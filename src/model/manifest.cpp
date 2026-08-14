#include "model/manifest.h"

#include <algorithm>
#include <array>
#include <iomanip>
#include <limits>
#include <map>
#include <regex>
#include <set>
#include <sstream>

#include "model/gemma4_26b_manifest.h"
#include "model/gemma4_26b_compiled_loader.h"
#include "model/safetensors.h"
#include "util/json.h"

namespace gem16::internal {
namespace {

struct CompiledRule {
  const QuantizationRule* rule = nullptr;
  std::vector<std::regex> targets;
};

bool EndsWith(std::string_view text, std::string_view suffix) {
  return text.size() >= suffix.size() && text.substr(text.size() - suffix.size()) == suffix;
}

std::string StripSuffix(std::string_view text, std::string_view suffix) {
  return std::string(text.substr(0, text.size() - suffix.size()));
}

std::string ModuleName(std::string_view tensor_name) {
  constexpr std::array<std::string_view, 8> suffixes = {
      ".weight_packed", ".weight_global_scale", ".input_global_scale", ".weight_scale_2",
      ".weight_scale", ".input_scale", ".weight", ".bias"};
  for (const auto suffix : suffixes) {
    if (EndsWith(tensor_name, suffix)) return StripSuffix(tensor_name, suffix);
  }
  return std::string(tensor_name);
}

Result<std::vector<CompiledRule>> CompileRules(const ModelConfig& config) {
  std::vector<CompiledRule> result;
  try {
    for (const auto& rule : config.quantization_rules) {
      CompiledRule compiled;
      compiled.rule = &rule;
      for (const auto& target : rule.regex_targets) compiled.targets.emplace_back(target, std::regex::ECMAScript | std::regex::optimize);
      result.push_back(std::move(compiled));
    }
  } catch (const std::regex_error& error) {
    return Status(StatusCode::kDataLoss, "invalid quantization target regex in config.json: " + std::string(error.what()));
  }
  return result;
}

const QuantizationRule* MatchRule(std::string_view module, const std::vector<CompiledRule>& rules) {
  const std::string owned(module);
  for (const auto& compiled : rules) {
    for (const auto& target : compiled.targets) {
      if (std::regex_match(owned, target)) return compiled.rule;
    }
  }
  return nullptr;
}

bool IsTextOnlyTensor(std::string_view name) {
  constexpr std::array<std::string_view, 8> modality_prefixes = {
      "model.vision_embedder.", "model.vision_tower.", "model.embed_vision.",
      "model.audio_embedder.", "model.audio_tower.", "model.embed_audio.",
      "model.video_tower.", "model.embed_video."};
  return std::none_of(modality_prefixes.begin(), modality_prefixes.end(),
                      [name](std::string_view prefix) { return name.starts_with(prefix); });
}

std::string StorageClass(std::string_view dtype) {
  if (dtype == "BF16") return "BF16";
  if (dtype == "F16") return "FP16";
  if (dtype == "F32") return "FP32";
  if (dtype == "F8_E4M3") return "FP8_E4M3";
  return "UNSUPPORTED";
}

std::string ExpectedRole(std::string_view name) {
  if (name == "pre_projection.weight") return "mtp_pre_projection";
  if (name == "post_projection.weight") return "mtp_post_projection";
  if (EndsWith(name, ".weight_packed")) return "projection_weight";
  if (EndsWith(name, ".weight_global_scale") || EndsWith(name, ".weight_scale_2")) return "weight_global_scale";
  if (EndsWith(name, ".input_global_scale")) return "activation_global_scale";
  if (EndsWith(name, ".weight_scale")) return "weight_local_or_channel_scale";
  if (EndsWith(name, ".input_scale")) return "activation_global_scale";
  if (name.find("embed_tokens.weight") != std::string_view::npos) return "tied_embedding_and_output";
  if (EndsWith(name, ".weight")) return "weight";
  if (EndsWith(name, ".bias")) return "bias";
  return "model_state";
}

std::string ShapeText(const std::vector<std::uint64_t>& shape) {
  std::ostringstream output;
  output << '[';
  for (std::size_t index = 0; index < shape.size(); ++index) {
    if (index != 0) output << ',';
    output << shape[index];
  }
  output << ']';
  return output.str();
}

Status ValidatePrimaryLayerTensorInventory(const ModelConfig& config, const std::set<std::string>& tensor_names) {
  constexpr std::array<std::string_view, 27> common_suffixes = {
      "input_layernorm.weight",
      "layer_scalar",
      "mlp.down_proj.input_global_scale",
      "mlp.down_proj.weight_global_scale",
      "mlp.down_proj.weight_packed",
      "mlp.down_proj.weight_scale",
      "mlp.gate_proj.input_global_scale",
      "mlp.gate_proj.weight_global_scale",
      "mlp.gate_proj.weight_packed",
      "mlp.gate_proj.weight_scale",
      "mlp.up_proj.input_global_scale",
      "mlp.up_proj.weight_global_scale",
      "mlp.up_proj.weight_packed",
      "mlp.up_proj.weight_scale",
      "post_attention_layernorm.weight",
      "post_feedforward_layernorm.weight",
      "pre_feedforward_layernorm.weight",
      "self_attn.k_norm.weight",
      "self_attn.k_proj.weight",
      "self_attn.k_proj.weight_scale",
      "self_attn.k_scale",
      "self_attn.o_proj.weight",
      "self_attn.o_proj.weight_scale",
      "self_attn.q_norm.weight",
      "self_attn.q_proj.weight",
      "self_attn.q_proj.weight_scale",
      "self_attn.v_scale",
  };
  constexpr std::array<std::string_view, 2> local_only_suffixes = {
      "self_attn.v_proj.weight", "self_attn.v_proj.weight_scale"};

  std::set<std::string> expected;
  for (std::size_t layer = 0; layer < config.layer_types.size(); ++layer) {
    const std::string prefix = "model.language_model.layers." + std::to_string(layer) + ".";
    for (const auto suffix : common_suffixes) expected.insert(prefix + std::string(suffix));
    if (config.layer_types[layer] == "sliding_attention") {
      for (const auto suffix : local_only_suffixes) expected.insert(prefix + std::string(suffix));
    }
  }

  std::set<std::string> actual;
  constexpr std::string_view layer_prefix = "model.language_model.layers.";
  for (const auto& name : tensor_names) {
    if (name.starts_with(layer_prefix)) actual.insert(name);
  }
  for (const auto& name : expected) {
    if (!actual.contains(name)) return Status(StatusCode::kDataLoss, "required decoder-layer tensor is missing: " + name);
  }
  for (const auto& name : actual) {
    if (!expected.contains(name)) return Status(StatusCode::kUnsupported, "unexpected decoder-layer tensor: " + name);
  }
  return Status::Ok();
}

Status ValidateAssistantTensorInventory(
    const ModelConfig& config, const std::set<std::string>& tensor_names,
    const std::map<std::string, const StoredTensor*, std::less<>>& stored_by_name) {
  std::map<std::string, std::vector<std::uint64_t>, std::less<>> expected;
  expected.emplace("model.embed_tokens.weight",
                   std::vector<std::uint64_t>{config.vocabulary_size, config.hidden_size});
  expected.emplace("model.norm.weight", std::vector<std::uint64_t>{config.hidden_size});
  expected.emplace("pre_projection.weight",
                   std::vector<std::uint64_t>{config.hidden_size,
                                              2U * config.backbone_hidden_size});
  expected.emplace("post_projection.weight",
                   std::vector<std::uint64_t>{config.backbone_hidden_size,
                                              config.hidden_size});

  for (std::size_t layer = 0; layer < config.layer_types.size(); ++layer) {
    const std::string prefix = "model.layers." + std::to_string(layer) + ".";
    expected.emplace(prefix + "input_layernorm.weight",
                     std::vector<std::uint64_t>{config.hidden_size});
    expected.emplace(prefix + "layer_scalar", std::vector<std::uint64_t>{1});
    expected.emplace(prefix + "mlp.down_proj.weight",
                     std::vector<std::uint64_t>{config.hidden_size,
                                                config.intermediate_size});
    expected.emplace(prefix + "mlp.gate_proj.weight",
                     std::vector<std::uint64_t>{config.intermediate_size,
                                                config.hidden_size});
    expected.emplace(prefix + "mlp.up_proj.weight",
                     std::vector<std::uint64_t>{config.intermediate_size,
                                                config.hidden_size});
    expected.emplace(prefix + "post_attention_layernorm.weight",
                     std::vector<std::uint64_t>{config.hidden_size});
    expected.emplace(prefix + "post_feedforward_layernorm.weight",
                     std::vector<std::uint64_t>{config.hidden_size});
    expected.emplace(prefix + "pre_feedforward_layernorm.weight",
                     std::vector<std::uint64_t>{config.hidden_size});
    const bool global = config.layer_types[layer] == "full_attention";
    const std::uint64_t projection_size =
        config.query_heads *
        (global ? config.global_head_dimension : config.local_head_dimension);
    expected.emplace(prefix + "self_attn.o_proj.weight",
                     std::vector<std::uint64_t>{config.hidden_size, projection_size});
    expected.emplace(prefix + "self_attn.q_norm.weight",
                     std::vector<std::uint64_t>{global ? config.global_head_dimension
                                                       : config.local_head_dimension});
    expected.emplace(prefix + "self_attn.q_proj.weight",
                     std::vector<std::uint64_t>{projection_size, config.hidden_size});
  }

  if (tensor_names.size() != expected.size()) {
    return Status(StatusCode::kDataLoss,
                  "assistant tensor count must be " +
                      std::to_string(expected.size()) + ", got " +
                      std::to_string(tensor_names.size()));
  }
  for (const auto& [name, shape] : expected) {
    const auto found = stored_by_name.find(name);
    if (found == stored_by_name.end()) {
      return Status(StatusCode::kDataLoss,
                    "required assistant tensor is missing: " + name);
    }
    if (found->second->dtype != "BF16" || found->second->shape != shape) {
      return Status(StatusCode::kDataLoss,
                    "assistant tensor schema mismatch: " + name +
                        " expected BF16 " + ShapeText(shape) + " got " +
                        found->second->dtype + " " +
                        ShapeText(found->second->shape));
    }
  }
  for (const auto& name : tensor_names) {
    if (!expected.contains(name)) {
      return Status(StatusCode::kUnsupported,
                    "unexpected assistant tensor: " + name);
    }
  }
  return Status::Ok();
}

void WriteString(std::ostream& output, std::string_view value) {
  output << '"' << json::Escape(value) << '"';
}

void WriteShape(std::ostream& output, const std::vector<std::uint64_t>& shape) {
  output << '[';
  for (std::size_t index = 0; index < shape.size(); ++index) {
    if (index != 0) output << ',';
    output << shape[index];
  }
  output << ']';
}

Status AddBytes(std::uint64_t bytes, std::uint64_t* total,
                std::string_view description) {
  if (*total > std::numeric_limits<std::uint64_t>::max() - bytes) {
    return Status(StatusCode::kDataLoss,
                  "manifest byte total overflows uint64: " +
                      std::string(description));
  }
  *total += bytes;
  return Status::Ok();
}

Status BuildManifestTotals(ModelManifest* manifest) {
  std::map<std::string, TensorClassTotal, std::less<>> by_class;
  std::map<std::string, TensorGroupTotal, std::less<>> by_role;
  std::map<std::string, TensorGroupTotal, std::less<>> by_residency;
  manifest->total_tensor_bytes = 0;
  manifest->text_only_tensor_bytes = 0;
  manifest->skipped_tensor_bytes = 0;
  for (auto& tensor : manifest->tensors) {
    if (tensor.logical_dtype.empty()) tensor.logical_dtype = tensor.storage_dtype;
    if (tensor.tensor_role.empty()) tensor.tensor_role = tensor.expected_role;
    if (tensor.residency_class.empty()) tensor.residency_class = "runtime_resident";
    if (tensor.source_family.empty()) tensor.source_family = manifest->model_variant;
    if (tensor.quantization_component.empty()) tensor.quantization_component = "value";
    if (tensor.quantization_producer.empty()) {
      tensor.quantization_producer = "checkpoint_declared";
    }
    if (tensor.local_scale_dtype.empty()) tensor.local_scale_dtype = "none";
    if (tensor.global_scale_role.empty()) tensor.global_scale_role = "none";
    if (tensor.activation_scale_role.empty()) tensor.activation_scale_role = "none";
    if (tensor.final_gpu_layout.empty()) tensor.final_gpu_layout = tensor.layout;

    auto& class_total = by_class[tensor.quantization_class];
    class_total.quantization_class = tensor.quantization_class;
    ++class_total.tensor_count;
    auto status = AddBytes(tensor.byte_length, &class_total.bytes,
                           tensor.quantization_class);
    if (!status.ok()) return status;

    auto& role_total = by_role[tensor.tensor_role];
    role_total.name = tensor.tensor_role;
    ++role_total.tensor_count;
    status = AddBytes(tensor.byte_length, &role_total.bytes, tensor.tensor_role);
    if (!status.ok()) return status;

    auto& residency_total = by_residency[tensor.residency_class];
    residency_total.name = tensor.residency_class;
    ++residency_total.tensor_count;
    status = AddBytes(tensor.byte_length, &residency_total.bytes,
                      tensor.residency_class);
    if (!status.ok()) return status;

    status = AddBytes(tensor.byte_length, &manifest->total_tensor_bytes,
                      "all tensors");
    if (!status.ok()) return status;
    auto* text_total = tensor.loaded_in_text_only_mode
                           ? &manifest->text_only_tensor_bytes
                           : &manifest->skipped_tensor_bytes;
    status = AddBytes(tensor.byte_length, text_total,
                      tensor.loaded_in_text_only_mode ? "text tensors"
                                                      : "skipped tensors");
    if (!status.ok()) return status;
  }
  manifest->totals.clear();
  manifest->totals_by_role.clear();
  manifest->totals_by_residency.clear();
  for (auto& [unused, total] : by_class) {
    (void)unused;
    manifest->totals.push_back(std::move(total));
  }
  for (auto& [unused, total] : by_role) {
    (void)unused;
    manifest->totals_by_role.push_back(std::move(total));
  }
  for (auto& [unused, total] : by_residency) {
    (void)unused;
    manifest->totals_by_residency.push_back(std::move(total));
  }
  return Status::Ok();
}

}  // namespace

Result<ModelManifest> BuildManifest(const std::filesystem::path& model_directory, const ModelConfig& config, bool validate) {
  auto stored = LoadSafetensorsDirectory(model_directory);
  if (!stored.ok()) return stored.status();
  auto compiled = CompileRules(config);
  if (!compiled.ok()) return compiled.status();

  std::set<std::string> names;
  std::map<std::string, const StoredTensor*, std::less<>> stored_by_name;
  for (const auto& tensor : stored.value()) {
    names.insert(tensor.name);
    stored_by_name.emplace(tensor.name, &tensor);
  }

  ModelManifest manifest;
  manifest.model_directory = model_directory.string();
  manifest.architecture = config.architecture;
  manifest.model_type = config.model_type;
  const auto variant = ClassifyModelVariant(config);
  const auto& traits = TraitsForModelVariant(variant);
  manifest.model_variant = std::string(traits.name);
  manifest.checkpoint_profile = manifest.model_variant;
  manifest.validation_contract = validate ? "generic_safetensors"
                                          : "unvalidated";
  manifest.layer_count = config.layer_count;
  manifest.hidden_size = config.hidden_size;
  manifest.intermediate_size = config.intermediate_size;
  manifest.moe_intermediate_size = config.moe_intermediate_size;
  manifest.expert_count = config.expert_count;
  manifest.top_k_experts = config.top_k_experts;
  manifest.runtime_supported = traits.executable;
  manifest.supports_text = traits.supports_text;
  manifest.supports_vision = traits.supports_vision;
  manifest.supports_audio = traits.supports_audio;
  manifest.supports_video = traits.supports_video;
  manifest.supports_mtp = traits.supports_mtp;
  const bool compiled_gemma4_26b =
      variant == ModelVariant::kGemma4Moe26BA4B &&
      std::filesystem::is_regular_file(model_directory /
                                       "gem16_compilation.json");

  for (const auto& stored_tensor : stored.value()) {
    TensorInfo tensor;
    tensor.name = stored_tensor.name;
    tensor.shape = stored_tensor.shape;
    tensor.logical_shape = stored_tensor.shape;
    tensor.storage_dtype = stored_tensor.dtype;
    tensor.byte_offset = stored_tensor.absolute_offset;
    tensor.byte_length = stored_tensor.length;
    tensor.alignment = stored_tensor.alignment;
    tensor.source_shard = stored_tensor.shard;
    tensor.expected_role = ExpectedRole(tensor.name);
    tensor.loaded_in_text_only_mode = IsTextOnlyTensor(tensor.name);
    tensor.aliased = tensor.name.find("embed_tokens.weight") != std::string::npos && config.tied_embeddings;
    tensor.layout = "source Safetensors order";

    const std::string module = ModuleName(tensor.name);
    const auto* rule = MatchRule(module, compiled.value());
    if (rule != nullptr && rule->format == "nvfp4-pack-quantized") {
      if (EndsWith(tensor.name, ".weight_packed")) {
        tensor.quantization_class = "NVFP4_PACKED";
        if (tensor.logical_shape.size() == 2 && tensor.logical_shape[1] <= std::numeric_limits<std::uint64_t>::max() / 2U) {
          tensor.logical_shape[1] *= 2U;
        }
        tensor.local_scale_tensor = module + ".weight_scale";
        tensor.global_scale_tensor = module + ".weight_global_scale";
        tensor.input_scale_tensor = module + ".input_global_scale";
        tensor.layout = "two E2M1 values per U8; contracting dimension packed by 2";
      } else if (EndsWith(tensor.name, ".weight_global_scale") || EndsWith(tensor.name, ".weight_scale_2")) {
        tensor.quantization_class = "NVFP4_GLOBAL_SCALE";
      } else if (EndsWith(tensor.name, ".weight_scale")) {
        tensor.quantization_class = "NVFP4_LOCAL_SCALE_E4M3";
      } else if (EndsWith(tensor.name, ".input_global_scale") || EndsWith(tensor.name, ".input_scale")) {
        tensor.quantization_class = "NVFP4_INPUT_SCALE";
      } else {
        tensor.quantization_class = StorageClass(tensor.storage_dtype);
      }
    } else if (rule != nullptr && rule->format == "float-quantized") {
      if (EndsWith(tensor.name, ".weight_scale")) {
        tensor.quantization_class = "FP8_WEIGHT_SCALE";
      } else if (EndsWith(tensor.name, ".input_scale")) {
        tensor.quantization_class = "FP8_INPUT_SCALE";
      } else if (EndsWith(tensor.name, ".weight")) {
        tensor.quantization_class = "FP8_WEIGHT_E4M3";
        tensor.local_scale_tensor = module + ".weight_scale";
        const std::string static_input_scale = module + ".input_scale";
        if (names.contains(static_input_scale)) tensor.input_scale_tensor = static_input_scale;
      } else {
        tensor.quantization_class = StorageClass(tensor.storage_dtype);
      }
    } else {
      tensor.quantization_class = StorageClass(tensor.storage_dtype);
    }

    if (validate && !compiled_gemma4_26b &&
        tensor.quantization_class == "UNSUPPORTED") {
      return Status(StatusCode::kUnsupported, "unsupported tensor: name=" + tensor.name + " shape=" + ShapeText(tensor.shape) + " dtype=" + tensor.storage_dtype + " quantization_group=" + (rule == nullptr ? "none" : rule->group_name));
    }
    if (validate && !compiled_gemma4_26b &&
        tensor.quantization_class == "NVFP4_PACKED") {
      for (const auto& required : {tensor.local_scale_tensor, tensor.global_scale_tensor, tensor.input_scale_tensor}) {
        if (!names.contains(required)) return Status(StatusCode::kDataLoss, "NVFP4 tensor is missing required scale tensor: " + tensor.name + " -> " + required);
      }
      if (tensor.logical_shape.size() != 2 || tensor.logical_shape[1] % 16U != 0) {
        return Status(StatusCode::kDataLoss, "NVFP4 logical contracting dimension must be divisible by 16: " + tensor.name);
      }
      const auto* local = stored_by_name.at(tensor.local_scale_tensor);
      const auto* global = stored_by_name.at(tensor.global_scale_tensor);
      const auto* input = stored_by_name.at(tensor.input_scale_tensor);
      if (tensor.storage_dtype != "U8" || local->dtype != "F8_E4M3" || global->dtype != "F32" || input->dtype != "F32") {
        return Status(StatusCode::kDataLoss, "NVFP4 storage or scale dtype mismatch: " + tensor.name);
      }
      if (local->shape.size() != 2 || local->shape[0] != tensor.logical_shape[0] || local->shape[1] * 16U != tensor.logical_shape[1]) {
        return Status(StatusCode::kDataLoss, "NVFP4 local scale shape does not describe groups of 16: " + tensor.name);
      }
      if (global->shape != std::vector<std::uint64_t>{1} || input->shape != std::vector<std::uint64_t>{1}) {
        return Status(StatusCode::kDataLoss, "NVFP4 global scales must have shape [1]: " + tensor.name);
      }
    }
    if (validate && !compiled_gemma4_26b &&
        tensor.quantization_class == "FP8_WEIGHT_E4M3") {
      const auto scale = names.find(tensor.local_scale_tensor);
      if (scale == names.end()) return Status(StatusCode::kDataLoss, "FP8 weight is missing its channel scale: " + tensor.name);
      const auto* stored_scale = stored_by_name.at(*scale);
      if (tensor.storage_dtype != "F8_E4M3" || stored_scale->dtype != "BF16" || tensor.shape.size() != 2 ||
          stored_scale->shape != std::vector<std::uint64_t>{tensor.shape[0], 1}) {
        return Status(StatusCode::kDataLoss, "FP8 weight/channel-scale schema mismatch: " + tensor.name);
      }
    }

    manifest.tensors.push_back(std::move(tensor));
  }
  if (validate && config.tied_embeddings && !compiled_gemma4_26b) {
    const std::string expected_embedding =
        IsAssistantModel(config) ? "model.embed_tokens.weight"
                                 : "model.language_model.embed_tokens.weight";
    if (!names.contains(expected_embedding)) {
      return Status(StatusCode::kDataLoss,
                    "tied input/output embedding tensor is missing: " +
                        expected_embedding);
    }
    if (names.contains("lm_head.weight") ||
        names.contains("model.language_model.lm_head.weight")) {
      return Status(StatusCode::kDataLoss,
                    "tied checkpoint unexpectedly stores a duplicate LM head");
    }
  }
  if (validate && compiled_gemma4_26b) {
    auto status = ValidateAndBindGemma4Moe26BCompiledArtifact(
        model_directory, &manifest.tensors);
    if (!status.ok()) return status;
    manifest.checkpoint_profile = "sm120-text-hybrid-v1";
    manifest.validation_contract = "gemma4_26b_m08_compiled_hybrid_v1";
    manifest.tensor_contract_validated = true;
    manifest.supports_text = true;
    manifest.supports_vision = false;
    manifest.supports_audio = false;
    manifest.supports_video = false;
    manifest.supports_mtp = false;
  } else if (validate && variant == ModelVariant::kGemma4Moe26BA4B) {
    auto profile =
        ValidateAndAnnotateGemma4Moe26BInventory(config, &manifest.tensors);
    if (!profile.ok()) return profile.status();
    manifest.checkpoint_profile =
        std::string(Gemma4Moe26BInventoryProfileName(profile.value()));
    manifest.validation_contract = "gemma4_26b_m03_exact_inventory_v1";
    manifest.tensor_contract_validated = true;
  } else if (validate) {
    const auto inventory_status = IsAssistantModel(config)
                                      ? ValidateAssistantTensorInventory(
                                            config, names, stored_by_name)
                                      : ValidatePrimaryLayerTensorInventory(config, names);
    if (!inventory_status.ok()) return inventory_status;
    manifest.validation_contract = IsAssistantModel(config)
                                       ? "gemma4_assistant_exact_inventory"
                                       : "gemma4_unified_12b_exact_inventory";
    manifest.tensor_contract_validated = true;
  }
  auto totals_status = BuildManifestTotals(&manifest);
  if (!totals_status.ok()) return totals_status;
  return manifest;
}

}  // namespace gem16::internal

namespace gem16 {

Status WriteManifestJson(const ModelManifest& manifest, std::ostream& output) {
  output << "{\n  \"schema_version\": 3,\n  \"model_directory\": ";
  internal::WriteString(output, manifest.model_directory);
  output << ",\n  \"architecture\": ";
  internal::WriteString(output, manifest.architecture);
  output << ",\n  \"model_type\": ";
  internal::WriteString(output, manifest.model_type);
  output << ",\n  \"model_variant\": ";
  internal::WriteString(output, manifest.model_variant);
  output << ",\n  \"checkpoint_profile\": ";
  internal::WriteString(output, manifest.checkpoint_profile);
  output << ",\n  \"validation_contract\": ";
  internal::WriteString(output, manifest.validation_contract);
  output << ",\n  \"layer_count\": " << manifest.layer_count
         << ",\n  \"hidden_size\": " << manifest.hidden_size
         << ",\n  \"intermediate_size\": " << manifest.intermediate_size
         << ",\n  \"moe_intermediate_size\": " << manifest.moe_intermediate_size
         << ",\n  \"expert_count\": " << manifest.expert_count
         << ",\n  \"top_k_experts\": " << manifest.top_k_experts
         << ",\n  \"runtime_supported\": "
         << (manifest.runtime_supported ? "true" : "false")
         << ",\n  \"tensor_contract_validated\": "
         << (manifest.tensor_contract_validated ? "true" : "false")
         << ",\n  \"capabilities\": {\"text\":"
         << (manifest.supports_text ? "true" : "false")
         << ",\"vision\":" << (manifest.supports_vision ? "true" : "false")
         << ",\"audio\":" << (manifest.supports_audio ? "true" : "false")
         << ",\"video\":" << (manifest.supports_video ? "true" : "false")
         << ",\"mtp\":" << (manifest.supports_mtp ? "true" : "false")
         << "},\n  \"tensors\": [\n";
  for (std::size_t index = 0; index < manifest.tensors.size(); ++index) {
    const auto& tensor = manifest.tensors[index];
    output << "    {\"name\":"; internal::WriteString(output, tensor.name);
    output << ",\"shape\":"; internal::WriteShape(output, tensor.shape);
    output << ",\"logical_shape\":"; internal::WriteShape(output, tensor.logical_shape);
    output << ",\"storage_dtype\":"; internal::WriteString(output, tensor.storage_dtype);
    output << ",\"logical_dtype\":"; internal::WriteString(output, tensor.logical_dtype);
    output << ",\"quantization_class\":"; internal::WriteString(output, tensor.quantization_class);
    output << ",\"byte_offset\":" << tensor.byte_offset << ",\"byte_length\":" << tensor.byte_length << ",\"alignment\":" << tensor.alignment;
    output << ",\"source_shard\":"; internal::WriteString(output, tensor.source_shard);
    output << ",\"expected_role\":"; internal::WriteString(output, tensor.expected_role);
    output << ",\"tensor_role\":"; internal::WriteString(output, tensor.tensor_role);
    output << ",\"residency_class\":"; internal::WriteString(output, tensor.residency_class);
    output << ",\"source_family\":"; internal::WriteString(output, tensor.source_family);
    output << ",\"quantization_component\":"; internal::WriteString(output, tensor.quantization_component);
    output << ",\"quantization_producer\":"; internal::WriteString(output, tensor.quantization_producer);
    output << ",\"local_scale_dtype\":"; internal::WriteString(output, tensor.local_scale_dtype);
    output << ",\"local_scale_vector_size\":" << tensor.local_scale_vector_size;
    output << ",\"global_scale_role\":"; internal::WriteString(output, tensor.global_scale_role);
    output << ",\"activation_scale_role\":"; internal::WriteString(output, tensor.activation_scale_role);
    output << ",\"final_gpu_layout\":"; internal::WriteString(output, tensor.final_gpu_layout);
    output << ",\"logical_axis_order\":"; internal::WriteString(output, tensor.logical_axis_order);
    output << ",\"layer_index\":" << tensor.layer_index
           << ",\"expert_index\":" << tensor.expert_index
           << ",\"expert_axis\":" << tensor.expert_axis;
    output << ",\"local_scale_tensor\":"; internal::WriteString(output, tensor.local_scale_tensor);
    output << ",\"global_scale_tensor\":"; internal::WriteString(output, tensor.global_scale_tensor);
    output << ",\"input_scale_tensor\":"; internal::WriteString(output, tensor.input_scale_tensor);
    output << ",\"layout\":"; internal::WriteString(output, tensor.layout);
    output << ",\"loaded_in_text_only_mode\":" << (tensor.loaded_in_text_only_mode ? "true" : "false");
    output << ",\"aliased\":" << (tensor.aliased ? "true" : "false") << '}';
    output << (index + 1U == manifest.tensors.size() ? "\n" : ",\n");
  }
  output << "  ],\n  \"totals_by_class\": [\n";
  for (std::size_t index = 0; index < manifest.totals.size(); ++index) {
    const auto& total = manifest.totals[index];
    output << "    {\"quantization_class\":"; internal::WriteString(output, total.quantization_class);
    output << ",\"tensor_count\":" << total.tensor_count << ",\"bytes\":" << total.bytes << '}';
    output << (index + 1U == manifest.totals.size() ? "\n" : ",\n");
  }
  output << "  ],\n  \"totals_by_role\": [\n";
  for (std::size_t index = 0; index < manifest.totals_by_role.size(); ++index) {
    const auto& total = manifest.totals_by_role[index];
    output << "    {\"role\":";
    internal::WriteString(output, total.name);
    output << ",\"tensor_count\":" << total.tensor_count
           << ",\"bytes\":" << total.bytes << '}';
    output << (index + 1U == manifest.totals_by_role.size() ? "\n" : ",\n");
  }
  output << "  ],\n  \"totals_by_residency\": [\n";
  for (std::size_t index = 0; index < manifest.totals_by_residency.size();
       ++index) {
    const auto& total = manifest.totals_by_residency[index];
    output << "    {\"residency_class\":";
    internal::WriteString(output, total.name);
    output << ",\"tensor_count\":" << total.tensor_count
           << ",\"bytes\":" << total.bytes << '}';
    output << (index + 1U == manifest.totals_by_residency.size() ? "\n"
                                                                 : ",\n");
  }
  output << "  ],\n  \"total_tensor_bytes\": " << manifest.total_tensor_bytes
         << ",\n  \"text_only_tensor_bytes\": " << manifest.text_only_tensor_bytes
         << ",\n  \"skipped_tensor_bytes\": " << manifest.skipped_tensor_bytes << "\n}\n";
  if (!output) return Status(StatusCode::kIoError, "failed while writing manifest JSON");
  return Status::Ok();
}

void PrintManifestSummary(const ModelManifest& manifest, std::ostream& output) {
  output << "Checkpoint: " << manifest.model_directory << '\n'
         << "Architecture: " << manifest.architecture << " (" << manifest.model_type << ")\n"
         << "Variant: " << manifest.model_variant
         << " profile=" << manifest.checkpoint_profile
         << " runtime=" << (manifest.runtime_supported ? "supported" : "unsupported")
         << " tensor_contract="
         << (manifest.tensor_contract_validated ? "validated" : "not-yet-validated")
         << " contract=" << manifest.validation_contract << '\n'
         << "MoE: layers=" << manifest.layer_count << " hidden=" << manifest.hidden_size
         << " shared_intermediate=" << manifest.intermediate_size
         << " expert_intermediate=" << manifest.moe_intermediate_size
         << " experts=" << manifest.expert_count << " top_k=" << manifest.top_k_experts << '\n'
         << "Tensors: " << manifest.tensors.size() << "\n\n";
  for (const auto& tensor : manifest.tensors) {
    output << tensor.name << " shape=" << internal::ShapeText(tensor.shape)
           << " logical=" << internal::ShapeText(tensor.logical_shape)
           << " dtype=" << tensor.storage_dtype << " class=" << tensor.quantization_class
           << " offset=" << tensor.byte_offset << " bytes=" << tensor.byte_length
           << " align=" << tensor.alignment << " shard=" << tensor.source_shard
           << " role=" << tensor.tensor_role
           << " residency=" << tensor.residency_class
           << " producer=" << tensor.quantization_producer
           << " text=" << (tensor.loaded_in_text_only_mode ? "load" : "skip")
           << " alias=" << (tensor.aliased ? "yes" : "no")
           << " layout=\"" << tensor.layout << "\"";
    if (!tensor.local_scale_tensor.empty()) output << " local_scale=" << tensor.local_scale_tensor;
    if (!tensor.global_scale_tensor.empty()) output << " global_scale=" << tensor.global_scale_tensor;
    if (!tensor.input_scale_tensor.empty()) output << " input_scale=" << tensor.input_scale_tensor;
    output << '\n';
  }
  output << "\nTotals by class:\n";
  for (const auto& total : manifest.totals) {
    output << "  " << std::setw(26) << std::left << total.quantization_class
           << " tensors=" << total.tensor_count << " bytes=" << total.bytes << '\n';
  }
  output << "\nTotals by role:\n";
  for (const auto& total : manifest.totals_by_role) {
    output << "  " << std::setw(36) << std::left << total.name
           << " tensors=" << total.tensor_count << " bytes=" << total.bytes
           << '\n';
  }
  output << "\nTotals by residency:\n";
  for (const auto& total : manifest.totals_by_residency) {
    output << "  " << std::setw(36) << std::left << total.name
           << " tensors=" << total.tensor_count << " bytes=" << total.bytes
           << '\n';
  }
  output << "Total tensor bytes: " << manifest.total_tensor_bytes << '\n'
         << "Text-only resident bytes: " << manifest.text_only_tensor_bytes << '\n'
         << "Text-only skipped bytes: " << manifest.skipped_tensor_bytes << '\n';
}

}  // namespace gem16
