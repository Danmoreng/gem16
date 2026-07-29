#include "model/config.h"

#include <array>
#include <cmath>
#include <fstream>
#include <sstream>

#include "util/json.h"

namespace gem16::internal {
namespace {

constexpr std::uint64_t kMaxConfigBytes = 16U * 1024U * 1024U;

Result<std::string> ReadConfig(const std::filesystem::path& path) {
  std::error_code error;
  const auto size = std::filesystem::file_size(path, error);
  if (error) {
    return Status(StatusCode::kIoError, "cannot stat config: " + path.string() + ": " + error.message());
  }
  if (size > kMaxConfigBytes) {
    return Status(StatusCode::kDataLoss, "config exceeds 16 MiB safety limit: " + path.string());
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return Status(StatusCode::kIoError, "cannot open config: " + path.string());
  }
  std::ostringstream contents;
  contents << input.rdbuf();
  if (!input.good() && !input.eof()) {
    return Status(StatusCode::kIoError, "failed while reading config: " + path.string());
  }
  return contents.str();
}

const json::Value* Member(const json::Value& object, std::string_view name) {
  return object.is_object() ? object.find(name) : nullptr;
}

const json::Value* Nested(const json::Value& root, std::string_view first, std::string_view second) {
  const auto* parent = Member(root, first);
  return parent == nullptr ? nullptr : Member(*parent, second);
}

std::string StringOrEmpty(const json::Value* value) {
  return value != nullptr && value->is_string() ? value->as_string() : std::string{};
}

std::uint64_t UnsignedOrZero(const json::Value* value) {
  if (value == nullptr || !value->is_integer() || value->as_integer() < 0) {
    return 0;
  }
  return static_cast<std::uint64_t>(value->as_integer());
}

bool BoolOrFalse(const json::Value* value) {
  return value != nullptr && value->is_bool() && value->as_bool();
}

std::vector<std::string> StringArray(const json::Value* value) {
  std::vector<std::string> result;
  if (value == nullptr || !value->is_array()) {
    return result;
  }
  for (const auto& item : value->as_array()) {
    if (item.is_string()) {
      result.push_back(item.as_string());
    }
  }
  return result;
}

double NumberOrZero(const json::Value* value) {
  return value != nullptr && value->is_number() ? value->as_number() : 0.0;
}

Status ContractError(std::string_view contract, std::string field,
                     std::string expected) {
  return Status(StatusCode::kUnsupported, "unsupported " + std::string(contract) +
                                              " checkpoint config: " +
                                              std::move(field) + " must be " +
                                              std::move(expected));
}

Status PrimaryContractError(std::string field, std::string expected) {
  return ContractError("primary", std::move(field), std::move(expected));
}

Status AssistantContractError(std::string field, std::string expected) {
  return ContractError("assistant", std::move(field), std::move(expected));
}

}  // namespace

Result<ModelConfig> LoadModelConfig(const std::filesystem::path& path) {
  auto text = ReadConfig(path);
  if (!text.ok()) {
    return text.status();
  }
  auto parsed = json::Parse(text.value(), {.max_depth = 128, .max_values = 100'000, .max_string_bytes = 8U * 1024U * 1024U});
  if (!parsed.ok()) {
    return Status(parsed.status().code(), path.string() + ": " + parsed.status().message());
  }
  const auto& root = parsed.value();
  if (!root.is_object()) {
    return Status(StatusCode::kDataLoss, "config root must be an object: " + path.string());
  }

  ModelConfig config;
  const auto architectures = StringArray(Member(root, "architectures"));
  if (!architectures.empty()) {
    config.architecture = architectures.front();
  }
  config.model_type = StringOrEmpty(Member(root, "model_type"));
  config.text_model_type = StringOrEmpty(Nested(root, "text_config", "model_type"));
  config.hidden_size = UnsignedOrZero(Nested(root, "text_config", "hidden_size"));
  config.backbone_hidden_size = UnsignedOrZero(Member(root, "backbone_hidden_size"));
  config.intermediate_size = UnsignedOrZero(Nested(root, "text_config", "intermediate_size"));
  config.layer_count = UnsignedOrZero(Nested(root, "text_config", "num_hidden_layers"));
  config.vocabulary_size = UnsignedOrZero(Nested(root, "text_config", "vocab_size"));
  config.max_positions = UnsignedOrZero(Nested(root, "text_config", "max_position_embeddings"));
  config.sliding_window = UnsignedOrZero(Nested(root, "text_config", "sliding_window"));
  config.query_heads = UnsignedOrZero(Nested(root, "text_config", "num_attention_heads"));
  config.local_kv_heads = UnsignedOrZero(Nested(root, "text_config", "num_key_value_heads"));
  config.global_kv_heads = UnsignedOrZero(Nested(root, "text_config", "num_global_key_value_heads"));
  config.local_head_dimension = UnsignedOrZero(Nested(root, "text_config", "head_dim"));
  config.global_head_dimension = UnsignedOrZero(Nested(root, "text_config", "global_head_dim"));
  config.shared_kv_layer_count = UnsignedOrZero(Nested(root, "text_config", "num_kv_shared_layers"));
  config.centroid_count = UnsignedOrZero(Member(root, "num_centroids"));
  config.centroid_intermediate_top_k = UnsignedOrZero(Member(root, "centroid_intermediate_top_k"));
  config.audio_embedding_dimension =
      UnsignedOrZero(Nested(root, "audio_config", "audio_embed_dim"));
  config.audio_token_id = UnsignedOrZero(Member(root, "audio_token_id"));
  config.begin_audio_token_id = UnsignedOrZero(Member(root, "boa_token_id"));
  config.end_audio_token_id = UnsignedOrZero(Member(root, "eoa_token_index"));
  config.vision_embedding_dimension =
      UnsignedOrZero(Nested(root, "vision_config", "mm_embed_dim"));
  config.vision_position_count =
      UnsignedOrZero(Nested(root, "vision_config", "mm_posemb_size"));
  config.vision_soft_token_count =
      UnsignedOrZero(Nested(root, "vision_config", "num_soft_tokens"));
  config.vision_patch_size =
      UnsignedOrZero(Nested(root, "vision_config", "patch_size"));
  config.vision_pooling_kernel_size =
      UnsignedOrZero(Nested(root, "vision_config", "pooling_kernel_size"));
  config.image_token_id = UnsignedOrZero(Member(root, "image_token_id"));
  config.begin_image_token_id = UnsignedOrZero(Member(root, "boi_token_id"));
  config.end_image_token_id = UnsignedOrZero(Member(root, "eoi_token_id"));
  config.attention_k_eq_v = BoolOrFalse(Nested(root, "text_config", "attention_k_eq_v"));
  config.tied_embeddings = BoolOrFalse(Member(root, "tie_word_embeddings")) ||
                           BoolOrFalse(Nested(root, "text_config", "tie_word_embeddings"));
  config.ordered_embeddings = BoolOrFalse(Member(root, "use_ordered_embeddings"));
  config.final_logit_softcap = NumberOrZero(Nested(root, "text_config", "final_logit_softcapping"));
  config.audio_rms_norm_epsilon =
      NumberOrZero(Nested(root, "audio_config", "rms_norm_eps"));
  config.vision_rms_norm_epsilon =
      NumberOrZero(Nested(root, "vision_config", "rms_norm_eps"));
  config.layer_types = StringArray(Nested(root, "text_config", "layer_types"));

  const auto* quantization = Member(root, "quantization_config");
  if (quantization != nullptr && quantization->is_object()) {
    config.quant_method = StringOrEmpty(Member(*quantization, "quant_method"));
    config.quant_format = StringOrEmpty(Member(*quantization, "format"));
    config.ignored_modules = StringArray(Member(*quantization, "ignore"));
    const auto* groups = Member(*quantization, "config_groups");
    if (groups != nullptr && groups->is_object()) {
      for (const auto& [group_name, value] : groups->as_object()) {
        if (!value.is_object()) {
          return Status(StatusCode::kDataLoss, "quantization group is not an object: " + group_name);
        }
        QuantizationRule rule;
        rule.group_name = group_name;
        rule.format = StringOrEmpty(Member(value, "format"));
        for (auto target : StringArray(Member(value, "targets"))) {
          if (target.starts_with("re:")) {
            rule.regex_targets.push_back(target.substr(3));
          } else {
            return Status(StatusCode::kUnsupported, "quantization target is not an explicit regex: " + target);
          }
        }
        const auto* weights = Member(value, "weights");
        const auto* activations = Member(value, "input_activations");
        rule.weight_bits = static_cast<std::uint32_t>(UnsignedOrZero(weights == nullptr ? nullptr : Member(*weights, "num_bits")));
        rule.activation_bits = static_cast<std::uint32_t>(UnsignedOrZero(activations == nullptr ? nullptr : Member(*activations, "num_bits")));
        rule.group_size = static_cast<std::uint32_t>(UnsignedOrZero(weights == nullptr ? nullptr : Member(*weights, "group_size")));
        rule.scale_dtype = StringOrEmpty(weights == nullptr ? nullptr : Member(*weights, "scale_dtype"));
        rule.weight_strategy = StringOrEmpty(weights == nullptr ? nullptr : Member(*weights, "strategy"));
        rule.activation_strategy = StringOrEmpty(activations == nullptr ? nullptr : Member(*activations, "strategy"));
        const auto* dynamic = activations == nullptr ? nullptr : Member(*activations, "dynamic");
        rule.activation_dynamic = dynamic != nullptr && dynamic->is_bool() && dynamic->as_bool();
        rule.activation_dynamic_local = dynamic != nullptr && dynamic->is_string() && dynamic->as_string() == "local";
        config.quantization_rules.push_back(std::move(rule));
      }
    }
  }
  return config;
}

bool IsPrimaryModel(const ModelConfig& config) {
  return config.architecture == "Gemma4UnifiedForConditionalGeneration" &&
         config.model_type == "gemma4_unified";
}

bool IsAssistantModel(const ModelConfig& config) {
  return config.architecture == "Gemma4UnifiedAssistantForCausalLM" &&
         config.model_type == "gemma4_unified_assistant";
}

Status ValidatePrimaryModelContract(const ModelConfig& config) {
  if (config.architecture != "Gemma4UnifiedForConditionalGeneration") return PrimaryContractError("architecture", "Gemma4UnifiedForConditionalGeneration");
  if (config.model_type != "gemma4_unified") return PrimaryContractError("model_type", "gemma4_unified");
  if (config.text_model_type != "gemma4_unified_text") return PrimaryContractError("text_config.model_type", "gemma4_unified_text");
  if (config.layer_count != 48) return PrimaryContractError("num_hidden_layers", "48");
  if (config.hidden_size != 3840) return PrimaryContractError("hidden_size", "3840");
  if (config.intermediate_size != 15360) return PrimaryContractError("intermediate_size", "15360");
  if (config.query_heads != 16 || config.local_kv_heads != 8 || config.global_kv_heads != 1) return PrimaryContractError("attention head counts", "16 query / 8 local KV / 1 global KV");
  if (config.local_head_dimension != 256 || config.global_head_dimension != 512) return PrimaryContractError("head dimensions", "256 local / 512 global");
  if (config.sliding_window != 1024 || config.max_positions != 262144) return PrimaryContractError("context dimensions", "1024 sliding / 262144 maximum");
  if (config.vocabulary_size != 262144) return PrimaryContractError("vocab_size", "262144");
  if (config.audio_embedding_dimension != 640 ||
      config.audio_token_id != 258881 ||
      config.begin_audio_token_id != 256000 ||
      config.end_audio_token_id != 258883 ||
      std::abs(config.audio_rms_norm_epsilon - 1.0e-6) > 1.0e-12) {
    return PrimaryContractError(
        "audio embedding/token contract",
        "640 dimensions, token IDs 258881/256000/258883, epsilon 1e-6");
  }
  if (config.vision_embedding_dimension != 3840 ||
      config.vision_position_count != 1120 ||
      config.vision_soft_token_count != 280 ||
      config.vision_patch_size != 16 ||
      config.vision_pooling_kernel_size != 3 ||
      config.image_token_id != 258880 ||
      config.begin_image_token_id != 255999 ||
      config.end_image_token_id != 258882 ||
      std::abs(config.vision_rms_norm_epsilon - 1.0e-6) > 1.0e-12) {
    return PrimaryContractError(
        "vision embedding/token contract",
        "3840 dimensions, 1120 positions, 280 tokens, 16x3 patches, token IDs 258880/255999/258882, epsilon 1e-6");
  }
  if (!config.tied_embeddings || !config.attention_k_eq_v) return PrimaryContractError("tied embeddings and attention_k_eq_v", "true");
  if (std::abs(config.final_logit_softcap - 30.0) > 1e-12) return PrimaryContractError("final_logit_softcapping", "30.0");
  if (config.quant_method != "compressed-tensors" || config.quant_format != "mixed-precision") return PrimaryContractError("quantization schema", "compressed-tensors mixed-precision");
  if (config.layer_types.size() != 48) return PrimaryContractError("layer_types length", "48");
  for (std::size_t index = 0; index < config.layer_types.size(); ++index) {
    const bool expected_global = (index % 6U) == 5U;
    const std::string_view expected = expected_global ? "full_attention" : "sliding_attention";
    if (config.layer_types[index] != expected) return PrimaryContractError("layer_types pattern", "five sliding layers followed by one full layer");
  }
  return Status::Ok();
}

Status ValidateAssistantModelContract(const ModelConfig& config) {
  if (config.architecture != "Gemma4UnifiedAssistantForCausalLM") return AssistantContractError("architecture", "Gemma4UnifiedAssistantForCausalLM");
  if (config.model_type != "gemma4_unified_assistant") return AssistantContractError("model_type", "gemma4_unified_assistant");
  if (config.text_model_type != "gemma4_unified_text") return AssistantContractError("text_config.model_type", "gemma4_unified_text");
  if (config.backbone_hidden_size != 3840 || config.hidden_size != 1024) return AssistantContractError("hidden dimensions", "3840 backbone / 1024 assistant");
  if (config.intermediate_size != 8192 || config.layer_count != 4) return AssistantContractError("decoder dimensions", "4 layers / 8192 intermediate");
  if (config.query_heads != 16 || config.local_kv_heads != 8 || config.global_kv_heads != 1) return AssistantContractError("attention head counts", "16 query / 8 local KV / 1 global KV");
  if (config.local_head_dimension != 256 || config.global_head_dimension != 512) return AssistantContractError("head dimensions", "256 local / 512 global");
  if (config.sliding_window != 1024 || config.max_positions != 262144) return AssistantContractError("context dimensions", "1024 sliding / 262144 maximum");
  if (config.vocabulary_size != 262144 || !config.tied_embeddings) return AssistantContractError("vocabulary and tied embeddings", "262144 / true");
  if (config.shared_kv_layer_count != 4) return AssistantContractError("num_kv_shared_layers", "4");
  if (config.centroid_count != 2048 || config.centroid_intermediate_top_k != 32) return AssistantContractError("centroid metadata", "2048 centroids / top-k 32");
  if (config.ordered_embeddings) return AssistantContractError("use_ordered_embeddings", "false");
  constexpr std::array<std::string_view, 4> expected = {
      "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"};
  if (config.layer_types.size() != expected.size()) return AssistantContractError("layer_types length", "4");
  for (std::size_t index = 0; index < expected.size(); ++index) {
    if (config.layer_types[index] != expected[index]) return AssistantContractError("layer_types pattern", "three sliding layers followed by one full layer");
  }
  if (!config.quant_method.empty() || !config.quant_format.empty() || !config.quantization_rules.empty()) {
    return AssistantContractError("quantization schema", "absent for the BF16 assistant");
  }
  return Status::Ok();
}

Status ValidateInspectableModelContract(const ModelConfig& config) {
  if (IsPrimaryModel(config)) return ValidatePrimaryModelContract(config);
  if (IsAssistantModel(config)) return ValidateAssistantModelContract(config);
  return Status(StatusCode::kUnsupported,
                "unsupported checkpoint architecture/model_type: " +
                    config.architecture + " / " + config.model_type);
}

}  // namespace gem16::internal
