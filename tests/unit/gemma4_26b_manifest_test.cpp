#include "test.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>

#include "model/config.h"
#include "model/gemma4_26b_manifest.h"
#include "model/gemma4_26b_attention.h"
#include "model/gemma4_26b_residency.h"
#include "util/json.h"

namespace {

using gem16::TensorInfo;
using gem16::internal::Gemma4Moe26BHeadFormat;
using gem16::internal::Gemma4Moe26BInventoryProfile;

TensorInfo* Find(std::vector<TensorInfo>* tensors, std::string_view name) {
  const auto found = std::find_if(
      tensors->begin(), tensors->end(),
      [name](const TensorInfo& tensor) { return tensor.name == name; });
  return found == tensors->end() ? nullptr : &*found;
}

const TensorInfo* Find(const std::vector<TensorInfo>* tensors,
                       std::string_view name) {
  const auto found = std::find_if(
      tensors->begin(), tensors->end(),
      [name](const TensorInfo& tensor) { return tensor.name == name; });
  return found == tensors->end() ? nullptr : &*found;
}

void Remove(std::vector<TensorInfo>* tensors, std::string_view name) {
  const auto found = std::find_if(
      tensors->begin(), tensors->end(),
      [name](const TensorInfo& tensor) { return tensor.name == name; });
  if (found != tensors->end()) tensors->erase(found);
}

void CheckCompiledFp8AttentionBinding(
    const std::vector<TensorInfo>& tensors,
    Gemma4Moe26BHeadFormat head_format) {
  constexpr std::uint64_t kHidden = 2816;
  constexpr std::uint64_t kLayerCount = 30;
  std::size_t expected_projection_count = 0;
  std::size_t weight_count = 0;
  std::size_t scale_count = 0;
  for (std::uint64_t layer = 0; layer < kLayerCount; ++layer) {
    const bool global = layer % 6U == 5U;
    const std::string prefix = "model.language_model.layers." +
                               std::to_string(layer) + ".self_attn.";
    const std::array<std::pair<std::string_view, std::string_view>, 4> projections = {{
        {"q", "attention_q_projection"},
        {"k", "attention_k_projection"},
        {"o", "attention_o_projection"},
        {"v", "attention_v_projection"},
    }};
    for (const auto& [projection, role] : projections) {
      if (global && projection == "v") continue;
      ++expected_projection_count;
      const std::uint64_t output_rows =
          projection == "q" ? (global ? 8192U : 4096U)
                             : projection == "k" || projection == "v"
                                   ? (global ? 1024U : 2048U)
                                   : kHidden;
      const std::uint64_t input_columns =
          projection == "o" ? (global ? 8192U : 4096U) : kHidden;
      const std::string module = prefix + std::string(projection) + "_proj";
      const auto weight_name = module + ".weight";
      const auto scale_name = module + ".weight_scale";
      const auto* weight = Find(&tensors, weight_name);
      const auto* scale = Find(&tensors, scale_name);
      GEM16_CHECK(weight != nullptr);
      GEM16_CHECK(scale != nullptr);
      if (weight == nullptr || scale == nullptr) continue;

      const std::vector<std::uint64_t> weight_shape = {output_rows,
                                                        input_columns};
      const std::vector<std::uint64_t> scale_shape = {output_rows, 1};
      GEM16_CHECK(weight->storage_dtype == "F8_E4M3");
      GEM16_CHECK(weight->shape == weight_shape);
      GEM16_CHECK(weight->logical_shape == weight_shape);
      GEM16_CHECK(weight->quantization_class == "FP8_WEIGHT_E4M3");
      GEM16_CHECK(weight->quantization_component == "weight");
      GEM16_CHECK(weight->quantization_producer == "gem16");
      GEM16_CHECK(weight->tensor_role == role);
      GEM16_CHECK(weight->expected_role == role);
      GEM16_CHECK(weight->residency_class == "immutable_device_text");
      GEM16_CHECK(weight->source_family == "gem16_compiled_hybrid");
      GEM16_CHECK(weight->local_scale_dtype == "BF16");
      GEM16_CHECK(weight->local_scale_vector_size == input_columns);
      GEM16_CHECK(weight->activation_scale_role ==
                  "dynamic_per_token_dequant_multiplier");
      GEM16_CHECK(weight->final_gpu_layout == "source_nk_fp8");
      GEM16_CHECK(weight->logical_axis_order == "output,input");
      GEM16_CHECK(weight->layer_index == static_cast<std::int64_t>(layer));
      GEM16_CHECK(weight->local_scale_tensor == scale_name);
      GEM16_CHECK(!weight->aliased);

      GEM16_CHECK(scale->storage_dtype == "BF16");
      GEM16_CHECK(scale->logical_dtype == "BF16");
      GEM16_CHECK(scale->shape == scale_shape);
      GEM16_CHECK(scale->logical_shape == scale_shape);
      GEM16_CHECK(scale->quantization_class == "FP8_WEIGHT_SCALE");
      GEM16_CHECK(scale->quantization_component == "weight_channel_scale");
      GEM16_CHECK(scale->quantization_producer == "gem16");
      GEM16_CHECK(scale->tensor_role == role);
      GEM16_CHECK(scale->expected_role == role);
      GEM16_CHECK(scale->residency_class == "immutable_device_text");
      GEM16_CHECK(scale->source_family == "gem16_compiled_hybrid");
      GEM16_CHECK(scale->local_scale_dtype == "BF16");
      GEM16_CHECK(scale->local_scale_vector_size == input_columns);
      GEM16_CHECK(scale->final_gpu_layout == "row_bf16");
      GEM16_CHECK(scale->logical_axis_order == "output,input");
      GEM16_CHECK(scale->layer_index == static_cast<std::int64_t>(layer));
      GEM16_CHECK(scale->local_scale_tensor.empty());
      GEM16_CHECK(!scale->aliased);
      ++weight_count;
      ++scale_count;
    }
  }

  // The public validator enforces uniqueness, while these counts pin the
  // source-defined attention set and ensure no global V pair is synthesized.
  GEM16_CHECK(expected_projection_count == 115);
  GEM16_CHECK(weight_count == expected_projection_count);
  GEM16_CHECK(scale_count == expected_projection_count);
  std::size_t attention_weight_count = 0;
  std::size_t attention_scale_count = 0;
  for (const auto& tensor : tensors) {
    if (tensor.name.find(".self_attn.") == std::string::npos) continue;
    if (tensor.name.size() >= 13 &&
        tensor.name.ends_with("_proj.weight")) {
      ++attention_weight_count;
    }
    if (tensor.name.size() >= 19 &&
        tensor.name.ends_with("_proj.weight_scale")) {
      ++attention_scale_count;
    }
  }
  GEM16_CHECK(attention_weight_count == expected_projection_count);
  GEM16_CHECK(attention_scale_count == expected_projection_count);
  GEM16_CHECK(gem16::internal::ValidateGemma4Moe26BCompiledHybridInventory(
                  tensors, head_format)
                  .ok());
}

gem16::internal::ModelConfig ExternalUnslothConfig(
    gem16::internal::ModelConfig config) {
  config.quant_method = "compressed-tensors";
  config.quant_format = "mixed-precision";
  config.quantization_status = "compressed";
  config.quantization_version = "0.17.2.a20260707";
  gem16::internal::QuantizationRule fp8;
  fp8.group_name = "group_0";
  fp8.format = "float-quantized";
  fp8.regex_targets = {".*self_attn\\.(q|k|v|o)_proj$"};
  fp8.weight_bits = 8;
  fp8.activation_bits = 8;
  fp8.weight_strategy = "channel";
  fp8.activation_strategy = "token";
  fp8.activation_dynamic = true;
  gem16::internal::QuantizationRule nvfp4;
  nvfp4.group_name = "group_1";
  nvfp4.format = "nvfp4-pack-quantized";
  nvfp4.regex_targets = {
      ".*\\.experts\\.\\d+\\.(gate|up|down)_proj$",
      ".*language_model.*\\.mlp\\.(gate|up|down)_proj$"};
  nvfp4.weight_bits = 4;
  nvfp4.activation_bits = 4;
  nvfp4.group_size = 16;
  nvfp4.scale_dtype = "torch.float8_e4m3fn";
  nvfp4.weight_strategy = "tensor_group";
  nvfp4.activation_strategy = "tensor_group";
  nvfp4.activation_dynamic_local = true;
  config.quantization_rules = {std::move(fp8), std::move(nvfp4)};
  return config;
}

}  // namespace

void RunGemma426BManifestTests() {
  const auto fixture = std::filesystem::path(__FILE__).parent_path().parent_path() /
                       "fixtures" / "gemma4_26b_config.json";
  auto config = gem16::internal::LoadModelConfig(fixture);
  GEM16_CHECK(config.ok());
  if (!config.ok()) return;

  auto source_contract =
      gem16::internal::BuildGemma4Moe26BSourceBf16Contract();
  GEM16_CHECK(source_contract.ok());
  if (!source_contract.ok()) return;
  GEM16_CHECK(source_contract.value().size() == 1013);
  auto source = source_contract.value();
  auto source_profile = gem16::internal::ValidateAndAnnotateGemma4Moe26BInventory(
      config.value(), &source);
  GEM16_CHECK(source_profile.ok());
  if (source_profile.ok()) {
    GEM16_CHECK(source_profile.value() ==
                Gemma4Moe26BInventoryProfile::kSourceBf16);
  }
  auto ambiguous_source_config = config.value();
  ambiguous_source_config.quantization_version = "undeclared-producer";
  auto ambiguous_source = source_contract.value();
  GEM16_CHECK(!gem16::internal::ValidateAndAnnotateGemma4Moe26BInventory(
                   ambiguous_source_config, &ambiguous_source)
                   .ok());
  const auto* embedding = Find(
      &source, "model.language_model.embed_tokens.weight");
  GEM16_CHECK(embedding != nullptr && embedding->aliased);
  GEM16_CHECK(embedding != nullptr &&
              embedding->tensor_role == "tied_embedding_and_output");
  const auto* gate_up = Find(
      &source, "model.language_model.layers.0.experts.gate_up_proj");
  GEM16_CHECK(gate_up != nullptr && gate_up->expert_axis == 0);
  GEM16_CHECK(gate_up != nullptr &&
              gate_up->logical_axis_order == "expert,gate_then_up,input");
  GEM16_CHECK(gate_up != nullptr &&
              gate_up->shape ==
                  std::vector<std::uint64_t>({128, 1408, 2816}));

  std::uint64_t vision_bytes = 0;
  std::uint64_t vision_count = 0;
  for (const auto& tensor : source) {
    GEM16_CHECK(!tensor.tensor_role.empty());
    GEM16_CHECK(!tensor.residency_class.empty());
    if (tensor.residency_class == "compile_excluded_vision") {
      vision_bytes += tensor.byte_length;
      ++vision_count;
    }
  }
  GEM16_CHECK(vision_count == 356);
  GEM16_CHECK(vision_bytes == 1'145'588'832);

  auto missing_expert = source_contract.value();
  Remove(&missing_expert,
         "model.language_model.layers.7.experts.down_proj");
  GEM16_CHECK(!gem16::internal::ValidateAndAnnotateGemma4Moe26BInventory(
                   config.value(), &missing_expert)
                   .ok());

  auto swapped_expert_axis = source_contract.value();
  auto* swapped = Find(
      &swapped_expert_axis,
      "model.language_model.layers.0.experts.gate_up_proj");
  GEM16_CHECK(swapped != nullptr);
  if (swapped != nullptr) {
    swapped->shape = {1408, 128, 2816};
  }
  GEM16_CHECK(!gem16::internal::ValidateAndAnnotateGemma4Moe26BInventory(
                   config.value(), &swapped_expert_axis)
                   .ok());

  auto duplicate = source_contract.value();
  duplicate.push_back(duplicate.front());
  GEM16_CHECK(!gem16::internal::ValidateAndAnnotateGemma4Moe26BInventory(
                   config.value(), &duplicate)
                   .ok());

  auto bad_router_shape = source_contract.value();
  auto* router = Find(
      &bad_router_shape,
      "model.language_model.layers.3.router.proj.weight");
  GEM16_CHECK(router != nullptr);
  if (router != nullptr) router->shape = {127, 2816};
  GEM16_CHECK(!gem16::internal::ValidateAndAnnotateGemma4Moe26BInventory(
                   config.value(), &bad_router_shape)
                   .ok());

  auto bad_router_dtype = source_contract.value();
  router = Find(&bad_router_dtype,
                "model.language_model.layers.3.router.per_expert_scale");
  GEM16_CHECK(router != nullptr);
  if (router != nullptr) router->storage_dtype = "F32";
  GEM16_CHECK(!gem16::internal::ValidateAndAnnotateGemma4Moe26BInventory(
                   config.value(), &bad_router_dtype)
                   .ok());

  auto local_missing_v = source_contract.value();
  Remove(&local_missing_v,
         "model.language_model.layers.0.self_attn.v_proj.weight");
  GEM16_CHECK(!gem16::internal::ValidateAndAnnotateGemma4Moe26BInventory(
                   config.value(), &local_missing_v)
                   .ok());

  auto global_with_v = source_contract.value();
  auto global_v = *Find(
      &global_with_v,
      "model.language_model.layers.0.self_attn.v_proj.weight");
  global_v.name = "model.language_model.layers.5.self_attn.v_proj.weight";
  global_with_v.push_back(std::move(global_v));
  GEM16_CHECK(!gem16::internal::ValidateAndAnnotateGemma4Moe26BInventory(
                   config.value(), &global_with_v)
                   .ok());

  auto duplicate_head = source_contract.value();
  auto lm_head = *Find(&duplicate_head,
                       "model.language_model.embed_tokens.weight");
  lm_head.name = "lm_head.weight";
  duplicate_head.push_back(std::move(lm_head));
  GEM16_CHECK(!gem16::internal::ValidateAndAnnotateGemma4Moe26BInventory(
                   config.value(), &duplicate_head)
                   .ok());

  auto missing_vision = source_contract.value();
  Remove(&missing_vision,
         "model.vision_tower.patch_embedder.position_embedding_table");
  GEM16_CHECK(!gem16::internal::ValidateAndAnnotateGemma4Moe26BInventory(
                   config.value(), &missing_vision)
                   .ok());

  auto unexpected_mtp = source_contract.value();
  auto mtp = unexpected_mtp.front();
  mtp.name = "model.mtp.layers.0.weight";
  unexpected_mtp.push_back(std::move(mtp));
  GEM16_CHECK(!gem16::internal::ValidateAndAnnotateGemma4Moe26BInventory(
                   config.value(), &unexpected_mtp)
                   .ok());

  auto external_contract =
      gem16::internal::BuildGemma4Moe26BExternalUnslothNvfp4Contract();
  GEM16_CHECK(external_contract.ok());
  if (!external_contract.ok()) return;
  GEM16_CHECK(external_contract.value().size() == 47478);
  auto external = external_contract.value();
  auto external_config = ExternalUnslothConfig(config.value());
  auto external_profile =
      gem16::internal::ValidateAndAnnotateGemma4Moe26BInventory(
          external_config, &external);
  GEM16_CHECK(external_profile.ok());
  if (external_profile.ok()) {
    GEM16_CHECK(external_profile.value() ==
                Gemma4Moe26BInventoryProfile::kExternalUnslothNvfp4);
  }
  const auto* external_expert = Find(
      &external,
      "model.language_model.layers.29.experts.127.down_proj.weight_packed");
  GEM16_CHECK(external_expert != nullptr);
  GEM16_CHECK(external_expert != nullptr &&
              external_expert->expert_index == 127);
  GEM16_CHECK(external_expert != nullptr &&
              external_expert->quantization_producer.find("llm-compressor") !=
                  std::string::npos);
  GEM16_CHECK(external_expert != nullptr &&
              external_expert->global_scale_role == "divisor");
  GEM16_CHECK(external_expert != nullptr &&
              external_expert->activation_scale_role == "divisor");

  auto missing_external_expert = external_contract.value();
  Remove(&missing_external_expert,
         "model.language_model.layers.29.experts.127.down_proj.weight_packed");
  GEM16_CHECK(!gem16::internal::ValidateAndAnnotateGemma4Moe26BInventory(
                   external_config, &missing_external_expert)
                   .ok());

  auto wrong_external_scale = external_contract.value();
  auto* scale = Find(
      &wrong_external_scale,
      "model.language_model.layers.0.experts.0.gate_proj.weight_scale");
  GEM16_CHECK(scale != nullptr);
  if (scale != nullptr) scale->storage_dtype = "BF16";
  GEM16_CHECK(!gem16::internal::ValidateAndAnnotateGemma4Moe26BInventory(
                   external_config, &wrong_external_scale)
                   .ok());

  auto wrong_producer_config = external_config;
  wrong_producer_config.quantization_version = "unknown";
  auto producer_inventory = external_contract.value();
  GEM16_CHECK(!gem16::internal::ValidateAndAnnotateGemma4Moe26BInventory(
                   wrong_producer_config, &producer_inventory)
                   .ok());

  auto compiled_q4 = gem16::internal::BuildGemma4Moe26BCompiledHybridContract(
      Gemma4Moe26BHeadFormat::kQ4_0);
  auto compiled_nvfp4 =
      gem16::internal::BuildGemma4Moe26BCompiledHybridContract(
          Gemma4Moe26BHeadFormat::kNvfp4);
  GEM16_CHECK(compiled_q4.ok() && compiled_nvfp4.ok());
  if (!compiled_q4.ok() || !compiled_nvfp4.ok()) return;
  GEM16_CHECK(compiled_q4.value().size() == 1282);
  GEM16_CHECK(compiled_nvfp4.value().size() == 1285);
  GEM16_CHECK(gem16::internal::ValidateGemma4Moe26BCompiledHybridInventory(
                  compiled_q4.value(), Gemma4Moe26BHeadFormat::kQ4_0)
                  .ok());
  GEM16_CHECK(gem16::internal::ValidateGemma4Moe26BCompiledHybridInventory(
                  compiled_nvfp4.value(), Gemma4Moe26BHeadFormat::kNvfp4)
                  .ok());
  CheckCompiledFp8AttentionBinding(compiled_q4.value(),
                                   Gemma4Moe26BHeadFormat::kQ4_0);
  CheckCompiledFp8AttentionBinding(compiled_nvfp4.value(),
                                   Gemma4Moe26BHeadFormat::kNvfp4);
  const auto* head_packed = Find(
      &compiled_nvfp4.value(),
      "model.language_model.embed_tokens.weight_packed");
  const auto* head_scale = Find(
      &compiled_nvfp4.value(),
      "model.language_model.embed_tokens.weight_scale");
  const auto* head_global = Find(
      &compiled_nvfp4.value(),
      "model.language_model.embed_tokens.weight_global_scale");
  const auto* head_input = Find(
      &compiled_nvfp4.value(),
      "model.language_model.embed_tokens.input_global_scale");
  GEM16_CHECK(head_packed != nullptr &&
              head_packed->final_gpu_layout == "sm120_row8_k64");
  GEM16_CHECK(head_scale != nullptr &&
              head_scale->final_gpu_layout ==
                  "sm120_row8_group16_e4m3");
  GEM16_CHECK(head_global != nullptr &&
              head_global->final_gpu_layout == "scalar_f32");
  GEM16_CHECK(head_input != nullptr &&
              head_input->final_gpu_layout == "scalar_f32");

  auto wrong_fp8_weight_dtype = compiled_q4.value();
  auto* compiled_q = Find(
      &wrong_fp8_weight_dtype,
      "model.language_model.layers.0.self_attn.q_proj.weight");
  GEM16_CHECK(compiled_q != nullptr);
  if (compiled_q != nullptr) compiled_q->storage_dtype = "BF16";
  GEM16_CHECK(!gem16::internal::ValidateGemma4Moe26BCompiledHybridInventory(
                   wrong_fp8_weight_dtype, Gemma4Moe26BHeadFormat::kQ4_0)
                   .ok());

  auto wrong_fp8_scale_dtype = compiled_q4.value();
  auto* compiled_q_scale = Find(
      &wrong_fp8_scale_dtype,
      "model.language_model.layers.0.self_attn.q_proj.weight_scale");
  GEM16_CHECK(compiled_q_scale != nullptr);
  if (compiled_q_scale != nullptr) compiled_q_scale->storage_dtype = "F32";
  GEM16_CHECK(!gem16::internal::ValidateGemma4Moe26BCompiledHybridInventory(
                   wrong_fp8_scale_dtype, Gemma4Moe26BHeadFormat::kQ4_0)
                   .ok());

  auto wrong_fp8_scale_shape = compiled_q4.value();
  compiled_q_scale = Find(
      &wrong_fp8_scale_shape,
      "model.language_model.layers.0.self_attn.q_proj.weight_scale");
  GEM16_CHECK(compiled_q_scale != nullptr);
  if (compiled_q_scale != nullptr) {
    compiled_q_scale->shape = {4096, 2};
    compiled_q_scale->logical_shape = {4096, 2};
  }
  GEM16_CHECK(!gem16::internal::ValidateGemma4Moe26BCompiledHybridInventory(
                   wrong_fp8_scale_shape, Gemma4Moe26BHeadFormat::kQ4_0)
                   .ok());

  auto missing_fp8_scale = compiled_q4.value();
  Remove(&missing_fp8_scale,
         "model.language_model.layers.0.self_attn.q_proj.weight_scale");
  GEM16_CHECK(!gem16::internal::ValidateGemma4Moe26BCompiledHybridInventory(
                   missing_fp8_scale, Gemma4Moe26BHeadFormat::kQ4_0)
                   .ok());

  auto mismatched_fp8_scale_link = compiled_q4.value();
  compiled_q = Find(&mismatched_fp8_scale_link,
                    "model.language_model.layers.0.self_attn.q_proj.weight");
  GEM16_CHECK(compiled_q != nullptr);
  if (compiled_q != nullptr) {
    compiled_q->local_scale_tensor =
        "model.language_model.layers.1.self_attn.q_proj.weight_scale";
  }
  GEM16_CHECK(!gem16::internal::ValidateGemma4Moe26BCompiledHybridInventory(
                   mismatched_fp8_scale_link, Gemma4Moe26BHeadFormat::kQ4_0)
                   .ok());

  auto fabricated_global_v = compiled_q4.value();
  auto* local_v = Find(
      &fabricated_global_v,
      "model.language_model.layers.0.self_attn.v_proj.weight");
  auto* local_v_scale = Find(
      &fabricated_global_v,
      "model.language_model.layers.0.self_attn.v_proj.weight_scale");
  GEM16_CHECK(local_v != nullptr && local_v_scale != nullptr);
  if (local_v != nullptr && local_v_scale != nullptr) {
    auto fabricated_v = *local_v;
    auto fabricated_v_scale = *local_v_scale;
    fabricated_v.name =
        "model.language_model.layers.5.self_attn.v_proj.weight";
    fabricated_v_scale.name =
        "model.language_model.layers.5.self_attn.v_proj.weight_scale";
    fabricated_v.local_scale_tensor = fabricated_v_scale.name;
    fabricated_v.layer_index = 5;
    fabricated_v_scale.layer_index = 5;
    fabricated_v.shape = {1024, 2816};
    fabricated_v.logical_shape = fabricated_v.shape;
    fabricated_v.byte_length = 1024ULL * 2816ULL;
    fabricated_v_scale.shape = {1024, 1};
    fabricated_v_scale.logical_shape = fabricated_v_scale.shape;
    fabricated_v_scale.byte_length = 1024ULL * 2ULL;
    fabricated_global_v.push_back(std::move(fabricated_v));
    fabricated_global_v.push_back(std::move(fabricated_v_scale));
  }
  GEM16_CHECK(!gem16::internal::ValidateGemma4Moe26BCompiledHybridInventory(
                   fabricated_global_v, Gemma4Moe26BHeadFormat::kQ4_0)
                   .ok());

  auto wrong_gate_up_order = compiled_q4.value();
  auto* compiled_gate_up = Find(
      &wrong_gate_up_order,
      "model.language_model.layers.0.experts.gate_up_proj.weight_packed");
  GEM16_CHECK(compiled_gate_up != nullptr);
  if (compiled_gate_up != nullptr) {
    compiled_gate_up->logical_axis_order = "expert,up_then_gate,input";
  }
  GEM16_CHECK(!gem16::internal::ValidateGemma4Moe26BCompiledHybridInventory(
                   wrong_gate_up_order, Gemma4Moe26BHeadFormat::kQ4_0)
                   .ok());

  auto wrong_scale_direction = compiled_q4.value();
  auto* compiled_scale = Find(
      &wrong_scale_direction,
      "model.language_model.layers.0.experts.down_proj.weight_packed");
  GEM16_CHECK(compiled_scale != nullptr);
  if (compiled_scale != nullptr) compiled_scale->global_scale_role = "multiplier";
  GEM16_CHECK(!gem16::internal::ValidateGemma4Moe26BCompiledHybridInventory(
                   wrong_scale_direction, Gemma4Moe26BHeadFormat::kQ4_0)
                   .ok());

  auto wrong_scale_link = compiled_q4.value();
  compiled_scale = Find(
      &wrong_scale_link,
      "model.language_model.layers.0.experts.down_proj.weight_packed");
  GEM16_CHECK(compiled_scale != nullptr);
  if (compiled_scale != nullptr) {
    compiled_scale->local_scale_tensor =
        "model.language_model.layers.1.experts.down_proj.weight_local_scale";
  }
  GEM16_CHECK(!gem16::internal::ValidateGemma4Moe26BCompiledHybridInventory(
                   wrong_scale_link, Gemma4Moe26BHeadFormat::kQ4_0)
                   .ok());

  auto compiled_duplicate_head = compiled_q4.value();
  auto extra_head = compiled_duplicate_head.front();
  extra_head.name = "lm_head.weight_q4_0";
  compiled_duplicate_head.push_back(std::move(extra_head));
  GEM16_CHECK(!gem16::internal::ValidateGemma4Moe26BCompiledHybridInventory(
                   compiled_duplicate_head, Gemma4Moe26BHeadFormat::kQ4_0)
                   .ok());

  auto compiled_vision = compiled_q4.value();
  auto extra_vision = compiled_vision.front();
  extra_vision.name = "model.vision_tower.weight";
  compiled_vision.push_back(std::move(extra_vision));
  GEM16_CHECK(!gem16::internal::ValidateGemma4Moe26BCompiledHybridInventory(
                   compiled_vision, Gemma4Moe26BHeadFormat::kQ4_0)
                   .ok());

  auto q4_arena = gem16::internal::Gemma4Moe26BAlignedArenaBytes(
      compiled_q4.value());
  auto nvfp4_arena = gem16::internal::Gemma4Moe26BAlignedArenaBytes(
      compiled_nvfp4.value());
  GEM16_CHECK(q4_arena.ok() && q4_arena.value() == 14'696'667'648ULL);
  GEM16_CHECK(nvfp4_arena.ok() &&
              nvfp4_arena.value() == 14'696'668'160ULL);
  GEM16_CHECK(!gem16::internal::Gemma4Moe26BAlignedArenaBytes(
                   compiled_q4.value(), 192)
                   .ok());

  gem16::ModelManifest residency_manifest;
  residency_manifest.model_variant = "gemma4_moe_26b_a4b";
  residency_manifest.checkpoint_profile = "sm120-text-hybrid-v1";
  residency_manifest.validation_contract =
      "gemma4_26b_m08_compiled_hybrid_v1";
  residency_manifest.tensor_contract_validated = true;
  residency_manifest.supports_text = true;
  residency_manifest.tensors = compiled_nvfp4.value();
  std::uint64_t source_offset = 4096U;
  for (auto& tensor : residency_manifest.tensors) {
    tensor.source_shard = "model-00001-of-00016.safetensors";
    tensor.byte_offset = source_offset;
    source_offset += tensor.byte_length;
    residency_manifest.total_tensor_bytes += tensor.byte_length;
  }
  const auto config_fixture = std::filesystem::path(__FILE__).parent_path().parent_path() /
                              "fixtures" / "gemma4_26b_config.json";
  auto residency_config = gem16::internal::LoadModelConfig(config_fixture);
  GEM16_CHECK(residency_config.ok());
  std::uint64_t computed_kv_32k = 0U;
  if (residency_config.ok()) {
    auto attention_traits =
        gem16::internal::BuildGemma4Moe26BAttentionTraits(
            residency_config.value());
    GEM16_CHECK(attention_traits.ok());
    if (attention_traits.ok()) {
      auto kv_32k = gem16::internal::Gemma4Moe26BFp8KvBytes(
          attention_traits.value(), 32768U);
      GEM16_CHECK(kv_32k.ok() && kv_32k.value() == 440'401'920ULL);
      if (kv_32k.ok()) computed_kv_32k = kv_32k.value();
      GEM16_CHECK(gem16::internal::ValidateGemma4Moe26BAttentionBindings(
                      compiled_nvfp4.value(), attention_traits.value())
                      .ok());
      auto unexpected_global_v = compiled_nvfp4.value();
      auto fake_v = unexpected_global_v.front();
      fake_v.name =
          "model.language_model.layers.5.self_attn.v_proj.weight";
      fake_v.storage_dtype = "F8_E4M3";
      fake_v.logical_shape = {1024U, 2816U};
      fake_v.residency_class = "immutable_device_text";
      unexpected_global_v.push_back(std::move(fake_v));
      GEM16_CHECK(!gem16::internal::ValidateGemma4Moe26BAttentionBindings(
                       unexpected_global_v, attention_traits.value())
                       .ok());
    }
  }
  auto residency = residency_config.ok()
      ? gem16::internal::BuildGemma4Moe26BResidencyPlan(
            residency_manifest, residency_config.value())
      : gem16::Result<gem16::internal::Gemma4Moe26BResidencyPlan>(
            residency_config.status());
  GEM16_CHECK(residency.ok());
  if (residency.ok()) {
    GEM16_CHECK(residency.value().upload_ranges.size() == 1285U);
    GEM16_CHECK(residency.value().artifact_payload_bytes ==
                14'696'569'196ULL);
    GEM16_CHECK(residency.value().immutable_weight_arena_bytes ==
                14'696'668'160ULL);
    GEM16_CHECK(residency.value().fixed_region_bytes == 469'762'048ULL);
    GEM16_CHECK(residency.value().fixed_regions.size() == 7U);
    GEM16_CHECK(residency.value().context_profiles.size() == 4U);
    constexpr std::array<std::uint64_t, 4> expected_contexts = {
        8192U, 16384U, 32768U, 65536U};
    constexpr std::array<std::uint64_t, 4> expected_kv = {
        188'743'680ULL, 272'629'760ULL, 440'401'920ULL,
        775'946'240ULL};
    for (std::size_t index = 0; index < expected_contexts.size(); ++index) {
      const auto& profile = residency.value().context_profiles[index];
      GEM16_CHECK(profile.context_tokens == expected_contexts[index]);
      GEM16_CHECK(profile.fp8_kv_bytes == expected_kv[index]);
    }
    const auto& standard = residency.value().context_profiles[2];
    GEM16_CHECK(standard.required_free_margin_bytes == 734'003'200ULL);
    GEM16_CHECK(residency.value().context_profiles[3]
                    .required_free_margin_bytes == 419'430'400ULL);
    GEM16_CHECK(gem16::internal::CheckGemma4Moe26BAdmission(
                    residency.value(), 32768U, standard.admission_bytes, true)
                    .ok());
    auto insufficient = gem16::internal::CheckGemma4Moe26BAdmission(
        residency.value(), 32768U, standard.admission_bytes - 1U, true);
    GEM16_CHECK(!insufficient.ok() &&
                insufficient.code() == gem16::StatusCode::kResourceExhausted);
    GEM16_CHECK(!gem16::internal::CheckGemma4Moe26BAdmission(
                     residency.value(), 12345U, standard.admission_bytes, true)
                     .ok());
  }
  auto unsafe_residency_manifest = residency_manifest;
  unsafe_residency_manifest.tensors.front().source_shard = "../unsafe.safetensors";
  GEM16_CHECK(!gem16::internal::BuildGemma4Moe26BResidencyPlan(
                   unsafe_residency_manifest, residency_config.value())
                   .ok());
  auto modality_residency_manifest = residency_manifest;
  modality_residency_manifest.supports_vision = true;
  GEM16_CHECK(!gem16::internal::BuildGemma4Moe26BResidencyPlan(
                   modality_residency_manifest, residency_config.value())
                   .ok());

  const auto canonical_path =
      std::filesystem::path(__FILE__).parent_path().parent_path() / "fixtures" /
      "gemma4_26b_inventory.json";
  std::ifstream canonical_input(canonical_path, std::ios::binary);
  std::ostringstream canonical_text;
  canonical_text << canonical_input.rdbuf();
  GEM16_CHECK(canonical_input.good() || canonical_input.eof());
  auto canonical = gem16::json::Parse(canonical_text.str());
  GEM16_CHECK(canonical.ok());
  if (canonical.ok()) {
    const auto* compiled = canonical.value().find("compiled_hybrid");
    const auto* profiles = compiled == nullptr ? nullptr : compiled->find("profiles");
    const auto* q4 = profiles == nullptr ? nullptr : profiles->find("q4_0_head");
    const auto* nvfp4 =
        profiles == nullptr ? nullptr : profiles->find("nvfp4_head");
    const auto* q4_bytes =
        q4 == nullptr ? nullptr : q4->find("aligned_weight_arena_bytes");
    const auto* nvfp4_bytes =
        nvfp4 == nullptr ? nullptr : nvfp4->find("aligned_weight_arena_bytes");
    const auto* canonical_kv =
        compiled == nullptr ? nullptr : compiled->find("fp8_kv_32k_bytes");
    GEM16_CHECK(q4_bytes != nullptr && q4_bytes->is_integer() &&
                static_cast<std::uint64_t>(q4_bytes->as_integer()) ==
                    q4_arena.value());
    GEM16_CHECK(nvfp4_bytes != nullptr && nvfp4_bytes->is_integer() &&
                static_cast<std::uint64_t>(nvfp4_bytes->as_integer()) ==
                    nvfp4_arena.value());
    GEM16_CHECK(canonical_kv != nullptr && canonical_kv->is_integer() &&
                static_cast<std::uint64_t>(canonical_kv->as_integer()) ==
                    computed_kv_32k);
  }
}
