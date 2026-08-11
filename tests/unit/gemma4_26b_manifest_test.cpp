#include "test.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string_view>

#include "model/config.h"
#include "model/gemma4_26b_manifest.h"
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

void Remove(std::vector<TensorInfo>* tensors, std::string_view name) {
  const auto found = std::find_if(
      tensors->begin(), tensors->end(),
      [name](const TensorInfo& tensor) { return tensor.name == name; });
  if (found != tensors->end()) tensors->erase(found);
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
  auto kv_32k = gem16::internal::Gemma4Moe26B32KFp8KvBytes();
  GEM16_CHECK(q4_arena.ok() && q4_arena.value() == 14'696'667'648ULL);
  GEM16_CHECK(nvfp4_arena.ok() &&
              nvfp4_arena.value() == 14'696'668'160ULL);
  GEM16_CHECK(kv_32k.ok() && kv_32k.value() == 440'401'920ULL);
  GEM16_CHECK(!gem16::internal::Gemma4Moe26BAlignedArenaBytes(
                   compiled_q4.value(), 192)
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
                    kv_32k.value());
  }
}
