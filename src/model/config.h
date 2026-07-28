#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "gem16/status.h"

namespace gem16::internal {

struct QuantizationRule {
  std::string group_name;
  std::string format;
  std::vector<std::string> regex_targets;
  std::uint32_t weight_bits = 0;
  std::uint32_t activation_bits = 0;
  std::uint32_t group_size = 0;
  std::string scale_dtype;
  std::string weight_strategy;
  std::string activation_strategy;
  bool activation_dynamic = false;
  bool activation_dynamic_local = false;
};

struct ModelConfig {
  std::string architecture;
  std::string model_type;
  std::string text_model_type;
  std::string quant_method;
  std::string quant_format;
  std::vector<std::string> ignored_modules;
  std::vector<QuantizationRule> quantization_rules;
  std::uint64_t hidden_size = 0;
  std::uint64_t backbone_hidden_size = 0;
  std::uint64_t intermediate_size = 0;
  std::uint64_t layer_count = 0;
  std::uint64_t vocabulary_size = 0;
  std::uint64_t max_positions = 0;
  std::uint64_t sliding_window = 0;
  std::uint64_t query_heads = 0;
  std::uint64_t local_kv_heads = 0;
  std::uint64_t global_kv_heads = 0;
  std::uint64_t local_head_dimension = 0;
  std::uint64_t global_head_dimension = 0;
  std::uint64_t shared_kv_layer_count = 0;
  std::uint64_t centroid_count = 0;
  std::uint64_t centroid_intermediate_top_k = 0;
  bool attention_k_eq_v = false;
  bool tied_embeddings = false;
  bool ordered_embeddings = false;
  double final_logit_softcap = 0.0;
  std::vector<std::string> layer_types;
};

[[nodiscard]] Result<ModelConfig> LoadModelConfig(const std::filesystem::path& path);
[[nodiscard]] Status ValidatePrimaryModelContract(const ModelConfig& config);
[[nodiscard]] Status ValidateAssistantModelContract(const ModelConfig& config);
[[nodiscard]] Status ValidateInspectableModelContract(const ModelConfig& config);
[[nodiscard]] bool IsPrimaryModel(const ModelConfig& config);
[[nodiscard]] bool IsAssistantModel(const ModelConfig& config);

}  // namespace gem16::internal

