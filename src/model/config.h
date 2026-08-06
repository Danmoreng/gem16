#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "gem16/status.h"
#include "model/model_variant.h"

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
  ModelVariant variant = ModelVariant::kUnsupported;
  std::string architecture;
  std::string model_type;
  std::string text_model_type;
  std::string hidden_activation;
  std::string bidirectional_attention_mode;
  std::string local_rope_type;
  std::string global_rope_type;
  std::string vision_model_type;
  std::string quant_method;
  std::string quant_format;
  std::vector<std::string> ignored_modules;
  std::vector<QuantizationRule> quantization_rules;
  std::uint64_t hidden_size = 0;
  std::uint64_t backbone_hidden_size = 0;
  std::uint64_t intermediate_size = 0;
  std::uint64_t moe_intermediate_size = 0;
  std::uint64_t expert_count = 0;
  std::uint64_t top_k_experts = 0;
  std::uint64_t hidden_size_per_layer_input = 0;
  std::uint64_t vocabulary_size_per_layer_input = 0;
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
  std::uint64_t audio_embedding_dimension = 0;
  std::uint64_t audio_token_id = 0;
  std::uint64_t begin_audio_token_id = 0;
  std::uint64_t end_audio_token_id = 0;
  std::uint64_t vision_embedding_dimension = 0;
  std::uint64_t vision_position_count = 0;
  std::uint64_t vision_soft_token_count = 0;
  std::uint64_t vision_patch_size = 0;
  std::uint64_t vision_pooling_kernel_size = 0;
  std::uint64_t image_token_id = 0;
  std::uint64_t begin_image_token_id = 0;
  std::uint64_t end_image_token_id = 0;
  std::uint64_t video_token_id = 0;
  std::uint64_t vision_hidden_size = 0;
  std::uint64_t vision_intermediate_size = 0;
  std::uint64_t vision_layer_count = 0;
  std::uint64_t vision_attention_heads = 0;
  std::uint64_t vision_kv_heads = 0;
  std::uint64_t vision_head_dimension = 0;
  std::uint64_t vision_max_positions = 0;
  std::uint64_t vision_position_embedding_size = 0;
  std::uint64_t vision_default_output_length = 0;
  bool enable_moe_block = false;
  bool attention_k_eq_v = false;
  bool attention_bias = false;
  bool tied_embeddings = false;
  bool ordered_embeddings = false;
  bool use_cache = false;
  bool use_double_wide_mlp = false;
  bool shared_kv_layer_count_present = false;
  bool hidden_size_per_layer_input_present = false;
  bool attention_bias_present = false;
  bool attention_dropout_present = false;
  bool use_double_wide_mlp_present = false;
  bool audio_config_present = false;
  bool audio_config_is_null = false;
  bool has_audio_config = false;
  bool has_vision_config = false;
  bool vision_standardize = false;
  double final_logit_softcap = 0.0;
  double rms_norm_epsilon = 0.0;
  double attention_dropout = 0.0;
  double local_rope_theta = 0.0;
  double global_rope_theta = 0.0;
  double global_partial_rotary_factor = 0.0;
  double audio_rms_norm_epsilon = 0.0;
  double vision_rms_norm_epsilon = 0.0;
  std::vector<std::string> layer_types;
};

[[nodiscard]] Result<ModelConfig> LoadModelConfig(const std::filesystem::path& path);
[[nodiscard]] Status ValidateGemma4Unified12BContract(const ModelConfig& config);
[[nodiscard]] Status ValidateGemma4Moe26BContract(const ModelConfig& config);
[[nodiscard]] Status ValidatePrimaryModelContract(const ModelConfig& config);
[[nodiscard]] Status ValidateAssistantModelContract(const ModelConfig& config);
[[nodiscard]] Status ValidateInspectableModelContract(const ModelConfig& config);
[[nodiscard]] bool IsPrimaryModel(const ModelConfig& config);
[[nodiscard]] bool IsGemma4Moe26BModel(const ModelConfig& config);
[[nodiscard]] bool IsAssistantModel(const ModelConfig& config);

}  // namespace gem16::internal
