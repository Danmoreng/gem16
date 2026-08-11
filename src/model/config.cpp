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

Status Moe26BContractError(std::string field, std::string expected) {
  return ContractError("Gemma 4 26B A4B", std::move(field),
                       std::move(expected));
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
  config.hidden_activation =
      StringOrEmpty(Nested(root, "text_config", "hidden_activation"));
  config.bidirectional_attention_mode =
      StringOrEmpty(Nested(root, "text_config", "use_bidirectional_attention"));
  const auto* rope_parameters = Nested(root, "text_config", "rope_parameters");
  const auto* local_rope =
      rope_parameters == nullptr ? nullptr : Member(*rope_parameters, "sliding_attention");
  const auto* global_rope =
      rope_parameters == nullptr ? nullptr : Member(*rope_parameters, "full_attention");
  config.local_rope_type =
      StringOrEmpty(local_rope == nullptr ? nullptr : Member(*local_rope, "rope_type"));
  config.global_rope_type =
      StringOrEmpty(global_rope == nullptr ? nullptr : Member(*global_rope, "rope_type"));
  config.local_rope_theta =
      NumberOrZero(local_rope == nullptr ? nullptr : Member(*local_rope, "rope_theta"));
  config.global_rope_theta =
      NumberOrZero(global_rope == nullptr ? nullptr : Member(*global_rope, "rope_theta"));
  config.global_partial_rotary_factor = NumberOrZero(
      global_rope == nullptr ? nullptr : Member(*global_rope, "partial_rotary_factor"));
  config.hidden_size = UnsignedOrZero(Nested(root, "text_config", "hidden_size"));
  config.backbone_hidden_size = UnsignedOrZero(Member(root, "backbone_hidden_size"));
  config.intermediate_size = UnsignedOrZero(Nested(root, "text_config", "intermediate_size"));
  config.moe_intermediate_size =
      UnsignedOrZero(Nested(root, "text_config", "moe_intermediate_size"));
  config.expert_count = UnsignedOrZero(Nested(root, "text_config", "num_experts"));
  config.top_k_experts = UnsignedOrZero(Nested(root, "text_config", "top_k_experts"));
  config.hidden_size_per_layer_input =
      UnsignedOrZero(Nested(root, "text_config", "hidden_size_per_layer_input"));
  config.hidden_size_per_layer_input_present =
      Nested(root, "text_config", "hidden_size_per_layer_input") != nullptr;
  config.vocabulary_size_per_layer_input =
      UnsignedOrZero(Nested(root, "text_config", "vocab_size_per_layer_input"));
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
  config.shared_kv_layer_count_present =
      Nested(root, "text_config", "num_kv_shared_layers") != nullptr;
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
  config.video_token_id = UnsignedOrZero(Member(root, "video_token_id"));
  const auto* audio_config = Member(root, "audio_config");
  const auto* vision_config = Member(root, "vision_config");
  config.audio_config_present = audio_config != nullptr;
  config.audio_config_is_null = audio_config != nullptr && audio_config->is_null();
  config.has_audio_config = audio_config != nullptr && audio_config->is_object();
  config.has_vision_config = vision_config != nullptr && vision_config->is_object();
  config.vision_model_type = StringOrEmpty(Nested(root, "vision_config", "model_type"));
  config.vision_hidden_size = UnsignedOrZero(Nested(root, "vision_config", "hidden_size"));
  config.vision_intermediate_size =
      UnsignedOrZero(Nested(root, "vision_config", "intermediate_size"));
  config.vision_layer_count =
      UnsignedOrZero(Nested(root, "vision_config", "num_hidden_layers"));
  config.vision_attention_heads =
      UnsignedOrZero(Nested(root, "vision_config", "num_attention_heads"));
  config.vision_kv_heads =
      UnsignedOrZero(Nested(root, "vision_config", "num_key_value_heads"));
  config.vision_head_dimension = UnsignedOrZero(Nested(root, "vision_config", "head_dim"));
  config.vision_max_positions =
      UnsignedOrZero(Nested(root, "vision_config", "max_position_embeddings"));
  config.vision_position_embedding_size =
      UnsignedOrZero(Nested(root, "vision_config", "position_embedding_size"));
  config.vision_default_output_length =
      UnsignedOrZero(Nested(root, "vision_config", "default_output_length"));
  config.vision_standardize = BoolOrFalse(Nested(root, "vision_config", "standardize"));
  config.enable_moe_block = BoolOrFalse(Nested(root, "text_config", "enable_moe_block"));
  config.attention_k_eq_v = BoolOrFalse(Nested(root, "text_config", "attention_k_eq_v"));
  config.attention_bias = BoolOrFalse(Nested(root, "text_config", "attention_bias"));
  config.attention_bias_present =
      Nested(root, "text_config", "attention_bias") != nullptr;
  config.tied_embeddings = BoolOrFalse(Member(root, "tie_word_embeddings")) ||
                           BoolOrFalse(Nested(root, "text_config", "tie_word_embeddings"));
  config.ordered_embeddings = BoolOrFalse(Member(root, "use_ordered_embeddings"));
  config.use_cache = BoolOrFalse(Nested(root, "text_config", "use_cache"));
  config.use_double_wide_mlp =
      BoolOrFalse(Nested(root, "text_config", "use_double_wide_mlp"));
  config.use_double_wide_mlp_present =
      Nested(root, "text_config", "use_double_wide_mlp") != nullptr;
  config.final_logit_softcap = NumberOrZero(Nested(root, "text_config", "final_logit_softcapping"));
  config.rms_norm_epsilon = NumberOrZero(Nested(root, "text_config", "rms_norm_eps"));
  config.attention_dropout =
      NumberOrZero(Nested(root, "text_config", "attention_dropout"));
  config.attention_dropout_present =
      Nested(root, "text_config", "attention_dropout") != nullptr;
  config.audio_rms_norm_epsilon =
      NumberOrZero(Nested(root, "audio_config", "rms_norm_eps"));
  config.vision_rms_norm_epsilon =
      NumberOrZero(Nested(root, "vision_config", "rms_norm_eps"));
  config.layer_types = StringArray(Nested(root, "text_config", "layer_types"));

  const auto* quantization = Member(root, "quantization_config");
  if (quantization != nullptr && quantization->is_object()) {
    config.quant_method = StringOrEmpty(Member(*quantization, "quant_method"));
    config.quant_format = StringOrEmpty(Member(*quantization, "format"));
    config.quantization_status =
        StringOrEmpty(Member(*quantization, "quantization_status"));
    config.quantization_version =
        StringOrEmpty(Member(*quantization, "version"));
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
  config.variant = ClassifyModelVariant(config);
  return config;
}

bool IsPrimaryModel(const ModelConfig& config) {
  return ClassifyModelVariant(config) == ModelVariant::kGemma4Unified12B;
}

bool IsGemma4Moe26BModel(const ModelConfig& config) {
  return ClassifyModelVariant(config) == ModelVariant::kGemma4Moe26BA4B;
}

bool IsAssistantModel(const ModelConfig& config) {
  return ClassifyModelVariant(config) == ModelVariant::kGemma4UnifiedAssistant;
}

Status ValidateGemma4Unified12BContract(const ModelConfig& config) {
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

Status ValidatePrimaryModelContract(const ModelConfig& config) {
  return ValidateGemma4Unified12BContract(config);
}

Status ValidateGemma4Moe26BContract(const ModelConfig& config) {
  if (config.architecture != "Gemma4ForConditionalGeneration") {
    return Moe26BContractError("architecture", "Gemma4ForConditionalGeneration");
  }
  if (config.model_type != "gemma4") {
    return Moe26BContractError("model_type", "gemma4");
  }
  if (config.text_model_type != "gemma4_text") {
    return Moe26BContractError("text_config.model_type", "gemma4_text");
  }
  if (!config.enable_moe_block) {
    return Moe26BContractError("enable_moe_block", "true");
  }
  if (config.layer_count != 30 || config.hidden_size != 2816) {
    return Moe26BContractError("decoder dimensions", "30 layers / hidden size 2816");
  }
  if (config.intermediate_size != 2112 || config.moe_intermediate_size != 704) {
    return Moe26BContractError("MLP dimensions", "2112 shared / 704 routed expert");
  }
  if (config.expert_count != 128 || config.top_k_experts != 8) {
    return Moe26BContractError("expert routing", "128 experts / top-k 8");
  }
  if (config.query_heads != 16 || config.local_kv_heads != 8 ||
      config.global_kv_heads != 2) {
    return Moe26BContractError("attention head counts",
                               "16 query / 8 local KV / 2 global KV");
  }
  if (config.local_head_dimension != 256 || config.global_head_dimension != 512) {
    return Moe26BContractError("head dimensions", "256 local / 512 global");
  }
  if (!config.shared_kv_layer_count_present || config.shared_kv_layer_count != 0) {
    return Moe26BContractError("num_kv_shared_layers", "present and 0");
  }
  if (config.sliding_window != 1024 || config.max_positions != 262144) {
    return Moe26BContractError("context dimensions", "1024 sliding / 262144 maximum");
  }
  if (config.vocabulary_size != 262144 ||
      config.vocabulary_size_per_layer_input != 262144 ||
      !config.hidden_size_per_layer_input_present ||
      config.hidden_size_per_layer_input != 0) {
    return Moe26BContractError(
        "input dimensions", "vocabulary 262144 / per-layer vocabulary 262144 / hidden 0");
  }
  if (!config.tied_embeddings || !config.attention_k_eq_v) {
    return Moe26BContractError("tied embeddings and attention_k_eq_v", "true");
  }
  if (!config.use_cache || !config.use_double_wide_mlp_present ||
      config.use_double_wide_mlp || !config.attention_bias_present ||
      config.attention_bias || !config.attention_dropout_present ||
      std::abs(config.attention_dropout) > 1.0e-12) {
    return Moe26BContractError(
        "attention/MLP controls",
        "use_cache true, double-wide false, bias false, dropout 0");
  }
  if (config.hidden_activation != "gelu_pytorch_tanh" ||
      std::abs(config.rms_norm_epsilon - 1.0e-6) > 1.0e-12 ||
      std::abs(config.final_logit_softcap - 30.0) > 1.0e-12) {
    return Moe26BContractError(
        "numeric controls", "gelu_pytorch_tanh / RMS epsilon 1e-6 / softcap 30");
  }
  if (config.local_rope_type != "default" ||
      std::abs(config.local_rope_theta - 10000.0) > 1.0e-9 ||
      config.global_rope_type != "proportional" ||
      std::abs(config.global_rope_theta - 1000000.0) > 1.0e-7 ||
      std::abs(config.global_partial_rotary_factor - 0.25) > 1.0e-12) {
    return Moe26BContractError(
        "RoPE contract",
        "local default/10000 and global proportional/1000000/0.25");
  }
  constexpr std::array<std::string_view, 30> expected_layers = {
      "sliding_attention", "sliding_attention", "sliding_attention",
      "sliding_attention", "sliding_attention", "full_attention",
      "sliding_attention", "sliding_attention", "sliding_attention",
      "sliding_attention", "sliding_attention", "full_attention",
      "sliding_attention", "sliding_attention", "sliding_attention",
      "sliding_attention", "sliding_attention", "full_attention",
      "sliding_attention", "sliding_attention", "sliding_attention",
      "sliding_attention", "sliding_attention", "full_attention",
      "sliding_attention", "sliding_attention", "sliding_attention",
      "sliding_attention", "sliding_attention", "full_attention"};
  if (config.layer_types.size() != expected_layers.size()) {
    return Moe26BContractError("layer_types length", "30");
  }
  for (std::size_t index = 0; index < expected_layers.size(); ++index) {
    if (config.layer_types[index] != expected_layers[index]) {
      return Moe26BContractError(
          "layer_types pattern",
          "five sliding layers followed by one full layer, repeated five times");
    }
  }
  if (config.bidirectional_attention_mode != "vision" ||
      !config.audio_config_present || !config.audio_config_is_null ||
      config.has_audio_config || !config.has_vision_config ||
      config.audio_token_id != 258881 ||
      config.begin_audio_token_id != 256000 || config.end_audio_token_id != 258883 ||
      config.image_token_id != 258880 || config.begin_image_token_id != 255999 ||
      config.end_image_token_id != 258882 || config.video_token_id != 258884) {
    return Moe26BContractError(
        "root modality metadata",
        "vision bidirectional mode, no audio config and token IDs 258881/256000/258883/258880/255999/258882/258884");
  }
  if (config.vision_model_type != "gemma4_vision" ||
      config.vision_hidden_size != 1152 ||
      config.vision_intermediate_size != 4304 || config.vision_layer_count != 27 ||
      config.vision_attention_heads != 16 || config.vision_kv_heads != 16 ||
      config.vision_head_dimension != 72 || config.vision_max_positions != 131072 ||
      config.vision_position_embedding_size != 10240 ||
      config.vision_default_output_length != 280 || config.vision_patch_size != 16 ||
      config.vision_pooling_kernel_size != 3 || !config.vision_standardize ||
      std::abs(config.vision_rms_norm_epsilon - 1.0e-6) > 1.0e-12) {
    return Moe26BContractError(
        "vision source metadata",
        "gemma4_vision 1152/4304/27/16/16/72/131072/10240/280/16/3/standardized/1e-6");
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
  switch (ClassifyModelVariant(config)) {
    case ModelVariant::kGemma4Unified12B:
      return ValidateGemma4Unified12BContract(config);
    case ModelVariant::kGemma4Moe26BA4B:
      return ValidateGemma4Moe26BContract(config);
    case ModelVariant::kGemma4UnifiedAssistant:
      return ValidateAssistantModelContract(config);
    case ModelVariant::kUnsupported:
      break;
  }
  const bool gemma4_family = config.architecture.starts_with("Gemma4") ||
                             config.model_type.starts_with("gemma4");
  return Status(
      StatusCode::kUnsupported,
      std::string(gemma4_family ? "inspectable but not executable Gemma 4 variant: "
                                : "unsupported checkpoint architecture/model_type: ") +
          config.architecture + " / " + config.model_type + " / " +
          config.text_model_type);
}

}  // namespace gem16::internal
