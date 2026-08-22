#include "test.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>

#include "gem16/model.h"
#include "model/config.h"
#include "model/gemma4_26b_attention.h"

namespace {

gem16::internal::ModelConfig Unified12BConfig() {
  gem16::internal::ModelConfig config;
  config.architecture = "Gemma4UnifiedForConditionalGeneration";
  config.model_type = "gemma4_unified";
  config.text_model_type = "gemma4_unified_text";
  config.hidden_size = 3840;
  config.intermediate_size = 15360;
  config.layer_count = 48;
  config.vocabulary_size = 262144;
  config.max_positions = 262144;
  config.sliding_window = 1024;
  config.query_heads = 16;
  config.local_kv_heads = 8;
  config.global_kv_heads = 1;
  config.local_head_dimension = 256;
  config.global_head_dimension = 512;
  config.audio_embedding_dimension = 640;
  config.audio_token_id = 258881;
  config.begin_audio_token_id = 256000;
  config.end_audio_token_id = 258883;
  config.audio_rms_norm_epsilon = 1.0e-6;
  config.vision_embedding_dimension = 3840;
  config.vision_position_count = 1120;
  config.vision_soft_token_count = 280;
  config.vision_patch_size = 16;
  config.vision_pooling_kernel_size = 3;
  config.image_token_id = 258880;
  config.begin_image_token_id = 255999;
  config.end_image_token_id = 258882;
  config.vision_rms_norm_epsilon = 1.0e-6;
  config.attention_k_eq_v = true;
  config.tied_embeddings = true;
  config.final_logit_softcap = 30.0;
  config.quant_method = "compressed-tensors";
  config.quant_format = "mixed-precision";
  for (std::size_t layer = 0; layer < 48; ++layer) {
    config.layer_types.push_back(layer % 6U == 5U ? "full_attention"
                                                  : "sliding_attention");
  }
  return config;
}

}  // namespace

void RunConfigTests() {
  gem16::internal::ModelConfig assistant;
  assistant.architecture = "Gemma4UnifiedAssistantForCausalLM";
  assistant.model_type = "gemma4_unified_assistant";
  assistant.text_model_type = "gemma4_unified_text";
  assistant.backbone_hidden_size = 3840;
  assistant.hidden_size = 1024;
  assistant.intermediate_size = 8192;
  assistant.layer_count = 4;
  assistant.vocabulary_size = 262144;
  assistant.max_positions = 262144;
  assistant.sliding_window = 1024;
  assistant.query_heads = 16;
  assistant.local_kv_heads = 8;
  assistant.global_kv_heads = 1;
  assistant.local_head_dimension = 256;
  assistant.global_head_dimension = 512;
  assistant.shared_kv_layer_count = 4;
  assistant.centroid_count = 2048;
  assistant.centroid_intermediate_top_k = 32;
  assistant.tied_embeddings = true;
  assistant.layer_types = {"sliding_attention", "sliding_attention",
                           "sliding_attention", "full_attention"};

  GEM16_CHECK(gem16::internal::IsAssistantModel(assistant));
  GEM16_CHECK(!gem16::internal::IsPrimaryModel(assistant));
  GEM16_CHECK(gem16::internal::ValidateAssistantModelContract(assistant).ok());
  GEM16_CHECK(gem16::internal::ValidateInspectableModelContract(assistant).ok());

  auto invalid = assistant;
  invalid.ordered_embeddings = true;
  GEM16_CHECK(!gem16::internal::ValidateAssistantModelContract(invalid).ok());
  invalid = assistant;
  invalid.layer_types[3] = "sliding_attention";
  GEM16_CHECK(!gem16::internal::ValidateAssistantModelContract(invalid).ok());
  invalid = assistant;
  invalid.hidden_size = 2048;
  GEM16_CHECK(!gem16::internal::ValidateAssistantModelContract(invalid).ok());

  const auto primary = Unified12BConfig();
  GEM16_CHECK(gem16::internal::IsPrimaryModel(primary));
  GEM16_CHECK(!gem16::internal::IsGemma4Moe26BModel(primary));
  GEM16_CHECK(gem16::internal::ValidatePrimaryModelContract(primary).ok());
  GEM16_CHECK(gem16::internal::ValidateGemma4Unified12BContract(primary).ok());
  const auto& primary_traits = gem16::internal::TraitsForModelVariant(
      gem16::internal::ClassifyModelVariant(primary));
  GEM16_CHECK(primary_traits.executable);
  GEM16_CHECK(primary_traits.layer_count == 48);
  GEM16_CHECK(primary_traits.supports_text && primary_traits.supports_vision &&
              primary_traits.supports_audio && primary_traits.supports_mtp);

  const auto fixture = std::filesystem::path(__FILE__).parent_path().parent_path() /
                       "fixtures" / "gemma4_26b_config.json";
  auto parsed_26b = gem16::internal::LoadModelConfig(fixture);
  GEM16_CHECK(parsed_26b.ok());
  if (parsed_26b.ok()) {
    const auto& moe = parsed_26b.value();
    GEM16_CHECK(moe.variant ==
                gem16::internal::ModelVariant::kGemma4Moe26BA4B);
    GEM16_CHECK(gem16::internal::IsGemma4Moe26BModel(moe));
    GEM16_CHECK(!gem16::internal::IsPrimaryModel(moe));
    GEM16_CHECK(gem16::internal::ValidateGemma4Moe26BContract(moe).ok());
    GEM16_CHECK(gem16::internal::ValidateInspectableModelContract(moe).ok());
    const auto& traits = gem16::internal::TraitsForModelVariant(moe.variant);
    GEM16_CHECK(traits.inspectable && !traits.executable);
    GEM16_CHECK(traits.layer_count == 30 && traits.supports_text);
    GEM16_CHECK(!traits.supports_vision && !traits.supports_audio &&
                !traits.supports_video && !traits.supports_mtp);
    GEM16_CHECK(gem16::internal::ModelVariantName(moe.variant) ==
                "gemma4_moe_26b_a4b");
    auto attention_traits =
        gem16::internal::BuildGemma4Moe26BAttentionTraits(moe);
    GEM16_CHECK(attention_traits.ok());
    if (attention_traits.ok()) {
      std::uint32_t sliding = 0U;
      std::uint32_t full = 0U;
      for (std::size_t layer = 0; layer < attention_traits.value().size();
           ++layer) {
        const auto& layer_traits = attention_traits.value()[layer];
        GEM16_CHECK(layer_traits.layer == layer);
        GEM16_CHECK(layer_traits.query_heads == 16U);
        GEM16_CHECK(layer_traits.kv_producer_layer ==
                    static_cast<std::int32_t>(layer));
        if (layer_traits.attention ==
            gem16::internal::Gemma4Moe26BAttentionType::kSliding) {
          ++sliding;
          GEM16_CHECK(layer_traits.kv_heads == 8U);
          GEM16_CHECK(layer_traits.head_dimension == 256U);
          GEM16_CHECK(layer_traits.cache_capacity == 1024U);
          GEM16_CHECK(layer_traits.stores_v_projection);
          GEM16_CHECK(!layer_traits.reuses_raw_k_for_v);
          GEM16_CHECK(layer_traits.rope_theta == 10000.0);
          GEM16_CHECK(layer_traits.rotary_factor == 1.0);
        } else {
          ++full;
          GEM16_CHECK(layer_traits.kv_heads == 2U);
          GEM16_CHECK(layer_traits.head_dimension == 512U);
          GEM16_CHECK(layer_traits.cache_capacity == 262144U);
          GEM16_CHECK(!layer_traits.stores_v_projection);
          GEM16_CHECK(layer_traits.reuses_raw_k_for_v);
          GEM16_CHECK(layer_traits.rope_theta == 1000000.0);
          GEM16_CHECK(layer_traits.rotary_factor == 0.25);
        }
      }
      GEM16_CHECK(sliding == 25U && full == 5U);
      for (const auto& [context, expected] :
           {std::pair{8192ULL, 188743680ULL},
            std::pair{32768ULL, 440401920ULL},
            std::pair{65536ULL, 775946240ULL}}) {
        auto bytes = gem16::internal::Gemma4Moe26BFp8KvBytes(
            attention_traits.value(), context);
        GEM16_CHECK(bytes.ok() && bytes.value() == expected);
      }
      GEM16_CHECK(!gem16::internal::Gemma4Moe26BFp8KvBytes(
                       attention_traits.value(), 0U)
                       .ok());
    }

    auto bad = moe;
    bad.layer_count = 29;
    GEM16_CHECK(!gem16::internal::ValidateGemma4Moe26BContract(bad).ok());
    bad = moe;
    bad.layer_count = 31;
    GEM16_CHECK(!gem16::internal::ValidateGemma4Moe26BContract(bad).ok());
    bad = moe;
    bad.hidden_size = 0;
    GEM16_CHECK(!gem16::internal::ValidateGemma4Moe26BContract(bad).ok());
    bad = moe;
    bad.hidden_size = std::numeric_limits<std::uint64_t>::max();
    GEM16_CHECK(!gem16::internal::ValidateGemma4Moe26BContract(bad).ok());
    bad = moe;
    bad.intermediate_size = 0;
    GEM16_CHECK(!gem16::internal::ValidateGemma4Moe26BContract(bad).ok());
    bad = moe;
    bad.intermediate_size = std::numeric_limits<std::uint64_t>::max();
    GEM16_CHECK(!gem16::internal::ValidateGemma4Moe26BContract(bad).ok());
    bad = moe;
    bad.moe_intermediate_size = 0;
    GEM16_CHECK(!gem16::internal::ValidateGemma4Moe26BContract(bad).ok());
    bad = moe;
    bad.moe_intermediate_size = std::numeric_limits<std::uint64_t>::max();
    GEM16_CHECK(!gem16::internal::ValidateGemma4Moe26BContract(bad).ok());
    bad = moe;
    bad.expert_count = 0;
    GEM16_CHECK(!gem16::internal::ValidateGemma4Moe26BContract(bad).ok());
    bad = moe;
    bad.expert_count = std::numeric_limits<std::uint64_t>::max();
    GEM16_CHECK(!gem16::internal::ValidateGemma4Moe26BContract(bad).ok());
    bad = moe;
    bad.top_k_experts = 0;
    GEM16_CHECK(!gem16::internal::ValidateGemma4Moe26BContract(bad).ok());
    bad = moe;
    bad.top_k_experts = std::numeric_limits<std::uint64_t>::max();
    GEM16_CHECK(!gem16::internal::ValidateGemma4Moe26BContract(bad).ok());
    bad = moe;
    bad.hidden_size_per_layer_input = std::numeric_limits<std::uint64_t>::max();
    GEM16_CHECK(!gem16::internal::ValidateGemma4Moe26BContract(bad).ok());
    bad = moe;
    bad.hidden_size_per_layer_input_present = false;
    GEM16_CHECK(!gem16::internal::ValidateGemma4Moe26BContract(bad).ok());
    bad = moe;
    bad.vocabulary_size_per_layer_input = 0;
    GEM16_CHECK(!gem16::internal::ValidateGemma4Moe26BContract(bad).ok());
    bad = moe;
    bad.vocabulary_size_per_layer_input =
        std::numeric_limits<std::uint64_t>::max();
    GEM16_CHECK(!gem16::internal::ValidateGemma4Moe26BContract(bad).ok());
    bad = moe;
    bad.global_kv_heads = 1;
    GEM16_CHECK(!gem16::internal::ValidateGemma4Moe26BContract(bad).ok());
    bad = moe;
    bad.shared_kv_layer_count = 1;
    GEM16_CHECK(!gem16::internal::ValidateGemma4Moe26BContract(bad).ok());
    bad = moe;
    bad.shared_kv_layer_count_present = false;
    GEM16_CHECK(!gem16::internal::ValidateGemma4Moe26BContract(bad).ok());
    bad = moe;
    bad.attention_bias_present = false;
    GEM16_CHECK(!gem16::internal::ValidateGemma4Moe26BContract(bad).ok());
    bad = moe;
    bad.audio_config_present = false;
    GEM16_CHECK(!gem16::internal::ValidateGemma4Moe26BContract(bad).ok());
    bad = moe;
    bad.layer_types[5] = "sliding_attention";
    GEM16_CHECK(!gem16::internal::ValidateGemma4Moe26BContract(bad).ok());
    GEM16_CHECK(!gem16::internal::BuildGemma4Moe26BAttentionTraits(bad).ok());
    bad = moe;
    bad.local_rope_theta = 1000000.0;
    GEM16_CHECK(!gem16::internal::ValidateGemma4Moe26BContract(bad).ok());
    bad = moe;
    bad.vision_hidden_size = 0;
    GEM16_CHECK(!gem16::internal::ValidateGemma4Moe26BContract(bad).ok());
    bad = moe;
    bad.has_audio_config = true;
    GEM16_CHECK(!gem16::internal::ValidateGemma4Moe26BContract(bad).ok());

    auto unknown = moe;
    unknown.enable_moe_block = false;
    unknown.variant = gem16::internal::ClassifyModelVariant(unknown);
    GEM16_CHECK(unknown.variant == gem16::internal::ModelVariant::kUnsupported);
    const auto unknown_status =
        gem16::internal::ValidateInspectableModelContract(unknown);
    GEM16_CHECK(!unknown_status.ok());
    GEM16_CHECK(unknown_status.message().find(
                    "inspectable but not executable Gemma 4 variant") !=
                std::string::npos);

    gem16::ModelManifest manifest;
    manifest.architecture = moe.architecture;
    manifest.model_type = moe.model_type;
    manifest.model_variant = std::string(gem16::internal::ModelVariantName(moe.variant));
    manifest.layer_count = moe.layer_count;
    manifest.hidden_size = moe.hidden_size;
    manifest.intermediate_size = moe.intermediate_size;
    manifest.moe_intermediate_size = moe.moe_intermediate_size;
    manifest.expert_count = moe.expert_count;
    manifest.top_k_experts = moe.top_k_experts;
    manifest.supports_text = true;
    std::ostringstream json;
    GEM16_CHECK(gem16::WriteManifestJson(manifest, json).ok());
    GEM16_CHECK(json.str().find("\"schema_version\": 3") != std::string::npos);
    GEM16_CHECK(json.str().find("\"model_variant\": \"gemma4_moe_26b_a4b\"") !=
                std::string::npos);
    GEM16_CHECK(json.str().find("\"moe_intermediate_size\": 704") !=
                std::string::npos);
    GEM16_CHECK(json.str().find("\"runtime_supported\": false") !=
                std::string::npos);
    GEM16_CHECK(json.str().find("\"capabilities\": {\"text\":true,\"vision\":false") !=
                std::string::npos);
  }

  const auto unique =
      std::chrono::steady_clock::now().time_since_epoch().count();
  const auto path = std::filesystem::temp_directory_path() /
                    ("gem16-assistant-config-test-" + std::to_string(unique) +
                     ".json");
  {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output << R"({
      "architectures":["Gemma4UnifiedAssistantForCausalLM"],
      "model_type":"gemma4_unified_assistant",
      "backbone_hidden_size":3840,
      "num_centroids":2048,
      "centroid_intermediate_top_k":32,
      "use_ordered_embeddings":false,
      "tie_word_embeddings":true,
      "text_config":{
        "model_type":"gemma4_unified_text",
        "hidden_size":1024,
        "intermediate_size":8192,
        "num_hidden_layers":4,
        "vocab_size":262144,
        "max_position_embeddings":262144,
        "sliding_window":1024,
        "num_attention_heads":16,
        "num_key_value_heads":8,
        "num_global_key_value_heads":1,
        "head_dim":256,
        "global_head_dim":512,
        "num_kv_shared_layers":4,
        "tie_word_embeddings":true,
        "layer_types":["sliding_attention","sliding_attention","sliding_attention","full_attention"]
      }
    })";
  }
  auto parsed = gem16::internal::LoadModelConfig(path);
  GEM16_CHECK(parsed.ok());
  if (parsed.ok()) {
    GEM16_CHECK(parsed.value().backbone_hidden_size == 3840);
    GEM16_CHECK(parsed.value().shared_kv_layer_count == 4);
    GEM16_CHECK(parsed.value().centroid_count == 2048);
    GEM16_CHECK(
        gem16::internal::ValidateAssistantModelContract(parsed.value()).ok());
  }
  std::error_code error;
  std::filesystem::remove(path, error);
  GEM16_CHECK(!error);
}
