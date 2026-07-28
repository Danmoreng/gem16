#include "test.h"

#include <chrono>
#include <filesystem>
#include <fstream>

#include "model/config.h"

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
