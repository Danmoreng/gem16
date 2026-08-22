#include "model/model_variant.h"

#include "model/config.h"

namespace gem16::internal {
namespace {

constexpr ModelVariantTraits kUnsupportedTraits{};
constexpr ModelVariantTraits kGemma4Unified12BTraits{
    .variant = ModelVariant::kGemma4Unified12B,
    .name = "gemma4_unified_12b",
    .layer_count = 48,
    .inspectable = true,
    .executable = true,
    .supports_text = true,
    .supports_vision = true,
    .supports_audio = true,
    .supports_video = false,
    .supports_mtp = true,
};
constexpr ModelVariantTraits kGemma4Moe26BA4BTraits{
    .variant = ModelVariant::kGemma4Moe26BA4B,
    .name = "gemma4_moe_26b_a4b",
    .layer_count = 30,
    .inspectable = true,
    .executable = true,
    .supports_text = true,
    .supports_vision = false,
    .supports_audio = false,
    .supports_video = false,
    .supports_mtp = false,
};
constexpr ModelVariantTraits kGemma4UnifiedAssistantTraits{
    .variant = ModelVariant::kGemma4UnifiedAssistant,
    .name = "gemma4_unified_assistant",
    .layer_count = 4,
    .inspectable = true,
    .executable = true,
    .supports_text = true,
    .supports_vision = false,
    .supports_audio = false,
    .supports_video = false,
    .supports_mtp = true,
};

}  // namespace

ModelVariant ClassifyModelVariant(const ModelConfig& config) {
  if (config.architecture == "Gemma4UnifiedForConditionalGeneration" &&
      config.model_type == "gemma4_unified" &&
      config.text_model_type == "gemma4_unified_text") {
    return ModelVariant::kGemma4Unified12B;
  }
  if (config.architecture == "Gemma4ForConditionalGeneration" &&
      config.model_type == "gemma4" && config.text_model_type == "gemma4_text" &&
      config.enable_moe_block) {
    return ModelVariant::kGemma4Moe26BA4B;
  }
  if (config.architecture == "Gemma4UnifiedAssistantForCausalLM" &&
      config.model_type == "gemma4_unified_assistant" &&
      config.text_model_type == "gemma4_unified_text") {
    return ModelVariant::kGemma4UnifiedAssistant;
  }
  return ModelVariant::kUnsupported;
}

const ModelVariantTraits& TraitsForModelVariant(ModelVariant variant) {
  switch (variant) {
    case ModelVariant::kGemma4Unified12B:
      return kGemma4Unified12BTraits;
    case ModelVariant::kGemma4Moe26BA4B:
      return kGemma4Moe26BA4BTraits;
    case ModelVariant::kGemma4UnifiedAssistant:
      return kGemma4UnifiedAssistantTraits;
    case ModelVariant::kUnsupported:
      return kUnsupportedTraits;
  }
  return kUnsupportedTraits;
}

std::string_view ModelVariantName(ModelVariant variant) {
  return TraitsForModelVariant(variant).name;
}

}  // namespace gem16::internal
