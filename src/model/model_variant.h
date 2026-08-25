#pragma once

#include <cstdint>
#include <string_view>

namespace gem16::internal {

struct ModelConfig;

enum class ModelVariant : std::uint8_t {
  kUnsupported = 0,
  kGemma4Unified12B,
  kGemma4Moe26BA4B,
  kGemma4UnifiedAssistant,
  kGemma4Moe26BAssistant,
};

struct ModelVariantTraits {
  ModelVariant variant = ModelVariant::kUnsupported;
  std::string_view name = "unsupported";
  std::uint64_t layer_count = 0;
  bool inspectable = false;
  bool executable = false;
  bool supports_text = false;
  bool supports_vision = false;
  bool supports_audio = false;
  bool supports_video = false;
  bool supports_mtp = false;
};

[[nodiscard]] ModelVariant ClassifyModelVariant(const ModelConfig& config);
[[nodiscard]] const ModelVariantTraits& TraitsForModelVariant(
    ModelVariant variant);
[[nodiscard]] std::string_view ModelVariantName(ModelVariant variant);

}  // namespace gem16::internal
