#pragma once

#include <cstdint>
#include <filesystem>
#include <string_view>

#include "gem16/status.h"

namespace gem16::internal {

// Artifact identity, resolved and frozen before CUDA initialization. This is
// deliberately not a kernel-policy switch: one engine owns exactly one routed
// expert representation for its complete lifetime.
enum class Gemma4Moe26BRoutedExpertFormat : std::uint8_t {
  kNvfp4,
  kTrellis35,
};

[[nodiscard]] constexpr bool IsTrellis35RoutedExpertFormat(
    Gemma4Moe26BRoutedExpertFormat format) {
  return format == Gemma4Moe26BRoutedExpertFormat::kTrellis35;
}

[[nodiscard]] constexpr std::string_view Gemma4Moe26BRoutedExpertFormatName(
    Gemma4Moe26BRoutedExpertFormat format) {
  return IsTrellis35RoutedExpertFormat(format) ? "trellis35" : "nvfp4";
}

// Marker inspection is host-only and fail-closed. It rejects missing,
// symlinked, and mixed format identities without parsing or allocating CUDA
// state. ResolveValidated additionally validates the selected format's exact
// artifact metadata.
[[nodiscard]] Result<Gemma4Moe26BRoutedExpertFormat>
DetectGemma4Moe26BRoutedExpertFormat(
    const std::filesystem::path& model_directory);

[[nodiscard]] Status ValidateGemma4Moe26BRoutedExpertFormat(
    Gemma4Moe26BRoutedExpertFormat actual,
    Gemma4Moe26BRoutedExpertFormat expected);

[[nodiscard]] Result<Gemma4Moe26BRoutedExpertFormat>
ResolveValidatedGemma4Moe26BRoutedExpertFormat(
    const std::filesystem::path& model_directory);

}  // namespace gem16::internal
