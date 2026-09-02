#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string_view>
#include <vector>

#include "gem16/status.h"

namespace gem16 {

struct ImageSourceIdentity {
  // SHA-256 of the original encoded bytes. An all-zero digest with zero bytes
  // denotes an in-process image without an authoritative source identity.
  std::array<std::uint8_t, 32> sha256{};
  std::uint64_t encoded_bytes = 0U;

  [[nodiscard]] bool available() const {
    if (encoded_bytes == 0U) return false;
    for (const std::uint8_t byte : sha256) {
      if (byte != 0U) return true;
    }
    return false;
  }

  bool operator==(const ImageSourceIdentity&) const = default;
};

struct VisionImage {
  // Row-major [patch, 48, 48, RGB], already rescaled to [0, 1].
  std::vector<float> patches;
  // Row-major [patch, xy].
  std::vector<std::int32_t> positions;
  std::uint32_t patch_count = 0U;
  std::uint32_t source_width = 0U;
  std::uint32_t source_height = 0U;
  std::uint32_t processed_width = 0U;
  std::uint32_t processed_height = 0U;
  std::uint32_t soft_token_budget = 280U;
  // Stable identity of the original encoded payload. Resident adapters use
  // this to retain the already-prefilled representation when a later request
  // recomputes a different aggregate image budget.
  ImageSourceIdentity source_identity;

  bool operator==(const VisionImage&) const = default;
};

struct VisionImageOptions {
  std::uint32_t maximum_soft_tokens = 280U;
  bool allow_upscale = false;
};

// Processed input for the Gemma 4 26B Vision tower. Unlike VisionImage, this
// keeps the teacher's individual 16x16 RGB patches. Nine adjacent raw patches
// become one text-side soft token only after the 26B Vision encoder.
struct Gemma4Moe26BVisionImage {
  // Row-major [raw_patch, 16, 16, RGB], rescaled to [0, 1]. Padding to the
  // selected 630/1,260/2,520-row capacity is an execution detail, not host
  // payload.
  std::vector<float> patches;
  // Row-major [raw_patch, xy].
  std::vector<std::int32_t> positions;
  std::uint32_t raw_patch_count = 0U;
  std::uint32_t soft_token_count = 0U;
  std::uint32_t source_width = 0U;
  std::uint32_t source_height = 0U;
  std::uint32_t processed_width = 0U;
  std::uint32_t processed_height = 0U;
  std::uint32_t soft_token_budget = 280U;
  ImageSourceIdentity source_identity;
  double decode_milliseconds = 0.0;
  double resize_patchify_milliseconds = 0.0;

  bool operator==(const Gemma4Moe26BVisionImage&) const = default;
};

// Optional diagnostic wall-clock boundaries for the host image processor.
// Filesystem I/O is excluded. Resize includes target-size selection and the
// bicubic transform; patchify includes allocation, /255 conversion, positions,
// and row-major 16x16 extraction.
struct Gemma4Moe26BVisionPreprocessTimings {
  double decode_milliseconds = 0.0;
  double resize_milliseconds = 0.0;
  double patchify_milliseconds = 0.0;
  double total_milliseconds = 0.0;
};

struct Gemma4Moe26BVisionImageOptions {
  // The pinned processor accepts 70, 140, or 280 for this v1 profile.
  std::uint32_t maximum_soft_tokens = 280U;
  // Caller-owned optional V10 diagnostic sink. Normal product calls leave it
  // null and retain the existing preprocessing path.
  Gemma4Moe26BVisionPreprocessTimings* timings = nullptr;
};

[[nodiscard]] std::uint32_t AutomaticVisionSoftTokenBudget(
    std::uint64_t context_tokens, std::uint64_t reserved_non_image_tokens,
    std::size_t image_count);

// Decodes the first frame and applies the pinned Gemma 4 Unified image
// processor: RGB, aspect-preserving antialiased bicubic resize, /255 rescale,
// 16x16 teacher patches, and 3x3 merge into 48x48 model patches.
[[nodiscard]] Result<VisionImage> LoadVisionImage(
    const std::filesystem::path& path,
    const VisionImageOptions& options = {});
[[nodiscard]] Result<VisionImage> LoadVisionImageBytes(
    std::span<const std::uint8_t> encoded, std::string_view source_name,
    const VisionImageOptions& options = {});

// Pinned Gemma 4 26B preprocessing: RGB, aspect-preserving antialiased
// bicubic resize (including reference upscaling), /255 rescale, and row-major
// 16x16 teacher patches. The returned host payload contains only real rows;
// the CUDA tower supplies the specified zero/-1 padding to 2,520 rows.
[[nodiscard]] Result<Gemma4Moe26BVisionImage>
LoadGemma4Moe26BVisionImage(
    const std::filesystem::path& path,
    const Gemma4Moe26BVisionImageOptions& options = {});
[[nodiscard]] Result<Gemma4Moe26BVisionImage>
LoadGemma4Moe26BVisionImageBytes(
    std::span<const std::uint8_t> encoded, std::string_view source_name,
    const Gemma4Moe26BVisionImageOptions& options = {});

}  // namespace gem16
