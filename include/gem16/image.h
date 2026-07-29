#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

#include "gem16/status.h"

namespace gem16 {

struct VisionImage {
  // Row-major [patch, 48, 48, RGB], already rescaled to [0, 1].
  std::vector<float> patches;
  // Row-major [patch, xy].
  std::vector<std::int32_t> positions;
  std::uint32_t patch_count = 0U;
  std::uint32_t source_width = 0U;
  std::uint32_t source_height = 0U;

  bool operator==(const VisionImage&) const = default;
};

// Decodes the first frame and applies the pinned Gemma 4 Unified image
// processor: RGB, aspect-preserving antialiased bicubic resize, /255 rescale,
// 16x16 teacher patches, and 3x3 merge into 48x48 model patches.
[[nodiscard]] Result<VisionImage> LoadVisionImage(
    const std::filesystem::path& path);

}  // namespace gem16
