#include "model/gemma4_26b_vision_contract.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace gem16::internal {
namespace {

Status Invalid(const char* message) {
  return Status(StatusCode::kInvalidArgument, message);
}

}  // namespace

Result<Gemma4Moe26BVisionGrid> ValidateGemma4Moe26BVisionGrid(
    std::uint32_t raw_patch_count, std::uint32_t soft_token_count,
    std::span<const std::int32_t> positions) {
  if (raw_patch_count == 0U || soft_token_count == 0U ||
      raw_patch_count != soft_token_count * 9U ||
      positions.size() != static_cast<std::size_t>(raw_patch_count) * 2U) {
    return Invalid("Gemma 4 26B Vision grid counts are invalid");
  }
  std::int32_t maximum_x = -1;
  std::int32_t maximum_y = -1;
  for (std::uint32_t patch = 0U; patch < raw_patch_count; ++patch) {
    const std::int32_t x = positions[patch * 2U];
    const std::int32_t y = positions[patch * 2U + 1U];
    if (x < 0 || y < 0 || x >= 10240 || y >= 10240) {
      return Invalid("Gemma 4 26B Vision patch position is out of range");
    }
    maximum_x = std::max(maximum_x, x);
    maximum_y = std::max(maximum_y, y);
  }
  const std::uint32_t width = static_cast<std::uint32_t>(maximum_x + 1);
  const std::uint32_t height = static_cast<std::uint32_t>(maximum_y + 1);
  if (width % 3U != 0U || height % 3U != 0U ||
      static_cast<std::uint64_t>(width) * height != raw_patch_count ||
      static_cast<std::uint64_t>(width / 3U) * (height / 3U) !=
          soft_token_count) {
    return Invalid("Gemma 4 26B Vision positions do not form a divisible grid");
  }
  for (std::uint32_t patch = 0U; patch < raw_patch_count; ++patch) {
    const std::int32_t expected_x =
        static_cast<std::int32_t>(patch % width);
    const std::int32_t expected_y =
        static_cast<std::int32_t>(patch / width);
    if (positions[patch * 2U] != expected_x ||
        positions[patch * 2U + 1U] != expected_y) {
      return Invalid(
          "Gemma 4 26B Vision positions must be canonical row-major");
    }
  }
  return Gemma4Moe26BVisionGrid{width, height};
}

}  // namespace gem16::internal
