#pragma once

#include <cstdint>
#include <span>

#include "gem16/status.h"

namespace gem16::internal {

struct Gemma4Moe26BVisionGrid {
  std::uint32_t width = 0U;
  std::uint32_t height = 0U;
};

// Validates the host-visible Vision grid before any input is copied to the
// device. Positions must enumerate the complete rectangle exactly once in
// canonical row-major order.
[[nodiscard]] Result<Gemma4Moe26BVisionGrid>
ValidateGemma4Moe26BVisionGrid(
    std::uint32_t raw_patch_count, std::uint32_t soft_token_count,
    std::span<const std::int32_t> positions);

}  // namespace gem16::internal
