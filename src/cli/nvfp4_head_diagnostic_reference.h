#pragma once

#include <cstdint>
#include <span>

namespace gem16::internal {

// Canonical row-major low-nibble-first access. `element` is logical, not a
// byte offset; retaining the row stride here prevents second-row aliasing.
inline std::uint8_t Nvfp4HeadNibble(std::span<const std::uint8_t> packed,
                                    std::uint64_t row,
                                    std::uint64_t hidden,
                                    std::uint64_t element) noexcept {
  const std::uint64_t logical = row * hidden + element;
  const auto byte = packed[static_cast<std::size_t>(logical / 2U)];
  return static_cast<std::uint8_t>((byte >> ((logical & 1U) == 0U ? 0U : 4U)) & 0x0FU);
}

}  // namespace gem16::internal
