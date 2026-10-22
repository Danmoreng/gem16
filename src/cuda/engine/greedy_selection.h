#pragma once

#include <cstdint>
#include <span>

namespace gem16::internal {

// Only the active suppression IDs belong in this span, never padded storage.
// With the same post-softcap ordering and tie break, an allowed global winner
// is also the winner after removing suppressed candidates.
inline bool GreedyWinnerIsAllowed(
    std::uint32_t winner,
    std::span<const std::uint32_t> suppressed_token_ids) {
  for (const std::uint32_t token : suppressed_token_ids) {
    if (token == winner) return false;
  }
  return true;
}

}  // namespace gem16::internal
