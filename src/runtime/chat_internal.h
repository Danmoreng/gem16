#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "gem16/chat.h"

namespace gem16::internal {

// Compares public conversation identity while allowing an image decoded from
// the same original payload to carry a different derived resize budget. The
// resident session continues with its canonical, already-prefilled image.
[[nodiscard]] bool ResidentMessageEquivalent(const GenerationMessage& cached,
                                              const GenerationMessage& supplied);
[[nodiscard]] std::vector<std::uint32_t> ExtractReasoningTokenIds(
    std::span<const std::uint32_t> token_ids,
    const GenerationTokenControls& controls);

}  // namespace gem16::internal
