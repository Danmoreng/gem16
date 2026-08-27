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
// The inference engine reports the model's physical channel tokens. API
// projection may flatten unexpected reasoning into visible text when the
// caller disabled thinking, while still discarding the channel controls.
[[nodiscard]] ResponseTokenChannel ProjectResponseChannel(
    ResponseTokenChannel channel, bool flatten_reasoning_to_text);
[[nodiscard]] std::vector<std::uint32_t> ExtractReasoningTokenIds(
    std::span<const std::uint32_t> token_ids,
    const GenerationTokenControls& controls,
    bool starts_in_reasoning = false);
[[nodiscard]] std::vector<std::uint32_t> ExtractVisibleTokenIds(
    std::span<const std::uint32_t> token_ids,
    const GenerationTokenControls& controls,
    bool starts_in_reasoning = false,
    bool flatten_reasoning_to_text = false);

}  // namespace gem16::internal
