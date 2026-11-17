#pragma once

#include <charconv>
#include <cstdint>
#include <optional>
#include <string_view>
#include <system_error>

#include "gem16/status.h"

namespace gem16::internal {

// Explicit benchmark control, resolved before reserving any inference arena.
// The automatic path and the BF16 correctness profile keep their old policy.
inline Result<std::uint64_t> ResolveTwelveBPrefillChunkTokens(
    std::uint64_t automatic_tokens, bool checkpoint_fp8,
    std::optional<std::string_view> requested) {
  if (!requested.has_value()) return automatic_tokens;
  if (!checkpoint_fp8) {
    return Status(StatusCode::kUnsupported,
                  "12B prefill chunk override requires checkpoint FP8 KV");
  }
  std::uint64_t tokens = 0U;
  const auto text = *requested;
  if (text.empty() || text.size() > 20U) {
    return Status(StatusCode::kInvalidArgument,
                  "12B prefill chunk override must be an unsigned integer");
  }
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), tokens);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
      tokens < 32U || tokens > 8192U || tokens > automatic_tokens ||
      tokens % 32U != 0U) {
    return Status(StatusCode::kInvalidArgument,
                  "12B prefill chunk override must be a 32-token multiple "
                  "within the automatic limit and at most 8192");
  }
  return tokens;
}

}  // namespace gem16::internal
