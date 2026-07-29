#pragma once

#include <algorithm>
#include <cstdint>
#include <span>

#include "gem16/engine.h"

namespace gem16::internal {

inline std::uint64_t PlanVisionAwarePrefillChunk(
    std::uint64_t proposed_begin, std::uint64_t maximum_tokens,
    std::span<const VisionEmbeddingSegment> vision_segments) {
  std::uint64_t tokens = maximum_tokens;
  for (const VisionEmbeddingSegment& segment : vision_segments) {
    const std::uint64_t patch_count = segment.patches.size() / 6912U;
    const std::uint64_t segment_end = segment.prompt_offset + patch_count;
    const std::uint64_t proposed_end = proposed_begin + tokens;
    if (proposed_begin < segment.prompt_offset &&
        proposed_end > segment.prompt_offset) {
      tokens = std::min(tokens, segment.prompt_offset - proposed_begin);
    } else if (proposed_begin >= segment.prompt_offset &&
               proposed_begin < segment_end) {
      return segment_end - proposed_begin;
    }
  }
  return tokens;
}

}  // namespace gem16::internal
