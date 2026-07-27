#pragma once

#include <cstdint>

namespace gem16 {

struct SamplingOptions {
  // Disabled preserves the existing greedy output-head path exactly.
  bool enabled = false;
  float temperature = 1.0F;
  float top_p = 1.0F;
  float min_p = 0.0F;
  std::uint32_t top_k = 0;
  float repetition_penalty = 1.0F;
  std::uint64_t seed = 0;
};

}  // namespace gem16
