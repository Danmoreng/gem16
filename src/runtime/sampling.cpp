#include "gem16/sampling.h"

#include <cmath>

namespace gem16 {

Status ValidateSamplingOptions(const SamplingOptions& options,
                               std::uint32_t vocabulary) {
  if (!options.enabled) return Status::Ok();
  if (vocabulary == 0U || !std::isfinite(options.temperature) ||
      options.temperature <= 0.0F || !std::isfinite(options.top_p) ||
      options.top_p <= 0.0F || options.top_p > 1.0F ||
      !std::isfinite(options.min_p) || options.min_p < 0.0F ||
      options.min_p > 1.0F || !std::isfinite(options.repetition_penalty) ||
      options.repetition_penalty <= 0.0F || options.top_k > vocabulary) {
    return Status(StatusCode::kInvalidArgument,
                  "sampling requires a positive vocabulary and temperature, "
                  "top-p in (0,1], min-p in [0,1], repetition penalty > 0, "
                  "and top-k no larger than the vocabulary");
  }
  return Status::Ok();
}

}  // namespace gem16
