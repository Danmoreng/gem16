#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

#include "gem16/status.h"

namespace gem16::internal {

// Correctness-first elementwise bridge between the Gate/Up projections and Down quantization.
// This is Gemma's GELU tanh approximation, multiplied by Up, with no host round trip.
[[nodiscard]] Status LaunchGeluTanhProduct(const float* gate,
                                           const float* up,
                                           float* output,
                                           std::uint64_t elements,
                                           cudaStream_t stream);

[[nodiscard]] Status LaunchAddResidual(const float* mlp_output,
                                       const float* residual,
                                       float* output,
                                       std::uint64_t elements,
                                       cudaStream_t stream);

}  // namespace gem16::internal
