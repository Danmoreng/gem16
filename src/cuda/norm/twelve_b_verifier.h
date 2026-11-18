#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

#include "gem16/status.h"

namespace gem16::internal {

// Finish a 12B verifier layer and prepare the following layer's FP8 input.
// The residual is materialized at the original BF16 boundary. Both row norms
// retain the original FP32 reduction order and BF16 rounding.
[[nodiscard]] Status LaunchTwelveBVerifierPostMlpInputFp8(
    const std::uint16_t* down_bf16, const std::uint16_t* post_mlp_weight,
    const float* residual, const std::uint16_t* layer_scalar,
    const std::uint16_t* next_input_weight, float* hidden,
    std::uint8_t* fp8, float* scales, std::uint64_t tokens, cudaStream_t stream);

}  // namespace gem16::internal
