#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>
#include "gem16/status.h"

namespace gem16::internal {
// Fixed 12B T1 boundaries. No allocation, precision change or weight transform.
Status LaunchTwelveBOrdinaryInputNormFp8(const float* input,
    const std::uint16_t* weight, std::uint8_t* output, float* scale,
    cudaStream_t stream);
Status LaunchTwelveBOrdinaryAttentionQuantize(const float* input,
    std::uint8_t* output, float* scale, std::uint64_t elements,
    cudaStream_t stream);
Status LaunchTwelveBOrdinaryPostAttention(float* projection,
    const std::uint16_t* post_weight, const float* residual, float* post_output,
    float* hidden_output, const std::uint16_t* pre_weight,
    std::uint8_t* packed, std::uint8_t* scales, float divisor,
    cudaStream_t stream);
Status LaunchTwelveBOrdinaryPostMlp(const float* input,
    const std::uint16_t* weight, const float* residual, float* post_output,
    float* output, const std::uint16_t* scalar, cudaStream_t stream);
}  // namespace gem16::internal
