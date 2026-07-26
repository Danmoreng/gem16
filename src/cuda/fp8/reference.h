#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

#include "gem16/status.h"

namespace gem16::internal {

[[nodiscard]] Status LaunchFp8ReferenceTokenQuantization(
    const float* input,
    std::uint8_t* output_e4m3fn,
    float* output_scale,
    std::uint64_t elements,
    cudaStream_t stream);

[[nodiscard]] Status LaunchFp8ReferenceTokenQuantizationBatch(
    const float* input,
    std::uint8_t* output_e4m3fn,
    float* output_scales,
    std::uint64_t tokens,
    std::uint64_t elements_per_token,
    cudaStream_t stream);

// Production prefill boundary: Gemma RMSNorm, BF16 state rounding, and
// dynamic per-token FP8 quantization in one CTA per token.
[[nodiscard]] Status LaunchRmsNormFp8TokenQuantizationBatch(
    const float* input,
    const std::uint16_t* weight_bf16,
    std::uint8_t* output_e4m3fn,
    float* output_scales,
    std::uint64_t tokens,
    std::uint64_t elements_per_token,
    float epsilon,
    cudaStream_t stream);

[[nodiscard]] Status LaunchFp8ReferenceProjection(
    const std::uint8_t* activation_e4m3fn,
    const float* activation_scale,
    const std::uint8_t* weight_e4m3fn,
    const std::uint16_t* weight_scales_bf16,
    float* output,
    std::uint64_t rows,
    std::uint64_t contracting_elements,
    cudaStream_t stream);

[[nodiscard]] Status LaunchFp8ReferenceProjectionBatch(
    const std::uint8_t* activation_e4m3fn,
    const float* activation_scales,
    const std::uint8_t* weight_e4m3fn,
    const std::uint16_t* weight_scales_bf16,
    float* output,
    std::uint64_t tokens,
    std::uint64_t rows,
    std::uint64_t contracting_elements,
    cudaStream_t stream);

}  // namespace gem16::internal
