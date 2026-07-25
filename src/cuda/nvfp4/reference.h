#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

#include "gem16gb/status.h"

namespace gem16gb::internal {

// Correctness-only CUDA route. These launchers deliberately make no native-NVFP4 performance
// claim and are never a fallback for the future SM120 MMA path.
[[nodiscard]] Status LaunchNvfp4ReferenceActivationQuantization(
    const float* input,
    std::uint8_t* packed_e2m1,
    std::uint8_t* block_scales_e4m3fn,
    std::uint64_t elements,
    float global_divisor,
    cudaStream_t stream);

// Production prefill boundary: Gemma RMSNorm, BF16 state rounding, and
// blockwise NVFP4 activation quantization in one CTA per token.
[[nodiscard]] Status LaunchRmsNormNvfp4ActivationQuantizationBatch(
    const float* input,
    const std::uint16_t* weight_bf16,
    std::uint8_t* packed_e2m1,
    std::uint8_t* block_scales_e4m3fn,
    std::uint64_t tokens,
    std::uint64_t elements_per_token,
    float epsilon,
    float global_divisor,
    cudaStream_t stream);

// Production prefill MLP boundary. Round Gate and Up to BF16, apply Gemma's
// tanh GELU product with its BF16 GELU boundary, round the product to BF16,
// and quantize directly into NVFP4 blocks.
[[nodiscard]] Status LaunchGatedGeluNvfp4ActivationQuantization(
    const float* gate,
    const float* up,
    std::uint8_t* packed_e2m1,
    std::uint8_t* block_scales_e4m3fn,
    std::uint64_t elements,
    float global_divisor,
    cudaStream_t stream);

// Same production boundary when Gate and Up already carry the required BF16
// projection epilogue values.
[[nodiscard]] Status LaunchGatedGeluNvfp4ActivationQuantizationBf16(
    const std::uint16_t* gate_bf16,
    const std::uint16_t* up_bf16,
    std::uint8_t* packed_e2m1,
    std::uint8_t* block_scales_e4m3fn,
    std::uint64_t elements,
    float global_divisor,
    cudaStream_t stream);

[[nodiscard]] Status LaunchNvfp4ReferenceProjection(
    const std::uint8_t* packed_activation_e2m1,
    const std::uint8_t* activation_scales_e4m3fn,
    const std::uint8_t* packed_weight_e2m1,
    const std::uint8_t* weight_scales_e4m3fn,
    float* output,
    std::uint64_t rows,
    std::uint64_t contracting_elements,
    float activation_global_divisor,
    float weight_global_divisor,
    cudaStream_t stream);

[[nodiscard]] Status LaunchNvfp4ReferenceProjectionBatch(
    const std::uint8_t* packed_activation_e2m1,
    const std::uint8_t* activation_scales_e4m3fn,
    const std::uint8_t* packed_weight_e2m1,
    const std::uint8_t* weight_scales_e4m3fn,
    float* output,
    std::uint64_t tokens,
    std::uint64_t rows,
    std::uint64_t contracting_elements,
    float activation_global_divisor,
    float weight_global_divisor,
    cudaStream_t stream);

}  // namespace gem16gb::internal
