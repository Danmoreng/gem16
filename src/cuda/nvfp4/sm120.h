#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

#include "gem16gb/status.h"

namespace gem16gb::internal {

// Native SM120a W4A4 projection. Packed operands remain in low-nibble-first source layout. Weight
// scale bytes must use TileSm120Nvfp4WeightScales' exact 8-row/K64 runtime order; activation
// scales remain row-major compact E4M3FN groups of 16.
[[nodiscard]] Status LaunchNvfp4Sm120DirectProjection(
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

[[nodiscard]] Status LaunchNvfp4Sm120DirectProjectionBatch(
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

// Closed batch-one Gate/Up operator for Gemma's native NVFP4 path. It reuses the activation MMA
// fragment for both projections, performs the required BF16 boundaries and GELU-tanh product in
// the projection epilogue. Gate/Up outputs may both be null when state diagnostics are disabled.
[[nodiscard]] Status LaunchNvfp4Sm120FusedGateUp(
    const std::uint8_t* packed_activation_e2m1,
    const std::uint8_t* activation_scales_e4m3fn,
    const std::uint8_t* packed_gate_weight_e2m1,
    const std::uint8_t* gate_weight_scales_e4m3fn,
    const std::uint8_t* packed_up_weight_e2m1,
    const std::uint8_t* up_weight_scales_e4m3fn,
    float* gate_output,
    float* up_output,
    float* product_output,
    std::uint64_t rows,
    std::uint64_t contracting_elements,
    float gate_activation_global_divisor,
    float gate_weight_global_divisor,
    float up_activation_global_divisor,
    float up_weight_global_divisor,
    cudaStream_t stream);

[[nodiscard]] Status LaunchNvfp4Sm120FusedGateUpBatch(
    const std::uint8_t* packed_activation_e2m1,
    const std::uint8_t* activation_scales_e4m3fn,
    const std::uint8_t* packed_gate_weight_e2m1,
    const std::uint8_t* gate_weight_scales_e4m3fn,
    const std::uint8_t* packed_up_weight_e2m1,
    const std::uint8_t* up_weight_scales_e4m3fn,
    float* gate_output,
    float* up_output,
    float* product_output,
    std::uint64_t tokens,
    std::uint64_t rows,
    std::uint64_t contracting_elements,
    float gate_activation_global_divisor,
    float gate_weight_global_divisor,
    float up_activation_global_divisor,
    float up_weight_global_divisor,
    cudaStream_t stream);

}  // namespace gem16gb::internal
