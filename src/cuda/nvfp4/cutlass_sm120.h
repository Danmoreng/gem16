#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime_api.h>

#include "gem16/status.h"

namespace gem16::internal {

// CUTLASS SM120 block-scaled MMA consumes scale factors in a padded 128x4
// interleave. This conversion writes only to the caller-owned prefill arena.
[[nodiscard]] Status LaunchNvfp4CutlassInterleaveActivationScales(
    const std::uint8_t* compact_scales_e4m3fn,
    std::uint8_t* interleaved_scales_e4m3fn,
    std::uint64_t tokens,
    std::uint64_t contracting_elements,
    cudaStream_t stream);

// Reorders one projection from the decode-optimized Row8/K64 arena layout into
// reusable prefill scratch, then launches the SM120 warp-specialized CUTLASS
// block-scaled GEMM. No persistent second weight copy is retained.
[[nodiscard]] Status LaunchNvfp4CutlassProjectionBf16Batch(
    const std::uint8_t* packed_activation_e2m1,
    const std::uint8_t* interleaved_activation_scales_e4m3fn,
    const std::uint8_t* tiled_weight_e2m1,
    const std::uint8_t* tiled_weight_scales_e4m3fn,
    std::uint8_t* row_major_weight_scratch_e2m1,
    std::uint8_t* interleaved_weight_scale_scratch_e4m3fn,
    void* cutlass_workspace,
    std::size_t cutlass_workspace_bytes,
    std::uint16_t* output_bf16,
    std::uint64_t tokens,
    std::uint64_t rows,
    std::uint64_t contracting_elements,
    float activation_global_divisor,
    float weight_global_divisor,
    cudaStream_t stream);

// Runs the Up projection while consuming the already materialized BF16 Gate
// projection in the epilogue. The exact BF16 GELU/product boundary is
// quantized directly to row-major E2M1 payload and CUTLASS-interleaved E4M3
// scales for the following Down projection.
[[nodiscard]] Status LaunchNvfp4CutlassUpGatedGeluQuantizedBatch(
    const std::uint8_t* packed_activation_e2m1,
    const std::uint8_t* interleaved_activation_scales_e4m3fn,
    const std::uint8_t* tiled_up_weight_e2m1,
    const std::uint8_t* tiled_up_weight_scales_e4m3fn,
    std::uint8_t* row_major_weight_scratch_e2m1,
    std::uint8_t* interleaved_weight_scale_scratch_e4m3fn,
    void* cutlass_workspace,
    std::size_t cutlass_workspace_bytes,
    const std::uint16_t* gate_bf16,
    std::uint8_t* product_packed_e2m1,
    std::uint8_t* product_interleaved_scales_e4m3fn,
    std::uint64_t tokens,
    std::uint64_t rows,
    std::uint64_t contracting_elements,
    float up_activation_global_divisor,
    float up_weight_global_divisor,
    float product_global_divisor,
    const float* product_global_divisor_device,
    cudaStream_t stream);

}  // namespace gem16::internal
