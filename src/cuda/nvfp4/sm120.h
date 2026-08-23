#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

#include "gem16/status.h"

namespace gem16::internal {

struct Gemma4MoePrefillAssignment;

// Native SM120a W4A4 projection. Packed weight and weight-scale bytes must use
// TileSm120Nvfp4Weights/TileSm120Nvfp4WeightScales' exact 8-row/K64 runtime order. Nibbles remain
// low-first within each packed byte; activation values and scales remain compact row-major.
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

// Batch-one projection with the model's recurrent BF16 boundary written into
// an FP32 container. This is used by the native 26B MoE decode path so its
// state semantics remain identical to the M11 reference implementation.
[[nodiscard]] Status LaunchNvfp4Sm120DirectProjectionBf16Float(
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

// Exact batched form of the batch-one kernel. Each token follows the same K64
// MMA accumulation and BF16 epilogue as decode while sharing one CUDA launch.
[[nodiscard]] Status LaunchNvfp4Sm120DirectProjectionBf16FloatExactBatch(
    const std::uint8_t* packed_activation_e2m1,
    const std::uint8_t* activation_scales_e4m3fn,
    const std::uint8_t* packed_weight_e2m1,
    const std::uint8_t* weight_scales_e4m3fn, float* output,
    std::uint64_t tokens, std::uint64_t rows,
    std::uint64_t contracting_elements, float activation_global_divisor,
    float weight_global_divisor, cudaStream_t stream);

// Device-selected batch-one projection over an expert-major row8/K64 matrix.
// selected_ids remains device-resident; output points at the caller-selected
// slot and contains rows_per_expert BF16-rounded values in FP32 containers.
[[nodiscard]] Status LaunchNvfp4Sm120SelectedDirectProjectionBf16Float(
    const std::uint8_t* packed_activation_e2m1,
    const std::uint8_t* activation_scales_e4m3fn,
    const std::uint8_t* packed_expert_weight_e2m1,
    const std::uint8_t* expert_weight_scales_e4m3fn,
    const std::uint32_t* selected_ids,
    std::uint32_t slot,
    float* output,
    std::uint64_t rows_per_expert,
    std::uint64_t contracting_elements,
    std::uint32_t experts,
    float activation_global_divisor,
    float weight_global_divisor,
    cudaStream_t stream);

// Batch-one decode projection for all selected Top-K experts in one launch.
// The activation is shared and output remains slot-major.
[[nodiscard]] Status LaunchNvfp4Sm120SelectedDirectProjectionBf16FloatBatch(
    const std::uint8_t* packed_activation_e2m1,
    const std::uint8_t* activation_scales_e4m3fn,
    const std::uint8_t* packed_expert_weight_e2m1,
    const std::uint8_t* expert_weight_scales_e4m3fn,
    const std::uint32_t* selected_ids,
    std::uint32_t top_k,
    float* output,
    std::uint64_t rows_per_expert,
    std::uint64_t contracting_elements,
    std::uint32_t experts,
    float activation_global_divisor,
    float weight_global_divisor,
    cudaStream_t stream);

// Grouped W2. Products are in stable expert-grouped order; the epilogue
// scatters rows back to original token/top-k assignment order.
[[nodiscard]] Status LaunchNvfp4Sm120GroupedExpertDown(
    const std::uint8_t* grouped_product_e2m1,
    const std::uint8_t* grouped_product_scales_e4m3fn,
    const std::uint8_t* packed_expert_weight_e2m1,
    const std::uint8_t* expert_weight_scales_e4m3fn,
    const Gemma4MoePrefillAssignment* assignments,
    const std::uint32_t* permutation, const std::uint32_t* prefix,
    const std::uint32_t* expert_tiles,
    const std::uint32_t* expert_tile_count, float* assignment_order_output,
    std::uint64_t assignment_count, std::uint64_t rows_per_expert,
    std::uint64_t contracting_elements, std::uint32_t experts,
    float activation_global_divisor, float weight_global_divisor,
    cudaStream_t stream);

// Physical-BF16-output form of grouped W2. It preserves the identical MMA,
// expert grouping, assignment scatter and BF16 epilogue while using two bytes
// rather than a four-byte FP32 container for every output value.
[[nodiscard]] Status LaunchNvfp4Sm120GroupedExpertDownBf16(
    const std::uint8_t* grouped_product_e2m1,
    const std::uint8_t* grouped_product_scales_e4m3fn,
    const std::uint8_t* packed_expert_weight_e2m1,
    const std::uint8_t* expert_weight_scales_e4m3fn,
    const Gemma4MoePrefillAssignment* assignments,
    const std::uint32_t* permutation, const std::uint32_t* prefix,
    const std::uint32_t* expert_tiles,
    const std::uint32_t* expert_tile_count,
    std::uint16_t* assignment_order_output_bf16,
    std::uint64_t assignment_count, std::uint64_t rows_per_expert,
    std::uint64_t contracting_elements, std::uint32_t experts,
    float activation_global_divisor, float weight_global_divisor,
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

// Production Gate/Up prefill projection. It writes the model's required BF16
// boundary directly, avoiding an FP32 workspace round trip before GELU.
[[nodiscard]] Status LaunchNvfp4Sm120DirectProjectionBf16Batch(
    const std::uint8_t* packed_activation_e2m1,
    const std::uint8_t* activation_scales_e4m3fn,
    const std::uint8_t* packed_weight_e2m1,
    const std::uint8_t* weight_scales_e4m3fn,
    std::uint16_t* output_bf16,
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

[[nodiscard]] Status LaunchNvfp4Sm120FusedGateUpExactBatch(
    const std::uint8_t* packed_activation_e2m1,
    const std::uint8_t* activation_scales_e4m3fn,
    const std::uint8_t* packed_gate_weight_e2m1,
    const std::uint8_t* gate_weight_scales_e4m3fn,
    const std::uint8_t* packed_up_weight_e2m1,
    const std::uint8_t* up_weight_scales_e4m3fn, float* product_output,
    std::uint64_t tokens, std::uint64_t rows,
    std::uint64_t contracting_elements,
    float gate_activation_global_divisor, float gate_weight_global_divisor,
    float up_activation_global_divisor, float up_weight_global_divisor,
    cudaStream_t stream);

// Device-selected fused Gate/Up over one expert-major matrix whose per-expert
// row order is [gate rows, up rows]. No selected ID crosses the host boundary.
[[nodiscard]] Status LaunchNvfp4Sm120SelectedFusedGateUp(
    const std::uint8_t* packed_activation_e2m1,
    const std::uint8_t* activation_scales_e4m3fn,
    const std::uint8_t* packed_expert_gate_up_weight_e2m1,
    const std::uint8_t* expert_gate_up_weight_scales_e4m3fn,
    const std::uint32_t* selected_ids,
    std::uint32_t slot,
    float* gate_output,
    float* up_output,
    float* product_output,
    std::uint64_t rows,
    std::uint64_t contracting_elements,
    std::uint32_t experts,
    float activation_global_divisor,
    float weight_global_divisor,
    cudaStream_t stream);

[[nodiscard]] Status LaunchNvfp4Sm120SelectedFusedGateUpBatch(
    const std::uint8_t* packed_activation_e2m1,
    const std::uint8_t* activation_scales_e4m3fn,
    const std::uint8_t* packed_expert_gate_up_weight_e2m1,
    const std::uint8_t* expert_gate_up_weight_scales_e4m3fn,
    const std::uint32_t* selected_ids,
    std::uint32_t top_k,
    float* gate_output,
    float* up_output,
    float* product_output,
    std::uint64_t rows,
    std::uint64_t contracting_elements,
    std::uint32_t experts,
    float activation_global_divisor,
    float weight_global_divisor,
    cudaStream_t stream);

// Grouped W13 over stable expert-grouped assignments. The product is emitted
// in grouped order for immediate batched quantization and W2 consumption.
[[nodiscard]] Status LaunchNvfp4Sm120GroupedExpertFusedGateUp(
    const std::uint8_t* token_activation_e2m1,
    const std::uint8_t* token_activation_scales_e4m3fn,
    const std::uint8_t* packed_expert_gate_up_weight_e2m1,
    const std::uint8_t* expert_gate_up_weight_scales_e4m3fn,
    const Gemma4MoePrefillAssignment* assignments,
    const std::uint32_t* permutation, const std::uint32_t* prefix,
    const std::uint32_t* expert_tiles,
    const std::uint32_t* expert_tile_count, float* grouped_product_output,
    std::uint64_t assignment_count, std::uint64_t rows,
    std::uint64_t contracting_elements, std::uint32_t experts,
    float activation_global_divisor, float weight_global_divisor,
    cudaStream_t stream);

// Physical-BF16-output form of grouped W13. The Gate, GELU and product
// boundaries are unchanged; only the representation of the already-rounded
// product changes from an FP32 container to its exact two-byte BF16 payload.
[[nodiscard]] Status LaunchNvfp4Sm120GroupedExpertFusedGateUpBf16(
    const std::uint8_t* token_activation_e2m1,
    const std::uint8_t* token_activation_scales_e4m3fn,
    const std::uint8_t* packed_expert_gate_up_weight_e2m1,
    const std::uint8_t* expert_gate_up_weight_scales_e4m3fn,
    const Gemma4MoePrefillAssignment* assignments,
    const std::uint32_t* permutation, const std::uint32_t* prefix,
    const std::uint32_t* expert_tiles,
    const std::uint32_t* expert_tile_count,
    std::uint16_t* grouped_product_output_bf16,
    std::uint64_t assignment_count, std::uint64_t rows,
    std::uint64_t contracting_elements, std::uint32_t experts,
    float activation_global_divisor, float weight_global_divisor,
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

}  // namespace gem16::internal
