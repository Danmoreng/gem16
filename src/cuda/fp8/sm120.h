#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

#include "gem16/status.h"

namespace gem16::internal {

// Experimental T=1 direct-source FP8 projection. E4M3 activation and checkpoint weight bytes
// are consumed without a persistent repack; FP32 MMA accumulators are scaled by the dynamic
// per-token FP32 input scale and the checkpoint's per-output-channel BF16 weight scale.
[[nodiscard]] Status LaunchFp8Sm120DirectProjection(
    const std::uint8_t* activation_e4m3fn,
    const float* activation_scale,
    const std::uint8_t* weight_e4m3fn,
    const std::uint16_t* weight_scales_bf16,
    float* output,
    std::uint64_t rows,
    std::uint64_t contracting_elements,
    cudaStream_t stream);

// Batch-one grouped Q/K/V projection. One CUDA launch contains the independent
// direct-source projection CTAs, reducing decode-graph node count without
// changing MMA ordering or projection outputs.
[[nodiscard]] Status LaunchFp8Sm120GroupedQkvProjection(
    const std::uint8_t* activation_e4m3fn,
    const float* activation_scale,
    const std::uint8_t* q_weight_e4m3fn,
    const std::uint16_t* q_weight_scales_bf16,
    float* q_output,
    std::uint64_t q_rows,
    const std::uint8_t* k_weight_e4m3fn,
    const std::uint16_t* k_weight_scales_bf16,
    float* k_output,
    std::uint64_t k_rows,
    const std::uint8_t* v_weight_e4m3fn,
    const std::uint16_t* v_weight_scales_bf16,
    float* v_output,
    std::uint64_t v_rows,
    std::uint64_t contracting_elements,
    cudaStream_t stream);

[[nodiscard]] Status LaunchFp8Sm120DirectProjectionBatch(
    const std::uint8_t* activation_e4m3fn,
    const float* activation_scales,
    const std::uint8_t* weight_e4m3fn,
    const std::uint16_t* weight_scales_bf16,
    float* output,
    std::uint64_t tokens,
    std::uint64_t rows,
    std::uint64_t contracting_elements,
    cudaStream_t stream);

// Q/K/V share the same per-token activation and contracting dimension. Local
// attention launches all three projections together; global attention passes
// null V pointers and launches Q/K together because K is reused as V.
[[nodiscard]] Status LaunchFp8Sm120GroupedQkvProjectionBatch(
    const std::uint8_t* activation_e4m3fn,
    const float* activation_scales,
    const std::uint8_t* q_weight_e4m3fn,
    const std::uint16_t* q_weight_scales_bf16,
    float* q_output,
    std::uint64_t q_rows,
    const std::uint8_t* k_weight_e4m3fn,
    const std::uint16_t* k_weight_scales_bf16,
    float* k_output,
    std::uint64_t k_rows,
    const std::uint8_t* v_weight_e4m3fn,
    const std::uint16_t* v_weight_scales_bf16,
    float* v_output,
    std::uint64_t v_rows,
    std::uint64_t tokens,
    std::uint64_t contracting_elements,
    cudaStream_t stream);

}  // namespace gem16::internal
