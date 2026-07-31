#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

#include "gem16/status.h"

namespace gem16::internal {

struct DecodeControl;

// Product-shape local Gemma prefill attention. Q is BF16-valued float storage;
// K/V use the checkpoint's physical E4M3 cache representation and BF16 scale.
// The kernel consumes current-chunk K/V directly, so callers append the chunk
// to the circular cache only after attention completes.
[[nodiscard]] Status LaunchOnlineCausalAttentionPrefillFp8LocalSm120(
    const float* query, const std::uint8_t* chunk_key,
    const std::uint8_t* chunk_value, const std::uint8_t* key_cache,
    const std::uint8_t* value_cache, const std::uint16_t* key_scale_bf16,
    const std::uint16_t* value_scale_bf16, float* output,
    std::uint64_t start_position, std::uint64_t tokens,
    std::uint64_t query_heads, std::uint64_t kv_heads,
    std::uint64_t head_dimension, std::uint64_t cache_capacity,
    cudaStream_t stream, std::uint64_t vision_begin = 0U,
    std::uint64_t vision_end = 0U);

// Product-shape global Gemma prefill attention. Two warps share Q/K/V staging
// and independently accumulate one 256-element half of each 512-element row.
[[nodiscard]] Status LaunchOnlineCausalAttentionPrefillFp8GlobalSm120(
    const float* query, const std::uint8_t* chunk_key,
    const std::uint8_t* chunk_value, const std::uint8_t* key_cache,
    const std::uint8_t* value_cache, const std::uint16_t* key_scale_bf16,
    const std::uint16_t* value_scale_bf16, float* output,
    std::uint64_t start_position, std::uint64_t tokens,
    std::uint64_t query_heads, std::uint64_t kv_heads,
    std::uint64_t head_dimension, std::uint64_t cache_capacity,
    cudaStream_t stream);

// Product-shape batch-one decode attention over the checkpoint's physical
// E4M3 K/V cache. Token ranges are split across CTAs; each CTA computes a
// normalized partial output and FP32 log-sum-exp state without materializing
// the score matrix. A second kernel merges the partials. Global-cache
// capacities of at least 64K use aligned E4M3x4 K/V loads; shorter tiers keep
// the lower-register scalar path. `workspace` requires
// DecodeAttentionWorkspaceElements(max_context) floats.
[[nodiscard]] Status LaunchOnlineAttentionDecodeFp8Sm120(
    const float* query, const std::uint8_t* key_cache,
    const std::uint8_t* value_cache,
    const std::uint16_t* key_scale_bf16,
    const std::uint16_t* value_scale_bf16, float* workspace, float* output,
    const DecodeControl* control, std::uint64_t query_heads,
    std::uint64_t kv_heads, std::uint64_t head_dimension,
    std::uint64_t cache_capacity, bool sliding, cudaStream_t stream);

// Exact three-row global-attention verifier. Historical K/V values are loaded
// once for all rows while each row retains the selected ordinary decode tier's
// independent accumulation, online-softmax, and split-merge order, including
// the aligned E4M3x4 path at 64K and above.
[[nodiscard]] Status LaunchOnlineAttentionDecodeFp8GlobalD2Sm120(
    const float* query, const std::uint8_t* key_cache,
    const std::uint8_t* value_cache,
    const std::uint16_t* key_scale_bf16,
    const std::uint16_t* value_scale_bf16, float* workspace, float* output,
    std::uint64_t start_position, std::uint64_t cache_capacity,
    cudaStream_t stream);

[[nodiscard]] Status LaunchOnlineAttentionDecodeFp8GlobalD2ControlledSm120(
    const float* query, const std::uint8_t* key_cache,
    const std::uint8_t* value_cache,
    const std::uint16_t* key_scale_bf16,
    const std::uint16_t* value_scale_bf16, float* workspace, float* output,
    const DecodeControl* row_controls, std::uint64_t cache_capacity,
    cudaStream_t stream);

// Covers both the local D256/256-token split and global D512/512-token split.
// The returned count includes normalized partial outputs and their LSE values.
[[nodiscard]] std::uint64_t DecodeAttentionWorkspaceElements(
    std::uint64_t max_context);

}  // namespace gem16::internal
