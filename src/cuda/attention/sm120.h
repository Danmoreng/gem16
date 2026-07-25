#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

#include "gem16gb/status.h"

namespace gem16gb::internal {

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
    cudaStream_t stream);

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

}  // namespace gem16gb::internal
