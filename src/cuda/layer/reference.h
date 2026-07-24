#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

#include "gem16gb/status.h"

namespace gem16gb::internal {

[[nodiscard]] Status LaunchRmsNorm(const float* input,
                                   const std::uint16_t* weight_bf16,
                                   float* output,
                                   std::uint64_t vectors,
                                   std::uint64_t width,
                                   float epsilon,
                                   cudaStream_t stream);

[[nodiscard]] Status LaunchRotaryEmbedding(float* states,
                                           std::uint64_t heads,
                                           std::uint64_t head_dimension,
                                           std::uint64_t rotary_dimensions,
                                           std::uint64_t position,
                                           double theta,
                                           cudaStream_t stream);

[[nodiscard]] Status LaunchProportionalRotaryEmbedding(float* states,
                                                       std::uint64_t heads,
                                                       std::uint64_t head_dimension,
                                                       double rotary_factor,
                                                       std::uint64_t position,
                                                       double theta,
                                                       double scaling_factor,
                                                       cudaStream_t stream);

[[nodiscard]] Status LaunchAppendKv(const float* key,
                                    const float* value,
                                    float* key_cache,
                                    float* value_cache,
                                    std::uint64_t slot,
                                    std::uint64_t kv_heads,
                                    std::uint64_t head_dimension,
                                    cudaStream_t stream);

[[nodiscard]] Status LaunchAppendKvFp8(const float* key,
                                       const float* value,
                                       std::uint8_t* key_cache,
                                       std::uint8_t* value_cache,
                                       const std::uint16_t* key_scale_bf16,
                                       const std::uint16_t* value_scale_bf16,
                                       std::uint64_t slot,
                                       std::uint64_t kv_heads,
                                       std::uint64_t head_dimension,
                                       cudaStream_t stream);

// Correctness-first batch-one decode attention. `scores` is a caller-owned
// workspace of query_heads * tokens floats and is overwritten with probabilities.
// Computes grouped-query attention over the exact cache view supplied by the
// caller. Sliding versus full attention is determined by that view's extent.
[[nodiscard]] Status LaunchLocalAttentionDecode(const float* query,
                                                const float* key_cache,
                                                const float* value_cache,
                                                float* scores,
                                                float* output,
                                                std::uint64_t query_heads,
                                                std::uint64_t kv_heads,
                                                std::uint64_t head_dimension,
                                                std::uint64_t tokens,
                                                cudaStream_t stream,
                                                std::uint64_t cache_capacity = 0,
                                                std::uint64_t first_slot = 0);

[[nodiscard]] Status LaunchLocalAttentionDecodeFp8(
    const float* query,
    const std::uint8_t* key_cache,
    const std::uint8_t* value_cache,
    const std::uint16_t* key_scale_bf16,
    const std::uint16_t* value_scale_bf16,
    float* scores,
    float* output,
    std::uint64_t query_heads,
    std::uint64_t kv_heads,
    std::uint64_t head_dimension,
    std::uint64_t tokens,
    cudaStream_t stream,
    std::uint64_t cache_capacity = 0,
    std::uint64_t first_slot = 0);

[[nodiscard]] Status LaunchScale(float* values,
                                 const std::uint16_t* scalar_bf16,
                                 std::uint64_t elements,
                                 cudaStream_t stream);

[[nodiscard]] Status LaunchRotaryEmbeddingBatch(
    float* states, std::uint64_t tokens, std::uint64_t heads,
    std::uint64_t head_dimension, std::uint64_t rotary_dimensions,
    std::uint64_t start_position, double theta, cudaStream_t stream);

[[nodiscard]] Status LaunchProportionalRotaryEmbeddingBatch(
    float* states, std::uint64_t tokens, std::uint64_t heads,
    std::uint64_t head_dimension, double rotary_factor,
    std::uint64_t start_position, double theta, double scaling_factor,
    cudaStream_t stream);

[[nodiscard]] Status LaunchQuantizeKvFp8Batch(
    const float* key, const float* value, std::uint8_t* key_fp8,
    std::uint8_t* value_fp8, const std::uint16_t* key_scale_bf16,
    const std::uint16_t* value_scale_bf16, std::uint64_t tokens,
    std::uint64_t elements_per_token, cudaStream_t stream);

[[nodiscard]] Status LaunchAppendKvBatch(
    const float* key, const float* value, float* key_cache,
    float* value_cache, std::uint64_t start_position, std::uint64_t tokens,
    std::uint64_t elements_per_token, std::uint64_t cache_capacity,
    cudaStream_t stream);

[[nodiscard]] Status LaunchAppendKvFp8Batch(
    const std::uint8_t* key, const std::uint8_t* value,
    std::uint8_t* key_cache, std::uint8_t* value_cache,
    std::uint64_t start_position, std::uint64_t tokens,
    std::uint64_t elements_per_token, std::uint64_t cache_capacity,
    cudaStream_t stream);

[[nodiscard]] Status LaunchCausalAttentionPrefill(
    const float* query, const float* chunk_key, const float* chunk_value,
    const float* key_cache, const float* value_cache, float* scores,
    float* output, std::uint64_t start_position, std::uint64_t tokens,
    std::uint64_t query_heads, std::uint64_t kv_heads,
    std::uint64_t head_dimension, std::uint64_t cache_capacity,
    bool sliding, cudaStream_t stream);

[[nodiscard]] Status LaunchCausalAttentionPrefillFp8(
    const float* query, const std::uint8_t* chunk_key,
    const std::uint8_t* chunk_value, const std::uint8_t* key_cache,
    const std::uint8_t* value_cache, const std::uint16_t* key_scale_bf16,
    const std::uint16_t* value_scale_bf16, float* scores, float* output,
    std::uint64_t start_position, std::uint64_t tokens,
    std::uint64_t query_heads, std::uint64_t kv_heads,
    std::uint64_t head_dimension, std::uint64_t cache_capacity,
    bool sliding, cudaStream_t stream);

[[nodiscard]] Status LaunchFusedCausalAttentionPrefill(
    const float* query, const float* chunk_key, const float* chunk_value,
    const float* key_cache, const float* value_cache, float* scores,
    float* output, std::uint64_t start_position, std::uint64_t tokens,
    std::uint64_t query_heads, std::uint64_t kv_heads,
    std::uint64_t head_dimension, std::uint64_t cache_capacity,
    bool sliding, cudaStream_t stream);

[[nodiscard]] Status LaunchFusedCausalAttentionPrefillFp8(
    const float* query, const std::uint8_t* chunk_key,
    const std::uint8_t* chunk_value, const std::uint8_t* key_cache,
    const std::uint8_t* value_cache, const std::uint16_t* key_scale_bf16,
    const std::uint16_t* value_scale_bf16, float* scores, float* output,
    std::uint64_t start_position, std::uint64_t tokens,
    std::uint64_t query_heads, std::uint64_t kv_heads,
    std::uint64_t head_dimension, std::uint64_t cache_capacity,
    bool sliding, cudaStream_t stream);

}  // namespace gem16gb::internal
