#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

#include "gem16/status.h"

namespace gem16::internal {

struct DecodeControl {
  std::uint32_t token = 0;
  std::uint32_t suppressed_token_count = 0;
  std::uint64_t position = 0;
  std::uint64_t sampling_step = 0;
};

[[nodiscard]] Status LaunchRmsNorm(const float* input,
                                   const std::uint16_t* weight_bf16,
                                   float* output,
                                   std::uint64_t vectors,
                                   std::uint64_t width,
                                   float epsilon,
                                   cudaStream_t stream);

// Production inference stores RMSNorm boundaries as BF16 values in FP32
// workspace slots. Perform that required rounding in the RMSNorm kernel.
[[nodiscard]] Status LaunchRmsNormBf16(const float* input,
                                       const std::uint16_t* weight_bf16,
                                       float* output,
                                       std::uint64_t vectors,
                                       std::uint64_t width,
                                       float epsilon,
                                       cudaStream_t stream);

[[nodiscard]] Status LaunchRmsNormBf16Input(
    const std::uint16_t* input_bf16,
    const std::uint16_t* weight_bf16, float* output,
    std::uint64_t vectors, std::uint64_t width, float epsilon,
    cudaStream_t stream);

// Gemma applies a BF16 RMSNorm boundary, residual addition, another BF16
// boundary, and optionally a BF16 layer scalar plus a final BF16 boundary.
// Preserve that exact ordering in one reduction kernel.
[[nodiscard]] Status LaunchRmsNormResidualBf16(
    const float* input, const std::uint16_t* weight_bf16,
    const float* residual, float* normalized_output, float* output,
    std::uint64_t vectors, std::uint64_t width, float epsilon,
    const std::uint16_t* scalar_bf16, cudaStream_t stream);

// Equivalent boundary for a projection that already materialized BF16 rather
// than BF16 values expanded into FP32 workspace slots.
[[nodiscard]] Status LaunchRmsNormResidualBf16Input(
    const std::uint16_t* input_bf16, const std::uint16_t* weight_bf16,
    const float* residual, float* normalized_output, float* output,
    std::uint64_t vectors, std::uint64_t width, float epsilon,
    const std::uint16_t* scalar_bf16, cudaStream_t stream);

// Production prefill boundary with both the recurrent residual stream and the
// result stored as physical BF16. The arithmetic and BF16 rounding order are
// identical to LaunchRmsNormResidualBf16Input.
[[nodiscard]] Status LaunchRmsNormResidualPhysicalBf16(
    const std::uint16_t* input_bf16,
    const std::uint16_t* weight_bf16,
    const std::uint16_t* residual_bf16,
    std::uint16_t* output_bf16,
    std::uint64_t vectors,
    std::uint64_t width,
    float epsilon,
    const std::uint16_t* scalar_bf16,
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

[[nodiscard]] Status LaunchRotaryEmbeddingControlled(
    float* states, std::uint64_t heads, std::uint64_t head_dimension,
    std::uint64_t rotary_dimensions, const DecodeControl* control,
    double theta, cudaStream_t stream);

[[nodiscard]] Status LaunchProportionalRotaryEmbeddingControlled(
    float* states, std::uint64_t heads, std::uint64_t head_dimension,
    double rotary_factor, const DecodeControl* control, double theta,
    double scaling_factor, cudaStream_t stream);

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

[[nodiscard]] Status LaunchAppendKvControlled(
    const float* key, const float* value, float* key_cache,
    float* value_cache, const DecodeControl* control,
    std::uint64_t kv_heads, std::uint64_t head_dimension,
    std::uint64_t cache_capacity, bool sliding, cudaStream_t stream);

[[nodiscard]] Status LaunchAppendKvFp8Controlled(
    const float* key, const float* value, std::uint8_t* key_cache,
    std::uint8_t* value_cache, const std::uint16_t* key_scale_bf16,
    const std::uint16_t* value_scale_bf16, const DecodeControl* control,
    std::uint64_t kv_heads, std::uint64_t head_dimension,
    std::uint64_t cache_capacity, bool sliding, cudaStream_t stream);

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

[[nodiscard]] Status LaunchLocalAttentionDecodeControlled(
    const float* query, const float* key_cache, const float* value_cache,
    float* scores, float* output, const DecodeControl* control,
    std::uint64_t query_heads, std::uint64_t kv_heads,
    std::uint64_t head_dimension, std::uint64_t cache_capacity,
    bool sliding, cudaStream_t stream);

[[nodiscard]] Status LaunchLocalAttentionDecodeFp8Controlled(
    const float* query, const std::uint8_t* key_cache,
    const std::uint8_t* value_cache, const std::uint16_t* key_scale_bf16,
    const std::uint16_t* value_scale_bf16, float* scores, float* output,
    const DecodeControl* control, std::uint64_t query_heads,
    std::uint64_t kv_heads, std::uint64_t head_dimension,
    std::uint64_t cache_capacity, bool sliding, cudaStream_t stream);

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

// Preserve projection BF16 rounding, per-head BF16 RMSNorm, RoPE, and the
// post-RoPE BF16 boundary in one CTA per token/head. Q and K share one launch
// and one exactly precomputed per-token trigonometric table while retaining
// independent per-head reductions.
[[nodiscard]] Status LaunchRotaryEmbeddingTableBatch(
    float* cosine, float* sine, std::uint64_t tokens,
    std::uint64_t rotating_pairs, std::uint64_t frequency_dimension,
    std::uint64_t start_position, double theta, double scaling_factor,
    cudaStream_t stream);

[[nodiscard]] Status LaunchRotaryEmbeddingTableControlled(
    float* cosine, float* sine, std::uint64_t rotating_pairs,
    std::uint64_t frequency_dimension, const DecodeControl* control,
    double theta, double scaling_factor, cudaStream_t stream);

// Device-controlled counterpart of LaunchRotaryEmbeddingTableBatch. Each row
// reads its position from controls[row], which keeps fixed-D graph replay to a
// single RoPE-table launch without assuming consecutive host-known positions.
[[nodiscard]] Status LaunchRotaryEmbeddingTableBatchControlled(
    float* cosine, float* sine, std::uint64_t tokens,
    std::uint64_t rotating_pairs, std::uint64_t frequency_dimension,
    const DecodeControl* controls, double theta, double scaling_factor,
    cudaStream_t stream);

[[nodiscard]] Status LaunchProjectionRmsNormRotaryBf16Batch(
    const float* query, const std::uint16_t* query_norm_bf16,
    float* normalized_query, const float* key,
    const std::uint16_t* key_norm_bf16, float* normalized_key,
    const float* rotary_cosine, const float* rotary_sine,
    std::uint64_t tokens, std::uint64_t query_heads,
    std::uint64_t kv_heads, std::uint64_t head_dimension,
    double rotary_factor, float epsilon, cudaStream_t stream);

[[nodiscard]] Status LaunchProjectionRmsNormRotaryBf16BatchInput(
    const std::uint16_t* query_bf16,
    const std::uint16_t* query_norm_bf16, float* normalized_query,
    const std::uint16_t* key_bf16,
    const std::uint16_t* key_norm_bf16, float* normalized_key,
    const float* rotary_cosine, const float* rotary_sine,
    std::uint64_t tokens, std::uint64_t query_heads,
    std::uint64_t kv_heads, std::uint64_t head_dimension,
    double rotary_factor, float epsilon, cudaStream_t stream);

[[nodiscard]] Status LaunchProjectionRmsNormRotaryBf16BatchControlled(
    const float* query, const std::uint16_t* query_norm_bf16,
    float* normalized_query, const float* key,
    const std::uint16_t* key_norm_bf16, float* normalized_key,
    const float* rotary_cosine, const float* rotary_sine,
    const DecodeControl* row_controls, std::uint64_t tokens,
    std::uint64_t query_heads, std::uint64_t kv_heads,
    std::uint64_t head_dimension, double rotary_factor, float epsilon,
    cudaStream_t stream);

[[nodiscard]] Status LaunchProjectionRmsNormRotaryBf16Controlled(
    const float* query, const std::uint16_t* query_norm_bf16,
    float* normalized_query, const float* key,
    const std::uint16_t* key_norm_bf16, float* normalized_key,
    const float* rotary_cosine, const float* rotary_sine,
    const DecodeControl* control, std::uint64_t query_heads,
    std::uint64_t kv_heads, std::uint64_t head_dimension,
    double rotary_factor, float epsilon, cudaStream_t stream);

// Single-row counterpart for a table produced by
// LaunchRotaryEmbeddingTableControlled. The table already represents
// control->position and therefore contains no position dimension.
[[nodiscard]] Status LaunchProjectionRmsNormRotaryBf16CurrentTableControlled(
    const float* query, const std::uint16_t* query_norm_bf16,
    float* normalized_query, const float* key,
    const std::uint16_t* key_norm_bf16, float* normalized_key,
    const float* rotary_cosine, const float* rotary_sine,
    const DecodeControl* control, std::uint64_t query_heads,
    std::uint64_t kv_heads, std::uint64_t head_dimension,
    double rotary_factor, float epsilon, cudaStream_t stream);

// Single-row 26B decode epilogue. Q preserves the existing BF16 math;
// K/V are written directly to the controlled cache slot, with no staging.
[[nodiscard]] Status LaunchGemma4Moe26BDecodeKvEpilogue(
    const float* query, const std::uint16_t* query_norm, float* normalized_query,
    const float* key, const std::uint16_t* key_norm, const float* value,
    const float* rotary_cosine, const float* rotary_sine,
    const DecodeControl* control, std::uint8_t* key_cache,
    std::uint8_t* value_cache, const std::uint16_t* key_scale,
    const std::uint16_t* value_scale, std::uint64_t kv_heads,
    std::uint64_t head_dimension, double rotary_factor,
    std::uint64_t cache_capacity, float epsilon, cudaStream_t stream);

// Fixed-depth verifier counterpart. Q retains the existing BF16 boundary;
// K/V are quantized directly into the row-major speculative staging buffers.
[[nodiscard]] Status LaunchGemma4Moe26BMtpKvEpilogue(
    const float* query, const std::uint16_t* query_norm,
    float* normalized_query, const float* key,
    const std::uint16_t* key_norm, const float* value,
    const float* rotary_cosine, const float* rotary_sine,
    std::uint8_t* staged_key, std::uint8_t* staged_value,
    const std::uint16_t* key_scale, const std::uint16_t* value_scale,
    std::uint64_t tokens, std::uint64_t kv_heads,
    std::uint64_t head_dimension, double rotary_factor, float epsilon,
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

[[nodiscard]] Status LaunchAppendKvFp8BatchControlled(
    const std::uint8_t* key, const std::uint8_t* value,
    std::uint8_t* key_cache, std::uint8_t* value_cache,
    const DecodeControl* row_controls, std::uint64_t tokens,
    std::uint64_t elements_per_token, std::uint64_t cache_capacity,
    cudaStream_t stream);

[[nodiscard]] Status LaunchBackupAppendKvFp8BatchControlled(
    const std::uint8_t* key, const std::uint8_t* value,
    std::uint8_t* key_cache, std::uint8_t* value_cache,
    std::uint8_t* backup_key, std::uint8_t* backup_value,
    const DecodeControl* row_controls, std::uint64_t tokens,
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

}  // namespace gem16::internal
