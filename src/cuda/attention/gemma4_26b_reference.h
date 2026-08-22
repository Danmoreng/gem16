#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

#include "gem16/status.h"
#include "model/gemma4_26b_attention.h"

namespace gem16::internal {

class Gemma4Moe26BDeviceArtifact;

struct Gemma4Moe26BFp8Matrix {
  const std::uint8_t* weight_e4m3 = nullptr;
  const std::uint16_t* weight_scales_bf16 = nullptr;
  std::uint64_t rows = 0;
  std::uint64_t columns = 0;
};

struct Gemma4Moe26BAttentionReferenceWeights {
  const std::uint16_t* input_norm_bf16 = nullptr;
  const std::uint16_t* post_attention_norm_bf16 = nullptr;
  const std::uint16_t* query_norm_bf16 = nullptr;
  const std::uint16_t* key_norm_bf16 = nullptr;
  const std::uint16_t* key_cache_scale_bf16 = nullptr;
  const std::uint16_t* value_cache_scale_bf16 = nullptr;
  Gemma4Moe26BFp8Matrix query;
  Gemma4Moe26BFp8Matrix key;
  Gemma4Moe26BFp8Matrix value;  // absent only for validated full attention
  Gemma4Moe26BFp8Matrix output;
};

struct Gemma4Moe26BAttentionReferenceWorkspace {
  std::uint8_t* input_fp8 = nullptr;       // 2816
  float* input_scale = nullptr;            // 1
  float* query_raw = nullptr;              // query_heads * head_dimension
  float* key_raw = nullptr;                // kv_heads * head_dimension
  float* value_raw = nullptr;              // kv_heads * head_dimension
  float* query_normalized = nullptr;       // query_heads * head_dimension
  float* key_normalized = nullptr;         // kv_heads * head_dimension
  float* value_normalized = nullptr;       // kv_heads * head_dimension
  float* rotary_cosine = nullptr;          // head_dimension / 2
  float* rotary_sine = nullptr;            // head_dimension / 2
  std::uint8_t* staged_key_fp8 = nullptr;  // kv_heads * head_dimension
  std::uint8_t* staged_value_fp8 = nullptr;
  float* scores = nullptr;                 // query_heads * cache capacity
  std::uint64_t score_elements = 0;
  float* attention = nullptr;              // query_heads * head_dimension
  std::uint8_t* output_fp8 = nullptr;       // query_heads * head_dimension
  float* output_scale = nullptr;            // 1
  float* output_projection = nullptr;       // 2816
  float* post_attention = nullptr;          // 2816, optional capture
};

struct Gemma4Moe26BKvCacheView {
  std::uint8_t* key = nullptr;
  std::uint8_t* value = nullptr;
  std::uint64_t capacity = 0;
};

// Initialization-only pointer resolution against the immutable M08/M09
// arena. The returned view does not own or duplicate any tensor.
[[nodiscard]] Result<Gemma4Moe26BAttentionReferenceWeights>
BindGemma4Moe26BAttentionReferenceWeights(
    const Gemma4Moe26BDeviceArtifact& artifact,
    const Gemma4Moe26BAttentionLayerTraits& traits);

// Correctness-first, one-token attention. It reads prior cache plus staged
// current K/V, then commits current K/V after attention. Every pointer is
// caller-owned fixed device storage; no allocation, synchronization, routing,
// filesystem access, JIT, quantization-policy choice, or repacking occurs.
[[nodiscard]] Status LaunchGemma4Moe26BAttentionReferenceLayer(
    const float* hidden, float* output, std::uint64_t position,
    const Gemma4Moe26BAttentionLayerTraits& traits,
    const Gemma4Moe26BAttentionReferenceWeights& weights,
    const Gemma4Moe26BKvCacheView& cache,
    const Gemma4Moe26BAttentionReferenceWorkspace& workspace,
    float epsilon, cudaStream_t stream);

}  // namespace gem16::internal
