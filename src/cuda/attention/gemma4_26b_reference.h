#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime_api.h>

#include "gem16/status.h"
#include "model/gemma4_26b_attention.h"

namespace gem16::internal {

inline constexpr std::size_t
    kGemma4Moe26BAttentionCutlassWorkspaceBytes = 8U * 1024U * 1024U;

class Gemma4Moe26BDeviceArtifact;
class Gemma4Moe26BTrellis35DeviceArtifact;
struct DecodeControl;

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
  // Prompt-only CUTLASS scratch. The integrated engine aliases this with the
  // routed-expert down-projection buffer whose lifetime begins only after the
  // attention layer has completed on the same stream.
  void* cutlass_workspace = nullptr;
  std::size_t cutlass_workspace_bytes = 0U;
  // Prompt-only, lifetime-aliased staging for the <=16K global-attention
  // path. The physical cache remains FP8; these buffers hold one layer's
  // exact BF16 dequantization only while that layer's attention is live.
  std::uint16_t* global_key_bf16 = nullptr;
  std::uint16_t* global_value_bf16 = nullptr;
  std::uint64_t global_bf16_capacity = 0U;
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
[[nodiscard]] Result<Gemma4Moe26BAttentionReferenceWeights>
BindGemma4Moe26BAttentionReferenceWeights(
    const Gemma4Moe26BTrellis35DeviceArtifact& artifact,
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

// Batched causal prefill over the same M12 arithmetic boundaries. Workspace
// pointers contain `tokens` contiguous rows and `score_elements` covers the
// complete causal score slab for the requested start position.
[[nodiscard]] Status LaunchGemma4Moe26BAttentionReferencePrefillLayer(
    const float* hidden, float* output, std::uint64_t start_position,
    std::uint64_t tokens,
    const Gemma4Moe26BAttentionLayerTraits& traits,
    const Gemma4Moe26BAttentionReferenceWeights& weights,
    const Gemma4Moe26BKvCacheView& cache,
    const Gemma4Moe26BAttentionReferenceWorkspace& workspace,
    float epsilon, cudaStream_t stream);

// Native SM120 batched prefill for the validated Gemma 4 26B local
// QH16/KVH8/D256 and global QH16/KVH2/D512 attention geometries. It preserves
// the reference path's FP8 cache, RoPE, BF16 and residual boundaries. Q/K/V
// use the prompt CUTLASS GEMM; O retains the exact native accumulation path,
// and causal attention uses the fixed SM120 kernels.
[[nodiscard]] Status LaunchGemma4Moe26BAttentionSm120PrefillLayer(
    const float* hidden, float* output, std::uint64_t start_position,
    std::uint64_t tokens,
    const Gemma4Moe26BAttentionLayerTraits& traits,
    const Gemma4Moe26BAttentionReferenceWeights& weights,
    const Gemma4Moe26BKvCacheView& cache,
    const Gemma4Moe26BAttentionReferenceWorkspace& workspace,
    float epsilon, cudaStream_t stream, bool rotary_prepared = false);

// Graph-capturable native T=1 path. Position is read from fixed device
// control; arithmetic and FP8 cache semantics remain the M12 contract.
// `workspace.scores` is the split-online scratch arena and requires at least
// DecodeAttentionWorkspaceElements(cache.capacity) FP32 elements.
[[nodiscard]] Status LaunchGemma4Moe26BAttentionReferenceControlledLayer(
    const float* hidden, float* output,
    const Gemma4Moe26BAttentionLayerTraits& traits,
    const Gemma4Moe26BAttentionReferenceWeights& weights,
    const Gemma4Moe26BKvCacheView& cache,
    const Gemma4Moe26BAttentionReferenceWorkspace& workspace,
    const DecodeControl* control, float epsilon, cudaStream_t stream,
    bool rotary_table_prepared = false);

// Exact fixed-D Target verifier. It reuses the production 12B decode-attention
// arithmetic because the 26B Target has the same local
// QH16/KVH8/D256 and global QH16/KVH2/D512 geometries. Q/K/V/O projection,
// recurrent BF16 boundaries and causal row order remain those of ordinary
// decode; only immutable weights and historical K/V are shared across rows.
[[nodiscard]] Status LaunchGemma4Moe26BAttentionSm120MtpFixedLayer(
    const float* hidden, float* output, std::uint64_t start_position,
    const Gemma4Moe26BAttentionLayerTraits& traits,
    const Gemma4Moe26BAttentionReferenceWeights& weights,
    const Gemma4Moe26BKvCacheView& cache,
    const Gemma4Moe26BAttentionReferenceWorkspace& workspace,
    float* decode_attention_workspace, const DecodeControl* row_controls,
    std::uint32_t tokens, bool shared_fixed_attention,
    bool batched_output_tail, bool controlled_positions, float epsilon,
    cudaStream_t stream, std::uint8_t* backup_key = nullptr,
    std::uint8_t* backup_value = nullptr);

[[nodiscard]] Status LaunchGemma4Moe26BAttentionSm120MtpD2Layer(
    const float* hidden, float* output, std::uint64_t start_position,
    const Gemma4Moe26BAttentionLayerTraits& traits,
    const Gemma4Moe26BAttentionReferenceWeights& weights,
    const Gemma4Moe26BKvCacheView& cache,
    const Gemma4Moe26BAttentionReferenceWorkspace& workspace,
    float* decode_attention_workspace, const DecodeControl* row_controls,
    bool shared_d2_attention, bool batched_output_tail,
    bool controlled_positions, float epsilon, cudaStream_t stream);

}  // namespace gem16::internal
