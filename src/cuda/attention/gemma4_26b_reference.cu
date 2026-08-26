#include "cuda/attention/gemma4_26b_reference.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <utility>

#include "cuda/attention/sm120.h"
#include "cuda/engine/gemma4_26b_artifact.h"
#include "cuda/fp8/cutlass_sm120.h"
#include "cuda/fp8/reference.h"
#include "cuda/fp8/sm120.h"
#include "cuda/layer/reference.h"

namespace gem16::internal {
namespace {

constexpr std::uint64_t kHidden = 2816U;
constexpr std::uint64_t kQueryHeads = 16U;

Status Invalid(std::string message) {
  return Status(StatusCode::kInvalidArgument, std::move(message));
}

Status CudaFailure(const char* operation, cudaError_t error) {
  return Status(StatusCode::kInternal,
                std::string(operation) + ": " + cudaGetErrorName(error) +
                    ": " + cudaGetErrorString(error));
}

template <typename T>
Result<const T*> Pointer(const Gemma4Moe26BDeviceArtifact& artifact,
                         const std::string& name) {
  auto pointer = artifact.Pointer(name);
  if (!pointer.ok()) return pointer.status();
  return reinterpret_cast<const T*>(pointer.value());
}

Result<Gemma4Moe26BFp8Matrix> Matrix(
    const Gemma4Moe26BDeviceArtifact& artifact, const std::string& module,
    std::uint64_t rows, std::uint64_t columns) {
  auto weight = Pointer<std::uint8_t>(artifact, module + ".weight");
  if (!weight.ok()) return weight.status();
  auto scales = Pointer<std::uint16_t>(artifact, module + ".weight_scale");
  if (!scales.ok()) return scales.status();
  return Gemma4Moe26BFp8Matrix{weight.value(), scales.value(), rows, columns};
}

bool ValidMatrix(const Gemma4Moe26BFp8Matrix& matrix, std::uint64_t rows,
                 std::uint64_t columns) {
  return matrix.weight_e4m3 != nullptr &&
         matrix.weight_scales_bf16 != nullptr && matrix.rows == rows &&
         matrix.columns == columns;
}

bool ValidPointers(const Gemma4Moe26BAttentionReferenceWorkspace& x) {
  return x.input_fp8 != nullptr && x.input_scale != nullptr &&
         x.query_raw != nullptr && x.key_raw != nullptr &&
         x.value_raw != nullptr && x.query_normalized != nullptr &&
         x.key_normalized != nullptr && x.value_normalized != nullptr &&
         x.rotary_cosine != nullptr && x.rotary_sine != nullptr &&
         x.staged_key_fp8 != nullptr && x.staged_value_fp8 != nullptr &&
         x.scores != nullptr && x.attention != nullptr &&
         x.output_fp8 != nullptr && x.output_scale != nullptr &&
         x.output_projection != nullptr;
}

}  // namespace

Result<Gemma4Moe26BAttentionReferenceWeights>
BindGemma4Moe26BAttentionReferenceWeights(
    const Gemma4Moe26BDeviceArtifact& artifact,
    const Gemma4Moe26BAttentionLayerTraits& traits) {
  if (traits.layer >= 30U || traits.query_heads != kQueryHeads ||
      traits.kv_producer_layer != static_cast<std::int32_t>(traits.layer)) {
    return Invalid("M12 cannot bind an invalid attention layer trait");
  }
  const std::string prefix = "model.language_model.layers." +
                             std::to_string(traits.layer) + ".";
  Gemma4Moe26BAttentionReferenceWeights result;
  auto bind_bf16 = [&](const char* suffix,
                       const std::uint16_t** destination) -> Status {
    auto pointer = Pointer<std::uint16_t>(artifact, prefix + suffix);
    if (!pointer.ok()) return pointer.status();
    *destination = pointer.value();
    return Status::Ok();
  };
  for (const auto& binding : {
           std::pair{"input_layernorm.weight", &result.input_norm_bf16},
           std::pair{"post_attention_layernorm.weight",
                     &result.post_attention_norm_bf16},
           std::pair{"self_attn.q_norm.weight", &result.query_norm_bf16},
           std::pair{"self_attn.k_norm.weight", &result.key_norm_bf16},
           std::pair{"self_attn.k_scale", &result.key_cache_scale_bf16},
           std::pair{"self_attn.v_scale", &result.value_cache_scale_bf16}}) {
    Status status = bind_bf16(binding.first, binding.second);
    if (!status.ok()) return status;
  }
  const std::uint64_t q_elements = traits.query_heads * traits.head_dimension;
  const std::uint64_t kv_elements = traits.kv_heads * traits.head_dimension;
  auto query = Matrix(artifact, prefix + "self_attn.q_proj", q_elements,
                      kHidden);
  if (!query.ok()) return query.status();
  result.query = query.value();
  auto key = Matrix(artifact, prefix + "self_attn.k_proj", kv_elements,
                    kHidden);
  if (!key.ok()) return key.status();
  result.key = key.value();
  if (traits.stores_v_projection) {
    auto value = Matrix(artifact, prefix + "self_attn.v_proj", kv_elements,
                        kHidden);
    if (!value.ok()) return value.status();
    result.value = value.value();
  }
  auto output = Matrix(artifact, prefix + "self_attn.o_proj", kHidden,
                       q_elements);
  if (!output.ok()) return output.status();
  result.output = output.value();
  return result;
}

Status LaunchGemma4Moe26BAttentionReferenceLayer(
    const float* hidden, float* output, std::uint64_t position,
    const Gemma4Moe26BAttentionLayerTraits& t,
    const Gemma4Moe26BAttentionReferenceWeights& w,
    const Gemma4Moe26BKvCacheView& cache,
    const Gemma4Moe26BAttentionReferenceWorkspace& x, float epsilon,
    cudaStream_t stream) {
  const bool sliding =
      t.attention == Gemma4Moe26BAttentionType::kSliding;
  const std::uint64_t q_elements = t.query_heads * t.head_dimension;
  const std::uint64_t kv_elements = t.kv_heads * t.head_dimension;
  const std::uint64_t required_scores =
      t.query_heads * (sliding ? cache.capacity : position + 1U);
  if (hidden == nullptr || output == nullptr || hidden == output ||
      cache.key == nullptr || cache.value == nullptr ||
      cache.key == cache.value || !ValidPointers(x) ||
      x.post_attention == nullptr ||
      w.input_norm_bf16 == nullptr ||
      w.post_attention_norm_bf16 == nullptr ||
      w.query_norm_bf16 == nullptr || w.key_norm_bf16 == nullptr ||
      w.key_cache_scale_bf16 == nullptr ||
      w.value_cache_scale_bf16 == nullptr || t.layer >= 30U ||
      t.query_heads != kQueryHeads || t.kv_heads == 0U ||
      t.head_dimension == 0U || t.query_heads % t.kv_heads != 0U ||
      t.kv_producer_layer != static_cast<std::int32_t>(t.layer) ||
      position >= 262144U ||
      cache.capacity == 0U || cache.capacity > t.cache_capacity ||
      (sliding && cache.capacity != t.cache_capacity) ||
      (!sliding && position >= cache.capacity) ||
      required_scores > x.score_elements ||
      !ValidMatrix(w.query, q_elements, kHidden) ||
      !ValidMatrix(w.key, kv_elements, kHidden) ||
      !ValidMatrix(w.output, kHidden, q_elements) ||
      (t.stores_v_projection != !t.reuses_raw_k_for_v) ||
      (t.stores_v_projection && !ValidMatrix(w.value, kv_elements, kHidden)) ||
      (!t.stores_v_projection &&
       (w.value.weight_e4m3 != nullptr ||
        w.value.weight_scales_bf16 != nullptr)) ||
      !std::isfinite(epsilon) || epsilon <= 0.0F) {
    return Invalid("M12 attention execution contract is invalid");
  }

  Status status = LaunchRmsNormFp8TokenQuantizationBatch(
      hidden, w.input_norm_bf16, x.input_fp8, x.input_scale, 1U, kHidden,
      epsilon, stream);
  if (!status.ok()) return status;
  status = LaunchFp8ReferenceProjection(
      x.input_fp8, x.input_scale, w.query.weight_e4m3,
      w.query.weight_scales_bf16, x.query_raw, q_elements, kHidden, stream);
  if (!status.ok()) return status;
  status = LaunchFp8ReferenceProjection(
      x.input_fp8, x.input_scale, w.key.weight_e4m3,
      w.key.weight_scales_bf16, x.key_raw, kv_elements, kHidden, stream);
  if (!status.ok()) return status;
  if (t.reuses_raw_k_for_v) {
    const cudaError_t copied = cudaMemcpyAsync(
        x.value_raw, x.key_raw, kv_elements * sizeof(float),
        cudaMemcpyDeviceToDevice, stream);
    if (copied != cudaSuccess) {
      return CudaFailure("reuse M12 raw K projection for V", copied);
    }
  } else {
    status = LaunchFp8ReferenceProjection(
        x.input_fp8, x.input_scale, w.value.weight_e4m3,
        w.value.weight_scales_bf16, x.value_raw, kv_elements, kHidden, stream);
    if (!status.ok()) return status;
  }

  const std::uint64_t rotating_pairs = static_cast<std::uint64_t>(
      t.rotary_factor * static_cast<double>(t.head_dimension / 2U));
  status = LaunchRotaryEmbeddingTableBatch(
      x.rotary_cosine, x.rotary_sine, 1U, rotating_pairs, t.head_dimension,
      position, t.rope_theta, t.rope_scaling_factor, stream);
  if (!status.ok()) return status;
  status = LaunchProjectionRmsNormRotaryBf16Batch(
      x.query_raw, w.query_norm_bf16, x.query_normalized, x.key_raw,
      w.key_norm_bf16, x.key_normalized, x.rotary_cosine, x.rotary_sine, 1U,
      t.query_heads, t.kv_heads, t.head_dimension, t.rotary_factor, epsilon,
      stream);
  if (!status.ok()) return status;
  status = LaunchRmsNormBf16(x.value_raw, nullptr, x.value_normalized,
                             t.kv_heads, t.head_dimension, epsilon, stream);
  if (!status.ok()) return status;
  status = LaunchQuantizeKvFp8Batch(
      x.key_normalized, x.value_normalized, x.staged_key_fp8,
      x.staged_value_fp8, w.key_cache_scale_bf16,
      w.value_cache_scale_bf16, 1U, kv_elements, stream);
  if (!status.ok()) return status;
  status = LaunchCausalAttentionPrefillFp8(
      x.query_normalized, x.staged_key_fp8, x.staged_value_fp8, cache.key,
      cache.value, w.key_cache_scale_bf16, w.value_cache_scale_bf16, x.scores,
      x.attention, position, 1U, t.query_heads, t.kv_heads, t.head_dimension,
      cache.capacity, sliding, stream);
  if (!status.ok()) return status;

  // Commit only after attention has consumed prior state plus staged current.
  status = LaunchAppendKvFp8Batch(
      x.staged_key_fp8, x.staged_value_fp8, cache.key, cache.value, position,
      1U, kv_elements, cache.capacity, stream);
  if (!status.ok()) return status;
  status = LaunchFp8ReferenceTokenQuantization(
      x.attention, x.output_fp8, x.output_scale, q_elements, stream);
  if (!status.ok()) return status;
  status = LaunchFp8ReferenceProjection(
      x.output_fp8, x.output_scale, w.output.weight_e4m3,
      w.output.weight_scales_bf16, x.output_projection, kHidden, q_elements,
      stream);
  if (!status.ok()) return status;
  return LaunchRmsNormResidualBf16(
      x.output_projection, w.post_attention_norm_bf16, hidden,
      x.post_attention, output, 1U, kHidden, epsilon, nullptr, stream);
}

Status LaunchGemma4Moe26BAttentionReferencePrefillLayer(
    const float* hidden, float* output, std::uint64_t start_position,
    std::uint64_t tokens, const Gemma4Moe26BAttentionLayerTraits& t,
    const Gemma4Moe26BAttentionReferenceWeights& w,
    const Gemma4Moe26BKvCacheView& cache,
    const Gemma4Moe26BAttentionReferenceWorkspace& x, float epsilon,
    cudaStream_t stream) {
  const bool sliding =
      t.attention == Gemma4Moe26BAttentionType::kSliding;
  const std::uint64_t q_elements = t.query_heads * t.head_dimension;
  const std::uint64_t kv_elements = t.kv_heads * t.head_dimension;
  if (tokens == 0U || tokens > 1024U ||
      start_position > std::numeric_limits<std::uint64_t>::max() - tokens ||
      start_position + tokens > 262144U || hidden == nullptr ||
      output == nullptr || hidden == output || cache.key == nullptr ||
      cache.value == nullptr || cache.key == cache.value || !ValidPointers(x) ||
      x.post_attention == nullptr ||
      w.input_norm_bf16 == nullptr ||
      w.post_attention_norm_bf16 == nullptr ||
      w.query_norm_bf16 == nullptr || w.key_norm_bf16 == nullptr ||
      w.key_cache_scale_bf16 == nullptr ||
      w.value_cache_scale_bf16 == nullptr || t.layer >= 30U ||
      t.query_heads != kQueryHeads || t.kv_heads == 0U ||
      t.head_dimension == 0U || t.query_heads % t.kv_heads != 0U ||
      t.kv_producer_layer != static_cast<std::int32_t>(t.layer) ||
      cache.capacity == 0U || cache.capacity > t.cache_capacity ||
      (sliding && cache.capacity != t.cache_capacity) ||
      (!sliding && start_position + tokens > cache.capacity) ||
      tokens > std::numeric_limits<std::uint64_t>::max() / t.query_heads ||
      tokens * t.query_heads > std::numeric_limits<std::uint64_t>::max() /
                                   (sliding ? cache.capacity
                                            : start_position + tokens) ||
      tokens * t.query_heads *
              (sliding ? cache.capacity : start_position + tokens) >
          x.score_elements ||
      !ValidMatrix(w.query, q_elements, kHidden) ||
      !ValidMatrix(w.key, kv_elements, kHidden) ||
      !ValidMatrix(w.output, kHidden, q_elements) ||
      (t.stores_v_projection != !t.reuses_raw_k_for_v) ||
      (t.stores_v_projection && !ValidMatrix(w.value, kv_elements, kHidden)) ||
      (!t.stores_v_projection &&
       (w.value.weight_e4m3 != nullptr ||
        w.value.weight_scales_bf16 != nullptr)) ||
      !std::isfinite(epsilon) || epsilon <= 0.0F) {
    return Invalid("M17 attention prefill execution contract is invalid");
  }
  Status status = LaunchRmsNormFp8TokenQuantizationBatch(
      hidden, w.input_norm_bf16, x.input_fp8, x.input_scale, tokens, kHidden,
      epsilon, stream);
  if (!status.ok()) return status;
  status = LaunchFp8ReferenceProjectionBatch(
      x.input_fp8, x.input_scale, w.query.weight_e4m3,
      w.query.weight_scales_bf16, x.query_raw, tokens, q_elements, kHidden,
      stream);
  if (!status.ok()) return status;
  status = LaunchFp8ReferenceProjectionBatch(
      x.input_fp8, x.input_scale, w.key.weight_e4m3,
      w.key.weight_scales_bf16, x.key_raw, tokens, kv_elements, kHidden,
      stream);
  if (!status.ok()) return status;
  if (t.reuses_raw_k_for_v) {
    const cudaError_t copied = cudaMemcpyAsync(
        x.value_raw, x.key_raw, tokens * kv_elements * sizeof(float),
        cudaMemcpyDeviceToDevice, stream);
    if (copied != cudaSuccess) {
      return CudaFailure("reuse M17 prefill raw K projection for V", copied);
    }
  } else {
    status = LaunchFp8ReferenceProjectionBatch(
        x.input_fp8, x.input_scale, w.value.weight_e4m3,
        w.value.weight_scales_bf16, x.value_raw, tokens, kv_elements, kHidden,
        stream);
    if (!status.ok()) return status;
  }
  const std::uint64_t rotating_pairs = static_cast<std::uint64_t>(
      t.rotary_factor * static_cast<double>(t.head_dimension / 2U));
  status = LaunchRotaryEmbeddingTableBatch(
      x.rotary_cosine, x.rotary_sine, tokens, rotating_pairs,
      t.head_dimension, start_position, t.rope_theta, t.rope_scaling_factor,
      stream);
  if (!status.ok()) return status;
  status = LaunchProjectionRmsNormRotaryBf16Batch(
      x.query_raw, w.query_norm_bf16, x.query_normalized, x.key_raw,
      w.key_norm_bf16, x.key_normalized, x.rotary_cosine, x.rotary_sine,
      tokens, t.query_heads, t.kv_heads, t.head_dimension, t.rotary_factor,
      epsilon, stream);
  if (!status.ok()) return status;
  status = LaunchRmsNormBf16(x.value_raw, nullptr, x.value_normalized,
                             tokens * t.kv_heads, t.head_dimension, epsilon,
                             stream);
  if (!status.ok()) return status;
  status = LaunchQuantizeKvFp8Batch(
      x.key_normalized, x.value_normalized, x.staged_key_fp8,
      x.staged_value_fp8, w.key_cache_scale_bf16,
      w.value_cache_scale_bf16, tokens, kv_elements, stream);
  if (!status.ok()) return status;
  status = LaunchCausalAttentionPrefillFp8(
      x.query_normalized, x.staged_key_fp8, x.staged_value_fp8, cache.key,
      cache.value, w.key_cache_scale_bf16, w.value_cache_scale_bf16, x.scores,
      x.attention, start_position, tokens, t.query_heads, t.kv_heads,
      t.head_dimension, cache.capacity, sliding, stream);
  if (!status.ok()) return status;
  status = LaunchAppendKvFp8Batch(
      x.staged_key_fp8, x.staged_value_fp8, cache.key, cache.value,
      start_position, tokens, kv_elements, cache.capacity, stream);
  if (!status.ok()) return status;
  status = LaunchFp8ReferenceTokenQuantizationBatch(
      x.attention, x.output_fp8, x.output_scale, tokens, q_elements, stream);
  if (!status.ok()) return status;
  status = LaunchFp8ReferenceProjectionBatch(
      x.output_fp8, x.output_scale, w.output.weight_e4m3,
      w.output.weight_scales_bf16, x.output_projection, tokens, kHidden,
      q_elements, stream);
  if (!status.ok()) return status;
  return LaunchRmsNormResidualBf16(
      x.output_projection, w.post_attention_norm_bf16, hidden,
      x.post_attention, output, tokens, kHidden, epsilon, nullptr, stream);
}

Status LaunchGemma4Moe26BAttentionSm120PrefillLayer(
    const float* hidden, float* output, std::uint64_t start_position,
    std::uint64_t tokens, const Gemma4Moe26BAttentionLayerTraits& t,
    const Gemma4Moe26BAttentionReferenceWeights& w,
    const Gemma4Moe26BKvCacheView& cache,
    const Gemma4Moe26BAttentionReferenceWorkspace& x, float epsilon,
    cudaStream_t stream) {
  constexpr std::uint64_t kMaximumChunkTokens = 1024U;
  constexpr std::uint64_t kMaximumRotaryPairs = 256U;
  const bool sliding =
      t.attention == Gemma4Moe26BAttentionType::kSliding;
  const std::uint64_t q_elements = t.query_heads * t.head_dimension;
  const std::uint64_t kv_elements = t.kv_heads * t.head_dimension;
  const bool native_geometry =
      sliding ? t.query_heads == 16U && t.kv_heads == 8U &&
                    t.head_dimension == 256U
              : t.query_heads == 16U && t.kv_heads == 2U &&
                    t.head_dimension == 512U;
  if (tokens == 0U || tokens > kMaximumChunkTokens ||
      start_position > std::numeric_limits<std::uint64_t>::max() - tokens ||
      start_position + tokens > 262144U || hidden == nullptr ||
      output == nullptr || hidden == output || cache.key == nullptr ||
      cache.value == nullptr || cache.key == cache.value || !ValidPointers(x) ||
      w.input_norm_bf16 == nullptr ||
      w.post_attention_norm_bf16 == nullptr ||
      w.query_norm_bf16 == nullptr || w.key_norm_bf16 == nullptr ||
      w.key_cache_scale_bf16 == nullptr ||
      w.value_cache_scale_bf16 == nullptr || t.layer >= 30U ||
      !native_geometry ||
      !std::isfinite(t.rotary_factor) || t.rotary_factor <= 0.0 ||
      t.rotary_factor > 1.0 ||
      t.rotary_factor * static_cast<double>(t.head_dimension / 2U) >
          static_cast<double>(kMaximumRotaryPairs) ||
      t.kv_producer_layer != static_cast<std::int32_t>(t.layer) ||
      cache.capacity == 0U || cache.capacity > t.cache_capacity ||
      (sliding && cache.capacity != t.cache_capacity) ||
      (!sliding && start_position + tokens > cache.capacity) ||
      !ValidMatrix(w.query, q_elements, kHidden) ||
      !ValidMatrix(w.key, kv_elements, kHidden) ||
      !ValidMatrix(w.output, kHidden, q_elements) ||
      (t.stores_v_projection != !t.reuses_raw_k_for_v) ||
      (t.stores_v_projection && !ValidMatrix(w.value, kv_elements, kHidden)) ||
      (!t.stores_v_projection &&
       (w.value.weight_e4m3 != nullptr ||
        w.value.weight_scales_bf16 != nullptr)) ||
      x.cutlass_workspace == nullptr ||
      x.cutlass_workspace_bytes <
          kGemma4Moe26BAttentionCutlassWorkspaceBytes ||
      !std::isfinite(epsilon) || epsilon <= 0.0F) {
    return Invalid("native SM120 Gemma 4 26B attention prefill contract is invalid");
  }

  auto* query_bf16 = reinterpret_cast<std::uint16_t*>(x.query_raw);
  auto* key_bf16 = reinterpret_cast<std::uint16_t*>(x.key_raw);
  auto* value_bf16 = reinterpret_cast<std::uint16_t*>(x.value_raw);
  Status status = LaunchRmsNormFp8TokenQuantizationBatch(
      hidden, w.input_norm_bf16, x.input_fp8, x.input_scale, tokens, kHidden,
      epsilon, stream);
  if (!status.ok()) return status;
  status = LaunchFp8CutlassProjectionBatch(
      x.input_fp8, x.input_scale, w.query.weight_e4m3,
      w.query.weight_scales_bf16, query_bf16, tokens, q_elements, kHidden,
      x.cutlass_workspace, x.cutlass_workspace_bytes, stream);
  if (!status.ok()) return status;
  status = LaunchFp8CutlassProjectionBatch(
      x.input_fp8, x.input_scale, w.key.weight_e4m3,
      w.key.weight_scales_bf16, key_bf16, tokens, kv_elements, kHidden,
      x.cutlass_workspace, x.cutlass_workspace_bytes, stream);
  if (!status.ok()) return status;
  if (t.reuses_raw_k_for_v) {
    const cudaError_t copied = cudaMemcpyAsync(
        value_bf16, key_bf16,
        tokens * kv_elements * sizeof(std::uint16_t),
        cudaMemcpyDeviceToDevice, stream);
    if (copied != cudaSuccess) {
      return CudaFailure("reuse native 26B prefill BF16 K projection for V",
                         copied);
    }
  } else {
    status = LaunchFp8CutlassProjectionBatch(
        x.input_fp8, x.input_scale, w.value.weight_e4m3,
        w.value.weight_scales_bf16, value_bf16, tokens, kv_elements, kHidden,
        x.cutlass_workspace, x.cutlass_workspace_bytes, stream);
    if (!status.ok()) return status;
  }

  const std::uint64_t rotating_pairs = static_cast<std::uint64_t>(
      t.rotary_factor * static_cast<double>(t.head_dimension / 2U));
  status = LaunchRotaryEmbeddingTableBatch(
      x.rotary_cosine, x.rotary_sine, tokens, rotating_pairs,
      t.head_dimension, start_position, t.rope_theta, t.rope_scaling_factor,
      stream);
  if (!status.ok()) return status;
  status = LaunchProjectionRmsNormRotaryBf16BatchInput(
      query_bf16, w.query_norm_bf16, x.query_normalized, key_bf16,
      w.key_norm_bf16, x.key_normalized, x.rotary_cosine, x.rotary_sine,
      tokens, t.query_heads, t.kv_heads, t.head_dimension, t.rotary_factor,
      epsilon, stream);
  if (!status.ok()) return status;
  status = LaunchRmsNormBf16Input(
      value_bf16, nullptr, x.value_normalized, tokens * t.kv_heads,
      t.head_dimension, epsilon, stream);
  if (!status.ok()) return status;
  status = LaunchQuantizeKvFp8Batch(
      x.key_normalized, x.value_normalized, x.staged_key_fp8,
      x.staged_value_fp8, w.key_cache_scale_bf16,
      w.value_cache_scale_bf16, tokens, kv_elements, stream);
  if (!status.ok()) return status;

  auto* attention_bf16 = reinterpret_cast<std::uint16_t*>(x.attention);
  const bool prepared_global =
      !sliding && x.global_key_bf16 != nullptr &&
      x.global_value_bf16 != nullptr &&
      tokens <= x.global_bf16_capacity &&
      start_position <= x.global_bf16_capacity - tokens;
  if (prepared_global) {
    // Prepared BF16 staging aliases projection-only temporaries, including
    // staged_key/value. Commit the physical FP8 bytes first so preparation
    // reads only the persistent cache while reusing those dead buffers.
    status = LaunchAppendKvFp8Batch(
        x.staged_key_fp8, x.staged_value_fp8, cache.key, cache.value,
        start_position, tokens, kv_elements, cache.capacity, stream);
    if (!status.ok()) return status;
  }
  status = sliding
               ? LaunchOnlineCausalAttentionPrefillFp8LocalSm120(
                     x.query_normalized, x.staged_key_fp8,
                     x.staged_value_fp8, cache.key, cache.value,
                     w.key_cache_scale_bf16, w.value_cache_scale_bf16,
                     attention_bf16, start_position, tokens, t.query_heads,
                     t.kv_heads, t.head_dimension, cache.capacity, stream)
               : (prepared_global
                      ? LaunchOnlineCausalAttentionPrefillFp8GlobalPreparedSm120(
                            x.query_normalized,
                            cache.key + start_position * kv_elements,
                            cache.value + start_position * kv_elements,
                            cache.key, cache.value,
                            w.key_cache_scale_bf16,
                            w.value_cache_scale_bf16, x.global_key_bf16,
                            x.global_value_bf16, attention_bf16,
                            start_position, tokens, t.query_heads, t.kv_heads,
                            t.head_dimension, cache.capacity,
                            x.global_bf16_capacity, stream)
                      : LaunchOnlineCausalAttentionPrefillFp8GlobalSm120(
                            x.query_normalized, x.staged_key_fp8,
                            x.staged_value_fp8, cache.key, cache.value,
                            w.key_cache_scale_bf16,
                            w.value_cache_scale_bf16, attention_bf16,
                            start_position, tokens, t.query_heads, t.kv_heads,
                            t.head_dimension, cache.capacity, stream));
  if (!status.ok()) return status;
  if (!prepared_global) {
    status = LaunchAppendKvFp8Batch(
        x.staged_key_fp8, x.staged_value_fp8, cache.key, cache.value,
        start_position, tokens, kv_elements, cache.capacity, stream);
    if (!status.ok()) return status;
  }
  status = LaunchFp8ReferenceTokenQuantizationBf16Batch(
      attention_bf16, x.output_fp8, x.output_scale, tokens, q_elements,
      stream);
  if (!status.ok()) return status;
  // Keep O on the source-order native MMA path. CUTLASS O is locally within
  // the BF16 differential gate, but the canonical 16K greedy run diverges;
  // Q/K/V CUTLASS plus native O preserves the accepted output hash.
  status = LaunchFp8Sm120DirectProjectionBatch(
      x.output_fp8, x.output_scale, w.output.weight_e4m3,
      w.output.weight_scales_bf16, x.output_projection, tokens, kHidden,
      q_elements, stream);
  if (!status.ok()) return status;
  return LaunchRmsNormResidualBf16(
      x.output_projection, w.post_attention_norm_bf16, hidden,
      x.post_attention, output, tokens, kHidden, epsilon, nullptr, stream);
}

Status LaunchGemma4Moe26BAttentionSm120MtpFixedLayer(
    const float* hidden, float* output, std::uint64_t start_position,
    const Gemma4Moe26BAttentionLayerTraits& t,
    const Gemma4Moe26BAttentionReferenceWeights& w,
    const Gemma4Moe26BKvCacheView& cache,
    const Gemma4Moe26BAttentionReferenceWorkspace& x,
    float* decode_attention_workspace, const DecodeControl* row_controls,
    std::uint32_t tokens, bool shared_fixed_attention, bool batched_output_tail,
    bool controlled_positions, float epsilon, cudaStream_t stream) {
  const bool sliding =
      t.attention == Gemma4Moe26BAttentionType::kSliding;
  const std::uint64_t q_elements = t.query_heads * t.head_dimension;
  const std::uint64_t kv_elements = t.kv_heads * t.head_dimension;
  const bool native_geometry =
      sliding ? t.query_heads == 16U && t.kv_heads == 8U &&
                    t.head_dimension == 256U
              : t.query_heads == 16U && t.kv_heads == 2U &&
                    t.head_dimension == 512U;
  if (hidden == nullptr || output == nullptr || hidden == output ||
      decode_attention_workspace == nullptr || row_controls == nullptr ||
      cache.key == nullptr || cache.value == nullptr ||
      cache.key == cache.value || !ValidPointers(x) ||
      w.input_norm_bf16 == nullptr ||
      w.post_attention_norm_bf16 == nullptr ||
      w.query_norm_bf16 == nullptr || w.key_norm_bf16 == nullptr ||
      w.key_cache_scale_bf16 == nullptr ||
      w.value_cache_scale_bf16 == nullptr || t.layer >= 30U ||
      (tokens != 2U && tokens != 3U && tokens != 5U) ||
      !native_geometry ||
      t.kv_producer_layer != static_cast<std::int32_t>(t.layer) ||
      cache.capacity == 0U || cache.capacity > t.cache_capacity ||
      (sliding && cache.capacity != t.cache_capacity) ||
      (!controlled_positions && !sliding &&
       (start_position >= cache.capacity ||
        tokens > cache.capacity - start_position)) ||
      !ValidMatrix(w.query, q_elements, kHidden) ||
      !ValidMatrix(w.key, kv_elements, kHidden) ||
      !ValidMatrix(w.output, kHidden, q_elements) ||
      (t.stores_v_projection != !t.reuses_raw_k_for_v) ||
      (t.stores_v_projection && !ValidMatrix(w.value, kv_elements, kHidden)) ||
      (!t.stores_v_projection &&
       (w.value.weight_e4m3 != nullptr ||
        w.value.weight_scales_bf16 != nullptr)) ||
      !std::isfinite(epsilon) || epsilon <= 0.0F) {
    return Invalid("M25 exact fixed-D attention contract is invalid");
  }

  Status status = LaunchRmsNormFp8TokenQuantizationBatch(
      hidden, w.input_norm_bf16, x.input_fp8, x.input_scale, tokens,
      kHidden, epsilon, stream);
  if (!status.ok()) return status;
  status = LaunchFp8Sm120GroupedQkvProjectionBatch(
      x.input_fp8, x.input_scale, w.query.weight_e4m3,
      w.query.weight_scales_bf16, x.query_raw, q_elements,
      w.key.weight_e4m3, w.key.weight_scales_bf16, x.key_raw, kv_elements,
      t.reuses_raw_k_for_v ? nullptr : w.value.weight_e4m3,
      t.reuses_raw_k_for_v ? nullptr : w.value.weight_scales_bf16,
      t.reuses_raw_k_for_v ? nullptr : x.value_raw,
      t.reuses_raw_k_for_v ? 0U : kv_elements, tokens, kHidden, stream);
  if (!status.ok()) return status;
  if (t.reuses_raw_k_for_v) {
    const cudaError_t copied = cudaMemcpyAsync(
        x.value_raw, x.key_raw, tokens * kv_elements * sizeof(float),
        cudaMemcpyDeviceToDevice, stream);
    if (copied != cudaSuccess) {
      return CudaFailure("reuse M25 exact fixed-D raw K for V", copied);
    }
  }

  const std::uint64_t rotating_pairs = static_cast<std::uint64_t>(
      t.rotary_factor * static_cast<double>(t.head_dimension / 2U));
  if (controlled_positions) {
    status = LaunchRotaryEmbeddingTableBatchControlled(
        x.rotary_cosine, x.rotary_sine, tokens, rotating_pairs,
        t.head_dimension, row_controls, t.rope_theta, t.rope_scaling_factor,
        stream);
  } else {
    status = LaunchRotaryEmbeddingTableBatch(
        x.rotary_cosine, x.rotary_sine, tokens, rotating_pairs,
        t.head_dimension, start_position, t.rope_theta,
        t.rope_scaling_factor, stream);
  }
  if (!status.ok()) return status;
  status = LaunchProjectionRmsNormRotaryBf16Batch(
      x.query_raw, w.query_norm_bf16, x.query_normalized, x.key_raw,
      w.key_norm_bf16, x.key_normalized, x.rotary_cosine, x.rotary_sine,
      tokens, t.query_heads, t.kv_heads, t.head_dimension, t.rotary_factor,
      epsilon, stream);
  if (!status.ok()) return status;
  status = LaunchRmsNormBf16(x.value_raw, nullptr, x.value_normalized,
                             tokens * t.kv_heads, t.head_dimension,
                             epsilon, stream);
  if (!status.ok()) return status;
  status = LaunchQuantizeKvFp8Batch(
      x.key_normalized, x.value_normalized, x.staged_key_fp8,
      x.staged_value_fp8, w.key_cache_scale_bf16,
      w.value_cache_scale_bf16, tokens, kv_elements, stream);
  if (!status.ok()) return status;

  if (!shared_fixed_attention) {
    for (std::uint64_t row = 0U; row < tokens; ++row) {
      status = controlled_positions
                   ? LaunchAppendKvFp8BatchControlled(
                         x.staged_key_fp8 + row * kv_elements,
                         x.staged_value_fp8 + row * kv_elements, cache.key,
                         cache.value, row_controls + row, 1U, kv_elements,
                         cache.capacity, stream)
                   : LaunchAppendKvFp8Batch(
                         x.staged_key_fp8 + row * kv_elements,
                         x.staged_value_fp8 + row * kv_elements, cache.key,
                         cache.value, start_position + row, 1U, kv_elements,
                         cache.capacity, stream);
      if (!status.ok()) return status;
      status = LaunchOnlineAttentionDecodeFp8Sm120(
          x.query_normalized + row * q_elements, cache.key, cache.value,
          w.key_cache_scale_bf16, w.value_cache_scale_bf16,
          decode_attention_workspace, x.attention + row * q_elements,
          row_controls + row, t.query_heads, t.kv_heads, t.head_dimension,
          cache.capacity, sliding, stream);
      if (!status.ok()) return status;
    }
  } else if (sliding) {
    status = LaunchOnlineAttentionDecodeFp8LocalFixedControlledSm120(
        x.query_normalized, cache.key, cache.value, x.staged_key_fp8,
        x.staged_value_fp8, w.key_cache_scale_bf16,
        w.value_cache_scale_bf16, decode_attention_workspace, x.attention,
        row_controls, tokens, cache.capacity, stream);
    if (!status.ok()) return status;
    status = controlled_positions
                 ? LaunchAppendKvFp8BatchControlled(
                       x.staged_key_fp8, x.staged_value_fp8, cache.key,
                       cache.value, row_controls, tokens, kv_elements,
                       cache.capacity, stream)
                 : LaunchAppendKvFp8Batch(
                       x.staged_key_fp8, x.staged_value_fp8, cache.key,
                       cache.value, start_position, tokens, kv_elements,
                       cache.capacity, stream);
  } else {
    status = controlled_positions
                 ? LaunchAppendKvFp8BatchControlled(
                       x.staged_key_fp8, x.staged_value_fp8, cache.key,
                       cache.value, row_controls, tokens, kv_elements,
                       cache.capacity, stream)
                 : LaunchAppendKvFp8Batch(
                       x.staged_key_fp8, x.staged_value_fp8, cache.key,
                       cache.value, start_position, tokens, kv_elements,
                       cache.capacity, stream);
    if (!status.ok()) return status;
    status = LaunchOnlineAttentionDecodeFp8GlobalFixedControlledSm120(
        x.query_normalized, cache.key, cache.value,
        w.key_cache_scale_bf16, w.value_cache_scale_bf16,
        decode_attention_workspace, x.attention, row_controls, tokens,
        t.kv_heads, cache.capacity, stream);
  }
  if (!status.ok()) return status;
  if (batched_output_tail) {
    status = LaunchFp8ReferenceTokenQuantizationBatch(
        x.attention, x.output_fp8, x.output_scale, tokens, q_elements,
        stream);
    if (!status.ok()) return status;
    status = LaunchFp8Sm120DirectProjectionBatch(
        x.output_fp8, x.output_scale, w.output.weight_e4m3,
        w.output.weight_scales_bf16, x.output_projection, tokens, kHidden,
        q_elements, stream);
    if (!status.ok()) return status;
    return LaunchRmsNormResidualBf16(
        x.output_projection, w.post_attention_norm_bf16, hidden, nullptr,
        output, tokens, kHidden, epsilon, nullptr, stream);
  }
  for (std::uint64_t row = 0U; row < tokens; ++row) {
    status = LaunchFp8ReferenceTokenQuantization(
        x.attention + row * q_elements, x.output_fp8 + row * q_elements,
        x.output_scale + row, q_elements, stream);
    if (!status.ok()) return status;
    status = LaunchFp8Sm120DirectProjection(
        x.output_fp8 + row * q_elements, x.output_scale + row,
        w.output.weight_e4m3, w.output.weight_scales_bf16,
        x.output_projection + row * kHidden, kHidden, q_elements, stream);
    if (!status.ok()) return status;
    status = LaunchRmsNormResidualBf16(
        x.output_projection + row * kHidden,
        w.post_attention_norm_bf16, hidden + row * kHidden, nullptr,
        output + row * kHidden, 1U, kHidden, epsilon, nullptr, stream);
    if (!status.ok()) return status;
  }
  return Status::Ok();
}

Status LaunchGemma4Moe26BAttentionSm120MtpD2Layer(
    const float* hidden, float* output, std::uint64_t start_position,
    const Gemma4Moe26BAttentionLayerTraits& t,
    const Gemma4Moe26BAttentionReferenceWeights& w,
    const Gemma4Moe26BKvCacheView& cache,
    const Gemma4Moe26BAttentionReferenceWorkspace& x,
    float* decode_attention_workspace, const DecodeControl* row_controls,
    bool shared_d2_attention, bool batched_output_tail,
    bool controlled_positions, float epsilon, cudaStream_t stream) {
  return LaunchGemma4Moe26BAttentionSm120MtpFixedLayer(
      hidden, output, start_position, t, w, cache, x,
      decode_attention_workspace, row_controls, 3U, shared_d2_attention,
      batched_output_tail, controlled_positions, epsilon, stream);
}

Status LaunchGemma4Moe26BAttentionReferenceControlledLayer(
    const float* hidden, float* output,
    const Gemma4Moe26BAttentionLayerTraits& t,
    const Gemma4Moe26BAttentionReferenceWeights& w,
    const Gemma4Moe26BKvCacheView& cache,
    const Gemma4Moe26BAttentionReferenceWorkspace& x,
    const DecodeControl* control, float epsilon, cudaStream_t stream,
    bool rotary_table_prepared) {
  const bool sliding =
      t.attention == Gemma4Moe26BAttentionType::kSliding;
  const std::uint64_t q_elements = t.query_heads * t.head_dimension;
  const std::uint64_t kv_elements = t.kv_heads * t.head_dimension;
  if (hidden == nullptr || output == nullptr || hidden == output ||
      control == nullptr || cache.key == nullptr || cache.value == nullptr ||
      cache.key == cache.value || !ValidPointers(x) ||
      x.post_attention == nullptr ||
      w.input_norm_bf16 == nullptr ||
      w.post_attention_norm_bf16 == nullptr ||
      w.query_norm_bf16 == nullptr || w.key_norm_bf16 == nullptr ||
      w.key_cache_scale_bf16 == nullptr ||
      w.value_cache_scale_bf16 == nullptr || t.layer >= 30U ||
      t.query_heads != kQueryHeads || t.kv_heads == 0U ||
      t.head_dimension == 0U || t.query_heads % t.kv_heads != 0U ||
      t.kv_producer_layer != static_cast<std::int32_t>(t.layer) ||
      cache.capacity == 0U || cache.capacity > t.cache_capacity ||
      (sliding && cache.capacity != t.cache_capacity) ||
      x.score_elements < t.query_heads * cache.capacity ||
      x.score_elements < DecodeAttentionWorkspaceElements(cache.capacity) ||
      !ValidMatrix(w.query, q_elements, kHidden) ||
      !ValidMatrix(w.key, kv_elements, kHidden) ||
      !ValidMatrix(w.output, kHidden, q_elements) ||
      (t.stores_v_projection != !t.reuses_raw_k_for_v) ||
      (t.stores_v_projection && !ValidMatrix(w.value, kv_elements, kHidden)) ||
      (!t.stores_v_projection &&
       (w.value.weight_e4m3 != nullptr ||
        w.value.weight_scales_bf16 != nullptr)) ||
      !std::isfinite(epsilon) || epsilon <= 0.0F) {
    return Invalid("M17 controlled attention contract is invalid");
  }
  Status status = LaunchRmsNormFp8TokenQuantizationBatch(
      hidden, w.input_norm_bf16, x.input_fp8, x.input_scale, 1U, kHidden,
      epsilon, stream);
  if (!status.ok()) return status;
  status = LaunchFp8Sm120GroupedQkvProjection(
      x.input_fp8, x.input_scale, w.query.weight_e4m3,
      w.query.weight_scales_bf16, x.query_raw, q_elements,
      w.key.weight_e4m3, w.key.weight_scales_bf16, x.key_raw, kv_elements,
      t.reuses_raw_k_for_v ? nullptr : w.value.weight_e4m3,
      t.reuses_raw_k_for_v ? nullptr : w.value.weight_scales_bf16,
      t.reuses_raw_k_for_v ? nullptr : x.value_raw,
      t.reuses_raw_k_for_v ? 0U : kv_elements, kHidden, stream);
  if (!status.ok()) return status;
  if (t.reuses_raw_k_for_v) {
    const cudaError_t copied = cudaMemcpyAsync(
        x.value_raw, x.key_raw, kv_elements * sizeof(float),
        cudaMemcpyDeviceToDevice, stream);
    if (copied != cudaSuccess) {
      return CudaFailure("reuse M17 controlled raw K for V", copied);
    }
  }
  const std::uint64_t rotating_pairs = static_cast<std::uint64_t>(
      t.rotary_factor * static_cast<double>(t.head_dimension / 2U));
  if (!rotary_table_prepared) {
    status = LaunchRotaryEmbeddingTableControlled(
        x.rotary_cosine, x.rotary_sine, rotating_pairs, t.head_dimension,
        control, t.rope_theta, t.rope_scaling_factor, stream);
    if (!status.ok()) return status;
  }
  status = LaunchProjectionRmsNormRotaryBf16CurrentTableControlled(
      x.query_raw, w.query_norm_bf16, x.query_normalized, x.key_raw,
      w.key_norm_bf16, x.key_normalized, x.rotary_cosine, x.rotary_sine,
      control, t.query_heads, t.kv_heads, t.head_dimension, t.rotary_factor,
      epsilon, stream);
  if (!status.ok()) return status;
  status = LaunchRmsNormBf16(x.value_raw, nullptr, x.value_normalized,
                             t.kv_heads, t.head_dimension, epsilon, stream);
  if (!status.ok()) return status;
  status = LaunchQuantizeKvFp8Batch(
      x.key_normalized, x.value_normalized, x.staged_key_fp8,
      x.staged_value_fp8, w.key_cache_scale_bf16,
      w.value_cache_scale_bf16, 1U, kv_elements, stream);
  if (!status.ok()) return status;
  status = LaunchAppendKvFp8BatchControlled(
      x.staged_key_fp8, x.staged_value_fp8, cache.key, cache.value, control,
      1U, kv_elements, cache.capacity, stream);
  if (!status.ok()) return status;
  status = LaunchOnlineAttentionDecodeFp8Sm120(
      x.query_normalized, cache.key, cache.value, w.key_cache_scale_bf16,
      w.value_cache_scale_bf16, x.scores, x.attention, control,
      t.query_heads, t.kv_heads, t.head_dimension, cache.capacity, sliding,
      stream);
  if (!status.ok()) return status;
  status = LaunchFp8ReferenceTokenQuantization(
      x.attention, x.output_fp8, x.output_scale, q_elements, stream);
  if (!status.ok()) return status;
  status = LaunchFp8Sm120DirectProjection(
      x.output_fp8, x.output_scale, w.output.weight_e4m3,
      w.output.weight_scales_bf16, x.output_projection, kHidden, q_elements,
      stream);
  if (!status.ok()) return status;
  return LaunchRmsNormResidualBf16(
      x.output_projection, w.post_attention_norm_bf16, hidden,
      x.post_attention, output, 1U, kHidden, epsilon, nullptr, stream);
}

}  // namespace gem16::internal
