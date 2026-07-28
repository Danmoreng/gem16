#include "cuda/layer/reference.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>

namespace gem16::internal {
namespace {

constexpr unsigned kThreads = 256;

Status Invalid(std::string message) {
  return Status(StatusCode::kInvalidArgument, std::move(message));
}

Status CudaFailure(const char* operation, cudaError_t error) {
  return Status(StatusCode::kInternal,
                std::string(operation) + ": " + cudaGetErrorName(error) + ": " +
                    cudaGetErrorString(error));
}

__device__ float BlockSum(float value) {
  __shared__ float scratch[kThreads];
  scratch[threadIdx.x] = value;
  __syncthreads();
  for (unsigned stride = kThreads / 2U; stride != 0U; stride >>= 1U) {
    if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    __syncthreads();
  }
  return scratch[0];
}

__device__ float BlockMaximum(float value) {
  __shared__ float scratch[kThreads];
  scratch[threadIdx.x] = value;
  __syncthreads();
  for (unsigned stride = kThreads / 2U; stride != 0U; stride >>= 1U) {
    if (threadIdx.x < stride) scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
    __syncthreads();
  }
  return scratch[0];
}

__global__ void AttentionScoreKernel(const float* query, const float* key_cache,
                                     float* scores, std::uint64_t kv_heads,
                                     std::uint64_t head_dimension, std::uint64_t tokens,
                                     std::uint64_t pairs, std::uint64_t queries_per_kv,
                                     std::uint64_t cache_capacity,
                                     std::uint64_t first_slot) {
  const std::uint64_t pair = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair >= pairs) return;
  const std::uint64_t query_head = pair / tokens;
  const std::uint64_t token = pair % tokens;
  const std::uint64_t unwrapped_slot = first_slot + token;
  const std::uint64_t cache_slot = unwrapped_slot < cache_capacity
                                       ? unwrapped_slot
                                       : unwrapped_slot - cache_capacity;
  const std::uint64_t kv_head = query_head / queries_per_kv;
  const float* query_head_data = query + query_head * head_dimension;
  const float* key = key_cache + (cache_slot * kv_heads + kv_head) * head_dimension;
  float score = 0.0F;
  for (std::uint64_t dimension = 0; dimension < head_dimension; ++dimension) {
    score = fmaf(query_head_data[dimension], key[dimension], score);
  }
  scores[pair] = score;
}

__global__ void AttentionScoreFp8Kernel(
    const float* query, const std::uint8_t* key_cache,
    const std::uint16_t* key_scale_bf16, float* scores,
    std::uint64_t kv_heads, std::uint64_t head_dimension,
    std::uint64_t tokens, std::uint64_t pairs,
    std::uint64_t queries_per_kv, std::uint64_t cache_capacity,
    std::uint64_t first_slot) {
  const std::uint64_t pair =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair >= pairs) return;
  const std::uint64_t query_head = pair / tokens;
  const std::uint64_t token = pair % tokens;
  const std::uint64_t unwrapped_slot = first_slot + token;
  const std::uint64_t cache_slot = unwrapped_slot < cache_capacity
                                       ? unwrapped_slot
                                       : unwrapped_slot - cache_capacity;
  const std::uint64_t kv_head = query_head / queries_per_kv;
  const float* query_head_data = query + query_head * head_dimension;
  const std::uint8_t* key =
      key_cache + (cache_slot * kv_heads + kv_head) * head_dimension;
  const float key_scale =
      static_cast<float>(__ushort_as_bfloat16(key_scale_bf16[0]));
  float score = 0.0F;
  for (std::uint64_t dimension = 0; dimension < head_dimension; ++dimension) {
    __nv_fp8_e4m3 quantized;
    quantized.__x = key[dimension];
    score = fmaf(query_head_data[dimension],
                 static_cast<float>(quantized) * key_scale, score);
  }
  scores[pair] = score;
}

__global__ void AttentionSoftmaxKernel(float* scores, std::uint64_t tokens) {
  float local_maximum = -FLT_MAX;
  const std::uint64_t base = static_cast<std::uint64_t>(blockIdx.x) * tokens;
  for (std::uint64_t token = threadIdx.x; token < tokens; token += blockDim.x) {
    local_maximum = fmaxf(local_maximum, scores[base + token]);
  }
  const float maximum = BlockMaximum(local_maximum);
  float local_sum = 0.0F;
  for (std::uint64_t token = threadIdx.x; token < tokens; token += blockDim.x) {
    const float probability = expf(scores[base + token] - maximum);
    scores[base + token] = probability;
    local_sum += probability;
  }
  const float denominator = BlockSum(local_sum);
  for (std::uint64_t token = threadIdx.x; token < tokens; token += blockDim.x) {
    scores[base + token] /= denominator;
  }
}

__global__ void AttentionValueKernel(const float* scores, const float* value_cache,
                                     float* output, std::uint64_t query_heads,
                                     std::uint64_t kv_heads, std::uint64_t head_dimension,
                                     std::uint64_t tokens, std::uint64_t elements,
                                     std::uint64_t queries_per_kv,
                                     std::uint64_t cache_capacity,
                                     std::uint64_t first_slot) {
  const std::uint64_t element = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= elements) return;
  const std::uint64_t query_head = element / head_dimension;
  const std::uint64_t dimension = element % head_dimension;
  const std::uint64_t kv_head = query_head / queries_per_kv;
  float value = 0.0F;
  for (std::uint64_t token = 0; token < tokens; ++token) {
    const std::uint64_t unwrapped_slot = first_slot + token;
    const std::uint64_t cache_slot = unwrapped_slot < cache_capacity
                                         ? unwrapped_slot
                                         : unwrapped_slot - cache_capacity;
    const std::uint64_t cache_offset =
        (cache_slot * kv_heads + kv_head) * head_dimension + dimension;
    value = fmaf(scores[query_head * tokens + token], value_cache[cache_offset], value);
  }
  output[element] = value;
  (void)query_heads;
}

__global__ void AttentionValueFp8Kernel(
    const float* scores, const std::uint8_t* value_cache,
    const std::uint16_t* value_scale_bf16, float* output,
    std::uint64_t query_heads, std::uint64_t kv_heads,
    std::uint64_t head_dimension, std::uint64_t tokens,
    std::uint64_t elements, std::uint64_t queries_per_kv,
    std::uint64_t cache_capacity, std::uint64_t first_slot) {
  const std::uint64_t element =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= elements) return;
  const std::uint64_t query_head = element / head_dimension;
  const std::uint64_t dimension = element % head_dimension;
  const std::uint64_t kv_head = query_head / queries_per_kv;
  const float value_scale =
      static_cast<float>(__ushort_as_bfloat16(value_scale_bf16[0]));
  float value = 0.0F;
  for (std::uint64_t token = 0; token < tokens; ++token) {
    const std::uint64_t unwrapped_slot = first_slot + token;
    const std::uint64_t cache_slot = unwrapped_slot < cache_capacity
                                         ? unwrapped_slot
                                         : unwrapped_slot - cache_capacity;
    const std::uint64_t cache_offset =
        (cache_slot * kv_heads + kv_head) * head_dimension + dimension;
    __nv_fp8_e4m3 quantized;
    quantized.__x = value_cache[cache_offset];
    value = fmaf(scores[query_head * tokens + token],
                 static_cast<float>(quantized) * value_scale, value);
  }
  output[element] = value;
  (void)query_heads;
}

__device__ std::uint64_t ControlledTokenCount(
    const DecodeControl* control, std::uint64_t cache_capacity,
    bool sliding) {
  const std::uint64_t count = control->position + 1U;
  return sliding && count > cache_capacity ? cache_capacity : count;
}

__device__ std::uint64_t ControlledFirstSlot(
    const DecodeControl* control, std::uint64_t cache_capacity,
    bool sliding) {
  const std::uint64_t count = control->position + 1U;
  return sliding && count > cache_capacity ? count % cache_capacity : 0U;
}

template <typename CacheType, bool kFp8>
__global__ void ControlledAttentionScoreKernel(
    const float* query, const CacheType* key_cache,
    const std::uint16_t* key_scale_bf16, float* scores,
    const DecodeControl* control, std::uint64_t query_heads,
    std::uint64_t kv_heads, std::uint64_t head_dimension,
    std::uint64_t cache_capacity, bool sliding) {
  const std::uint64_t query_head = blockIdx.x;
  if (query_head >= query_heads) return;
  const std::uint64_t tokens =
      ControlledTokenCount(control, cache_capacity, sliding);
  const std::uint64_t first_slot =
      ControlledFirstSlot(control, cache_capacity, sliding);
  const std::uint64_t kv_head = query_head / (query_heads / kv_heads);
  const float* query_head_data = query + query_head * head_dimension;
  const float key_scale =
      kFp8 ? static_cast<float>(__ushort_as_bfloat16(key_scale_bf16[0]))
           : 1.0F;
  for (std::uint64_t token = threadIdx.x; token < tokens;
       token += blockDim.x) {
    const std::uint64_t unwrapped_slot = first_slot + token;
    const std::uint64_t cache_slot = unwrapped_slot < cache_capacity
                                         ? unwrapped_slot
                                         : unwrapped_slot - cache_capacity;
    const CacheType* key =
        key_cache + (cache_slot * kv_heads + kv_head) * head_dimension;
    float score = 0.0F;
    for (std::uint64_t dimension = 0; dimension < head_dimension;
         ++dimension) {
      float key_value;
      if constexpr (kFp8) {
        __nv_fp8_e4m3 quantized;
        quantized.__x = key[dimension];
        key_value = static_cast<float>(quantized) * key_scale;
      } else {
        key_value = key[dimension];
      }
      score = fmaf(query_head_data[dimension], key_value, score);
    }
    scores[query_head * tokens + token] = score;
  }
}

__global__ void ControlledAttentionSoftmaxKernel(
    float* scores, const DecodeControl* control,
    std::uint64_t cache_capacity, bool sliding) {
  const std::uint64_t tokens =
      ControlledTokenCount(control, cache_capacity, sliding);
  float local_maximum = -FLT_MAX;
  const std::uint64_t base = static_cast<std::uint64_t>(blockIdx.x) * tokens;
  for (std::uint64_t token = threadIdx.x; token < tokens;
       token += blockDim.x) {
    local_maximum = fmaxf(local_maximum, scores[base + token]);
  }
  const float maximum = BlockMaximum(local_maximum);
  float local_sum = 0.0F;
  for (std::uint64_t token = threadIdx.x; token < tokens;
       token += blockDim.x) {
    const float probability = expf(scores[base + token] - maximum);
    scores[base + token] = probability;
    local_sum += probability;
  }
  const float denominator = BlockSum(local_sum);
  for (std::uint64_t token = threadIdx.x; token < tokens;
       token += blockDim.x) {
    scores[base + token] /= denominator;
  }
}

template <typename CacheType, bool kFp8>
__global__ void ControlledAttentionValueKernel(
    const float* scores, const CacheType* value_cache,
    const std::uint16_t* value_scale_bf16, float* output,
    const DecodeControl* control, std::uint64_t query_heads,
    std::uint64_t kv_heads, std::uint64_t head_dimension,
    std::uint64_t cache_capacity, bool sliding, std::uint64_t elements) {
  const std::uint64_t element =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= elements) return;
  const std::uint64_t tokens =
      ControlledTokenCount(control, cache_capacity, sliding);
  const std::uint64_t first_slot =
      ControlledFirstSlot(control, cache_capacity, sliding);
  const std::uint64_t query_head = element / head_dimension;
  const std::uint64_t dimension = element % head_dimension;
  const std::uint64_t kv_head = query_head / (query_heads / kv_heads);
  const float value_scale =
      kFp8 ? static_cast<float>(__ushort_as_bfloat16(value_scale_bf16[0]))
           : 1.0F;
  float value = 0.0F;
  for (std::uint64_t token = 0; token < tokens; ++token) {
    const std::uint64_t unwrapped_slot = first_slot + token;
    const std::uint64_t cache_slot = unwrapped_slot < cache_capacity
                                         ? unwrapped_slot
                                         : unwrapped_slot - cache_capacity;
    const std::uint64_t offset =
        (cache_slot * kv_heads + kv_head) * head_dimension + dimension;
    float cache_value;
    if constexpr (kFp8) {
      __nv_fp8_e4m3 quantized;
      quantized.__x = value_cache[offset];
      cache_value = static_cast<float>(quantized) * value_scale;
    } else {
      cache_value = value_cache[offset];
    }
    value = fmaf(scores[query_head * tokens + token], cache_value, value);
  }
  output[element] = value;
}

__device__ __forceinline__ std::uint64_t PrefillWindowStart(
    std::uint64_t position, std::uint64_t capacity, bool sliding) {
  return sliding && position + 1U > capacity ? position + 1U - capacity : 0U;
}

template <typename CacheType, bool kFp8>
__global__ void PrefillScoreKernel(
    const float* query, const CacheType* chunk_key,
    const CacheType* key_cache, const std::uint16_t* key_scale_bf16,
    float* scores, std::uint64_t start_position, std::uint64_t tokens,
    std::uint64_t query_heads, std::uint64_t kv_heads,
    std::uint64_t head_dimension, std::uint64_t cache_capacity,
    std::uint64_t score_stride, bool sliding, std::uint64_t total_scores) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total_scores) return;
  const std::uint64_t query_token = index / (query_heads * score_stride);
  const std::uint64_t remainder = index % (query_heads * score_stride);
  const std::uint64_t query_head = remainder / score_stride;
  const std::uint64_t score_slot = remainder % score_stride;
  const std::uint64_t position = start_position + query_token;
  const std::uint64_t window_start =
      PrefillWindowStart(position, cache_capacity, sliding);
  const std::uint64_t key_count = position - window_start + 1U;
  if (score_slot >= key_count) {
    scores[index] = -FLT_MAX;
    return;
  }
  const std::uint64_t absolute_key = window_start + score_slot;
  const std::uint64_t kv_head = query_head / (query_heads / kv_heads);
  const std::uint64_t source_token =
      absolute_key >= start_position ? absolute_key - start_position
                                     : absolute_key % cache_capacity;
  const CacheType* source = absolute_key >= start_position ? chunk_key : key_cache;
  source += (source_token * kv_heads + kv_head) * head_dimension;
  const float* query_head_data =
      query + (query_token * query_heads + query_head) * head_dimension;
  const float scale = kFp8
                          ? static_cast<float>(__ushort_as_bfloat16(key_scale_bf16[0]))
                          : 1.0F;
  float score = 0.0F;
  for (std::uint64_t dimension = 0; dimension < head_dimension; ++dimension) {
    float key_value;
    if constexpr (kFp8) {
      __nv_fp8_e4m3 quantized;
      quantized.__x = source[dimension];
      key_value = static_cast<float>(quantized) * scale;
    } else {
      key_value = source[dimension];
    }
    score = fmaf(query_head_data[dimension], key_value, score);
  }
  scores[index] = score;
}

__global__ void PrefillSoftmaxKernel(
    float* scores, std::uint64_t start_position, std::uint64_t query_heads,
    std::uint64_t cache_capacity, std::uint64_t score_stride, bool sliding) {
  const std::uint64_t vector = blockIdx.x;
  const std::uint64_t query_token = vector / query_heads;
  const std::uint64_t position = start_position + query_token;
  const std::uint64_t window_start =
      PrefillWindowStart(position, cache_capacity, sliding);
  const std::uint64_t key_count = position - window_start + 1U;
  float* vector_scores = scores + vector * score_stride;
  float local_maximum = -FLT_MAX;
  for (std::uint64_t key = threadIdx.x; key < key_count; key += blockDim.x) {
    local_maximum = fmaxf(local_maximum, vector_scores[key]);
  }
  const float maximum = BlockMaximum(local_maximum);
  float local_sum = 0.0F;
  for (std::uint64_t key = threadIdx.x; key < key_count; key += blockDim.x) {
    const float probability = expf(vector_scores[key] - maximum);
    vector_scores[key] = probability;
    local_sum += probability;
  }
  const float denominator = BlockSum(local_sum);
  for (std::uint64_t key = threadIdx.x; key < key_count; key += blockDim.x) {
    vector_scores[key] /= denominator;
  }
}

template <typename CacheType, bool kFp8>
__global__ void PrefillValueKernel(
    const float* scores, const CacheType* chunk_value,
    const CacheType* value_cache, const std::uint16_t* value_scale_bf16,
    float* output, std::uint64_t start_position, std::uint64_t query_heads,
    std::uint64_t kv_heads, std::uint64_t head_dimension,
    std::uint64_t cache_capacity, std::uint64_t score_stride, bool sliding,
    std::uint64_t total_elements) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total_elements) return;
  const std::uint64_t elements_per_token = query_heads * head_dimension;
  const std::uint64_t query_token = index / elements_per_token;
  const std::uint64_t in_token = index % elements_per_token;
  const std::uint64_t query_head = in_token / head_dimension;
  const std::uint64_t dimension = in_token % head_dimension;
  const std::uint64_t kv_head = query_head / (query_heads / kv_heads);
  const std::uint64_t position = start_position + query_token;
  const std::uint64_t window_start =
      PrefillWindowStart(position, cache_capacity, sliding);
  const std::uint64_t key_count = position - window_start + 1U;
  const float scale = kFp8
                          ? static_cast<float>(__ushort_as_bfloat16(value_scale_bf16[0]))
                          : 1.0F;
  float sum = 0.0F;
  for (std::uint64_t key = 0; key < key_count; ++key) {
    const std::uint64_t absolute_key = window_start + key;
    const std::uint64_t source_token =
        absolute_key >= start_position ? absolute_key - start_position
                                       : absolute_key % cache_capacity;
    const CacheType* source = absolute_key >= start_position ? chunk_value : value_cache;
    const std::uint64_t source_index =
        (source_token * kv_heads + kv_head) * head_dimension + dimension;
    float value;
    if constexpr (kFp8) {
      __nv_fp8_e4m3 quantized;
      quantized.__x = source[source_index];
      value = static_cast<float>(quantized) * scale;
    } else {
      value = source[source_index];
    }
    sum = fmaf(scores[(query_token * query_heads + query_head) * score_stride + key],
               value, sum);
  }
  output[index] = sum;
}

// Fuses score, softmax, and value phases while preserving the reference
// operation order within every dot product, reduction, and value accumulator.
template <typename CacheType, bool kFp8>
__global__ void FusedCausalAttentionPrefillKernel(
    const float* query, const CacheType* chunk_key,
    const CacheType* chunk_value, const CacheType* key_cache,
    const CacheType* value_cache, const std::uint16_t* key_scale_bf16,
    const std::uint16_t* value_scale_bf16, float* scores, float* output,
    std::uint64_t start_position, std::uint64_t tokens,
    std::uint64_t query_heads, std::uint64_t kv_heads,
    std::uint64_t head_dimension, std::uint64_t cache_capacity,
    std::uint64_t score_stride, bool sliding) {
  const std::uint64_t vector = blockIdx.x;
  const std::uint64_t query_token = vector / query_heads;
  const std::uint64_t query_head = vector % query_heads;
  const std::uint64_t kv_head =
      query_head / (query_heads / kv_heads);
  const std::uint64_t position = start_position + query_token;
  const std::uint64_t window_start =
      PrefillWindowStart(position, cache_capacity, sliding);
  const std::uint64_t key_count = position - window_start + 1U;
  const float* query_head_data =
      query + (query_token * query_heads + query_head) * head_dimension;
  float* vector_scores = scores + vector * score_stride;
  const float key_scale =
      kFp8 ? static_cast<float>(__ushort_as_bfloat16(key_scale_bf16[0]))
           : 1.0F;
  const float value_scale =
      kFp8 ? static_cast<float>(__ushort_as_bfloat16(value_scale_bf16[0]))
           : 1.0F;

  for (std::uint64_t key = threadIdx.x; key < key_count;
       key += blockDim.x) {
    const std::uint64_t absolute_key = window_start + key;
    const bool in_chunk = absolute_key >= start_position;
    const std::uint64_t source_token =
        in_chunk ? absolute_key - start_position
                 : absolute_key % cache_capacity;
    const CacheType* source = in_chunk ? chunk_key : key_cache;
    source += (source_token * kv_heads + kv_head) * head_dimension;
    float score = 0.0F;
    if constexpr (kFp8) {
      std::uint64_t dimension = 0U;
      // CUDA device allocations are at least 256-byte aligned. Requiring the
      // row extent and the resolved row address to retain uint4 alignment
      // makes the wide load explicit; uncommon test geometries stay scalar.
      if (head_dimension % sizeof(uint4) == 0U &&
          reinterpret_cast<std::uintptr_t>(source) % alignof(uint4) == 0U) {
        for (; dimension < head_dimension; dimension += 16U) {
          const uint4 packed =
              *reinterpret_cast<const uint4*>(source + dimension);
          const std::uint32_t words[4] = {packed.x, packed.y, packed.z,
                                          packed.w};
#pragma unroll
          for (unsigned index = 0U; index < 16U; ++index) {
            __nv_fp8_e4m3 quantized;
            quantized.__x = static_cast<std::uint8_t>(
                words[index >> 2U] >> ((index & 3U) * 8U));
            score = fmaf(query_head_data[dimension + index],
                         static_cast<float>(quantized) * key_scale, score);
          }
        }
      }
      for (; dimension < head_dimension; ++dimension) {
        __nv_fp8_e4m3 quantized;
        quantized.__x = source[dimension];
        score = fmaf(query_head_data[dimension],
                     static_cast<float>(quantized) * key_scale, score);
      }
    } else {
      for (std::uint64_t dimension = 0; dimension < head_dimension;
           ++dimension) {
        score = fmaf(query_head_data[dimension], source[dimension], score);
      }
    }
    vector_scores[key] = score;
  }
  __syncthreads();

  float local_maximum = -FLT_MAX;
  for (std::uint64_t key = threadIdx.x; key < key_count;
       key += blockDim.x) {
    local_maximum = fmaxf(local_maximum, vector_scores[key]);
  }
  const float maximum = BlockMaximum(local_maximum);
  float local_sum = 0.0F;
  for (std::uint64_t key = threadIdx.x; key < key_count;
       key += blockDim.x) {
    const float probability = expf(vector_scores[key] - maximum);
    vector_scores[key] = probability;
    local_sum += probability;
  }
  const float denominator = BlockSum(local_sum);
  for (std::uint64_t key = threadIdx.x; key < key_count;
       key += blockDim.x) {
    vector_scores[key] /= denominator;
  }
  __syncthreads();

  float* output_head =
      output + (query_token * query_heads + query_head) * head_dimension;
  for (std::uint64_t dimension = threadIdx.x; dimension < head_dimension;
       dimension += blockDim.x) {
    float sum = 0.0F;
    for (std::uint64_t key = 0; key < key_count; ++key) {
      const std::uint64_t absolute_key = window_start + key;
      const bool in_chunk = absolute_key >= start_position;
      const std::uint64_t source_token =
          in_chunk ? absolute_key - start_position
                   : absolute_key % cache_capacity;
      const CacheType* source = in_chunk ? chunk_value : value_cache;
      const std::uint64_t source_index =
          (source_token * kv_heads + kv_head) * head_dimension + dimension;
      float value;
      if constexpr (kFp8) {
        __nv_fp8_e4m3 quantized;
        quantized.__x = source[source_index];
        value = static_cast<float>(quantized) * value_scale;
      } else {
        value = source[source_index];
      }
      sum = fmaf(vector_scores[key], value, sum);
    }
    output_head[dimension] = sum;
  }
}

std::uint64_t Blocks(std::uint64_t elements) {
  return (elements + kThreads - 1U) / kThreads;
}

bool ValidGrid(std::uint64_t blocks) {
  return blocks <= static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max());
}


}  // namespace

Status LaunchLocalAttentionDecode(const float* query, const float* key_cache,
                                  const float* value_cache, float* scores, float* output,
                                  std::uint64_t query_heads, std::uint64_t kv_heads,
                                  std::uint64_t head_dimension, std::uint64_t tokens,
                                  cudaStream_t stream, std::uint64_t cache_capacity,
                                  std::uint64_t first_slot) {
  if (query == nullptr || key_cache == nullptr || value_cache == nullptr || scores == nullptr ||
      output == nullptr) return Invalid("local attention requires non-null pointers");
  if (cache_capacity == 0U) cache_capacity = tokens;
  if (query_heads == 0U || kv_heads == 0U || head_dimension == 0U || tokens == 0U ||
      tokens > cache_capacity || first_slot >= cache_capacity ||
      query_heads % kv_heads != 0U || query_heads > std::numeric_limits<std::uint64_t>::max() / tokens ||
      query_heads > std::numeric_limits<std::uint64_t>::max() / head_dimension) {
    return Invalid("local attention geometry is invalid");
  }
  const std::uint64_t pairs = query_heads * tokens;
  const std::uint64_t elements = query_heads * head_dimension;
  const std::uint64_t score_blocks = Blocks(pairs);
  const std::uint64_t value_blocks = Blocks(elements);
  if (!ValidGrid(score_blocks) || !ValidGrid(value_blocks) || !ValidGrid(query_heads)) {
    return Invalid("local attention grid exceeds CUDA limits");
  }
  const std::uint64_t queries_per_kv = query_heads / kv_heads;
  AttentionScoreKernel<<<static_cast<unsigned>(score_blocks), kThreads, 0, stream>>>(
      query, key_cache, scores, kv_heads, head_dimension, tokens, pairs,
      queries_per_kv, cache_capacity, first_slot);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) return CudaFailure("launch attention scores", error);
  AttentionSoftmaxKernel<<<static_cast<unsigned>(query_heads), kThreads, 0, stream>>>(scores, tokens);
  error = cudaGetLastError();
  if (error != cudaSuccess) return CudaFailure("launch attention softmax", error);
  AttentionValueKernel<<<static_cast<unsigned>(value_blocks), kThreads, 0, stream>>>(
      scores, value_cache, output, query_heads, kv_heads, head_dimension, tokens, elements,
      queries_per_kv, cache_capacity, first_slot);
  error = cudaGetLastError();
  return error == cudaSuccess ? Status::Ok() : CudaFailure("launch attention values", error);
}

Status LaunchLocalAttentionDecodeFp8(
    const float* query, const std::uint8_t* key_cache,
    const std::uint8_t* value_cache, const std::uint16_t* key_scale_bf16,
    const std::uint16_t* value_scale_bf16, float* scores, float* output,
    std::uint64_t query_heads, std::uint64_t kv_heads,
    std::uint64_t head_dimension, std::uint64_t tokens,
    cudaStream_t stream, std::uint64_t cache_capacity,
    std::uint64_t first_slot) {
  if (query == nullptr || key_cache == nullptr || value_cache == nullptr ||
      key_scale_bf16 == nullptr || value_scale_bf16 == nullptr ||
      scores == nullptr || output == nullptr) {
    return Invalid("FP8 local attention requires non-null pointers and scales");
  }
  if (cache_capacity == 0U) cache_capacity = tokens;
  if (query_heads == 0U || kv_heads == 0U || head_dimension == 0U ||
      tokens == 0U || query_heads % kv_heads != 0U ||
      tokens > cache_capacity || first_slot >= cache_capacity ||
      query_heads > std::numeric_limits<std::uint64_t>::max() / tokens ||
      query_heads > std::numeric_limits<std::uint64_t>::max() / head_dimension) {
    return Invalid("FP8 local attention geometry is invalid");
  }
  const std::uint64_t pairs = query_heads * tokens;
  const std::uint64_t elements = query_heads * head_dimension;
  const std::uint64_t score_blocks = Blocks(pairs);
  const std::uint64_t value_blocks = Blocks(elements);
  if (!ValidGrid(score_blocks) || !ValidGrid(value_blocks) ||
      !ValidGrid(query_heads)) {
    return Invalid("FP8 local attention grid exceeds CUDA limits");
  }
  const std::uint64_t queries_per_kv = query_heads / kv_heads;
  AttentionScoreFp8Kernel<<<static_cast<unsigned>(score_blocks), kThreads, 0,
                            stream>>>(
      query, key_cache, key_scale_bf16, scores, kv_heads, head_dimension,
      tokens, pairs, queries_per_kv, cache_capacity, first_slot);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaFailure("launch FP8 attention scores", error);
  }
  AttentionSoftmaxKernel<<<static_cast<unsigned>(query_heads), kThreads, 0,
                           stream>>>(scores, tokens);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaFailure("launch FP8 attention softmax", error);
  }
  AttentionValueFp8Kernel<<<static_cast<unsigned>(value_blocks), kThreads, 0,
                            stream>>>(
      scores, value_cache, value_scale_bf16, output, query_heads, kv_heads,
      head_dimension, tokens, elements, queries_per_kv, cache_capacity,
      first_slot);
  error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch FP8 attention values", error);
}

template <typename CacheType, bool kFp8>
Status LaunchLocalAttentionDecodeControlledImpl(
    const float* query, const CacheType* key_cache,
    const CacheType* value_cache, const std::uint16_t* key_scale_bf16,
    const std::uint16_t* value_scale_bf16, float* scores, float* output,
    const DecodeControl* control, std::uint64_t query_heads,
    std::uint64_t kv_heads, std::uint64_t head_dimension,
    std::uint64_t cache_capacity, bool sliding, cudaStream_t stream) {
  if (query == nullptr || key_cache == nullptr || value_cache == nullptr ||
      scores == nullptr || output == nullptr || control == nullptr ||
      (kFp8 && (key_scale_bf16 == nullptr || value_scale_bf16 == nullptr)) ||
      query_heads == 0U || kv_heads == 0U || head_dimension == 0U ||
      cache_capacity == 0U || query_heads % kv_heads != 0U ||
      query_heads > std::numeric_limits<std::uint64_t>::max() /
                        cache_capacity ||
      query_heads > std::numeric_limits<std::uint64_t>::max() /
                        head_dimension) {
    return Invalid("controlled attention arguments are invalid");
  }
  const std::uint64_t elements = query_heads * head_dimension;
  const std::uint64_t value_blocks = Blocks(elements);
  if (!ValidGrid(value_blocks) ||
      !ValidGrid(query_heads)) {
    return Invalid("controlled attention grid exceeds CUDA limits");
  }
  ControlledAttentionScoreKernel<CacheType, kFp8>
      <<<static_cast<unsigned>(query_heads), kThreads, 0, stream>>>(
          query, key_cache, key_scale_bf16, scores, control, query_heads,
          kv_heads, head_dimension, cache_capacity, sliding);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaFailure("launch controlled attention scores", error);
  }
  ControlledAttentionSoftmaxKernel
      <<<static_cast<unsigned>(query_heads), kThreads, 0, stream>>>(
          scores, control, cache_capacity, sliding);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaFailure("launch controlled attention softmax", error);
  }
  ControlledAttentionValueKernel<CacheType, kFp8>
      <<<static_cast<unsigned>(value_blocks), kThreads, 0, stream>>>(
          scores, value_cache, value_scale_bf16, output, control, query_heads,
          kv_heads, head_dimension, cache_capacity, sliding, elements);
  error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch controlled attention values", error);
}

Status LaunchLocalAttentionDecodeControlled(
    const float* query, const float* key_cache, const float* value_cache,
    float* scores, float* output, const DecodeControl* control,
    std::uint64_t query_heads, std::uint64_t kv_heads,
    std::uint64_t head_dimension, std::uint64_t cache_capacity,
    bool sliding, cudaStream_t stream) {
  return LaunchLocalAttentionDecodeControlledImpl<float, false>(
      query, key_cache, value_cache, nullptr, nullptr, scores, output, control,
      query_heads, kv_heads, head_dimension, cache_capacity, sliding, stream);
}

Status LaunchLocalAttentionDecodeFp8Controlled(
    const float* query, const std::uint8_t* key_cache,
    const std::uint8_t* value_cache, const std::uint16_t* key_scale_bf16,
    const std::uint16_t* value_scale_bf16, float* scores, float* output,
    const DecodeControl* control, std::uint64_t query_heads,
    std::uint64_t kv_heads, std::uint64_t head_dimension,
    std::uint64_t cache_capacity, bool sliding, cudaStream_t stream) {
  return LaunchLocalAttentionDecodeControlledImpl<std::uint8_t, true>(
      query, key_cache, value_cache, key_scale_bf16, value_scale_bf16, scores,
      output, control, query_heads, kv_heads, head_dimension, cache_capacity,
      sliding, stream);
}

template <typename CacheType, bool kFp8>
Status LaunchCausalAttentionPrefillImpl(
    const float* query, const CacheType* chunk_key,
    const CacheType* chunk_value, const CacheType* key_cache,
    const CacheType* value_cache, const std::uint16_t* key_scale_bf16,
    const std::uint16_t* value_scale_bf16, float* scores, float* output,
    std::uint64_t start_position, std::uint64_t tokens,
    std::uint64_t query_heads, std::uint64_t kv_heads,
    std::uint64_t head_dimension, std::uint64_t cache_capacity, bool sliding,
    cudaStream_t stream) {
  if (query == nullptr || chunk_key == nullptr || chunk_value == nullptr ||
      key_cache == nullptr || value_cache == nullptr || scores == nullptr ||
      output == nullptr || (kFp8 && (key_scale_bf16 == nullptr ||
                                    value_scale_bf16 == nullptr)) ||
      tokens == 0U || query_heads == 0U || kv_heads == 0U ||
      query_heads % kv_heads != 0U || head_dimension == 0U ||
      cache_capacity == 0U || tokens > cache_capacity) {
    return Invalid("causal prefill attention arguments are invalid");
  }
  const std::uint64_t final_position = start_position + tokens - 1U;
  const std::uint64_t score_stride =
      sliding ? std::min(final_position + 1U, cache_capacity)
              : final_position + 1U;
  const std::uint64_t total_scores = tokens * query_heads * score_stride;
  const std::uint64_t total_elements = tokens * query_heads * head_dimension;
  const std::uint64_t score_blocks = Blocks(total_scores);
  const std::uint64_t value_blocks = Blocks(total_elements);
  if (!ValidGrid(score_blocks) || !ValidGrid(value_blocks) ||
      tokens * query_heads > static_cast<std::uint64_t>(
                                   std::numeric_limits<unsigned>::max())) {
    return Invalid("causal prefill attention grid exceeds CUDA limits");
  }
  PrefillScoreKernel<CacheType, kFp8>
      <<<static_cast<unsigned>(score_blocks), kThreads, 0, stream>>>(
          query, chunk_key, key_cache, key_scale_bf16, scores, start_position,
          tokens, query_heads, kv_heads, head_dimension, cache_capacity,
          score_stride, sliding, total_scores);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) return CudaFailure("launch prefill scores", error);
  PrefillSoftmaxKernel<<<static_cast<unsigned>(tokens * query_heads), kThreads,
                         0, stream>>>(scores, start_position, query_heads,
                                     cache_capacity, score_stride, sliding);
  error = cudaGetLastError();
  if (error != cudaSuccess) return CudaFailure("launch prefill softmax", error);
  PrefillValueKernel<CacheType, kFp8>
      <<<static_cast<unsigned>(value_blocks), kThreads, 0, stream>>>(
          scores, chunk_value, value_cache, value_scale_bf16, output,
          start_position, query_heads, kv_heads, head_dimension, cache_capacity,
          score_stride, sliding, total_elements);
  error = cudaGetLastError();
  return error == cudaSuccess ? Status::Ok()
                              : CudaFailure("launch prefill values", error);
}

Status LaunchCausalAttentionPrefill(
    const float* query, const float* chunk_key, const float* chunk_value,
    const float* key_cache, const float* value_cache, float* scores,
    float* output, std::uint64_t start_position, std::uint64_t tokens,
    std::uint64_t query_heads, std::uint64_t kv_heads,
    std::uint64_t head_dimension, std::uint64_t cache_capacity, bool sliding,
    cudaStream_t stream) {
  return LaunchCausalAttentionPrefillImpl<float, false>(
      query, chunk_key, chunk_value, key_cache, value_cache, nullptr, nullptr,
      scores, output, start_position, tokens, query_heads, kv_heads,
      head_dimension, cache_capacity, sliding, stream);
}

Status LaunchCausalAttentionPrefillFp8(
    const float* query, const std::uint8_t* chunk_key,
    const std::uint8_t* chunk_value, const std::uint8_t* key_cache,
    const std::uint8_t* value_cache, const std::uint16_t* key_scale_bf16,
    const std::uint16_t* value_scale_bf16, float* scores, float* output,
    std::uint64_t start_position, std::uint64_t tokens,
    std::uint64_t query_heads, std::uint64_t kv_heads,
    std::uint64_t head_dimension, std::uint64_t cache_capacity, bool sliding,
    cudaStream_t stream) {
  return LaunchCausalAttentionPrefillImpl<std::uint8_t, true>(
      query, chunk_key, chunk_value, key_cache, value_cache, key_scale_bf16,
      value_scale_bf16, scores, output, start_position, tokens, query_heads,
      kv_heads, head_dimension, cache_capacity, sliding, stream);
}

template <typename CacheType, bool kFp8>
Status LaunchFusedCausalAttentionPrefillImpl(
    const float* query, const CacheType* chunk_key,
    const CacheType* chunk_value, const CacheType* key_cache,
    const CacheType* value_cache, const std::uint16_t* key_scale_bf16,
    const std::uint16_t* value_scale_bf16, float* scores, float* output,
    std::uint64_t start_position, std::uint64_t tokens,
    std::uint64_t query_heads, std::uint64_t kv_heads,
    std::uint64_t head_dimension, std::uint64_t cache_capacity, bool sliding,
    cudaStream_t stream) {
  if (query == nullptr || chunk_key == nullptr || chunk_value == nullptr ||
      key_cache == nullptr || value_cache == nullptr || scores == nullptr ||
      output == nullptr || (kFp8 && (key_scale_bf16 == nullptr ||
                                    value_scale_bf16 == nullptr)) ||
      tokens == 0U || query_heads == 0U || kv_heads == 0U ||
      query_heads % kv_heads != 0U || head_dimension == 0U ||
      cache_capacity == 0U || tokens > cache_capacity ||
      tokens * query_heads > static_cast<std::uint64_t>(
                                   std::numeric_limits<unsigned>::max())) {
    return Invalid("fused causal prefill attention arguments are invalid");
  }
  const std::uint64_t final_position = start_position + tokens - 1U;
  const std::uint64_t score_stride =
      sliding ? std::min(final_position + 1U, cache_capacity)
              : final_position + 1U;
  FusedCausalAttentionPrefillKernel<CacheType, kFp8>
      <<<static_cast<unsigned>(tokens * query_heads), kThreads, 0, stream>>>(
          query, chunk_key, chunk_value, key_cache, value_cache,
          key_scale_bf16, value_scale_bf16, scores, output, start_position,
          tokens, query_heads, kv_heads, head_dimension, cache_capacity,
          score_stride, sliding);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch fused causal prefill attention", error);
}

Status LaunchFusedCausalAttentionPrefill(
    const float* query, const float* chunk_key, const float* chunk_value,
    const float* key_cache, const float* value_cache, float* scores,
    float* output, std::uint64_t start_position, std::uint64_t tokens,
    std::uint64_t query_heads, std::uint64_t kv_heads,
    std::uint64_t head_dimension, std::uint64_t cache_capacity, bool sliding,
    cudaStream_t stream) {
  return LaunchFusedCausalAttentionPrefillImpl<float, false>(
      query, chunk_key, chunk_value, key_cache, value_cache, nullptr, nullptr,
      scores, output, start_position, tokens, query_heads, kv_heads,
      head_dimension, cache_capacity, sliding, stream);
}

Status LaunchFusedCausalAttentionPrefillFp8(
    const float* query, const std::uint8_t* chunk_key,
    const std::uint8_t* chunk_value, const std::uint8_t* key_cache,
    const std::uint8_t* value_cache, const std::uint16_t* key_scale_bf16,
    const std::uint16_t* value_scale_bf16, float* scores, float* output,
    std::uint64_t start_position, std::uint64_t tokens,
    std::uint64_t query_heads, std::uint64_t kv_heads,
    std::uint64_t head_dimension, std::uint64_t cache_capacity, bool sliding,
    cudaStream_t stream) {
  return LaunchFusedCausalAttentionPrefillImpl<std::uint8_t, true>(
      query, chunk_key, chunk_value, key_cache, value_cache, key_scale_bf16,
      value_scale_bf16, scores, output, start_position, tokens, query_heads,
      kv_heads, head_dimension, cache_capacity, sliding, stream);
}


}  // namespace gem16::internal
