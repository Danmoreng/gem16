#include "cuda/layer/reference.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

namespace gem16gb::internal {
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

template <bool kRoundBf16>
__global__ void RmsNormKernel(const float* input, const std::uint16_t* weight,
                              float* output, std::uint64_t width, float epsilon) {
  const std::uint64_t vector = blockIdx.x;
  const std::uint64_t base = vector * width;
  float squared_sum = 0.0F;
  for (std::uint64_t index = threadIdx.x; index < width; index += blockDim.x) {
    const float value = input[base + index];
    squared_sum = fmaf(value, value, squared_sum);
  }
  const float inverse_rms = rsqrtf(BlockSum(squared_sum) / static_cast<float>(width) + epsilon);
  for (std::uint64_t index = threadIdx.x; index < width; index += blockDim.x) {
    const float scale = weight == nullptr ? 1.0F :
        static_cast<float>(__ushort_as_bfloat16(weight[index]));
    const float value = input[base + index] * inverse_rms * scale;
    if constexpr (kRoundBf16) {
      output[base + index] = static_cast<float>(__float2bfloat16_rn(value));
    } else {
      output[base + index] = value;
    }
  }
}

__global__ void RmsNormResidualBf16Kernel(
    const float* input, const std::uint16_t* weight, const float* residual,
    float* normalized_output, float* output, std::uint64_t width,
    float epsilon, const std::uint16_t* scalar) {
  const std::uint64_t vector = blockIdx.x;
  const std::uint64_t base = vector * width;
  float squared_sum = 0.0F;
  for (std::uint64_t index = threadIdx.x; index < width;
       index += blockDim.x) {
    const float value = input[base + index];
    squared_sum = fmaf(value, value, squared_sum);
  }
  const float inverse_rms =
      rsqrtf(BlockSum(squared_sum) / static_cast<float>(width) + epsilon);
  const float layer_scale =
      scalar == nullptr
          ? 1.0F
          : static_cast<float>(__ushort_as_bfloat16(scalar[0]));
  for (std::uint64_t index = threadIdx.x; index < width;
       index += blockDim.x) {
    const float norm_scale =
        weight == nullptr
            ? 1.0F
            : static_cast<float>(__ushort_as_bfloat16(weight[index]));
    const float normalized = static_cast<float>(__float2bfloat16_rn(
        input[base + index] * inverse_rms * norm_scale));
    if (normalized_output != nullptr) {
      normalized_output[base + index] = normalized;
    }
    float value = static_cast<float>(
        __float2bfloat16_rn(normalized + residual[base + index]));
    if (scalar != nullptr) {
      value = static_cast<float>(__float2bfloat16_rn(value * layer_scale));
    }
    output[base + index] = value;
  }
}

__global__ void RotaryKernel(float* states, std::uint64_t head_dimension,
                             std::uint64_t pair_stride, std::uint64_t rotating_pairs,
                             std::uint64_t frequency_dimension, std::uint64_t position,
                             double theta, double scaling_factor, std::uint64_t pairs) {
  const std::uint64_t pair = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair >= pairs) return;
  const std::uint64_t head = pair / rotating_pairs;
  const std::uint64_t index = pair % rotating_pairs;
  const double exponent = 2.0 * static_cast<double>(index) /
                          static_cast<double>(frequency_dimension);
  const double angle = static_cast<double>(position) /
                       (pow(theta, exponent) * scaling_factor);
  const float cosine = static_cast<float>(cos(angle));
  const float sine = static_cast<float>(sin(angle));
  const std::uint64_t first = head * head_dimension + index;
  const std::uint64_t second = first + pair_stride;
  const float first_value = states[first];
  const float second_value = states[second];
  states[first] = first_value * cosine - second_value * sine;
  states[second] = second_value * cosine + first_value * sine;
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

__global__ void AppendKvFp8Kernel(
    const float* key, const float* value, std::uint8_t* key_cache,
    std::uint8_t* value_cache, const std::uint16_t* key_scale_bf16,
    const std::uint16_t* value_scale_bf16, std::uint64_t offset,
    std::uint64_t elements) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  const float key_scale =
      static_cast<float>(__ushort_as_bfloat16(key_scale_bf16[0]));
  const float value_scale =
      static_cast<float>(__ushort_as_bfloat16(value_scale_bf16[0]));
  const __nv_fp8_e4m3 quantized_key(key[index] / key_scale);
  const __nv_fp8_e4m3 quantized_value(value[index] / value_scale);
  key_cache[offset + index] = quantized_key.__x;
  value_cache[offset + index] = quantized_value.__x;
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

__global__ void ControlledRotaryKernel(
    float* states, std::uint64_t head_dimension, std::uint64_t pair_stride,
    std::uint64_t rotating_pairs, std::uint64_t frequency_dimension,
    const DecodeControl* control, double theta, double scaling_factor,
    std::uint64_t pairs) {
  const std::uint64_t pair =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair >= pairs) return;
  const std::uint64_t head = pair / rotating_pairs;
  const std::uint64_t index = pair % rotating_pairs;
  const double exponent = 2.0 * static_cast<double>(index) /
                          static_cast<double>(frequency_dimension);
  const double angle = static_cast<double>(control->position) /
                       (pow(theta, exponent) * scaling_factor);
  const float cosine = static_cast<float>(cos(angle));
  const float sine = static_cast<float>(sin(angle));
  const std::uint64_t first = head * head_dimension + index;
  const std::uint64_t second = first + pair_stride;
  const float first_value = states[first];
  const float second_value = states[second];
  states[first] = first_value * cosine - second_value * sine;
  states[second] = second_value * cosine + first_value * sine;
}

template <typename CacheType, bool kFp8>
__global__ void ControlledAppendKvKernel(
    const float* key, const float* value, CacheType* key_cache,
    CacheType* value_cache, const std::uint16_t* key_scale_bf16,
    const std::uint16_t* value_scale_bf16, const DecodeControl* control,
    std::uint64_t elements, std::uint64_t cache_capacity, bool sliding) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  const std::uint64_t slot =
      sliding ? control->position % cache_capacity : control->position;
  const std::uint64_t offset = slot * elements + index;
  if constexpr (kFp8) {
    const float key_scale =
        static_cast<float>(__ushort_as_bfloat16(key_scale_bf16[0]));
    const float value_scale =
        static_cast<float>(__ushort_as_bfloat16(value_scale_bf16[0]));
    const __nv_fp8_e4m3 quantized_key(key[index] / key_scale);
    const __nv_fp8_e4m3 quantized_value(value[index] / value_scale);
    key_cache[offset] = quantized_key.__x;
    value_cache[offset] = quantized_value.__x;
  } else {
    key_cache[offset] = key[index];
    value_cache[offset] = value[index];
  }
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

__global__ void ScaleKernel(float* values, const std::uint16_t* scalar,
                            std::uint64_t elements) {
  const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    values[index] *= static_cast<float>(__ushort_as_bfloat16(scalar[0]));
  }
}

__global__ void RotaryBatchKernel(
    float* states, std::uint64_t heads, std::uint64_t head_dimension,
    std::uint64_t pair_stride, std::uint64_t rotating_pairs,
    std::uint64_t frequency_dimension, std::uint64_t start_position,
    double theta, double scaling_factor, std::uint64_t pairs_per_token,
    std::uint64_t total_pairs) {
  const std::uint64_t pair =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair >= total_pairs) return;
  const std::uint64_t token = pair / pairs_per_token;
  const std::uint64_t in_token = pair % pairs_per_token;
  const std::uint64_t head = in_token / rotating_pairs;
  const std::uint64_t index = in_token % rotating_pairs;
  const double exponent = 2.0 * static_cast<double>(index) /
                          static_cast<double>(frequency_dimension);
  const double angle = static_cast<double>(start_position + token) /
                       (pow(theta, exponent) * scaling_factor);
  const float cosine = static_cast<float>(cos(angle));
  const float sine = static_cast<float>(sin(angle));
  float* token_states = states + token * heads * head_dimension;
  const std::uint64_t first = head * head_dimension + index;
  const std::uint64_t second = first + pair_stride;
  const float first_value = token_states[first];
  const float second_value = token_states[second];
  token_states[first] = first_value * cosine - second_value * sine;
  token_states[second] = second_value * cosine + first_value * sine;
}

__global__ void QuantizeKvBatchKernel(
    const float* key, const float* value, std::uint8_t* key_fp8,
    std::uint8_t* value_fp8, const std::uint16_t* key_scale_bf16,
    const std::uint16_t* value_scale_bf16, std::uint64_t elements) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  const float key_scale =
      static_cast<float>(__ushort_as_bfloat16(key_scale_bf16[0]));
  const float value_scale =
      static_cast<float>(__ushort_as_bfloat16(value_scale_bf16[0]));
  const __nv_fp8_e4m3 quantized_key(key[index] / key_scale);
  const __nv_fp8_e4m3 quantized_value(value[index] / value_scale);
  key_fp8[index] = quantized_key.__x;
  value_fp8[index] = quantized_value.__x;
}

template <typename T>
__global__ void AppendKvBatchKernel(
    const T* key, const T* value, T* key_cache, T* value_cache,
    std::uint64_t start_position, std::uint64_t elements_per_token,
    std::uint64_t cache_capacity, std::uint64_t total_elements) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total_elements) return;
  const std::uint64_t token = index / elements_per_token;
  const std::uint64_t element = index % elements_per_token;
  const std::uint64_t slot = (start_position + token) % cache_capacity;
  const std::uint64_t destination = slot * elements_per_token + element;
  key_cache[destination] = key[index];
  value_cache[destination] = value[index];
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

Status LaunchRmsNorm(const float* input, const std::uint16_t* weight_bf16, float* output,
                     std::uint64_t vectors, std::uint64_t width, float epsilon,
                     cudaStream_t stream) {
  if (input == nullptr || output == nullptr) return Invalid("RMSNorm requires non-null input and output");
  if (vectors == 0U || width == 0U || !std::isfinite(epsilon) || epsilon <= 0.0F ||
      !ValidGrid(vectors)) return Invalid("RMSNorm geometry or epsilon is invalid");
  RmsNormKernel<false><<<static_cast<unsigned>(vectors), kThreads, 0, stream>>>(
      input, weight_bf16, output, width, epsilon);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess ? Status::Ok() : CudaFailure("launch RMSNorm", error);
}

Status LaunchRmsNormBf16(const float* input,
                         const std::uint16_t* weight_bf16, float* output,
                         std::uint64_t vectors, std::uint64_t width,
                         float epsilon, cudaStream_t stream) {
  if (input == nullptr || output == nullptr) {
    return Invalid("BF16 RMSNorm requires non-null input and output");
  }
  if (vectors == 0U || width == 0U || !std::isfinite(epsilon) ||
      epsilon <= 0.0F || !ValidGrid(vectors)) {
    return Invalid("BF16 RMSNorm geometry or epsilon is invalid");
  }
  RmsNormKernel<true><<<static_cast<unsigned>(vectors), kThreads, 0, stream>>>(
      input, weight_bf16, output, width, epsilon);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch BF16-boundary RMSNorm", error);
}

Status LaunchRmsNormResidualBf16(
    const float* input, const std::uint16_t* weight_bf16,
    const float* residual, float* normalized_output, float* output,
    std::uint64_t vectors, std::uint64_t width, float epsilon,
    const std::uint16_t* scalar_bf16, cudaStream_t stream) {
  if (input == nullptr || residual == nullptr || output == nullptr) {
    return Invalid("fused RMSNorm residual requires non-null input, residual, and output");
  }
  if (vectors == 0U || width == 0U || !std::isfinite(epsilon) ||
      epsilon <= 0.0F || !ValidGrid(vectors)) {
    return Invalid("fused RMSNorm residual geometry or epsilon is invalid");
  }
  RmsNormResidualBf16Kernel<<<static_cast<unsigned>(vectors), kThreads, 0,
                              stream>>>(
      input, weight_bf16, residual, normalized_output, output, width, epsilon,
      scalar_bf16);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch fused RMSNorm residual", error);
}

Status LaunchRotaryEmbedding(float* states, std::uint64_t heads,
                             std::uint64_t head_dimension, std::uint64_t rotary_dimensions,
                             std::uint64_t position, double theta, cudaStream_t stream) {
  if (states == nullptr) return Invalid("RoPE requires a non-null state pointer");
  if (heads == 0U || rotary_dimensions == 0U || rotary_dimensions > head_dimension ||
      rotary_dimensions % 2U != 0U || !std::isfinite(theta) || theta <= 0.0) {
    return Invalid("RoPE geometry or theta is invalid");
  }
  const std::uint64_t pairs = heads * (rotary_dimensions / 2U);
  const std::uint64_t blocks = Blocks(pairs);
  if (!ValidGrid(blocks)) return Invalid("RoPE grid exceeds CUDA limits");
  RotaryKernel<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
      states, head_dimension, rotary_dimensions / 2U, rotary_dimensions / 2U,
      rotary_dimensions, position, theta, 1.0, pairs);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess ? Status::Ok() : CudaFailure("launch RoPE", error);
}

Status LaunchProportionalRotaryEmbedding(float* states, std::uint64_t heads,
                                         std::uint64_t head_dimension, double rotary_factor,
                                         std::uint64_t position, double theta,
                                         double scaling_factor, cudaStream_t stream) {
  if (states == nullptr) return Invalid("proportional RoPE requires a non-null state pointer");
  if (heads == 0U || head_dimension == 0U || head_dimension % 2U != 0U ||
      !std::isfinite(rotary_factor) || rotary_factor <= 0.0 || rotary_factor > 1.0 ||
      !std::isfinite(theta) || theta <= 0.0 || !std::isfinite(scaling_factor) ||
      scaling_factor <= 0.0) {
    return Invalid("proportional RoPE geometry, factors, or theta are invalid");
  }
  const std::uint64_t half = head_dimension / 2U;
  const std::uint64_t rotating_pairs =
      static_cast<std::uint64_t>(rotary_factor * static_cast<double>(half));
  if (rotating_pairs == 0U || rotating_pairs > half ||
      heads > std::numeric_limits<std::uint64_t>::max() / rotating_pairs) {
    return Invalid("proportional RoPE factor selects no valid pairs");
  }
  const std::uint64_t pairs = heads * rotating_pairs;
  const std::uint64_t blocks = Blocks(pairs);
  if (!ValidGrid(blocks)) return Invalid("proportional RoPE grid exceeds CUDA limits");
  RotaryKernel<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
      states, head_dimension, half, rotating_pairs, head_dimension, position, theta,
      scaling_factor, pairs);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess ? Status::Ok() : CudaFailure("launch proportional RoPE", error);
}

Status LaunchRotaryEmbeddingControlled(
    float* states, std::uint64_t heads, std::uint64_t head_dimension,
    std::uint64_t rotary_dimensions, const DecodeControl* control,
    double theta, cudaStream_t stream) {
  if (states == nullptr || control == nullptr || heads == 0U ||
      rotary_dimensions == 0U || rotary_dimensions > head_dimension ||
      rotary_dimensions % 2U != 0U || !std::isfinite(theta) || theta <= 0.0) {
    return Invalid("controlled RoPE arguments are invalid");
  }
  const std::uint64_t pairs = heads * (rotary_dimensions / 2U);
  const std::uint64_t blocks = Blocks(pairs);
  if (!ValidGrid(blocks)) return Invalid("controlled RoPE grid exceeds CUDA limits");
  ControlledRotaryKernel<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
      states, head_dimension, rotary_dimensions / 2U,
      rotary_dimensions / 2U, rotary_dimensions, control, theta, 1.0, pairs);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch controlled RoPE", error);
}

Status LaunchProportionalRotaryEmbeddingControlled(
    float* states, std::uint64_t heads, std::uint64_t head_dimension,
    double rotary_factor, const DecodeControl* control, double theta,
    double scaling_factor, cudaStream_t stream) {
  if (states == nullptr || control == nullptr || heads == 0U ||
      head_dimension == 0U || head_dimension % 2U != 0U ||
      !std::isfinite(rotary_factor) || rotary_factor <= 0.0 ||
      rotary_factor > 1.0 || !std::isfinite(theta) || theta <= 0.0 ||
      !std::isfinite(scaling_factor) || scaling_factor <= 0.0) {
    return Invalid("controlled proportional RoPE arguments are invalid");
  }
  const std::uint64_t half = head_dimension / 2U;
  const std::uint64_t rotating_pairs =
      static_cast<std::uint64_t>(rotary_factor * static_cast<double>(half));
  if (rotating_pairs == 0U || rotating_pairs > half ||
      heads > std::numeric_limits<std::uint64_t>::max() / rotating_pairs) {
    return Invalid("controlled proportional RoPE factor is invalid");
  }
  const std::uint64_t pairs = heads * rotating_pairs;
  const std::uint64_t blocks = Blocks(pairs);
  if (!ValidGrid(blocks)) {
    return Invalid("controlled proportional RoPE grid exceeds CUDA limits");
  }
  ControlledRotaryKernel<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
      states, head_dimension, half, rotating_pairs, head_dimension, control,
      theta, scaling_factor, pairs);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch controlled proportional RoPE", error);
}

Status LaunchAppendKv(const float* key, const float* value, float* key_cache,
                      float* value_cache, std::uint64_t slot, std::uint64_t kv_heads,
                      std::uint64_t head_dimension, cudaStream_t stream) {
  if (key == nullptr || value == nullptr || key_cache == nullptr || value_cache == nullptr) {
    return Invalid("KV append requires non-null pointers");
  }
  if (kv_heads == 0U || head_dimension == 0U ||
      kv_heads > std::numeric_limits<std::uint64_t>::max() / head_dimension) {
    return Invalid("KV append geometry is invalid");
  }
  const std::uint64_t elements = kv_heads * head_dimension;
  if (slot > std::numeric_limits<std::uint64_t>::max() / elements ||
      elements > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
    return Invalid("KV append offset exceeds addressable memory");
  }
  const std::size_t bytes = static_cast<std::size_t>(elements) * sizeof(float);
  cudaError_t error = cudaMemcpyAsync(key_cache + slot * elements, key, bytes,
                                     cudaMemcpyDeviceToDevice, stream);
  if (error != cudaSuccess) return CudaFailure("append K cache", error);
  error = cudaMemcpyAsync(value_cache + slot * elements, value, bytes,
                          cudaMemcpyDeviceToDevice, stream);
  return error == cudaSuccess ? Status::Ok() : CudaFailure("append V cache", error);
}

Status LaunchAppendKvFp8(
    const float* key, const float* value, std::uint8_t* key_cache,
    std::uint8_t* value_cache, const std::uint16_t* key_scale_bf16,
    const std::uint16_t* value_scale_bf16, std::uint64_t slot,
    std::uint64_t kv_heads, std::uint64_t head_dimension,
    cudaStream_t stream) {
  if (key == nullptr || value == nullptr || key_cache == nullptr ||
      value_cache == nullptr || key_scale_bf16 == nullptr ||
      value_scale_bf16 == nullptr) {
    return Invalid("FP8 KV append requires non-null pointers and scales");
  }
  if (kv_heads == 0U || head_dimension == 0U ||
      kv_heads > std::numeric_limits<std::uint64_t>::max() / head_dimension) {
    return Invalid("FP8 KV append geometry is invalid");
  }
  const std::uint64_t elements = kv_heads * head_dimension;
  if (slot > std::numeric_limits<std::uint64_t>::max() / elements) {
    return Invalid("FP8 KV append offset exceeds addressable memory");
  }
  const std::uint64_t blocks = Blocks(elements);
  if (!ValidGrid(blocks)) return Invalid("FP8 KV append grid exceeds CUDA limits");
  AppendKvFp8Kernel<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
      key, value, key_cache, value_cache, key_scale_bf16,
      value_scale_bf16, slot * elements, elements);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess ? Status::Ok()
                              : CudaFailure("launch FP8 KV append", error);
}

Status LaunchAppendKvControlled(
    const float* key, const float* value, float* key_cache,
    float* value_cache, const DecodeControl* control,
    std::uint64_t kv_heads, std::uint64_t head_dimension,
    std::uint64_t cache_capacity, bool sliding, cudaStream_t stream) {
  if (key == nullptr || value == nullptr || key_cache == nullptr ||
      value_cache == nullptr || control == nullptr || kv_heads == 0U ||
      head_dimension == 0U || cache_capacity == 0U ||
      kv_heads > std::numeric_limits<std::uint64_t>::max() / head_dimension) {
    return Invalid("controlled KV append arguments are invalid");
  }
  const std::uint64_t elements = kv_heads * head_dimension;
  const std::uint64_t blocks = Blocks(elements);
  if (!ValidGrid(blocks)) {
    return Invalid("controlled KV append grid exceeds CUDA limits");
  }
  ControlledAppendKvKernel<float, false>
      <<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
          key, value, key_cache, value_cache, nullptr, nullptr, control,
          elements, cache_capacity, sliding);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch controlled KV append", error);
}

Status LaunchAppendKvFp8Controlled(
    const float* key, const float* value, std::uint8_t* key_cache,
    std::uint8_t* value_cache, const std::uint16_t* key_scale_bf16,
    const std::uint16_t* value_scale_bf16, const DecodeControl* control,
    std::uint64_t kv_heads, std::uint64_t head_dimension,
    std::uint64_t cache_capacity, bool sliding, cudaStream_t stream) {
  if (key == nullptr || value == nullptr || key_cache == nullptr ||
      value_cache == nullptr || key_scale_bf16 == nullptr ||
      value_scale_bf16 == nullptr || control == nullptr || kv_heads == 0U ||
      head_dimension == 0U || cache_capacity == 0U ||
      kv_heads > std::numeric_limits<std::uint64_t>::max() / head_dimension) {
    return Invalid("controlled FP8 KV append arguments are invalid");
  }
  const std::uint64_t elements = kv_heads * head_dimension;
  const std::uint64_t blocks = Blocks(elements);
  if (!ValidGrid(blocks)) {
    return Invalid("controlled FP8 KV append grid exceeds CUDA limits");
  }
  ControlledAppendKvKernel<std::uint8_t, true>
      <<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
          key, value, key_cache, value_cache, key_scale_bf16,
          value_scale_bf16, control, elements, cache_capacity, sliding);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch controlled FP8 KV append", error);
}

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

Status LaunchRotaryEmbeddingBatch(
    float* states, std::uint64_t tokens, std::uint64_t heads,
    std::uint64_t head_dimension, std::uint64_t rotary_dimensions,
    std::uint64_t start_position, double theta, cudaStream_t stream) {
  if (states == nullptr || tokens == 0U || heads == 0U ||
      rotary_dimensions == 0U || rotary_dimensions > head_dimension ||
      rotary_dimensions % 2U != 0U || !std::isfinite(theta) || theta <= 0.0) {
    return Invalid("batched RoPE geometry is invalid");
  }
  const std::uint64_t pairs_per_token = heads * (rotary_dimensions / 2U);
  const std::uint64_t total_pairs = tokens * pairs_per_token;
  const std::uint64_t blocks = Blocks(total_pairs);
  if (!ValidGrid(blocks)) return Invalid("batched RoPE grid exceeds CUDA limits");
  RotaryBatchKernel<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
      states, heads, head_dimension, rotary_dimensions / 2U,
      rotary_dimensions / 2U, rotary_dimensions, start_position, theta, 1.0,
      pairs_per_token, total_pairs);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess ? Status::Ok()
                              : CudaFailure("launch batched RoPE", error);
}

Status LaunchProportionalRotaryEmbeddingBatch(
    float* states, std::uint64_t tokens, std::uint64_t heads,
    std::uint64_t head_dimension, double rotary_factor,
    std::uint64_t start_position, double theta, double scaling_factor,
    cudaStream_t stream) {
  if (states == nullptr || tokens == 0U || heads == 0U ||
      head_dimension == 0U || head_dimension % 2U != 0U ||
      !std::isfinite(rotary_factor) || rotary_factor <= 0.0 ||
      rotary_factor > 1.0 || !std::isfinite(theta) || theta <= 0.0 ||
      !std::isfinite(scaling_factor) || scaling_factor <= 0.0) {
    return Invalid("batched proportional RoPE geometry is invalid");
  }
  const std::uint64_t half = head_dimension / 2U;
  const std::uint64_t rotating_pairs =
      static_cast<std::uint64_t>(rotary_factor * static_cast<double>(half));
  const std::uint64_t pairs_per_token = heads * rotating_pairs;
  const std::uint64_t total_pairs = tokens * pairs_per_token;
  const std::uint64_t blocks = Blocks(total_pairs);
  if (rotating_pairs == 0U || !ValidGrid(blocks)) {
    return Invalid("batched proportional RoPE extent is invalid");
  }
  RotaryBatchKernel<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
      states, heads, head_dimension, half, rotating_pairs, head_dimension,
      start_position, theta, scaling_factor, pairs_per_token, total_pairs);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch batched proportional RoPE", error);
}

Status LaunchQuantizeKvFp8Batch(
    const float* key, const float* value, std::uint8_t* key_fp8,
    std::uint8_t* value_fp8, const std::uint16_t* key_scale_bf16,
    const std::uint16_t* value_scale_bf16, std::uint64_t tokens,
    std::uint64_t elements_per_token, cudaStream_t stream) {
  if (key == nullptr || value == nullptr || key_fp8 == nullptr ||
      value_fp8 == nullptr || key_scale_bf16 == nullptr ||
      value_scale_bf16 == nullptr || tokens == 0U || elements_per_token == 0U) {
    return Invalid("batched FP8 KV quantization arguments are invalid");
  }
  const std::uint64_t elements = tokens * elements_per_token;
  const std::uint64_t blocks = Blocks(elements);
  if (!ValidGrid(blocks)) return Invalid("batched FP8 KV grid exceeds CUDA limits");
  QuantizeKvBatchKernel<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
      key, value, key_fp8, value_fp8, key_scale_bf16, value_scale_bf16,
      elements);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch batched FP8 KV quantization", error);
}

Status LaunchAppendKvBatch(
    const float* key, const float* value, float* key_cache,
    float* value_cache, std::uint64_t start_position, std::uint64_t tokens,
    std::uint64_t elements_per_token, std::uint64_t cache_capacity,
    cudaStream_t stream) {
  if (key == nullptr || value == nullptr || key_cache == nullptr ||
      value_cache == nullptr || tokens == 0U || elements_per_token == 0U ||
      cache_capacity == 0U || tokens > cache_capacity) {
    return Invalid("batched KV append arguments are invalid");
  }
  const std::uint64_t elements = tokens * elements_per_token;
  const std::uint64_t blocks = Blocks(elements);
  if (!ValidGrid(blocks)) return Invalid("batched KV append grid exceeds CUDA limits");
  AppendKvBatchKernel<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
      key, value, key_cache, value_cache, start_position, elements_per_token,
      cache_capacity, elements);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess ? Status::Ok()
                              : CudaFailure("launch batched KV append", error);
}

Status LaunchAppendKvFp8Batch(
    const std::uint8_t* key, const std::uint8_t* value,
    std::uint8_t* key_cache, std::uint8_t* value_cache,
    std::uint64_t start_position, std::uint64_t tokens,
    std::uint64_t elements_per_token, std::uint64_t cache_capacity,
    cudaStream_t stream) {
  if (key == nullptr || value == nullptr || key_cache == nullptr ||
      value_cache == nullptr || tokens == 0U || elements_per_token == 0U ||
      cache_capacity == 0U || tokens > cache_capacity) {
    return Invalid("batched FP8 KV append arguments are invalid");
  }
  const std::uint64_t elements = tokens * elements_per_token;
  const std::uint64_t blocks = Blocks(elements);
  if (!ValidGrid(blocks)) return Invalid("batched FP8 KV append grid exceeds CUDA limits");
  AppendKvBatchKernel<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
      key, value, key_cache, value_cache, start_position, elements_per_token,
      cache_capacity, elements);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch batched FP8 KV append", error);
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

Status LaunchScale(float* values, const std::uint16_t* scalar_bf16,
                   std::uint64_t elements, cudaStream_t stream) {
  if (values == nullptr || scalar_bf16 == nullptr) return Invalid("scale requires non-null pointers");
  if (elements == 0U || !ValidGrid(Blocks(elements))) return Invalid("scale extent is invalid");
  ScaleKernel<<<static_cast<unsigned>(Blocks(elements)), kThreads, 0, stream>>>(
      values, scalar_bf16, elements);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess ? Status::Ok() : CudaFailure("launch layer scale", error);
}

}  // namespace gem16gb::internal
