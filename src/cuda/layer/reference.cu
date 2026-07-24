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
    output[base + index] = input[base + index] * inverse_rms * scale;
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

__global__ void ScaleKernel(float* values, const std::uint16_t* scalar,
                            std::uint64_t elements) {
  const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    values[index] *= static_cast<float>(__ushort_as_bfloat16(scalar[0]));
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
  RmsNormKernel<<<static_cast<unsigned>(vectors), kThreads, 0, stream>>>(
      input, weight_bf16, output, width, epsilon);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess ? Status::Ok() : CudaFailure("launch RMSNorm", error);
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
