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

template <typename T>
__global__ void AppendKvBatchControlledKernel(
    const T* key, const T* value, T* key_cache, T* value_cache,
    const DecodeControl* row_controls, std::uint64_t elements_per_token,
    std::uint64_t cache_capacity, std::uint64_t total_elements) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total_elements) return;
  const std::uint64_t token = index / elements_per_token;
  const std::uint64_t element = index % elements_per_token;
  const std::uint64_t slot =
      row_controls[token].position % cache_capacity;
  const std::uint64_t destination = slot * elements_per_token + element;
  key_cache[destination] = key[index];
  value_cache[destination] = value[index];
}

__global__ void BackupAppendKvFp8BatchControlledKernel(
    const std::uint8_t* key, const std::uint8_t* value,
    std::uint8_t* key_cache, std::uint8_t* value_cache,
    std::uint8_t* backup_key, std::uint8_t* backup_value,
    const DecodeControl* row_controls, std::uint64_t elements_per_token,
    std::uint64_t cache_capacity, std::uint64_t total_elements) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total_elements) return;
  const std::uint64_t token = index / elements_per_token;
  const std::uint64_t element = index % elements_per_token;
  const std::uint64_t slot =
      row_controls[token].position % cache_capacity;
  const std::uint64_t destination = slot * elements_per_token + element;
  const std::uint8_t original_key = key_cache[destination];
  const std::uint8_t original_value = value_cache[destination];
  backup_key[index] = original_key;
  backup_value[index] = original_value;
  key_cache[destination] = key[index];
  value_cache[destination] = value[index];
}

std::uint64_t Blocks(std::uint64_t elements) {
  return (elements + kThreads - 1U) / kThreads;
}

bool ValidGrid(std::uint64_t blocks) {
  return blocks <= static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max());
}


}  // namespace

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

Status LaunchAppendKvFp8BatchControlled(
    const std::uint8_t* key, const std::uint8_t* value,
    std::uint8_t* key_cache, std::uint8_t* value_cache,
    const DecodeControl* row_controls, std::uint64_t tokens,
    std::uint64_t elements_per_token, std::uint64_t cache_capacity,
    cudaStream_t stream) {
  if (key == nullptr || value == nullptr || key_cache == nullptr ||
      value_cache == nullptr || row_controls == nullptr || tokens == 0U ||
      elements_per_token == 0U || cache_capacity == 0U ||
      tokens > cache_capacity) {
    return Invalid("controlled batched FP8 KV append arguments are invalid");
  }
  const std::uint64_t elements = tokens * elements_per_token;
  const std::uint64_t blocks = Blocks(elements);
  if (!ValidGrid(blocks)) {
    return Invalid("controlled batched FP8 KV append grid exceeds CUDA limits");
  }
  AppendKvBatchControlledKernel<<<static_cast<unsigned>(blocks), kThreads, 0,
                                  stream>>>(
      key, value, key_cache, value_cache, row_controls, elements_per_token,
      cache_capacity, elements);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch controlled batched FP8 KV append", error);
}

Status LaunchBackupAppendKvFp8BatchControlled(
    const std::uint8_t* key, const std::uint8_t* value,
    std::uint8_t* key_cache, std::uint8_t* value_cache,
    std::uint8_t* backup_key, std::uint8_t* backup_value,
    const DecodeControl* row_controls, std::uint64_t tokens,
    std::uint64_t elements_per_token, std::uint64_t cache_capacity,
    cudaStream_t stream) {
  if (key == nullptr || value == nullptr || key_cache == nullptr ||
      value_cache == nullptr || backup_key == nullptr ||
      backup_value == nullptr || row_controls == nullptr || tokens == 0U ||
      elements_per_token == 0U || cache_capacity == 0U ||
      tokens > cache_capacity) {
    return Invalid("controlled backup/append FP8 KV arguments are invalid");
  }
  const std::uint64_t elements = tokens * elements_per_token;
  const std::uint64_t blocks = Blocks(elements);
  if (!ValidGrid(blocks)) {
    return Invalid("controlled backup/append FP8 KV grid exceeds CUDA limits");
  }
  BackupAppendKvFp8BatchControlledKernel
      <<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
          key, value, key_cache, value_cache, backup_key, backup_value,
          row_controls, elements_per_token, cache_capacity, elements);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch controlled backup/append FP8 KV", error);
}


}  // namespace gem16::internal
