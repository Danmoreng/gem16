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

__global__ void RotaryTableBatchKernel(
    float* cosine, float* sine, std::uint64_t rotating_pairs,
    std::uint64_t frequency_dimension, std::uint64_t start_position,
    double theta, double scaling_factor, std::uint64_t total_pairs) {
  const std::uint64_t pair =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair >= total_pairs) return;
  const std::uint64_t token = pair / rotating_pairs;
  const std::uint64_t index = pair % rotating_pairs;
  const double exponent = 2.0 * static_cast<double>(index) /
                          static_cast<double>(frequency_dimension);
  const double angle = static_cast<double>(start_position + token) /
                       (pow(theta, exponent) * scaling_factor);
  cosine[pair] = static_cast<float>(cos(angle));
  sine[pair] = static_cast<float>(sin(angle));
}

__global__ void RotaryTableControlledKernel(
    float* cosine, float* sine, std::uint64_t rotating_pairs,
    std::uint64_t frequency_dimension, const DecodeControl* control,
    double theta, double scaling_factor) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= rotating_pairs) return;
  const double exponent = 2.0 * static_cast<double>(index) /
                          static_cast<double>(frequency_dimension);
  const double angle = static_cast<double>(control->position) /
                       (pow(theta, exponent) * scaling_factor);
  cosine[index] = static_cast<float>(cos(angle));
  sine[index] = static_cast<float>(sin(angle));
}

__global__ void RotaryTableBatchControlledKernel(
    float* cosine, float* sine, std::uint64_t rotating_pairs,
    std::uint64_t frequency_dimension, const DecodeControl* controls,
    double theta, double scaling_factor, std::uint64_t total_pairs) {
  const std::uint64_t pair =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair >= total_pairs) return;
  const std::uint64_t token = pair / rotating_pairs;
  const std::uint64_t index = pair % rotating_pairs;
  const double exponent = 2.0 * static_cast<double>(index) /
                          static_cast<double>(frequency_dimension);
  const double angle = static_cast<double>(controls[token].position) /
                       (pow(theta, exponent) * scaling_factor);
  cosine[pair] = static_cast<float>(cos(angle));
  sine[pair] = static_cast<float>(sin(angle));
}

template <typename Input>
__device__ float LoadProjectionBoundary(const Input* input,
                                        std::uint64_t index) {
  if constexpr (std::is_same_v<Input, std::uint16_t>) {
    return static_cast<float>(__ushort_as_bfloat16(input[index]));
  } else {
    return static_cast<float>(__float2bfloat16_rn(input[index]));
  }
}

template <typename Input>
__global__ void ProjectionRmsNormRotaryBf16BatchKernel(
    const Input* query, const std::uint16_t* query_norm,
    float* normalized_query, const Input* key,
    const std::uint16_t* key_norm, float* normalized_key,
    const float* rotary_cosine, const float* rotary_sine,
    std::uint64_t query_heads, std::uint64_t kv_heads,
    std::uint64_t head_dimension, std::uint64_t rotating_pairs,
    float epsilon) {
  const std::uint64_t heads_per_token = query_heads + kv_heads;
  const std::uint64_t token = blockIdx.x / heads_per_token;
  const std::uint64_t combined_head = blockIdx.x % heads_per_token;
  const bool is_query = combined_head < query_heads;
  const std::uint64_t heads = is_query ? query_heads : kv_heads;
  const std::uint64_t head =
      is_query ? combined_head : combined_head - query_heads;
  const Input* input = is_query ? query : key;
  const std::uint16_t* weight = is_query ? query_norm : key_norm;
  float* output = is_query ? normalized_query : normalized_key;
  const std::uint64_t base =
      (token * heads + head) * head_dimension;

  float squared_sum = 0.0F;
  for (std::uint64_t index = threadIdx.x; index < head_dimension;
       index += blockDim.x) {
    const float value = LoadProjectionBoundary(input, base + index);
    squared_sum = fmaf(value, value, squared_sum);
  }
  const float inverse_rms =
      rsqrtf(BlockSum(squared_sum) / static_cast<float>(head_dimension) +
             epsilon);

  const std::uint64_t half = head_dimension / 2U;
  for (std::uint64_t index = threadIdx.x; index < head_dimension;
       index += blockDim.x) {
    const bool rotated = index < rotating_pairs ||
                         (index >= half && index < half + rotating_pairs);
    if (rotated) continue;
    const float rounded_input = LoadProjectionBoundary(input, base + index);
    const float scale =
        static_cast<float>(__ushort_as_bfloat16(weight[index]));
    output[base + index] = static_cast<float>(__float2bfloat16_rn(
        rounded_input * inverse_rms * scale));
  }

  if (threadIdx.x >= rotating_pairs) return;
  const std::uint64_t index = threadIdx.x;
  const std::uint64_t first = base + index;
  const std::uint64_t second = first + half;
  const float first_input = LoadProjectionBoundary(input, first);
  const float second_input = LoadProjectionBoundary(input, second);
  const float first_scale =
      static_cast<float>(__ushort_as_bfloat16(weight[index]));
  const float second_scale =
      static_cast<float>(__ushort_as_bfloat16(weight[index + half]));
  const float first_value = static_cast<float>(__float2bfloat16_rn(
      first_input * inverse_rms * first_scale));
  const float second_value = static_cast<float>(__float2bfloat16_rn(
      second_input * inverse_rms * second_scale));
  const std::uint64_t rotary_index = token * rotating_pairs + index;
  const float cosine = rotary_cosine[rotary_index];
  const float sine = rotary_sine[rotary_index];
  output[first] = static_cast<float>(__float2bfloat16_rn(
      first_value * cosine - second_value * sine));
  output[second] = static_cast<float>(__float2bfloat16_rn(
      second_value * cosine + first_value * sine));
}

__global__ void ProjectionRmsNormRotaryBf16BatchControlledKernel(
    const float* query, const std::uint16_t* query_norm,
    float* normalized_query, const float* key,
    const std::uint16_t* key_norm, float* normalized_key,
    const float* rotary_cosine, const float* rotary_sine,
    const DecodeControl* row_controls, std::uint64_t query_heads,
    std::uint64_t kv_heads, std::uint64_t head_dimension,
    std::uint64_t rotating_pairs, float epsilon) {
  const std::uint64_t heads_per_token = query_heads + kv_heads;
  const std::uint64_t token = blockIdx.x / heads_per_token;
  const std::uint64_t combined_head = blockIdx.x % heads_per_token;
  const bool is_query = combined_head < query_heads;
  const std::uint64_t heads = is_query ? query_heads : kv_heads;
  const std::uint64_t head =
      is_query ? combined_head : combined_head - query_heads;
  const float* input = is_query ? query : key;
  const std::uint16_t* weight = is_query ? query_norm : key_norm;
  float* output = is_query ? normalized_query : normalized_key;
  const std::uint64_t base =
      (token * heads + head) * head_dimension;

  float squared_sum = 0.0F;
  for (std::uint64_t index = threadIdx.x; index < head_dimension;
       index += blockDim.x) {
    const float value = static_cast<float>(
        __float2bfloat16_rn(input[base + index]));
    squared_sum = fmaf(value, value, squared_sum);
  }
  const float inverse_rms =
      rsqrtf(BlockSum(squared_sum) / static_cast<float>(head_dimension) +
             epsilon);

  const std::uint64_t half = head_dimension / 2U;
  for (std::uint64_t index = threadIdx.x; index < head_dimension;
       index += blockDim.x) {
    const bool rotated = index < rotating_pairs ||
                         (index >= half && index < half + rotating_pairs);
    if (rotated) continue;
    const float rounded_input = static_cast<float>(
        __float2bfloat16_rn(input[base + index]));
    const float scale =
        static_cast<float>(__ushort_as_bfloat16(weight[index]));
    output[base + index] = static_cast<float>(__float2bfloat16_rn(
        rounded_input * inverse_rms * scale));
  }

  if (threadIdx.x >= rotating_pairs) return;
  const std::uint64_t index = threadIdx.x;
  const std::uint64_t first = base + index;
  const std::uint64_t second = first + half;
  const float first_input = static_cast<float>(
      __float2bfloat16_rn(input[first]));
  const float second_input = static_cast<float>(
      __float2bfloat16_rn(input[second]));
  const float first_scale =
      static_cast<float>(__ushort_as_bfloat16(weight[index]));
  const float second_scale =
      static_cast<float>(__ushort_as_bfloat16(weight[index + half]));
  const float first_value = static_cast<float>(__float2bfloat16_rn(
      first_input * inverse_rms * first_scale));
  const float second_value = static_cast<float>(__float2bfloat16_rn(
      second_input * inverse_rms * second_scale));
  const std::uint64_t rotary_index =
      row_controls[token].position * rotating_pairs + index;
  const float cosine = rotary_cosine[rotary_index];
  const float sine = rotary_sine[rotary_index];
  output[first] = static_cast<float>(__float2bfloat16_rn(
      first_value * cosine - second_value * sine));
  output[second] = static_cast<float>(__float2bfloat16_rn(
      second_value * cosine + first_value * sine));
}

__global__ void Gemma4Moe26BMtpKvEpilogueKernel(
    const float* query, const std::uint16_t* query_norm,
    float* normalized_query, const float* key,
    const std::uint16_t* key_norm, const float* value,
    const float* rotary_cosine, const float* rotary_sine,
    std::uint8_t* staged_key, std::uint8_t* staged_value,
    const std::uint16_t* key_scale, const std::uint16_t* value_scale,
    std::uint64_t kv_heads, std::uint64_t head_dimension,
    std::uint64_t rotating_pairs, float epsilon) {
  constexpr std::uint64_t kQueryHeads = 16U;
  const std::uint64_t heads_per_token = kQueryHeads + 2U * kv_heads;
  const std::uint64_t token = blockIdx.x / heads_per_token;
  const std::uint64_t combined_head = blockIdx.x % heads_per_token;
  if (combined_head >= kQueryHeads + kv_heads) {
    const std::uint64_t head = combined_head - kQueryHeads - kv_heads;
    const std::uint64_t base = (token * kv_heads + head) * head_dimension;
    float squared_sum = 0.0F;
    for (std::uint64_t index = threadIdx.x; index < head_dimension;
         index += blockDim.x) {
      squared_sum = fmaf(value[base + index], value[base + index],
                         squared_sum);
    }
    const float inverse_rms =
        rsqrtf(BlockSum(squared_sum) / static_cast<float>(head_dimension) +
               epsilon);
    const float scale =
        static_cast<float>(__ushort_as_bfloat16(value_scale[0]));
    for (std::uint64_t index = threadIdx.x; index < head_dimension;
         index += blockDim.x) {
      const float rounded = static_cast<float>(
          __float2bfloat16_rn(value[base + index] * inverse_rms));
      staged_value[base + index] = __nv_fp8_e4m3(rounded / scale).__x;
    }
    return;
  }

  const bool is_query = combined_head < kQueryHeads;
  const std::uint64_t heads = is_query ? kQueryHeads : kv_heads;
  const std::uint64_t head =
      is_query ? combined_head : combined_head - kQueryHeads;
  const float* input = is_query ? query : key;
  const std::uint16_t* weight = is_query ? query_norm : key_norm;
  const std::uint64_t base = (token * heads + head) * head_dimension;
  float squared_sum = 0.0F;
  for (std::uint64_t index = threadIdx.x; index < head_dimension;
       index += blockDim.x) {
    const float rounded = LoadProjectionBoundary(input, base + index);
    squared_sum = fmaf(rounded, rounded, squared_sum);
  }
  const float inverse_rms =
      rsqrtf(BlockSum(squared_sum) / static_cast<float>(head_dimension) +
             epsilon);
  const float cache_scale = is_query
                                ? 1.0F
                                : static_cast<float>(
                                      __ushort_as_bfloat16(key_scale[0]));
  auto store = [&](std::uint64_t index, float normalized) {
    if (is_query) {
      normalized_query[index] = normalized;
    } else {
      staged_key[index] = __nv_fp8_e4m3(normalized / cache_scale).__x;
    }
  };
  const std::uint64_t half = head_dimension / 2U;
  for (std::uint64_t index = threadIdx.x; index < head_dimension;
       index += blockDim.x) {
    const bool rotated = index < rotating_pairs ||
                         (index >= half && index < half + rotating_pairs);
    if (rotated) continue;
    const float rounded = LoadProjectionBoundary(input, base + index);
    const float scale =
        static_cast<float>(__ushort_as_bfloat16(weight[index]));
    store(base + index, static_cast<float>(
                            __float2bfloat16_rn(rounded * inverse_rms * scale)));
  }
  if (threadIdx.x >= rotating_pairs) return;
  const std::uint64_t index = threadIdx.x;
  const std::uint64_t first = base + index;
  const std::uint64_t second = first + half;
  const float first_value = static_cast<float>(__float2bfloat16_rn(
      LoadProjectionBoundary(input, first) * inverse_rms *
      static_cast<float>(__ushort_as_bfloat16(weight[index]))));
  const float second_value = static_cast<float>(__float2bfloat16_rn(
      LoadProjectionBoundary(input, second) * inverse_rms *
      static_cast<float>(__ushort_as_bfloat16(weight[index + half]))));
  const std::uint64_t rotary_index = token * rotating_pairs + index;
  const float cosine = rotary_cosine[rotary_index];
  const float sine = rotary_sine[rotary_index];
  store(first, static_cast<float>(__float2bfloat16_rn(
                   first_value * cosine - second_value * sine)));
  store(second, static_cast<float>(__float2bfloat16_rn(
                    second_value * cosine + first_value * sine)));
}

template <bool kDirectKv>
__device__ void ProjectionRmsNormRotaryBf16ControlledBody(
    const float* query, const std::uint16_t* query_norm,
    float* normalized_query, const float* key,
    const std::uint16_t* key_norm, float* normalized_key,
    const float* rotary_cosine, const float* rotary_sine,
    const DecodeControl* control, std::uint64_t query_heads,
    std::uint64_t kv_heads, std::uint64_t head_dimension,
    std::uint64_t rotating_pairs, float epsilon,
    bool table_contains_all_positions,
    const float* value = nullptr, std::uint8_t* key_cache = nullptr,
    std::uint8_t* value_cache = nullptr,
    const std::uint16_t* key_scale = nullptr,
    const std::uint16_t* value_scale = nullptr,
    std::uint64_t cache_capacity = 0U) {
  const std::uint64_t combined_head = blockIdx.x;
  if constexpr (kDirectKv) {
    if (combined_head >= query_heads + kv_heads) {
      const std::uint64_t base =
          (combined_head - query_heads - kv_heads) * head_dimension;
      float squared_sum = 0.0F;
      for (std::uint64_t i = threadIdx.x; i < head_dimension; i += blockDim.x) {
        // V deliberately uses raw FP32 projection values, unlike Q/K.
        squared_sum = fmaf(value[base + i], value[base + i], squared_sum);
      }
      const float inverse_rms =
          rsqrtf(BlockSum(squared_sum) / static_cast<float>(head_dimension) + epsilon);
      const float scale = static_cast<float>(__ushort_as_bfloat16(value_scale[0]));
      const std::uint64_t offset =
          (control->position % cache_capacity) * kv_heads * head_dimension;
      for (std::uint64_t i = threadIdx.x; i < head_dimension; i += blockDim.x) {
        const float rounded = static_cast<float>(
            __float2bfloat16_rn(value[base + i] * inverse_rms));
        value_cache[offset + base + i] = __nv_fp8_e4m3(rounded / scale).__x;
      }
      return;
    }
  }
  const bool is_query = combined_head < query_heads;
  const std::uint64_t head =
      is_query ? combined_head : combined_head - query_heads;
  const float* input = is_query ? query : key;
  const std::uint16_t* weight = is_query ? query_norm : key_norm;
  float* output = is_query ? normalized_query : normalized_key;
  const std::uint64_t base = head * head_dimension;

  float squared_sum = 0.0F;
  for (std::uint64_t index = threadIdx.x; index < head_dimension;
       index += blockDim.x) {
    const float value = static_cast<float>(
        __float2bfloat16_rn(input[base + index]));
    squared_sum = fmaf(value, value, squared_sum);
  }
  const float inverse_rms =
      rsqrtf(BlockSum(squared_sum) / static_cast<float>(head_dimension) +
             epsilon);

  const std::uint64_t half = head_dimension / 2U;
  for (std::uint64_t index = threadIdx.x; index < head_dimension;
       index += blockDim.x) {
    const bool rotated = index < rotating_pairs ||
                         (index >= half && index < half + rotating_pairs);
    if (rotated) continue;
    const float rounded_input = static_cast<float>(
        __float2bfloat16_rn(input[base + index]));
    const float scale =
        static_cast<float>(__ushort_as_bfloat16(weight[index]));
    const float normalized = static_cast<float>(__float2bfloat16_rn(
        rounded_input * inverse_rms * scale));
    if constexpr (kDirectKv) {
      if (!is_query) {
        const std::uint64_t offset =
            (control->position % cache_capacity) * kv_heads * head_dimension;
        const float cache_scale = static_cast<float>(__ushort_as_bfloat16(key_scale[0]));
        key_cache[offset + base + index] = __nv_fp8_e4m3(normalized / cache_scale).__x;
      } else {
        output[base + index] = normalized;
      }
    } else {
      output[base + index] = normalized;
    }
  }

  if (threadIdx.x >= rotating_pairs) return;
  const std::uint64_t index = threadIdx.x;
  const std::uint64_t first = base + index;
  const std::uint64_t second = first + half;
  const float first_input = static_cast<float>(
      __float2bfloat16_rn(input[first]));
  const float second_input = static_cast<float>(
      __float2bfloat16_rn(input[second]));
  const float first_scale =
      static_cast<float>(__ushort_as_bfloat16(weight[index]));
  const float second_scale =
      static_cast<float>(__ushort_as_bfloat16(weight[index + half]));
  const float first_value = static_cast<float>(__float2bfloat16_rn(
      first_input * inverse_rms * first_scale));
  const float second_value = static_cast<float>(__float2bfloat16_rn(
      second_input * inverse_rms * second_scale));
  const std::uint64_t rotary_index =
      table_contains_all_positions
          ? control->position * rotating_pairs + index
          : index;
  const float cosine = rotary_cosine[rotary_index];
  const float sine = rotary_sine[rotary_index];
  const float first_rotated = static_cast<float>(__float2bfloat16_rn(
      first_value * cosine - second_value * sine));
  const float second_rotated = static_cast<float>(__float2bfloat16_rn(
      second_value * cosine + first_value * sine));
  if constexpr (kDirectKv) {
    if (!is_query) {
      const std::uint64_t offset =
          (control->position % cache_capacity) * kv_heads * head_dimension;
      const float cache_scale = static_cast<float>(__ushort_as_bfloat16(key_scale[0]));
      key_cache[offset + first] = __nv_fp8_e4m3(first_rotated / cache_scale).__x;
      key_cache[offset + second] = __nv_fp8_e4m3(second_rotated / cache_scale).__x;
    } else {
      output[first] = first_rotated;
      output[second] = second_rotated;
    }
  } else {
    output[first] = first_rotated;
    output[second] = second_rotated;
  }
}

__global__ void ProjectionRmsNormRotaryBf16ControlledKernel(
    const float* query, const std::uint16_t* query_norm,
    float* normalized_query, const float* key,
    const std::uint16_t* key_norm, float* normalized_key,
    const float* rotary_cosine, const float* rotary_sine,
    const DecodeControl* control, std::uint64_t query_heads,
    std::uint64_t kv_heads, std::uint64_t head_dimension,
    std::uint64_t rotating_pairs, float epsilon,
    bool table_contains_all_positions) {
  ProjectionRmsNormRotaryBf16ControlledBody<false>(
      query, query_norm, normalized_query, key, key_norm, normalized_key,
      rotary_cosine, rotary_sine, control, query_heads, kv_heads,
      head_dimension, rotating_pairs, epsilon, table_contains_all_positions);
}

__global__ void Gemma4Moe26BDecodeKvEpilogueKernel(
    const float* query, const std::uint16_t* query_norm,
    float* normalized_query, const float* key,
    const std::uint16_t* key_norm, const float* value,
    const float* rotary_cosine, const float* rotary_sine,
    const DecodeControl* control, std::uint64_t kv_heads,
    std::uint64_t head_dimension, std::uint64_t rotating_pairs, float epsilon,
    std::uint8_t* key_cache, std::uint8_t* value_cache,
    const std::uint16_t* key_scale, const std::uint16_t* value_scale,
    std::uint64_t cache_capacity) {
  ProjectionRmsNormRotaryBf16ControlledBody<true>(
      query, query_norm, normalized_query, key, key_norm, nullptr,
      rotary_cosine, rotary_sine, control, 16U, kv_heads, head_dimension,
      rotating_pairs, epsilon, false, value, key_cache, value_cache, key_scale,
      value_scale, cache_capacity);
}

std::uint64_t Blocks(std::uint64_t elements) {
  return (elements + kThreads - 1U) / kThreads;
}

bool ValidGrid(std::uint64_t blocks) {
  return blocks <= static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max());
}


}  // namespace

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

Status LaunchProjectionRmsNormRotaryBf16Batch(
    const float* query, const std::uint16_t* query_norm_bf16,
    float* normalized_query, const float* key,
    const std::uint16_t* key_norm_bf16, float* normalized_key,
    const float* rotary_cosine, const float* rotary_sine,
    std::uint64_t tokens, std::uint64_t query_heads,
    std::uint64_t kv_heads, std::uint64_t head_dimension,
    double rotary_factor, float epsilon, cudaStream_t stream) {
  if (query == nullptr || query_norm_bf16 == nullptr ||
      normalized_query == nullptr || key == nullptr ||
      key_norm_bf16 == nullptr || normalized_key == nullptr || tokens == 0U ||
      rotary_cosine == nullptr || rotary_sine == nullptr ||
      query_heads == 0U || kv_heads == 0U || head_dimension == 0U ||
      head_dimension > kThreads * 2U || head_dimension % 2U != 0U ||
      !std::isfinite(rotary_factor) || rotary_factor <= 0.0 ||
      rotary_factor > 1.0 ||
      !std::isfinite(epsilon) || epsilon <= 0.0F) {
    return Invalid("fused projection RMSNorm/RoPE geometry is invalid");
  }
  const std::uint64_t rotating_pairs = static_cast<std::uint64_t>(
      rotary_factor * static_cast<double>(head_dimension / 2U));
  const std::uint64_t blocks = tokens * (query_heads + kv_heads);
  if (rotating_pairs == 0U || rotating_pairs > kThreads ||
      !ValidGrid(blocks)) {
    return Invalid("fused projection RMSNorm/RoPE extent is invalid");
  }
  ProjectionRmsNormRotaryBf16BatchKernel<float>
      <<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
          query, query_norm_bf16, normalized_query, key, key_norm_bf16,
          normalized_key, rotary_cosine, rotary_sine, query_heads, kv_heads,
          head_dimension, rotating_pairs, epsilon);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch fused projection RMSNorm/RoPE", error);
}

Status LaunchProjectionRmsNormRotaryBf16BatchInput(
    const std::uint16_t* query_bf16,
    const std::uint16_t* query_norm_bf16, float* normalized_query,
    const std::uint16_t* key_bf16, const std::uint16_t* key_norm_bf16,
    float* normalized_key, const float* rotary_cosine,
    const float* rotary_sine, std::uint64_t tokens,
    std::uint64_t query_heads, std::uint64_t kv_heads,
    std::uint64_t head_dimension, double rotary_factor, float epsilon,
    cudaStream_t stream) {
  if (query_bf16 == nullptr || query_norm_bf16 == nullptr ||
      normalized_query == nullptr || key_bf16 == nullptr ||
      key_norm_bf16 == nullptr || normalized_key == nullptr || tokens == 0U ||
      rotary_cosine == nullptr || rotary_sine == nullptr ||
      query_heads == 0U || kv_heads == 0U || head_dimension == 0U ||
      head_dimension > kThreads * 2U || head_dimension % 2U != 0U ||
      !std::isfinite(rotary_factor) || rotary_factor <= 0.0 ||
      rotary_factor > 1.0 || !std::isfinite(epsilon) || epsilon <= 0.0F) {
    return Invalid("physical-BF16 projection RMSNorm/RoPE geometry is invalid");
  }
  const std::uint64_t rotating_pairs = static_cast<std::uint64_t>(
      rotary_factor * static_cast<double>(head_dimension / 2U));
  const std::uint64_t blocks = tokens * (query_heads + kv_heads);
  if (rotating_pairs == 0U || rotating_pairs > kThreads ||
      !ValidGrid(blocks)) {
    return Invalid("physical-BF16 projection RMSNorm/RoPE extent is invalid");
  }
  ProjectionRmsNormRotaryBf16BatchKernel<std::uint16_t>
      <<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
          query_bf16, query_norm_bf16, normalized_query, key_bf16,
          key_norm_bf16, normalized_key, rotary_cosine, rotary_sine,
          query_heads, kv_heads, head_dimension, rotating_pairs, epsilon);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure(
                   "launch physical-BF16 projection RMSNorm/RoPE", error);
}

Status LaunchProjectionRmsNormRotaryBf16BatchControlled(
    const float* query, const std::uint16_t* query_norm_bf16,
    float* normalized_query, const float* key,
    const std::uint16_t* key_norm_bf16, float* normalized_key,
    const float* rotary_cosine, const float* rotary_sine,
    const DecodeControl* row_controls, std::uint64_t tokens,
    std::uint64_t query_heads, std::uint64_t kv_heads,
    std::uint64_t head_dimension, double rotary_factor, float epsilon,
    cudaStream_t stream) {
  if (query == nullptr || query_norm_bf16 == nullptr ||
      normalized_query == nullptr || key == nullptr ||
      key_norm_bf16 == nullptr || normalized_key == nullptr ||
      rotary_cosine == nullptr || rotary_sine == nullptr ||
      row_controls == nullptr || tokens == 0U || query_heads == 0U ||
      kv_heads == 0U || head_dimension == 0U ||
      head_dimension > kThreads * 2U || head_dimension % 2U != 0U ||
      !std::isfinite(rotary_factor) || rotary_factor <= 0.0 ||
      rotary_factor > 1.0 || !std::isfinite(epsilon) || epsilon <= 0.0F) {
    return Invalid("controlled fused projection RMSNorm/RoPE geometry is invalid");
  }
  const std::uint64_t rotating_pairs = static_cast<std::uint64_t>(
      rotary_factor * static_cast<double>(head_dimension / 2U));
  const std::uint64_t blocks = tokens * (query_heads + kv_heads);
  if (rotating_pairs == 0U || rotating_pairs > kThreads ||
      !ValidGrid(blocks)) {
    return Invalid("controlled fused projection RMSNorm/RoPE extent is invalid");
  }
  ProjectionRmsNormRotaryBf16BatchControlledKernel
      <<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
          query, query_norm_bf16, normalized_query, key, key_norm_bf16,
          normalized_key, rotary_cosine, rotary_sine, row_controls,
          query_heads, kv_heads, head_dimension, rotating_pairs, epsilon);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure(
                   "launch controlled fused projection RMSNorm/RoPE", error);
}

Status LaunchProjectionRmsNormRotaryBf16Controlled(
    const float* query, const std::uint16_t* query_norm_bf16,
    float* normalized_query, const float* key,
    const std::uint16_t* key_norm_bf16, float* normalized_key,
    const float* rotary_cosine, const float* rotary_sine,
    const DecodeControl* control, std::uint64_t query_heads,
    std::uint64_t kv_heads, std::uint64_t head_dimension,
    double rotary_factor, float epsilon, cudaStream_t stream) {
  if (query == nullptr || query_norm_bf16 == nullptr ||
      normalized_query == nullptr || key == nullptr ||
      key_norm_bf16 == nullptr || normalized_key == nullptr ||
      rotary_cosine == nullptr || rotary_sine == nullptr || control == nullptr ||
      query_heads == 0U || kv_heads == 0U || head_dimension == 0U ||
      head_dimension > kThreads * 2U || head_dimension % 2U != 0U ||
      !std::isfinite(rotary_factor) || rotary_factor <= 0.0 ||
      rotary_factor > 1.0 || !std::isfinite(epsilon) || epsilon <= 0.0F) {
    return Invalid("controlled fused projection RMSNorm/RoPE geometry is invalid");
  }
  const std::uint64_t rotating_pairs = static_cast<std::uint64_t>(
      rotary_factor * static_cast<double>(head_dimension / 2U));
  const std::uint64_t blocks = query_heads + kv_heads;
  if (rotating_pairs == 0U || rotating_pairs > kThreads ||
      !ValidGrid(blocks)) {
    return Invalid("controlled fused projection RMSNorm/RoPE extent is invalid");
  }
  ProjectionRmsNormRotaryBf16ControlledKernel
      <<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
          query, query_norm_bf16, normalized_query, key, key_norm_bf16,
          normalized_key, rotary_cosine, rotary_sine, control, query_heads,
          kv_heads, head_dimension, rotating_pairs, epsilon, true);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch controlled fused projection RMSNorm/RoPE",
                           error);
}

Status LaunchProjectionRmsNormRotaryBf16CurrentTableControlled(
    const float* query, const std::uint16_t* query_norm_bf16,
    float* normalized_query, const float* key,
    const std::uint16_t* key_norm_bf16, float* normalized_key,
    const float* rotary_cosine, const float* rotary_sine,
    const DecodeControl* control, std::uint64_t query_heads,
    std::uint64_t kv_heads, std::uint64_t head_dimension,
    double rotary_factor, float epsilon, cudaStream_t stream) {
  if (query == nullptr || query_norm_bf16 == nullptr ||
      normalized_query == nullptr || key == nullptr ||
      key_norm_bf16 == nullptr || normalized_key == nullptr ||
      rotary_cosine == nullptr || rotary_sine == nullptr || control == nullptr ||
      query_heads == 0U || kv_heads == 0U || head_dimension == 0U ||
      head_dimension > kThreads * 2U || head_dimension % 2U != 0U ||
      !std::isfinite(rotary_factor) || rotary_factor <= 0.0 ||
      rotary_factor > 1.0 || !std::isfinite(epsilon) || epsilon <= 0.0F) {
    return Invalid("current-table controlled fused projection RMSNorm/RoPE "
                   "geometry is invalid");
  }
  const std::uint64_t rotating_pairs = static_cast<std::uint64_t>(
      rotary_factor * static_cast<double>(head_dimension / 2U));
  const std::uint64_t blocks = query_heads + kv_heads;
  if (rotating_pairs == 0U || rotating_pairs > kThreads ||
      !ValidGrid(blocks)) {
    return Invalid("current-table controlled fused projection RMSNorm/RoPE "
                   "extent is invalid");
  }
  ProjectionRmsNormRotaryBf16ControlledKernel
      <<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
          query, query_norm_bf16, normalized_query, key, key_norm_bf16,
          normalized_key, rotary_cosine, rotary_sine, control, query_heads,
          kv_heads, head_dimension, rotating_pairs, epsilon, false);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure(
                   "launch current-table controlled fused projection "
                   "RMSNorm/RoPE",
                   error);
}

Status LaunchGemma4Moe26BDecodeKvEpilogue(
    const float* query, const std::uint16_t* query_norm, float* normalized_query,
    const float* key, const std::uint16_t* key_norm, const float* value,
    const float* rotary_cosine, const float* rotary_sine,
    const DecodeControl* control, std::uint8_t* key_cache,
    std::uint8_t* value_cache, const std::uint16_t* key_scale,
    const std::uint16_t* value_scale, std::uint64_t kv_heads,
    std::uint64_t head_dimension, double rotary_factor,
    std::uint64_t cache_capacity, float epsilon, cudaStream_t stream) {
  if (query == nullptr || query_norm == nullptr || normalized_query == nullptr ||
      key == nullptr || key_norm == nullptr || value == nullptr ||
      rotary_cosine == nullptr || rotary_sine == nullptr || control == nullptr ||
      key_cache == nullptr || value_cache == nullptr || key_cache == value_cache ||
      key_scale == nullptr || value_scale == nullptr ||
      !((kv_heads == 8U && head_dimension == 256U && rotary_factor == 1.0) ||
        (kv_heads == 2U && head_dimension == 512U && rotary_factor == 0.25)) ||
      cache_capacity == 0U ||
      cache_capacity > std::numeric_limits<std::uint64_t>::max() / (kv_heads * head_dimension) ||
      !std::isfinite(epsilon) || epsilon <= 0.0F) {
    return Invalid("26B decode KV epilogue contract is invalid");
  }
  const auto pairs = static_cast<std::uint64_t>(rotary_factor * (head_dimension / 2U));
  Gemma4Moe26BDecodeKvEpilogueKernel
      <<<static_cast<unsigned>(16U + 2U * kv_heads), kThreads, 0, stream>>>(
          query, query_norm, normalized_query, key, key_norm, value,
          rotary_cosine, rotary_sine, control, kv_heads, head_dimension, pairs,
          epsilon, key_cache, value_cache, key_scale, value_scale,
          cache_capacity);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess ? Status::Ok()
      : CudaFailure("launch 26B decode KV epilogue", error);
}

Status LaunchGemma4Moe26BMtpKvEpilogue(
    const float* query, const std::uint16_t* query_norm,
    float* normalized_query, const float* key,
    const std::uint16_t* key_norm, const float* value,
    const float* rotary_cosine, const float* rotary_sine,
    std::uint8_t* staged_key, std::uint8_t* staged_value,
    const std::uint16_t* key_scale, const std::uint16_t* value_scale,
    std::uint64_t tokens, std::uint64_t kv_heads,
    std::uint64_t head_dimension, double rotary_factor, float epsilon,
    cudaStream_t stream) {
  const bool local = kv_heads == 8U && head_dimension == 256U &&
                     rotary_factor == 1.0;
  const bool global = kv_heads == 2U && head_dimension == 512U &&
                      rotary_factor == 0.25;
  if (query == nullptr || query_norm == nullptr || normalized_query == nullptr ||
      key == nullptr || key_norm == nullptr || value == nullptr ||
      rotary_cosine == nullptr || rotary_sine == nullptr ||
      staged_key == nullptr || staged_value == nullptr ||
      staged_key == staged_value || key_scale == nullptr ||
      value_scale == nullptr || (tokens != 2U && tokens != 3U && tokens != 5U) ||
      (!local && !global) ||
      !std::isfinite(epsilon) || epsilon <= 0.0F) {
    return Invalid("26B fixed-depth KV epilogue contract is invalid");
  }
  const auto pairs = static_cast<std::uint64_t>(
      rotary_factor * static_cast<double>(head_dimension / 2U));
  const std::uint64_t blocks = tokens * (16U + 2U * kv_heads);
  if (!ValidGrid(blocks)) {
    return Invalid("26B fixed-depth KV epilogue grid is invalid");
  }
  Gemma4Moe26BMtpKvEpilogueKernel
      <<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
          query, query_norm, normalized_query, key, key_norm, value,
          rotary_cosine, rotary_sine, staged_key, staged_value, key_scale,
          value_scale, kv_heads, head_dimension, pairs, epsilon);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch 26B fixed-depth KV epilogue", error);
}

Status LaunchRotaryEmbeddingTableBatch(
    float* cosine, float* sine, std::uint64_t tokens,
    std::uint64_t rotating_pairs, std::uint64_t frequency_dimension,
    std::uint64_t start_position, double theta, double scaling_factor,
    cudaStream_t stream) {
  if (cosine == nullptr || sine == nullptr || tokens == 0U ||
      rotating_pairs == 0U || frequency_dimension == 0U ||
      rotating_pairs * 2U > frequency_dimension ||
      !std::isfinite(theta) || theta <= 0.0 ||
      !std::isfinite(scaling_factor) || scaling_factor <= 0.0) {
    return Invalid("batched RoPE table geometry is invalid");
  }
  const std::uint64_t total_pairs = tokens * rotating_pairs;
  const std::uint64_t blocks = Blocks(total_pairs);
  if (!ValidGrid(blocks)) return Invalid("batched RoPE table grid exceeds CUDA limits");
  RotaryTableBatchKernel<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
      cosine, sine, rotating_pairs, frequency_dimension, start_position,
      theta, scaling_factor, total_pairs);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch batched RoPE table", error);
}

Status LaunchRotaryEmbeddingTableControlled(
    float* cosine, float* sine, std::uint64_t rotating_pairs,
    std::uint64_t frequency_dimension, const DecodeControl* control,
    double theta, double scaling_factor, cudaStream_t stream) {
  if (cosine == nullptr || sine == nullptr || control == nullptr ||
      rotating_pairs == 0U || frequency_dimension == 0U ||
      rotating_pairs * 2U > frequency_dimension ||
      !std::isfinite(theta) || theta <= 0.0 ||
      !std::isfinite(scaling_factor) || scaling_factor <= 0.0) {
    return Invalid("controlled RoPE table geometry is invalid");
  }
  const std::uint64_t blocks = Blocks(rotating_pairs);
  if (!ValidGrid(blocks)) {
    return Invalid("controlled RoPE table grid exceeds CUDA limits");
  }
  RotaryTableControlledKernel<<<static_cast<unsigned>(blocks), kThreads, 0,
                                stream>>>(
      cosine, sine, rotating_pairs, frequency_dimension, control, theta,
      scaling_factor);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch controlled RoPE table", error);
}

Status LaunchRotaryEmbeddingTableBatchControlled(
    float* cosine, float* sine, std::uint64_t tokens,
    std::uint64_t rotating_pairs, std::uint64_t frequency_dimension,
    const DecodeControl* controls, double theta, double scaling_factor,
    cudaStream_t stream) {
  if (cosine == nullptr || sine == nullptr || controls == nullptr ||
      tokens == 0U || rotating_pairs == 0U || frequency_dimension == 0U ||
      rotating_pairs * 2U > frequency_dimension ||
      !std::isfinite(theta) || theta <= 0.0 ||
      !std::isfinite(scaling_factor) || scaling_factor <= 0.0) {
    return Invalid("batch-controlled RoPE table geometry is invalid");
  }
  const std::uint64_t total_pairs = tokens * rotating_pairs;
  const std::uint64_t blocks = Blocks(total_pairs);
  if (!ValidGrid(blocks)) {
    return Invalid("batch-controlled RoPE table grid exceeds CUDA limits");
  }
  RotaryTableBatchControlledKernel
      <<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
          cosine, sine, rotating_pairs, frequency_dimension, controls, theta,
          scaling_factor, total_pairs);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch batch-controlled RoPE table", error);
}


}  // namespace gem16::internal
