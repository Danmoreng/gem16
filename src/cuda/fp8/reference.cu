#include "cuda/fp8/reference.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

namespace gem16gb::internal {
namespace {

constexpr unsigned kThreads = 256;
constexpr float kE4M3Maximum = 448.0F;

Status Invalid(std::string message) {
  return Status(StatusCode::kInvalidArgument, std::move(message));
}

Status CudaFailure(const char* operation, cudaError_t error) {
  return Status(StatusCode::kInternal,
                std::string(operation) + ": " + cudaGetErrorName(error) + ": " +
                    cudaGetErrorString(error));
}

__global__ void QuantizeTokenReferenceKernel(const float* input, std::uint8_t* output,
                                             float* output_scale, std::uint64_t elements) {
  const std::uint64_t token = blockIdx.x;
  input += token * elements;
  output += token * elements;
  output_scale += token;
  __shared__ float maxima[kThreads];
  float local_maximum = 0.0F;
  for (std::uint64_t index = threadIdx.x; index < elements; index += blockDim.x) {
    local_maximum = fmaxf(local_maximum, fabsf(input[index]));
  }
  maxima[threadIdx.x] = local_maximum;
  __syncthreads();
  for (unsigned stride = kThreads / 2U; stride != 0U; stride >>= 1U) {
    if (threadIdx.x < stride) maxima[threadIdx.x] = fmaxf(maxima[threadIdx.x], maxima[threadIdx.x + stride]);
    __syncthreads();
  }
  if (threadIdx.x == 0U) {
    output_scale[0] = maxima[0] == 0.0F ? 1.0F : maxima[0] / kE4M3Maximum;
  }
  __syncthreads();
  const float scale = output_scale[0];
  for (std::uint64_t index = threadIdx.x; index < elements; index += blockDim.x) {
    const __nv_fp8_e4m3 encoded(input[index] / scale);
    output[index] = encoded.__x;
  }
}

__global__ void RmsNormQuantizeTokenKernel(
    const float* input, const std::uint16_t* weight, std::uint8_t* output,
    float* output_scale, std::uint64_t elements, float epsilon) {
  const std::uint64_t token = blockIdx.x;
  input += token * elements;
  output += token * elements;
  output_scale += token;
  __shared__ float reduction[kThreads];
  float squared_sum = 0.0F;
  for (std::uint64_t index = threadIdx.x; index < elements;
       index += blockDim.x) {
    const float value = input[index];
    squared_sum = fmaf(value, value, squared_sum);
  }
  reduction[threadIdx.x] = squared_sum;
  __syncthreads();
  for (unsigned stride = kThreads / 2U; stride != 0U; stride >>= 1U) {
    if (threadIdx.x < stride) {
      reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inverse_rms =
      rsqrtf(reduction[0] / static_cast<float>(elements) + epsilon);
  float local_maximum = 0.0F;
  for (std::uint64_t index = threadIdx.x; index < elements;
       index += blockDim.x) {
    const float norm_scale =
        weight == nullptr
            ? 1.0F
            : static_cast<float>(__ushort_as_bfloat16(weight[index]));
    const float normalized = static_cast<float>(__float2bfloat16_rn(
        input[index] * inverse_rms * norm_scale));
    local_maximum = fmaxf(local_maximum, fabsf(normalized));
  }
  reduction[threadIdx.x] = local_maximum;
  __syncthreads();
  for (unsigned stride = kThreads / 2U; stride != 0U; stride >>= 1U) {
    if (threadIdx.x < stride) {
      reduction[threadIdx.x] =
          fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0U) {
    output_scale[0] =
        reduction[0] == 0.0F ? 1.0F : reduction[0] / kE4M3Maximum;
  }
  __syncthreads();
  const float quantization_scale = output_scale[0];
  for (std::uint64_t index = threadIdx.x; index < elements;
       index += blockDim.x) {
    const float norm_scale =
        weight == nullptr
            ? 1.0F
            : static_cast<float>(__ushort_as_bfloat16(weight[index]));
    const float normalized = static_cast<float>(__float2bfloat16_rn(
        input[index] * inverse_rms * norm_scale));
    const __nv_fp8_e4m3 encoded(normalized / quantization_scale);
    output[index] = encoded.__x;
  }
}

__global__ void ProjectionReferenceKernel(const std::uint8_t* activation,
                                          const float* activation_scale,
                                          const std::uint8_t* weight,
                                          const std::uint16_t* weight_scales,
                                          float* output,
                                          std::uint64_t tokens,
                                          std::uint64_t rows,
                                          std::uint64_t contracting_elements) {
  const std::uint64_t token = blockIdx.y;
  const std::uint64_t row = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (token >= tokens || row >= rows) return;
  activation += token * contracting_elements;
  activation_scale += token;
  output += token * rows;
  const std::uint8_t* weight_row = weight + row * contracting_elements;
  float accumulator = 0.0F;
  for (std::uint64_t index = 0; index < contracting_elements; ++index) {
    __nv_fp8_e4m3 activation_value;
    activation_value.__x = activation[index];
    __nv_fp8_e4m3 weight_value;
    weight_value.__x = weight_row[index];
    accumulator = fmaf(static_cast<float>(activation_value),
                       static_cast<float>(weight_value), accumulator);
  }
  const __nv_bfloat16 channel_scale = __ushort_as_bfloat16(weight_scales[row]);
  output[row] = accumulator * activation_scale[0] * static_cast<float>(channel_scale);
}

}  // namespace

Status LaunchFp8ReferenceTokenQuantization(const float* input, std::uint8_t* output_e4m3fn,
                                           float* output_scale, std::uint64_t elements,
                                           cudaStream_t stream) {
  if (input == nullptr || output_e4m3fn == nullptr || output_scale == nullptr) {
    return Invalid("FP8 token quantization requires non-null device pointers");
  }
  if (elements == 0U) return Invalid("FP8 token quantization requires a nonzero extent");
  QuantizeTokenReferenceKernel<<<1, kThreads, 0, stream>>>(input, output_e4m3fn,
                                                           output_scale, elements);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess ? Status::Ok()
                              : CudaFailure("launch FP8 token quantization", error);
}

Status LaunchFp8ReferenceTokenQuantizationBatch(
    const float* input, std::uint8_t* output_e4m3fn, float* output_scales,
    std::uint64_t tokens, std::uint64_t elements_per_token,
    cudaStream_t stream) {
  if (input == nullptr || output_e4m3fn == nullptr || output_scales == nullptr) {
    return Invalid("FP8 batched token quantization requires non-null device pointers");
  }
  if (tokens == 0U || elements_per_token == 0U ||
      tokens > static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max())) {
    return Invalid("FP8 batched token quantization extent is invalid");
  }
  QuantizeTokenReferenceKernel<<<static_cast<unsigned>(tokens), kThreads, 0, stream>>>(
      input, output_e4m3fn, output_scales, elements_per_token);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch batched FP8 token quantization", error);
}

Status LaunchRmsNormFp8TokenQuantizationBatch(
    const float* input, const std::uint16_t* weight_bf16,
    std::uint8_t* output_e4m3fn, float* output_scales,
    std::uint64_t tokens, std::uint64_t elements_per_token, float epsilon,
    cudaStream_t stream) {
  if (input == nullptr || output_e4m3fn == nullptr ||
      output_scales == nullptr) {
    return Invalid("fused RMSNorm FP8 quantization requires non-null pointers");
  }
  if (tokens == 0U || elements_per_token == 0U ||
      !std::isfinite(epsilon) || epsilon <= 0.0F ||
      tokens > static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max())) {
    return Invalid("fused RMSNorm FP8 quantization extent is invalid");
  }
  RmsNormQuantizeTokenKernel<<<static_cast<unsigned>(tokens), kThreads, 0,
                               stream>>>(
      input, weight_bf16, output_e4m3fn, output_scales, elements_per_token,
      epsilon);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch fused RMSNorm FP8 quantization", error);
}

Status LaunchFp8ReferenceProjection(const std::uint8_t* activation_e4m3fn,
                                    const float* activation_scale,
                                    const std::uint8_t* weight_e4m3fn,
                                    const std::uint16_t* weight_scales_bf16, float* output,
                                    std::uint64_t rows, std::uint64_t contracting_elements,
                                    cudaStream_t stream) {
  if (activation_e4m3fn == nullptr || activation_scale == nullptr || weight_e4m3fn == nullptr ||
      weight_scales_bf16 == nullptr || output == nullptr) {
    return Invalid("FP8 reference projection requires non-null device pointers");
  }
  if (rows == 0U || contracting_elements == 0U) {
    return Invalid("FP8 reference projection dimensions must be positive");
  }
  const std::uint64_t blocks = (rows + kThreads - 1U) / kThreads;
  if (blocks > static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max())) {
    return Invalid("FP8 reference projection grid exceeds CUDA limits");
  }
  ProjectionReferenceKernel<<<dim3(static_cast<unsigned>(blocks), 1U), kThreads, 0, stream>>>(
      activation_e4m3fn, activation_scale, weight_e4m3fn, weight_scales_bf16, output, 1U,
      rows, contracting_elements);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess ? Status::Ok()
                              : CudaFailure("launch FP8 reference projection", error);
}

Status LaunchFp8ReferenceProjectionBatch(
    const std::uint8_t* activation_e4m3fn, const float* activation_scales,
    const std::uint8_t* weight_e4m3fn,
    const std::uint16_t* weight_scales_bf16, float* output,
    std::uint64_t tokens, std::uint64_t rows,
    std::uint64_t contracting_elements, cudaStream_t stream) {
  if (activation_e4m3fn == nullptr || activation_scales == nullptr ||
      weight_e4m3fn == nullptr || weight_scales_bf16 == nullptr ||
      output == nullptr) {
    return Invalid("FP8 batched reference projection requires non-null device pointers");
  }
  if (tokens == 0U || rows == 0U || contracting_elements == 0U ||
      tokens > 65535U) {
    return Invalid("FP8 batched reference projection dimensions are invalid");
  }
  const std::uint64_t blocks = (rows + kThreads - 1U) / kThreads;
  if (blocks > static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max())) {
    return Invalid("FP8 batched reference projection grid exceeds CUDA limits");
  }
  ProjectionReferenceKernel<<<dim3(static_cast<unsigned>(blocks),
                                   static_cast<unsigned>(tokens)),
                              kThreads, 0, stream>>>(
      activation_e4m3fn, activation_scales, weight_e4m3fn,
      weight_scales_bf16, output, tokens, rows, contracting_elements);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch batched FP8 reference projection", error);
}

}  // namespace gem16gb::internal
