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

template <typename Input>
__device__ float LoadBoundaryValue(const Input* input, std::uint64_t index) {
  if constexpr (std::is_same_v<Input, std::uint16_t>) {
    return static_cast<float>(__ushort_as_bfloat16(input[index]));
  } else {
    return input[index];
  }
}

template <typename Input, bool kRoundBf16>
__global__ void RmsNormKernel(const Input* input, const std::uint16_t* weight,
                              float* output, std::uint64_t width, float epsilon) {
  const std::uint64_t vector = blockIdx.x;
  const std::uint64_t base = vector * width;
  float squared_sum = 0.0F;
  for (std::uint64_t index = threadIdx.x; index < width; index += blockDim.x) {
    const float value = LoadBoundaryValue(input, base + index);
    squared_sum = fmaf(value, value, squared_sum);
  }
  const float inverse_rms = rsqrtf(BlockSum(squared_sum) / static_cast<float>(width) + epsilon);
  for (std::uint64_t index = threadIdx.x; index < width; index += blockDim.x) {
    const float scale = weight == nullptr ? 1.0F :
        static_cast<float>(__ushort_as_bfloat16(weight[index]));
    const float value =
        LoadBoundaryValue(input, base + index) * inverse_rms * scale;
    if constexpr (kRoundBf16) {
      output[base + index] = static_cast<float>(__float2bfloat16_rn(value));
    } else {
      output[base + index] = value;
    }
  }
}

template <typename Input>
__global__ void RmsNormResidualBf16Kernel(
    const Input* input, const std::uint16_t* weight, const float* residual,
    float* normalized_output, float* output, std::uint64_t width,
    float epsilon, const std::uint16_t* scalar) {
  const std::uint64_t vector = blockIdx.x;
  const std::uint64_t base = vector * width;
  float squared_sum = 0.0F;
  for (std::uint64_t index = threadIdx.x; index < width;
       index += blockDim.x) {
    const float value = LoadBoundaryValue(input, base + index);
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
        LoadBoundaryValue(input, base + index) * inverse_rms * norm_scale));
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

__global__ void RmsNormResidualPhysicalBf16Kernel(
    const std::uint16_t* input, const std::uint16_t* weight,
    const std::uint16_t* residual, std::uint16_t* output,
    std::uint64_t width, float epsilon, const std::uint16_t* scalar) {
  const std::uint64_t vector = blockIdx.x;
  const std::uint64_t base = vector * width;
  float squared_sum = 0.0F;
  for (std::uint64_t index = threadIdx.x; index < width;
       index += blockDim.x) {
    const float value = LoadBoundaryValue(input, base + index);
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
        LoadBoundaryValue(input, base + index) * inverse_rms * norm_scale));
    float value = static_cast<float>(__float2bfloat16_rn(
        normalized + LoadBoundaryValue(residual, base + index)));
    if (scalar != nullptr) {
      value = static_cast<float>(__float2bfloat16_rn(value * layer_scale));
    }
    output[base + index] =
        __bfloat16_as_ushort(__float2bfloat16_rn(value));
  }
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
  RmsNormKernel<float, false>
      <<<static_cast<unsigned>(vectors), kThreads, 0, stream>>>(
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
  RmsNormKernel<float, true>
      <<<static_cast<unsigned>(vectors), kThreads, 0, stream>>>(
      input, weight_bf16, output, width, epsilon);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch BF16-boundary RMSNorm", error);
}

Status LaunchRmsNormBf16Input(
    const std::uint16_t* input_bf16,
    const std::uint16_t* weight_bf16, float* output,
    std::uint64_t vectors, std::uint64_t width, float epsilon,
    cudaStream_t stream) {
  if (input_bf16 == nullptr || output == nullptr) {
    return Invalid("physical-BF16 RMSNorm requires non-null input and output");
  }
  if (vectors == 0U || width == 0U || !std::isfinite(epsilon) ||
      epsilon <= 0.0F || !ValidGrid(vectors)) {
    return Invalid("physical-BF16 RMSNorm geometry or epsilon is invalid");
  }
  RmsNormKernel<std::uint16_t, true>
      <<<static_cast<unsigned>(vectors), kThreads, 0, stream>>>(
          input_bf16, weight_bf16, output, width, epsilon);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch physical-BF16 RMSNorm", error);
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
  RmsNormResidualBf16Kernel<float>
      <<<static_cast<unsigned>(vectors), kThreads, 0, stream>>>(
      input, weight_bf16, residual, normalized_output, output, width, epsilon,
      scalar_bf16);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch fused RMSNorm residual", error);
}

Status LaunchRmsNormResidualBf16Input(
    const std::uint16_t* input_bf16, const std::uint16_t* weight_bf16,
    const float* residual, float* normalized_output, float* output,
    std::uint64_t vectors, std::uint64_t width, float epsilon,
    const std::uint16_t* scalar_bf16, cudaStream_t stream) {
  if (input_bf16 == nullptr || residual == nullptr || output == nullptr) {
    return Invalid(
        "fused BF16-input RMSNorm residual requires non-null input, residual, "
        "and output");
  }
  if (vectors == 0U || width == 0U || !std::isfinite(epsilon) ||
      epsilon <= 0.0F || !ValidGrid(vectors)) {
    return Invalid(
        "fused BF16-input RMSNorm residual geometry or epsilon is invalid");
  }
  RmsNormResidualBf16Kernel<std::uint16_t>
      <<<static_cast<unsigned>(vectors), kThreads, 0, stream>>>(
          input_bf16, weight_bf16, residual, normalized_output, output, width,
          epsilon, scalar_bf16);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch fused BF16-input RMSNorm residual", error);
}

Status LaunchRmsNormResidualPhysicalBf16(
    const std::uint16_t* input_bf16, const std::uint16_t* weight_bf16,
    const std::uint16_t* residual_bf16, std::uint16_t* output_bf16,
    std::uint64_t vectors, std::uint64_t width, float epsilon,
    const std::uint16_t* scalar_bf16, cudaStream_t stream) {
  if (input_bf16 == nullptr || residual_bf16 == nullptr ||
      output_bf16 == nullptr) {
    return Invalid(
        "physical-BF16 RMSNorm residual requires non-null input, residual, and output");
  }
  if (vectors == 0U || width == 0U || !std::isfinite(epsilon) ||
      epsilon <= 0.0F || !ValidGrid(vectors)) {
    return Invalid("physical-BF16 RMSNorm residual geometry is invalid");
  }
  RmsNormResidualPhysicalBf16Kernel<<<static_cast<unsigned>(vectors), kThreads,
                                      0, stream>>>(
      input_bf16, weight_bf16, residual_bf16, output_bf16, width, epsilon,
      scalar_bf16);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch physical-BF16 RMSNorm residual", error);
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


}  // namespace gem16::internal
