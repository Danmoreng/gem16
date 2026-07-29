#include "cuda/vision/projection.h"

#include <cuda_bf16.h>

#include <cmath>
#include <limits>
#include <string>

#include "cuda/layer/reference.h"

namespace gem16::internal {
namespace {

constexpr unsigned kThreads = 256U;
constexpr std::uint64_t kPatchWidth = 6912U;
constexpr std::uint64_t kHidden = 3840U;
constexpr float kLayerNormEpsilon = 1.0e-5F;
constexpr float kRmsNormEpsilon = 1.0e-6F;

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

__global__ void RoundBf16Kernel(float* values, std::uint64_t elements) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    values[index] = static_cast<float>(__float2bfloat16_rn(values[index]));
  }
}

__global__ void LayerNormBf16Kernel(
    const float* input, const std::uint16_t* weight,
    const std::uint16_t* bias, float* output, std::uint64_t width,
    float epsilon) {
  const std::uint64_t vector = blockIdx.x;
  const std::uint64_t base = vector * width;
  float sum = 0.0F;
  for (std::uint64_t column = threadIdx.x; column < width;
       column += blockDim.x) {
    sum += input[base + column];
  }
  const float mean = BlockSum(sum) / static_cast<float>(width);
  float squared_sum = 0.0F;
  for (std::uint64_t column = threadIdx.x; column < width;
       column += blockDim.x) {
    const float centered = input[base + column] - mean;
    squared_sum = fmaf(centered, centered, squared_sum);
  }
  const float inverse_std = rsqrtf(
      BlockSum(squared_sum) / static_cast<float>(width) + epsilon);
  for (std::uint64_t column = threadIdx.x; column < width;
       column += blockDim.x) {
    const float gamma = static_cast<float>(
        __ushort_as_bfloat16(weight[column]));
    const float beta = static_cast<float>(
        __ushort_as_bfloat16(bias[column]));
    const float value =
        (input[base + column] - mean) * inverse_std * gamma + beta;
    output[base + column] =
        static_cast<float>(__float2bfloat16_rn(value));
  }
}

__global__ void Bf16LinearBatchKernel(
    const float* input, const std::uint16_t* weight,
    const std::uint16_t* bias, float* output, std::uint64_t contracting) {
  const std::uint64_t row = blockIdx.x;
  const std::uint64_t vector = blockIdx.y;
  float sum = 0.0F;
  for (std::uint64_t column = threadIdx.x; column < contracting;
       column += blockDim.x) {
    const float coefficient = static_cast<float>(__ushort_as_bfloat16(
        weight[row * contracting + column]));
    sum = fmaf(input[vector * contracting + column], coefficient, sum);
  }
  float reduced = BlockSum(sum);
  if (threadIdx.x == 0U) {
    if (bias != nullptr) {
      reduced += static_cast<float>(__ushort_as_bfloat16(bias[row]));
    }
    output[vector * kHidden + row] =
        static_cast<float>(__float2bfloat16_rn(reduced));
  }
}

__global__ void AddPositionBf16Kernel(
    const float* input, const std::uint16_t* position_embedding,
    const std::int32_t* positions, float* output,
    std::uint64_t elements) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  const std::uint64_t patch = index / kHidden;
  const std::uint64_t column = index % kHidden;
  const std::int32_t x = positions[patch * 2U];
  const std::int32_t y = positions[patch * 2U + 1U];
  const float x_embedding = static_cast<float>(__ushort_as_bfloat16(
      position_embedding[(static_cast<std::uint64_t>(x) * 2U) * kHidden +
                         column]));
  const float y_embedding = static_cast<float>(__ushort_as_bfloat16(
      position_embedding[(static_cast<std::uint64_t>(y) * 2U + 1U) * kHidden +
                         column]));
  output[index] = static_cast<float>(
      __float2bfloat16_rn(input[index] + x_embedding + y_embedding));
}

Status CudaFailure(const char* operation, cudaError_t error) {
  return Status(StatusCode::kInternal,
                std::string(operation) + ": " + cudaGetErrorName(error) +
                    ": " + cudaGetErrorString(error));
}

Status LaunchLayerNorm(const float* input, const std::uint16_t* weight,
                       const std::uint16_t* bias, float* output,
                       std::uint64_t vectors, std::uint64_t width,
                       cudaStream_t stream) {
  LayerNormBf16Kernel<<<static_cast<unsigned>(vectors), kThreads, 0, stream>>>(
      input, weight, bias, output, width, kLayerNormEpsilon);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch vision LayerNorm", error);
}

Status LaunchLinear(const float* input, const std::uint16_t* weight,
                    const std::uint16_t* bias, float* output,
                    std::uint64_t vectors, std::uint64_t rows,
                    std::uint64_t contracting, cudaStream_t stream) {
  const dim3 grid(static_cast<unsigned>(rows),
                  static_cast<unsigned>(vectors));
  Bf16LinearBatchKernel<<<grid, kThreads, 0, stream>>>(
      input, weight, bias, output, contracting);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch vision BF16 projection", error);
}

}  // namespace

Status LaunchVisionProjection(
    float* patches, const std::int32_t* positions, float* patch_normalized,
    float* hidden_a, float* hidden_b, const VisionBinding& weights,
    float* output, std::uint64_t patch_count, cudaStream_t stream) {
  if (patches == nullptr || positions == nullptr ||
      patch_normalized == nullptr || hidden_a == nullptr ||
      hidden_b == nullptr || output == nullptr || patch_count == 0U ||
      patch_count > 280U || weights.patch_ln1_weight == nullptr ||
      weights.projection_weight == nullptr) {
    return Status(StatusCode::kInvalidArgument,
                  "vision projection pointers or patch count are invalid");
  }
  const std::uint64_t patch_elements = patch_count * kPatchWidth;
  const std::uint64_t round_blocks =
      (patch_elements + kThreads - 1U) / kThreads;
  if (round_blocks > std::numeric_limits<unsigned>::max()) {
    return Status(StatusCode::kInvalidArgument,
                  "vision patch grid is too large");
  }
  RoundBf16Kernel<<<static_cast<unsigned>(round_blocks), kThreads, 0, stream>>>(
      patches, patch_elements);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) return CudaFailure("round vision patches", error);

  Status status = LaunchLayerNorm(
      patches, weights.patch_ln1_weight, weights.patch_ln1_bias,
      patch_normalized, patch_count, kPatchWidth, stream);
  if (!status.ok()) return status;
  status = LaunchLinear(
      patch_normalized, weights.patch_dense_weight, weights.patch_dense_bias,
      hidden_a, patch_count, kHidden, kPatchWidth, stream);
  if (!status.ok()) return status;
  status = LaunchLayerNorm(
      hidden_a, weights.patch_ln2_weight, weights.patch_ln2_bias,
      hidden_b, patch_count, kHidden, stream);
  if (!status.ok()) return status;

  const std::uint64_t hidden_elements = patch_count * kHidden;
  const std::uint64_t blocks = (hidden_elements + kThreads - 1U) / kThreads;
  AddPositionBf16Kernel<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
      hidden_b, weights.position_embedding, positions, hidden_a,
      hidden_elements);
  error = cudaGetLastError();
  if (error != cudaSuccess) return CudaFailure("add vision position embeddings", error);
  status = LaunchLayerNorm(
      hidden_a, weights.position_norm_weight, weights.position_norm_bias,
      hidden_b, patch_count, kHidden, stream);
  if (!status.ok()) return status;
  status = LaunchRmsNormBf16(hidden_b, nullptr, hidden_a, patch_count,
                             kHidden, kRmsNormEpsilon, stream);
  if (!status.ok()) return status;
  return LaunchLinear(hidden_a, weights.projection_weight, nullptr, output,
                      patch_count, kHidden, kHidden, stream);
}

}  // namespace gem16::internal
