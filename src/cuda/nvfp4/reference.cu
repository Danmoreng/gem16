#include "cuda/nvfp4/reference.h"

#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

namespace gem16gb::internal {
namespace {

constexpr std::uint64_t kBlockElements = 16;

Status Invalid(std::string message) {
  return Status(StatusCode::kInvalidArgument, std::move(message));
}

Status CudaFailure(const char* operation, cudaError_t error) {
  return Status(StatusCode::kInternal,
                std::string(operation) + ": " + cudaGetErrorName(error) + ": " +
                    cudaGetErrorString(error));
}

__device__ std::uint8_t LoadNibble(const std::uint8_t* packed, std::uint64_t index) {
  const std::uint8_t byte = packed[index / 2U];
  const unsigned shift = (index & 1U) == 0U ? 0U : 4U;
  return static_cast<std::uint8_t>((byte >> shift) & 0x0FU);
}

__device__ __forceinline__ float ReciprocalApproximateFtz(float value) {
  float result;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
  return result;
}

__global__ void QuantizeActivationReferenceKernel(const float* input,
                                                  std::uint8_t* packed_e2m1,
                                                  std::uint8_t* block_scales_e4m3fn,
                                                  std::uint64_t blocks,
                                                  float global_divisor) {
  const std::uint64_t block = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (block >= blocks) return;

  const std::uint64_t begin = block * kBlockElements;
  float amax = 0.0F;
#pragma unroll
  for (std::uint64_t local = 0; local < kBlockElements; ++local) {
    amax = fmaxf(amax, fabsf(input[begin + local]));
  }

  const __nv_fp8_e4m3 scale(
      (amax * ReciprocalApproximateFtz(6.0F)) * global_divisor);
  block_scales_e4m3fn[block] = scale.__x;
  const float decoded_scale = static_cast<float>(scale);
  const float output_scale =
      decoded_scale == 0.0F
          ? 0.0F
          : ReciprocalApproximateFtz(
                decoded_scale * ReciprocalApproximateFtz(global_divisor));

#pragma unroll
  for (std::uint64_t pair = 0; pair < kBlockElements / 2U; ++pair) {
    const std::uint64_t index = begin + pair * 2U;
    const float low_value = input[index] * output_scale;
    const float high_value = input[index + 1U] * output_scale;
    const __nv_fp4_e2m1 low(low_value);
    const __nv_fp4_e2m1 high(high_value);
    packed_e2m1[index / 2U] =
        static_cast<std::uint8_t>((low.__x & 0x0FU) | ((high.__x & 0x0FU) << 4U));
  }
}

__global__ void ProjectionReferenceKernel(const std::uint8_t* packed_activation_e2m1,
                                          const std::uint8_t* activation_scales_e4m3fn,
                                          const std::uint8_t* packed_weight_e2m1,
                                          const std::uint8_t* weight_scales_e4m3fn,
                                          float* output,
                                          std::uint64_t tokens,
                                          std::uint64_t rows,
                                          std::uint64_t contracting_elements,
                                          float output_divisor) {
  const std::uint64_t token = blockIdx.y;
  const std::uint64_t row = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (token >= tokens || row >= rows) return;

  const std::uint64_t packed_row_bytes = contracting_elements / 2U;
  const std::uint64_t scale_row_bytes = contracting_elements / kBlockElements;
  packed_activation_e2m1 += token * packed_row_bytes;
  activation_scales_e4m3fn += token * scale_row_bytes;
  output += token * rows;
  const std::uint8_t* weight_row = packed_weight_e2m1 + row * packed_row_bytes;
  const std::uint8_t* weight_scale_row = weight_scales_e4m3fn + row * scale_row_bytes;
  float accumulator = 0.0F;
  for (std::uint64_t index = 0; index < contracting_elements; ++index) {
    __nv_fp4_e2m1 activation_value;
    activation_value.__x = LoadNibble(packed_activation_e2m1, index);
    __nv_fp4_e2m1 weight_value;
    weight_value.__x = LoadNibble(weight_row, index);
    __nv_fp8_e4m3 activation_scale;
    activation_scale.__x = activation_scales_e4m3fn[index / kBlockElements];
    __nv_fp8_e4m3 weight_scale;
    weight_scale.__x = weight_scale_row[index / kBlockElements];
    const float left = static_cast<float>(activation_value) *
                       static_cast<float>(activation_scale);
    const float right = static_cast<float>(weight_value) * static_cast<float>(weight_scale);
    accumulator = fmaf(left, right, accumulator);
  }
  output[row] = accumulator / output_divisor;
}

bool PositiveFinite(float value) {
  return std::isfinite(value) && value > 0.0F;
}

}  // namespace

Status LaunchNvfp4ReferenceActivationQuantization(const float* input,
                                                  std::uint8_t* packed_e2m1,
                                                  std::uint8_t* block_scales_e4m3fn,
                                                  std::uint64_t elements,
                                                  float global_divisor,
                                                  cudaStream_t stream) {
  if (input == nullptr || packed_e2m1 == nullptr || block_scales_e4m3fn == nullptr) {
    return Invalid("NVFP4 reference activation quantization requires non-null device pointers");
  }
  if (elements == 0U || elements % kBlockElements != 0U) {
    return Invalid("NVFP4 reference activation extent must be a nonzero multiple of 16");
  }
  if (!PositiveFinite(global_divisor)) {
    return Invalid("NVFP4 reference activation global divisor must be positive and finite");
  }
  const std::uint64_t blocks = elements / kBlockElements;
  constexpr unsigned threads = 128;
  const std::uint64_t grid = (blocks + threads - 1U) / threads;
  if (grid > static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max())) {
    return Invalid("NVFP4 reference activation grid exceeds CUDA limits");
  }
  QuantizeActivationReferenceKernel<<<static_cast<unsigned>(grid), threads, 0, stream>>>(
      input, packed_e2m1, block_scales_e4m3fn, blocks, global_divisor);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess ? Status::Ok()
                              : CudaFailure("launch NVFP4 reference activation quantization", error);
}

Status LaunchNvfp4ReferenceProjection(const std::uint8_t* packed_activation_e2m1,
                                      const std::uint8_t* activation_scales_e4m3fn,
                                      const std::uint8_t* packed_weight_e2m1,
                                      const std::uint8_t* weight_scales_e4m3fn,
                                      float* output,
                                      std::uint64_t rows,
                                      std::uint64_t contracting_elements,
                                      float activation_global_divisor,
                                      float weight_global_divisor,
                                      cudaStream_t stream) {
  if (packed_activation_e2m1 == nullptr || activation_scales_e4m3fn == nullptr ||
      packed_weight_e2m1 == nullptr || weight_scales_e4m3fn == nullptr || output == nullptr) {
    return Invalid("NVFP4 reference projection requires non-null device pointers");
  }
  if (rows == 0U || contracting_elements == 0U ||
      contracting_elements % kBlockElements != 0U) {
    return Invalid("NVFP4 reference projection dimensions must be positive and K divisible by 16");
  }
  if (!PositiveFinite(activation_global_divisor) || !PositiveFinite(weight_global_divisor)) {
    return Invalid("NVFP4 reference projection global divisors must be positive and finite");
  }
  const float output_divisor = activation_global_divisor * weight_global_divisor;
  if (!PositiveFinite(output_divisor)) {
    return Invalid("NVFP4 reference projection global-divisor product overflowed");
  }
  constexpr unsigned threads = 128;
  const std::uint64_t grid = (rows + threads - 1U) / threads;
  if (grid > static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max())) {
    return Invalid("NVFP4 reference projection grid exceeds CUDA limits");
  }
  ProjectionReferenceKernel<<<dim3(static_cast<unsigned>(grid), 1U), threads, 0, stream>>>(
      packed_activation_e2m1, activation_scales_e4m3fn, packed_weight_e2m1,
      weight_scales_e4m3fn, output, 1U, rows, contracting_elements, output_divisor);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess ? Status::Ok()
                              : CudaFailure("launch NVFP4 reference projection", error);
}

Status LaunchNvfp4ReferenceProjectionBatch(
    const std::uint8_t* packed_activation_e2m1,
    const std::uint8_t* activation_scales_e4m3fn,
    const std::uint8_t* packed_weight_e2m1,
    const std::uint8_t* weight_scales_e4m3fn, float* output,
    std::uint64_t tokens, std::uint64_t rows,
    std::uint64_t contracting_elements, float activation_global_divisor,
    float weight_global_divisor, cudaStream_t stream) {
  if (packed_activation_e2m1 == nullptr || activation_scales_e4m3fn == nullptr ||
      packed_weight_e2m1 == nullptr || weight_scales_e4m3fn == nullptr ||
      output == nullptr) {
    return Invalid("batched NVFP4 reference projection requires non-null device pointers");
  }
  if (tokens == 0U || tokens > 65535U || rows == 0U ||
      contracting_elements == 0U ||
      contracting_elements % kBlockElements != 0U) {
    return Invalid("batched NVFP4 reference projection dimensions are invalid");
  }
  if (!PositiveFinite(activation_global_divisor) ||
      !PositiveFinite(weight_global_divisor)) {
    return Invalid("batched NVFP4 reference projection divisors are invalid");
  }
  const float output_divisor = activation_global_divisor * weight_global_divisor;
  if (!PositiveFinite(output_divisor)) {
    return Invalid("batched NVFP4 reference projection divisor product overflowed");
  }
  constexpr unsigned threads = 128;
  const std::uint64_t grid = (rows + threads - 1U) / threads;
  if (grid > static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max())) {
    return Invalid("batched NVFP4 reference projection grid exceeds CUDA limits");
  }
  ProjectionReferenceKernel<<<dim3(static_cast<unsigned>(grid),
                                   static_cast<unsigned>(tokens)),
                              threads, 0, stream>>>(
      packed_activation_e2m1, activation_scales_e4m3fn, packed_weight_e2m1,
      weight_scales_e4m3fn, output, tokens, rows, contracting_elements,
      output_divisor);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch batched NVFP4 reference projection", error);
}

}  // namespace gem16gb::internal
