#include "cuda/nvfp4/gemv.h"

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

constexpr std::uint64_t kScaleGroupElements = 16;
constexpr std::uint64_t kElementsPerLane = 8;
constexpr unsigned kWarpSize = 32;
constexpr unsigned kWarpsPerBlock = 8;
constexpr unsigned kThreadsPerBlock = kWarpSize * kWarpsPerBlock;

Status Invalid(std::string message) {
  return Status(StatusCode::kInvalidArgument, std::move(message));
}

Status CudaFailure(const char* operation, cudaError_t error) {
  return Status(StatusCode::kInternal,
                std::string(operation) + ": " + cudaGetErrorName(error) + ": " +
                    cudaGetErrorString(error));
}

bool PositiveFinite(float value) {
  return std::isfinite(value) && value > 0.0F;
}

__device__ __forceinline__ std::uint32_t LoadU32(const std::uint8_t* source) {
  return *reinterpret_cast<const std::uint32_t*>(source);
}

__device__ __forceinline__ float DecodeScale(std::uint8_t code) {
  __nv_fp8_e4m3 value;
  value.__x = code;
  return static_cast<float>(value);
}

__device__ __forceinline__ float DecodeE2M1(std::uint8_t code) {
  const unsigned magnitude_code = code & 7U;
  const float magnitude = magnitude_code < 4U
                              ? static_cast<float>(magnitude_code) * 0.5F
                              : static_cast<float>(magnitude_code - 2U +
                                                   (magnitude_code == 7U ? 1U : 0U));
  return (code & 8U) == 0U ? magnitude : -magnitude;
}

__device__ __forceinline__ float2 DecodePair(std::uint8_t packed) {
  return make_float2(DecodeE2M1(packed & 0x0FU), DecodeE2M1(packed >> 4U));
}

__global__ void SimtGemvKernel(const std::uint8_t* packed_activation_e2m1,
                               const std::uint8_t* activation_scales_e4m3fn,
                               const std::uint8_t* packed_weight_e2m1,
                               const std::uint8_t* weight_scales_e4m3fn,
                               float* output,
                               std::uint64_t rows,
                               std::uint64_t contracting_elements,
                               float output_divisor) {
  extern __shared__ __align__(16) std::uint8_t shared[];
  const std::uint64_t packed_bytes = contracting_elements / 2U;
  const std::uint64_t scale_bytes = contracting_elements / kScaleGroupElements;
  std::uint8_t* shared_activation = shared;
  std::uint8_t* shared_scales = shared + packed_bytes;

  for (std::uint64_t index = threadIdx.x; index < packed_bytes; index += blockDim.x) {
    shared_activation[index] = packed_activation_e2m1[index];
  }
  for (std::uint64_t index = threadIdx.x; index < scale_bytes; index += blockDim.x) {
    shared_scales[index] = activation_scales_e4m3fn[index];
  }
  __syncthreads();

  const unsigned warp = threadIdx.x / kWarpSize;
  const unsigned lane = threadIdx.x & (kWarpSize - 1U);
  const std::uint64_t row = static_cast<std::uint64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  if (row >= rows) return;

  const std::uint64_t packed_row_bytes = packed_bytes;
  const std::uint64_t scale_row_bytes = scale_bytes;
  const std::uint8_t* weight_row = packed_weight_e2m1 + row * packed_row_bytes;
  const std::uint8_t* weight_scale_row = weight_scales_e4m3fn + row * scale_row_bytes;
  float accumulator = 0.0F;

  constexpr std::uint64_t kElementsPerWarpIteration = kWarpSize * kElementsPerLane;
  for (std::uint64_t base = 0; base < contracting_elements;
       base += kElementsPerWarpIteration) {
    const std::uint64_t element = base + static_cast<std::uint64_t>(lane) * kElementsPerLane;
    if (element >= contracting_elements) continue;

    const std::uint32_t activation_word = LoadU32(shared_activation + element / 2U);
    const std::uint32_t weight_word = LoadU32(weight_row + element / 2U);
    const float activation_scale = DecodeScale(shared_scales[element / kScaleGroupElements]);
    const float weight_scale = DecodeScale(weight_scale_row[element / kScaleGroupElements]);

#pragma unroll
    for (unsigned byte_index = 0; byte_index < sizeof(std::uint32_t); ++byte_index) {
      const unsigned shift = byte_index * 8U;
      const float2 activation = DecodePair(
          static_cast<std::uint8_t>((activation_word >> shift) & 0xFFU));
      const float2 weight =
          DecodePair(static_cast<std::uint8_t>((weight_word >> shift) & 0xFFU));
      accumulator = fmaf(activation.x * activation_scale, weight.x * weight_scale, accumulator);
      accumulator = fmaf(activation.y * activation_scale, weight.y * weight_scale, accumulator);
    }
  }

#pragma unroll
  for (unsigned offset = kWarpSize / 2U; offset != 0U; offset >>= 1U) {
    accumulator += __shfl_down_sync(0xFFFFFFFFU, accumulator, offset);
  }
  if (lane == 0U) output[row] = accumulator / output_divisor;
}

}  // namespace

Status LaunchNvfp4SimtGemvProjection(const std::uint8_t* packed_activation_e2m1,
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
    return Invalid("NVFP4 SIMT GEMV requires non-null device pointers");
  }
  if (rows == 0U || contracting_elements == 0U ||
      contracting_elements % 64U != 0U) {
    return Invalid("NVFP4 SIMT GEMV requires positive dimensions and K divisible by 64");
  }
  if (!PositiveFinite(activation_global_divisor) || !PositiveFinite(weight_global_divisor)) {
    return Invalid("NVFP4 SIMT GEMV global divisors must be positive and finite");
  }
  const float output_divisor = activation_global_divisor * weight_global_divisor;
  if (!PositiveFinite(output_divisor)) {
    return Invalid("NVFP4 SIMT GEMV global-divisor product overflowed");
  }

  const std::uint64_t blocks = (rows + kWarpsPerBlock - 1U) / kWarpsPerBlock;
  if (blocks > static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max())) {
    return Invalid("NVFP4 SIMT GEMV grid exceeds CUDA limits");
  }
  const std::uint64_t shared_bytes = contracting_elements / 2U +
                                     contracting_elements / kScaleGroupElements;
  if (shared_bytes > static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max())) {
    return Invalid("NVFP4 SIMT GEMV shared-memory extent exceeds CUDA limits");
  }

  SimtGemvKernel<<<static_cast<unsigned>(blocks), kThreadsPerBlock,
                   static_cast<unsigned>(shared_bytes), stream>>>(
      packed_activation_e2m1, activation_scales_e4m3fn, packed_weight_e2m1,
      weight_scales_e4m3fn, output, rows, contracting_elements, output_divisor);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess ? Status::Ok()
                              : CudaFailure("launch direct-source NVFP4 SIMT GEMV", error);
}

}  // namespace gem16gb::internal
