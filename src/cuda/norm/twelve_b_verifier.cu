#include "cuda/norm/twelve_b_verifier.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <string>

namespace gem16::internal {
namespace {

constexpr unsigned kWidth = 3840;
constexpr unsigned kThreads = 256;
constexpr unsigned kItems = kWidth / kThreads;
constexpr unsigned kWarp = 32;
constexpr float kEpsilon = 1.0e-6F;
constexpr float kE4M3Maximum = 448.0F;

__device__ __forceinline__ float RoundBf16(float value) {
  return __bfloat162float(__float2bfloat16_rn(value));
}

__device__ __forceinline__ float Bf16Value(std::uint16_t value) {
  return __bfloat162float(__ushort_as_bfloat16(value));
}

template <bool kMax>
__device__ __forceinline__ float Combine(float a, float b) {
  if constexpr (kMax) return fmaxf(a, b);
  return a + b;
}

template <bool kMax>
__device__ __forceinline__ float ReduceRow(float value, float* scratch,
                                           float& broadcast) {
  const unsigned lane = threadIdx.x;
  scratch[lane] = value;
  __syncthreads();
  if (lane < kWarp) {
    // The reference halves a 256-lane tree in descending stride order.
    // Gather its first three stages in one warp, preserving all pairings.
    const float a = Combine<kMax>(scratch[lane], scratch[lane + 128]);
    const float b = Combine<kMax>(scratch[lane + 64], scratch[lane + 192]);
    const float c = Combine<kMax>(scratch[lane + 32], scratch[lane + 160]);
    const float d = Combine<kMax>(scratch[lane + 96], scratch[lane + 224]);
    value = Combine<kMax>(Combine<kMax>(a, b), Combine<kMax>(c, d));
#pragma unroll
    for (unsigned stride = kWarp / 2; stride != 0; stride >>= 1) {
      const float other = __shfl_down_sync(0xffffffffU, value, stride);
      if (lane < stride) value = Combine<kMax>(value, other);
    }
    if (lane == 0) broadcast = value;
  }
  // Keep the result separate from the reusable lane scratch. The first
  // barrier of a subsequent reduction protects this broadcast from reuse.
  __syncthreads();
  return broadcast;
}

__global__ void TwelveBVerifierPostMlpInputFp8Kernel(
    const std::uint16_t* down, const std::uint16_t* post_weight,
    const float* residual, const std::uint16_t* layer_scalar,
    const std::uint16_t* next_weight, float* hidden,
    std::uint8_t* fp8, float* scales) {
  __shared__ float scratch[kThreads];
  __shared__ float broadcast;
  const unsigned base = blockIdx.x * kWidth;
  float values[kItems];
  float sum = 0.0F;
#pragma unroll
  for (unsigned j = 0; j < kItems; ++j) {
    const unsigned i = threadIdx.x + j * kThreads;
    values[j] = Bf16Value(down[base + i]);
    sum = fmaf(values[j], values[j], sum);
  }
  const float post_inverse =
      rsqrtf(ReduceRow<false>(sum, scratch, broadcast) / float(kWidth) + kEpsilon);
  const float scale = Bf16Value(layer_scalar[0]);
  sum = 0.0F;
#pragma unroll
  for (unsigned j = 0; j < kItems; ++j) {
    const unsigned i = threadIdx.x + j * kThreads;
    const float normalized =
        RoundBf16(values[j] * post_inverse * Bf16Value(post_weight[i]));
    values[j] = RoundBf16(RoundBf16(normalized + residual[base + i]) * scale);
    hidden[base + i] = values[j];
    sum = fmaf(values[j], values[j], sum);
  }
  const float input_inverse =
      rsqrtf(ReduceRow<false>(sum, scratch, broadcast) / float(kWidth) + kEpsilon);
  float maximum = 0.0F;
#pragma unroll
  for (unsigned j = 0; j < kItems; ++j) {
    const unsigned i = threadIdx.x + j * kThreads;
    values[j] = RoundBf16(values[j] * input_inverse * Bf16Value(next_weight[i]));
    maximum = fmaxf(maximum, fabsf(values[j]));
  }
  maximum = ReduceRow<true>(maximum, scratch, broadcast);
  const float fp8_scale = maximum == 0.0F ? 1.0F : maximum / kE4M3Maximum;
  if (threadIdx.x == 0) scales[blockIdx.x] = fp8_scale;
#pragma unroll
  for (unsigned j = 0; j < kItems; ++j) {
    const unsigned i = threadIdx.x + j * kThreads;
    fp8[base + i] = __nv_fp8_e4m3(values[j] / fp8_scale).__x;
  }
}

}  // namespace

Status LaunchTwelveBVerifierPostMlpInputFp8(
    const std::uint16_t* down_bf16, const std::uint16_t* post_mlp_weight,
    const float* residual, const std::uint16_t* layer_scalar,
    const std::uint16_t* next_input_weight, float* hidden,
    std::uint8_t* fp8, float* scales, std::uint64_t tokens, cudaStream_t stream) {
  if (tokens < 2 || tokens > 5 || !down_bf16 || !post_mlp_weight ||
      !residual || !layer_scalar || !next_input_weight || !hidden || !fp8 ||
      !scales) {
    return Status(StatusCode::kInvalidArgument,
                  "invalid 12B verifier post-MLP and input FP8 boundary");
  }
  TwelveBVerifierPostMlpInputFp8Kernel<<<static_cast<unsigned>(tokens), kThreads,
                                        0, stream>>>(
      down_bf16, post_mlp_weight, residual, layer_scalar, next_input_weight,
      hidden, fp8, scales);
  const auto error = cudaGetLastError();
  if (error != cudaSuccess) {
    return Status(StatusCode::kInternal,
                  std::string("launch 12B verifier post-MLP and input FP8: ") +
                      cudaGetErrorString(error));
  }
  return Status::Ok();
}

}  // namespace gem16::internal
