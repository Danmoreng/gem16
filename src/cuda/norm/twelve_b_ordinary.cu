#include "cuda/norm/twelve_b_ordinary.h"
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cmath>
#include <string>

namespace gem16::internal {
namespace {
constexpr unsigned kWidth = 3840;
constexpr unsigned kThreads = 256;
constexpr unsigned kItems = kWidth / kThreads;

__device__ __forceinline__ float Bf16(float x) {
  return static_cast<float>(__float2bfloat16_rn(x));
}
__device__ __forceinline__ float Weight(const std::uint16_t* w, unsigned i) {
  return w == nullptr ? 1.0F : static_cast<float>(__ushort_as_bfloat16(w[i]));
}
__device__ __forceinline__ float Reciprocal(float x) {
  float result;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
  return result;
}

// Exact parent 256-lane descending-stride tree:128,64,32,16,8,4,2,1.
// Only the last five shared-memory stages move into warp registers.
template <bool kMaximum>
__device__ __forceinline__ float Reduce256(float value) {
  __shared__ float scratch[kThreads];
  scratch[threadIdx.x] = value;
  __syncthreads();
#pragma unroll
  for (unsigned stride = 128; stride >= 32; stride >>= 1) {
    if (threadIdx.x < stride) {
      const float other = scratch[threadIdx.x + stride];
      scratch[threadIdx.x] = kMaximum ? fmaxf(scratch[threadIdx.x], other)
                                    : scratch[threadIdx.x] + other;
    }
    __syncthreads();
  }
  if (threadIdx.x < 32) {
    value = scratch[threadIdx.x];
#pragma unroll
    for (unsigned stride = 16; stride != 0; stride >>= 1) {
      const float other = __shfl_down_sync(0xffffffffU, value, stride);
      if (threadIdx.x < stride) value = kMaximum ? fmaxf(value, other) : value + other;
    }
    if (threadIdx.x == 0) scratch[0] = value;
  }
  __syncthreads();
  const float result = scratch[0];
  // All lanes consume the result before another invocation reuses scratch.
  __syncthreads();
  return result;
}

__global__ void TwelveBOrdinaryInputNormFp8Kernel(const float* input,
    const std::uint16_t* weight, std::uint8_t* output, float* scale) {
  float values[kItems];
  float sum = 0.0F;
#pragma unroll
  for (unsigned j = 0; j < kItems; ++j) {
    values[j] = input[threadIdx.x + j * kThreads];
    sum = fmaf(values[j], values[j], sum);
  }
  const float inverse = rsqrtf(Reduce256<false>(sum) / float(kWidth) + 1.0e-6F);
  float maximum = 0.0F;
#pragma unroll
  for (unsigned j = 0; j < kItems; ++j) {
    const unsigned i = threadIdx.x + j * kThreads;
    values[j] = Bf16(values[j] * inverse * Weight(weight, i));
    maximum = fmaxf(maximum, fabsf(values[j]));
  }
  maximum = Reduce256<true>(maximum);
  const float s = maximum == 0.0F ? 1.0F : maximum / 448.0F;
  if (threadIdx.x == 0) scale[0] = s;
#pragma unroll
  for (unsigned j = 0; j < kItems; ++j) {
    const __nv_fp8_e4m3 value(values[j] / s);
    output[threadIdx.x + j * kThreads] = value.__x;
  }
}

template <unsigned kElements>
__global__ void TwelveBOrdinaryAttentionQuantizeKernel(const float* input,
    std::uint8_t* output, float* scale) {
  float values[kElements / kThreads];
  float maximum = 0.0F;
#pragma unroll
  for (unsigned j = 0; j < kElements / kThreads; ++j) {
    values[j] = Bf16(input[threadIdx.x + j * kThreads]);
    maximum = fmaxf(maximum, fabsf(values[j]));
  }
  maximum = Reduce256<true>(maximum);
  const float s = maximum == 0.0F ? 1.0F : maximum / 448.0F;
  if (threadIdx.x == 0) scale[0] = s;
#pragma unroll
  for (unsigned j = 0; j < kElements / kThreads; ++j) {
    const __nv_fp8_e4m3 value(values[j] / s);
    output[threadIdx.x + j * kThreads] = value.__x;
  }
}

__global__ void TwelveBOrdinaryPostAttentionKernel(float* projection,
    const std::uint16_t* post_weight, const float* residual, float* post_output,
    float* hidden_output, const std::uint16_t* pre_weight,
    std::uint8_t* packed, std::uint8_t* scales, float divisor) {
  float values[kItems];
  float sum = 0.0F;
#pragma unroll
  for (unsigned j = 0; j < kItems; ++j) {
    const unsigned i = threadIdx.x + j * kThreads;
    values[j] = Bf16(projection[i]);
    projection[i] = values[j];
    sum = fmaf(values[j], values[j], sum);
  }
  const float first_inverse = rsqrtf(Reduce256<false>(sum) / float(kWidth) + 1.0e-6F);
  sum = 0.0F;
#pragma unroll
  for (unsigned j = 0; j < kItems; ++j) {
    const unsigned i = threadIdx.x + j * kThreads;
    const float normalized = Bf16(values[j] * first_inverse * Weight(post_weight, i));
    if (post_output != nullptr) post_output[i] = normalized;
    values[j] = Bf16(normalized + residual[i]);
    hidden_output[i] = values[j];
    sum = fmaf(values[j], values[j], sum);
  }
  const float inverse = rsqrtf(Reduce256<false>(sum) / float(kWidth) + 1.0e-6F);
#pragma unroll
  for (unsigned j = 0; j < kItems; ++j) {
    const unsigned i = threadIdx.x + j * kThreads;
    const float normalized = Bf16(values[j] * inverse * Weight(pre_weight, i));
    // Parent NVFP4 block16 amax, distributed over sixteen contiguous lanes.
    float maximum = fmaxf(0.0F, fabsf(normalized));
#pragma unroll
    for (unsigned distance = 8; distance != 0; distance >>= 1)
      maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffU, maximum, distance, 16));
    const __nv_fp8_e4m3 encoded_scale((maximum * Reciprocal(6.0F)) * divisor);
    if ((threadIdx.x & 15U) == 0) scales[i / 16] = encoded_scale.__x;
    const float decoded = static_cast<float>(encoded_scale);
    const float output_scale = decoded == 0.0F ? 0.0F : Reciprocal(decoded * Reciprocal(divisor));
    const __nv_fp4_e2m1 value(normalized * output_scale);
    const unsigned low = value.__x;
    const unsigned high = __shfl_down_sync(0xffffffffU, low, 1);
    if ((threadIdx.x & 1U) == 0) packed[i / 2] = (low & 15U) | ((high & 15U) << 4U);
  }
}

__global__ void TwelveBOrdinaryPostMlpKernel(const float* input,
    const std::uint16_t* weight, const float* residual, float* post_output,
    float* output, const std::uint16_t* scalar) {
  float values[kItems];
  float sum = 0.0F;
#pragma unroll
  for (unsigned j = 0; j < kItems; ++j) {
    values[j] = input[threadIdx.x + j * kThreads];
    sum = fmaf(values[j], values[j], sum);
  }
  const float inverse = rsqrtf(Reduce256<false>(sum) / float(kWidth) + 1.0e-6F);
  const float layer_scale = Weight(scalar, 0);
#pragma unroll
  for (unsigned j = 0; j < kItems; ++j) {
    const unsigned i = threadIdx.x + j * kThreads;
    const float normalized = Bf16(values[j] * inverse * Weight(weight, i));
    if (post_output != nullptr) post_output[i] = normalized;
    float value = Bf16(normalized + residual[i]);
    if (scalar != nullptr) value = Bf16(value * layer_scale);
    output[i] = value;
  }
}

Status LaunchStatus(const char* operation) {
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess ? Status::Ok() : Status(StatusCode::kInternal,
      std::string(operation) + ": " + cudaGetErrorString(error));
}
}  // namespace

Status LaunchTwelveBOrdinaryInputNormFp8(const float* input,
    const std::uint16_t* weight, std::uint8_t* output, float* scale,
    cudaStream_t stream) {
  if (!input || !output || !scale) return Status(StatusCode::kInvalidArgument,"12B input norm null buffer");
  TwelveBOrdinaryInputNormFp8Kernel<<<1, kThreads, 0, stream>>>(input,weight,output,scale);
  return LaunchStatus("12B Ordinary input norm/FP8");
}
Status LaunchTwelveBOrdinaryAttentionQuantize(const float* input,
    std::uint8_t* output, float* scale, std::uint64_t elements,
    cudaStream_t stream) {
  if (!input || !output || !scale) return Status(StatusCode::kInvalidArgument,"12B attention quant null buffer");
  if (elements == 4096) TwelveBOrdinaryAttentionQuantizeKernel<4096><<<1,kThreads,0,stream>>>(input,output,scale);
  else if (elements == 8192) TwelveBOrdinaryAttentionQuantizeKernel<8192><<<1,kThreads,0,stream>>>(input,output,scale);
  else return Status(StatusCode::kInvalidArgument,"12B Ordinary attention quant requires4096/8192 elements");
  return LaunchStatus("12B Ordinary attention round/FP8");
}
Status LaunchTwelveBOrdinaryPostAttention(float* projection,
    const std::uint16_t* post_weight, const float* residual, float* post_output,
    float* hidden_output, const std::uint16_t* pre_weight,
    std::uint8_t* packed, std::uint8_t* scales, float divisor,
    cudaStream_t stream) {
  if (!projection || !post_weight || !residual || !hidden_output || !packed || !scales || !(divisor>0.0F) || !std::isfinite(divisor))
    return Status(StatusCode::kInvalidArgument,"12B post-attention boundary invalid buffer/divisor");
  TwelveBOrdinaryPostAttentionKernel<<<1,kThreads,0,stream>>>(projection,post_weight,residual,post_output,hidden_output,pre_weight,packed,scales,divisor);
  return LaunchStatus("12B Ordinary post-attention/FP4");
}
Status LaunchTwelveBOrdinaryPostMlp(const float* input,
    const std::uint16_t* weight, const float* residual, float* post_output,
    float* output, const std::uint16_t* scalar, cudaStream_t stream) {
  if (!input || !residual || !output) return Status(StatusCode::kInvalidArgument,"12B post-MLP norm null buffer");
  TwelveBOrdinaryPostMlpKernel<<<1,kThreads,0,stream>>>(input,weight,residual,post_output,output,scalar);
  return LaunchStatus("12B Ordinary post-MLP norm/residual");
}
}  // namespace gem16::internal
