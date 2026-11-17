// Explicit experimental T4 boundary. Warp-first RMS changes FP32 reduction order.
#include "cuda/norm/twelve_b_d3_paired.h"
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cmath>
#include <string>

namespace gem16::internal {
namespace {
constexpr unsigned kThreads = 128, kPairs = 1920, kItems = 15;
__device__ __forceinline__ float Bf(float x) {
  return float(__float2bfloat16_rn(x));
}
__device__ __forceinline__ float Rcp(float x) {
  float y;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float2 WeightPair(const std::uint16_t* weight, unsigned i) {
  if (!weight) return make_float2(1.0F, 1.0F);
  return __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(weight)[i]);
}
__device__ __forceinline__ float Sum128(float x) {
  __shared__ float sums[4];
#pragma unroll
  for (unsigned delta=16; delta; delta>>=1)
    x += __shfl_down_sync(0xffffffffU, x, delta);
  const unsigned lane = threadIdx.x & 31U, warp = threadIdx.x >> 5U;
  if (lane == 0) sums[warp] = x;
  __syncthreads();
  // No mutable shared broadcast slot: all threads read the four stable sums.
  x = (sums[0] + sums[1]) + (sums[2] + sums[3]);
  __syncthreads();
  return x;
}
__global__ void TwelveBD3PairedPostAttentionKernel(float* projection,
    const std::uint16_t* post_weight, const float* residual, float* post_output,
    float* hidden_output, const std::uint16_t* pre_weight,
    std::uint8_t* packed, std::uint8_t* scales, float divisor) {
  const unsigned row = blockIdx.x;
  auto* p = reinterpret_cast<float2*>(projection) + row*kPairs;
  const auto* r = reinterpret_cast<const float2*>(residual) + row*kPairs;
  auto* hidden = reinterpret_cast<float2*>(hidden_output) + row*kPairs;
  auto* post = post_output ? reinterpret_cast<float2*>(post_output)+row*kPairs : nullptr;
  packed += row*kPairs; scales += row*240;
  float2 values[kItems];
  float sum=0.0F;
#pragma unroll
  for (unsigned j=0;j<kItems;++j) {
    const unsigned i=threadIdx.x+j*kThreads;
    const float2 x=p[i];
    values[j]=make_float2(Bf(x.x),Bf(x.y)); p[i]=values[j];
    sum=fmaf(values[j].x,values[j].x,sum);
    sum=fmaf(values[j].y,values[j].y,sum);
  }
  const float inverse1=rsqrtf(Sum128(sum)/3840.0F+1.0e-6F);
  sum=0.0F;
#pragma unroll
  for(unsigned j=0;j<kItems;++j) {
    const unsigned i=threadIdx.x+j*kThreads;
    const float2 w=WeightPair(post_weight,i),res=r[i];
    const float2 norm=make_float2(Bf(values[j].x*inverse1*w.x),Bf(values[j].y*inverse1*w.y));
    if(post)post[i]=norm;
    values[j]=make_float2(Bf(norm.x+res.x),Bf(norm.y+res.y));hidden[i]=values[j];
    sum=fmaf(values[j].x,values[j].x,sum);
    sum=fmaf(values[j].y,values[j].y,sum);
  }
  const float inverse2=rsqrtf(Sum128(sum)/3840.0F+1.0e-6F);
  const float inverse_divisor=Rcp(divisor);
#pragma unroll
  for(unsigned j=0;j<kItems;++j) {
    const unsigned i=threadIdx.x+j*kThreads;
    const float2 w=WeightPair(pre_weight,i);
    const float2 norm=make_float2(Bf(values[j].x*inverse2*w.x),Bf(values[j].y*inverse2*w.y));
    float maximum=fmaxf(fabsf(norm.x),fabsf(norm.y));
#pragma unroll
    for(unsigned delta=4;delta;delta>>=1)
      maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffffU,maximum,delta,8));
    const __nv_fp8_e4m3 encoded((maximum*Rcp(6.0F))*divisor);
    if((threadIdx.x&7U)==0)scales[i/8]=encoded.__x;
    const float decoded=float(encoded);
    const float output_scale=decoded==0.0F?0.0F:Rcp(decoded*inverse_divisor);
    packed[i]=__nv_fp4x2_e2m1(make_float2(norm.x*output_scale,norm.y*output_scale)).__x;
  }
}
} // namespace
Status LaunchTwelveBD3PairedPostAttention(float* projection,
    const std::uint16_t* post_weight, const float* residual, float* post_output,
    float* hidden_output, const std::uint16_t* pre_weight,
    std::uint8_t* packed, std::uint8_t* scales, float divisor,
    cudaStream_t stream) {
  if(!projection||!post_weight||!residual||!hidden_output||!packed||!scales||
     !(divisor>0.0F)||!std::isfinite(divisor))
    return Status(StatusCode::kInvalidArgument,"12B paired D3 boundary invalid resources");
  TwelveBD3PairedPostAttentionKernel<<<4,kThreads,0,stream>>>(projection,
      post_weight,residual,post_output,hidden_output,pre_weight,packed,scales,divisor);
  const auto e=cudaGetLastError();
  return e==cudaSuccess?Status::Ok():Status(StatusCode::kInternal,
      std::string("12B paired D3 boundary: ")+cudaGetErrorString(e));
}
} // namespace gem16::internal
