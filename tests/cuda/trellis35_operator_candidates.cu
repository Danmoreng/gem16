#include "cuda/trellis35/reference.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <string>
#include <utility>
#include "exllamav3_quant/util.cuh"
#include "exllamav3_quant/quant/codebook.cuh"

namespace gem16::internal {
namespace {
constexpr float kHadamardScale = 0.08838834764831845F;
#include "cuda/trellis35/detail/codec.cuh"
#include "cuda/trellis35/detail/mma_w4a8.cuh"
#include "cuda/trellis35/detail/h128_warp.cuh"
#include "trellis35_m1_eight_warp_candidate.cuh"
#include "cuda/trellis35/detail/t3_pair_fragments.cuh"

template <bool CaptureTransform>
__global__ void SelectedGateInputTransformQuantizeCandidateKernel(
    const float* input, const std::uint16_t* all_suh,
    const std::uint32_t* ids, std::uint8_t* quantized, float* scales,
    float* diagnostic_transform, unsigned assignments) {
  constexpr unsigned kWidth = 2816U;
  const unsigned assignment = blockIdx.x;
  if (assignment >= assignments) return;
  const unsigned expert = ids[assignment];
  if (expert >= 128U) return;
  input += static_cast<std::uint64_t>(assignment / 8U) * kWidth;
  all_suh += static_cast<std::uint64_t>(expert) * kWidth;
  quantized += static_cast<std::uint64_t>(assignment) * kWidth;
  __shared__ float transformed[kWidth];
  __shared__ float maxima[256];
  const unsigned warp = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
  float maximum = 0.0F;
  for (unsigned block = warp; block < kWidth / 128U; block += 8U) {
    const unsigned base = block * 128U + lane * 4U;
    float values[4];
#pragma unroll
    for (unsigned i = 0U; i < 4U; ++i) values[i] = input[base + i] * F16(all_suh + base + i);
    H128Warp(values);
#pragma unroll
    for (unsigned i = 0U; i < 4U; ++i) {
      transformed[base + i] = values[i];
      maximum = fmaxf(maximum, fabsf(values[i]));
      if constexpr (CaptureTransform) diagnostic_transform[static_cast<std::uint64_t>(assignment) * kWidth + base + i] = values[i];
    }
  }
  maxima[threadIdx.x] = maximum;
  __syncthreads();
  for (unsigned stride = 128U; stride; stride >>= 1U) {
    if (threadIdx.x < stride) maxima[threadIdx.x] = fmaxf(maxima[threadIdx.x], maxima[threadIdx.x + stride]);
    __syncthreads();
  }
  const float scale = maxima[0] == 0.0F ? 1.0F : maxima[0] / 448.0F;
  if (threadIdx.x == 0U) scales[assignment] = scale;
  for (unsigned i = threadIdx.x; i < kWidth; i += blockDim.x) {
    const __nv_fp8_e4m3 encoded(transformed[i] / scale);
    quantized[i] = encoded.__x;
  }
}
}  // namespace
}  // namespace gem16::internal

extern "C" cudaError_t gem16_lab_launch_m1_eight(
    const std::uint8_t* activation, const float* scales,
    const gem16::internal::Trellis35DeviceFamilyBinding* family,
    const std::uint32_t* ids, float* output, std::uint64_t input_elements,
    std::uint64_t output_elements, cudaStream_t stream) {
  using namespace gem16::internal;
  MmaW4A8ProjectionSelectedN128EightWarpCandidateKernel<<<
      dim3(static_cast<unsigned>(output_elements / 128U), 8U), 256U, 0U, stream>>>(
          activation, scales, *family, ids, output, input_elements, output_elements);
  return cudaGetLastError();
}

extern "C" cudaError_t gem16_lab_launch_gate_fused(
    const float* input, const gem16::internal::Trellis35DeviceFamilyBinding* family,
    const std::uint32_t* ids, std::uint8_t* quantized, float* scales,
    float* diagnostic_transform, unsigned assignments, bool capture, cudaStream_t stream) {
  using namespace gem16::internal;
  if (capture) SelectedGateInputTransformQuantizeCandidateKernel<true><<<assignments, 256U, 0U, stream>>>(
      input, family->suh_f16, ids, quantized, scales, diagnostic_transform, assignments);
  else SelectedGateInputTransformQuantizeCandidateKernel<false><<<assignments, 256U, 0U, stream>>>(
      input, family->suh_f16, ids, quantized, scales, diagnostic_transform, assignments);
  return cudaGetLastError();
}

extern "C" cudaError_t gem16_lab_launch_t3_pair(
    const std::uint8_t* activation, const float* scales,
    const gem16::internal::Trellis35DeviceFamilyBinding* family,
    const std::uint32_t* ids, float* output, std::uint64_t inputs,
    std::uint64_t outputs, cudaStream_t stream) {
  using namespace gem16::internal;
  MmaW4A8ProjectionGroupedT3N128EightWarpKernel<<<
      dim3(static_cast<unsigned>(outputs / 128U), 24U), 256U, 0U, stream>>>(
          activation, scales, *family, ids, output, inputs, outputs);
  return cudaGetLastError();
}
