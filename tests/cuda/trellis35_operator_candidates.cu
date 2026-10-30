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
#include "cuda/trellis35/detail/t3_gate_pair_codec.cuh"
#include "cuda/trellis35/detail/t3_gate_pair_window.cuh"
#include "cuda/trellis35/detail/t3_down_pair_window.cuh"
#include "cuda/trellis35/detail/m1_gate_pair_window.cuh"
#include "trellis35_t3_gate_pipeline_candidate.cuh"

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

template <unsigned Depth>
cudaError_t LaunchT3GatePipelineLab(
    const std::uint8_t* activation, const float* scales,
    const gem16::internal::Trellis35DeviceFamilyBinding* family,
    const std::uint32_t* ids, float* output, std::uint64_t inputs,
    std::uint64_t outputs, cudaStream_t stream) {
  using namespace gem16::internal;
  if (inputs == 2816U && outputs == 1408U) {
    MmaW4A8ProjectionGroupedT3GatePipelineCandidateKernel<Depth><<<
        dim3(static_cast<unsigned>(outputs / 128U), 24U), 256U, 0U, stream>>>(
            activation, scales, *family, ids, output, inputs, outputs);
  } else {
    MmaW4A8ProjectionGroupedT3DownWindowKernel<<<
        dim3(static_cast<unsigned>(outputs / 128U), 24U), 256U, 0U, stream>>>(
            activation, scales, *family, ids, output, inputs, outputs);
  }
  return cudaGetLastError();
}

extern "C" cudaError_t gem16_lab_launch_t3_gate_pipe2(
    const std::uint8_t* activation, const float* scales,
    const gem16::internal::Trellis35DeviceFamilyBinding* family,
    const std::uint32_t* ids, float* output, std::uint64_t inputs,
    std::uint64_t outputs, cudaStream_t stream) {
  return LaunchT3GatePipelineLab<2U>(activation, scales, family, ids, output, inputs, outputs, stream);
}

extern "C" cudaError_t gem16_lab_launch_t3_gate_pipe4(
    const std::uint8_t* activation, const float* scales,
    const gem16::internal::Trellis35DeviceFamilyBinding* family,
    const std::uint32_t* ids, float* output, std::uint64_t inputs,
    std::uint64_t outputs, cudaStream_t stream) {
  return LaunchT3GatePipelineLab<4U>(activation, scales, family, ids, output, inputs, outputs, stream);
}

extern "C" cudaError_t gem16_lab_launch_t3_gate_bound2(
    const std::uint8_t* activation, const float* scales,
    const gem16::internal::Trellis35DeviceFamilyBinding* family,
    const std::uint32_t* ids, float* output, std::uint64_t inputs,
    std::uint64_t outputs, cudaStream_t stream) {
  using namespace gem16::internal;
  if (inputs == 2816U && outputs == 1408U) {
    MmaW4A8ProjectionGroupedT3GatePipelineBoundedCandidateKernel<2U><<<
        dim3(static_cast<unsigned>(outputs / 128U), 24U), 256U, 0U, stream>>>(
            activation, scales, *family, ids, output, inputs, outputs);
  } else {
    MmaW4A8ProjectionGroupedT3DownWindowKernel<<<
        dim3(static_cast<unsigned>(outputs / 128U), 24U), 256U, 0U, stream>>>(
            activation, scales, *family, ids, output, inputs, outputs);
  }
  return cudaGetLastError();
}

extern "C" cudaError_t gem16_lab_launch_m1_window(
    const std::uint8_t* activation, const float* scales,
    const gem16::internal::Trellis35DeviceFamilyBinding* family,
    const std::uint32_t* ids, float* output, std::uint64_t inputs,
    std::uint64_t outputs, cudaStream_t stream) {
  using namespace gem16::internal;
  if (inputs == 2816U && outputs == 1408U) {
    MmaW4A8ProjectionSelectedGateWindowKernel<<<
        dim3(static_cast<unsigned>(outputs / 128U), 8U), 256U, 0U, stream>>>(
            activation, scales, *family, ids, output, inputs, outputs);
    return cudaGetLastError();
  }
  return gem16_lab_launch_m1_eight(activation, scales, family, ids, output, inputs, outputs, stream);
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

extern "C" cudaError_t gem16_lab_launch_t3_window(
    const std::uint8_t* activation, const float* scales,
    const gem16::internal::Trellis35DeviceFamilyBinding* family,
    const std::uint32_t* ids, float* output, std::uint64_t inputs,
    std::uint64_t outputs, cudaStream_t stream) {
  using namespace gem16::internal;
  if (inputs == 2816U && outputs == 1408U) {
    MmaW4A8ProjectionGroupedT3GateWindowKernel<<<
        dim3(static_cast<unsigned>(outputs / 128U), 24U), 256U, 0U, stream>>>(
            activation, scales, *family, ids, output, inputs, outputs);
    return cudaGetLastError();
  }
  return gem16_lab_launch_t3_pair(activation, scales, family, ids, output, inputs, outputs, stream);
}

extern "C" cudaError_t gem16_lab_launch_t3_down_window(
    const std::uint8_t* activation, const float* scales,
    const gem16::internal::Trellis35DeviceFamilyBinding* family,
    const std::uint32_t* ids, float* output, std::uint64_t inputs,
    std::uint64_t outputs, cudaStream_t stream) {
  using namespace gem16::internal;
  if (inputs == 768U && outputs == 2816U) {
    MmaW4A8ProjectionGroupedT3DownWindowKernel<<<
        dim3(static_cast<unsigned>(outputs / 128U), 24U), 256U, 0U, stream>>>(
            activation, scales, *family, ids, output, inputs, outputs);
    return cudaGetLastError();
  }
  return gem16_lab_launch_t3_window(activation, scales, family, ids, output, inputs, outputs, stream);
}

namespace {
template <int Rate>
__global__ void CheckT3PairWindowCodecKernel(unsigned* errors) {
  const unsigned lane = threadIdx.x;
  std::uint32_t word = (blockIdx.x + 1U) * 2654435761U ^ (lane + 3U) * 2246822519U;
  word ^= word >> 15U; word *= 3266489917U; word ^= word >> 16U;
  if (blockIdx.x == 0U) word = 0U;
  if (blockIdx.x == 1U) word = 0xffffffffU;
  const unsigned position = gem16::internal::TensorCoreIndex((blockIdx.y & 3U) * 4U, blockIdx.y >> 2U);
  std::uint32_t left = 0U, right = 0U;
  gem16::internal::DecodeT3GatePairWindow<Rate>(word, position, left, right);
  const auto expected_left = gem16::internal::DecodeLanePayloadNativeE4M3x4<Rate>(word, position);
  const auto expected_right = gem16::internal::DecodeLanePayloadNativeE4M3x4<Rate>(word, position + 4U);
  if (left != expected_left || right != expected_right) atomicAdd(errors, 1U);
}
}
extern "C" cudaError_t gem16_lab_check_t3_window(unsigned* host_errors) {
  unsigned* device = nullptr;
  auto status = cudaMalloc(&device, sizeof(unsigned));
  if (status != cudaSuccess) return status;
  status = cudaMemset(device, 0, sizeof(unsigned));
  if (status == cudaSuccess) {
    CheckT3PairWindowCodecKernel<3><<<dim3(1024U, 32U), 32U>>>(device);
    CheckT3PairWindowCodecKernel<4><<<dim3(1024U, 32U), 32U>>>(device);
    status = cudaGetLastError();
    if (status == cudaSuccess) status = cudaMemcpy(host_errors, device, sizeof(unsigned), cudaMemcpyDeviceToHost);
  }
  const auto freed = cudaFree(device);
  return status == cudaSuccess ? freed : status;
}
