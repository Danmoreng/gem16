#include "cuda/audio/projection.h"

#include <cuda_bf16.h>

#include <limits>
#include <string>

#include "cuda/layer/reference.h"

namespace gem16::internal {
namespace {

constexpr unsigned kThreads = 256U;
constexpr std::uint64_t kAudioWidth = 640U;
constexpr std::uint64_t kHidden = 3840U;
constexpr float kEpsilon = 1.0e-6F;

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

__global__ void RoundFramesBf16Kernel(float* frames,
                                      std::uint64_t elements) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    frames[index] = static_cast<float>(__float2bfloat16_rn(frames[index]));
  }
}

__global__ void AudioProjectionKernel(
    const float* frames, const std::uint16_t* weight, float* output) {
  const std::uint64_t row = blockIdx.x;
  const std::uint64_t frame = blockIdx.y;
  float sum = 0.0F;
  for (std::uint64_t column = threadIdx.x; column < kAudioWidth;
       column += blockDim.x) {
    const float coefficient = static_cast<float>(__ushort_as_bfloat16(
        weight[row * kAudioWidth + column]));
    sum = fmaf(frames[frame * kAudioWidth + column], coefficient, sum);
  }
  const float reduced = BlockSum(sum);
  if (threadIdx.x == 0U) {
    output[frame * kHidden + row] =
        static_cast<float>(__float2bfloat16_rn(reduced));
  }
}

Status CudaFailure(const char* operation, cudaError_t error) {
  return Status(StatusCode::kInternal,
                std::string(operation) + ": " + cudaGetErrorName(error) +
                    ": " + cudaGetErrorString(error));
}

}  // namespace

Status LaunchAudioProjection(float* frames, float* normalized_frames,
                             const std::uint16_t* projection_weight,
                             float* output, std::uint64_t frame_count,
                             cudaStream_t stream) {
  if (frames == nullptr || normalized_frames == nullptr ||
      projection_weight == nullptr || output == nullptr || frame_count == 0U ||
      frame_count > 750U) {
    return Status(StatusCode::kInvalidArgument,
                  "audio projection pointers or frame count are invalid");
  }
  const std::uint64_t elements = frame_count * kAudioWidth;
  const std::uint64_t blocks = (elements + kThreads - 1U) / kThreads;
  if (blocks > std::numeric_limits<unsigned>::max()) {
    return Status(StatusCode::kInvalidArgument,
                  "audio projection grid is too large");
  }
  RoundFramesBf16Kernel<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
      frames, elements);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) return CudaFailure("round audio frames to BF16", error);
  Status status = LaunchRmsNormBf16(frames, nullptr, normalized_frames,
                                    frame_count, kAudioWidth, kEpsilon, stream);
  if (!status.ok()) return status;
  const dim3 grid(static_cast<unsigned>(kHidden),
                  static_cast<unsigned>(frame_count));
  AudioProjectionKernel<<<grid, kThreads, 0, stream>>>(
      normalized_frames, projection_weight, output);
  error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch audio embedding projection", error);
}

}  // namespace gem16::internal
