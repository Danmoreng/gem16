#include "cuda/attention/twelve_b_d3_fast.h"
#include "cuda/attention/detail/twelve_b_d3_fast_embedded.cuh"
#include "cuda/layer/reference.h"
#include <cuda_runtime.h>
#include <cstddef>
#include <string>

namespace gem16::internal {
namespace {
Status Failure(const char* where, cudaError_t error) {
  return Status(StatusCode::kInternal, std::string(where) + ": " + cudaGetErrorString(error));
}

__global__ void MergeD3Fast16Kernel(const float* partial, const float* lse,
                                   float* output) {
  __shared__ float weights[16];
  __shared__ float inverse;
  const unsigned head = blockIdx.x, row = blockIdx.y, lane = threadIdx.x;
  if (lane < 16U) {
    const float value = lse[(row * 16U + lane) * 16U + head];
    float maximum = value;
#pragma unroll
    for (unsigned offset = 8; offset > 0; offset >>= 1) {
      maximum = fmaxf(maximum, __shfl_down_sync(0xffffU, maximum, offset, 16));
    }
    maximum = __shfl_sync(0xffffU, maximum, 0, 16);
    float weight = expf(value - maximum);
    weights[lane] = weight;
#pragma unroll
    for (unsigned offset = 8; offset > 0; offset >>= 1) {
      weight += __shfl_down_sync(0xffffU, weight, offset, 16);
    }
    if (lane == 0U) inverse = 1.0F / weight;
  }
  __syncthreads();
#pragma unroll
  for (unsigned part = 0; part < 4; ++part) {
    const unsigned dim = lane + part * 128U;
    float result = 0.0F;
#pragma unroll
    for (unsigned segment = 0; segment < 16; ++segment) {
      result = fmaf(weights[segment],
          partial[((row * 16U + segment) * 16U + head) * 512U + dim], result);
    }
    output[(row * 16U + head) * 512U + dim] = result * inverse;
  }
}
}  // namespace

TwelveBD3FastGlobalAttention::~TwelveBD3FastGlobalAttention() {
  if (library_ != nullptr) (void)cudaLibraryUnload(library_);
}

Status TwelveBD3FastGlobalAttention::Initialize() {
  if (library_ != nullptr) return Status::Ok();
  cudaDeviceProp device{};
  auto error = cudaGetDeviceProperties(&device, 0);
  if (error != cudaSuccess) return Failure("D3 fast device properties", error);
  if (device.major != 12 || device.minor != 0) {
    return Status(StatusCode::kUnsupported, "experimental D3 FP8 attention requires SM120");
  }
  error = cudaLibraryLoadData(&library_, d3_fast::kCubin, nullptr, nullptr, 0,
                            nullptr, nullptr, 0);
  if (error != cudaSuccess) return Failure("load offline D3 FP8 cubin", error);
  error = cudaLibraryGetKernel(&kernel_, library_, d3_fast::kKernel);
  if (error != cudaSuccess) return Failure("find offline D3 FP8 kernel", error);
  // Force device module/resource realization before any capture or recurring
  // inference. The embedded artifact contains SASS, not a runtime JIT recipe.
  cudaFuncAttributes attributes{};
  error = cudaFuncGetAttributes(&attributes, reinterpret_cast<const void*>(kernel_));
  if (error != cudaSuccess) return Failure("prepare D3 FP8 kernel resources", error);
  error = cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel_),
      cudaFuncAttributeMaxDynamicSharedMemorySize, d3_fast::kShared);
  return error == cudaSuccess ? Status::Ok() : Failure("configure D3 FP8 shared memory", error);
}

Status TwelveBD3FastGlobalAttention::Launch(const float* query,
    const std::uint8_t* key, const std::uint8_t* value,
    const std::uint16_t* key_scale, const std::uint16_t* value_scale,
    float* workspace, std::uint64_t workspace_elements, float* output,
    const DecodeControl* controls, std::uint64_t capacity, cudaStream_t stream,
    unsigned rows) const {
  static_assert(offsetof(DecodeControl, position) == 8);
  if (!kernel_ || !query || !key || !value || !key_scale || !value_scale ||
      !workspace || !output || !controls || capacity < 4 || capacity > 262144U ||
      (rows != 1U && rows != 4U) || workspace_elements < WorkspaceElements()) {
    return Status(StatusCode::kInvalidArgument, "invalid experimental D3 global FP8 geometry/resources");
  }
  float* lse = workspace + 4U * 16U * 16U * 512U;
  void* no_scratch = nullptr;
  static_assert(d3_fast::kGlobalScratchPerCTA <= 256U);
  void* global_scratch = d3_fast::kGlobalScratchPerCTA == 0U ? nullptr :
      static_cast<void*>(workspace + 4U * 16U * (16U * 512U + 16U));
  auto descriptor_rows = static_cast<std::int32_t>(capacity);
  void* arguments[] = {&query, &key, &value, &key_scale, &value_scale,
                      &controls, &workspace, &lse, &descriptor_rows, &global_scratch, &no_scratch};
  auto error = cudaLaunchKernel(reinterpret_cast<const void*>(kernel_),
      dim3(16, rows), dim3(128), arguments, d3_fast::kShared, stream);
  if (error != cudaSuccess) return Failure("launch experimental D3 FP8 attention", error);
  MergeD3Fast16Kernel<<<dim3(16, rows), 128, 0, stream>>>(workspace, lse, output);
  error = cudaGetLastError();
  return error == cudaSuccess ? Status::Ok() : Failure("merge experimental D3 FP8 attention", error);
}
}  // namespace gem16::internal
