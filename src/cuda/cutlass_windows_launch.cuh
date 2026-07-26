#pragma once

#if defined(_WIN32)

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

#include "cutlass/arch/synclog.hpp"
#include "cutlass/cutlass.h"

namespace gem16::internal::cutlass_windows {

// MSVC cannot pass an over-aligned type by value. CUDA 13.3's CUtensorMap is
// alignas(128), which makes CUTLASS' normal device_kernel(Params) entry point
// fail in NVCC's generated host stub. Keep the parameter object in aligned
// device workspace and pass only its pointer across the Windows host ABI.
template <typename Operator>
__global__ __launch_bounds__(Operator::MaxThreadsPerBlock,
                             Operator::MinBlocksPerMultiprocessor)
void DeviceKernel(const typename Operator::Params* params) {
  extern __shared__ char smem[];
  Operator op;
  op(*params, smem);
  cutlass::arch::synclog_print();
}

template <typename Gemm>
cutlass::Status InitializeAndRun(
    const typename Gemm::Arguments& arguments, void* workspace,
    std::size_t workspace_bytes, cudaStream_t stream) {
  using GemmKernel = typename Gemm::GemmKernel;
  using Params = typename Gemm::Params;

  if (workspace == nullptr) return cutlass::Status::kErrorWorkspaceNull;
  if (Gemm::can_implement(arguments) != cutlass::Status::kSuccess) {
    return cutlass::Status::kInvalid;
  }

  const std::size_t cutlass_bytes = Gemm::get_workspace_size(arguments);
  const std::uintptr_t base = reinterpret_cast<std::uintptr_t>(workspace);
  if (cutlass_bytes > workspace_bytes ||
      base > std::numeric_limits<std::uintptr_t>::max() - cutlass_bytes) {
    return cutlass::Status::kErrorWorkspaceNull;
  }
  const std::uintptr_t after_cutlass = base + cutlass_bytes;
  constexpr std::size_t kParamsAlignment = alignof(Params);
  static_assert((kParamsAlignment & (kParamsAlignment - 1U)) == 0U);
  if (after_cutlass > std::numeric_limits<std::uintptr_t>::max() -
                          (kParamsAlignment - 1U)) {
    return cutlass::Status::kErrorWorkspaceNull;
  }
  const std::uintptr_t params_address =
      (after_cutlass + kParamsAlignment - 1U) &
      ~(static_cast<std::uintptr_t>(kParamsAlignment) - 1U);
  const std::size_t params_offset =
      static_cast<std::size_t>(params_address - base);
  if (params_offset > workspace_bytes ||
      sizeof(Params) > workspace_bytes - params_offset) {
    return cutlass::Status::kErrorWorkspaceNull;
  }

  cutlass::Status status =
      GemmKernel::initialize_workspace(arguments, workspace, stream);
  if (status != cutlass::Status::kSuccess) return status;

  Params params = GemmKernel::to_underlying_arguments(arguments, workspace);
  auto* device_params = reinterpret_cast<Params*>(params_address);
  cudaError_t error = cudaMemcpyAsync(
      device_params, &params, sizeof(Params), cudaMemcpyHostToDevice, stream);
  if (error != cudaSuccess) return cutlass::Status::kErrorInternal;

  constexpr int kSharedMemoryBytes = GemmKernel::SharedStorageSize;
  if constexpr (kSharedMemoryBytes >= (48 << 10)) {
    error = cudaFuncSetAttribute(
        DeviceKernel<GemmKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        kSharedMemoryBytes);
    if (error != cudaSuccess) return cutlass::Status::kErrorInternal;
  }

  const dim3 grid = GemmKernel::get_grid_shape(params);
  const dim3 block = GemmKernel::get_block_shape();
  DeviceKernel<GemmKernel><<<grid, block, kSharedMemoryBytes, stream>>>(
      device_params);
  error = cudaGetLastError();
  return error == cudaSuccess ? cutlass::Status::kSuccess
                              : cutlass::Status::kErrorInternal;
}

}  // namespace gem16::internal::cutlass_windows

#endif
