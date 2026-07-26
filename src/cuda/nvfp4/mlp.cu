#include "cuda/nvfp4/mlp.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <string>

namespace gem16::internal {
namespace {

constexpr unsigned kThreads = 256;
constexpr float kSqrtTwoOverPi = 0.7978845608028654F;
constexpr float kGeluCubic = 0.044715F;

Status Invalid(std::string message) {
  return Status(StatusCode::kInvalidArgument, std::move(message));
}

Status CudaFailure(const char* operation, cudaError_t error) {
  return Status(StatusCode::kInternal,
                std::string(operation) + ": " + cudaGetErrorName(error) + ": " +
                    cudaGetErrorString(error));
}

__global__ void GeluTanhProductKernel(const float* gate, const float* up, float* output,
                                      std::uint64_t elements) {
  const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  const float value = gate[index];
  const float inner = kSqrtTwoOverPi * (value + kGeluCubic * value * value * value);
  const float gelu =
      static_cast<float>(__float2bfloat16_rn(0.5F * value * (1.0F + tanhf(inner))));
  output[index] = gelu * up[index];
}

__global__ void AddResidualKernel(const float* mlp_output, const float* residual, float* output,
                                  std::uint64_t elements) {
  const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) output[index] = mlp_output[index] + residual[index];
}

template <typename Kernel, typename... Arguments>
Status LaunchElementwise(const char* label, Kernel kernel, std::uint64_t elements,
                         cudaStream_t stream, Arguments... arguments) {
  if (elements == 0U) return Invalid(std::string(label) + " requires a nonzero extent");
  const std::uint64_t blocks = (elements + kThreads - 1U) / kThreads;
  if (blocks > static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max())) {
    return Invalid(std::string(label) + " grid exceeds CUDA limits");
  }
  kernel<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(arguments..., elements);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess ? Status::Ok() : CudaFailure(label, error);
}

}  // namespace

Status LaunchGeluTanhProduct(const float* gate, const float* up, float* output,
                             std::uint64_t elements, cudaStream_t stream) {
  if (gate == nullptr || up == nullptr || output == nullptr) {
    return Invalid("GELU-tanh product requires non-null device pointers");
  }
  return LaunchElementwise("launch GELU-tanh product", GeluTanhProductKernel, elements, stream,
                           gate, up, output);
}

Status LaunchAddResidual(const float* mlp_output, const float* residual, float* output,
                         std::uint64_t elements, cudaStream_t stream) {
  if (mlp_output == nullptr || residual == nullptr || output == nullptr) {
    return Invalid("MLP residual addition requires non-null device pointers");
  }
  return LaunchElementwise("launch MLP residual addition", AddResidualKernel, elements, stream,
                           mlp_output, residual, output);
}

}  // namespace gem16::internal
