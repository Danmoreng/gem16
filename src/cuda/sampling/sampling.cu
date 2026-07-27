#include "cuda/sampling/sampling.h"

#include "cuda/layer/reference.h"

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <string>

namespace gem16::internal {
namespace {

constexpr unsigned kThreads = 256U;

Status CudaFailure(const char* operation, cudaError_t error) {
  return Status(StatusCode::kInternal,
                std::string(operation) + ": " + cudaGetErrorString(error));
}

__device__ bool IsSuppressed(std::uint32_t token,
                             const std::uint32_t* suppressed,
                             std::uint32_t suppressed_count) {
  for (std::uint32_t index = 0; index < suppressed_count; ++index) {
    if (token == suppressed[index]) return true;
  }
  return false;
}

__global__ void MarkRepetitionTokensKernel(const std::uint32_t* tokens,
                                            std::uint64_t count,
                                            std::uint32_t* mask) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    const std::uint32_t token = tokens[index];
    atomicOr(mask + token / 32U, 1U << (token % 32U));
  }
}

__global__ void MarkRepetitionTokenKernel(std::uint32_t token,
                                          std::uint32_t* mask) {
  if (threadIdx.x == 0U) {
    atomicOr(mask + token / 32U, 1U << (token % 32U));
  }
}

__global__ void MarkControlledRepetitionTokenKernel(
    const DecodeControl* control, std::uint32_t* mask) {
  if (threadIdx.x == 0U) {
    const std::uint32_t token = control->token;
    atomicOr(mask + token / 32U, 1U << (token % 32U));
  }
}

__global__ void PrepareSamplingLogitsKernel(
    const float* logits, float* adjusted, std::uint32_t* token_ids,
    const std::uint32_t* repetition_mask, float repetition_penalty,
    float inverse_temperature, const std::uint32_t* suppressed,
    std::uint32_t suppressed_count, std::uint32_t vocabulary) {
  const std::uint32_t token = blockIdx.x * blockDim.x + threadIdx.x;
  if (token >= vocabulary) return;
  float value = logits[token];
  const bool repeated =
      (repetition_mask[token / 32U] & (1U << (token % 32U))) != 0U;
  if (repetition_penalty != 1.0F && repeated) {
    value = value < 0.0F ? value * repetition_penalty
                         : value / repetition_penalty;
  }
  if (IsSuppressed(token, suppressed, suppressed_count)) value = -FLT_MAX;
  adjusted[token] = value * inverse_temperature;
  token_ids[token] = token;
}

__device__ std::uint64_t SplitMix64(std::uint64_t value) {
  value += 0x9E3779B97F4A7C15ULL;
  value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
  return value ^ (value >> 31U);
}

__global__ void PrepareSamplingProbabilitiesKernel(
    const float* sorted_logits, double* probabilities,
    std::uint32_t vocabulary, std::uint32_t top_k, float min_p) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= vocabulary) return;
  const std::uint32_t top_limit = top_k == 0U ? vocabulary : min(top_k, vocabulary);
  const float relative = index < top_limit
                             ? expf(sorted_logits[index] - sorted_logits[0])
                             : 0.0F;
  probabilities[index] = relative >= min_p ? static_cast<double>(relative) : 0.0;
}

__global__ void SampleCumulativeProbabilitiesKernel(
    const float* sorted_logits, const double* cumulative,
    const std::uint32_t* sorted_token_ids, std::uint32_t vocabulary,
    std::uint32_t top_k, float top_p, float min_p, std::uint64_t seed,
    const DecodeControl* control, std::uint64_t step,
    std::uint32_t* selected) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) return;
  std::uint32_t eligible = top_k == 0U ? vocabulary : min(top_k, vocabulary);
  if (!isfinite(sorted_logits[0])) {
    selected[0] = 0U;
    return;
  }
  if (min_p > 0.0F) {
    std::uint32_t low = 1U;
    std::uint32_t high = eligible;
    while (low < high) {
      const std::uint32_t middle = low + (high - low) / 2U;
      if (expf(sorted_logits[middle] - sorted_logits[0]) >= min_p) {
        low = middle + 1U;
      } else {
        high = middle;
      }
    }
    eligible = low;
  }
  if (top_p < 1.0F) {
    const double threshold = static_cast<double>(top_p) * cumulative[eligible - 1U];
    std::uint32_t low = 0U;
    std::uint32_t high = eligible;
    while (low < high) {
      const std::uint32_t middle = low + (high - low) / 2U;
      if (cumulative[middle] < threshold) low = middle + 1U;
      else high = middle;
    }
    eligible = min(low + 1U, eligible);
  }
  const std::uint64_t effective_step =
      control == nullptr ? step : control->sampling_step;
  const std::uint64_t random_bits =
      SplitMix64(seed ^ SplitMix64(effective_step));
  const double uniform =
      (static_cast<double>(random_bits >> 11U) + 0.5) *
      (1.0 / 9007199254740992.0);
  const double target = uniform * cumulative[eligible - 1U];
  std::uint32_t low = 0U;
  std::uint32_t high = eligible;
  while (low < high) {
    const std::uint32_t middle = low + (high - low) / 2U;
    if (cumulative[middle] <= target) low = middle + 1U;
    else high = middle;
  }
  selected[0] = sorted_token_ids[min(low, eligible - 1U)];
}

}  // namespace

Result<std::size_t> SamplingWorkspaceBytes(
    std::uint32_t vocabulary, cudaStream_t stream) {
  if (vocabulary == 0U) {
    return Status(StatusCode::kInvalidArgument,
                  "sampling vocabulary must be positive");
  }
  std::size_t sort_bytes = 0U;
  cudaError_t error = cub::DeviceRadixSort::SortPairsDescending(
      nullptr, sort_bytes, static_cast<float*>(nullptr), static_cast<float*>(nullptr),
      static_cast<std::uint32_t*>(nullptr),
      static_cast<std::uint32_t*>(nullptr), static_cast<int>(vocabulary), 0,
      sizeof(float) * 8, stream);
  if (error != cudaSuccess) return CudaFailure("size sampling radix sort", error);
  std::size_t scan_bytes = 0U;
  error = cub::DeviceScan::InclusiveSum(
      nullptr, scan_bytes, static_cast<double*>(nullptr),
      static_cast<double*>(nullptr), static_cast<int>(vocabulary), stream);
  if (error != cudaSuccess) return CudaFailure("size sampling probability scan", error);
  return std::max(sort_bytes, scan_bytes);
}

Status LaunchMarkRepetitionTokens(const std::uint32_t* tokens,
                                  std::uint64_t count, std::uint32_t* mask,
                                  cudaStream_t stream) {
  if (tokens == nullptr || mask == nullptr || count == 0U) {
    return Status(StatusCode::kInvalidArgument,
                  "repetition-token mark requires non-empty buffers");
  }
  const unsigned blocks = static_cast<unsigned>((count + kThreads - 1U) /
                                                 kThreads);
  MarkRepetitionTokensKernel<<<blocks, kThreads, 0, stream>>>(tokens, count,
                                                              mask);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess ? Status::Ok()
                              : CudaFailure("mark repetition tokens", error);
}

Status LaunchMarkRepetitionToken(std::uint32_t token, std::uint32_t* mask,
                                 cudaStream_t stream) {
  if (mask == nullptr) {
    return Status(StatusCode::kInvalidArgument,
                  "repetition-token mark requires a mask");
  }
  MarkRepetitionTokenKernel<<<1U, 1U, 0, stream>>>(token, mask);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess ? Status::Ok()
                              : CudaFailure("mark repetition token", error);
}

Status LaunchMarkControlledRepetitionToken(const DecodeControl* control,
                                           std::uint32_t* mask,
                                           cudaStream_t stream) {
  if (control == nullptr || mask == nullptr) {
    return Status(StatusCode::kInvalidArgument,
                  "controlled repetition mark requires control and mask");
  }
  MarkControlledRepetitionTokenKernel<<<1U, 1U, 0, stream>>>(control, mask);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("mark controlled repetition token", error);
}

Status LaunchSampleToken(
    float* logits, float* adjusted_logits, double* cumulative_probabilities,
    std::uint32_t* token_ids, std::uint32_t* sorted_token_ids,
    std::uint32_t* repetition_mask,
    const std::uint32_t* suppressed, std::uint32_t suppressed_count,
    std::uint32_t vocabulary, const SamplingOptions& options,
    std::uint64_t step, const DecodeControl* control, std::uint32_t* selected,
    void* algorithm_workspace, std::size_t algorithm_workspace_bytes,
    cudaStream_t stream) {
  Status validation = ValidateSamplingOptions(options, vocabulary);
  if (!validation.ok()) return validation;
  if (logits == nullptr || adjusted_logits == nullptr || token_ids == nullptr ||
      cumulative_probabilities == nullptr || sorted_token_ids == nullptr ||
      repetition_mask == nullptr || selected == nullptr ||
      algorithm_workspace == nullptr ||
      vocabulary == 0U || (suppressed_count != 0U && suppressed == nullptr)) {
    return Status(StatusCode::kInvalidArgument,
                  "sampling launch has an invalid buffer or vocabulary");
  }
  const unsigned blocks = (vocabulary + kThreads - 1U) / kThreads;
  PrepareSamplingLogitsKernel<<<blocks, kThreads, 0, stream>>>(
      logits, adjusted_logits, token_ids, repetition_mask,
      options.repetition_penalty, 1.0F / options.temperature, suppressed,
      suppressed_count, vocabulary);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) return CudaFailure("prepare sampling logits", error);
  error = cub::DeviceRadixSort::SortPairsDescending(
      algorithm_workspace, algorithm_workspace_bytes, adjusted_logits, logits,
      token_ids, sorted_token_ids, static_cast<int>(vocabulary), 0,
      sizeof(float) * 8, stream);
  if (error != cudaSuccess) return CudaFailure("sort sampling logits", error);
  PrepareSamplingProbabilitiesKernel<<<blocks, kThreads, 0, stream>>>(
      logits, cumulative_probabilities, vocabulary, options.top_k,
      options.min_p);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaFailure("prepare sampling probabilities", error);
  }
  error = cub::DeviceScan::InclusiveSum(
      algorithm_workspace, algorithm_workspace_bytes, cumulative_probabilities,
      cumulative_probabilities, static_cast<int>(vocabulary), stream);
  if (error != cudaSuccess) {
    return CudaFailure("scan sampling probabilities", error);
  }
  SampleCumulativeProbabilitiesKernel<<<1U, 1U, 0, stream>>>(
      logits, cumulative_probabilities, sorted_token_ids, vocabulary,
      options.top_k, options.top_p, options.min_p, options.seed, control, step,
      selected);
  error = cudaGetLastError();
  return error == cudaSuccess ? Status::Ok()
                              : CudaFailure("select sampled token", error);
}

}  // namespace gem16::internal
