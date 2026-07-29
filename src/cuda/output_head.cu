#include "cuda/output_head.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>
#include <string>
#include <utility>

namespace gem16::internal {
namespace {

constexpr std::uint64_t kHidden = 3840U;
constexpr std::uint64_t kVocabulary = 262144U;
constexpr std::uint64_t kMaximumMtpVerifyTokens = 5U;
constexpr unsigned kThreads = 256U;

Status CudaFailure(const char* operation, cudaError_t error) {
  return Status(StatusCode::kInternal,
                std::string(operation) + ": " + cudaGetErrorName(error) +
                    ": " + cudaGetErrorString(error));
}

__device__ bool IsSuppressed(std::uint64_t token,
                             const std::uint32_t* suppressed,
                             std::uint32_t suppressed_count) {
  for (std::uint32_t index = 0U; index < suppressed_count; ++index) {
    if (token == suppressed[index]) return true;
  }
  return false;
}

__global__ void OutputHeadKernel(const std::uint16_t* weights,
                                 const float* hidden, float* logits) {
  __shared__ float scratch[kThreads];
  const std::uint64_t token = blockIdx.x;
  float sum = 0.0F;
  const std::uint64_t base = token * kHidden;
  for (std::uint64_t index = threadIdx.x; index < kHidden;
       index += blockDim.x) {
    const float weight = static_cast<float>(
        __ushort_as_bfloat16(weights[base + index]));
    sum = fmaf(weight, hidden[index], sum);
  }
  scratch[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned stride = kThreads / 2U; stride != 0U; stride >>= 1U) {
    if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0U) logits[token] = tanhf(scratch[0] / 30.0F) * 30.0F;
}

__global__ void FusedOutputHeadCandidatesKernel(
    const std::uint16_t* weights, const float* hidden,
    const std::uint32_t* suppressed, std::uint32_t suppressed_count,
    const DecodeControl* control, OutputHeadCandidate* candidates,
    float* diagnostic_logits) {
  constexpr unsigned kWarpSize = 32U;
  constexpr unsigned kWarpsPerBlock = kThreads / kWarpSize;
  __shared__ OutputHeadCandidate warp_candidates[kWarpsPerBlock];
  const unsigned warp = threadIdx.x / kWarpSize;
  const unsigned lane = threadIdx.x % kWarpSize;
  OutputHeadCandidate best{-FLT_MAX, 0U};
  const std::uint32_t dynamic_suppressed_count =
      control == nullptr ? suppressed_count : control->suppressed_token_count;
  for (std::uint64_t token =
           static_cast<std::uint64_t>(blockIdx.x) * kWarpsPerBlock + warp;
       token < kVocabulary;
       token += static_cast<std::uint64_t>(gridDim.x) * kWarpsPerBlock) {
    float sum = 0.0F;
    const std::uint64_t base = token * kHidden;
    for (std::uint64_t index = lane; index < kHidden; index += kWarpSize) {
      const float weight = static_cast<float>(
          __ushort_as_bfloat16(weights[base + index]));
      sum = fmaf(weight, hidden[index], sum);
    }
    for (unsigned offset = kWarpSize / 2U; offset != 0U; offset >>= 1U) {
      sum += __shfl_down_sync(0xFFFFFFFFU, sum, offset);
    }
    if (lane == 0U) {
      const float softcapped = tanhf(sum / 30.0F) * 30.0F;
      if (diagnostic_logits != nullptr) diagnostic_logits[token] = softcapped;
      if (!IsSuppressed(token, suppressed, dynamic_suppressed_count) &&
          (softcapped > best.value ||
           (softcapped == best.value && token < best.token))) {
        best = {softcapped, static_cast<std::uint32_t>(token)};
      }
    }
  }
  if (lane == 0U) warp_candidates[warp] = best;
  __syncthreads();
  if (warp == 0U) {
    best = lane < kWarpsPerBlock ? warp_candidates[lane]
                                 : OutputHeadCandidate{-FLT_MAX, 0U};
    for (unsigned offset = kWarpSize / 2U; offset != 0U; offset >>= 1U) {
      const float other_value =
          __shfl_down_sync(0xFFFFFFFFU, best.value, offset);
      const std::uint32_t other_token =
          __shfl_down_sync(0xFFFFFFFFU, best.token, offset);
      if (other_value > best.value ||
          (other_value == best.value && other_token < best.token)) {
        best = {other_value, other_token};
      }
    }
    if (lane == 0U) candidates[blockIdx.x] = best;
  }
}

__global__ void FusedOutputHeadBatchCandidatesKernel(
    const std::uint16_t* weights, const float* hidden,
    const std::uint32_t* suppressed, std::uint32_t suppressed_count,
    std::uint64_t rows, OutputHeadCandidate* candidates,
    float* diagnostic_logits) {
  constexpr unsigned kWarpSize = 32U;
  constexpr unsigned kWarpsPerBlock = kThreads / kWarpSize;
  __shared__ OutputHeadCandidate
      warp_candidates[kMaximumMtpVerifyTokens][kWarpsPerBlock];
  const unsigned warp = threadIdx.x / kWarpSize;
  const unsigned lane = threadIdx.x % kWarpSize;
  OutputHeadCandidate best[kMaximumMtpVerifyTokens];
#pragma unroll
  for (unsigned row = 0U; row < kMaximumMtpVerifyTokens; ++row) {
    best[row] = {-FLT_MAX, 0U};
  }
  for (std::uint64_t token =
           static_cast<std::uint64_t>(blockIdx.x) * kWarpsPerBlock + warp;
       token < kVocabulary;
       token += static_cast<std::uint64_t>(gridDim.x) * kWarpsPerBlock) {
    float sums[kMaximumMtpVerifyTokens]{};
    const std::uint64_t base = token * kHidden;
    for (std::uint64_t index = lane; index < kHidden; index += kWarpSize) {
      const float weight = static_cast<float>(
          __ushort_as_bfloat16(weights[base + index]));
#pragma unroll
      for (unsigned row = 0U; row < kMaximumMtpVerifyTokens; ++row) {
        if (row < rows) {
          sums[row] = fmaf(weight, hidden[row * kHidden + index], sums[row]);
        }
      }
    }
#pragma unroll
    for (unsigned row = 0U; row < kMaximumMtpVerifyTokens; ++row) {
      if (row >= rows) continue;
      for (unsigned offset = kWarpSize / 2U; offset != 0U; offset >>= 1U) {
        sums[row] += __shfl_down_sync(0xFFFFFFFFU, sums[row], offset);
      }
      if (lane == 0U) {
        const float value = tanhf(sums[row] / 30.0F) * 30.0F;
        if (diagnostic_logits != nullptr) {
          diagnostic_logits[row * kVocabulary + token] = value;
        }
        if (!IsSuppressed(token, suppressed, suppressed_count) &&
            (value > best[row].value ||
             (value == best[row].value && token < best[row].token))) {
          best[row] = {value, static_cast<std::uint32_t>(token)};
        }
      }
    }
  }
  if (lane == 0U) {
#pragma unroll
    for (unsigned row = 0U; row < kMaximumMtpVerifyTokens; ++row) {
      if (row < rows) warp_candidates[row][warp] = best[row];
    }
  }
  __syncthreads();
  if (warp == 0U) {
#pragma unroll
    for (unsigned row = 0U; row < kMaximumMtpVerifyTokens; ++row) {
      if (row >= rows) continue;
      OutputHeadCandidate value =
          lane < kWarpsPerBlock ? warp_candidates[row][lane]
                                : OutputHeadCandidate{-FLT_MAX, 0U};
      for (unsigned offset = kWarpSize / 2U; offset != 0U; offset >>= 1U) {
        const float other_value =
            __shfl_down_sync(0xFFFFFFFFU, value.value, offset);
        const std::uint32_t other_token =
            __shfl_down_sync(0xFFFFFFFFU, value.token, offset);
        if (other_value > value.value ||
            (other_value == value.value && other_token < value.token)) {
          value = {other_value, other_token};
        }
      }
      if (lane == 0U) {
        candidates[row * kOutputHeadCandidateBlocks + blockIdx.x] = value;
      }
    }
  }
}

template <unsigned kRows>
__global__ void FusedOutputHeadFixedBatchCandidatesKernel(
    const std::uint16_t* weights, const float* hidden,
    const std::uint32_t* suppressed, std::uint32_t suppressed_count,
    OutputHeadCandidate* candidates, float* diagnostic_logits) {
  constexpr unsigned kWarpSize = 32U;
  constexpr unsigned kWarpsPerBlock = kThreads / kWarpSize;
  __shared__ OutputHeadCandidate warp_candidates[kRows][kWarpsPerBlock];
  const unsigned warp = threadIdx.x / kWarpSize;
  const unsigned lane = threadIdx.x % kWarpSize;
  OutputHeadCandidate best[kRows];
#pragma unroll
  for (unsigned row = 0U; row < kRows; ++row) {
    best[row] = {-FLT_MAX, 0U};
  }
  for (std::uint64_t token =
           static_cast<std::uint64_t>(blockIdx.x) * kWarpsPerBlock + warp;
       token < kVocabulary;
       token += static_cast<std::uint64_t>(gridDim.x) * kWarpsPerBlock) {
    float sums[kRows]{};
    const std::uint64_t base = token * kHidden;
    for (std::uint64_t index = lane; index < kHidden; index += kWarpSize) {
      const float weight = static_cast<float>(
          __ushort_as_bfloat16(weights[base + index]));
#pragma unroll
      for (unsigned row = 0U; row < kRows; ++row) {
        sums[row] =
            fmaf(weight, hidden[row * kHidden + index], sums[row]);
      }
    }
#pragma unroll
    for (unsigned row = 0U; row < kRows; ++row) {
      for (unsigned offset = kWarpSize / 2U; offset != 0U; offset >>= 1U) {
        sums[row] += __shfl_down_sync(0xFFFFFFFFU, sums[row], offset);
      }
      if (lane == 0U) {
        const float value = tanhf(sums[row] / 30.0F) * 30.0F;
        if (diagnostic_logits != nullptr) {
          diagnostic_logits[row * kVocabulary + token] = value;
        }
        if (!IsSuppressed(token, suppressed, suppressed_count) &&
            (value > best[row].value ||
             (value == best[row].value && token < best[row].token))) {
          best[row] = {value, static_cast<std::uint32_t>(token)};
        }
      }
    }
  }
  if (lane == 0U) {
#pragma unroll
    for (unsigned row = 0U; row < kRows; ++row) {
      warp_candidates[row][warp] = best[row];
    }
  }
  __syncthreads();
  if (warp == 0U) {
#pragma unroll
    for (unsigned row = 0U; row < kRows; ++row) {
      OutputHeadCandidate value =
          lane < kWarpsPerBlock ? warp_candidates[row][lane]
                                : OutputHeadCandidate{-FLT_MAX, 0U};
      for (unsigned offset = kWarpSize / 2U; offset != 0U; offset >>= 1U) {
        const float other_value =
            __shfl_down_sync(0xFFFFFFFFU, value.value, offset);
        const std::uint32_t other_token =
            __shfl_down_sync(0xFFFFFFFFU, value.token, offset);
        if (other_value > value.value ||
            (other_value == value.value && other_token < value.token)) {
          value = {other_value, other_token};
        }
      }
      if (lane == 0U) {
        candidates[row * kOutputHeadCandidateBlocks + blockIdx.x] = value;
      }
    }
  }
}

template <bool kBatch>
__global__ void OutputHeadArgmaxKernel(const OutputHeadCandidate* candidates,
                                       std::uint64_t rows,
                                       std::uint32_t* selected) {
  const std::uint64_t row = kBatch ? blockIdx.x : 0U;
  if (row >= rows) return;
  __shared__ OutputHeadCandidate scratch[kThreads];
  OutputHeadCandidate best{-FLT_MAX, 0U};
  const OutputHeadCandidate* row_candidates =
      candidates + row * kOutputHeadCandidateBlocks;
  for (std::uint32_t index = threadIdx.x; index < kOutputHeadCandidateBlocks;
       index += blockDim.x) {
    const OutputHeadCandidate candidate = row_candidates[index];
    if (candidate.value > best.value ||
        (candidate.value == best.value && candidate.token < best.token)) {
      best = candidate;
    }
  }
  scratch[threadIdx.x] = best;
  __syncthreads();
  for (unsigned stride = kThreads / 2U; stride != 0U; stride >>= 1U) {
    if (threadIdx.x < stride) {
      const OutputHeadCandidate other = scratch[threadIdx.x + stride];
      if (other.value > scratch[threadIdx.x].value ||
          (other.value == scratch[threadIdx.x].value &&
           other.token < scratch[threadIdx.x].token)) {
        scratch[threadIdx.x] = other;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0U) selected[row] = scratch[0].token;
}

__global__ void ArgmaxKernel(const float* logits,
                             const std::uint32_t* suppressed,
                             std::uint32_t suppressed_count,
                             std::uint32_t* selected) {
  __shared__ OutputHeadCandidate scratch[kThreads];
  OutputHeadCandidate best{-FLT_MAX, 0U};
  for (std::uint64_t index = threadIdx.x; index < kVocabulary;
       index += blockDim.x) {
    if (IsSuppressed(index, suppressed, suppressed_count)) continue;
    const float value = logits[index];
    if (value > best.value || (value == best.value && index < best.token)) {
      best = {value, static_cast<std::uint32_t>(index)};
    }
  }
  scratch[threadIdx.x] = best;
  __syncthreads();
  for (unsigned stride = kThreads / 2U; stride != 0U; stride >>= 1U) {
    if (threadIdx.x < stride) {
      const OutputHeadCandidate other = scratch[threadIdx.x + stride];
      if (other.value > scratch[threadIdx.x].value ||
          (other.value == scratch[threadIdx.x].value &&
           other.token < scratch[threadIdx.x].token)) {
        scratch[threadIdx.x] = other;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0U) selected[0] = scratch[0].token;
}

}  // namespace

Status LaunchFusedOutputHeadCandidates(
    const std::uint16_t* weights, const float* hidden,
    const std::uint32_t* suppressed, std::uint32_t suppressed_count,
    const DecodeControl* control, OutputHeadCandidate* candidates,
    float* diagnostic_logits, cudaStream_t stream) {
  FusedOutputHeadCandidatesKernel<<<kOutputHeadCandidateBlocks, kThreads, 0,
                                    stream>>>(
      weights, hidden, suppressed, suppressed_count, control, candidates,
      diagnostic_logits);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch fused output-head candidates", error);
}

Status LaunchFusedOutputHeadBatchCandidates(
    const std::uint16_t* weights, const float* hidden,
    const std::uint32_t* suppressed, std::uint32_t suppressed_count,
    std::uint64_t rows, OutputHeadCandidate* candidates,
    float* diagnostic_logits, cudaStream_t stream) {
  if (rows == 0U || rows > kMaximumMtpVerifyTokens) {
    return Status(StatusCode::kInvalidArgument,
                  "batched output-head row count is invalid");
  }
  if (rows == 3U) {
    FusedOutputHeadFixedBatchCandidatesKernel<3U>
        <<<kOutputHeadCandidateBlocks, kThreads, 0, stream>>>(
            weights, hidden, suppressed, suppressed_count, candidates,
            diagnostic_logits);
  } else {
    FusedOutputHeadBatchCandidatesKernel<<<kOutputHeadCandidateBlocks,
                                           kThreads, 0, stream>>>(
        weights, hidden, suppressed, suppressed_count, rows, candidates,
        diagnostic_logits);
  }
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch batched output-head candidates", error);
}

Status LaunchOutputHeadCandidateArgmax(const OutputHeadCandidate* candidates,
                                       std::uint32_t* selected,
                                       cudaStream_t stream) {
  OutputHeadArgmaxKernel<false><<<1U, kThreads, 0, stream>>>(candidates, 1U,
                                                               selected);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch output-head candidate argmax", error);
}

Status LaunchOutputHeadBatchArgmax(const OutputHeadCandidate* candidates,
                                   std::uint64_t rows,
                                   std::uint32_t* selected,
                                   cudaStream_t stream) {
  if (rows == 0U || rows > kMaximumMtpVerifyTokens) {
    return Status(StatusCode::kInvalidArgument,
                  "batched output-head argmax row count is invalid");
  }
  OutputHeadArgmaxKernel<true><<<static_cast<unsigned>(rows), kThreads, 0,
                                  stream>>>(candidates, rows, selected);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch batched output-head argmax", error);
}

Status LaunchOutputHeadLogits(const std::uint16_t* weights, const float* hidden,
                              float* logits, cudaStream_t stream) {
  OutputHeadKernel<<<static_cast<unsigned>(kVocabulary), kThreads, 0, stream>>>(
      weights, hidden, logits);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess ? Status::Ok()
                              : CudaFailure("launch output-head logits", error);
}

Status LaunchLogitArgmax(const float* logits, const std::uint32_t* suppressed,
                         std::uint32_t suppressed_count,
                         std::uint32_t* selected, cudaStream_t stream) {
  ArgmaxKernel<<<1U, kThreads, 0, stream>>>(logits, suppressed,
                                              suppressed_count, selected);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess ? Status::Ok()
                              : CudaFailure("launch logit argmax", error);
}

}  // namespace gem16::internal
