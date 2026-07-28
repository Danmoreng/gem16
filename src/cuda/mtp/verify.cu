#include "cuda/mtp/verify.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <string>
#include <utility>

namespace gem16::internal {
namespace {

constexpr unsigned kThreads = 256U;

Status CudaFailure(const char* operation, cudaError_t error) {
  return Status(StatusCode::kInternal,
                std::string(operation) + ": " + cudaGetErrorName(error) +
                    ": " + cudaGetErrorString(error));
}

template <typename T, bool kRestore>
__global__ void CopyCircularMtpKvKernel(
    T* cache_key, T* cache_value, T* compact_key, T* compact_value,
    std::uint64_t start_position, std::uint64_t tokens,
    std::uint64_t elements_per_token, std::uint64_t capacity) {
  const std::uint64_t total = tokens * elements_per_token;
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) return;
  const std::uint64_t token = index / elements_per_token;
  const std::uint64_t element = index % elements_per_token;
  const std::uint64_t cache_index =
      ((start_position + token) % capacity) * elements_per_token + element;
  if constexpr (kRestore) {
    cache_key[cache_index] = compact_key[index];
    cache_value[cache_index] = compact_value[index];
  } else {
    compact_key[index] = cache_key[cache_index];
    compact_value[index] = cache_value[cache_index];
  }
}

__global__ void SetMtpAttentionPositionKernel(DecodeControl* control,
                                              std::uint64_t position) {
  if (blockIdx.x == 0U && threadIdx.x == 0U) control->position = position;
}

__global__ void BuildMtpVerificationInputsKernel(
    std::uint32_t input_token, const std::uint32_t* drafts,
    std::uint32_t proposal_count, std::uint32_t* inputs) {
  const std::uint32_t index = threadIdx.x;
  if (blockIdx.x != 0U || index > proposal_count) return;
  inputs[index] = index == 0U ? input_token : drafts[index - 1U];
}

__global__ void AcceptMtpGroupKernel(
    const std::uint32_t* drafts, const std::uint32_t* verified,
    std::uint32_t proposal_count, const std::uint32_t* stop_tokens,
    std::uint32_t stop_count, MtpGroupResult* result) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) return;
  result->proposal_count = proposal_count;
  result->accepted_count = 0U;
  result->output_count = 0U;
  result->stop_token = 0U;
  result->stopped = 0U;
  for (std::uint32_t index = 0U; index < kMaximumMtpDraftTokens; ++index) {
    result->proposed[index] = index < proposal_count ? drafts[index] : 0U;
  }
  for (std::uint32_t index = 0U; index < kMaximumMtpVerifyTokens; ++index) {
    result->verified[index] = index <= proposal_count ? verified[index] : 0U;
  }
  while (result->accepted_count < proposal_count &&
         verified[result->accepted_count] == drafts[result->accepted_count]) {
    ++result->accepted_count;
  }
  result->output_count = result->accepted_count == proposal_count
                             ? proposal_count + 1U
                             : result->accepted_count + 1U;
  for (std::uint32_t index = 0U; index < result->output_count; ++index) {
    for (std::uint32_t stop = 0U; stop < stop_count; ++stop) {
      if (verified[index] == stop_tokens[stop]) {
        result->output_count = index + 1U;
        result->accepted_count = result->accepted_count < index
                                     ? result->accepted_count
                                     : index;
        result->stop_token = verified[index];
        result->stopped = 1U;
        return;
      }
    }
  }
}

template <typename T>
__global__ void CommitMtpKvKernel(
    const T* compact_key, const T* compact_value, T* cache_key,
    T* cache_value, std::uint64_t start_position,
    std::uint64_t elements_per_token, std::uint64_t capacity,
    const MtpGroupResult* result) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t total =
      static_cast<std::uint64_t>(result->output_count) * elements_per_token;
  if (index >= total) return;
  const std::uint64_t token = index / elements_per_token;
  const std::uint64_t element = index % elements_per_token;
  const std::uint64_t cache_index =
      ((start_position + token) % capacity) * elements_per_token + element;
  cache_key[cache_index] = compact_key[index];
  cache_value[cache_index] = compact_value[index];
}

__global__ void CommitMtpHiddenKernel(const float* verified_hidden,
                                      float* committed_hidden,
                                      std::uint64_t hidden,
                                      const MtpGroupResult* result) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= hidden) return;
  const std::uint64_t row = result->output_count - 1U;
  committed_hidden[index] = verified_hidden[row * hidden + index];
}

template <typename T>
Status LaunchCommitMtpKv(const T* compact_key, const T* compact_value,
                         T* cache_key, T* cache_value,
                         std::uint64_t start_position,
                         std::uint64_t elements_per_token,
                         std::uint64_t capacity,
                         const MtpGroupResult* result, cudaStream_t stream,
                         const char* label) {
  const std::uint64_t maximum_elements =
      kMaximumMtpVerifyTokens * elements_per_token;
  if (maximum_elements >
      std::numeric_limits<unsigned>::max() * static_cast<std::uint64_t>(kThreads)) {
    return Status(StatusCode::kInvalidArgument,
                  "MTP KV commit grid exceeds CUDA limits");
  }
  const unsigned blocks = static_cast<unsigned>(
      (maximum_elements + kThreads - 1U) / kThreads);
  CommitMtpKvKernel<<<blocks, kThreads, 0, stream>>>(
      compact_key, compact_value, cache_key, cache_value, start_position,
      elements_per_token, capacity, result);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess ? Status::Ok() : CudaFailure(label, error);
}

template <typename T>
Status LaunchCopyCircularMtpKv(T* cache_key, T* cache_value, T* compact_key,
                               T* compact_value,
                               std::uint64_t start_position,
                               std::uint64_t tokens,
                               std::uint64_t elements_per_token,
                               std::uint64_t capacity, bool restore,
                               cudaStream_t stream, const char* label) {
  const std::uint64_t elements = tokens * elements_per_token;
  if (elements >
      std::numeric_limits<unsigned>::max() * static_cast<std::uint64_t>(kThreads)) {
    return Status(StatusCode::kInvalidArgument,
                  "MTP circular KV copy grid exceeds CUDA limits");
  }
  const unsigned blocks =
      static_cast<unsigned>((elements + kThreads - 1U) / kThreads);
  if (restore) {
    CopyCircularMtpKvKernel<T, true><<<blocks, kThreads, 0, stream>>>(
        cache_key, cache_value, compact_key, compact_value, start_position,
        tokens, elements_per_token, capacity);
  } else {
    CopyCircularMtpKvKernel<T, false><<<blocks, kThreads, 0, stream>>>(
        cache_key, cache_value, compact_key, compact_value, start_position,
        tokens, elements_per_token, capacity);
  }
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess ? Status::Ok() : CudaFailure(label, error);
}

}  // namespace

Status LaunchBuildMtpVerificationInputs(
    std::uint32_t input_token, const std::uint32_t* drafts,
    std::uint32_t proposal_count, std::uint32_t* inputs, cudaStream_t stream) {
  BuildMtpVerificationInputsKernel<<<1U, kMaximumMtpVerifyTokens, 0, stream>>>(
      input_token, drafts, proposal_count, inputs);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("build MTP verification token IDs", error);
}

Status LaunchAcceptMtpGroup(
    const std::uint32_t* drafts, const std::uint32_t* verified,
    std::uint32_t proposal_count, const std::uint32_t* stop_tokens,
    std::uint32_t stop_count, MtpGroupResult* result, cudaStream_t stream) {
  AcceptMtpGroupKernel<<<1U, 1U, 0, stream>>>(
      drafts, verified, proposal_count, stop_tokens, stop_count, result);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess ? Status::Ok()
                              : CudaFailure("launch GPU MTP acceptance", error);
}

Status LaunchCommitMtpKvFp8(
    const std::uint8_t* compact_key, const std::uint8_t* compact_value,
    std::uint8_t* cache_key, std::uint8_t* cache_value,
    std::uint64_t start_position, std::uint64_t elements_per_token,
    std::uint64_t capacity, const MtpGroupResult* result, cudaStream_t stream) {
  return LaunchCommitMtpKv(compact_key, compact_value, cache_key, cache_value,
                           start_position, elements_per_token, capacity, result,
                           stream, "launch GPU MTP FP8 KV commit");
}

Status LaunchCommitMtpKvBf16(
    const float* compact_key, const float* compact_value, float* cache_key,
    float* cache_value, std::uint64_t start_position,
    std::uint64_t elements_per_token, std::uint64_t capacity,
    const MtpGroupResult* result, cudaStream_t stream) {
  return LaunchCommitMtpKv(compact_key, compact_value, cache_key, cache_value,
                           start_position, elements_per_token, capacity, result,
                           stream, "launch GPU MTP BF16 KV commit");
}

Status LaunchCopyCircularMtpKvFp8(
    std::uint8_t* cache_key, std::uint8_t* cache_value,
    std::uint8_t* compact_key, std::uint8_t* compact_value,
    std::uint64_t start_position, std::uint64_t tokens,
    std::uint64_t elements_per_token, std::uint64_t capacity, bool restore,
    cudaStream_t stream) {
  return LaunchCopyCircularMtpKv(cache_key, cache_value, compact_key,
                                 compact_value, start_position, tokens,
                                 elements_per_token, capacity, restore, stream,
                                 restore ? "restore local speculative FP8 KV"
                                         : "backup local speculative FP8 KV");
}

Status LaunchCopyCircularMtpKvBf16(
    float* cache_key, float* cache_value, float* compact_key,
    float* compact_value, std::uint64_t start_position, std::uint64_t tokens,
    std::uint64_t elements_per_token, std::uint64_t capacity, bool restore,
    cudaStream_t stream) {
  return LaunchCopyCircularMtpKv(cache_key, cache_value, compact_key,
                                 compact_value, start_position, tokens,
                                 elements_per_token, capacity, restore, stream,
                                 restore ? "restore local speculative BF16 KV"
                                         : "backup local speculative BF16 KV");
}

Status LaunchSetMtpAttentionPosition(DecodeControl* control,
                                     std::uint64_t position,
                                     cudaStream_t stream) {
  SetMtpAttentionPositionKernel<<<1U, 1U, 0, stream>>>(control, position);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("set MTP attention position", error);
}

Status LaunchCommitMtpHidden(const float* verified_hidden,
                             float* committed_hidden, std::uint64_t hidden,
                             const MtpGroupResult* result, cudaStream_t stream) {
  const unsigned blocks =
      static_cast<unsigned>((hidden + kThreads - 1U) / kThreads);
  CommitMtpHiddenKernel<<<blocks, kThreads, 0, stream>>>(
      verified_hidden, committed_hidden, hidden, result);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch GPU MTP hidden-state commit", error);
}

}  // namespace gem16::internal
