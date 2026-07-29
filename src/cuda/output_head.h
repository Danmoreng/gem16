#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

#include "cuda/layer/reference.h"
#include "gem16/status.h"

namespace gem16::internal {

constexpr unsigned kOutputHeadCandidateBlocks = 4096U;

struct OutputHeadCandidate {
  float value;
  std::uint32_t token;
};

[[nodiscard]] Status LaunchFusedOutputHeadCandidates(
    const std::uint16_t* weights, const float* hidden,
    const std::uint32_t* suppressed, std::uint32_t suppressed_count,
    const DecodeControl* control, OutputHeadCandidate* candidates,
    float* diagnostic_logits, cudaStream_t stream);

[[nodiscard]] Status LaunchFusedOutputHeadBatchCandidates(
    const std::uint16_t* weights, const float* hidden,
    const std::uint32_t* suppressed, std::uint32_t suppressed_count,
    std::uint64_t rows, OutputHeadCandidate* candidates,
    float* diagnostic_logits, cudaStream_t stream);

[[nodiscard]] Status LaunchOutputHeadCandidateArgmax(
    const OutputHeadCandidate* candidates, std::uint32_t* selected,
    cudaStream_t stream);

[[nodiscard]] Status LaunchOutputHeadBatchArgmax(
    const OutputHeadCandidate* candidates, std::uint64_t rows,
    std::uint32_t* selected, cudaStream_t stream);

[[nodiscard]] Status LaunchOutputHeadLogits(const std::uint16_t* weights,
                                            const float* hidden,
                                            float* logits,
                                            cudaStream_t stream);

[[nodiscard]] Status LaunchLogitArgmax(const float* logits,
                                       const std::uint32_t* suppressed,
                                       std::uint32_t suppressed_count,
                                       std::uint32_t* selected,
                                       cudaStream_t stream);

}  // namespace gem16::internal
