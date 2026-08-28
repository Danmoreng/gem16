#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime_api.h>

#include "gem16/sampling.h"
#include "gem16/status.h"

namespace gem16::internal {

struct DecodeControl;

[[nodiscard]] Result<std::size_t> SamplingWorkspaceBytes(
    std::uint32_t vocabulary, cudaStream_t stream);

[[nodiscard]] Status LaunchMarkRepetitionTokens(
    const std::uint32_t* tokens, std::uint64_t count, std::uint32_t* mask,
    cudaStream_t stream);
[[nodiscard]] Status LaunchMarkRepetitionToken(
    std::uint32_t token, std::uint32_t* mask, cudaStream_t stream);
[[nodiscard]] Status LaunchMarkControlledRepetitionToken(
    const DecodeControl* control, std::uint32_t* mask, cudaStream_t stream);

// Build one speculative repetition-history row per verification input. Row r
// contains the committed base history plus inputs [0, r]. The caller commits
// only the row selected by the target-verified output count.
[[nodiscard]] Status LaunchBuildSpeculativeRepetitionMasks(
    const std::uint32_t* base_mask, const std::uint32_t* verification_inputs,
    std::uint32_t rows, std::uint32_t mask_words, std::uint32_t* row_masks,
    cudaStream_t stream);
[[nodiscard]] Status LaunchCommitSpeculativeRepetitionMask(
    const std::uint32_t* row_masks, std::uint32_t mask_words,
    const std::uint32_t* committed_row_count, std::uint32_t* base_mask,
    cudaStream_t stream);

// logits is both the unmodified input and the sorted-key output. The repetition
// mask is a ceil(vocabulary/32)-word atomic bitset. Callers that capture
// diagnostics must enqueue that copy before this launch. Every buffer,
// including radix_sort_workspace, is caller-owned and preallocated.
[[nodiscard]] Status LaunchSampleToken(
    float* logits, float* adjusted_logits, double* cumulative_probabilities,
    std::uint32_t* token_ids, std::uint32_t* sorted_token_ids,
    std::uint32_t* repetition_mask,
    const std::uint32_t* suppressed, std::uint32_t suppressed_count,
    std::uint32_t vocabulary, const SamplingOptions& options,
    std::uint64_t step, const DecodeControl* control, std::uint32_t* selected,
    void* algorithm_workspace, std::size_t algorithm_workspace_bytes,
    cudaStream_t stream);

// Variant with a separate radix-sort output, used when source_logits belongs
// to a verifier batch that must remain intact for commit/diagnostics.
[[nodiscard]] Status LaunchSampleTokenFromLogits(
    float* source_logits, float* sorted_logits, float* adjusted_logits,
    double* cumulative_probabilities, std::uint32_t* token_ids,
    std::uint32_t* sorted_token_ids, std::uint32_t* repetition_mask,
    const std::uint32_t* suppressed, std::uint32_t suppressed_count,
    std::uint32_t vocabulary, const SamplingOptions& options,
    std::uint64_t step, const DecodeControl* control, std::uint32_t* selected,
    void* algorithm_workspace, std::size_t algorithm_workspace_bytes,
    cudaStream_t stream);

// Sample directly from a verifier-logit row while applying the model softcap
// in place and recording the same finite-state diagnostic as the ordinary
// softcap/argmax path. sorted_logits is separate because radix sort overwrites
// its output; this avoids staging the full verifier row before sampling.
[[nodiscard]] Status LaunchSampleTokenSoftcapInPlace(
    float* source_logits, float* sorted_logits, float* adjusted_logits,
    double* cumulative_probabilities, std::uint32_t* token_ids,
    std::uint32_t* sorted_token_ids, std::uint32_t* repetition_mask,
    const std::uint32_t* suppressed, std::uint32_t suppressed_count,
    std::uint32_t vocabulary, float softcap, int* all_finite,
    const SamplingOptions& options, std::uint64_t step,
    const DecodeControl* control, std::uint32_t* selected,
    void* algorithm_workspace, std::size_t algorithm_workspace_bytes,
    cudaStream_t stream);

}  // namespace gem16::internal
