#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <cuda_runtime_api.h>

#include "cuda/layer/reference.h"
#include "gem16/status.h"

namespace gem16::internal {

constexpr std::uint64_t kMaximumMtpDraftTokens = 4U;
constexpr std::uint64_t kMaximumMtpVerifyTokens =
    kMaximumMtpDraftTokens + 1U;
constexpr std::uint32_t kMaximumMtpKvCommitLayers = 30U;
constexpr std::uint64_t kMtpStreamingRingCapacity = 256U;
constexpr std::uint32_t kMaximumThinkingOpenTokens = 8U;

struct alignas(64) MtpStreamingRing {
  unsigned long long producer = 0U;
  unsigned long long consumer = 0U;
  unsigned long long cancelled = 0U;
  unsigned long long backpressure_events = 0U;
  std::array<std::uint32_t, kMtpStreamingRingCapacity> tokens{};
};

struct MtpGroupResult {
  std::array<std::uint32_t, kMaximumMtpDraftTokens> proposed{};
  std::array<std::uint32_t, kMaximumMtpVerifyTokens> verified{};
  std::uint32_t proposal_count = 0U;
  std::uint32_t accepted_count = 0U;
  std::uint32_t output_count = 0U;
  std::uint32_t stop_token = 0U;
  std::uint32_t stopped = 0U;
};

struct MtpDeviceState {
  std::uint32_t input_token = 0U;
  std::uint32_t stopped = 0U;
  std::uint32_t stop_token = 0U;
  std::uint32_t reserved = 0U;
  std::uint64_t processed_position = 0U;
  std::uint64_t remaining_output_capacity = 0U;
  std::uint64_t output_write_position = 0U;
  std::uint64_t sampling_step = 0U;
};

struct MtpReasoningState {
  std::array<std::uint32_t, kMaximumThinkingOpenTokens> open_token_ids{};
  std::uint32_t close_token_id = 0U;
  std::uint32_t open_token_count = 0U;
  std::uint32_t open_match_length = 0U;
  std::uint32_t enabled = 0U;
  std::uint32_t started = 0U;
  std::uint32_t complete = 0U;
  std::uint32_t in_reasoning = 0U;
  std::uint32_t budget_forced = 0U;
  std::uint32_t reserved = 0U;
  std::uint64_t reasoning_token_count = 0U;
  std::uint64_t max_reasoning_tokens = 0U;
};

struct alignas(16) MtpDeviceControl {
  MtpDeviceState current{};
  MtpDeviceState next{};
  MtpReasoningState reasoning{};
  std::uint32_t fixed_draft_tokens = 0U;
  std::uint32_t proposal_count = 0U;
  std::uint32_t transition_valid = 0U;
  std::uint32_t sampling_enabled = 0U;
};

struct alignas(16) MtpGroupTransaction {
  MtpGroupResult result{};
  std::array<std::byte, 8U> control_alignment_padding{};
  MtpDeviceControl control{};
};

struct alignas(16) MtpChainResult {
  std::uint64_t output_count = 0U;
  std::uint64_t proposed_count = 0U;
  std::uint64_t accepted_count = 0U;
  std::uint64_t rejected_count = 0U;
  std::uint64_t group_count = 0U;
  std::uint64_t ordinary_tail_count = 0U;
  // Sticky device-side fault counter. Every verifier group and ordinary tail
  // latches non-finite Target logits/router state before the chain advances.
  std::uint64_t non_finite_step_count = 0U;
  std::uint64_t reasoning_ordinary_count = 0U;
  std::uint64_t reasoning_token_count = 0U;
  std::uint32_t stopped = 0U;
  std::uint32_t stop_token = 0U;
  std::uint32_t reasoning_complete = 0U;
  std::uint32_t reasoning_budget_forced = 0U;
};

struct MtpKvCommitLayer {
  const std::uint8_t* compact_key = nullptr;
  const std::uint8_t* compact_value = nullptr;
  std::uint8_t* cache_key = nullptr;
  std::uint8_t* cache_value = nullptr;
  std::uint64_t elements_per_token = 0U;
  std::uint64_t capacity = 0U;
};

static_assert(std::is_trivially_copyable_v<MtpGroupTransaction>);
static_assert(offsetof(MtpGroupTransaction, control) % 16U == 0U);

[[nodiscard]] Status LaunchBuildMtpVerificationInputs(
    std::uint32_t input_token, const std::uint32_t* drafts,
    std::uint32_t proposal_count, std::uint32_t* inputs, cudaStream_t stream);

[[nodiscard]] Status LaunchBuildControlledMtpD2Inputs(
    const MtpDeviceControl* control, const std::uint32_t* drafts,
    std::uint32_t* inputs, DecodeControl* row_controls,
    std::uint32_t suppressed_token_count, cudaStream_t stream);

[[nodiscard]] Status LaunchBuildControlledMtpInputs(
    const MtpDeviceControl* control, const std::uint32_t* drafts,
    std::uint32_t draft_count, std::uint32_t* inputs,
    DecodeControl* row_controls, std::uint32_t suppressed_token_count,
    cudaStream_t stream);

[[nodiscard]] Status LaunchAcceptMtpGroup(
    const std::uint32_t* drafts, const std::uint32_t* verified,
    std::uint32_t proposal_count, const std::uint32_t* stop_tokens,
    std::uint32_t stop_count, MtpGroupResult* result, MtpDeviceControl* control,
    cudaStream_t stream);

[[nodiscard]] Status LaunchCommitMtpKvFp8(
    const std::uint8_t* compact_key, const std::uint8_t* compact_value,
    std::uint8_t* cache_key, std::uint8_t* cache_value,
    std::uint64_t start_position, std::uint64_t elements_per_token,
    std::uint64_t capacity, const MtpGroupResult* result, cudaStream_t stream);

[[nodiscard]] Status LaunchCommitMtpKvBf16(
    const float* compact_key, const float* compact_value, float* cache_key,
    float* cache_value, std::uint64_t start_position,
    std::uint64_t elements_per_token, std::uint64_t capacity,
    const MtpGroupResult* result, cudaStream_t stream);

[[nodiscard]] Status LaunchCopyCircularMtpKvFp8(
    std::uint8_t* cache_key, std::uint8_t* cache_value,
    std::uint8_t* compact_key, std::uint8_t* compact_value,
    std::uint64_t start_position, std::uint64_t tokens,
    std::uint64_t elements_per_token, std::uint64_t capacity, bool restore,
    cudaStream_t stream);

[[nodiscard]] Status LaunchCopyCircularMtpKvBf16(
    float* cache_key, float* cache_value, float* compact_key,
    float* compact_value, std::uint64_t start_position, std::uint64_t tokens,
    std::uint64_t elements_per_token, std::uint64_t capacity, bool restore,
    cudaStream_t stream);

[[nodiscard]] Status LaunchCopyCircularMtpKvFp8ControlledD2(
    std::uint8_t* cache_key, std::uint8_t* cache_value,
    std::uint8_t* compact_key, std::uint8_t* compact_value,
    std::uint64_t elements_per_token, std::uint64_t capacity, bool restore,
    const MtpDeviceControl* control, cudaStream_t stream);

[[nodiscard]] Status LaunchCopyCircularMtpKvFp8Controlled(
    std::uint8_t* cache_key, std::uint8_t* cache_value,
    std::uint8_t* compact_key, std::uint8_t* compact_value,
    std::uint32_t tokens, std::uint64_t elements_per_token,
    std::uint64_t capacity, bool restore,
    const MtpDeviceControl* control, cudaStream_t stream);

[[nodiscard]] Status LaunchCompactRestoreCircularMtpKvFp8Controlled(
    std::uint8_t* cache_key, std::uint8_t* cache_value,
    std::uint8_t* compact_key, std::uint8_t* compact_value,
    const std::uint8_t* backup_key, const std::uint8_t* backup_value,
    std::uint32_t tokens, std::uint64_t elements_per_token,
    std::uint64_t capacity, const MtpDeviceControl* control,
    cudaStream_t stream);

[[nodiscard]] Status LaunchSetMtpAttentionPosition(DecodeControl* control,
                                                    std::uint64_t position,
                                                    cudaStream_t stream);

[[nodiscard]] Status LaunchCommitMtpHidden(const float* verified_hidden,
                                           float* committed_hidden,
                                           std::uint64_t hidden,
                                           const MtpGroupResult* result,
                                           cudaStream_t stream);

[[nodiscard]] Status LaunchCommitMtpKvFp8ControlledD2(
    const std::uint8_t* compact_key, const std::uint8_t* compact_value,
    std::uint8_t* cache_key, std::uint8_t* cache_value,
    std::uint64_t elements_per_token, std::uint64_t capacity,
    const MtpGroupResult* result, const MtpDeviceControl* control,
    cudaStream_t stream);

[[nodiscard]] Status LaunchCommitMtpKvFp8Controlled(
    const std::uint8_t* compact_key, const std::uint8_t* compact_value,
    std::uint8_t* cache_key, std::uint8_t* cache_value,
    std::uint64_t elements_per_token, std::uint64_t capacity,
    const MtpGroupResult* result, const MtpDeviceControl* control,
    cudaStream_t stream);

[[nodiscard]] Status LaunchCommitMtpKvFp8ControlledLayers(
    const MtpKvCommitLayer* layers, std::uint32_t layer_count,
    const MtpGroupResult* result, const MtpDeviceControl* control,
    cudaStream_t stream);

[[nodiscard]] Status LaunchAdvanceMtpD2Chain(
    MtpGroupTransaction* transaction, MtpChainResult* chain_result,
    std::uint32_t* output_tokens, std::uint32_t* proposed_tokens,
    MtpStreamingRing* streaming_ring, cudaStream_t stream);

[[nodiscard]] Status LaunchAdvanceMtpChain(
    MtpGroupTransaction* transaction, MtpChainResult* chain_result,
    std::uint32_t* output_tokens, std::uint32_t* proposed_tokens,
    MtpStreamingRing* streaming_ring, cudaStream_t stream);

[[nodiscard]] Status LaunchSelectMtpChainBranch(
    const MtpGroupTransaction* transaction,
    cudaGraphConditionalHandle d2_condition,
    cudaGraphConditionalHandle ordinary_condition, cudaStream_t stream);

[[nodiscard]] Status LaunchContinueMtpChain(
    const MtpGroupTransaction* transaction,
    const MtpStreamingRing* streaming_ring,
    cudaGraphConditionalHandle loop_condition, cudaStream_t stream);

[[nodiscard]] Status LaunchInitializeMtpOrdinaryTail(
    const MtpGroupTransaction* transaction, DecodeControl* decode_control,
    std::uint32_t suppressed_token_count, cudaStream_t stream);

[[nodiscard]] Status LaunchFinalizeMtpOrdinaryTail(
    const std::uint32_t* selected, const std::uint32_t* stop_tokens,
    std::uint32_t stop_count, MtpGroupTransaction* transaction,
    MtpChainResult* chain_result, std::uint32_t* output_tokens,
    MtpStreamingRing* streaming_ring, cudaStream_t stream);

}  // namespace gem16::internal
