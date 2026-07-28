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
};

struct alignas(16) MtpDeviceControl {
  MtpDeviceState current{};
  MtpDeviceState next{};
  std::uint32_t fixed_draft_tokens = 0U;
  std::uint32_t proposal_count = 0U;
  std::uint32_t transition_valid = 0U;
  std::uint32_t reserved = 0U;
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
  std::uint32_t stopped = 0U;
  std::uint32_t stop_token = 0U;
};

static_assert(std::is_trivially_copyable_v<MtpGroupTransaction>);
static_assert(offsetof(MtpGroupTransaction, control) % 16U == 0U);

[[nodiscard]] Status LaunchBuildMtpVerificationInputs(
    std::uint32_t input_token, const std::uint32_t* drafts,
    std::uint32_t proposal_count, std::uint32_t* inputs, cudaStream_t stream);

[[nodiscard]] Status LaunchBuildControlledMtpD2Inputs(
    const MtpDeviceControl* control, const std::uint32_t* drafts,
    std::uint32_t* inputs, DecodeControl* row_controls, cudaStream_t stream);

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

[[nodiscard]] Status LaunchAdvanceMtpD2Chain(
    MtpGroupTransaction* transaction, MtpChainResult* chain_result,
    std::uint32_t* output_tokens, std::uint32_t* proposed_tokens,
    cudaGraphConditionalHandle condition, cudaStream_t stream);

}  // namespace gem16::internal
