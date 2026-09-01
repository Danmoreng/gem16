#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>

#include <cuda_runtime_api.h>

#include "cuda/moe/reference.h"
#include "cuda/mtp/assistant.h"
#include "gem16/status.h"

namespace gem16::internal {

struct Gemma4Moe26BAssistantProposalContext {
  Gemma4MoeNvfp4Matrix target_embedding;
  const float* target_hidden = nullptr;
  AssistantSharedKvView sliding_kv;
  AssistantSharedKvView full_kv;
  std::uint32_t input_token = 0;
  std::uint64_t position = 0;
};

enum class Gemma4Moe26BAssistantContextPolicy {
  // Preserve the accepted M25/NVFP4 qualification boundary.
  kQualifiedM25,
  // Let the Trellis35 profile use the model's physical 262144-token limit.
  kTrellis35PhysicalCapacity,
};

// Dedicated M25 Assistant path. It is deliberately separate from the qualified
// 12B BF16 AssistantModel: the 26B variant has a 2816-wide backbone interface,
// two global KV heads and the fixed NVFP4/FP8 compiled storage contract.
class Gemma4Moe26BAssistantModel {
 public:
  Gemma4Moe26BAssistantModel();
  ~Gemma4Moe26BAssistantModel();
  Gemma4Moe26BAssistantModel(const Gemma4Moe26BAssistantModel&) = delete;
  Gemma4Moe26BAssistantModel& operator=(
      const Gemma4Moe26BAssistantModel&) = delete;
  Gemma4Moe26BAssistantModel(Gemma4Moe26BAssistantModel&&) noexcept;
  Gemma4Moe26BAssistantModel& operator=(
      Gemma4Moe26BAssistantModel&&) noexcept;

  [[nodiscard]] Status Load(const std::filesystem::path& directory);
  [[nodiscard]] Status Prepare(
      std::uint64_t max_context,
      Gemma4Moe26BAssistantContextPolicy context_policy);
  [[nodiscard]] Status GenerateDrafts(
      const Gemma4Moe26BAssistantProposalContext& context,
      std::span<std::uint32_t> draft_token_ids, cudaStream_t stream);
  [[nodiscard]] Status GenerateDraftsDevice(
      const Gemma4Moe26BAssistantProposalContext& context,
      std::uint32_t draft_count, cudaStream_t stream,
      const MtpDeviceControl* control = nullptr);
  // Explicit diagnostic copies for the BF16 whole-Assistant oracle. These
  // synchronize the supplied stream and are never used by recurring MTP.
  [[nodiscard]] Status CopyLastOracleState(
      std::span<float> concatenated_input, std::span<float> logits,
      cudaStream_t stream) const;

  [[nodiscard]] const std::uint32_t* device_draft_tokens() const;
  [[nodiscard]] bool loaded() const;
  [[nodiscard]] bool prepared() const;
  [[nodiscard]] std::uint64_t arena_bytes() const;
  [[nodiscard]] std::uint64_t workspace_bytes() const;
  [[nodiscard]] std::uint64_t source_bytes() const;
  [[nodiscard]] std::uint64_t tensor_count() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gem16::internal
