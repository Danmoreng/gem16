#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>

#include <cuda_runtime_api.h>

#include "gem16/status.h"

namespace gem16::internal {

struct MtpDeviceControl;

struct AssistantLayerBinding {
  bool global = false;
  std::uint64_t query_elements = 0;
  const std::uint16_t* input_norm = nullptr;
  const std::uint16_t* q_projection = nullptr;
  const std::uint16_t* q_norm = nullptr;
  const std::uint16_t* o_projection = nullptr;
  const std::uint16_t* post_attention_norm = nullptr;
  const std::uint16_t* pre_feedforward_norm = nullptr;
  const std::uint16_t* gate_projection = nullptr;
  const std::uint16_t* up_projection = nullptr;
  const std::uint16_t* down_projection = nullptr;
  const std::uint16_t* post_feedforward_norm = nullptr;
  const std::uint16_t* layer_scalar = nullptr;
};

struct AssistantBindings {
  const std::uint16_t* embedding = nullptr;
  const std::uint16_t* pre_projection = nullptr;
  const std::uint16_t* post_projection = nullptr;
  const std::uint16_t* final_norm = nullptr;
  std::array<AssistantLayerBinding, 4> layers{};
};

enum class AssistantKvCacheMode {
  kCheckpointFp8,
  kBf16,
};

struct AssistantSharedKvView {
  AssistantKvCacheMode mode = AssistantKvCacheMode::kCheckpointFp8;
  const std::uint8_t* key_fp8 = nullptr;
  const std::uint8_t* value_fp8 = nullptr;
  const float* key_bf16 = nullptr;
  const float* value_bf16 = nullptr;
  const std::uint16_t* key_scale_bf16 = nullptr;
  const std::uint16_t* value_scale_bf16 = nullptr;
  std::uint64_t tokens = 0;
  std::uint64_t capacity = 0;
  std::uint64_t first_slot = 0;
  std::uint64_t kv_heads = 0;
  std::uint64_t head_dimension = 0;
};

struct AssistantProposalContext {
  const std::uint16_t* target_embedding = nullptr;
  const float* target_hidden = nullptr;
  AssistantSharedKvView sliding_kv;
  AssistantSharedKvView full_kv;
  std::uint32_t input_token = 0;
  std::uint64_t position = 0;
};

// Owns the official BF16 MTP assistant in one independent, fixed-address
// device arena. The target model and its KV cache remain separate owners.
class AssistantModel {
 public:
  AssistantModel();
  AssistantModel(const AssistantModel&) = delete;
  AssistantModel& operator=(const AssistantModel&) = delete;
  ~AssistantModel();

  [[nodiscard]] Status Load(const std::filesystem::path& directory);
  [[nodiscard]] Status Prepare(std::uint64_t max_context);
  [[nodiscard]] Status GenerateDrafts(
      const AssistantProposalContext& context,
      std::span<std::uint32_t> draft_token_ids, cudaStream_t stream);
  [[nodiscard]] Status GenerateDraftsDevice(
      const AssistantProposalContext& context, std::uint32_t draft_count,
      cudaStream_t stream, const MtpDeviceControl* control = nullptr);
  [[nodiscard]] const std::uint32_t* device_draft_tokens() const;
  [[nodiscard]] bool loaded() const;
  [[nodiscard]] bool prepared() const;
  [[nodiscard]] std::uint64_t arena_bytes() const;
  [[nodiscard]] std::uint64_t workspace_bytes() const;
  [[nodiscard]] std::uint64_t source_bytes() const;
  [[nodiscard]] std::uint64_t tensor_count() const;
  [[nodiscard]] const AssistantBindings& bindings() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gem16::internal
