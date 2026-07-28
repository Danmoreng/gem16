#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>

#include "cuda/mtp/verify.h"
#include "gem16/engine.h"
#include "gem16/sampling.h"
#include "gem16/status.h"

namespace gem16 {

class InferenceEngine {
 public:
  InferenceEngine();
  InferenceEngine(const InferenceEngine&) = delete;
  InferenceEngine& operator=(const InferenceEngine&) = delete;
  InferenceEngine(InferenceEngine&&) noexcept;
  InferenceEngine& operator=(InferenceEngine&&) noexcept;
  ~InferenceEngine();

  [[nodiscard]] Status Initialize(
      const std::filesystem::path& model_directory,
      std::uint64_t max_context, KvCacheMode kv_cache_mode,
      const SamplingOptions& sampling = {},
      const std::filesystem::path& assistant_model_directory = {},
      std::uint32_t mtp_draft_tokens = 0U);
  [[nodiscard]] Result<std::uint32_t> Forward(
      std::uint32_t token, std::uint64_t position, bool select_token,
      std::span<float> host_logits = {}, std::span<float> host_state = {});
  [[nodiscard]] Result<std::uint32_t> Prefill(
      std::span<const std::uint32_t> token_ids,
      std::span<float> host_logits = {});
  [[nodiscard]] Result<std::uint32_t> PrefillAt(
      std::span<const std::uint32_t> token_ids,
      std::uint64_t start_position, std::span<float> host_logits = {});
  [[nodiscard]] Status GenerateAssistantDraftsDevice(
      std::uint32_t input_token, std::uint64_t processed_position,
      std::uint32_t draft_count);
  [[nodiscard]] Status PrepareMtpDeviceControl(
      std::uint32_t input_token, std::uint64_t processed_position,
      std::uint64_t remaining_output_capacity,
      std::uint64_t output_write_position, bool stopped,
      std::uint32_t stop_token);
  [[nodiscard]] Status VerifyAcceptCommitAssistantBatch(
      std::uint32_t input_token, std::uint64_t start_position,
      std::uint32_t proposal_count, internal::MtpGroupResult* host_result);
  [[nodiscard]] Status ExecuteFixedD2GraphGroup(
      std::uint32_t input_token, std::uint64_t start_position,
      internal::MtpGroupResult* host_result);
  [[nodiscard]] Status PrepareFixedD2Graph();
  [[nodiscard]] Status ExecuteFixedD2GraphChain(
      internal::MtpChainResult* host_result);
  [[nodiscard]] const std::uint32_t* mtp_chain_outputs() const;
  [[nodiscard]] const std::uint32_t* mtp_chain_proposals() const;
  [[nodiscard]] Status CheckMtpDeviceControlParity(
      std::uint32_t input_token, std::uint64_t processed_position,
      std::uint64_t remaining_output_capacity,
      std::uint64_t output_write_position, bool stopped,
      std::uint32_t stop_token) const;
  [[nodiscard]] Status ResetCache();
  [[nodiscard]] Status SetSampling(const SamplingOptions& options);
  [[nodiscard]] Status SetSuppressedTokens(
      std::span<const std::uint32_t> tokens);
  [[nodiscard]] Status SetMtpStopTokens(
      std::span<const std::uint32_t> tokens);

  [[nodiscard]] std::uint64_t weight_bytes() const;
  [[nodiscard]] bool assistant_loaded() const;
  [[nodiscard]] std::uint64_t assistant_source_bytes() const;
  [[nodiscard]] std::uint64_t assistant_weight_bytes() const;
  [[nodiscard]] std::uint64_t assistant_device_memory_delta_bytes() const;
  [[nodiscard]] std::uint64_t assistant_tensor_count() const;
  [[nodiscard]] std::uint64_t assistant_workspace_bytes() const;
  [[nodiscard]] std::uint64_t cache_bytes() const;
  [[nodiscard]] std::uint64_t workspace_bytes() const;
  [[nodiscard]] std::uint64_t decode_graph_device_bytes() const;
  [[nodiscard]] std::uint64_t prefill_chunk_tokens() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gem16
