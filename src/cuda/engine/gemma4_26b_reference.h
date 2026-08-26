#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>

#include "gem16/status.h"
#include "gem16/sampling.h"
#include "cuda/moe/reference.h"
#include "cuda/mtp/verify.h"

namespace gem16::internal {

enum class Gemma4Moe26BBackend {
  kReference,
  kSm120MoeHead,
  kSm120Integrated,
};

enum class Gemma4Moe26BMtpVerifierBackend {
  kExactDecodeParent,
  kBatchedAttention,
  kBatchedMoe,
  kExactSharedBatchedMoe,
  kFullyBatched,
};

struct Gemma4Moe26BReferencePrediction {
  std::uint32_t token = 0;
  float logit = 0.0F;
  bool all_logits_finite = false;
};

// Source-backed execution observations for qualification. These counters are
// advanced only after the corresponding native launch path succeeds. They are
// intentionally separate from capability claims and from external SASS or
// process-memory evidence.
struct Gemma4Moe26BExecutionEvidence {
  bool integrated_native_backend = false;
  bool decode_graph_ready = false;
  bool tensor_core_prefill_router = false;
  std::uint64_t prefill_calls = 0;
  std::uint64_t prefill_chunks = 0;
  std::uint64_t decode_graph_launches = 0;
  std::uint64_t token_selections = 0;
  std::uint64_t sliding_ring_wraps = 0;
  std::uint64_t maximum_global_position_exclusive = 0;
  std::uint64_t fallback_count = 0;
  std::uint64_t recurring_allocation_count = 0;
};

// M13 reference plus initialization-selected M16/M17 execution profiles and
// M21 long-context qualification. The context extent is validated against the
// model's declared maximum and the M12 262144-token attention/KV ceiling.
// Initialization binds one immutable M08 arena, all 30 layers, one
// partitioned FP8 K/V arena and fixed decode/prefill workspaces. Recurring
// execution performs no allocation, filesystem access, JIT, repacking, host
// expert routing, or precision fallback.
class Gemma4Moe26BReferenceEngine {
 public:
  Gemma4Moe26BReferenceEngine();
  ~Gemma4Moe26BReferenceEngine();
  Gemma4Moe26BReferenceEngine(const Gemma4Moe26BReferenceEngine&) = delete;
  Gemma4Moe26BReferenceEngine& operator=(
      const Gemma4Moe26BReferenceEngine&) = delete;
  Gemma4Moe26BReferenceEngine(Gemma4Moe26BReferenceEngine&&) noexcept;
  Gemma4Moe26BReferenceEngine& operator=(
      Gemma4Moe26BReferenceEngine&&) noexcept;

  [[nodiscard]] static Result<Gemma4Moe26BReferenceEngine> Create(
      const std::filesystem::path& model_directory,
      std::uint64_t context_tokens = 32768U, int device = 0,
      Gemma4Moe26BBackend backend = Gemma4Moe26BBackend::kReference);

  [[nodiscard]] Status Reset();
  [[nodiscard]] Status ForwardToken(std::uint32_t token);
  [[nodiscard]] Status PrefillTokens(std::span<const std::uint32_t> tokens);
  [[nodiscard]] Result<Gemma4Moe26BReferencePrediction> Prediction();
  // Product token selection is configured once per resident session. All
  // device buffers are reserved by Create; recurring selection performs no
  // allocation and preserves the existing Gemma sampling semantics.
  [[nodiscard]] Status ConfigureTokenSelection(
      const SamplingOptions& options,
      std::span<const std::uint32_t> suppressed_token_ids);
  // Initialization-time precision/dispatch selection. It is immutable after
  // the first prefill/decode token so one session cannot silently mix router
  // arithmetic or leak a process-global diagnostic choice.
  [[nodiscard]] Status ConfigurePrefillRouter(
      Gemma4MoePrefillRouter router);
  [[nodiscard]] Result<std::uint32_t> SelectToken();

  // M25 initialization and proposal hook. The Assistant owns only its fixed
  // compiled weights/workspace and aliases the target's final hidden state and
  // final sliding/full FP8 KV views.
  [[nodiscard]] Status LoadMtpAssistant(
      const std::filesystem::path& assistant_directory);
  // Configure Target stop tokens before the fixed MTP graphs are captured.
  // Stop tokens are distinct from sampling-suppressed tokens: they may be
  // emitted, then terminate the device-controlled chain.
  [[nodiscard]] Status ConfigureMtpStopTokens(
      std::span<const std::uint32_t> stop_token_ids);
  [[nodiscard]] Status GenerateMtpAssistantDrafts(
      std::span<std::uint32_t> draft_token_ids);
  [[nodiscard]] Status GenerateMtpAssistantDraftsForPending(
      std::uint32_t pending_token,
      std::span<std::uint32_t> draft_token_ids);
  // Production M25 group primitive. The caller supplies the pending Target
  // token (already emitted but not yet processed). The Assistant proposes from
  // the last committed Target hidden/KV state, then one multi-row Target batch
  // verifies [pending, drafts...] and commits only the accepted transaction.
  // The returned verified tokens are the only tokens callers may emit.
  [[nodiscard]] Status RunMtpAssistantGroup(
      std::uint32_t pending_token, std::uint32_t proposal_count,
      MtpGroupResult* host_result);
  // Fixed-D graph-chain execution. The caller has already emitted
  // pending_token; output_token_ids receives exactly the requested number of
  // subsequent Target-verified tokens in one device-controlled graph replay.
  [[nodiscard]] Status RunFixedMtpGraphChain(
      std::uint32_t pending_token, std::uint32_t draft_count,
      std::span<std::uint32_t> output_token_ids,
      MtpChainResult* host_result);
  [[nodiscard]] Status ConfigureMtpVerifierBackend(
      Gemma4Moe26BMtpVerifierBackend backend);
  [[nodiscard]] bool mtp_assistant_loaded() const;
  [[nodiscard]] std::uint64_t mtp_assistant_weight_bytes() const;
  [[nodiscard]] std::uint64_t mtp_assistant_workspace_bytes() const;
  [[nodiscard]] bool mtp_group_graph_prepared(
      std::uint32_t draft_count) const;
  [[nodiscard]] std::uint64_t mtp_group_graph_device_bytes() const;
  [[nodiscard]] std::uint64_t mtp_group_graph_launches() const;
  [[nodiscard]] bool mtp_chain_graph_prepared(
      std::uint32_t draft_count) const;
  [[nodiscard]] std::uint64_t mtp_chain_graph_device_bytes(
      std::uint32_t draft_count) const;
  [[nodiscard]] std::uint64_t mtp_chain_graph_launches(
      std::uint32_t draft_count) const;
  [[nodiscard]] float last_mtp_verification_min_margin() const;
  // One-token diagnostic export used only to compare the compiled Assistant
  // with Google's immutable BF16 source checkpoint. Cache bytes remain in the
  // target's FP8 representation and the four BF16 scale bit patterns are
  // copied separately.
  [[nodiscard]] Status CopyMtpAssistantOracleInputs(
      std::span<float> concatenated_input, std::span<float> assistant_logits,
      std::span<std::uint8_t> sliding_key,
      std::span<std::uint8_t> sliding_value,
      std::span<std::uint8_t> full_key,
      std::span<std::uint8_t> full_value,
      std::span<std::uint16_t> kv_scale_bf16_bits);

  // Diagnostic copies are explicit synchronization points and never occur
  // implicitly in ForwardToken. Callers provide fixed-size host storage.
  [[nodiscard]] Status CopyLogits(std::span<float> output);
  [[nodiscard]] Status CopyLayerOutput(std::uint32_t layer,
                                       std::span<float> output);
  [[nodiscard]] Status CopyRouterProbabilities(std::uint32_t layer,
                                               std::span<float> output);
  [[nodiscard]] Status CopyRouterTopIds(std::uint32_t layer,
                                        std::span<std::uint32_t> output);

  [[nodiscard]] std::uint64_t position() const;
  [[nodiscard]] std::uint64_t context_capacity() const;
  [[nodiscard]] std::uint64_t weight_arena_bytes() const;
  [[nodiscard]] std::uint64_t kv_cache_bytes() const;
  [[nodiscard]] std::uint64_t workspace_bytes() const;
  [[nodiscard]] std::uint64_t sliding_cache_capacity() const;
  [[nodiscard]] std::uint64_t prefill_chunk_count() const;
  [[nodiscard]] std::uint64_t minimum_prefill_chunk_tokens() const;
  [[nodiscard]] Gemma4Moe26BExecutionEvidence execution_evidence() const;

 private:
  struct Impl;
  explicit Gemma4Moe26BReferenceEngine(std::unique_ptr<Impl> implementation);
  std::unique_ptr<Impl> implementation_;
};

}  // namespace gem16::internal
