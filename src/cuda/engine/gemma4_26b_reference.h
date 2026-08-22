#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>

#include "gem16/status.h"
#include "gem16/sampling.h"

namespace gem16::internal {

enum class Gemma4Moe26BBackend {
  kReference,
  kSm120MoeHead,
  kSm120Integrated,
};

struct Gemma4Moe26BReferencePrediction {
  std::uint32_t token = 0;
  float logit = 0.0F;
  bool all_logits_finite = false;
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
  [[nodiscard]] Result<std::uint32_t> SelectToken();

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

 private:
  struct Impl;
  explicit Gemma4Moe26BReferenceEngine(std::unique_ptr<Impl> implementation);
  std::unique_ptr<Impl> implementation_;
};

}  // namespace gem16::internal
