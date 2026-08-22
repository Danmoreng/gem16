#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>

#include "gem16/status.h"

namespace gem16::internal {

enum class Gemma4Moe26BBackend {
  kReference,
  kSm120MoeHead,
};

struct Gemma4Moe26BReferencePrediction {
  std::uint32_t token = 0;
  float logit = 0.0F;
  bool all_logits_finite = false;
};

// M13 correctness-only full-model path. Initialization binds one immutable
// M08 arena, all 30 M11/M12 layers, one partitioned FP8 K/V arena and one
// fixed workspace. ForwardToken performs no allocation, filesystem access,
// JIT, repacking, host expert routing, or precision fallback.
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
  [[nodiscard]] Result<Gemma4Moe26BReferencePrediction> Prediction();

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

 private:
  struct Impl;
  explicit Gemma4Moe26BReferenceEngine(std::unique_ptr<Impl> implementation);
  std::unique_ptr<Impl> implementation_;
};

}  // namespace gem16::internal
