#pragma once

#include <array>
#include <cstdint>
#include <span>

#include "gem16/status.h"
#include "gem16/types.h"
#include "model/config.h"

namespace gem16::internal {

enum class Gemma4Moe26BAttentionType : std::uint8_t {
  kSliding = 0,
  kFull,
};

enum class Gemma4Moe26BKvSource : std::uint8_t {
  kOwnedProjection = 0,
};

struct Gemma4Moe26BAttentionLayerTraits {
  std::uint32_t layer = 0;
  Gemma4Moe26BAttentionType attention =
      Gemma4Moe26BAttentionType::kSliding;
  std::uint32_t query_heads = 0;
  std::uint32_t kv_heads = 0;
  std::uint32_t head_dimension = 0;
  std::uint32_t cache_capacity = 0;
  bool stores_v_projection = false;
  bool reuses_raw_k_for_v = false;
  Gemma4Moe26BKvSource kv_source =
      Gemma4Moe26BKvSource::kOwnedProjection;
  std::int32_t kv_producer_layer = -1;
  double rope_theta = 0.0;
  double rotary_factor = 0.0;
  double rope_scaling_factor = 0.0;
};

using Gemma4Moe26BAttentionTraits =
    std::array<Gemma4Moe26BAttentionLayerTraits, 30>;

// Contract validation happens before this immutable table is constructed.
// Layer kinds are copied from the validated config; they are never inferred
// from the layer index in execution code.
[[nodiscard]] Result<Gemma4Moe26BAttentionTraits>
BuildGemma4Moe26BAttentionTraits(const ModelConfig& config);

// Counts physical one-byte FP8 K and V payloads. Local layers own fixed 1024
// rings; full layers own context-sized append caches. Scale metadata is part
// of the immutable checkpoint and is therefore not counted here.
[[nodiscard]] Result<std::uint64_t> Gemma4Moe26BFp8KvBytes(
    const Gemma4Moe26BAttentionTraits& traits,
    std::uint64_t context_tokens);

// Revalidates the attention-facing subset of the compiled M08 inventory at
// the execution boundary, including the deliberate absence of global V.
[[nodiscard]] Status ValidateGemma4Moe26BAttentionBindings(
    std::span<const TensorInfo> tensors,
    const Gemma4Moe26BAttentionTraits& traits);

}  // namespace gem16::internal
