#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#include "gem16/status.h"

namespace gem16::internal {

struct Gemma4MoePrefillAssignment {
  std::uint16_t expert_id = 0;
  std::uint16_t topk_slot = 0;
  std::uint32_t token_id = 0;
  float weight = 0.0F;
};
static_assert(sizeof(Gemma4MoePrefillAssignment) == 12U);

struct Gemma4MoePrefillRegion {
  std::string_view name;
  std::uint64_t offset = 0;
  std::uint64_t bytes = 0;
};

struct Gemma4MoePrefillPlan {
  std::uint32_t chunk_tokens = 0;
  std::uint64_t moe_workspace_bytes = 0;
  std::uint64_t permutation_workspace_bytes = 0;
  std::vector<Gemma4MoePrefillRegion> moe_regions;
  std::vector<Gemma4MoePrefillRegion> permutation_regions;
};

// Runtime 26B prompt chunking stays allocation-free. At the 262144-token
// engine ceiling, ordinary 1024-token chunks need at most 256 entries; one
// image-alignment split can add one more entry.
constexpr std::size_t kGemma4Moe26BMaximumPrefillChunks = 257U;

struct Gemma4Moe26BPrefillChunkPlan {
  std::array<std::uint32_t, kGemma4Moe26BMaximumPrefillChunks> chunks{};
  std::size_t count = 0U;
};

enum class Gemma4Moe26BVisionCacheRelation {
  kFullyCached,
  kFullyUncached,
  kSplit,
};

// Classifies a non-empty absolute Vision span against the resident prompt
// prefix. A split span is invalid because its embeddings cannot be partially
// reconstructed from the text KV cache.
[[nodiscard]] Result<Gemma4Moe26BVisionCacheRelation>
ClassifyGemma4Moe26BVisionCacheSpan(
    std::uint64_t prefix_tokens, std::uint64_t vision_begin,
    std::uint64_t vision_end);

// Builds the complete bounded prompt schedule before GPU work begins.
// `vision_begin`/`vision_end` are absolute prompt positions. A non-empty
// Vision span is kept inside exactly one chunk while the no-Vision plan
// preserves the established 2048-before-16K/1024-after-16K schedule.
[[nodiscard]] Result<Gemma4Moe26BPrefillChunkPlan>
BuildGemma4Moe26BPrefillChunkPlan(
    std::uint64_t start_position, std::uint64_t tokens,
    std::uint32_t prepared_chunk_tokens,
    std::uint64_t vision_begin = 0U, std::uint64_t vision_end = 0U);

// M15 initialization-only bounded workspace plan. It materializes exactly 8T
// assignments and selected-expert products/partials, never T*128 activations.
[[nodiscard]] Result<Gemma4MoePrefillPlan> BuildGemma4MoePrefillPlan(
    std::uint32_t requested_chunk_tokens,
    std::uint64_t moe_workspace_cap_bytes = 192U * 1024U * 1024U,
    std::uint64_t permutation_workspace_cap_bytes = 64U * 1024U * 1024U);

}  // namespace gem16::internal
