#pragma once

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

// M15 initialization-only bounded workspace plan. It materializes exactly 8T
// assignments and selected-expert products/partials, never T*128 activations.
[[nodiscard]] Result<Gemma4MoePrefillPlan> BuildGemma4MoePrefillPlan(
    std::uint32_t requested_chunk_tokens,
    std::uint64_t moe_workspace_cap_bytes = 192U * 1024U * 1024U,
    std::uint64_t permutation_workspace_cap_bytes = 64U * 1024U * 1024U);

}  // namespace gem16::internal
