#include "cuda/moe/prefill_plan.h"

#include <array>
#include <limits>
#include <string>

namespace gem16::internal {
namespace {

constexpr std::uint64_t kAlignment = 256U;
constexpr std::uint64_t kWidth = 2816U;
constexpr std::uint64_t kShared = 2112U;
constexpr std::uint64_t kExpert = 704U;
constexpr std::uint64_t kExperts = 128U;
constexpr std::uint64_t kTopK = 8U;
constexpr std::array<std::uint32_t, 4> kCandidates{128U, 256U, 512U, 1024U};
constexpr std::uint64_t kMaximumContextTokens = 262144U;
constexpr std::uint64_t kPreparedGlobalPrefillTokens = 16384U;
constexpr std::uint64_t kLongContextChunkTokens = 1024U;

Status Invalid(std::string message) {
  return Status(StatusCode::kInvalidArgument, std::move(message));
}

bool AddRegion(std::vector<Gemma4MoePrefillRegion>* regions,
               std::uint64_t* cursor, std::string_view name,
               std::uint64_t elements, std::uint64_t element_bytes) {
  if (elements > std::numeric_limits<std::uint64_t>::max() / element_bytes) {
    return false;
  }
  const std::uint64_t aligned =
      (*cursor + kAlignment - 1U) & ~(kAlignment - 1U);
  const std::uint64_t bytes = elements * element_bytes;
  if (aligned > std::numeric_limits<std::uint64_t>::max() - bytes) {
    return false;
  }
  regions->push_back({name, aligned, bytes});
  *cursor = aligned + bytes;
  return true;
}

Result<Gemma4MoePrefillPlan> BuildCandidate(std::uint32_t tokens) {
  Gemma4MoePrefillPlan plan;
  plan.chunk_tokens = tokens;
  const std::uint64_t t = tokens;
  const std::uint64_t assignments = kTopK * t;
  std::uint64_t permutation = 0U;
  if (!AddRegion(&plan.permutation_regions, &permutation, "assignments",
                 assignments, sizeof(Gemma4MoePrefillAssignment)) ||
      !AddRegion(&plan.permutation_regions, &permutation, "histogram",
                 kExperts, sizeof(std::uint32_t)) ||
      !AddRegion(&plan.permutation_regions, &permutation, "prefix",
                 kExperts + 1U, sizeof(std::uint32_t)) ||
      !AddRegion(&plan.permutation_regions, &permutation, "permutation",
                 assignments, sizeof(std::uint32_t)) ||
      !AddRegion(&plan.permutation_regions, &permutation, "inverse",
                 assignments, sizeof(std::uint32_t))) {
    return Invalid("M15 permutation workspace arithmetic overflow");
  }
  plan.permutation_workspace_bytes =
      (permutation + kAlignment - 1U) & ~(kAlignment - 1U);

  std::uint64_t moe = 0U;
  // Router values, one reusable token hidden buffer, compact NVFP4 inputs,
  // selected-expert W13 products/W2 partials and shared-branch product/output.
  if (!AddRegion(&plan.moe_regions, &moe, "router_logits_probabilities",
                 2U * t * kExperts, sizeof(float)) ||
      !AddRegion(&plan.moe_regions, &moe, "token_hidden", t * kWidth,
                 sizeof(float)) ||
      !AddRegion(&plan.moe_regions, &moe, "token_nvfp4",
                 t * (kWidth / 2U + kWidth / 16U), sizeof(std::uint8_t)) ||
      !AddRegion(&plan.moe_regions, &moe, "expert_product",
                 assignments * kExpert, sizeof(float)) ||
      !AddRegion(&plan.moe_regions, &moe, "expert_product_nvfp4",
                 assignments * (kExpert / 2U + kExpert / 16U),
                 sizeof(std::uint8_t)) ||
      !AddRegion(&plan.moe_regions, &moe, "expert_down_partials",
                 assignments * kWidth, sizeof(float)) ||
      !AddRegion(&plan.moe_regions, &moe, "shared_product", t * kShared,
                 sizeof(float)) ||
      !AddRegion(&plan.moe_regions, &moe, "shared_product_nvfp4",
                 t * (kShared / 2U + kShared / 16U), sizeof(std::uint8_t)) ||
      !AddRegion(&plan.moe_regions, &moe, "shared_output", t * kWidth,
                 sizeof(float)) ||
      !AddRegion(&plan.moe_regions, &moe, "reduced_output", t * kWidth,
                 sizeof(float))) {
    return Invalid("M15 MoE workspace arithmetic overflow");
  }
  plan.moe_workspace_bytes =
      (moe + kAlignment - 1U) & ~(kAlignment - 1U);
  return plan;
}

}  // namespace

Result<Gemma4Moe26BVisionCacheRelation>
ClassifyGemma4Moe26BVisionCacheSpan(
    std::uint64_t prefix_tokens, std::uint64_t vision_begin,
    std::uint64_t vision_end) {
  if (vision_begin >= vision_end) {
    return Invalid("Gemma 4 26B Vision cache span is invalid");
  }
  if (vision_begin < prefix_tokens && vision_end > prefix_tokens) {
    return Gemma4Moe26BVisionCacheRelation::kSplit;
  }
  return vision_end <= prefix_tokens
             ? Gemma4Moe26BVisionCacheRelation::kFullyCached
             : Gemma4Moe26BVisionCacheRelation::kFullyUncached;
}

Result<Gemma4Moe26BPrefillChunkPlan> BuildGemma4Moe26BPrefillChunkPlan(
    std::uint64_t start_position, std::uint64_t tokens,
    std::uint32_t prepared_chunk_tokens, std::uint64_t vision_begin,
    std::uint64_t vision_end) {
  const bool has_vision = vision_begin < vision_end;
  if (tokens == 0U || prepared_chunk_tokens == 0U ||
      prepared_chunk_tokens > 2048U || start_position > kMaximumContextTokens ||
      tokens > kMaximumContextTokens - start_position ||
      vision_begin > vision_end ||
      (has_vision &&
       (vision_begin < start_position ||
        vision_end > start_position + tokens ||
        vision_end - vision_begin > prepared_chunk_tokens))) {
    return Invalid("Gemma 4 26B prefill chunk request is invalid");
  }

  Gemma4Moe26BPrefillChunkPlan plan;
  std::uint64_t cursor = start_position;
  const std::uint64_t end = start_position + tokens;
  while (cursor < end) {
    const std::uint64_t prepared_remaining =
        cursor < kPreparedGlobalPrefillTokens
            ? kPreparedGlobalPrefillTokens - cursor
            : 0U;
    std::uint64_t limit =
        prepared_remaining == 0U
            ? kLongContextChunkTokens
            : std::min<std::uint64_t>(prepared_chunk_tokens,
                                      prepared_remaining);
    std::uint64_t chunk = std::min(limit, end - cursor);

    // End the preceding chunk exactly at the image. The next iteration can
    // then admit the whole image block, including when it crosses the 16K
    // prepared-global boundary.
    if (has_vision && cursor < vision_begin &&
        cursor + chunk > vision_begin) {
      chunk = vision_begin - cursor;
    } else if (has_vision && cursor >= vision_begin &&
               cursor < vision_end && cursor + chunk < vision_end) {
      chunk = vision_end - cursor;
    }

    if (chunk == 0U || chunk > prepared_chunk_tokens ||
        plan.count == plan.chunks.size()) {
      return Status(StatusCode::kResourceExhausted,
                    "Gemma 4 26B prefill chunk plan exceeds its fixed bounds");
    }
    plan.chunks[plan.count++] = static_cast<std::uint32_t>(chunk);
    cursor += chunk;
  }
  return plan;
}

Result<Gemma4MoePrefillPlan> BuildGemma4MoePrefillPlan(
    std::uint32_t requested_chunk_tokens,
    std::uint64_t moe_workspace_cap_bytes,
    std::uint64_t permutation_workspace_cap_bytes) {
  if (requested_chunk_tokens == 0U || moe_workspace_cap_bytes == 0U ||
      permutation_workspace_cap_bytes == 0U) {
    return Invalid("M15 prefill plan requires non-zero request and caps");
  }
  for (auto candidate = kCandidates.rbegin(); candidate != kCandidates.rend();
       ++candidate) {
    if (*candidate > requested_chunk_tokens) continue;
    auto plan = BuildCandidate(*candidate);
    if (!plan.ok()) return plan.status();
    if (plan.value().moe_workspace_bytes <= moe_workspace_cap_bytes &&
        plan.value().permutation_workspace_bytes <=
            permutation_workspace_cap_bytes) {
      return plan;
    }
  }
  return Status(StatusCode::kResourceExhausted,
                "no M15 chunk candidate fits the fixed workspace caps");
}

}  // namespace gem16::internal
