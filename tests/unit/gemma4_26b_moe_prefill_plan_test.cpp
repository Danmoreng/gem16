#include "cuda/moe/prefill_plan.h"

#include <array>
#include <cstdint>

#include "test.h"

namespace {

void TestBoundedAndCompact() {
  auto plan = gem16::internal::BuildGemma4MoePrefillPlan(1024U);
  GEM16_CHECK(plan.ok());
  GEM16_CHECK(plan.value().chunk_tokens == 1024U);
  GEM16_CHECK(plan.value().moe_workspace_bytes <= 192U * 1024U * 1024U);
  GEM16_CHECK(plan.value().permutation_workspace_bytes <=
              64U * 1024U * 1024U);
  GEM16_CHECK(plan.value().permutation_regions.front().name == "assignments");
  GEM16_CHECK(plan.value().permutation_regions.front().bytes ==
              8U * 1024U *
                  sizeof(gem16::internal::Gemma4MoePrefillAssignment));
}

void TestLargestFittingCandidate() {
  auto plan = gem16::internal::BuildGemma4MoePrefillPlan(700U);
  GEM16_CHECK(plan.ok());
  GEM16_CHECK(plan.value().chunk_tokens == 512U);
  auto too_small = gem16::internal::BuildGemma4MoePrefillPlan(127U);
  GEM16_CHECK(!too_small.ok());
}

void TestVisionAwareRuntimeChunks() {
  constexpr std::uint64_t kTotalTokens = 20000U;
  constexpr std::uint64_t kVisionTokens = 280U;
  constexpr std::array<std::uint64_t, 11U> kVisionOffsets{
      0U, 1U, 1023U, 1024U, 1940U, 2047U, 2048U, 4095U,
      16105U, 16383U, 16384U};
  for (const std::uint64_t vision_begin : kVisionOffsets) {
    const std::uint64_t vision_end = vision_begin + kVisionTokens;
    auto plan = gem16::internal::BuildGemma4Moe26BPrefillChunkPlan(
        0U, kTotalTokens, 2048U, vision_begin, vision_end);
    GEM16_CHECK(plan.ok());
    if (!plan.ok()) continue;
    std::uint64_t cursor = 0U;
    std::uint32_t containing_chunks = 0U;
    for (std::size_t index = 0U; index < plan.value().count; ++index) {
      const std::uint64_t next = cursor + plan.value().chunks[index];
      GEM16_CHECK(!(cursor < vision_begin && next > vision_begin));
      GEM16_CHECK(!(next > vision_begin && next < vision_end));
      if (cursor <= vision_begin && next >= vision_end) ++containing_chunks;
      GEM16_CHECK(plan.value().chunks[index] <= 2048U);
      cursor = next;
    }
    GEM16_CHECK(cursor == kTotalTokens);
    GEM16_CHECK(containing_chunks == 1U);
  }
}

void TestTextRuntimeChunksStayStable() {
  auto plan = gem16::internal::BuildGemma4Moe26BPrefillChunkPlan(
      0U, 20000U, 2048U);
  GEM16_CHECK(plan.ok());
  if (plan.ok()) {
    GEM16_CHECK(plan.value().count == 12U);
    for (std::size_t index = 0U; index < 8U; ++index) {
      GEM16_CHECK(plan.value().chunks[index] == 2048U);
    }
    GEM16_CHECK(plan.value().chunks[8U] == 1024U);
    GEM16_CHECK(plan.value().chunks[9U] == 1024U);
    GEM16_CHECK(plan.value().chunks[10U] == 1024U);
    GEM16_CHECK(plan.value().chunks[11U] == 544U);
  }
  auto legacy = gem16::internal::BuildGemma4Moe26BPrefillChunkPlan(
      0U, 262144U, 1024U);
  GEM16_CHECK(legacy.ok());
  if (legacy.ok()) GEM16_CHECK(legacy.value().count == 256U);
}

void TestRuntimeChunkValidation() {
  GEM16_CHECK(!gem16::internal::BuildGemma4Moe26BPrefillChunkPlan(
                   0U, 0U, 2048U)
                   .ok());
  GEM16_CHECK(!gem16::internal::BuildGemma4Moe26BPrefillChunkPlan(
                   0U, 4096U, 2048U, 100U, 2149U)
                   .ok());
  GEM16_CHECK(!gem16::internal::BuildGemma4Moe26BPrefillChunkPlan(
                   100U, 4096U, 2048U, 99U, 379U)
                   .ok());
}

void TestVisionCacheSpanClassification() {
  using gem16::internal::ClassifyGemma4Moe26BVisionCacheSpan;
  using gem16::internal::Gemma4Moe26BVisionCacheRelation;
  auto cached = ClassifyGemma4Moe26BVisionCacheSpan(512U, 100U, 380U);
  GEM16_CHECK(cached.ok());
  if (cached.ok()) {
    GEM16_CHECK(cached.value() ==
                Gemma4Moe26BVisionCacheRelation::kFullyCached);
  }
  auto uncached = ClassifyGemma4Moe26BVisionCacheSpan(100U, 100U, 380U);
  GEM16_CHECK(uncached.ok());
  if (uncached.ok()) {
    GEM16_CHECK(uncached.value() ==
                Gemma4Moe26BVisionCacheRelation::kFullyUncached);
  }
  auto split = ClassifyGemma4Moe26BVisionCacheSpan(200U, 100U, 380U);
  GEM16_CHECK(split.ok());
  if (split.ok()) {
    GEM16_CHECK(split.value() == Gemma4Moe26BVisionCacheRelation::kSplit);
  }
  GEM16_CHECK(!ClassifyGemma4Moe26BVisionCacheSpan(100U, 100U, 100U).ok());
}

}  // namespace

void RunGemma426BMoePrefillPlanTests() {
  TestBoundedAndCompact();
  TestLargestFittingCandidate();
  TestVisionAwareRuntimeChunks();
  TestTextRuntimeChunksStayStable();
  TestRuntimeChunkValidation();
  TestVisionCacheSpanClassification();
}
