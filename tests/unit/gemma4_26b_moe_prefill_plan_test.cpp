#include "cuda/moe/prefill_plan.h"

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

}  // namespace

void RunGemma426BMoePrefillPlanTests() {
  TestBoundedAndCompact();
  TestLargestFittingCandidate();
}
