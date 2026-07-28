#include "cuda/mtp/scheduler.h"
#include "test.h"

void RunMtpSchedulerTests() {
  using gem16::internal::AdaptiveMtpScheduler;

  AdaptiveMtpScheduler explicit_d4(4U, 16384U, false);
  for (unsigned index = 0U; index < 32U; ++index) {
    explicit_d4.Observe(16384U + index, 0U);
  }
  GEM16_CHECK(explicit_d4.active_drafts() == 4U);
  GEM16_CHECK(!explicit_d4.use_ordinary_fallback());

  AdaptiveMtpScheduler long_context(4U, 16384U, true);
  GEM16_CHECK(long_context.active_drafts() == 2U);
  for (unsigned index = 0U; index < 16U; ++index) {
    long_context.Observe(16384U + index, 0U);
  }
  GEM16_CHECK(long_context.active_drafts() == 1U);
  for (unsigned index = 0U; index < 16U; ++index) {
    long_context.Observe(16400U + index, 0U);
  }
  GEM16_CHECK(long_context.use_ordinary_fallback());
  GEM16_CHECK(long_context.ordinary_fallback_remaining() == 16U);
  for (unsigned index = 0U; index < 16U; ++index) {
    long_context.ConsumeOrdinaryFallback();
  }
  GEM16_CHECK(!long_context.use_ordinary_fallback());

  AdaptiveMtpScheduler short_context(4U, 128U, true);
  GEM16_CHECK(short_context.active_drafts() == 4U);
  for (unsigned index = 0U; index < 16U; ++index) {
    short_context.Observe(128U + index, 1U);
  }
  GEM16_CHECK(short_context.active_drafts() == 2U);
  for (unsigned index = 0U; index < 16U; ++index) {
    short_context.Observe(144U + index, 2U);
  }
  GEM16_CHECK(short_context.active_drafts() == 4U);
}
