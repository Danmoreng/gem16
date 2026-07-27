#include "gem16/sampling.h"

#include "test.h"

#include <limits>

namespace {

void TestSamplingValidation() {
  gem16::SamplingOptions options;
  GEM16_CHECK(gem16::ValidateSamplingOptions(options, 8U).ok());
  options.enabled = true;
  GEM16_CHECK(gem16::ValidateSamplingOptions(options, 8U).ok());

  options.temperature = 0.0F;
  GEM16_CHECK(!gem16::ValidateSamplingOptions(options, 8U).ok());
  options.temperature = 1.0F;
  options.top_p = 0.0F;
  GEM16_CHECK(!gem16::ValidateSamplingOptions(options, 8U).ok());
  options.top_p = 1.0F;
  options.min_p = 1.01F;
  GEM16_CHECK(!gem16::ValidateSamplingOptions(options, 8U).ok());
  options.min_p = 0.0F;
  options.repetition_penalty = -1.0F;
  GEM16_CHECK(!gem16::ValidateSamplingOptions(options, 8U).ok());
  options.repetition_penalty = 1.0F;
  options.top_k = 9U;
  GEM16_CHECK(!gem16::ValidateSamplingOptions(options, 8U).ok());
  options.top_k = 0U;
  options.temperature = std::numeric_limits<float>::quiet_NaN();
  GEM16_CHECK(!gem16::ValidateSamplingOptions(options, 8U).ok());
  options.temperature = 1.0F;
  GEM16_CHECK(!gem16::ValidateSamplingOptions(options, 0U).ok());
}

}  // namespace

void RunSamplingTests() { TestSamplingValidation(); }
