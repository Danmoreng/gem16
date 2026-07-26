#include "gem16/layer.h"

#include "test.h"

#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace {

void TestRmsNorm() {
  constexpr std::array<float, 8> input = {1.0F, 2.0F, 3.0F, 4.0F,
                                          -2.0F, 0.0F, 2.0F, 0.0F};
  constexpr std::array<float, 4> weight = {1.0F, 0.5F, 2.0F, 1.5F};
  const auto output = gem16::layer::RmsNorm(input, weight, 2, 4, 1.0e-6F);
  GEM16_CHECK(output.ok());
  if (!output.ok()) return;
  for (std::size_t vector = 0; vector < 2; ++vector) {
    double mean_square = 0.0;
    for (std::size_t index = 0; index < 4; ++index) {
      const double value = input[vector * 4 + index];
      mean_square += value * value / 4.0;
    }
    for (std::size_t index = 0; index < 4; ++index) {
      const double expected = input[vector * 4 + index] /
                              std::sqrt(mean_square + 1.0e-6) * weight[index];
      GEM16_CHECK(std::fabs(output.value()[vector * 4 + index] - expected) < 1.0e-6);
    }
  }
  const auto unscaled = gem16::layer::RmsNorm(
      std::span<const float>(input.data(), 4), {}, 1, 4, 1.0e-6F);
  GEM16_CHECK(unscaled.ok());
}

void TestRotaryEmbedding() {
  std::array<float, 8> states = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F};
  const auto original = states;
  GEM16_CHECK(gem16::layer::ApplyRotaryEmbedding(states, 1, 8, 4, 0, 10000.0).ok());
  GEM16_CHECK(states == original);
  GEM16_CHECK(gem16::layer::ApplyRotaryEmbedding(states, 1, 8, 4, 3, 10000.0).ok());
  GEM16_CHECK(states[4] == original[4] && states[7] == original[7]);
  const double first_norm = static_cast<double>(states[0]) * states[0] +
                            static_cast<double>(states[2]) * states[2];
  const double original_norm = static_cast<double>(original[0]) * original[0] +
                               static_cast<double>(original[2]) * original[2];
  GEM16_CHECK(std::fabs(first_norm - original_norm) < 1.0e-5);
}

void TestProportionalRotaryEmbedding() {
  std::vector<float> states(512);
  for (std::size_t index = 0; index < states.size(); ++index) {
    states[index] = static_cast<float>(index + 1U) * 0.01F;
  }
  const auto original = states;
  GEM16_CHECK(gem16::layer::ApplyProportionalRotaryEmbedding(
                    states, 1, 512, 0.25, 31, 1'000'000.0)
                    .ok());
  GEM16_CHECK(states[0] != original[0]);
  GEM16_CHECK(states[256] != original[256]);
  GEM16_CHECK(states[63] != original[63]);
  GEM16_CHECK(states[319] != original[319]);
  GEM16_CHECK(states[64] == original[64]);
  GEM16_CHECK(states[320] == original[320]);
  GEM16_CHECK(states[255] == original[255]);
  GEM16_CHECK(states[511] == original[511]);
  const double rotated_norm = static_cast<double>(states[0]) * states[0] +
                              static_cast<double>(states[256]) * states[256];
  const double original_norm = static_cast<double>(original[0]) * original[0] +
                               static_cast<double>(original[256]) * original[256];
  GEM16_CHECK(std::fabs(rotated_norm - original_norm) < 1.0e-5);
}

void TestGroupedQueryAttention() {
  constexpr std::array<float, 8> query = {
      1.0F, 0.0F, 0.0F, 1.0F, 1.0F, 0.0F, 1.0F, 0.0F};
  constexpr std::array<float, 8> keys = {
      1.0F, 0.0F, 0.0F, 1.0F, 0.0F, 1.0F, 1.0F, 0.0F};
  constexpr std::array<float, 8> values = {
      2.0F, 4.0F, 10.0F, 20.0F, 6.0F, 8.0F, 30.0F, 40.0F};
  const auto output = gem16::layer::LocalAttentionDecode(
      query, keys, values, 4, 2, 2, 2);
  GEM16_CHECK(output.ok());
  if (!output.ok()) return;
  const double strong = std::exp(1.0) / (std::exp(1.0) + 1.0);
  GEM16_CHECK(std::fabs(output.value()[0] - (2.0 * strong + 6.0 * (1.0 - strong))) < 1.0e-6);
  GEM16_CHECK(std::fabs(output.value()[1] - (4.0 * strong + 8.0 * (1.0 - strong))) < 1.0e-6);
  GEM16_CHECK(std::fabs(output.value()[4] - (10.0 * (1.0 - strong) + 30.0 * strong)) < 1.0e-5);
  GEM16_CHECK(std::fabs(output.value()[7] - (20.0 * (1.0 - strong) + 40.0 * strong)) < 1.0e-5);
}

void TestInvalidLayerInputs() {
  std::array<float, 1> value = {std::numeric_limits<float>::quiet_NaN()};
  GEM16_CHECK(!gem16::layer::RmsNorm(value, {}, 1, 1, 1.0e-6F).ok());
  std::array<float, 4> rope{};
  GEM16_CHECK(!gem16::layer::ApplyRotaryEmbedding(rope, 1, 4, 3, 0, 10000.0).ok());
  GEM16_CHECK(!gem16::layer::ApplyProportionalRotaryEmbedding(
                     rope, 1, 4, 0.0, 0, 1'000'000.0)
                     .ok());
  GEM16_CHECK(!gem16::layer::ApplyProportionalRotaryEmbedding(
                     rope, 1, 4, 0.25, 0, 1'000'000.0, 0.0)
                     .ok());
  GEM16_CHECK(!gem16::layer::LocalAttentionDecode({}, {}, {}, 16, 8, 256, 0).ok());
}

}  // namespace

void RunLayerTests() {
  TestRmsNorm();
  TestRotaryEmbedding();
  TestProportionalRotaryEmbedding();
  TestGroupedQueryAttention();
  TestInvalidLayerInputs();
}
