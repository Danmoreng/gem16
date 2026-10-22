#include "cuda/engine/greedy_selection.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>

#include "test.h"

namespace {

// Independent exhaustive oracle: visit tokens in ID order, retaining the first
// maximum. The input contains the FP32 values after softcap, as the engine does.
std::uint32_t MaskedWinner(std::span<const float> values,
                           std::span<const std::uint32_t> mask) {
  float best = -std::numeric_limits<float>::infinity();
  std::uint32_t winner = 0U;
  for (std::uint32_t token = 0U; token < values.size(); ++token) {
    if (std::find(mask.begin(), mask.end(), token) == mask.end() &&
        values[token] > best) {
      best = values[token];
      winner = token;
    }
  }
  return winner;
}

void TestGreedyAllowedWinner() {
  using gem16::internal::GreedyWinnerIsAllowed;
  const std::array<std::uint32_t, 16> padded{1U, 50U, 106U, 258882U, 258883U};
  const auto active = std::span(padded).first(5U);
  GEM16_CHECK(GreedyWinnerIsAllowed(0U, active));
  GEM16_CHECK(GreedyWinnerIsAllowed(61684U, active));
  GEM16_CHECK(!GreedyWinnerIsAllowed(258883U, active));
  GEM16_CHECK(GreedyWinnerIsAllowed(1U, {}));
  GEM16_CHECK(!GreedyWinnerIsAllowed(0U, std::span(padded)));

  // Cover every non-total suppression subset against varied orderings, ties,
  // and FP32 softcap saturation. A disallowed global winner must use fallback;
  // an allowed one must equal exhaustive masked selection.
  constexpr std::array<std::array<float, 5>, 6> logits{{
      {{1.0F, 2.0F, 3.0F, 4.0F, 5.0F}},
      {{5.0F, 4.0F, 3.0F, 2.0F, 1.0F}},
      {{4.0F, 4.0F, 3.0F, 4.0F, 2.0F}},
      {{-4.0F, -4.0F, -4.0F, -4.0F, -4.0F}},
      {{900.0F, 1000.0F, 1100.0F, 1.0F, 2.0F}},
      {{0.0F, -0.0F, -1.0F, -2.0F, -3.0F}},
  }};
  for (const auto& input : logits) {
    std::array<float, 5> values{};
    std::transform(input.begin(), input.end(), values.begin(),
                   [](float v) { return std::tanh(v / 30.0F) * 30.0F; });
    const auto global = MaskedWinner(values, {});
    for (unsigned bits = 0U; bits < 31U; ++bits) {
      std::array<std::uint32_t, 5> mask{};
      std::size_t count = 0U;
      for (std::uint32_t token = 0U; token < values.size(); ++token) {
        if ((bits & (1U << token)) != 0U) mask[count++] = token;
      }
      const auto ids = std::span(mask).first(count);
      const auto expected = MaskedWinner(values, ids);
      const bool allowed = GreedyWinnerIsAllowed(global, ids);
      GEM16_CHECK(allowed == ((bits & (1U << global)) == 0U));
      GEM16_CHECK((allowed ? global : MaskedWinner(values, ids)) == expected);
    }
  }
}

}  // namespace

void RunGreedySelectionTests() { TestGreedyAllowedWinner(); }
