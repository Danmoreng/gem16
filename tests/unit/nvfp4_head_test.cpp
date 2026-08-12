#include "gem16/nvfp4_head.h"
#include "cli/nvfp4_head_diagnostic_reference.h"

#include "gem16/nvfp4.h"
#include "test.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace {

constexpr std::uint64_t kVocabulary = 4U;
constexpr std::uint64_t kHidden = 32U;
constexpr std::size_t kPackedPerRow = kHidden / 2U;
constexpr std::size_t kScalesPerRow = kHidden / 16U;

void SetNibble(std::vector<std::uint8_t>& packed, std::size_t index,
               std::uint8_t nibble) {
  const std::size_t byte = index / 2U;
  const unsigned shift = (index & 1U) == 0U ? 0U : 4U;
  packed[byte] = static_cast<std::uint8_t>(
      (packed[byte] & static_cast<std::uint8_t>(0x0FU << (4U - shift))) |
      static_cast<std::uint8_t>((nibble & 0x0FU) << shift));
}

std::uint8_t Nibble(const std::vector<std::uint8_t>& packed, std::size_t index) {
  return static_cast<std::uint8_t>((packed[index / 2U] >>
                                    ((index & 1U) == 0U ? 0U : 4U)) & 0x0FU);
}

float ManualE2M1(std::uint8_t nibble) {
  constexpr float magnitudes[] = {0.0F, 0.5F, 1.0F, 1.5F,
                                  2.0F, 3.0F, 4.0F, 6.0F};
  const float value = magnitudes[nibble & 7U];
  return (nibble & 8U) == 0U ? value : -value;
}

float ManualE4M3(std::uint8_t bits) {
  const int exponent = static_cast<int>((bits >> 3U) & 0x0FU);
  const int mantissa = static_cast<int>(bits & 7U);
  const float magnitude = exponent == 0
                              ? std::ldexp(static_cast<float>(mantissa), -9)
                              : std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F,
                                           exponent - 7);
  return (bits & 0x80U) == 0U ? magnitude : -magnitude;
}

float ManualBf16Rne(float value) {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  const std::uint32_t rounded = bits + 0x7FFFU + ((bits >> 16U) & 1U);
  return std::bit_cast<float>(rounded & 0xFFFF0000U);
}

std::uint16_t Bf16Bits(float value) {
  return static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(value) >> 16U);
}

std::uint8_t ManualEncodeE2M1(float value) {
  constexpr float magnitudes[] = {0.0F, 0.5F, 1.0F, 1.5F,
                                   2.0F, 3.0F, 4.0F, 6.0F};
  std::uint8_t best = 0U;
  float best_error = std::numeric_limits<float>::infinity();
  for (std::uint8_t candidate = 0U; candidate < 8U; ++candidate) {
    const float error = std::fabs(std::fabs(value) - magnitudes[candidate]);
    if (error < best_error ||
        (error == best_error && (candidate & 1U) == 0U && (best & 1U) != 0U)) {
      best = candidate;
      best_error = error;
    }
  }
  return static_cast<std::uint8_t>(best | (std::signbit(value) ? 0x8U : 0U));
}

std::uint8_t ManualEncodeE4M3(float value) {
  std::uint8_t best = 0U;
  float best_error = std::numeric_limits<float>::infinity();
  for (std::uint16_t candidate = 0U; candidate <= 0x7EU; ++candidate) {
    const auto bits = static_cast<std::uint8_t>(candidate);
    const float error = std::fabs(std::fabs(value) - ManualE4M3(bits));
    if (error < best_error ||
        (error == best_error && (bits & 1U) == 0U && (best & 1U) != 0U)) {
      best = bits;
      best_error = error;
    }
  }
  if (std::fabs(value) >= 448.0F) best = 0x7EU;
  return best;
}

std::vector<std::uint8_t> ManualActivationCodes(std::span<const float> hidden,
                                                float divisor,
                                                std::vector<std::uint8_t>* scales) {
  std::vector<std::uint8_t> codes(hidden.size(), 0U);
  scales->assign(hidden.size() / 16U, 0U);
  std::vector<float> rounded(hidden.size());
  for (std::size_t i = 0; i < hidden.size(); ++i) rounded[i] = ManualBf16Rne(hidden[i]);
  for (std::size_t block = 0; block < scales->size(); ++block) {
    float amax = 0.0F;
    for (std::size_t i = 0; i < 16U; ++i) amax = std::max(amax, std::fabs(rounded[block * 16U + i]));
    (*scales)[block] = ManualEncodeE4M3(amax * divisor / 6.0F);
    const float local = ManualE4M3((*scales)[block]);
    for (std::size_t i = 0; i < 16U; ++i) {
      const float normalized = local == 0.0F ? 0.0F : rounded[block * 16U + i] * divisor / local;
      codes[block * 16U + i] = ManualEncodeE2M1(normalized);
    }
  }
  return codes;
}

struct Fixture {
  std::vector<std::uint8_t> packed =
      std::vector<std::uint8_t>(kVocabulary * kPackedPerRow, 0U);
  std::vector<std::uint8_t> scales =
      std::vector<std::uint8_t>(kVocabulary * kScalesPerRow, 0x40U);
};

Fixture MakeFixture() {
  Fixture fixture;
  for (std::size_t row = 0; row < kVocabulary; ++row) {
    for (std::size_t index = 0U; index < kHidden; ++index) {
      std::uint8_t nibble = static_cast<std::uint8_t>((index % 8U) + 1U);
      if ((index % 5U) == 1U) nibble = static_cast<std::uint8_t>(nibble | 0x8U);
      if (row == 3U) nibble = static_cast<std::uint8_t>(nibble ^ 0x8U);
      SetNibble(fixture.packed, row * kHidden + index, nibble);
    }
  }
  for (std::size_t row = 0U; row < kVocabulary; ++row) {
    fixture.scales[row * kScalesPerRow] = 0x40U;
    fixture.scales[row * kScalesPerRow + 1U] = 0x48U;
  }
  return fixture;
}

float ManualProjectionLogit(const Fixture& fixture, std::span<const float> hidden,
                            std::size_t row, float weight_divisor,
                            float input_divisor, float softcap) {
  std::vector<std::uint8_t> activation_scales;
  const auto activation = ManualActivationCodes(hidden, input_divisor, &activation_scales);
  double dot = 0.0;
  for (std::size_t i = 0; i < hidden.size(); ++i) {
    const auto block = i / 16U;
    const float av = ManualE2M1(activation[i]) * ManualE4M3(activation_scales[block]);
    const float wv = ManualE2M1(Nibble(fixture.packed, row * kHidden + i)) *
                     ManualE4M3(fixture.scales[row * kScalesPerRow + block]);
    dot += static_cast<double>(av) * static_cast<double>(wv);
  }
  dot /= static_cast<double>(input_divisor) * static_cast<double>(weight_divisor);
  return std::tanh(static_cast<float>(dot) / softcap) * softcap;
}

gem16::Result<gem16::nvfp4::TiedNvfp4HeadView> MakeView(Fixture& fixture) {
  return gem16::nvfp4::TiedNvfp4HeadView::Create(
      fixture.packed, fixture.scales, kVocabulary, kHidden, 2.0F, 1.5F);
}

void TestManualRowNibbleIndexing() {
  const std::vector<std::uint8_t> packed = {0x21U, 0x43U, 0xA5U, 0xC7U};
  GEM16_CHECK(gem16::internal::Nvfp4HeadNibble(packed, 0U, 4U, 0U) == 1U);
  GEM16_CHECK(gem16::internal::Nvfp4HeadNibble(packed, 0U, 4U, 3U) == 4U);
  GEM16_CHECK(gem16::internal::Nvfp4HeadNibble(packed, 1U, 4U, 0U) == 5U);
  GEM16_CHECK(gem16::internal::Nvfp4HeadNibble(packed, 1U, 4U, 3U) == 0xCU);
}

void TestValidationAndBorrowedStorage() {
  Fixture fixture = MakeFixture();
  const auto view = MakeView(fixture);
  GEM16_CHECK(view.ok());
  if (!view.ok()) return;
  GEM16_CHECK(view.value().packed_weights().data() == fixture.packed.data());
  GEM16_CHECK(view.value().local_scales().data() == fixture.scales.data());
  GEM16_CHECK(view.value().packed_weights().size() == fixture.packed.size());

  GEM16_CHECK(!gem16::nvfp4::TiedNvfp4HeadView::Create(
                   std::span<const std::uint8_t>(fixture.packed).first(
                       fixture.packed.size() - 1U),
                   fixture.scales, kVocabulary, kHidden, 2.0F, 2.0F)
                   .ok());
  GEM16_CHECK(!gem16::nvfp4::TiedNvfp4HeadView::Create(
                   fixture.packed, fixture.scales, 0U, kHidden, 2.0F, 2.0F)
                   .ok());
  GEM16_CHECK(!gem16::nvfp4::TiedNvfp4HeadView::Create(
                   fixture.packed, fixture.scales, kVocabulary, 24U, 2.0F, 2.0F)
                   .ok());
  GEM16_CHECK(!gem16::nvfp4::TiedNvfp4HeadView::Create(
                   fixture.packed, fixture.scales, kVocabulary, kHidden, 0.0F,
                   2.0F)
                   .ok());
  GEM16_CHECK(!gem16::nvfp4::TiedNvfp4HeadView::Create(
                   fixture.packed, fixture.scales, kVocabulary, kHidden, -1.0F,
                   2.0F)
                   .ok());
  GEM16_CHECK(!gem16::nvfp4::TiedNvfp4HeadView::Create(
                   fixture.packed, fixture.scales, kVocabulary, kHidden, 2.0F,
                   -1.0F)
                   .ok());
  GEM16_CHECK(!gem16::nvfp4::TiedNvfp4HeadView::Create(
                   fixture.packed, fixture.scales, kVocabulary, kHidden,
                   std::numeric_limits<float>::quiet_NaN(), 2.0F)
                   .ok());
  Fixture bad_scale = fixture;
  bad_scale.scales[0] = 0x7FU;
  GEM16_CHECK(!MakeView(bad_scale).ok());
  bad_scale = fixture;
  bad_scale.scales[0] = 0xB8U;
  GEM16_CHECK(!MakeView(bad_scale).ok());

  const auto overflow_view = gem16::nvfp4::TiedNvfp4HeadView::Create(
      fixture.packed, fixture.scales, kVocabulary, kHidden, 1.0e-38F, 1.0F);
  GEM16_CHECK(overflow_view.ok());
  if (overflow_view.ok()) {
    GEM16_CHECK(!overflow_view.value().Lookup(0U).ok());
  }
}

void TestLookupIndependentOracle() {
  Fixture fixture = MakeFixture();
  const auto view = MakeView(fixture);
  GEM16_CHECK(view.ok());
  if (!view.ok()) return;

  const auto lookup = view.value().Lookup(0U);
  GEM16_CHECK(lookup.ok());
  if (!lookup.ok()) return;
  const float embedding_scale = ManualBf16Rne(std::sqrt(static_cast<float>(kHidden)));
  GEM16_CHECK(Bf16Bits(embedding_scale) == 0x40B5U);
  GEM16_CHECK(lookup.value().size() == kHidden);
  for (std::size_t index = 0U; index < kHidden; ++index) {
    const std::size_t block = index / 16U;
    const float expected = ManualBf16Rne(
        ManualE2M1(Nibble(fixture.packed, index)) *
        ManualE4M3(fixture.scales[block]) / 2.0F * embedding_scale);
    GEM16_CHECK(lookup.value()[index] == expected);
  }
  GEM16_CHECK(Bf16Bits(ManualBf16Rne(3.1415926F)) == 0x4049U);

  const auto negative = view.value().Lookup(3U);
  GEM16_CHECK(negative.ok());
  if (negative.ok()) {
    for (std::size_t index = 0U; index < kHidden; ++index) {
      GEM16_CHECK(negative.value()[index] == -lookup.value()[index]);
    }
  }
  GEM16_CHECK(!view.value().Lookup(kVocabulary).ok());
}

void TestProjectionSemantics() {
  Fixture fixture = MakeFixture();
  const auto view = MakeView(fixture);
  GEM16_CHECK(view.ok());
  if (!view.ok()) return;

  std::vector<float> hidden(kHidden, 0.0F);
  hidden[0] = 1.0F;
  const auto no_logits = view.value().ProjectT1(hidden);
  GEM16_CHECK(no_logits.ok());
  if (!no_logits.ok()) return;
  GEM16_CHECK(no_logits.value().diagnostic_logits.empty());

  const float expected_logit = ManualProjectionLogit(
      fixture, hidden, 0U, 2.0F, 1.5F, 30.0F);
  GEM16_CHECK(no_logits.value().token == 0U);
  GEM16_CHECK(no_logits.value().value == expected_logit);

  gem16::nvfp4::TiedNvfp4HeadView::ProjectionOptions all_logits_options;
  all_logits_options.return_diagnostic_logits = true;
  const auto all_logits = view.value().ProjectT1(hidden, all_logits_options);
  GEM16_CHECK(all_logits.ok());
  if (all_logits.ok()) {
    GEM16_CHECK(all_logits.value().diagnostic_logits.size() == kVocabulary);
    for (std::uint32_t row = 0U; row < kVocabulary; ++row) {
      GEM16_CHECK(all_logits.value().diagnostic_logits[row] ==
                  ManualProjectionLogit(fixture, hidden, row, 2.0F, 1.5F, 30.0F));
    }
  }

  const std::array<std::uint32_t, 1> suppressed_zero = {0U};
  gem16::nvfp4::TiedNvfp4HeadView::ProjectionOptions options;
  options.suppressed_tokens = suppressed_zero;
  options.return_diagnostic_logits = true;
  const auto suppressed = view.value().ProjectT1(hidden, options);
  GEM16_CHECK(suppressed.ok());
  if (suppressed.ok()) {
    GEM16_CHECK(suppressed.value().token == 1U);
    GEM16_CHECK(suppressed.value().diagnostic_logits.size() == kVocabulary);
    GEM16_CHECK(suppressed.value().diagnostic_logits[0] == expected_logit);
    GEM16_CHECK(suppressed.value().diagnostic_logits[1] == expected_logit);
    GEM16_CHECK(suppressed.value().diagnostic_logits[2] != 0.0F);
    GEM16_CHECK(suppressed.value().diagnostic_logits[3] != expected_logit);
  }

  const std::array<std::uint32_t, 17> too_many = {
      0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 9U, 10U, 11U, 12U, 13U, 14U, 15U, 16U};
  GEM16_CHECK(!view.value().ProjectT1(
      hidden, {.softcap = 30.0F, .suppressed_tokens = too_many,
               .return_diagnostic_logits = false}).ok());

  std::vector<float> nonfinite = hidden;
  nonfinite[1] = std::numeric_limits<float>::quiet_NaN();
  GEM16_CHECK(!view.value().ProjectT1(nonfinite).ok());
  nonfinite[1] = std::numeric_limits<float>::infinity();
  GEM16_CHECK(!view.value().ProjectT1(nonfinite).ok());
  GEM16_CHECK(!view.value().ProjectT1(
                       hidden, {.softcap = 0.0F,
                               .suppressed_tokens = {},
                               .return_diagnostic_logits = false})
                   .ok());
  GEM16_CHECK(!view.value().ProjectT1(
                       hidden, {.softcap = 30.0F,
                               .suppressed_tokens = std::array<std::uint32_t, 2>{1U, 1U},
                               .return_diagnostic_logits = false})
                   .ok());
  GEM16_CHECK(!view.value().ProjectT1(
                       hidden, {.softcap = 30.0F,
                               .suppressed_tokens = std::array<std::uint32_t, 1>{kVocabulary},
                               .return_diagnostic_logits = false})
                   .ok());
  const std::array<std::uint32_t, kVocabulary> all_tokens = {0U, 1U, 2U, 3U};
  GEM16_CHECK(!view.value().ProjectT1(
                       hidden, {.softcap = 30.0F,
                               .suppressed_tokens = all_tokens,
                               .return_diagnostic_logits = false})
                   .ok());
  GEM16_CHECK(!view.value().ProjectT1(std::span<const float>(hidden).first(kHidden - 1U))
                   .ok());
}

void TestCanonicalZeroAndPhysicalBf16Boundary() {
  Fixture malformed = MakeFixture();
  malformed.scales[0] = 0x80U;  // E4M3FN negative zero is not canonical.
  GEM16_CHECK(!MakeView(malformed).ok());
  malformed = MakeFixture();
  malformed.scales[0] = 0U;
  malformed.packed[0] = 1U;
  GEM16_CHECK(!MakeView(malformed).ok());

  Fixture fixture = MakeFixture();
  const auto view = MakeView(fixture);
  GEM16_CHECK(view.ok());
  if (!view.ok()) return;
  std::vector<float> hidden(kHidden, 0.0F);
  hidden[0] = 0.0508F;
  const auto projected = view.value().ProjectT1(
      hidden, {.softcap = 7.0F, .suppressed_tokens = {},
               .return_diagnostic_logits = false});
  GEM16_CHECK(projected.ok());
  if (!projected.ok()) return;
  const float expected = ManualProjectionLogit(fixture, hidden, 0U, 2.0F, 1.5F, 7.0F);
  GEM16_CHECK(projected.value().value == expected);
  // This value crosses an NVFP4 scale boundary after the required BF16-RNE
  // boundary; a direct FP32 quantization would produce a different logit.
  std::vector<std::uint8_t> unrounded_scales(2U, 0U);
  std::vector<std::uint8_t> unrounded_codes(kHidden, 0U);
  for (std::size_t block = 0U; block < 2U; ++block) {
    float amax = 0.0F;
    for (std::size_t i = 0U; i < 16U; ++i) {
      amax = std::max(amax, std::fabs(hidden[block * 16U + i]));
    }
    unrounded_scales[block] = ManualEncodeE4M3(amax * 1.5F / 6.0F);
    const float local = ManualE4M3(unrounded_scales[block]);
    for (std::size_t i = 0U; i < 16U; ++i) {
      const float normalized = local == 0.0F
                                   ? 0.0F
                                   : hidden[block * 16U + i] * 1.5F / local;
      unrounded_codes[block * 16U + i] = ManualEncodeE2M1(normalized);
    }
  }
  const float rounded_input = ManualBf16Rne(hidden[0]);
  hidden[0] = rounded_input;
  const float rounded_again = ManualProjectionLogit(fixture, hidden, 0U, 2.0F, 1.5F, 7.0F);
  GEM16_CHECK(rounded_again == projected.value().value);
  hidden[0] = 0.0508F;
  double direct_dot = 0.0;
  for (std::size_t index = 0U; index < kHidden; ++index) {
    const std::size_t block = index / 16U;
    direct_dot += static_cast<double>(ManualE2M1(unrounded_codes[index]) *
                                      ManualE4M3(unrounded_scales[block])) *
                  static_cast<double>(ManualE2M1(Nibble(fixture.packed, index)) *
                                      ManualE4M3(fixture.scales[block]));
  }
  const float direct_logit = std::tanh(static_cast<float>(direct_dot / 3.0) / 7.0F) * 7.0F;
  GEM16_CHECK(direct_logit != projected.value().value);
}

void TestSuppressionLimit() {
  constexpr std::uint64_t vocabulary = 18U;
  constexpr std::uint64_t hidden_size = 16U;
  std::vector<std::uint8_t> packed(vocabulary * 8U, 0U);
  std::vector<std::uint8_t> scales(vocabulary, 0x40U);
  const auto view = gem16::nvfp4::TiedNvfp4HeadView::Create(
      packed, scales, vocabulary, hidden_size, 1.5F, 1.25F);
  GEM16_CHECK(view.ok());
  if (!view.ok()) return;
  const std::array<std::uint32_t, 17> too_many = {
      0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 9U, 10U, 11U, 12U, 13U, 14U, 15U, 16U};
  const std::vector<float> hidden(hidden_size, 0.0F);
  GEM16_CHECK(!view.value().ProjectT1(
      hidden, {.suppressed_tokens = too_many, .return_diagnostic_logits = false}).ok());
}

}  // namespace

void RunNvfp4HeadTests() {
  TestManualRowNibbleIndexing();
  TestValidationAndBorrowedStorage();
  TestLookupIndependentOracle();
  TestProjectionSemantics();
  TestCanonicalZeroAndPhysicalBf16Boundary();
  TestSuppressionLimit();
}
