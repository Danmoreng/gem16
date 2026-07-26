#include "gem16/nvfp4.h"

#include "test.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

void TestE2M1Codec() {
  constexpr std::array<float, 8> expected = {
      0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F,
  };
  for (std::uint8_t bits = 0; bits < 16U; ++bits) {
    const float decoded = gem16::nvfp4::DecodeE2M1(bits);
    const float magnitude = expected[bits & 0x07U];
    GEM16_CHECK(decoded == ((bits & 0x08U) != 0U ? -magnitude : magnitude));
    const auto encoded = gem16::nvfp4::EncodeE2M1(decoded);
    GEM16_CHECK(encoded.ok());
    if (encoded.ok()) GEM16_CHECK(encoded.value() == bits);
  }

  const auto tie_zero = gem16::nvfp4::EncodeE2M1(0.25F);
  const auto tie_one = gem16::nvfp4::EncodeE2M1(0.75F);
  const auto tie_two = gem16::nvfp4::EncodeE2M1(1.75F);
  const auto tie_four = gem16::nvfp4::EncodeE2M1(3.5F);
  const auto saturated = gem16::nvfp4::EncodeE2M1(100.0F);
  GEM16_CHECK(tie_zero.ok() && tie_zero.value() == 0U);
  GEM16_CHECK(tie_one.ok() && tie_one.value() == 2U);
  GEM16_CHECK(tie_two.ok() && tie_two.value() == 4U);
  GEM16_CHECK(tie_four.ok() && tie_four.value() == 6U);
  GEM16_CHECK(saturated.ok() && saturated.value() == 7U);
  GEM16_CHECK(!gem16::nvfp4::EncodeE2M1(
                       std::numeric_limits<float>::infinity())
                       .ok());
}

void TestE4M3FnCodec() {
  GEM16_CHECK(gem16::nvfp4::DecodeE4M3Fn(0x00U) == 0.0F);
  GEM16_CHECK(gem16::nvfp4::DecodeE4M3Fn(0x01U) == 1.0F / 512.0F);
  GEM16_CHECK(gem16::nvfp4::DecodeE4M3Fn(0x08U) == 1.0F / 64.0F);
  GEM16_CHECK(gem16::nvfp4::DecodeE4M3Fn(0x38U) == 1.0F);
  GEM16_CHECK(gem16::nvfp4::DecodeE4M3Fn(0x61U) == 36.0F);
  GEM16_CHECK(gem16::nvfp4::DecodeE4M3Fn(0x7EU) == 448.0F);
  GEM16_CHECK(gem16::nvfp4::DecodeE4M3Fn(0xB8U) == -1.0F);
  GEM16_CHECK(std::isnan(gem16::nvfp4::DecodeE4M3Fn(0x7FU)));
  GEM16_CHECK(std::isnan(gem16::nvfp4::DecodeE4M3Fn(0xFFU)));

  for (std::uint16_t word = 0; word <= 0xFFU; ++word) {
    const auto bits = static_cast<std::uint8_t>(word);
    if (!gem16::nvfp4::IsFiniteE4M3Fn(bits)) continue;
    const auto encoded = gem16::nvfp4::EncodeE4M3Fn(
        gem16::nvfp4::DecodeE4M3Fn(bits));
    GEM16_CHECK(encoded.ok());
    if (encoded.ok()) GEM16_CHECK(encoded.value() == bits);
  }

  const auto saturated = gem16::nvfp4::EncodeE4M3Fn(1000.0F);
  GEM16_CHECK(saturated.ok() && saturated.value() == 0x7EU);
  GEM16_CHECK(!gem16::nvfp4::EncodeE4M3Fn(
                       std::numeric_limits<float>::quiet_NaN())
                       .ok());
}

void TestActivationQuantizationAndDivisors() {
  std::array<float, 16> activation{};
  for (std::size_t index = 0; index < activation.size(); ++index) {
    activation[index] = gem16::nvfp4::DecodeE2M1(static_cast<std::uint8_t>(index)) / 2.0F;
  }
  const auto quantized = gem16::nvfp4::QuantizeActivation(activation, 2.0F);
  GEM16_CHECK(quantized.ok());
  if (!quantized.ok()) return;

  constexpr std::array<std::uint8_t, 8> expected_packed = {
      0x10U, 0x32U, 0x54U, 0x76U, 0x98U, 0xBAU, 0xDCU, 0xFEU,
  };
  GEM16_CHECK(quantized.value().packed_e2m1.size() == expected_packed.size());
  GEM16_CHECK(std::equal(quantized.value().packed_e2m1.begin(),
                          quantized.value().packed_e2m1.end(), expected_packed.begin()));
  GEM16_CHECK(quantized.value().block_scales_e4m3fn.size() == 1U);
  GEM16_CHECK(quantized.value().block_scales_e4m3fn[0] == 0x38U);

  const auto dot = gem16::nvfp4::ReferenceDotProduct(
      quantized.value(), expected_packed, std::array<std::uint8_t, 1>{0x38U}, 4.0F);
  GEM16_CHECK(dot.ok());
  if (dot.ok()) GEM16_CHECK(dot.value() == 17.125);

  std::array<float, 16> zeros{};
  const auto zero_quantized = gem16::nvfp4::QuantizeActivation(zeros, 11.0F);
  GEM16_CHECK(zero_quantized.ok());
  if (zero_quantized.ok()) {
    GEM16_CHECK(zero_quantized.value().block_scales_e4m3fn[0] == 0U);
    for (const std::uint8_t byte : zero_quantized.value().packed_e2m1) {
      GEM16_CHECK(byte == 0U);
    }
  }
}

void TestPinnedCheckpointByteFixture() {
  // First 16 packed values of layer 0 Gate row 0 at pinned revision
  // b1f649734b34aa5575b03d186abd1b9be3d0d5c4. The first local scale byte is
  // 0x61 (36.0), and the stored weight global divisor is 9600.0.
  constexpr std::array<std::uint8_t, 8> packed_weight = {
      0x37U, 0xC1U, 0x53U, 0xA0U, 0xFBU, 0x5DU, 0xADU, 0xFEU,
  };
  constexpr std::array<std::uint8_t, 16> expected_nibbles = {
      7U, 3U, 1U, 12U, 3U, 5U, 0U, 10U, 11U, 15U, 13U, 5U, 13U, 10U, 14U, 15U,
  };

  for (std::size_t selected = 0; selected < expected_nibbles.size(); ++selected) {
    gem16::nvfp4::QuantizedActivation one_hot;
    one_hot.logical_elements = 16;
    one_hot.global_divisor = 1.0F;
    one_hot.packed_e2m1.resize(8, 0U);
    one_hot.block_scales_e4m3fn = {0x38U};
    const unsigned shift = selected % 2U == 0U ? 0U : 4U;
    one_hot.packed_e2m1[selected / 2U] = static_cast<std::uint8_t>(2U << shift);

    const auto dot = gem16::nvfp4::ReferenceDotProduct(
        one_hot, packed_weight, std::array<std::uint8_t, 1>{0x61U}, 9600.0F);
    GEM16_CHECK(dot.ok());
    if (dot.ok()) {
      const double expected =
          static_cast<double>(gem16::nvfp4::DecodeE2M1(expected_nibbles[selected])) *
          36.0 / 9600.0;
      GEM16_CHECK(std::fabs(dot.value() - expected) < 1.0e-12);
    }
  }
}

void TestInvalidOracleInputs() {
  std::array<float, 15> short_activation{};
  GEM16_CHECK(!gem16::nvfp4::QuantizeActivation(short_activation, 1.0F).ok());
  std::array<float, 16> activation{};
  GEM16_CHECK(!gem16::nvfp4::QuantizeActivation(activation, 0.0F).ok());
  activation[3] = std::numeric_limits<float>::quiet_NaN();
  GEM16_CHECK(!gem16::nvfp4::QuantizeActivation(activation, 1.0F).ok());

  gem16::nvfp4::QuantizedActivation malformed;
  malformed.logical_elements = 16;
  malformed.global_divisor = 1.0F;
  malformed.packed_e2m1.resize(8, 0U);
  malformed.block_scales_e4m3fn = {0x7FU};
  GEM16_CHECK(!gem16::nvfp4::ReferenceDotProduct(
                       malformed, std::array<std::uint8_t, 8>{},
                       std::array<std::uint8_t, 1>{0x38U}, 1.0F)
                       .ok());
}

}  // namespace

void RunNvfp4Tests() {
  TestE2M1Codec();
  TestE4M3FnCodec();
  TestActivationQuantizationAndDivisors();
  TestPinnedCheckpointByteFixture();
  TestInvalidOracleInputs();
}
