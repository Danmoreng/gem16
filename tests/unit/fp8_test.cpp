#include "gem16/fp8.h"

#include "test.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {

void TestBf16Decode() {
  GEM16_CHECK(gem16::fp8::DecodeBf16(0x3F80U) == 1.0F);
  GEM16_CHECK(gem16::fp8::DecodeBf16(0xC020U) == -2.5F);
  GEM16_CHECK(std::isinf(gem16::fp8::DecodeBf16(0x7F80U)));
}

void TestDynamicTokenQuantization() {
  constexpr std::array<float, 8> activation = {
      -4.0F, -2.0F, -1.0F, -0.5F, 0.0F, 0.5F, 2.0F, 4.0F};
  const auto quantized = gem16::fp8::QuantizeToken(activation);
  GEM16_CHECK(quantized.ok());
  if (!quantized.ok()) return;
  GEM16_CHECK(quantized.value().scale == 4.0F / 448.0F);
  GEM16_CHECK(quantized.value().values_e4m3fn.front() == 0xFEU);
  GEM16_CHECK(quantized.value().values_e4m3fn.back() == 0x7EU);

  std::array<float, 8> reconstructed{};
  for (std::size_t index = 0; index < reconstructed.size(); ++index) {
    reconstructed[index] =
        gem16::fp8::DecodeE4M3Fn(quantized.value().values_e4m3fn[index]) *
        quantized.value().scale;
    GEM16_CHECK(std::fabs(reconstructed[index] - activation[index]) < 0.02F);
  }

  constexpr std::array<float, 4> zeros{};
  const auto zero = gem16::fp8::QuantizeToken(zeros);
  GEM16_CHECK(zero.ok());
  if (zero.ok()) {
    GEM16_CHECK(zero.value().scale == 1.0F);
    for (const std::uint8_t value : zero.value().values_e4m3fn) {
      GEM16_CHECK(value == 0U);
    }
  }
}

void TestProjectionOracle() {
  constexpr std::array<float, 4> activation = {1.0F, -2.0F, 3.0F, -4.0F};
  const auto quantized = gem16::fp8::QuantizeToken(activation);
  GEM16_CHECK(quantized.ok());
  if (!quantized.ok()) return;
  constexpr std::array<std::uint8_t, 4> weight = {0x38U, 0xB8U, 0x40U, 0xC0U};
  const auto dot = gem16::fp8::ReferenceDotProduct(
      quantized.value(), weight, 0x3F00U);  // BF16 0.5
  GEM16_CHECK(dot.ok());
  if (dot.ok()) {
    double expected = 0.0;
    for (std::size_t index = 0; index < weight.size(); ++index) {
      expected += static_cast<double>(gem16::fp8::DecodeE4M3Fn(
                      quantized.value().values_e4m3fn[index])) *
                  static_cast<double>(gem16::fp8::DecodeE4M3Fn(weight[index]));
    }
    expected *= static_cast<double>(quantized.value().scale) * 0.5;
    GEM16_CHECK(dot.value() == expected);
  }
}

void TestInvalidInputs() {
  std::array<float, 1> invalid = {std::numeric_limits<float>::quiet_NaN()};
  GEM16_CHECK(!gem16::fp8::QuantizeToken(invalid).ok());
  GEM16_CHECK(!gem16::fp8::QuantizeToken(std::span<const float>{}).ok());

  gem16::fp8::QuantizedToken malformed;
  malformed.logical_elements = 1;
  malformed.scale = 1.0F;
  malformed.values_e4m3fn = {0x7FU};
  GEM16_CHECK(!gem16::fp8::ReferenceDotProduct(
                       malformed, std::array<std::uint8_t, 1>{0x38U}, 0x3F80U)
                       .ok());
}

}  // namespace

void RunFp8Tests() {
  TestBf16Decode();
  TestDynamicTokenQuantization();
  TestProjectionOracle();
  TestInvalidInputs();
}
