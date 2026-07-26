#include "gem16/fp8.h"

#include "gem16/nvfp4.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <utility>

namespace gem16::fp8 {
namespace {

Status Invalid(std::string message) {
  return Status(StatusCode::kInvalidArgument, std::move(message));
}

}  // namespace

float DecodeE4M3Fn(std::uint8_t bits) noexcept {
  return nvfp4::DecodeE4M3Fn(bits);
}

bool IsFiniteE4M3Fn(std::uint8_t bits) noexcept {
  return nvfp4::IsFiniteE4M3Fn(bits);
}

Result<std::uint8_t> EncodeE4M3Fn(float value) {
  return nvfp4::EncodeE4M3Fn(value);
}

float DecodeBf16(std::uint16_t bits) noexcept {
  return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U);
}

Result<QuantizedToken> QuantizeToken(std::span<const float> activation) {
  if (activation.empty()) return Invalid("FP8 token activation must be nonempty");

  float absolute_maximum = 0.0F;
  for (const float value : activation) {
    if (!std::isfinite(value)) return Invalid("FP8 token activation values must be finite");
    absolute_maximum = std::max(absolute_maximum, std::fabs(value));
  }

  QuantizedToken output;
  output.logical_elements = activation.size();
  output.scale = absolute_maximum == 0.0F ? 1.0F : absolute_maximum / kE4M3FnMax;
  if (!std::isfinite(output.scale) || output.scale <= 0.0F) {
    return Invalid("FP8 token activation scale must be positive and finite");
  }
  output.values_e4m3fn.reserve(activation.size());
  for (const float value : activation) {
    const auto encoded = EncodeE4M3Fn(value / output.scale);
    if (!encoded.ok()) return encoded.status();
    output.values_e4m3fn.push_back(encoded.value());
  }
  return output;
}

Result<double> ReferenceDotProduct(const QuantizedToken& activation,
                                   std::span<const std::uint8_t> weight_e4m3fn,
                                   std::uint16_t weight_scale_bf16) {
  if (activation.logical_elements == 0U ||
      activation.logical_elements > std::numeric_limits<std::size_t>::max() ||
      activation.values_e4m3fn.size() != activation.logical_elements ||
      weight_e4m3fn.size() != activation.logical_elements) {
    return Invalid("FP8 oracle storage does not match the logical token extent");
  }
  if (!std::isfinite(activation.scale) || activation.scale <= 0.0F) {
    return Invalid("FP8 oracle activation scale must be positive and finite");
  }
  const float weight_scale = DecodeBf16(weight_scale_bf16);
  if (!std::isfinite(weight_scale) || weight_scale <= 0.0F) {
    return Invalid("FP8 oracle weight scale must be positive and finite");
  }

  double accumulator = 0.0;
  for (std::size_t index = 0; index < weight_e4m3fn.size(); ++index) {
    if (!IsFiniteE4M3Fn(activation.values_e4m3fn[index]) ||
        !IsFiniteE4M3Fn(weight_e4m3fn[index])) {
      return Invalid("FP8 oracle encountered a non-finite E4M3FN value");
    }
    accumulator += static_cast<double>(DecodeE4M3Fn(activation.values_e4m3fn[index])) *
                   static_cast<double>(DecodeE4M3Fn(weight_e4m3fn[index]));
  }
  return accumulator * static_cast<double>(activation.scale) *
         static_cast<double>(weight_scale);
}

}  // namespace gem16::fp8
