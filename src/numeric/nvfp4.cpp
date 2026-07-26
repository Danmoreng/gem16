#include "gem16/nvfp4.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <utility>

namespace gem16::nvfp4 {
namespace {

constexpr std::array<float, 8> kPositiveE2M1 = {
    0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F,
};
constexpr float kE2M1MaxReciprocal = 1.0F / kE2M1Max;

Status Invalid(std::string message) {
  return Status(StatusCode::kInvalidArgument, std::move(message));
}

bool IsPositiveFinite(float value) {
  return std::isfinite(value) && value > 0.0F;
}

std::uint8_t ChooseNearestE2M1(float magnitude) {
  std::uint8_t best = 0;
  float best_error = std::numeric_limits<float>::infinity();
  for (std::uint8_t candidate = 0; candidate < kPositiveE2M1.size(); ++candidate) {
    const float error = std::fabs(magnitude - kPositiveE2M1[candidate]);
    if (error < best_error ||
        (error == best_error && (candidate & 1U) == 0U && (best & 1U) != 0U)) {
      best = candidate;
      best_error = error;
    }
  }
  return best;
}

std::uint8_t ChooseNearestPositiveE4M3Fn(float magnitude) {
  if (magnitude >= kE4M3FnMax) return 0x7EU;

  std::uint8_t best = 0;
  float best_error = std::numeric_limits<float>::infinity();
  for (std::uint16_t candidate = 0; candidate <= 0x7EU; ++candidate) {
    const auto bits = static_cast<std::uint8_t>(candidate);
    const float decoded = DecodeE4M3Fn(bits);
    const float error = std::fabs(magnitude - decoded);
    if (error < best_error ||
        (error == best_error && (bits & 1U) == 0U && (best & 1U) != 0U)) {
      best = bits;
      best_error = error;
    }
  }
  return best;
}

std::uint8_t PackedNibble(std::span<const std::uint8_t> packed, std::size_t index) {
  const std::uint8_t byte = packed[index / kPackedElementsPerByte];
  const unsigned shift = (index % kPackedElementsPerByte == 0U) ? 0U : 4U;
  return static_cast<std::uint8_t>((byte >> shift) & 0x0FU);
}

void StorePackedNibble(std::span<std::uint8_t> packed, std::size_t index, std::uint8_t nibble) {
  std::uint8_t& byte = packed[index / kPackedElementsPerByte];
  const unsigned shift = (index % kPackedElementsPerByte == 0U) ? 0U : 4U;
  const std::uint8_t mask = static_cast<std::uint8_t>(0x0FU << shift);
  byte = static_cast<std::uint8_t>((byte & static_cast<std::uint8_t>(~mask)) |
                                   static_cast<std::uint8_t>((nibble & 0x0FU) << shift));
}

}  // namespace

float DecodeE2M1(std::uint8_t nibble) noexcept {
  nibble &= 0x0FU;
  const float magnitude = kPositiveE2M1[nibble & 0x07U];
  return (nibble & 0x08U) != 0U ? -magnitude : magnitude;
}

Result<std::uint8_t> EncodeE2M1(float value) {
  if (!std::isfinite(value)) return Invalid("E2M1 input must be finite");
  const std::uint8_t magnitude = ChooseNearestE2M1(std::fabs(value));
  const std::uint8_t sign = std::signbit(value) ? 0x08U : 0x00U;
  return static_cast<std::uint8_t>(sign | magnitude);
}

bool IsFiniteE4M3Fn(std::uint8_t bits) noexcept {
  return (bits & 0x7FU) != 0x7FU;
}

float DecodeE4M3Fn(std::uint8_t bits) noexcept {
  if (!IsFiniteE4M3Fn(bits)) return std::numeric_limits<float>::quiet_NaN();

  const bool negative = (bits & 0x80U) != 0U;
  const int exponent = static_cast<int>((bits >> 3U) & 0x0FU);
  const int mantissa = static_cast<int>(bits & 0x07U);
  float magnitude = 0.0F;
  if (exponent == 0) {
    magnitude = std::ldexp(static_cast<float>(mantissa), -9);
  } else {
    magnitude = std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F, exponent - 7);
  }
  return negative ? -magnitude : magnitude;
}

Result<std::uint8_t> EncodeE4M3Fn(float value) {
  if (!std::isfinite(value)) return Invalid("E4M3FN input must be finite");
  const std::uint8_t magnitude = ChooseNearestPositiveE4M3Fn(std::fabs(value));
  const std::uint8_t sign = std::signbit(value) ? 0x80U : 0x00U;
  return static_cast<std::uint8_t>(sign | magnitude);
}

Result<QuantizedActivation> QuantizeActivation(std::span<const float> activation,
                                               float global_divisor) {
  if (activation.empty() || activation.size() % kBlockElements != 0U) {
    return Invalid("NVFP4 activation extent must be a nonzero multiple of 16");
  }
  if (!IsPositiveFinite(global_divisor)) {
    return Invalid("NVFP4 activation global divisor must be positive and finite");
  }

  QuantizedActivation output;
  output.logical_elements = activation.size();
  output.global_divisor = global_divisor;
  output.packed_e2m1.resize(activation.size() / kPackedElementsPerByte, 0U);
  output.block_scales_e4m3fn.resize(activation.size() / kBlockElements, 0U);

  for (std::size_t block = 0; block < output.block_scales_e4m3fn.size(); ++block) {
    const std::size_t begin = block * kBlockElements;
    float amax = 0.0F;
    for (std::size_t local = 0; local < kBlockElements; ++local) {
      const float value = activation[begin + local];
      if (!std::isfinite(value)) return Invalid("NVFP4 activation values must be finite");
      const float scaled = value * global_divisor;
      if (!std::isfinite(scaled)) return Invalid("NVFP4 activation scaling overflowed");
      amax = std::max(amax, std::fabs(value));
    }

    const auto encoded_scale =
        EncodeE4M3Fn((amax * kE2M1MaxReciprocal) * global_divisor);
    if (!encoded_scale.ok()) return encoded_scale.status();
    output.block_scales_e4m3fn[block] = encoded_scale.value();
    const float decoded_scale = DecodeE4M3Fn(encoded_scale.value());

    for (std::size_t local = 0; local < kBlockElements; ++local) {
      const float scaled = activation[begin + local] * global_divisor;
      const float normalized = decoded_scale == 0.0F ? 0.0F : scaled / decoded_scale;
      const auto encoded = EncodeE2M1(normalized);
      if (!encoded.ok()) return encoded.status();
      StorePackedNibble(output.packed_e2m1, begin + local, encoded.value());
    }
  }
  return output;
}

Result<double> ReferenceDotProduct(const QuantizedActivation& activation,
                                   std::span<const std::uint8_t> packed_weight_e2m1,
                                   std::span<const std::uint8_t> weight_scales_e4m3fn,
                                   float weight_global_divisor) {
  if (activation.logical_elements == 0U ||
      activation.logical_elements % kBlockElements != 0U) {
    return Invalid("NVFP4 oracle activation extent must be a nonzero multiple of 16");
  }
  if (activation.logical_elements > std::numeric_limits<std::size_t>::max()) {
    return Invalid("NVFP4 oracle activation extent exceeds the host address space");
  }
  const auto elements = static_cast<std::size_t>(activation.logical_elements);
  if (activation.packed_e2m1.size() != elements / kPackedElementsPerByte ||
      activation.block_scales_e4m3fn.size() != elements / kBlockElements) {
    return Invalid("NVFP4 oracle activation storage does not match its logical extent");
  }
  if (packed_weight_e2m1.size() != elements / kPackedElementsPerByte ||
      weight_scales_e4m3fn.size() != elements / kBlockElements) {
    return Invalid("NVFP4 oracle weight storage does not match the activation extent");
  }
  if (!IsPositiveFinite(activation.global_divisor) ||
      !IsPositiveFinite(weight_global_divisor)) {
    return Invalid("NVFP4 oracle global divisors must be positive and finite");
  }

  double accumulator = 0.0;
  for (std::size_t index = 0; index < elements; ++index) {
    const std::size_t block = index / kBlockElements;
    const std::uint8_t activation_scale_bits = activation.block_scales_e4m3fn[block];
    const std::uint8_t weight_scale_bits = weight_scales_e4m3fn[block];
    if (!IsFiniteE4M3Fn(activation_scale_bits) || !IsFiniteE4M3Fn(weight_scale_bits)) {
      return Invalid("NVFP4 oracle encountered a non-finite E4M3FN scale");
    }
    const double activation_value =
        static_cast<double>(DecodeE2M1(PackedNibble(activation.packed_e2m1, index))) *
        static_cast<double>(DecodeE4M3Fn(activation_scale_bits));
    const double weight_value =
        static_cast<double>(DecodeE2M1(PackedNibble(packed_weight_e2m1, index))) *
        static_cast<double>(DecodeE4M3Fn(weight_scale_bits));
    accumulator += activation_value * weight_value;
  }

  const double divisor = static_cast<double>(activation.global_divisor) *
                         static_cast<double>(weight_global_divisor);
  return accumulator / divisor;
}

}  // namespace gem16::nvfp4
