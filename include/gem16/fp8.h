#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "gem16/status.h"

namespace gem16::fp8 {

inline constexpr float kE4M3FnMax = 448.0F;

struct QuantizedToken {
  std::uint64_t logical_elements = 0;
  float scale = 1.0F;
  std::vector<std::uint8_t> values_e4m3fn;
};

[[nodiscard]] float DecodeE4M3Fn(std::uint8_t bits) noexcept;
[[nodiscard]] bool IsFiniteE4M3Fn(std::uint8_t bits) noexcept;
[[nodiscard]] Result<std::uint8_t> EncodeE4M3Fn(float value);
[[nodiscard]] float DecodeBf16(std::uint16_t bits) noexcept;

// Symmetric dynamic per-token quantization used by the checkpoint's attention projections.
// A nonzero token uses scale=max(abs(x))/448 and q=round_e4m3fn(x/scale). An all-zero token
// uses scale 1 so that both the stored scale and reconstructed values remain finite.
[[nodiscard]] Result<QuantizedToken> QuantizeToken(std::span<const float> activation);

// Binary64 oracle for one output row. Checkpoint weight bytes are row-major E4M3FN and the
// BF16 scale is per output channel: y[row] = sum(qx * qw) * input_scale * weight_scale[row].
[[nodiscard]] Result<double> ReferenceDotProduct(
    const QuantizedToken& activation,
    std::span<const std::uint8_t> weight_e4m3fn,
    std::uint16_t weight_scale_bf16);

}  // namespace gem16::fp8
