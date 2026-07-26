#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "gem16/status.h"

namespace gem16::nvfp4 {

inline constexpr std::size_t kBlockElements = 16;
inline constexpr std::size_t kPackedElementsPerByte = 2;
inline constexpr float kE2M1Max = 6.0F;
inline constexpr float kE4M3FnMax = 448.0F;

struct QuantizedActivation {
  std::uint64_t logical_elements = 0;
  float global_divisor = 1.0F;
  std::vector<std::uint8_t> packed_e2m1;
  std::vector<std::uint8_t> block_scales_e4m3fn;
};

[[nodiscard]] float DecodeE2M1(std::uint8_t nibble) noexcept;
[[nodiscard]] Result<std::uint8_t> EncodeE2M1(float value);

[[nodiscard]] float DecodeE4M3Fn(std::uint8_t bits) noexcept;
[[nodiscard]] bool IsFiniteE4M3Fn(std::uint8_t bits) noexcept;
[[nodiscard]] Result<std::uint8_t> EncodeE4M3Fn(float value);

// Quantizes `activation * global_divisor` in independent groups of 16. The local scale is
// round-to-nearest-even E4M3FN and two E2M1 values are stored per byte, with the even logical
// element in the low nibble. Inputs must be finite and the logical extent must be divisible by 16.
[[nodiscard]] Result<QuantizedActivation> QuantizeActivation(
    std::span<const float> activation,
    float global_divisor);

// Independent mathematical oracle for one W4A4 projection row. The packed weight uses the same
// low-nibble-first source layout as the pinned Safetensors checkpoint. Local represented values
// are accumulated in binary64, then divided by the stored activation and weight global divisors.
[[nodiscard]] Result<double> ReferenceDotProduct(
    const QuantizedActivation& activation,
    std::span<const std::uint8_t> packed_weight_e2m1,
    std::span<const std::uint8_t> weight_scales_e4m3fn,
    float weight_global_divisor);

}  // namespace gem16::nvfp4
