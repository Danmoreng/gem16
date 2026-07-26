#include "cuda/nvfp4/sm120_layout.h"

#include "test.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

std::uint8_t SourceNibble(const std::vector<std::uint8_t>& packed, std::size_t row,
                          std::size_t k) {
  const std::uint8_t byte = packed[row * 32U + k / 2U];
  return static_cast<std::uint8_t>((byte >> (k % 2U == 0U ? 0U : 4U)) & 0x0FU);
}

void StoreSourceNibble(std::vector<std::uint8_t>& packed, std::size_t row, std::size_t k,
                       std::uint8_t nibble) {
  std::uint8_t& byte = packed[row * 32U + k / 2U];
  const unsigned shift = k % 2U == 0U ? 0U : 4U;
  byte = static_cast<std::uint8_t>(byte | static_cast<std::uint8_t>(nibble << shift));
}

void TestRealCheckpointGeometry() {
  const auto gate = gem16::internal::PlanSm120Nvfp4SourceLayout(15360, 3840);
  GEM16_CHECK(gate.ok());
  if (gate.ok()) {
    GEM16_CHECK(gate.value().row_tiles == 1920U);
    GEM16_CHECK(gate.value().k_blocks == 60U);
    GEM16_CHECK(gate.value().packed_weight_bytes == 29491200U);
    GEM16_CHECK(gate.value().scale_bytes == 3686400U);
    GEM16_CHECK(gate.value().persistent_repack_bytes == 0U);
  }

  const auto down = gem16::internal::PlanSm120Nvfp4SourceLayout(3840, 15360);
  GEM16_CHECK(down.ok());
  if (down.ok()) {
    GEM16_CHECK(down.value().row_tiles == 480U);
    GEM16_CHECK(down.value().k_blocks == 240U);
    GEM16_CHECK(down.value().packed_weight_bytes == 29491200U);
    GEM16_CHECK(down.value().scale_bytes == 3686400U);
    GEM16_CHECK(down.value().persistent_repack_bytes == 0U);
  }
}

void TestDirectLaneMappingRoundTrip() {
  constexpr std::size_t rows = 8;
  constexpr std::size_t k_size = 64;
  std::vector<std::uint8_t> packed(rows * k_size / 2U, 0U);
  std::vector<std::uint8_t> scales(rows * k_size / 16U, 0U);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t k = 0; k < k_size; ++k) {
      StoreSourceNibble(packed, row, k,
                        static_cast<std::uint8_t>((row * 5U + k * 3U) & 0x0FU));
    }
    for (std::size_t group = 0; group < k_size / 16U; ++group) {
      scales[row * 4U + group] = static_cast<std::uint8_t>(0x20U + row * 4U + group);
    }
  }

  const auto layout = gem16::internal::PlanSm120Nvfp4SourceLayout(rows, k_size);
  GEM16_CHECK(layout.ok());
  if (!layout.ok()) return;
  const auto tiled_weights =
      gem16::internal::TileSm120Nvfp4Weights(layout.value(), packed);
  const auto tiled_scales =
      gem16::internal::TileSm120Nvfp4WeightScales(layout.value(), scales);
  GEM16_CHECK(tiled_weights.ok());
  GEM16_CHECK(tiled_scales.ok());
  if (!tiled_weights.ok() || !tiled_scales.ok()) return;

  std::vector<std::uint8_t> reconstructed(packed.size(), 0U);
  for (std::uint32_t lane = 0; lane < 32U; ++lane) {
    const auto fragment = gem16::internal::LoadSm120Nvfp4WeightLaneFragment(
        layout.value(), tiled_weights.value(), tiled_scales.value(), 0, 0, lane);
    GEM16_CHECK(fragment.ok());
    if (!fragment.ok()) continue;
    GEM16_CHECK(fragment.value().active);
    const std::size_t row = lane / 4U;
    const std::size_t quarter = lane % 4U;
    for (std::size_t word = 0; word < 2U; ++word) {
      for (std::size_t nibble = 0; nibble < 8U; ++nibble) {
        const std::size_t k = quarter * 8U + nibble + word * 32U;
        const auto value = static_cast<std::uint8_t>(
            (fragment.value().packed_e2m1[word] >> (nibble * 4U)) & 0x0FU);
        StoreSourceNibble(reconstructed, row, k, value);
      }
    }
    const std::uint32_t expected_scales =
        static_cast<std::uint32_t>(scales[row * 4U]) |
        (static_cast<std::uint32_t>(scales[row * 4U + 1U]) << 8U) |
        (static_cast<std::uint32_t>(scales[row * 4U + 2U]) << 16U) |
        (static_cast<std::uint32_t>(scales[row * 4U + 3U]) << 24U);
    GEM16_CHECK(fragment.value().packed_e4m3fn_scales == expected_scales);
  }

  GEM16_CHECK(reconstructed == packed);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t k = 0; k < k_size; ++k) {
      GEM16_CHECK(SourceNibble(reconstructed, row, k) == SourceNibble(packed, row, k));
    }
  }
}

void TestWeightTilingIsExactAndCoalesced() {
  constexpr std::size_t rows = 9;
  constexpr std::size_t k_size = 128;
  constexpr std::size_t packed_bytes_per_k_block = 32;
  const auto layout = gem16::internal::PlanSm120Nvfp4SourceLayout(rows, k_size);
  GEM16_CHECK(layout.ok());
  if (!layout.ok()) return;
  std::vector<std::uint8_t> source(layout.value().packed_weight_bytes);
  for (std::size_t index = 0; index < source.size(); ++index) {
    source[index] = static_cast<std::uint8_t>((index * 29U + 7U) & 0xFFU);
  }
  const auto tiled =
      gem16::internal::TileSm120Nvfp4Weights(layout.value(), source);
  GEM16_CHECK(tiled.ok());
  if (!tiled.ok()) return;
  GEM16_CHECK(tiled.value().size() == source.size());

  const std::size_t k_blocks = k_size / 64U;
  const std::size_t source_row_bytes = k_size / 2U;
  for (std::size_t row_tile = 0; row_tile < 2U; ++row_tile) {
    const std::size_t first_row = row_tile * 8U;
    const std::size_t tile_rows = std::min<std::size_t>(8U, rows - first_row);
    for (std::size_t k_block = 0; k_block < k_blocks; ++k_block) {
      for (std::size_t row = 0; row < tile_rows; ++row) {
        for (std::size_t byte = 0; byte < packed_bytes_per_k_block; ++byte) {
          const std::size_t source_offset =
              (first_row + row) * source_row_bytes +
              k_block * packed_bytes_per_k_block + byte;
          const std::size_t tiled_offset =
              first_row * k_blocks * packed_bytes_per_k_block +
              (k_block * tile_rows + row) * packed_bytes_per_k_block + byte;
          GEM16_CHECK(tiled.value()[tiled_offset] == source[source_offset]);
        }
      }
    }
  }
  GEM16_CHECK(!gem16::internal::TileSm120Nvfp4Weights(
                       layout.value(), std::span<const std::uint8_t>(source).first(1U))
                       .ok());
}

void TestScaleTilingIsExactAndCoalesced() {
  constexpr std::size_t rows = 9;
  constexpr std::size_t k_size = 128;
  const auto layout = gem16::internal::PlanSm120Nvfp4SourceLayout(rows, k_size);
  GEM16_CHECK(layout.ok());
  if (!layout.ok()) return;
  std::vector<std::uint8_t> source(layout.value().scale_bytes);
  for (std::size_t index = 0; index < source.size(); ++index) {
    source[index] = static_cast<std::uint8_t>((index * 37U + 11U) & 0xFFU);
  }
  const auto tiled =
      gem16::internal::TileSm120Nvfp4WeightScales(layout.value(), source);
  GEM16_CHECK(tiled.ok());
  if (!tiled.ok()) return;
  GEM16_CHECK(tiled.value().size() == source.size());

  constexpr std::size_t scales_per_k_block = 4;
  const std::size_t k_blocks = k_size / 64U;
  for (std::size_t row_tile = 0; row_tile < 2U; ++row_tile) {
    const std::size_t first_row = row_tile * 8U;
    const std::size_t tile_rows = std::min<std::size_t>(8U, rows - first_row);
    for (std::size_t k_block = 0; k_block < k_blocks; ++k_block) {
      for (std::size_t row = 0; row < tile_rows; ++row) {
        for (std::size_t scale = 0; scale < scales_per_k_block; ++scale) {
          const std::size_t source_offset =
              (first_row + row) * (k_size / 16U) +
              k_block * scales_per_k_block + scale;
          const std::size_t tiled_offset =
              first_row * k_blocks * scales_per_k_block +
              (k_block * tile_rows + row) * scales_per_k_block + scale;
          GEM16_CHECK(tiled.value()[tiled_offset] == source[source_offset]);
        }
      }
    }
  }
  GEM16_CHECK(!gem16::internal::TileSm120Nvfp4WeightScales(
                       layout.value(), std::span<const std::uint8_t>(source).first(1U))
                       .ok());
}

void TestTailRowsAndValidation() {
  const auto layout = gem16::internal::PlanSm120Nvfp4SourceLayout(9, 64);
  GEM16_CHECK(layout.ok());
  if (!layout.ok()) return;
  std::vector<std::uint8_t> packed(layout.value().packed_weight_bytes, 0U);
  std::vector<std::uint8_t> scales(layout.value().scale_bytes, 0x38U);
  const auto tiled_weights =
      gem16::internal::TileSm120Nvfp4Weights(layout.value(), packed);
  const auto tiled_scales =
      gem16::internal::TileSm120Nvfp4WeightScales(layout.value(), scales);
  GEM16_CHECK(tiled_weights.ok());
  GEM16_CHECK(tiled_scales.ok());
  if (!tiled_weights.ok() || !tiled_scales.ok()) return;
  const auto active = gem16::internal::LoadSm120Nvfp4WeightLaneFragment(
      layout.value(), tiled_weights.value(), tiled_scales.value(), 1, 0, 0);
  const auto inactive = gem16::internal::LoadSm120Nvfp4WeightLaneFragment(
      layout.value(), tiled_weights.value(), tiled_scales.value(), 1, 0, 4);
  GEM16_CHECK(active.ok() && active.value().active && active.value().source_row == 8U);
  GEM16_CHECK(inactive.ok() && !inactive.value().active && inactive.value().source_row == 9U);
  GEM16_CHECK(!gem16::internal::PlanSm120Nvfp4SourceLayout(8, 48).ok());
  GEM16_CHECK(!gem16::internal::LoadSm120Nvfp4WeightLaneFragment(
                       layout.value(), tiled_weights.value(), tiled_scales.value(), 2, 0, 0)
                       .ok());
}

}  // namespace

void RunSm120LayoutTests() {
  TestRealCheckpointGeometry();
  TestDirectLaneMappingRoundTrip();
  TestWeightTilingIsExactAndCoalesced();
  TestScaleTilingIsExactAndCoalesced();
  TestTailRowsAndValidation();
}
