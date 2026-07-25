#include "cuda/nvfp4/sm120_layout.h"

#include <algorithm>
#include <limits>
#include <string>
#include <utility>

namespace gem16gb::internal {
namespace {

constexpr std::uint64_t kRowsPerTile = 8;
constexpr std::uint64_t kElementsPerKBlock = 64;
constexpr std::uint64_t kElementsPerScale = 16;
constexpr std::uint64_t kElementsPerPackedByte = 2;
constexpr std::uint32_t kWarpLanes = 32;

Status Invalid(std::string message) {
  return Status(StatusCode::kInvalidArgument, std::move(message));
}

Result<std::uint64_t> CheckedMultiply(std::uint64_t left, std::uint64_t right,
                                      const char* label) {
  if (right != 0U && left > std::numeric_limits<std::uint64_t>::max() / right) {
    return Invalid(std::string(label) + " overflow");
  }
  return left * right;
}

std::uint32_t LoadLittleU32(std::span<const std::uint8_t> source, std::uint64_t offset) {
  return static_cast<std::uint32_t>(source[static_cast<std::size_t>(offset)]) |
         (static_cast<std::uint32_t>(source[static_cast<std::size_t>(offset + 1U)]) << 8U) |
         (static_cast<std::uint32_t>(source[static_cast<std::size_t>(offset + 2U)]) << 16U) |
         (static_cast<std::uint32_t>(source[static_cast<std::size_t>(offset + 3U)]) << 24U);
}

}  // namespace

Result<Sm120Nvfp4SourceLayout> PlanSm120Nvfp4SourceLayout(
    std::uint64_t rows, std::uint64_t contracting_elements) {
  if (rows == 0U || contracting_elements == 0U ||
      contracting_elements % kElementsPerKBlock != 0U) {
    return Invalid("SM120 NVFP4 source layout requires positive dimensions and K divisible by 64");
  }
  const auto logical_elements = CheckedMultiply(rows, contracting_elements, "NVFP4 elements");
  if (!logical_elements.ok()) return logical_elements.status();

  Sm120Nvfp4SourceLayout layout;
  layout.rows = rows;
  layout.contracting_elements = contracting_elements;
  layout.row_tiles = (rows + kRowsPerTile - 1U) / kRowsPerTile;
  layout.k_blocks = contracting_elements / kElementsPerKBlock;
  layout.packed_weight_bytes = logical_elements.value() / kElementsPerPackedByte;
  layout.scale_bytes = logical_elements.value() / kElementsPerScale;
  layout.persistent_repack_bytes = 0;
  return layout;
}

Result<std::vector<std::uint8_t>> TileSm120Nvfp4WeightScales(
    const Sm120Nvfp4SourceLayout& layout,
    std::span<const std::uint8_t> source_weight_scales_e4m3fn) {
  const auto expected = PlanSm120Nvfp4SourceLayout(layout.rows, layout.contracting_elements);
  if (!expected.ok()) return expected.status();
  if (layout.row_tiles != expected.value().row_tiles ||
      layout.k_blocks != expected.value().k_blocks ||
      layout.packed_weight_bytes != expected.value().packed_weight_bytes ||
      layout.scale_bytes != expected.value().scale_bytes ||
      layout.persistent_repack_bytes != 0U) {
    return Invalid("SM120 NVFP4 source layout metadata is inconsistent");
  }
  if (source_weight_scales_e4m3fn.size() != layout.scale_bytes ||
      layout.scale_bytes > std::numeric_limits<std::size_t>::max()) {
    return Invalid("SM120 NVFP4 source scales do not match the planned byte count");
  }

  std::vector<std::uint8_t> tiled(static_cast<std::size_t>(layout.scale_bytes));
  const std::uint64_t source_row_bytes =
      layout.contracting_elements / kElementsPerScale;
  constexpr std::uint64_t kScalesPerKBlock =
      kElementsPerKBlock / kElementsPerScale;
  for (std::uint64_t row_tile = 0; row_tile < layout.row_tiles; ++row_tile) {
    const std::uint64_t first_row = row_tile * kRowsPerTile;
    const std::uint64_t tile_rows =
        std::min(kRowsPerTile, layout.rows - first_row);
    const std::uint64_t tile_offset = first_row * layout.k_blocks * kScalesPerKBlock;
    for (std::uint64_t k_block = 0; k_block < layout.k_blocks; ++k_block) {
      for (std::uint64_t row = 0; row < tile_rows; ++row) {
        const std::uint64_t source_offset =
            (first_row + row) * source_row_bytes +
            k_block * kScalesPerKBlock;
        const std::uint64_t destination_offset =
            tile_offset + (k_block * tile_rows + row) * kScalesPerKBlock;
        std::copy_n(source_weight_scales_e4m3fn.begin() +
                        static_cast<std::ptrdiff_t>(source_offset),
                    static_cast<std::size_t>(kScalesPerKBlock),
                    tiled.begin() + static_cast<std::ptrdiff_t>(destination_offset));
      }
    }
  }
  return tiled;
}

Result<Sm120Nvfp4WeightLaneFragment> LoadSm120Nvfp4WeightLaneFragment(
    const Sm120Nvfp4SourceLayout& layout,
    std::span<const std::uint8_t> packed_weight_e2m1,
    std::span<const std::uint8_t> weight_scales_e4m3fn,
    std::uint64_t row_tile,
    std::uint64_t k_block,
    std::uint32_t lane) {
  const auto expected = PlanSm120Nvfp4SourceLayout(layout.rows, layout.contracting_elements);
  if (!expected.ok()) return expected.status();
  if (layout.row_tiles != expected.value().row_tiles ||
      layout.k_blocks != expected.value().k_blocks ||
      layout.packed_weight_bytes != expected.value().packed_weight_bytes ||
      layout.scale_bytes != expected.value().scale_bytes || layout.persistent_repack_bytes != 0U) {
    return Invalid("SM120 NVFP4 source layout metadata is inconsistent");
  }
  if (packed_weight_e2m1.size() != layout.packed_weight_bytes ||
      weight_scales_e4m3fn.size() != layout.scale_bytes) {
    return Invalid("SM120 NVFP4 source buffers do not match the planned byte counts");
  }
  if (row_tile >= layout.row_tiles || k_block >= layout.k_blocks || lane >= kWarpLanes) {
    return Invalid("SM120 NVFP4 lane-fragment coordinate is out of range");
  }

  Sm120Nvfp4WeightLaneFragment fragment;
  const std::uint64_t row_in_tile = lane / 4U;
  const std::uint64_t k_quarter = lane % 4U;
  fragment.source_row = row_tile * kRowsPerTile + row_in_tile;
  if (fragment.source_row >= layout.rows) return fragment;

  const std::uint64_t packed_row_bytes = layout.contracting_elements / 2U;
  const std::uint64_t packed_k_block_offset = k_block * (kElementsPerKBlock / 2U);
  const std::uint64_t first_offset = fragment.source_row * packed_row_bytes +
                                     packed_k_block_offset + k_quarter * 4U;
  const std::uint64_t second_offset = first_offset + 16U;
  fragment.packed_e2m1[0] = LoadLittleU32(packed_weight_e2m1, first_offset);
  fragment.packed_e2m1[1] = LoadLittleU32(packed_weight_e2m1, second_offset);

  const std::uint64_t scale_row_bytes = layout.contracting_elements / kElementsPerScale;
  const std::uint64_t scale_offset = fragment.source_row * scale_row_bytes + k_block * 4U;
  fragment.packed_e4m3fn_scales = LoadLittleU32(weight_scales_e4m3fn, scale_offset);
  fragment.active = true;
  return fragment;
}

}  // namespace gem16gb::internal
