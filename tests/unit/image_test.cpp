#include "gem16/image.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

#include "test.h"

namespace {

void Put16(std::vector<std::uint8_t>& bytes, std::uint16_t value) {
  bytes.push_back(static_cast<std::uint8_t>(value));
  bytes.push_back(static_cast<std::uint8_t>(value >> 8U));
}

void Put32(std::vector<std::uint8_t>& bytes, std::uint32_t value) {
  for (unsigned shift = 0U; shift < 32U; shift += 8U) {
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
  }
}

std::filesystem::path WriteSolidBmp() {
  constexpr std::uint32_t kWidth = 768U;
  constexpr std::uint32_t kHeight = 768U;
  constexpr std::uint32_t kStride = kWidth * 3U;
  constexpr std::uint32_t kPixelBytes = kStride * kHeight;
  std::vector<std::uint8_t> bytes;
  bytes.reserve(54U + kPixelBytes);
  bytes.push_back('B');
  bytes.push_back('M');
  Put32(bytes, 54U + kPixelBytes);
  Put16(bytes, 0U);
  Put16(bytes, 0U);
  Put32(bytes, 54U);
  Put32(bytes, 40U);
  Put32(bytes, kWidth);
  Put32(bytes, kHeight);
  Put16(bytes, 1U);
  Put16(bytes, 24U);
  Put32(bytes, 0U);
  Put32(bytes, kPixelBytes);
  Put32(bytes, 2835U);
  Put32(bytes, 2835U);
  Put32(bytes, 0U);
  Put32(bytes, 0U);
  for (std::uint32_t pixel = 0U; pixel < kWidth * kHeight; ++pixel) {
    bytes.push_back(255U);  // B
    bytes.push_back(128U);  // G
    bytes.push_back(64U);   // R
  }
  const auto path = std::filesystem::temp_directory_path() /
                    "gem16_vision_solid.bmp";
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  return path;
}

}  // namespace

void RunImageTests() {
  GEM16_CHECK(gem16::AutomaticVisionSoftTokenBudget(1024U, 200U, 2U) ==
              280U);
  GEM16_CHECK(gem16::AutomaticVisionSoftTokenBudget(512U, 200U, 2U) ==
              156U);
  GEM16_CHECK(gem16::AutomaticVisionSoftTokenBudget(128U, 200U, 2U) ==
              1U);
  const auto path = WriteSolidBmp();
  auto image = gem16::LoadVisionImage(path);
  GEM16_CHECK(image.ok());
  if (image.ok()) {
    const auto& value = image.value();
    GEM16_CHECK(value.source_width == 768U);
    GEM16_CHECK(value.source_height == 768U);
    GEM16_CHECK(value.processed_width == 768U);
    GEM16_CHECK(value.processed_height == 768U);
    GEM16_CHECK(value.soft_token_budget == 280U);
    GEM16_CHECK(value.patch_count == 256U);
    GEM16_CHECK(value.positions.size() == 512U);
    GEM16_CHECK(value.positions[0] == 0);
    GEM16_CHECK(value.positions[1] == 0);
    GEM16_CHECK(value.positions[510] == 15);
    GEM16_CHECK(value.positions[511] == 15);
    GEM16_CHECK(value.patches.size() == 256U * 48U * 48U * 3U);
    GEM16_CHECK(std::abs(value.patches[0] - 64.0F / 255.0F) < 1.0e-6F);
    GEM16_CHECK(std::abs(value.patches[1] - 128.0F / 255.0F) < 1.0e-6F);
    GEM16_CHECK(value.patches[2] == 1.0F);
    GEM16_CHECK(value.patches.back() == 1.0F);
  }
  std::ifstream encoded_input(path, std::ios::binary);
  const std::vector<std::uint8_t> encoded(
      std::istreambuf_iterator<char>(encoded_input), {});
  auto memory_image = gem16::LoadVisionImageBytes(encoded, "unit BMP");
  GEM16_CHECK(memory_image.ok());
  if (memory_image.ok()) {
    GEM16_CHECK(memory_image.value().patch_count == 256U);
    GEM16_CHECK(memory_image.value().source_fingerprint != 0U);
  }
  auto compact = gem16::LoadVisionImage(
      path, gem16::VisionImageOptions{70U, false});
  GEM16_CHECK(compact.ok());
  if (compact.ok()) {
    GEM16_CHECK(compact.value().patch_count == 64U);
    GEM16_CHECK(compact.value().processed_width == 384U);
    GEM16_CHECK(compact.value().processed_height == 384U);
    GEM16_CHECK(compact.value().soft_token_budget == 70U);
    if (memory_image.ok()) {
      GEM16_CHECK(compact.value().source_fingerprint ==
                  memory_image.value().source_fingerprint);
    }
  }
  GEM16_CHECK(!gem16::LoadVisionImage(
                   path, gem16::VisionImageOptions{281U, false})
                   .ok());
  auto moe26b = gem16::LoadGemma4Moe26BVisionImage(path);
  GEM16_CHECK(moe26b.ok());
  if (moe26b.ok()) {
    const auto& value = moe26b.value();
    GEM16_CHECK(value.source_width == 768U);
    GEM16_CHECK(value.source_height == 768U);
    GEM16_CHECK(value.processed_width == 768U);
    GEM16_CHECK(value.processed_height == 768U);
    GEM16_CHECK(value.raw_patch_count == 2304U);
    GEM16_CHECK(value.soft_token_count == 256U);
    GEM16_CHECK(value.positions.size() == 4608U);
    GEM16_CHECK(value.positions[0] == 0);
    GEM16_CHECK(value.positions[1] == 0);
    GEM16_CHECK(value.positions[4606] == 47);
    GEM16_CHECK(value.positions[4607] == 47);
    GEM16_CHECK(value.patches.size() == 2304U * 16U * 16U * 3U);
    GEM16_CHECK(std::abs(value.patches[0] - 64.0F / 255.0F) < 1.0e-6F);
    GEM16_CHECK(std::abs(value.patches[1] - 128.0F / 255.0F) < 1.0e-6F);
    GEM16_CHECK(value.patches[2] == 1.0F);
    GEM16_CHECK(value.patches.back() == 1.0F);
  }
  auto moe26b_compact = gem16::LoadGemma4Moe26BVisionImage(
      path, gem16::Gemma4Moe26BVisionImageOptions{70U});
  GEM16_CHECK(moe26b_compact.ok());
  if (moe26b_compact.ok()) {
    GEM16_CHECK(moe26b_compact.value().processed_width == 384U);
    GEM16_CHECK(moe26b_compact.value().processed_height == 384U);
    GEM16_CHECK(moe26b_compact.value().raw_patch_count == 576U);
    GEM16_CHECK(moe26b_compact.value().soft_token_count == 64U);
  }
  GEM16_CHECK(!gem16::LoadGemma4Moe26BVisionImage(
                   path, gem16::Gemma4Moe26BVisionImageOptions{71U})
                   .ok());
  std::error_code ignored;
  std::filesystem::remove(path, ignored);
}
