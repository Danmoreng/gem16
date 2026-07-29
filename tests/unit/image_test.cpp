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
  const auto path = WriteSolidBmp();
  auto image = gem16::LoadVisionImage(path);
  GEM16_CHECK(image.ok());
  if (image.ok()) {
    const auto& value = image.value();
    GEM16_CHECK(value.source_width == 768U);
    GEM16_CHECK(value.source_height == 768U);
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
  std::error_code ignored;
  std::filesystem::remove(path, ignored);
}
