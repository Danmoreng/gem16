#include "test.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>

#include "util/file_sha256.h"

void RunFileSha256Tests() {
  using gem16::internal::Sha256File;
  using gem16::internal::Sha256Range;
  const auto root = std::filesystem::temp_directory_path() /
      ("gem16-file-sha256-" + std::to_string(
          std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::create_directory(root);
  const auto path = root / "payload.bin";
  // Exercise both actual loader hash paths on the default Windows stack,
  // including exact chunk boundaries and a final partial read.
  for (const std::size_t size : {0U, 3U, 1024U * 1024U, 1024U * 1024U + 17U}) {
    std::string payload(size, '\0');
    for (std::size_t i = 0; i < size; ++i) {
      payload[i] = static_cast<char>(i % 251U);
    }
    {
      std::ofstream output(path, std::ios::binary);
      output.write(payload.data(), static_cast<std::streamsize>(size));
      GEM16_CHECK(output.good());
    }
    const auto expected = gem16::compiler::Sha256Hex(payload.data(), size);
    const auto file = Sha256File(path);
    GEM16_CHECK(file.ok());
    if (file.ok()) GEM16_CHECK(file.value() == expected);
    const auto range = Sha256Range(path, 0U, size);
    GEM16_CHECK(range.ok());
    if (range.ok()) GEM16_CHECK(range.value() == expected);
    if (size >= 3U) {
      const auto slice = Sha256Range(path, 2U, size - 3U);
      GEM16_CHECK(slice.ok());
      if (slice.ok()) {
        GEM16_CHECK(slice.value() == gem16::compiler::Sha256Hex(
            payload.data() + 2U, size - 3U));
      }
    }
    const auto short_read = Sha256Range(path, 0U, size + 1U);
    GEM16_CHECK(!short_read.ok());
    GEM16_CHECK(short_read.status().code() == gem16::StatusCode::kDataLoss);
  }
  GEM16_CHECK(!Sha256File(root / "missing.bin").ok());
  GEM16_CHECK(!Sha256Range(root / "missing.bin", 0U, 1U).ok());
  GEM16_CHECK(!Sha256Range(path, std::numeric_limits<std::uint64_t>::max(), 1U).ok());
  std::filesystem::remove_all(root);
}
