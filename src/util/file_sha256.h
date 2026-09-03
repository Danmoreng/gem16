#pragma once

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "compiler/sha256.h"
#include "gem16/status.h"

namespace gem16::internal {

// Startup-only hashing. Keep the bounded 1 MiB staging allocation off the
// stack: it alone would exhaust the default Windows thread stack. The buffer
// is released before inference; no full checkpoint copy is retained.
inline constexpr std::size_t kFileHashBufferBytes = 1024U * 1024U;

[[nodiscard]] inline Result<std::string> Sha256File(
    const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return Status(StatusCode::kIoError,
                  "cannot hash Trellis35 file: " + path.string());
  }
  compiler::Sha256 digest;
  std::vector<char> buffer(kFileHashBufferBytes);
  while (input) {
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    if (input.gcount() > 0) {
      digest.Update(buffer.data(), static_cast<std::size_t>(input.gcount()));
    }
  }
  if (!input.eof()) {
    return Status(StatusCode::kIoError,
                  "failed while hashing Trellis35 file: " + path.string());
  }
  return digest.HexDigest();
}

[[nodiscard]] inline Result<std::string> Sha256Range(
    const std::filesystem::path& path, std::uint64_t offset,
    std::uint64_t length) {
  std::ifstream input(path, std::ios::binary);
  if (!input || offset > static_cast<std::uint64_t>(
                             std::numeric_limits<std::streamoff>::max())) {
    return Status(StatusCode::kIoError, "cannot open artifact file: " + path.string());
  }
  input.seekg(static_cast<std::streamoff>(offset));
  if (!input) {
    return Status(StatusCode::kIoError, "cannot seek artifact file: " + path.string());
  }
  compiler::Sha256 digest;
  std::vector<char> buffer(kFileHashBufferBytes);
  std::uint64_t remaining = length;
  while (remaining != 0) {
    const auto count = static_cast<std::streamsize>(
        std::min<std::uint64_t>(remaining, buffer.size()));
    input.read(buffer.data(), count);
    if (input.gcount() != count) {
      return Status(StatusCode::kDataLoss,
                    "short read while hashing artifact file: " + path.string());
    }
    digest.Update(buffer.data(), static_cast<std::size_t>(count));
    remaining -= static_cast<std::uint64_t>(count);
  }
  return digest.HexDigest();
}

}  // namespace gem16::internal
