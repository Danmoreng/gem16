#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

#include "gem16/status.h"

namespace gem16::internal {

struct Trellis35DeviceImageUploadStats {
  std::uint64_t uploaded_bytes = 0U;
  std::uint64_t host_staging_peak_bytes = 0U;
  double upload_milliseconds = 0.0;
  bool cufile_attempted = false;
  bool used_cufile = false;
  std::string load_path;
};

// Startup-only upload of an already verified, final-layout payload. This path
// performs no payload SHA-256 pass. Integrity belongs to offline packaging and
// download/install verification; runtime validates structure and exact extent.
[[nodiscard]] Result<Trellis35DeviceImageUploadStats>
UploadGemma4Moe26BTrellis35DeviceImage(
    const std::filesystem::path& path, std::byte* destination,
    std::uint64_t bytes);

}  // namespace gem16::internal
