#pragma once

#include <cstdint>
#include <filesystem>
#include <string_view>

#include "gem16/status.h"

namespace gem16::internal {

inline constexpr std::string_view kGemma4Moe26BDeviceImageFormat =
    "gem16-sm120-device-image-v1";
inline constexpr std::string_view kAcceptedM08DeviceImageSha256 =
    "1ed73cf105b68db937ac0992283d31fdb2225474204341440721f41fe871bb72";
inline constexpr std::uint64_t kAcceptedM08DeviceImageBytes =
    14'696'668'160ULL;

[[nodiscard]] std::filesystem::path Gemma4Moe26BDeviceImagePath(
    const std::filesystem::path& model_directory);

// A present but malformed image is an integrity error, not permission to
// silently fall back to the transformation loader. The image payload hash is
// verified while it is uploaded by the CUDA loader.
[[nodiscard]] Result<bool> ProbeAcceptedGemma4Moe26BDeviceImage(
    const std::filesystem::path& model_directory);

}  // namespace gem16::internal
