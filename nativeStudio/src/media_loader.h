#pragma once

#include "types.h"

#include <filesystem>
#include <string>

namespace gem16::studio {

inline constexpr std::uint64_t kMaxSingleAttachmentBytes = 10U * 1024U * 1024U;
inline constexpr std::size_t kMaxDocumentCharacters = 500000U;

[[nodiscard]] bool LoadMediaAttachment(const std::filesystem::path& path,
                                       MediaAttachment& attachment,
                                       std::string& error);
[[nodiscard]] std::string EncodeBase64(const std::vector<std::uint8_t>& bytes);
[[nodiscard]] std::string AttachmentPolicyError(ModelProfile profile,
                                                MediaKind kind,
                                                std::size_t existing_images);
[[nodiscard]] std::uint32_t EstimateVisionSoftTokens(
    const MediaAttachment& attachment, std::uint32_t budget);

}  // namespace gem16::studio
