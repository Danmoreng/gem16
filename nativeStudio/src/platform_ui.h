#pragma once

#include <filesystem>
#include <vector>
#include <string_view>

namespace gem16::studio {

[[nodiscard]] std::vector<std::filesystem::path> OpenAttachmentDialog();
[[nodiscard]] std::filesystem::path OpenExecutableDialog();
[[nodiscard]] std::filesystem::path OpenDirectoryDialog();
[[nodiscard]] std::vector<std::filesystem::path> DrainDroppedFiles();
void QueueDroppedFiles(const std::vector<std::filesystem::path>& paths);
void OpenInFileManager(const std::filesystem::path& path);
[[nodiscard]] bool IsSafeWebLink(std::string_view url);
bool OpenWebLink(std::string_view url);

}  // namespace gem16::studio
