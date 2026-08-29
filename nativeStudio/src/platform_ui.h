#pragma once

#include <filesystem>
#include <vector>

namespace gem16::studio {

[[nodiscard]] std::vector<std::filesystem::path> OpenAttachmentDialog();
[[nodiscard]] std::filesystem::path OpenExecutableDialog();
[[nodiscard]] std::filesystem::path OpenDirectoryDialog();
[[nodiscard]] std::vector<std::filesystem::path> DrainDroppedFiles();
void QueueDroppedFiles(const std::vector<std::filesystem::path>& paths);
void OpenInFileManager(const std::filesystem::path& path);

}  // namespace gem16::studio
