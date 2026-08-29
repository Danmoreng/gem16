#pragma once

#include "types.h"

#include <filesystem>

namespace gem16::studio {

[[nodiscard]] std::filesystem::path RepositoryRoot();
[[nodiscard]] std::filesystem::path HuggingFaceHubRoot();
[[nodiscard]] std::filesystem::path ProfileTargetDirectory(ModelProfile profile);
[[nodiscard]] std::filesystem::path ProfileAssistantDirectory(ModelProfile profile);
[[nodiscard]] std::filesystem::path SettingsPath();
[[nodiscard]] StudioSettings DefaultSettings();
[[nodiscard]] StudioSettings LoadSettings();
[[nodiscard]] bool SaveSettings(const StudioSettings& settings);
[[nodiscard]] float ResolveUiScale(float configured_scale,
                                   float platform_scale,
                                   bool linux_platform);
void ApplyProfileDefaults(ServerConfig& config, ModelProfile profile);

}  // namespace gem16::studio
