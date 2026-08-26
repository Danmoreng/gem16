#pragma once

#include "types.h"

#include <filesystem>

namespace gem16::studio {

[[nodiscard]] std::filesystem::path RepositoryRoot();
[[nodiscard]] std::filesystem::path SettingsPath();
[[nodiscard]] StudioSettings DefaultSettings();
[[nodiscard]] StudioSettings LoadSettings();
[[nodiscard]] bool SaveSettings(const StudioSettings& settings);
void ApplyProfileDefaults(ServerConfig& config, ModelProfile profile);

}  // namespace gem16::studio

