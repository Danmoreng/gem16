#pragma once

#include <filesystem>

#include "gem16/status.h"
#include "gem16/types.h"
#include "model/config.h"

namespace gem16::internal {

[[nodiscard]] Result<ModelManifest> BuildManifest(
    const std::filesystem::path& model_directory,
    const ModelConfig& config,
    bool validate);

}  // namespace gem16::internal

