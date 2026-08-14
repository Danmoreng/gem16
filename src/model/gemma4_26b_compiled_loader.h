#pragma once

#include <filesystem>
#include <vector>

#include "gem16/status.h"
#include "gem16/types.h"

namespace gem16::internal {

[[nodiscard]] Status ValidateAndBindGemma4Moe26BCompiledArtifact(
    const std::filesystem::path& model_directory,
    std::vector<TensorInfo>* tensors);

}  // namespace gem16::internal
