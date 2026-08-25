#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "gem16/status.h"
#include "gem16/types.h"

namespace gem16::internal {

// Immutable identity copied from the already validated M08 compilation and
// external lock. Product reporting keeps these values in host memory so no
// model-repository access is needed after runtime initialization.
struct Gemma4Moe26BCompiledIdentity {
  std::string artifact_profile;
  std::string head_format;
  std::string artifact_content_sha256;
  std::string source_lock_sha256;
  std::string compiler_commit;
};

[[nodiscard]] Status ValidateAndBindGemma4Moe26BCompiledArtifact(
    const std::filesystem::path& model_directory,
    std::vector<TensorInfo>* tensors);

// Validates the M25 Assistant candidate independently from the accepted M08
// target artifact. This contract pins the official Google source lock and the
// exact 97-tensor hybrid layout, but intentionally does not freeze a candidate
// artifact hash before M25 acceptance.
[[nodiscard]] Status ValidateAndBindGemma4Moe26BAssistantCompiledArtifact(
    const std::filesystem::path& model_directory,
    std::vector<TensorInfo>* tensors);

[[nodiscard]] Result<Gemma4Moe26BCompiledIdentity>
LoadGemma4Moe26BCompiledIdentity(
    const std::filesystem::path& model_directory);

}  // namespace gem16::internal
