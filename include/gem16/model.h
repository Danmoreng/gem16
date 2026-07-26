#pragma once

#include <filesystem>
#include <iosfwd>

#include "gem16/status.h"
#include "gem16/types.h"

namespace gem16 {

struct InspectOptions {
  std::filesystem::path model_directory;
  bool validate = false;
};

[[nodiscard]] Result<ModelManifest> InspectCheckpoint(const InspectOptions& options);
[[nodiscard]] Status WriteManifestJson(const ModelManifest& manifest, std::ostream& output);
void PrintManifestSummary(const ModelManifest& manifest, std::ostream& output);

}  // namespace gem16
