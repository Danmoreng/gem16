#include "gem16gb/model.h"

#include <array>

#include "model/config.h"
#include "model/manifest.h"
#include "model/tokenizer_config.h"

namespace gem16gb {

Result<ModelManifest> InspectCheckpoint(const InspectOptions& options) {
  if (options.model_directory.empty()) {
    return Status(StatusCode::kInvalidArgument, "--model requires a checkpoint directory");
  }
  std::error_code error;
  if (!std::filesystem::is_directory(options.model_directory, error)) {
    return Status(StatusCode::kNotFound, "model directory does not exist: " + options.model_directory.string());
  }

  constexpr std::array required_files = {
      "config.json", "tokenizer.json", "tokenizer_config.json"};
  for (const char* name : required_files) {
    const auto path = options.model_directory / name;
    if (!std::filesystem::is_regular_file(path)) {
      return Status(StatusCode::kNotFound, "required checkpoint file is missing: " + path.string());
    }
  }
  auto config = internal::LoadModelConfig(options.model_directory / "config.json");
  if (!config.ok()) return config.status();
  auto tokenizer_config = internal::LoadTokenizerConfig(
      options.model_directory / "tokenizer_config.json");
  if (!tokenizer_config.ok()) return tokenizer_config.status();
  if (options.validate) {
    auto validation = internal::ValidatePrimaryModelContract(config.value());
    if (!validation.ok()) return validation;
    validation =
        internal::ValidatePrimaryTokenizerConfig(tokenizer_config.value());
    if (!validation.ok()) return validation;
  }
  return internal::BuildManifest(options.model_directory, config.value(), options.validate);
}

}  // namespace gem16gb
