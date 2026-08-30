#include "cuda/engine/gemma4_26b_routed_expert_format.h"

#include <system_error>
#include <string>

#include "model/gemma4_26b_compiled_loader.h"
#include "model/gemma4_26b_trellis35.h"

namespace gem16::internal {
namespace {

Result<bool> IsSafeMarker(const std::filesystem::path& path,
                          const char* label) {
  std::error_code error;
  const auto status = std::filesystem::symlink_status(path, error);
  if (error == std::errc::no_such_file_or_directory ||
      (!error && status.type() == std::filesystem::file_type::not_found)) {
    return false;
  }
  if (error) {
    return Status(StatusCode::kIoError,
                  std::string("cannot inspect ") + label + ": " +
                      error.message());
  }
  if (std::filesystem::is_symlink(status) ||
      !std::filesystem::is_regular_file(status)) {
    return Status(StatusCode::kDataLoss,
                  std::string(label) + " is not a safe regular file");
  }
  return true;
}

}  // namespace

Result<Gemma4Moe26BRoutedExpertFormat>
DetectGemma4Moe26BRoutedExpertFormat(
    const std::filesystem::path& model_directory) {
  auto nvfp4 =
      IsSafeMarker(model_directory / "gem16_compilation.json",
                   "Gemma 4 26B NVFP4 compilation metadata");
  if (!nvfp4.ok()) return nvfp4.status();
  auto trellis_marker =
      IsSafeMarker(model_directory / "trellis35-checkpoint.json",
                   "Gemma 4 26B Trellis35 checkpoint metadata");
  if (!trellis_marker.ok()) return trellis_marker.status();
  if (nvfp4.value() == trellis_marker.value()) {
    return Status(
        StatusCode::kDataLoss,
        nvfp4.value()
            ? "Gemma 4 26B artifact mixes NVFP4 and Trellis35 routed experts"
            : "Gemma 4 26B artifact has no explicit routed expert format");
  }
  return trellis_marker.value()
             ? Gemma4Moe26BRoutedExpertFormat::kTrellis35
             : Gemma4Moe26BRoutedExpertFormat::kNvfp4;
}

Status ValidateGemma4Moe26BRoutedExpertFormat(
    Gemma4Moe26BRoutedExpertFormat actual,
    Gemma4Moe26BRoutedExpertFormat expected) {
  if (actual == expected) return Status::Ok();
  return Status(
      StatusCode::kDataLoss,
      "Gemma 4 26B routed expert format disagrees with the expected profile: " +
          std::string(Gemma4Moe26BRoutedExpertFormatName(actual)) +
          " artifact, " +
          std::string(Gemma4Moe26BRoutedExpertFormatName(expected)) +
          " expected");
}

Result<Gemma4Moe26BRoutedExpertFormat>
ResolveValidatedGemma4Moe26BRoutedExpertFormat(
    const std::filesystem::path& model_directory) {
  auto format = DetectGemma4Moe26BRoutedExpertFormat(model_directory);
  if (!format.ok()) return format.status();
  if (IsTrellis35RoutedExpertFormat(format.value())) {
    auto plan = LoadGemma4Moe26BTrellis35CheckpointPlan(model_directory);
    if (!plan.ok()) return plan.status();
  } else {
    auto identity = LoadGemma4Moe26BCompiledIdentity(model_directory);
    if (!identity.ok()) return identity.status();
  }
  return format.value();
}

}  // namespace gem16::internal
