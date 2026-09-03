#include "cuda/engine/gemma4_26b_routed_expert_format.h"

#include <system_error>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string>

#include "model/gemma4_26b_compiled_loader.h"
#include "model/gemma4_26b_trellis35.h"
#include "util/environment.h"
#include "util/json.h"

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

Result<bool> IsTrellis35V2Marker(const std::filesystem::path& directory) {
  const auto path = directory / "gem16_model.json";
  auto present = IsSafeMarker(path, "Gemma 4 26B model metadata");
  if (!present.ok() || !present.value()) return present;
  std::error_code error;
  const auto bytes = std::filesystem::file_size(path, error);
  if (error || bytes > 4U * 1024U * 1024U ||
      bytes > static_cast<std::uint64_t>(
                  std::numeric_limits<std::size_t>::max())) {
    return Status(StatusCode::kDataLoss,
                  "Gemma 4 26B model metadata is oversized");
  }
  std::string payload(static_cast<std::size_t>(bytes), '\0');
  std::ifstream input(path, std::ios::binary);
  if (!input ||
      (bytes != 0U &&
       !input.read(payload.data(), static_cast<std::streamsize>(bytes)))) {
    return Status(StatusCode::kIoError,
                  "cannot read Gemma 4 26B model metadata");
  }
  auto document = json::Parse(payload);
  if (!document.ok() || !document.value().is_object()) {
    return Status(StatusCode::kDataLoss,
                  "Gemma 4 26B model metadata is invalid JSON");
  }
  const auto* format = document.value().find("format");
  return format != nullptr && format->is_string() &&
         format->as_string() == kGemma4Moe26BTrellis35DeviceImageFormat;
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
  auto trellis_v2 = IsTrellis35V2Marker(model_directory);
  if (!trellis_v2.ok()) return trellis_v2.status();
  if (trellis_marker.value() && trellis_v2.value()) {
    return Status(StatusCode::kDataLoss,
                  "Gemma 4 26B artifact mixes Trellis35 v1 and v2");
  }
  if (trellis_v2.value()) {
    return Gemma4Moe26BRoutedExpertFormat::kTrellis35;
  }
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
    auto plan = LoadValidatedGemma4Moe26BTrellis35RuntimePlan(model_directory);
    if (!plan.ok()) return plan.status();
  } else {
    auto identity = LoadGemma4Moe26BCompiledIdentity(model_directory);
    if (!identity.ok()) return identity.status();
  }
  return format.value();
}

Result<Gemma4Moe26BTrellis35CheckpointPlan>
LoadValidatedGemma4Moe26BTrellis35RuntimePlan(
    const std::filesystem::path& model_directory) {
  auto v2 = IsTrellis35V2Marker(model_directory);
  if (!v2.ok()) return v2.status();
  if (v2.value()) {
    const char* requested = GetEnvironmentVariable("GEM16_TRELLIS35_FORMAT");
    if (requested != nullptr && std::string_view(requested) == "legacy-v1") {
      return Status(StatusCode::kDataLoss,
                    "legacy Trellis35 v1 was requested but v2 files are "
                    "present");
    }
    return LoadGemma4Moe26BTrellis35DeviceImagePlan(model_directory);
  }
  const char* requested = GetEnvironmentVariable("GEM16_TRELLIS35_FORMAT");
  if (requested == nullptr || std::string_view(requested) != "legacy-v1") {
    return Status(StatusCode::kUnsupported,
                  "Trellis35 v1 requires explicit "
                  "GEM16_TRELLIS35_FORMAT=legacy-v1");
  }
  return LoadGemma4Moe26BTrellis35CheckpointPlan(model_directory);
}

}  // namespace gem16::internal
