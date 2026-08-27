#include "model/gemma4_26b_device_image.h"

#include <system_error>

namespace gem16::internal {

std::filesystem::path Gemma4Moe26BDeviceImagePath(
    const std::filesystem::path& model_directory) {
  return model_directory.parent_path() /
         (model_directory.filename().string() + "." +
          std::string(kGemma4Moe26BDeviceImageFormat) + ".bin");
}

Result<bool> ProbeAcceptedGemma4Moe26BDeviceImage(
    const std::filesystem::path& model_directory) {
  const auto path = Gemma4Moe26BDeviceImagePath(model_directory);
  std::error_code error;
  const auto status = std::filesystem::symlink_status(path, error);
  if (error == std::errc::no_such_file_or_directory ||
      status.type() == std::filesystem::file_type::not_found) {
    return false;
  }
  if (error || std::filesystem::is_symlink(status) ||
      !std::filesystem::is_regular_file(status)) {
    return Status(StatusCode::kDataLoss,
                  "SM120 device image is unsafe: " + path.string());
  }
  const auto bytes = std::filesystem::file_size(path, error);
  if (error || bytes != kAcceptedM08DeviceImageBytes) {
    return Status(StatusCode::kDataLoss,
                  "SM120 device image size does not match the accepted image: " +
                      path.string());
  }
  return true;
}

}  // namespace gem16::internal
