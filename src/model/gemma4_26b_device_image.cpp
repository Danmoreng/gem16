#include "model/gemma4_26b_device_image.h"

#include <system_error>

namespace gem16::internal {

std::filesystem::path Gemma4Moe26BDeviceImagePath(
    const std::filesystem::path& model_directory) {
  const auto product_image = model_directory / "model.gem16";
  std::error_code error;
  const auto product_status =
      std::filesystem::symlink_status(product_image, error);
  if (!error &&
      product_status.type() != std::filesystem::file_type::not_found) {
    return product_image;
  }
  return model_directory.parent_path() /
         (model_directory.filename().string() + "." +
          std::string(kGemma4Moe26BDeviceImageFormat) + ".bin");
}

Result<bool> ProbeAcceptedGemma4Moe26BDeviceImage(
    const std::filesystem::path& model_directory) {
  const auto path = Gemma4Moe26BDeviceImagePath(model_directory);
  const auto product_image = model_directory / "model.gem16";
  const auto legacy_image =
      model_directory.parent_path() /
      (model_directory.filename().string() + "." +
       std::string(kGemma4Moe26BDeviceImageFormat) + ".bin");
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
  if (path == product_image) {
    std::error_code legacy_error;
    const auto legacy_status =
        std::filesystem::symlink_status(legacy_image, legacy_error);
    if (!legacy_error &&
        legacy_status.type() != std::filesystem::file_type::not_found) {
      return Status(StatusCode::kDataLoss,
                    "both product and legacy SM120 device images are present");
    }
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
