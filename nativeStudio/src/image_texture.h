#pragma once

#include "imgui.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gem16::studio {

struct DecodedImage {
  std::vector<std::uint8_t> rgba;
  int width = 0;
  int height = 0;
};

struct ImageDimensions {
  int width = 0;
  int height = 0;
  [[nodiscard]] bool Valid() const { return width > 0 && height > 0; }
};

[[nodiscard]] ImageDimensions ProbePreviewImage(const std::uint8_t* encoded,
                                                std::size_t size);

[[nodiscard]] DecodedImage DecodePreviewImage(const std::uint8_t* encoded,
                                              std::size_t size);

class ImageTexture final {
 public:
  ImageTexture() = default;
  ~ImageTexture();
  ImageTexture(const ImageTexture&) = delete;
  ImageTexture& operator=(const ImageTexture&) = delete;

  static void InitializeRenderer(void* device);
  [[nodiscard]] bool Load(const std::uint8_t* encoded, std::size_t size);
  [[nodiscard]] bool Valid() const { return texture_id_ != ImTextureID_Invalid; }
  [[nodiscard]] ImTextureID Id() const { return texture_id_; }
  [[nodiscard]] int Width() const { return width_; }
  [[nodiscard]] int Height() const { return height_; }

 private:
  void Reset();

  ImTextureID texture_id_ = ImTextureID_Invalid;
  int width_ = 0;
  int height_ = 0;
};

}  // namespace gem16::studio
