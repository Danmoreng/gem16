#pragma once
#include "image_texture.h"
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace gem16::studio {
bool IsSvgCode(std::string_view language, std::string_view source);
struct SvgRaster {
  DecodedImage image;
  std::string error;
};
// In-memory, bounded static SVG subset; never fetches external resources.
SvgRaster RasterizeSvg(std::string_view source);

class SvgPreviewCache {
 public:
  struct Entry {
    std::string source;
    std::string error;
    ImageTexture texture;
    int last_frame = -1;
  };
  Entry* Get(std::string_view source);
  void Clear() { entries_.clear(); }
 private:
  // Eight <= 1024x1024 RGBA textures: at most 32 MiB GPU preview storage.
  std::vector<std::unique_ptr<Entry>> entries_;
};
}
