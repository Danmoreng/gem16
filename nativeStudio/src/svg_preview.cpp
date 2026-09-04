#include "svg_preview.h"
#include "lunasvg.h"
#include "tinyxml2.h"
#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <set>

namespace gem16::studio {
namespace {
constexpr std::size_t kMaxSource = 256 * 1024;
void EnsureSvgFontFallback() {
  static std::once_flag initialized;
  std::call_once(initialized, [] {
#ifdef _WIN32
    const char* paths[] = {"C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf"};
#else
    const char* paths[] = {"/usr/share/fonts/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"};
#endif
    for (const char* path : paths) {
      if (std::filesystem::is_regular_file(path) &&
          lunasvg_add_font_face_from_file("", false, false, path)) break;
    }
  });
}
std::string Lower(std::string_view value) {
  std::string out(value);
  for (char& ch : out) if (ch >= 'A' && ch <= 'Z') ch += 'a' - 'A';
  return out;
}
std::string Validate(const tinyxml2::XMLNode& node, unsigned depth, unsigned& count, bool resource = false) {
  if (++count > 2048 || depth > 32) return "SVG exceeds the element/depth limit.";
  if (node.ToUnknown()) return "SVG declarations and entities are not supported.";
  if (const auto* element = node.ToElement()) {
    static const std::set<std::string_view> tags = {"svg", "g", "defs", "title", "desc", "rect", "circle",
        "ellipse", "line", "polyline", "polygon", "path", "text", "tspan", "linearGradient", "radialGradient",
        "stop", "clipPath", "marker", "style"};
    const std::string_view tag(element->Name());
    if (!tags.contains(tag)) return "SVG element is not supported: " + std::string(tag);
    resource |= tag == "marker" || tag == "clipPath";
    for (auto* attr = element->FirstAttribute(); attr; attr = attr->Next()) {
      const auto name = Lower(attr->Name()), value = Lower(attr->Value());
      if (value.size() > 16384) return "SVG attribute exceeds the size limit.";
      if (name.starts_with("on") || name == "href" || name == "xlink:href" || name == "xml:base")
        return "SVG events and resource links are disabled.";
      if (value.find('\\') != value.npos || value.find('@') != value.npos)
        return "SVG CSS escapes and imports are disabled.";
      std::size_t pos = 0;
      while ((pos = value.find("url", pos)) != value.npos) {
        // Permit only simple local paint/marker/clip references. References
        // inside marker/clip definitions are rejected to prevent cycles.
        if (resource || value.compare(pos, 5, "url(#") != 0)
          return "SVG external or recursive resource references are disabled.";
        const auto end = value.find(')', pos + 5);
        if (end == value.npos || end == pos + 5) return "Invalid SVG resource reference.";
        for (auto i = pos + 5; i < end; ++i)
          if (!(std::isalnum(static_cast<unsigned char>(value[i])) || value[i] == '_' || value[i] == '-'))
            return "Invalid SVG resource identifier.";
        pos = end + 1;
      }
    }
    if (tag == "style") {
      const auto css = Lower(element->GetText() ? element->GetText() : "");
      if (css.find('@') != css.npos || css.find('\\') != css.npos || css.find("url") != css.npos)
        return "SVG stylesheets cannot import or reference resources.";
    }
  }
  for (auto* child = node.FirstChild(); child; child = child->NextSibling()) {
    if (auto error = Validate(*child, depth + 1, count, resource); !error.empty()) return error;
  }
  return {};
}
}
bool IsSvgCode(std::string_view language, std::string_view source) {
  const auto lang = Lower(language);
  if (lang == "svg") return true;
  if (!lang.empty() && lang != "xml" && lang != "html") return false;
  const auto pos = source.find("<svg");
  return pos != source.npos && pos < 1024 && pos + 4 < source.size() &&
      (source[pos + 4] == '>' || std::isspace(static_cast<unsigned char>(source[pos + 4])));
}
SvgRaster RasterizeSvg(std::string_view source) {
  SvgRaster result;
  if (source.empty() || source.size() > kMaxSource || source.find('\0') != source.npos) {
    result.error = "SVG preview requires nonempty XML up to 256 KiB."; return result;
  }
  tinyxml2::XMLDocument xml;
  if (xml.Parse(source.data(), source.size()) != tinyxml2::XML_SUCCESS) {
    result.error = "SVG is incomplete or invalid; source is shown until it is complete."; return result;
  }
  const auto* root = xml.RootElement();
  if (!root || std::strcmp(root->Name(), "svg") != 0 || root->NextSiblingElement()) {
    result.error = "SVG preview requires exactly one svg root."; return result;
  }
  unsigned count = 0;
  result.error = Validate(xml, 0, count);
  if (!result.error.empty()) return result;
  try {
    EnsureSvgFontFallback();
    auto document = lunasvg::Document::loadFromData(source.data(), source.size());
    if (!document) { result.error = "SVG could not be parsed."; return result; }
    const double w = document->width(), h = document->height();
    if (!std::isfinite(w) || !std::isfinite(h) || w <= 0 || h <= 0 || w > 16384 || h > 16384) {
      result.error = "SVG dimensions must be between 0 and 16384 pixels."; return result;
    }
    // Rasterize small vectors at preview resolution too: enlarging the canvas
    // should not merely stretch their low-resolution intrinsic-size bitmap.
    const double scale = 1024.0 / std::max(w, h);
    const int width = std::max(1, static_cast<int>(std::ceil(w * scale)));
    const int height = std::max(1, static_cast<int>(std::ceil(h * scale)));
    auto bitmap = document->renderToBitmap(width, height);
    if (!bitmap.data()) { result.error = "SVG rasterization failed."; return result; }
    bitmap.convertToRGBA();
    result.image.width = width; result.image.height = height;
    result.image.rgba.resize(static_cast<std::size_t>(width) * height * 4);
    for (int y = 0; y < height; ++y)
      std::memcpy(result.image.rgba.data() + static_cast<std::size_t>(y) * width * 4,
          bitmap.data() + static_cast<std::size_t>(y) * bitmap.stride(), static_cast<std::size_t>(width) * 4);
  } catch (...) { result.error = "SVG preview failed; source remains available."; }
  return result;
}
SvgPreviewCache::Entry* SvgPreviewCache::Get(std::string_view source) {
  if (source.size() > kMaxSource) return nullptr;
  const int frame = ImGui::GetFrameCount();
  for (auto& entry : entries_) if (entry->source == source) { entry->last_frame = frame; return entry.get(); }
  if (entries_.size() >= 8) {
    const auto oldest = std::min_element(entries_.begin(), entries_.end(), [](const auto& a, const auto& b) {
      return a->last_frame < b->last_frame;
    });
    // Never destroy a texture already referenced by this frame's draw lists.
    if ((*oldest)->last_frame == frame) return nullptr;
    entries_.erase(oldest);
  }
  auto entry = std::make_unique<Entry>();
  entry->source = source.substr(0, kMaxSource + 1); entry->last_frame = frame;
  auto raster = RasterizeSvg(source);
  entry->error = std::move(raster.error);
  if (entry->error.empty() && !entry->texture.LoadRgba(raster.image)) entry->error = "SVG preview texture is unavailable.";
  entries_.push_back(std::move(entry));
  return entries_.back().get();
}
}
