#include "gem16/image.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define NOMINMAX
#include <objbase.h>
#include <wincodec.h>
#endif

namespace gem16 {
namespace {

constexpr std::uint32_t kTeacherPatch = 16U;
constexpr std::uint32_t kPool = 3U;
constexpr std::uint32_t kModelPatch = kTeacherPatch * kPool;
constexpr std::uint32_t kMaximumSoftTokens = 280U;
constexpr std::uint64_t kMaximumPixels = 100'000'000ULL;

struct RgbImage {
  std::uint32_t width = 0U;
  std::uint32_t height = 0U;
  std::vector<std::uint8_t> pixels;
};

Status Error(StatusCode code, std::string message) {
  return Status(code, std::move(message));
}

#if defined(_WIN32)

template <typename T>
class ComPtr {
 public:
  ~ComPtr() { if (value_ != nullptr) value_->Release(); }
  T** put() { return &value_; }
  T* get() const { return value_; }
 private:
  T* value_ = nullptr;
};

Result<RgbImage> DecodeImage(const std::filesystem::path& path) {
  const HRESULT initialized =
      CoInitializeEx(nullptr, COINIT_MULTITHREADED);
  const bool uninitialize = initialized == S_OK || initialized == S_FALSE;
  if (FAILED(initialized) && initialized != RPC_E_CHANGED_MODE) {
    return Error(StatusCode::kInternal,
                 "cannot initialize Windows image decoding");
  }
  struct CoScope {
    bool active;
    ~CoScope() { if (active) CoUninitialize(); }
  } scope{uninitialize};

  ComPtr<IWICImagingFactory> factory;
  HRESULT result = CoCreateInstance(
      CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
      IID_PPV_ARGS(factory.put()));
  if (FAILED(result)) {
    return Error(StatusCode::kInternal,
                 "cannot create Windows Imaging Component factory");
  }
  ComPtr<IWICBitmapDecoder> decoder;
  result = factory.get()->CreateDecoderFromFilename(
      path.c_str(), nullptr, GENERIC_READ, WICDecodeMetadataCacheOnDemand,
      decoder.put());
  if (FAILED(result)) {
    return Error(StatusCode::kUnsupported,
                 "cannot decode image file: " + path.string());
  }
  ComPtr<IWICBitmapFrameDecode> frame;
  result = decoder.get()->GetFrame(0U, frame.put());
  if (FAILED(result)) {
    return Error(StatusCode::kDataLoss,
                 "cannot decode the first image frame: " + path.string());
  }
  UINT width = 0U;
  UINT height = 0U;
  result = frame.get()->GetSize(&width, &height);
  if (FAILED(result) || width == 0U || height == 0U ||
      static_cast<std::uint64_t>(width) * height > kMaximumPixels) {
    return Error(StatusCode::kUnsupported,
                 "image dimensions are empty or exceed the safety limit");
  }
  ComPtr<IWICFormatConverter> converter;
  result = factory.get()->CreateFormatConverter(converter.put());
  if (SUCCEEDED(result)) {
    result = converter.get()->Initialize(
        frame.get(), GUID_WICPixelFormat24bppRGB,
        WICBitmapDitherTypeNone, nullptr, 0.0,
        WICBitmapPaletteTypeCustom);
  }
  if (FAILED(result)) {
    return Error(StatusCode::kUnsupported,
                 "image cannot be converted to 24-bit RGB");
  }
  const std::uint64_t stride64 = static_cast<std::uint64_t>(width) * 3U;
  const std::uint64_t bytes64 = stride64 * height;
  if (stride64 > std::numeric_limits<UINT>::max() ||
      bytes64 > std::numeric_limits<UINT>::max()) {
    return Error(StatusCode::kUnsupported,
                 "decoded image buffer exceeds the WIC limit");
  }
  RgbImage image;
  image.width = width;
  image.height = height;
  image.pixels.resize(static_cast<std::size_t>(bytes64));
  result = converter.get()->CopyPixels(
      nullptr, static_cast<UINT>(stride64), static_cast<UINT>(bytes64),
      image.pixels.data());
  if (FAILED(result)) {
    return Error(StatusCode::kDataLoss,
                 "cannot read converted RGB image pixels");
  }
  return image;
}

#else

Result<RgbImage> DecodeImage(const std::filesystem::path& path) {
  return Error(StatusCode::kUnsupported,
               "this build currently requires Windows WIC for PNG/JPEG/BMP image decoding: " +
                   path.string());
}

#endif

std::pair<std::uint32_t, std::uint32_t> TargetSize(
    std::uint32_t height, std::uint32_t width) {
  const double target_pixels = static_cast<double>(
      kMaximumSoftTokens * kPool * kPool * kTeacherPatch * kTeacherPatch);
  const double factor = std::sqrt(
      target_pixels / static_cast<double>(height * static_cast<std::uint64_t>(width)));
  const double ideal_height = factor * height;
  const double ideal_width = factor * width;
  std::uint32_t target_height = static_cast<std::uint32_t>(
      std::floor(ideal_height / kModelPatch)) * kModelPatch;
  std::uint32_t target_width = static_cast<std::uint32_t>(
      std::floor(ideal_width / kModelPatch)) * kModelPatch;
  const std::uint32_t max_side = kMaximumSoftTokens * kModelPatch;
  if (target_height == 0U) {
    target_height = kModelPatch;
    target_width = std::min(
        std::max(kModelPatch, (width / height) * kModelPatch), max_side);
  } else if (target_width == 0U) {
    target_width = kModelPatch;
    target_height = std::min(
        std::max(kModelPatch, (height / width) * kModelPatch), max_side);
  }
  return {target_height, target_width};
}

double Cubic(double value) {
  constexpr double a = -0.75;
  value = std::abs(value);
  if (value < 1.0) {
    return ((a + 2.0) * value - (a + 3.0)) * value * value + 1.0;
  }
  if (value < 2.0) {
    return ((a * value - 5.0 * a) * value + 8.0 * a) * value - 4.0 * a;
  }
  return 0.0;
}

struct FilterEntry {
  std::vector<std::int32_t> indices;
  std::vector<float> weights;
};

std::vector<FilterEntry> Filters(std::uint32_t source,
                                 std::uint32_t target) {
  std::vector<FilterEntry> result(target);
  const double scale = static_cast<double>(source) / target;
  const double support_scale = std::max(1.0, scale);
  const double radius = 2.0 * support_scale;
  for (std::uint32_t output = 0U; output < target; ++output) {
    const double center = (static_cast<double>(output) + 0.5) * scale - 0.5;
    const std::int32_t first = static_cast<std::int32_t>(
        std::ceil(center - radius));
    const std::int32_t last = static_cast<std::int32_t>(
        std::floor(center + radius));
    double sum = 0.0;
    for (std::int32_t input = first; input <= last; ++input) {
      const double weight = Cubic((center - input) / support_scale);
      if (weight == 0.0) continue;
      result[output].indices.push_back(std::clamp<std::int32_t>(
          input, 0, static_cast<std::int32_t>(source) - 1));
      result[output].weights.push_back(static_cast<float>(weight));
      sum += weight;
    }
    for (float& weight : result[output].weights) {
      weight = static_cast<float>(weight / sum);
    }
  }
  return result;
}

std::vector<std::uint8_t> Resize(const RgbImage& source,
                                 std::uint32_t target_width,
                                 std::uint32_t target_height) {
  if (source.width == target_width && source.height == target_height) {
    return source.pixels;
  }
  const auto horizontal = Filters(source.width, target_width);
  const auto vertical = Filters(source.height, target_height);
  std::vector<float> intermediate(
      static_cast<std::size_t>(source.height) * target_width * 3U);
  for (std::uint32_t y = 0U; y < source.height; ++y) {
    for (std::uint32_t x = 0U; x < target_width; ++x) {
      for (std::uint32_t channel = 0U; channel < 3U; ++channel) {
        float value = 0.0F;
        for (std::size_t tap = 0U; tap < horizontal[x].indices.size(); ++tap) {
          const std::size_t source_index =
              (static_cast<std::size_t>(y) * source.width +
               static_cast<std::uint32_t>(horizontal[x].indices[tap])) *
                  3U +
              channel;
          value += horizontal[x].weights[tap] * source.pixels[source_index];
        }
        intermediate[(static_cast<std::size_t>(y) * target_width + x) * 3U +
                     channel] = value;
      }
    }
  }
  std::vector<std::uint8_t> output(
      static_cast<std::size_t>(target_height) * target_width * 3U);
  for (std::uint32_t y = 0U; y < target_height; ++y) {
    for (std::uint32_t x = 0U; x < target_width; ++x) {
      for (std::uint32_t channel = 0U; channel < 3U; ++channel) {
        float value = 0.0F;
        for (std::size_t tap = 0U; tap < vertical[y].indices.size(); ++tap) {
          const std::size_t source_index =
              (static_cast<std::size_t>(vertical[y].indices[tap]) *
                   target_width +
               x) *
                  3U +
              channel;
          value += vertical[y].weights[tap] * intermediate[source_index];
        }
        output[(static_cast<std::size_t>(y) * target_width + x) * 3U +
               channel] = static_cast<std::uint8_t>(
            std::clamp(std::lround(value), 0L, 255L));
      }
    }
  }
  return output;
}

}  // namespace

Result<VisionImage> LoadVisionImage(const std::filesystem::path& path) {
  auto decoded = DecodeImage(path);
  if (!decoded.ok()) return decoded.status();
  const auto [target_height, target_width] =
      TargetSize(decoded.value().height, decoded.value().width);
  if (target_height == 0U || target_width == 0U ||
      target_height % kModelPatch != 0U ||
      target_width % kModelPatch != 0U) {
    return Error(StatusCode::kDataLoss,
                 "image aspect ratio cannot produce a valid patch grid");
  }
  const std::uint32_t grid_height = target_height / kModelPatch;
  const std::uint32_t grid_width = target_width / kModelPatch;
  const std::uint32_t patch_count = grid_height * grid_width;
  if (patch_count == 0U || patch_count > kMaximumSoftTokens) {
    return Error(StatusCode::kDataLoss,
                 "image patch count exceeds the model limit");
  }
  auto resized = Resize(decoded.value(), target_width, target_height);
  VisionImage result;
  result.patch_count = patch_count;
  result.source_width = decoded.value().width;
  result.source_height = decoded.value().height;
  result.patches.resize(static_cast<std::size_t>(patch_count) * 48U * 48U * 3U);
  result.positions.resize(static_cast<std::size_t>(patch_count) * 2U);
  std::size_t destination = 0U;
  for (std::uint32_t patch_y = 0U; patch_y < grid_height; ++patch_y) {
    for (std::uint32_t patch_x = 0U; patch_x < grid_width; ++patch_x) {
      const std::uint32_t patch = patch_y * grid_width + patch_x;
      result.positions[patch * 2U] = static_cast<std::int32_t>(patch_x);
      result.positions[patch * 2U + 1U] = static_cast<std::int32_t>(patch_y);
      for (std::uint32_t y = 0U; y < kModelPatch; ++y) {
        for (std::uint32_t x = 0U; x < kModelPatch; ++x) {
          const std::size_t source =
              ((static_cast<std::size_t>(patch_y) * kModelPatch + y) *
                   target_width +
               static_cast<std::size_t>(patch_x) * kModelPatch + x) *
              3U;
          for (std::uint32_t channel = 0U; channel < 3U; ++channel) {
            result.patches[destination++] =
                static_cast<float>(resized[source + channel]) / 255.0F;
          }
        }
      }
    }
  }
  return result;
}

}  // namespace gem16
