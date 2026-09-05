#include "gem16/image.h"
#include "model/image_decode_budget.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "compiler/sha256.h"

#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#define STBI_MAX_DIMENSIONS 32768
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace gem16 {
namespace {

constexpr std::uint32_t kTeacherPatch = 16U;
constexpr std::uint32_t kPool = 3U;
constexpr std::uint32_t kModelPatch = kTeacherPatch * kPool;
constexpr std::uint32_t kMaximumSoftTokens = 280U;
constexpr std::uint64_t kMaximumPixels = 100'000'000ULL;
constexpr std::uint64_t kMaximumEncodedBytes = 256ULL * 1024ULL * 1024ULL;

double Milliseconds(std::chrono::steady_clock::time_point begin,
                    std::chrono::steady_clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

struct RgbImage {
  std::uint32_t width = 0U;
  std::uint32_t height = 0U;
  std::vector<std::uint8_t> pixels;
};

Status Error(StatusCode code, std::string message) {
  return Status(code, std::move(message));
}

ImageSourceIdentity SourceIdentity(std::span<const std::uint8_t> encoded) {
  compiler::Sha256 hash;
  hash.Update(encoded.data(), encoded.size());
  return ImageSourceIdentity{hash.Final(),
                             static_cast<std::uint64_t>(encoded.size())};
}

Result<RgbImage> DecodeImage(std::span<const std::uint8_t> encoded,
                             std::string_view source_name) {
  if (encoded.empty() || encoded.size() > kMaximumEncodedBytes ||
      encoded.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return Error(StatusCode::kUnsupported,
                 "encoded image size is empty or exceeds the safety limit");
  }
  int width = 0;
  int height = 0;
  int channels = 0;
  const int encoded_size = static_cast<int>(encoded.size());
  if (stbi_info_from_memory(encoded.data(), encoded_size, &width, &height,
                            &channels) == 0 ||
      width <= 0 || height <= 0 ||
      static_cast<std::uint64_t>(width) *
              static_cast<std::uint64_t>(height) >
          kMaximumPixels) {
    return Error(StatusCode::kUnsupported,
                 "image is malformed, unsupported, or exceeds the pixel limit: " +
                     std::string(source_name));
  }

  if (!internal::ImageDecodeBudget::Pixels(
          static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height)))
    return Error(StatusCode::kResourceExhausted, "request image pixel budget exceeded");
  std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> decoded(
      stbi_load_from_memory(encoded.data(), encoded_size, &width, &height,
                            &channels, 3),
      &stbi_image_free);
  if (decoded == nullptr) {
    const char* reason = stbi_failure_reason();
    return Error(StatusCode::kDataLoss,
                 "cannot decode image " + std::string(source_name) +
                     (reason == nullptr ? std::string() :
                                          ": " + std::string(reason)));
  }
  const std::size_t decoded_bytes = static_cast<std::size_t>(width) *
                                    static_cast<std::size_t>(height) * 3U;
  RgbImage image;
  image.width = static_cast<std::uint32_t>(width);
  image.height = static_cast<std::uint32_t>(height);
  image.pixels.assign(decoded.get(), decoded.get() + decoded_bytes);
  return image;
}

std::pair<std::uint32_t, std::uint32_t> TargetSize(
    std::uint32_t height, std::uint32_t width,
    std::uint32_t maximum_soft_tokens, bool allow_upscale) {
  const double target_pixels = static_cast<double>(
      maximum_soft_tokens * kPool * kPool * kTeacherPatch * kTeacherPatch);
  double factor = std::sqrt(
      target_pixels / static_cast<double>(height * static_cast<std::uint64_t>(width)));
  if (!allow_upscale) factor = std::min(1.0, factor);
  const double ideal_height = factor * height;
  const double ideal_width = factor * width;
  std::uint32_t target_height = static_cast<std::uint32_t>(
      std::floor(ideal_height / kModelPatch)) * kModelPatch;
  std::uint32_t target_width = static_cast<std::uint32_t>(
      std::floor(ideal_width / kModelPatch)) * kModelPatch;
  const std::uint32_t max_side = maximum_soft_tokens * kModelPatch;
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

bool IsGemma4Moe26BSoftTokenBudget(std::uint32_t value) {
  return value == 70U || value == 140U || value == 280U;
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

std::uint32_t AutomaticVisionSoftTokenBudget(
    std::uint64_t context_tokens, std::uint64_t reserved_non_image_tokens,
    std::size_t image_count) {
  if (image_count == 0U) return kMaximumSoftTokens;
  const std::uint64_t available = context_tokens > reserved_non_image_tokens
                                      ? context_tokens - reserved_non_image_tokens
                                      : image_count;
  return static_cast<std::uint32_t>(std::clamp<std::uint64_t>(
      available / image_count, 1U, kMaximumSoftTokens));
}

Result<VisionImage> LoadVisionImageBytes(
    std::span<const std::uint8_t> encoded, std::string_view source_name,
    const VisionImageOptions& options) {
  if (options.maximum_soft_tokens == 0U ||
      options.maximum_soft_tokens > kMaximumSoftTokens) {
    return Error(StatusCode::kInvalidArgument,
                 "image soft-token budget must be between 1 and 280");
  }
  auto decoded = DecodeImage(encoded, source_name);
  if (!decoded.ok()) return decoded.status();
  const auto [target_height, target_width] =
      TargetSize(decoded.value().height, decoded.value().width,
                 options.maximum_soft_tokens, options.allow_upscale);
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
  // Account for retained float patches, positions and resize scratch before
  // allocating any of them. The pixel budget separately bounds codec/RGB work.
  if (!internal::ImageDecodeBudget::PreparedBytes(
          static_cast<std::uint64_t>(target_width) * target_height * 16U +
          static_cast<std::uint64_t>(target_width / 16U) * (target_height / 16U) * 8U))
    return Error(StatusCode::kResourceExhausted, "request prepared image memory budget exceeded");
  auto resized = Resize(decoded.value(), target_width, target_height);
  VisionImage result;
  result.patch_count = patch_count;
  result.source_width = decoded.value().width;
  result.source_height = decoded.value().height;
  result.processed_width = target_width;
  result.processed_height = target_height;
  result.soft_token_budget = options.maximum_soft_tokens;
  result.source_identity = SourceIdentity(encoded);
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

Result<VisionImage> LoadVisionImage(const std::filesystem::path& path,
                                    const VisionImageOptions& options) {
  std::error_code file_error;
  const std::uint64_t file_size = std::filesystem::file_size(path, file_error);
  if (file_error) {
    return Error(StatusCode::kIoError,
                 "cannot stat image " + path.string() + ": " +
                     file_error.message());
  }
  if (file_size == 0U || file_size > kMaximumEncodedBytes ||
      file_size > std::numeric_limits<std::size_t>::max()) {
    return Error(StatusCode::kUnsupported,
                 "encoded image size is empty or exceeds the safety limit");
  }
  std::vector<std::uint8_t> encoded(static_cast<std::size_t>(file_size));
  std::ifstream input(path, std::ios::binary);
  if (!input ||
      !input.read(reinterpret_cast<char*>(encoded.data()),
                  static_cast<std::streamsize>(encoded.size()))) {
    return Error(StatusCode::kIoError,
                 "cannot read image " + path.string());
  }
  return LoadVisionImageBytes(encoded, path.string(), options);
}

Result<Gemma4Moe26BVisionImage> LoadGemma4Moe26BVisionImageBytes(
    std::span<const std::uint8_t> encoded, std::string_view source_name,
    const Gemma4Moe26BVisionImageOptions& options) {
  const auto total_begin = std::chrono::steady_clock::now();
  if (options.timings != nullptr) {
    *options.timings = Gemma4Moe26BVisionPreprocessTimings{};
  }
  if (!IsGemma4Moe26BSoftTokenBudget(options.maximum_soft_tokens)) {
    return Error(StatusCode::kInvalidArgument,
                 "Gemma 4 26B image soft-token budget must be 70, 140, or 280");
  }
  const auto decode_begin = std::chrono::steady_clock::now();
  auto decoded = DecodeImage(encoded, source_name);
  const auto decode_end = std::chrono::steady_clock::now();
  if (options.timings != nullptr) {
    options.timings->decode_milliseconds =
        Milliseconds(decode_begin, decode_end);
  }
  if (!decoded.ok()) return decoded.status();
  // Google's processor always scales toward the selected budget. In
  // particular, small images are intentionally upscaled here.
  const auto [target_height, target_width] =
      TargetSize(decoded.value().height, decoded.value().width,
                 options.maximum_soft_tokens, true);
  if (target_height == 0U || target_width == 0U ||
      target_height % kModelPatch != 0U ||
      target_width % kModelPatch != 0U) {
    return Error(StatusCode::kDataLoss,
                 "image aspect ratio cannot produce a valid Gemma 4 26B patch grid");
  }
  const std::uint32_t raw_grid_height = target_height / kTeacherPatch;
  const std::uint32_t raw_grid_width = target_width / kTeacherPatch;
  const std::uint32_t raw_patch_count = raw_grid_height * raw_grid_width;
  const std::uint32_t soft_token_count = raw_patch_count / (kPool * kPool);
  if (raw_patch_count == 0U ||
      raw_patch_count > options.maximum_soft_tokens * kPool * kPool ||
      raw_grid_height % kPool != 0U || raw_grid_width % kPool != 0U ||
      soft_token_count == 0U ||
      soft_token_count > options.maximum_soft_tokens) {
    return Error(StatusCode::kDataLoss,
                 "Gemma 4 26B image patch count exceeds the selected budget");
  }

  const auto resize_begin = std::chrono::steady_clock::now();
  // Account for retained float patches, positions and resize scratch before
  // allocating any of them. The pixel budget separately bounds codec/RGB work.
  if (!internal::ImageDecodeBudget::PreparedBytes(
          static_cast<std::uint64_t>(target_width) * target_height * 16U +
          static_cast<std::uint64_t>(target_width / 16U) * (target_height / 16U) * 8U))
    return Error(StatusCode::kResourceExhausted, "request prepared image memory budget exceeded");
  auto resized = Resize(decoded.value(), target_width, target_height);
  const auto resize_end = std::chrono::steady_clock::now();
  if (options.timings != nullptr) {
    options.timings->resize_milliseconds =
        Milliseconds(resize_begin, resize_end);
  }
  const auto patchify_begin = std::chrono::steady_clock::now();
  Gemma4Moe26BVisionImage result;
  result.raw_patch_count = raw_patch_count;
  result.soft_token_count = soft_token_count;
  result.source_width = decoded.value().width;
  result.source_height = decoded.value().height;
  result.processed_width = target_width;
  result.processed_height = target_height;
  result.soft_token_budget = options.maximum_soft_tokens;
  result.source_identity = SourceIdentity(encoded);
  result.patches.resize(static_cast<std::size_t>(raw_patch_count) * 16U * 16U * 3U);
  result.positions.resize(static_cast<std::size_t>(raw_patch_count) * 2U);
  std::size_t destination = 0U;
  for (std::uint32_t patch_y = 0U; patch_y < raw_grid_height; ++patch_y) {
    for (std::uint32_t patch_x = 0U; patch_x < raw_grid_width; ++patch_x) {
      const std::uint32_t patch = patch_y * raw_grid_width + patch_x;
      result.positions[patch * 2U] = static_cast<std::int32_t>(patch_x);
      result.positions[patch * 2U + 1U] = static_cast<std::int32_t>(patch_y);
      for (std::uint32_t y = 0U; y < kTeacherPatch; ++y) {
        for (std::uint32_t x = 0U; x < kTeacherPatch; ++x) {
          const std::size_t source =
              ((static_cast<std::size_t>(patch_y) * kTeacherPatch + y) *
                   target_width +
               static_cast<std::size_t>(patch_x) * kTeacherPatch + x) *
              3U;
          for (std::uint32_t channel = 0U; channel < 3U; ++channel) {
            result.patches[destination++] =
                static_cast<float>(resized[source + channel]) / 255.0F;
          }
        }
      }
    }
  }
  const auto patchify_end = std::chrono::steady_clock::now();
  result.decode_milliseconds = Milliseconds(decode_begin, decode_end);
  result.resize_patchify_milliseconds =
      Milliseconds(resize_begin, resize_end) +
      Milliseconds(patchify_begin, patchify_end);
  if (options.timings != nullptr) {
    options.timings->patchify_milliseconds =
        Milliseconds(patchify_begin, patchify_end);
    options.timings->total_milliseconds =
        Milliseconds(total_begin, patchify_end);
  }
  return result;
}

Result<Gemma4Moe26BVisionImage> LoadGemma4Moe26BVisionImage(
    const std::filesystem::path& path,
    const Gemma4Moe26BVisionImageOptions& options) {
  std::error_code file_error;
  const std::uint64_t file_size = std::filesystem::file_size(path, file_error);
  if (file_error) {
    return Error(StatusCode::kIoError,
                 "cannot stat image " + path.string() + ": " +
                     file_error.message());
  }
  if (file_size == 0U || file_size > kMaximumEncodedBytes ||
      file_size > std::numeric_limits<std::size_t>::max()) {
    return Error(StatusCode::kUnsupported,
                 "encoded image size is empty or exceeds the safety limit");
  }
  std::vector<std::uint8_t> encoded(static_cast<std::size_t>(file_size));
  std::ifstream input(path, std::ios::binary);
  if (!input ||
      !input.read(reinterpret_cast<char*>(encoded.data()),
                  static_cast<std::streamsize>(encoded.size()))) {
    return Error(StatusCode::kIoError,
                 "cannot read image " + path.string());
  }
  return LoadGemma4Moe26BVisionImageBytes(encoded, path.string(), options);
}

}  // namespace gem16
