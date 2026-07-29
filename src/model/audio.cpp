#include "gem16/audio.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define NOMINMAX
#endif
#define MA_NO_DEVICE_IO
#define MA_NO_ENCODING
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MA_NO_ENGINE
#define MA_NO_GENERATION
#define MINIAUDIO_IMPLEMENTATION
#include <miniaudio.h>

namespace gem16 {
namespace {

constexpr std::uint64_t kMaximumEncodedBytes = 64ULL * 1024ULL * 1024ULL;
constexpr std::size_t kMaximumSamples = 16000U * 30U;

Status Invalid(const std::filesystem::path& path, std::string message) {
  return Status(StatusCode::kDataLoss,
                "invalid audio file " + path.string() + ": " +
                    std::move(message));
}

}  // namespace

Result<AudioWaveform> LoadAudioFile(const std::filesystem::path& path) {
  std::error_code file_error;
  const std::uint64_t file_size = std::filesystem::file_size(path, file_error);
  if (file_error) {
    return Status(StatusCode::kIoError,
                  "cannot stat audio file " + path.string() + ": " +
                      file_error.message());
  }
  if (file_size == 0U || file_size > kMaximumEncodedBytes ||
      file_size > std::numeric_limits<std::size_t>::max()) {
    return Invalid(path, "encoded size is empty or exceeds the safety limit");
  }

  std::vector<unsigned char> encoded(static_cast<std::size_t>(file_size));
  std::ifstream input(path, std::ios::binary);
  if (!input ||
      !input.read(reinterpret_cast<char*>(encoded.data()),
                  static_cast<std::streamsize>(encoded.size()))) {
    return Status(StatusCode::kIoError,
                  "cannot read audio file " + path.string());
  }

  ma_decoder_config config =
      ma_decoder_config_init(ma_format_f32, 1U, 16000U);
  ma_decoder decoder{};
  const ma_result init = ma_decoder_init_memory(
      encoded.data(), encoded.size(), &config, &decoder);
  if (init != MA_SUCCESS) {
    return Status(StatusCode::kUnsupported,
                  "cannot decode WAV, FLAC, or MP3 audio file " +
                      path.string() + ": miniaudio error " +
                      std::to_string(init));
  }
  struct DecoderScope {
    ma_decoder* decoder;
    ~DecoderScope() { ma_decoder_uninit(decoder); }
  } decoder_scope{&decoder};

  // Read one frame beyond the public duration limit. This keeps allocation and
  // decompression bounded even when a tiny compressed file declares a huge
  // duration.
  AudioWaveform waveform;
  waveform.samples.resize(kMaximumSamples + 1U);
  ma_uint64 frames_read = 0U;
  const ma_result read = ma_decoder_read_pcm_frames(
      &decoder, waveform.samples.data(),
      static_cast<ma_uint64>(waveform.samples.size()), &frames_read);
  if ((read != MA_SUCCESS && read != MA_AT_END) || frames_read == 0U) {
    return Invalid(path, "decoder produced no complete samples");
  }
  if (frames_read > kMaximumSamples) {
    return Status(StatusCode::kUnsupported,
                  "audio duration exceeds the 30-second model limit");
  }
  waveform.samples.resize(static_cast<std::size_t>(frames_read));
  if (!std::all_of(waveform.samples.begin(), waveform.samples.end(),
                   [](float sample) { return std::isfinite(sample); })) {
    return Invalid(path, "decoded sample is not finite");
  }
  return waveform;
}

Result<AudioWaveform> LoadAudioWav(const std::filesystem::path& path) {
  return LoadAudioFile(path);
}

}  // namespace gem16
