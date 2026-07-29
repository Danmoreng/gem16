#include "gem16/audio.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>

namespace gem16 {
namespace {

constexpr std::uint64_t kMaximumWaveBytes = 64ULL * 1024ULL * 1024ULL;
constexpr std::size_t kMaximumSamples = 16000U * 30U;

std::uint16_t U16(const std::byte* value) {
  return static_cast<std::uint16_t>(std::to_integer<std::uint8_t>(value[0])) |
         static_cast<std::uint16_t>(std::to_integer<std::uint8_t>(value[1]) << 8U);
}

std::uint32_t U32(const std::byte* value) {
  return static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(value[0])) |
         (static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(value[1])) << 8U) |
         (static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(value[2])) << 16U) |
         (static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(value[3])) << 24U);
}

bool FourCc(const std::byte* value, const char* expected) {
  return std::memcmp(value, expected, 4U) == 0;
}

Status Invalid(const std::filesystem::path& path, std::string message) {
  return Status(StatusCode::kDataLoss,
                "invalid audio WAV " + path.string() + ": " + std::move(message));
}

}  // namespace

Result<AudioWaveform> LoadAudioWav(const std::filesystem::path& path) {
  std::error_code error;
  const std::uint64_t size = std::filesystem::file_size(path, error);
  if (error) {
    return Status(StatusCode::kIoError,
                  "cannot stat audio WAV " + path.string() + ": " + error.message());
  }
  if (size < 44U || size > kMaximumWaveBytes ||
      size > std::numeric_limits<std::size_t>::max()) {
    return Invalid(path, "file size is outside the supported range");
  }
  std::vector<std::byte> bytes(static_cast<std::size_t>(size));
  std::ifstream input(path, std::ios::binary);
  if (!input || !input.read(reinterpret_cast<char*>(bytes.data()),
                            static_cast<std::streamsize>(bytes.size()))) {
    return Status(StatusCode::kIoError, "cannot read audio WAV " + path.string());
  }
  if (!FourCc(bytes.data(), "RIFF") || !FourCc(bytes.data() + 8U, "WAVE")) {
    return Invalid(path, "expected RIFF/WAVE header");
  }

  const std::byte* format = nullptr;
  std::size_t format_bytes = 0U;
  const std::byte* data = nullptr;
  std::size_t data_bytes = 0U;
  for (std::size_t cursor = 12U; cursor + 8U <= bytes.size();) {
    const std::uint32_t chunk_size = U32(bytes.data() + cursor + 4U);
    cursor += 8U;
    if (chunk_size > bytes.size() - cursor) {
      return Invalid(path, "chunk extends beyond the file");
    }
    if (FourCc(bytes.data() + cursor - 8U, "fmt ")) {
      format = bytes.data() + cursor;
      format_bytes = chunk_size;
    } else if (FourCc(bytes.data() + cursor - 8U, "data")) {
      data = bytes.data() + cursor;
      data_bytes = chunk_size;
    }
    cursor += chunk_size;
    if ((chunk_size & 1U) != 0U && cursor < bytes.size()) ++cursor;
  }
  if (format == nullptr || format_bytes < 16U || data == nullptr) {
    return Invalid(path, "required fmt/data chunk is missing");
  }
  const std::uint16_t encoding = U16(format);
  const std::uint16_t channels = U16(format + 2U);
  const std::uint32_t sample_rate = U32(format + 4U);
  const std::uint16_t block_align = U16(format + 12U);
  const std::uint16_t bits = U16(format + 14U);
  const bool pcm16 = encoding == 1U && bits == 16U;
  const bool float32 = encoding == 3U && bits == 32U;
  const std::uint16_t sample_bytes = static_cast<std::uint16_t>(bits / 8U);
  if ((channels != 1U && channels != 2U) || sample_rate < 8000U ||
      sample_rate > 48000U) {
    return Status(StatusCode::kUnsupported,
                  "audio WAV must be mono/stereo with an 8..48 kHz sample rate");
  }
  if ((!pcm16 && !float32) ||
      block_align != static_cast<std::uint16_t>(channels * sample_bytes) ||
      sample_bytes == 0U || data_bytes % block_align != 0U) {
    return Status(StatusCode::kUnsupported,
                  "audio WAV must use PCM16 or IEEE float32 samples");
  }
  const std::size_t input_frames = data_bytes / block_align;
  if (input_frames == 0U ||
      input_frames > static_cast<std::size_t>(sample_rate) * 30U) {
    return Status(StatusCode::kUnsupported,
                  "audio WAV duration must be greater than zero and at most 30 seconds");
  }

  std::vector<float> mono(input_frames);
  for (std::size_t frame = 0; frame < input_frames; ++frame) {
    float sum = 0.0F;
    for (std::uint16_t channel = 0U; channel < channels; ++channel) {
      const std::size_t index = frame * channels + channel;
      float value = 0.0F;
      if (pcm16) {
        value = static_cast<float>(std::bit_cast<std::int16_t>(
                    U16(data + index * 2U))) /
                32768.0F;
      } else {
        value = std::bit_cast<float>(U32(data + index * 4U));
        if (!std::isfinite(value)) return Invalid(path, "sample is not finite");
      }
      sum += value;
    }
    mono[frame] = sum / static_cast<float>(channels);
  }

  const std::size_t output_frames = static_cast<std::size_t>(
      (static_cast<std::uint64_t>(input_frames) * 16000U + sample_rate / 2U) /
      sample_rate);
  if (output_frames == 0U || output_frames > kMaximumSamples) {
    return Status(StatusCode::kUnsupported,
                  "converted audio exceeds the 30-second model limit");
  }
  AudioWaveform waveform;
  waveform.samples.resize(output_frames);
  if (sample_rate == 16000U) {
    waveform.samples = std::move(mono);
    return waveform;
  }
  for (std::size_t output = 0U; output < output_frames; ++output) {
    const double source =
        static_cast<double>(output) * static_cast<double>(sample_rate) / 16000.0;
    const std::size_t left = std::min(
        static_cast<std::size_t>(source), input_frames - 1U);
    const std::size_t right = std::min(left + 1U, input_frames - 1U);
    const float fraction = static_cast<float>(source - static_cast<double>(left));
    waveform.samples[output] =
        mono[left] + (mono[right] - mono[left]) * fraction;
  }
  return waveform;
}

}  // namespace gem16
