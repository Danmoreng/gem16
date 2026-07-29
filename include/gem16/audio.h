#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <string_view>
#include <vector>

#include "gem16/status.h"

namespace gem16 {

struct AudioWaveform {
  std::vector<float> samples;
  std::uint32_t sample_rate = 16000U;

  bool operator==(const AudioWaveform&) const = default;
};

// Decodes a bounded WAV, FLAC, or MP3 file and converts it to mono float32 at
// the model's fixed 16-kHz sample rate.
[[nodiscard]] Result<AudioWaveform> LoadAudioFile(
    const std::filesystem::path& path);
[[nodiscard]] Result<AudioWaveform> LoadAudioBytes(
    std::span<const std::uint8_t> encoded, std::string_view source_name);

// Backward-compatible name retained for callers built against the WAV-only
// milestone. The decoder now accepts every format supported by LoadAudioFile.
[[nodiscard]] Result<AudioWaveform> LoadAudioWav(
    const std::filesystem::path& path);

}  // namespace gem16
