#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

#include "gem16/status.h"

namespace gem16 {

struct AudioWaveform {
  std::vector<float> samples;
  std::uint32_t sample_rate = 16000U;

  bool operator==(const AudioWaveform&) const = default;
};

// Loads a bounded RIFF/WAVE file. PCM16 and IEEE float32, mono/stereo, and
// 8..48 kHz are accepted and deterministically converted to mono 16 kHz.
[[nodiscard]] Result<AudioWaveform> LoadAudioWav(
    const std::filesystem::path& path);

}  // namespace gem16
