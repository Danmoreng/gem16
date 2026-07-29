#include "gem16/audio.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

#include "test.h"

namespace {

void Put16(std::vector<std::uint8_t>& bytes, std::uint16_t value) {
  bytes.push_back(static_cast<std::uint8_t>(value));
  bytes.push_back(static_cast<std::uint8_t>(value >> 8U));
}

void Put32(std::vector<std::uint8_t>& bytes, std::uint32_t value) {
  for (unsigned shift = 0U; shift < 32U; shift += 8U) {
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
  }
}

void Text(std::vector<std::uint8_t>& bytes, const char* value) {
  for (unsigned index = 0U; index < 4U; ++index) {
    bytes.push_back(static_cast<std::uint8_t>(value[index]));
  }
}

std::filesystem::path WritePcm16Wav(std::uint32_t sample_rate,
                                    std::uint16_t channels) {
  const std::vector<std::int16_t> samples = {-32768, 0, 16384, 32767};
  const std::uint32_t data_bytes =
      static_cast<std::uint32_t>(samples.size() * sizeof(std::int16_t));
  std::vector<std::uint8_t> bytes;
  Text(bytes, "RIFF"); Put32(bytes, 36U + data_bytes); Text(bytes, "WAVE");
  Text(bytes, "fmt "); Put32(bytes, 16U); Put16(bytes, 1U);
  Put16(bytes, channels); Put32(bytes, sample_rate);
  Put32(bytes, sample_rate * channels * 2U); Put16(bytes, channels * 2U);
  Put16(bytes, 16U); Text(bytes, "data"); Put32(bytes, data_bytes);
  for (const std::int16_t sample : samples) {
    Put16(bytes, static_cast<std::uint16_t>(sample));
  }
  const auto path = std::filesystem::temp_directory_path() /
                    (sample_rate == 16000U && channels == 1U
                         ? "gem16_audio_valid.wav"
                         : "gem16_audio_invalid.wav");
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  return path;
}

std::filesystem::path WriteInvalidAudio() {
  const auto path = std::filesystem::temp_directory_path() /
                    "gem16_audio_malformed.bin";
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write("not audio", 9);
  return path;
}

}  // namespace

void RunAudioTests() {
  const auto valid_path = WritePcm16Wav(16000U, 1U);
  auto valid = gem16::LoadAudioFile(valid_path);
  GEM16_CHECK(valid.ok());
  if (valid.ok()) {
    GEM16_CHECK(valid.value().sample_rate == 16000U);
    GEM16_CHECK(valid.value().samples.size() == 4U);
    GEM16_CHECK(valid.value().samples[0] == -1.0F);
    GEM16_CHECK(valid.value().samples[2] == 0.5F);
    GEM16_CHECK(valid.value().samples[3] > 0.999F);
  }
  std::ifstream encoded_input(valid_path, std::ios::binary);
  const std::vector<std::uint8_t> encoded(
      std::istreambuf_iterator<char>(encoded_input), {});
  auto memory_audio = gem16::LoadAudioBytes(encoded, "unit WAV");
  GEM16_CHECK(memory_audio.ok());
  if (memory_audio.ok()) {
    GEM16_CHECK(memory_audio.value().samples.size() == 4U);
  }
  std::error_code ignored;
  std::filesystem::remove(valid_path, ignored);

  const auto resampled_path = WritePcm16Wav(22050U, 1U);
  auto resampled = gem16::LoadAudioFile(resampled_path);
  GEM16_CHECK(resampled.ok());
  if (resampled.ok()) {
    GEM16_CHECK(resampled.value().sample_rate == 16000U);
    GEM16_CHECK(resampled.value().samples.size() == 3U);
  }
  std::filesystem::remove(resampled_path, ignored);

  const auto invalid_path = WriteInvalidAudio();
  auto invalid = gem16::LoadAudioFile(invalid_path);
  GEM16_CHECK(!invalid.ok());
  if (!invalid.ok()) {
    GEM16_CHECK(invalid.status().code() == gem16::StatusCode::kUnsupported);
  }
  std::filesystem::remove(invalid_path, ignored);
}
