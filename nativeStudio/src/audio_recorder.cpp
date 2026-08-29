#include "audio_recorder.h"

#define MINIAUDIO_IMPLEMENTATION
#include <miniaudio.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>
#include <vector>

namespace gem16::studio {
namespace {

constexpr ma_uint32 kSampleRate = 16000;
constexpr ma_uint32 kChannels = 1;
constexpr std::size_t kMaximumFrames = 5U * 60U * kSampleRate;

void AppendU16(std::vector<std::uint8_t>& output, std::uint16_t value) {
  output.push_back(static_cast<std::uint8_t>(value));
  output.push_back(static_cast<std::uint8_t>(value >> 8U));
}

void AppendU32(std::vector<std::uint8_t>& output, std::uint32_t value) {
  for (int shift = 0; shift < 32; shift += 8)
    output.push_back(static_cast<std::uint8_t>(value >> shift));
}

std::vector<std::uint8_t> WavBytes(const std::int16_t* samples,
                                   std::size_t sample_count) {
  const auto data_bytes = static_cast<std::uint32_t>(sample_count * sizeof(std::int16_t));
  std::vector<std::uint8_t> output;
  output.reserve(44U + data_bytes);
  output.insert(output.end(), {'R', 'I', 'F', 'F'});
  AppendU32(output, 36U + data_bytes);
  output.insert(output.end(), {'W', 'A', 'V', 'E', 'f', 'm', 't', ' '});
  AppendU32(output, 16U);
  AppendU16(output, 1U);
  AppendU16(output, static_cast<std::uint16_t>(kChannels));
  AppendU32(output, kSampleRate);
  AppendU32(output, kSampleRate * kChannels * sizeof(std::int16_t));
  AppendU16(output, static_cast<std::uint16_t>(kChannels * sizeof(std::int16_t)));
  AppendU16(output, 16U);
  output.insert(output.end(), {'d', 'a', 't', 'a'});
  AppendU32(output, data_bytes);
  const auto* begin = reinterpret_cast<const std::uint8_t*>(samples);
  output.insert(output.end(), begin, begin + data_bytes);
  return output;
}

}  // namespace

struct AudioRecorder::Impl {
  ma_device device{};
  std::vector<std::int16_t> samples;
  std::atomic<std::size_t> frames{0};
  std::atomic<bool> recording{false};
  bool initialized = false;
};

AudioRecorder::AudioRecorder() : impl_(std::make_unique<Impl>()) {}

AudioRecorder::~AudioRecorder() {
  if (impl_->initialized) {
    if (impl_->recording.exchange(false)) ma_device_stop(&impl_->device);
    ma_device_uninit(&impl_->device);
  }
}

bool AudioRecorder::Start(std::string& error) {
  error.clear();
  if (impl_->recording.load()) return true;
  if (impl_->initialized) {
    ma_device_uninit(&impl_->device);
    impl_->initialized = false;
  }
  impl_->samples.assign(kMaximumFrames * kChannels, 0);
  impl_->frames.store(0);
  ma_device_config config = ma_device_config_init(ma_device_type_capture);
  config.capture.format = ma_format_s16;
  config.capture.channels = kChannels;
  config.sampleRate = kSampleRate;
  config.pUserData = impl_.get();
  config.dataCallback = [](ma_device* device, void*, const void* input,
                           ma_uint32 frame_count) {
    auto* state = static_cast<Impl*>(device->pUserData);
    if (!state->recording.load() || input == nullptr) return;
    const std::size_t begin = state->frames.fetch_add(frame_count);
    const std::size_t accepted = begin < kMaximumFrames
                                     ? std::min<std::size_t>(frame_count,
                                                             kMaximumFrames - begin)
                                     : 0U;
    if (accepted != 0U) {
      std::memcpy(state->samples.data() + begin * kChannels, input,
                  accepted * kChannels * sizeof(std::int16_t));
    }
    if (accepted != frame_count) state->recording.store(false);
  };
  const ma_result initialized = ma_device_init(nullptr, &config, &impl_->device);
  if (initialized != MA_SUCCESS) {
    error = std::string("Could not open the default microphone: ") +
            ma_result_description(initialized);
    return false;
  }
  impl_->initialized = true;
  impl_->recording.store(true);
  const ma_result started = ma_device_start(&impl_->device);
  if (started != MA_SUCCESS) {
    impl_->recording.store(false);
    ma_device_uninit(&impl_->device);
    impl_->initialized = false;
    error = std::string("Could not start microphone capture: ") +
            ma_result_description(started);
    return false;
  }
  return true;
}

bool AudioRecorder::Stop(MediaAttachment& attachment, std::string& error) {
  error.clear();
  if (!impl_->initialized) {
    error = "Microphone recording is not active";
    return false;
  }
  impl_->recording.store(false);
  ma_device_stop(&impl_->device);
  const std::size_t frames = std::min(impl_->frames.load(), kMaximumFrames);
  ma_device_uninit(&impl_->device);
  impl_->initialized = false;
  if (frames < kSampleRate / 4U) {
    error = "Recording was shorter than 250 ms";
    return false;
  }
  attachment = {};
  attachment.kind = MediaKind::kAudio;
  attachment.file_name = "microphone.wav";
  attachment.mime_type = "audio/wav";
  attachment.format = "wav";
  attachment.bytes = WavBytes(impl_->samples.data(), frames * kChannels);
  attachment.byte_size = attachment.bytes.size();
  return true;
}

bool AudioRecorder::Recording() const { return impl_->recording.load(); }

}  // namespace gem16::studio
