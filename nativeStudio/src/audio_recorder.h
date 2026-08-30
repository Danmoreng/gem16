#pragma once

#include "types.h"

#include <memory>
#include <string>

namespace gem16::studio {

inline constexpr std::uint64_t kMaximumRecordingSeconds = 30;

class AudioRecorder final {
 public:
  AudioRecorder();
  ~AudioRecorder();
  AudioRecorder(const AudioRecorder&) = delete;
  AudioRecorder& operator=(const AudioRecorder&) = delete;

  [[nodiscard]] bool Start(std::string& error);
  [[nodiscard]] bool Stop(MediaAttachment& attachment, std::string& error);
  [[nodiscard]] bool Recording() const;
  [[nodiscard]] bool Active() const;
  [[nodiscard]] std::uint64_t ElapsedMilliseconds() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gem16::studio
