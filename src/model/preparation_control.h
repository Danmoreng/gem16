#pragma once

#include "gem16/status.h"

namespace gem16::internal {

enum class PreparationPhase { kRequest, kBase64, kImageDecode, kImageResize, kImagePatchify, kAudioDecode };

inline const char* PreparationPhaseName(PreparationPhase phase) {
  switch (phase) {
    case PreparationPhase::kRequest: return "request_prepare";
    case PreparationPhase::kBase64: return "media_base64";
    case PreparationPhase::kImageDecode: return "image_decode";
    case PreparationPhase::kImageResize: return "image_resize";
    case PreparationPhase::kImagePatchify: return "image_patchify";
    case PreparationPhase::kAudioDecode: return "audio_decode";
  }
  return "request_prepare";
}

// Request-thread scope, like ImageDecodeBudget. It never reaches a CUDA hot
// loop, and standalone model acquisition retains its existing behavior.
class PreparationControl {
 public:
  using Callback = Status (*)(void*, PreparationPhase);
  PreparationControl(Callback callback, void* context)
      : callback_(callback), context_(context), previous_(current_) { current_ = this; }
  ~PreparationControl() { current_ = previous_; }
  PreparationControl(const PreparationControl&) = delete;
  PreparationControl& operator=(const PreparationControl&) = delete;
  static Status Check(PreparationPhase phase) {
    return current_ && current_->callback_ ? current_->callback_(current_->context_, phase) : Status::Ok();
  }
 private:
  Callback callback_;
  void* context_;
  PreparationControl* previous_;
  static inline thread_local PreparationControl* current_ = nullptr;
};

}  // namespace gem16::internal
