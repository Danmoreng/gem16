#pragma once
#include <cstdint>

namespace gem16::internal {

// A parser owns this scope on its request thread. Standalone/offline image
// loading retains its existing limits. No cross-request mutable budget.
class ImageDecodeBudget {
 public:
  ImageDecodeBudget() : previous_(current_) { current_ = this; }
  ImageDecodeBudget(const ImageDecodeBudget&) = delete;
  ImageDecodeBudget& operator=(const ImageDecodeBudget&) = delete;
  ~ImageDecodeBudget() { current_ = previous_; }
  static bool Pixels(std::uint64_t count) {
    return !current_ || Charge(current_->pixels_, count);
  }
  static bool PreparedBytes(std::uint64_t count) {
    return !current_ || Charge(current_->prepared_bytes_, count);
  }
 private:
  static bool Charge(std::uint64_t& remaining, std::uint64_t count) {
    if (count > remaining) return false;
    remaining -= count;
    return true;
  }
  std::uint64_t pixels_ = 32'000'000U;
  std::uint64_t prepared_bytes_ = 256U * 1024U * 1024U;
  ImageDecodeBudget* previous_;
  static inline thread_local ImageDecodeBudget* current_ = nullptr;
};
}  // namespace gem16::internal
