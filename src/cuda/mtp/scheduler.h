#pragma once

#include <cstdint>

namespace gem16::internal {

// Deterministic host policy for optional MTP adaptation. Explicit D1/D2/D4
// bypasses all transitions. Thresholds come from the qualified 16K cost
// curve and are evaluated only after complete 16-group windows.
class AdaptiveMtpScheduler {
 public:
  AdaptiveMtpScheduler(std::uint32_t maximum_drafts,
                       std::uint64_t processed_position, bool enabled)
      : maximum_drafts_(maximum_drafts),
        active_drafts_(maximum_drafts),
        enabled_(enabled) {
    if (enabled_ && active_drafts_ == 4U && processed_position > 2048U) {
      active_drafts_ = 2U;
    }
  }

  [[nodiscard]] std::uint32_t active_drafts() const {
    return active_drafts_;
  }
  [[nodiscard]] bool use_ordinary_fallback() const {
    return enabled_ && ordinary_fallback_remaining_ != 0U;
  }
  [[nodiscard]] std::uint32_t ordinary_fallback_remaining() const {
    return ordinary_fallback_remaining_;
  }

  void ConsumeOrdinaryFallback() {
    if (ordinary_fallback_remaining_ != 0U) {
      --ordinary_fallback_remaining_;
    }
  }

  void Observe(std::uint64_t processed_position,
               std::uint32_t accepted_tokens) {
    if (!enabled_) return;
    ++window_groups_;
    window_accepted_tokens_ += accepted_tokens;
    if (window_groups_ < 16U) return;

    const float mean_acceptance =
        static_cast<float>(window_accepted_tokens_) /
        static_cast<float>(window_groups_);
    if (active_drafts_ == 4U &&
        (processed_position > 2048U || mean_acceptance < 1.25F)) {
      active_drafts_ = 2U;
    } else if (active_drafts_ == 2U && mean_acceptance < 0.65F) {
      active_drafts_ = 1U;
    } else if (active_drafts_ == 1U && mean_acceptance < 0.40F) {
      ordinary_fallback_remaining_ = 16U;
    } else if (active_drafts_ == 1U && maximum_drafts_ >= 2U &&
               mean_acceptance >= 0.70F) {
      active_drafts_ = 2U;
    } else if (active_drafts_ == 2U && maximum_drafts_ == 4U &&
               processed_position <= 2048U && mean_acceptance >= 1.60F) {
      active_drafts_ = 4U;
    }
    window_groups_ = 0U;
    window_accepted_tokens_ = 0U;
  }

 private:
  std::uint32_t maximum_drafts_ = 0U;
  std::uint32_t active_drafts_ = 0U;
  std::uint32_t window_groups_ = 0U;
  std::uint32_t window_accepted_tokens_ = 0U;
  std::uint32_t ordinary_fallback_remaining_ = 0U;
  bool enabled_ = false;
};

}  // namespace gem16::internal
