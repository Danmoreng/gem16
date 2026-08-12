#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "gem16/status.h"

namespace gem16::nvfp4 {

// A non-owning, immutable view of the single canonical NVFP4 matrix used for
// both embedding lookup and output projection. The caller owns the two
// borrowed spans and must keep their bytes immutable and alive for the
// lifetime of the view. The two scalar divisors are copied into the view.
struct TiedNvfp4HeadView {
  struct ProjectionOptions {
    float softcap = 30.0F;
    std::span<const std::uint32_t> suppressed_tokens;
    bool return_diagnostic_logits = false;
  };

  struct ProjectionResult {
    std::uint32_t token = 0;
    float value = 0.0F;
    // Pre-suppression, post-softcap logits. This is populated only when
    // explicitly requested in ProjectionOptions.
    std::vector<float> diagnostic_logits;
  };

  // The packed and scale spans must exactly describe a row-major
  // [vocabulary_size, hidden_size] matrix. Packed values are low-nibble-first
  // E2M1 and scales are one E4M3FN value per group of 16 hidden elements.
  [[nodiscard]] static Result<TiedNvfp4HeadView> Create(
      std::span<const std::uint8_t> packed_weights,
      std::span<const std::uint8_t> local_scales,
      std::uint64_t vocabulary_size, std::uint64_t hidden_size,
      float weight_global_divisor, float input_global_divisor);

  [[nodiscard]] Result<std::vector<float>> Lookup(
      std::uint32_t token) const;

  // Reference-only T=1 output projection, selection, and optional diagnostic
  // logits. `hidden` is the post-final-normalization FP32 stream. The reference
  // applies the model's BF16-RNE physical boundary before NVFP4 activation
  // quantization. Suppression IDs are caller supplied, must contain at most 16
  // unique in-range IDs, and must leave one vocabulary item selectable.
  [[nodiscard]] Result<ProjectionResult> ProjectT1(
      std::span<const float> hidden,
      const ProjectionOptions& options) const;
  [[nodiscard]] Result<ProjectionResult> ProjectT1(
      std::span<const float> hidden) const {
    return ProjectT1(hidden, ProjectionOptions{});
  }

  // These accessors expose the borrowed storage identity for integration tests;
  // no data is copied or repacked by this view.
  [[nodiscard]] std::span<const std::uint8_t> packed_weights() const noexcept {
    return packed_weights_;
  }
  [[nodiscard]] std::span<const std::uint8_t> local_scales() const noexcept {
    return local_scales_;
  }
  [[nodiscard]] std::uint64_t vocabulary_size() const noexcept {
    return vocabulary_size_;
  }
  [[nodiscard]] std::uint64_t hidden_size() const noexcept { return hidden_size_; }
  [[nodiscard]] float weight_global_divisor() const noexcept {
    return weight_global_divisor_;
  }
  [[nodiscard]] float input_global_divisor() const noexcept {
    return input_global_divisor_;
  }

 private:
  TiedNvfp4HeadView(std::span<const std::uint8_t> packed_weights,
                    std::span<const std::uint8_t> local_scales,
                    std::uint64_t vocabulary_size,
                    std::uint64_t hidden_size,
                    float weight_global_divisor,
                    float input_global_divisor)
      : packed_weights_(packed_weights),
        local_scales_(local_scales),
        vocabulary_size_(vocabulary_size),
        hidden_size_(hidden_size),
        weight_global_divisor_(weight_global_divisor),
        input_global_divisor_(input_global_divisor) {}

  std::span<const std::uint8_t> packed_weights_;
  std::span<const std::uint8_t> local_scales_;
  std::uint64_t vocabulary_size_ = 0;
  std::uint64_t hidden_size_ = 0;
  float weight_global_divisor_ = 1.0F;
  float input_global_divisor_ = 1.0F;
};

}  // namespace gem16::nvfp4
