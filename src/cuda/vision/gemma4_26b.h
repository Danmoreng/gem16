#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <span>

#include <cuda_runtime_api.h>

#include "cuda/engine/gemma4_26b_vision_artifact.h"
#include "gem16/engine.h"
#include "gem16/status.h"

namespace gem16::internal {

struct Gemma4Moe26BVisionLayerTimings {
  float input_norm_quant_milliseconds = 0.0F;
  float qkv_projection_milliseconds = 0.0F;
  float qkv_norm_rope_milliseconds = 0.0F;
  float attention_milliseconds = 0.0F;
  float output_projection_residual_milliseconds = 0.0F;
  float ffn_norm_quant_milliseconds = 0.0F;
  float gate_up_milliseconds = 0.0F;
  float gelu_milliseconds = 0.0F;
  float product_quant_milliseconds = 0.0F;
  float down_residual_milliseconds = 0.0F;
};

struct Gemma4Moe26BVisionRuntimeTimings {
  std::uint32_t budget = 0U;
  std::uint32_t raw_patch_count = 0U;
  std::uint32_t soft_token_count = 0U;
  float upload_milliseconds = 0.0F;
  float patch_project_milliseconds = 0.0F;
  float position_add_milliseconds = 0.0F;
  std::array<Gemma4Moe26BVisionLayerTimings, 27U> layers{};
  // The correctness-first runtime physically fuses these two boundaries.
  float pool_standardize_milliseconds = 0.0F;
  float final_norm_project_milliseconds = 0.0F;
  float total_gpu_milliseconds = 0.0F;
};

// Diagnostic-only fixed CUDA-event recorder. Events are allocated before
// Encode, so the measured path retains the recurring-allocation contract.
class Gemma4Moe26BVisionTimingRecorder {
 public:
  Gemma4Moe26BVisionTimingRecorder();
  ~Gemma4Moe26BVisionTimingRecorder();
  Gemma4Moe26BVisionTimingRecorder(
      const Gemma4Moe26BVisionTimingRecorder&) = delete;
  Gemma4Moe26BVisionTimingRecorder& operator=(
      const Gemma4Moe26BVisionTimingRecorder&) = delete;
  Gemma4Moe26BVisionTimingRecorder(
      Gemma4Moe26BVisionTimingRecorder&&) noexcept;
  Gemma4Moe26BVisionTimingRecorder& operator=(
      Gemma4Moe26BVisionTimingRecorder&&) noexcept;

  [[nodiscard]] static Result<Gemma4Moe26BVisionTimingRecorder> Create();
  [[nodiscard]] Status Resolve(
      Gemma4Moe26BVisionRuntimeTimings* timings);

 private:
  struct Impl;
  explicit Gemma4Moe26BVisionTimingRecorder(std::unique_ptr<Impl> impl);
  [[nodiscard]] Status Begin(
      cudaStream_t stream, const Gemma4Moe26BVisionInputSegment& segment);
  [[nodiscard]] Status Boundary(cudaStream_t stream);
  std::unique_ptr<Impl> impl_;
  friend class Gemma4Moe26BVisionRuntime;
};

// Correctness-first Gemma 4 26B Vision executor. It consumes the immutable
// FP8/BF16 module directly and owns one fixed workspace plus pinned input
// staging. Encode performs no allocation, filesystem access, JIT, or repack.
class Gemma4Moe26BVisionRuntime {
 public:
  Gemma4Moe26BVisionRuntime();
  ~Gemma4Moe26BVisionRuntime();
  Gemma4Moe26BVisionRuntime(const Gemma4Moe26BVisionRuntime&) = delete;
  Gemma4Moe26BVisionRuntime& operator=(
      const Gemma4Moe26BVisionRuntime&) = delete;
  Gemma4Moe26BVisionRuntime(Gemma4Moe26BVisionRuntime&&) noexcept;
  Gemma4Moe26BVisionRuntime& operator=(
      Gemma4Moe26BVisionRuntime&&) noexcept;

  [[nodiscard]] static Result<Gemma4Moe26BVisionRuntime> Create(
      const Gemma4Moe26BVisionDeviceArtifact& artifact);
  [[nodiscard]] Status Encode(
      const Gemma4Moe26BVisionInputSegment& segment, cudaStream_t stream,
      Gemma4Moe26BVisionTimingRecorder* timing = nullptr,
      std::span<cudaEvent_t> phase_events = {});

  // Physical BF16-rounded values expanded into float slots, row-major
  // [soft_token_count, 2816]. Stable until the next Encode call.
  [[nodiscard]] const float* output() const;
  [[nodiscard]] std::uint32_t output_tokens() const;
  [[nodiscard]] std::uint64_t workspace_bytes() const;
  [[nodiscard]] std::uint64_t host_pinned_bytes() const;

 private:
  struct Impl;
  explicit Gemma4Moe26BVisionRuntime(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gem16::internal
