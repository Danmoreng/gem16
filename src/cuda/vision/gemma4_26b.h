#pragma once

#include <cstdint>
#include <memory>

#include <cuda_runtime_api.h>

#include "cuda/engine/gemma4_26b_vision_artifact.h"
#include "gem16/engine.h"
#include "gem16/status.h"

namespace gem16::internal {

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
      const Gemma4Moe26BVisionInputSegment& segment, cudaStream_t stream);

  // Physical BF16-rounded values expanded into float slots, row-major
  // [soft_token_count, 2816]. Stable until the next Encode call.
  [[nodiscard]] const float* output() const;
  [[nodiscard]] std::uint32_t output_tokens() const;
  [[nodiscard]] std::uint64_t workspace_bytes() const;

 private:
  struct Impl;
  explicit Gemma4Moe26BVisionRuntime(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gem16::internal
