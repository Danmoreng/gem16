#pragma once
#include <cstdint>
#include <cuda_runtime_api.h>
#include "gem16/status.h"

namespace gem16::internal {
struct DecodeControl;

// The optimized verifier and Assistant share an offline-compiled kernel.
// Launch uses the physical slot capacity and preallocated graph workspace.
class TwelveBD3FastGlobalAttention {
 public:
  TwelveBD3FastGlobalAttention() = default;
  TwelveBD3FastGlobalAttention(const TwelveBD3FastGlobalAttention&) = delete;
  TwelveBD3FastGlobalAttention& operator=(const TwelveBD3FastGlobalAttention&) = delete;
  ~TwelveBD3FastGlobalAttention();
  [[nodiscard]] Status Initialize();
  [[nodiscard]] Status Launch(const float* query, const std::uint8_t* key,
      const std::uint8_t* value, const std::uint16_t* key_scale,
      const std::uint16_t* value_scale, float* workspace,
      std::uint64_t workspace_elements, float* output,
      const DecodeControl* controls, std::uint64_t capacity,
      cudaStream_t stream, unsigned rows = 4U) const;
  static constexpr std::uint64_t WorkspaceElements() {
    // Includes16KiB of per-CTA TMA descriptor scratch, reserved before capture.
    return 4U * 16U * (16U * 512U + 16U) + 4096U;
  }
 private:
  cudaLibrary_t library_ = nullptr;
  cudaKernel_t kernel_ = nullptr;
};
}  // namespace gem16::internal
