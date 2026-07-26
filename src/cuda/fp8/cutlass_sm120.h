#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime_api.h>

#include "gem16gb/status.h"

namespace gem16gb::internal {

// Prompt-only FP8 GEMM. The checkpoint stores each [N,K] weight tensor in the
// column-major B memory order expected by CUTLASS, so no weight repack is
// required. The GEMM accumulates into FP32; a second device kernel applies the
// dynamic per-token activation scale and checkpoint per-channel BF16 scale.
[[nodiscard]] Status LaunchFp8CutlassProjectionBatch(
    const std::uint8_t* activation_e4m3fn,
    const float* activation_scales,
    const std::uint8_t* weight_e4m3fn,
    const std::uint16_t* weight_scales_bf16,
    float* output,
    std::uint64_t tokens,
    std::uint64_t rows,
    std::uint64_t contracting_elements,
    void* workspace,
    std::size_t workspace_bytes,
    cudaStream_t stream);

}  // namespace gem16gb::internal
