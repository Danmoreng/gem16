#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime_api.h>

#include "gem16/status.h"

namespace gem16::internal {

// Prompt-only FP8 GEMM. The checkpoint stores each [N,K] weight tensor in the
// column-major B memory order expected by CUTLASS, so no weight repack is
// required. The GEMM accumulates into FP32; its epilogue applies the dynamic
// per-token activation scale and checkpoint per-channel BF16 scale, rounds at
// the model's BF16 projection boundary, and stores physical BF16 values.
[[nodiscard]] Status LaunchFp8CutlassProjectionBatch(
    const std::uint8_t* activation_e4m3fn,
    const float* activation_scales,
    const std::uint8_t* weight_e4m3fn,
    const std::uint16_t* weight_scales_bf16,
    std::uint16_t* output_bf16,
    std::uint64_t tokens,
    std::uint64_t rows,
    std::uint64_t contracting_elements,
    void* workspace,
    std::size_t workspace_bytes,
    cudaStream_t stream);

}  // namespace gem16::internal
