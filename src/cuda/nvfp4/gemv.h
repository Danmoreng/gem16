#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

#include "gem16gb/status.h"

namespace gem16gb::internal {

// Experimental batch-one, direct-source NVFP4 GEMV. One warp owns one output row while the
// block cooperatively stages the packed activation and its E4M3 scales once for eight rows.
// The source checkpoint's packed values and compact group-16 scales are consumed directly.
[[nodiscard]] Status LaunchNvfp4SimtGemvProjection(
    const std::uint8_t* packed_activation_e2m1,
    const std::uint8_t* activation_scales_e4m3fn,
    const std::uint8_t* packed_weight_e2m1,
    const std::uint8_t* weight_scales_e4m3fn,
    float* output,
    std::uint64_t rows,
    std::uint64_t contracting_elements,
    float activation_global_divisor,
    float weight_global_divisor,
    cudaStream_t stream);

}  // namespace gem16gb::internal
