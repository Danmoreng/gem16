#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

#include "gem16/status.h"

namespace gem16::internal {

[[nodiscard]] Status LaunchAudioProjection(
    float* frames, float* normalized_frames,
    const std::uint16_t* projection_weight, float* output,
    std::uint64_t frame_count, cudaStream_t stream);

}  // namespace gem16::internal
