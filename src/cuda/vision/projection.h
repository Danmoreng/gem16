#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

#include "cuda/engine/target_model.h"
#include "gem16/status.h"

namespace gem16::internal {

[[nodiscard]] Status LaunchVisionProjection(
    float* patches, const std::int32_t* positions, float* patch_normalized,
    float* hidden_a, float* hidden_b, const VisionBinding& weights,
    float* output, std::uint64_t patch_count, cudaStream_t stream);

}  // namespace gem16::internal
