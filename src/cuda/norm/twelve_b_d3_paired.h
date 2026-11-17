#pragma once
#include "gem16/status.h"
#include <cuda_runtime_api.h>
#include <cstdint>
namespace gem16::internal {
Status LaunchTwelveBD3PairedPostAttention(float*,const std::uint16_t*,const float*,
    float*,float*,const std::uint16_t*,std::uint8_t*,std::uint8_t*,float,cudaStream_t);
}
