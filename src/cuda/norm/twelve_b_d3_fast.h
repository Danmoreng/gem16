#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>
#include "gem16/status.h"
namespace gem16::internal {
Status LaunchTwelveBD3FastInputNormFp8(const float*,const std::uint16_t*,std::uint8_t*,float*,cudaStream_t);
Status LaunchTwelveBD3FastAttentionQuantize(const float*,std::uint8_t*,float*,std::uint64_t,cudaStream_t);
Status LaunchTwelveBD3FastPostAttention(float*,const std::uint16_t*,const float*,float*,float*,const std::uint16_t*,std::uint8_t*,std::uint8_t*,float,cudaStream_t);
Status LaunchTwelveBD3FastPostMlp(const std::uint16_t*,const std::uint16_t*,const float*,float*,float*,const std::uint16_t*,cudaStream_t);
}
