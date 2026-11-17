#include "cuda/norm/twelve_b_ordinary.h"
extern "C" int gem16_12b_ordinary_post_attention(float* projection,
 const std::uint16_t* post_weight,const float* residual,float* post_output,
 float* hidden_output,const std::uint16_t* pre_weight,std::uint8_t* packed,
 std::uint8_t* scales,float divisor,cudaStream_t stream) {
 return gem16::internal::LaunchTwelveBOrdinaryPostAttention(projection,post_weight,
     residual,post_output,hidden_output,pre_weight,packed,scales,divisor,stream).ok()?0:1;
}
