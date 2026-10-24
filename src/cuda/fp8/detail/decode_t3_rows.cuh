#pragma once
// Native fixed-R3 projections, included inside the FP8 private namespace.
// Native fragment mapping is the existing M16 matrix mapping, restricted to R3.
template <unsigned Padding>
__launch_bounds__(kThreadsPerBlock,1) __global__
void Sm120DirectProjectionPackedT3Kernel(const std::uint8_t* activation,
    const float* activation_scale,Fp8MatrixBinding first,Fp8MatrixBinding second,
    Fp8MatrixBinding third,unsigned binding_count,std::uint64_t contracting_elements){
  static_assert(Padding==32U);
  if(blockIdx.z>=binding_count)return;
  const auto binding=blockIdx.z==0?first:blockIdx.z==1?second:third;
  const unsigned warp=threadIdx.x/32U,lane=threadIdx.x&31U,group=lane>>2U,quarter=lane&3U;
  const std::uint64_t global_warp=std::uint64_t(blockIdx.x)*kWarpsPerBlock+warp;
  const std::uint64_t row_tiles=(binding.rows+7U)/8U;
  extern __shared__ __align__(16) std::uint8_t staged[];
  const std::uint64_t stride=contracting_elements+Padding;
  const std::uint64_t vectors_per_row=contracting_elements/16U;
  for(std::uint64_t vector=threadIdx.x;vector<3U*vectors_per_row;vector+=blockDim.x){
    const auto token=vector/vectors_per_row,chunk=vector%vectors_per_row;
    *reinterpret_cast<uint4*>(staged+token*stride+chunk*16U)=
        *reinterpret_cast<const uint4*>(activation+token*contracting_elements+chunk*16U);
  }
  __syncthreads();if(global_warp>=row_tiles)return;
  const auto source_row=global_warp*8U+group;
  Fp8Accumulator accumulator{};
  for(std::uint64_t k=0;k<contracting_elements/32U;++k){
    const auto byte=k*32U+quarter*4U;
    std::uint32_t b0=0,b1=0,a0=0,a2=0;
    if(source_row<binding.rows){b0=LoadU32(binding.weight+source_row*contracting_elements+byte);b1=LoadU32(binding.weight+source_row*contracting_elements+byte+16U);}
    if(group<3U){a0=LoadU32(staged+group*stride+byte);a2=LoadU32(staged+group*stride+byte+16U);}
    AccumulateFp8(a0,0U,a2,0U,b0,b1,accumulator);
  }
  if(group<3U){
    const auto column=global_warp*8U+quarter*2U;
    const float scale=activation_scale[group];
    if(column<binding.rows)binding.output[std::uint64_t(group)*binding.rows+column]=accumulator.x0*scale*DecodeBf16(binding.weight_scales+column);
    if(column+1U<binding.rows)binding.output[std::uint64_t(group)*binding.rows+column+1U]=accumulator.x1*scale*DecodeBf16(binding.weight_scales+column+1U);
  }
}

__launch_bounds__(kSplitKThreadsPerBlock,1) __global__
void Sm120DirectProjectionSplitK2PackedT3Kernel(const std::uint8_t* activation,
    const float* activation_scale,const std::uint8_t* weight,
    const std::uint16_t* weight_scales,float* output,std::uint64_t rows,
    std::uint64_t contracting_elements){
  __shared__ float second_half[3][8];
  const unsigned split=threadIdx.x/32U,lane=threadIdx.x&31U,group=lane>>2U,quarter=lane&3U;
  const std::uint64_t tile=blockIdx.x,source_row=tile*8U+group;
  const auto half=contracting_elements/32U/2U;
  Fp8Accumulator accumulator{};
  if(tile<(rows+7U)/8U)for(std::uint64_t k=split*half;k<(split+1U)*half;++k){
    const auto byte=k*32U+quarter*4U;
    std::uint32_t b0=0,b1=0,a0=0,a2=0;
    if(source_row<rows){b0=LoadU32(weight+source_row*contracting_elements+byte);b1=LoadU32(weight+source_row*contracting_elements+byte+16U);}
    if(group<3U){a0=LoadU32(activation+group*contracting_elements+byte);a2=LoadU32(activation+group*contracting_elements+byte+16U);}
    AccumulateFp8(a0,0U,a2,0U,b0,b1,accumulator);
  }
  if(split==1U&&group<3U){second_half[group][quarter*2U]=accumulator.x0;second_half[group][quarter*2U+1U]=accumulator.x1;}
  __syncthreads();
  if(split==0U&&group<3U){
    const auto column=tile*8U+quarter*2U;const float scale=activation_scale[group];
    if(column<rows)output[std::uint64_t(group)*rows+column]=(accumulator.x0+second_half[group][quarter*2U])*scale*DecodeBf16(weight_scales+column);
    if(column+1U<rows)output[std::uint64_t(group)*rows+column+1U]=(accumulator.x1+second_half[group][quarter*2U+1U])*scale*DecodeBf16(weight_scales+column+1U);
  }
}

