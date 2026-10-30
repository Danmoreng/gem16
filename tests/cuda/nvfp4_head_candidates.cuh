// Offline bounded native NVFP4 head candidates, included in the private namespace.
// SF_A layout is the existing matrix kernel's thread-id-a0 map: lower two
// lanes of each quad select row group and group+8; only first three rows exist.
__launch_bounds__(kThreadsPerBlock,1) __global__
void Sm120HeadPackedT3CandidateKernel(const std::uint8_t* a,const std::uint8_t* sa,
    const std::uint8_t* b,const std::uint8_t* sb,float* out,std::uint64_t rows,
    std::uint64_t width,float divisor){
  const unsigned warp=threadIdx.x/32U,lane=threadIdx.x&31U,group=lane>>2U,quarter=lane&3U;
  const std::uint64_t tile=std::uint64_t(blockIdx.x)*4U+warp;
  if(tile>=rows/8U)return;
  const auto blocks=width/64U,first_row=tile*8U;
  float d0=0.F,d1=0.F,d2=0.F,d3=0.F;
  for(std::uint64_t k=0;k<blocks;++k){
    const auto offset=k*32U+quarter*4U;
    std::uint32_t a0=0,a2=0,scale_a=0;
    if(group<3U){a0=LoadU32(a+group*(width/2U)+offset);a2=LoadU32(a+group*(width/2U)+offset+16U);}
    if(quarter==0U&&group<3U)scale_a=LoadU32(sa+group*(width/16U)+k*4U);
    const auto woffset=first_row*blocks*32U+(k*8U+group)*32U+quarter*4U;
    const auto scale_b=LoadU32(sb+first_row*blocks*4U+(k*8U+group)*4U);
    MmaNvfp4(d0,d1,d2,d3,a0,0U,a2,0U,LoadU32(b+woffset),LoadU32(b+woffset+16U),scale_a,scale_b);
  }
  if(group<3U){const auto column=first_row+quarter*2U;
    out[group*rows+column]=float(__float2bfloat16_rn(d0/divisor));
    out[group*rows+column+1U]=float(__float2bfloat16_rn(d1/divisor));}
}

__launch_bounds__(kHeadCandidateThreadsPerBlock,1) __global__
void Sm120HeadPairedCandidateKernel(const std::uint8_t* a,const std::uint8_t* sa,
    const std::uint8_t* b,const std::uint8_t* sb,Nvfp4ProjectionCandidate* out,
    std::uint64_t rows,std::uint64_t width,float divisor,std::uint32_t mask0,std::uint32_t mask1){
  __shared__ Nvfp4ProjectionCandidate best_warp[8];
  const unsigned warp=threadIdx.x/32U,lane=threadIdx.x&31U,group=lane>>2U,quarter=lane&3U;
  const std::uint64_t tile=(std::uint64_t(blockIdx.x)*8U+warp)*2U;
  const auto first_row=tile*8U,blocks=width/64U;
  float acc[2][4]={};
  for(std::uint64_t k=0;k<blocks;++k){
    const auto offset=k*32U+quarter*4U;
    const auto a0=LoadU32(a+offset),a2=LoadU32(a+offset+16U),scale_a=LoadU32(sa+k*4U);
#pragma unroll
    for(unsigned j=0;j<2U;++j){const auto row=first_row+j*8U;
      const auto woffset=row*blocks*32U+(k*8U+group)*32U+quarter*4U;
      const auto scale_b=LoadU32(sb+row*blocks*4U+(k*8U+group)*4U);
      MmaNvfp4(acc[j][0],acc[j][1],acc[j][2],acc[j][3],a0,a0,a2,a2,LoadU32(b+woffset),LoadU32(b+woffset+16U),scale_a,scale_b);
    }
  }
  Nvfp4ProjectionCandidate best{-3.402823466e+38F,0U};
  if(lane<4U)for(unsigned j=0;j<2U;++j){
    const auto token=static_cast<std::uint32_t>(first_row+j*8U+lane*2U);
    const float v0=float(__float2bfloat16_rn(acc[j][0]/divisor)),v1=float(__float2bfloat16_rn(acc[j][1]/divisor));
    if(isfinite(v0)&&token!=mask0&&token!=mask1)best=BetterCandidate(best,{v0,token});
    if(isfinite(v1)&&token+1U!=mask0&&token+1U!=mask1)best=BetterCandidate(best,{v1,token+1U});
  }
  for(unsigned offset=16;offset;offset>>=1){const Nvfp4ProjectionCandidate other{__shfl_down_sync(0xffffffffU,best.value,offset),__shfl_down_sync(0xffffffffU,best.token,offset)};best=BetterCandidate(best,other);}
  if(lane==0U)best_warp[warp]=best;__syncthreads();
  // Two separate original64-token partitions; the total-order max over finite
  // BF16 values, then lower token, is unchanged by grouping adjacent N8 tiles.
  if(warp<2U){best=lane<4U?best_warp[warp*4U+lane]:Nvfp4ProjectionCandidate{-3.402823466e+38F,0U};
    for(unsigned offset=16;offset;offset>>=1){const Nvfp4ProjectionCandidate other{__shfl_down_sync(0xffffffffU,best.value,offset),__shfl_down_sync(0xffffffffU,best.token,offset)};best=BetterCandidate(best,other);}
    if(lane==0U)out[blockIdx.x*2U+warp]=best;
  }
  (void)rows;
}
