"""One exact Down ILP candidate: two N8 tiles per warp, same CTA N64 range."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('destination', type=Path)
a = p.parse_args()
source = ROOT / 'src/cuda/nvfp4/sm120.cu'
s = source.read_text() + r'''
namespace gem16::internal { namespace {
// Independent N-tile accumulators retain each output's sequential K64 sum.
// Two N8 tiles per warp and four warps cover the same N64 CTA as the parent.
__global__ void Sm120D2DownDualTileKernel(
 const std::uint8_t*a,const std::uint8_t*sa,const std::uint8_t*b,
 const std::uint8_t*sb,std::uint16_t*y,float divisor){
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
 constexpr unsigned KBlocks=240U,Width=15360U,Rows=3840U;
 const unsigned warp=threadIdx.x/32U,lane=threadIdx.x%32U;
 const unsigned group=lane>>2U,q=lane&3U;
 const unsigned tile0=(blockIdx.x*4U+warp)*2U;
 float accum[2][4]={};
 for(unsigned k=0;k<KBlocks;++k){
  const unsigned off=k*32U+q*4U;
  const std::uint32_t a0=group<3U?LoadU32(a+group*(Width/2U)+off):0U;
  const std::uint32_t a2=group<3U?LoadU32(a+group*(Width/2U)+off+16U):0U;
  const std::uint32_t scale_a=q==0U&&group<3U?LoadU32(sa+group*(Width/16U)+k*4U):0U;
#pragma unroll
  for(unsigned n=0;n<2U;++n){
   const unsigned first_row=(tile0+n)*8U;
   const unsigned bo=first_row*KBlocks*32U+(k*8U+group)*32U+q*4U;
   const unsigned so=first_row*KBlocks*4U+(k*8U+group)*4U;
   const std::uint32_t b0=LoadU32(b+bo),b1=LoadU32(b+bo+16U),scale_b=LoadU32(sb+so);
   MmaNvfp4(accum[n][0],accum[n][1],accum[n][2],accum[n][3],a0,0U,a2,0U,b0,b1,scale_a,scale_b);
  }
 }
 if(group<3U){
#pragma unroll
  for(unsigned n=0;n<2U;++n){
   const unsigned col=(tile0+n)*8U+q*2U;
   const std::uint32_t low=__bfloat16_as_ushort(__float2bfloat16_rn(accum[n][0]/divisor));
   const std::uint32_t high=__bfloat16_as_ushort(__float2bfloat16_rn(accum[n][1]/divisor));
   *reinterpret_cast<std::uint32_t*>(y+group*Rows+col)=low|(high<<16U);
  }
 }
#endif
}
}}
extern "C" cudaError_t gem16_12b_down(unsigned candidate,
 const std::uint8_t*a,const std::uint8_t*sa,const std::uint8_t*b,
 const std::uint8_t*sb,std::uint16_t*y,float divisor,cudaStream_t stream){
 using namespace gem16::internal;
 if(candidate==0U)Sm120MatrixProjectionKernel<false,1U,8U,false,std::uint16_t>
  <<<60U,256U,0U,stream>>>(a,sa,b,sb,nullptr,nullptr,nullptr,nullptr,y,3U,3840U,15360U,divisor,1.0F);
 else if(candidate==1U)Sm120D2DownDualTileKernel<<<60U,128U,0U,stream>>>(a,sa,b,sb,y,divisor);
 else return cudaErrorInvalidValue;
 return cudaGetLastError();
}
'''
with a.destination.open('x') as f: f.write(s)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
a.destination.with_suffix('.provenance.json').write_text(json.dumps({
    'source_sha256': sha(source), 'generator_sha256': sha(Path(__file__)),
    'output_sha256': sha(a.destination),
    'scope': 'Only T3/N3840/K15360 Down; two independent N8 tiles per warp, four warps; same60CTA N64 ranges, same existing tiled weights/SFB, K64/MMA/division/BF16 per output; no splitK/repack/new weights'}, indent=2) + '\n')
