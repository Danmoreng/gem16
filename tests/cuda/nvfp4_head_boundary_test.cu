// Standalone guarded native-scale tests, no model/primary throughput claim.
#include "cuda/nvfp4/sm120.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
using Kernel=cudaError_t(*)(unsigned,const std::uint8_t*,const std::uint8_t*,const std::uint8_t*,const std::uint8_t*,void*,std::uint64_t,std::uint64_t,float,std::uint32_t,std::uint32_t,dim3,dim3,cudaStream_t);
extern "C" cudaError_t gem16_lab_launch_nvfp4_head(unsigned,const std::uint8_t*,const std::uint8_t*,const std::uint8_t*,const std::uint8_t*,void*,std::uint64_t,std::uint64_t,float,std::uint32_t,std::uint32_t,dim3,dim3,cudaStream_t);
extern "C" cudaError_t gem16_lab_launch_nvfp4_head_parent(unsigned,const std::uint8_t*,const std::uint8_t*,const std::uint8_t*,const std::uint8_t*,void*,std::uint64_t,std::uint64_t,float,std::uint32_t,std::uint32_t,dim3,dim3,cudaStream_t);
namespace {
void Check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Device{void*p=nullptr;explicit Device(std::size_t n){Check(cudaMalloc(&p,n));}~Device(){if(p)(void)cudaFree(p);}Device(const Device&)=delete;Device&operator=(const Device&)=delete;template<class T>T*As()const{return static_cast<T*>(p);}};
std::uint32_t Next(std::uint32_t&x){x^=x<<13;x^=x>>17;x^=x<<5;return x;}
void Run(unsigned kind,std::uint64_t rows,std::uint64_t width,unsigned pattern){
  constexpr unsigned guard=32U,sentinel=0x7fc12345U;const unsigned tokens=kind==1?3U:1U;
  const std::size_t bytes=kind==1?tokens*rows*4U:(rows/64U)*8U;
  Device a(tokens*width/2U),sa(tokens*width/16U),b(rows*width/2U),sb(rows*width/16U),p(bytes+guard*8U),c(bytes+guard*8U);
  std::vector<std::uint8_t> ah(tokens*width/2U),sah(tokens*width/16U),bh(rows*width/2U),sbh(rows*width/16U);
  std::uint32_t random=1234567U+pattern*97U;for(auto&v:ah)v=static_cast<std::uint8_t>(Next(random));for(auto&v:bh)v=static_cast<std::uint8_t>(Next(random));
  constexpr std::array<std::uint8_t,10> sf{0U,8U,32U,48U,56U,62U,72U,96U,112U,126U};
  for(std::size_t i=0;i<sah.size();++i)sah[i]=sf[(i*7U+i/4U*3U+pattern)%sf.size()];
  for(std::size_t i=0;i<sbh.size();++i)sbh[i]=sf[(i*3U+i/4U+pattern)%sf.size()];
  if(pattern==1U)std::fill(ah.begin(),ah.end(),0U);
  if(pattern==2U)std::fill(ah.begin(),ah.begin()+width/2U,0U);
  if(pattern==3U)std::fill(bh.begin(),bh.end(),0U);
  if(pattern==5U)for(std::size_t i=0;i<sah.size();++i)if((i%3U)==0U)sah[i]=0U;
  for(auto [dst,src]:{std::pair{a.p,&ah},std::pair{sa.p,&sah},std::pair{b.p,&bh},std::pair{sb.p,&sbh}})Check(cudaMemcpy(dst,src->data(),src->size(),cudaMemcpyHostToDevice));
  std::vector<std::uint32_t> expected(bytes/4U+guard*2U,sentinel),actual(expected);
  Check(cudaMemcpy(p.p,expected.data(),expected.size()*4U,cudaMemcpyHostToDevice));Check(cudaMemcpy(c.p,actual.data(),actual.size()*4U,cudaMemcpyHostToDevice));
  const float divisor=pattern==4U?1.e-38F:pattern==5U?0.03125F:pattern==2U?4736.F:7.75F;
  const std::uint32_t m0=pattern==3U?0U:pattern==1U?63U:258883U,m1=pattern==3U?1U:pattern==1U?64U:258882U;
  const dim3 grid(static_cast<unsigned>(kind==1?(rows/8U+3U)/4U:rows/64U)),block(kind==1?128U:256U);
  auto launch=[&](Kernel fn,Device&out){Check(fn(kind,a.As<std::uint8_t>(),sa.As<std::uint8_t>(),b.As<std::uint8_t>(),sb.As<std::uint8_t>(),out.As<std::uint32_t>()+guard,rows,width,divisor,m0,m1,grid,block,nullptr));};
  launch(gem16_lab_launch_nvfp4_head_parent,p);launch(gem16_lab_launch_nvfp4_head,c);Check(cudaDeviceSynchronize());
  Check(cudaMemcpy(expected.data(),p.p,expected.size()*4U,cudaMemcpyDeviceToHost));Check(cudaMemcpy(actual.data(),c.p,actual.size()*4U,cudaMemcpyDeviceToHost));
  if(expected!=actual)throw std::runtime_error("native head output mismatch kind"+std::to_string(kind)+" pattern"+std::to_string(pattern)+" width"+std::to_string(width));
  for(unsigned i=0;i<guard;++i)if(expected[i]!=sentinel||expected[expected.size()-1U-i]!=sentinel)throw std::runtime_error("head output canary");
  if(kind==2U&&(pattern==1U||pattern==3U))for(std::uint32_t partition=0;partition<rows/64U;++partition){
    auto token=partition*64U;while(token==m0||token==m1)++token;
    if(actual[guard+partition*2U+1U]!=token)throw std::runtime_error("head mask/tie partition mismatch");
  }
}
}
int main(){try{unsigned cases=0;for(unsigned pattern=0;pattern<6U;++pattern){
 for(auto shape:{std::array<std::uint64_t,2>{8,64},{24,128},{40,2816},{64,1024}}){Run(1U,shape[0],shape[1],pattern);++cases;}
 for(auto shape:{std::array<std::uint64_t,2>{128,64},{256,128},{384,1024}}){Run(2U,shape[0],shape[1],pattern);++cases;}
 }std::printf("%u native NVFP4 head cases exact: diverse per-row/K scales, zero/tie/mask/overflow, partial warps; guards intact\n",cases);return 0;
 }catch(const std::exception&e){std::fprintf(stderr,"head boundary: %s\n",e.what());return 1;}}
