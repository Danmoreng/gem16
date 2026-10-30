// Offline numerical/memory test with diverse finite FP8 bytes, independent
// per-row/per-channel scales and guarded outputs. No model or throughput claim.
#include "fp8_projection_lab_abi.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>
extern "C" cudaError_t gem16_lab_launch_fp8_qkv_pack3(const std::uint8_t*,const float*,Fp8LabBinding,Fp8LabBinding,Fp8LabBinding,unsigned,std::uint64_t,unsigned,dim3,dim3,cudaStream_t);
extern "C" cudaError_t gem16_lab_launch_fp8_qkv_t3_parent(const std::uint8_t*,const float*,Fp8LabBinding,Fp8LabBinding,Fp8LabBinding,unsigned,std::uint64_t,unsigned,dim3,dim3,cudaStream_t);
extern "C" cudaError_t gem16_lab_launch_fp8_o_pack3(const std::uint8_t*,const float*,Fp8LabBinding,std::uint64_t,dim3,dim3,cudaStream_t);
extern "C" cudaError_t gem16_lab_launch_fp8_o_t3_parent(const std::uint8_t*,const float*,Fp8LabBinding,std::uint64_t,dim3,dim3,cudaStream_t);
namespace {
void Check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Device{
  void* p=nullptr;explicit Device(std::size_t n){Check(cudaMalloc(&p,n));}~Device(){if(p)(void)cudaFree(p);}
  Device(const Device&)=delete;Device& operator=(const Device&)=delete;
  template<class T>T* As()const{return static_cast<T*>(p);}
};
constexpr unsigned kGuard=32U,kSentinel=0x7fc12345U;
void Run(bool qkv,unsigned count,std::array<std::uint64_t,3> rows,std::uint64_t width,unsigned variant,unsigned pattern){
  Device activation(3U*width),scales(3U*sizeof(float));
  std::vector<std::uint8_t> a(3U*width);
  for(std::size_t i=0;i<a.size();++i){auto v=std::uint8_t(i*29U+17U);if((v&127U)==127U)v^=1U;a[i]=pattern==1U?0U:pattern==2U&&i<width?0U:v;}
  Check(cudaMemcpy(activation.p,a.data(),a.size(),cudaMemcpyHostToDevice));
  const std::array<float,3> input_scales{0.0078125F,0.3125F,1.75F};Check(cudaMemcpy(scales.p,input_scales.data(),sizeof(input_scales),cudaMemcpyHostToDevice));
  std::array<std::unique_ptr<Device>,3> weights,weight_scales,parent,child;
  std::array<Fp8LabBinding,3> p{},c{};
  for(unsigned matrix=0;matrix<count;++matrix){
    weights[matrix]=std::make_unique<Device>(rows[matrix]*width);weight_scales[matrix]=std::make_unique<Device>(rows[matrix]*2U);
    const auto elements=3U*rows[matrix]+2U*kGuard;
    parent[matrix]=std::make_unique<Device>(elements*4U);child[matrix]=std::make_unique<Device>(elements*4U);
    std::vector<std::uint8_t> w(rows[matrix]*width);for(std::size_t i=0;i<w.size();++i){auto v=std::uint8_t(i*37U+matrix*13U+3U);if((v&127U)==127U)v^=1U;w[i]=v;}
    std::vector<std::uint16_t> ws(rows[matrix]);constexpr std::array<std::uint16_t,5> sf{0x3b80U,0x3d00U,0x3e80U,0x3f80U,0x4000U};
    for(std::size_t i=0;i<ws.size();++i)ws[i]=sf[(i+matrix)%sf.size()];
    Check(cudaMemcpy(weights[matrix]->p,w.data(),w.size(),cudaMemcpyHostToDevice));Check(cudaMemcpy(weight_scales[matrix]->p,ws.data(),ws.size()*2U,cudaMemcpyHostToDevice));
    std::vector<unsigned> guard(elements,kSentinel);Check(cudaMemcpy(parent[matrix]->p,guard.data(),elements*4U,cudaMemcpyHostToDevice));Check(cudaMemcpy(child[matrix]->p,guard.data(),elements*4U,cudaMemcpyHostToDevice));
    p[matrix]={weights[matrix]->As<std::uint8_t>(),weight_scales[matrix]->As<std::uint16_t>(),parent[matrix]->As<float>()+kGuard,rows[matrix]};c[matrix]=p[matrix];c[matrix].output=child[matrix]->As<float>()+kGuard;
  }
  if(qkv){
    const auto maximum=*std::max_element(rows.begin(),rows.end());const dim3 grid(static_cast<unsigned>(((maximum+7U)/8U+3U)/4U),1U,count),block(128U);
    Check(gem16_lab_launch_fp8_qkv_t3_parent(activation.As<std::uint8_t>(),scales.As<float>(),p[0],p[1],p[2],count,width,0U,grid,block,nullptr));
    Check(gem16_lab_launch_fp8_qkv_pack3(activation.As<std::uint8_t>(),scales.As<float>(),c[0],c[1],c[2],count,width,variant,grid,block,nullptr));
  }else{
    const dim3 grid(static_cast<unsigned>((rows[0]+7U)/8U)),block(64U);
    Check(gem16_lab_launch_fp8_o_t3_parent(activation.As<std::uint8_t>(),scales.As<float>(),p[0],width,grid,block,nullptr));
    Check(gem16_lab_launch_fp8_o_pack3(activation.As<std::uint8_t>(),scales.As<float>(),c[0],width,grid,block,nullptr));
  }
  Check(cudaDeviceSynchronize());
  for(unsigned matrix=0;matrix<count;++matrix){
    const auto elements=3U*rows[matrix]+2U*kGuard;std::vector<unsigned> expected(elements),actual(elements);
    Check(cudaMemcpy(expected.data(),parent[matrix]->p,elements*4U,cudaMemcpyDeviceToHost));Check(cudaMemcpy(actual.data(),child[matrix]->p,elements*4U,cudaMemcpyDeviceToHost));
    if(expected!=actual)throw std::runtime_error("FP8 row/scale/split-K output mismatch");
    for(unsigned j=0;j<kGuard;++j)if(expected[j]!=kSentinel||expected[elements-1U-j]!=kSentinel)throw std::runtime_error("FP8 output canary changed");
  }
}
}
int main(int argc,char** argv){
  if(argc!=2||(std::strcmp(argv[1],"0")!=0&&std::strcmp(argv[1],"1")!=0))return 2;
  const unsigned variant=unsigned(argv[1][0]-'0');
  try{
    unsigned cases=0;
    for(unsigned pattern=0;pattern<3;++pattern){
      for(auto shape:{std::array<std::uint64_t,4>{4096,2048,2048,2816},std::array<std::uint64_t,4>{8192,1024,0,2816},std::array<std::uint64_t,4>{10,18,2,32},std::array<std::uint64_t,4>{2,34,0,64}}){Run(true,shape[2]?3U:2U,{shape[0],shape[1],shape[2]},shape[3],variant,pattern);++cases;}
      for(auto shape:{std::array<std::uint64_t,2>{2816,4096},std::array<std::uint64_t,2>{2816,8192},std::array<std::uint64_t,2>{10,128},std::array<std::uint64_t,2>{34,64}}){Run(false,1U,{shape[0],0,0},shape[1],variant,pattern);++cases;}
    }
    std::printf("%u FP8 R3 cases: diverse bytes/scales, zero rows, tile tails and split-K bit exact; guards intact; variant=%u\n",cases,variant);return 0;
  }catch(const std::exception& e){std::fprintf(stderr,"FP8 boundary test: %s\n",e.what());return 1;}
}
