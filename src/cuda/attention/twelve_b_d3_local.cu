#include "cuda/attention/twelve_b_d3_local.h"
#include "cuda/attention/detail/twelve_b_d3_local_embedded.cuh"
#include "cuda/layer/reference.h"
#include <cuda_runtime.h>
#include <cstddef>
#include <string>
namespace gem16::internal {
namespace {
Status Failure(const char* where,cudaError_t e){return Status(StatusCode::kInternal,std::string(where)+": "+cudaGetErrorString(e));}
__global__ void MergeD3Local4Kernel(const float* partial,const float* lse,float* out){
 __shared__ float weight[4],inverse;
 const unsigned head=blockIdx.x,row=blockIdx.y,lane=threadIdx.x;
 if(lane<4){
  const float value=lse[(row*4+lane)*16+head];float maximum=value;
#pragma unroll
  for(unsigned d=2;d;d>>=1)maximum=fmaxf(maximum,__shfl_down_sync(15U,maximum,d,4));
  maximum=__shfl_sync(15U,maximum,0,4);
  float w=expf(value-maximum);weight[lane]=w;
#pragma unroll
  for(unsigned d=2;d;d>>=1)w+=__shfl_down_sync(15U,w,d,4);
  if(lane==0)inverse=1.0F/w;
 }
 __syncthreads();
#pragma unroll
 for(unsigned item=0;item<2;++item){
  const unsigned dim=lane+item*128;float result=0.0F;
#pragma unroll
  for(unsigned s=0;s<4;++s)result=fmaf(weight[s],partial[((row*4+s)*16+head)*256+dim],result);
  out[(row*16+head)*256+dim]=result*inverse;
 }
}
}
TwelveBD3FastLocalAttention::~TwelveBD3FastLocalAttention(){if(library_)(void)cudaLibraryUnload(library_);}
Status TwelveBD3FastLocalAttention::Initialize(){
 if(library_)return Status::Ok();
 cudaDeviceProp device{};auto e=cudaGetDeviceProperties(&device,0);
 if(e!=cudaSuccess)return Failure("D3 local device",e);
 if(device.major!=12||device.minor!=0)return Status(StatusCode::kUnsupported,"experimental D3 local attention requires SM120");
 e=cudaLibraryLoadData(&library_,d3_local::kCubin,nullptr,nullptr,0,nullptr,nullptr,0);
 if(e!=cudaSuccess)return Failure("load offline D3 local cubin",e);
 e=cudaLibraryGetKernel(&kernel_,library_,d3_local::kKernel);
 if(e!=cudaSuccess)return Failure("find offline D3 local kernel",e);
 cudaFuncAttributes attr{};e=cudaFuncGetAttributes(&attr,reinterpret_cast<const void*>(kernel_));
 if(e!=cudaSuccess)return Failure("prepare D3 local resources",e);
 e=cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel_),cudaFuncAttributeMaxDynamicSharedMemorySize,d3_local::kShared);
 return e==cudaSuccess?Status::Ok():Failure("D3 local shared memory",e);
}
Status TwelveBD3FastLocalAttention::Launch(const float*q,const std::uint8_t*k,
 const std::uint8_t*v,const std::uint8_t*sk,const std::uint8_t*sv,
 const std::uint16_t*ks,const std::uint16_t*vs,float*workspace,
 std::uint64_t elements,float*out,const DecodeControl*control,cudaStream_t stream)const{
 static_assert(offsetof(DecodeControl,position)==8);
 if(!kernel_||!q||!k||!v||!sk||!sv||!ks||!vs||!workspace||!out||!control||elements<WorkspaceElements())
  return Status(StatusCode::kInvalidArgument,"experimental D3 local attention invalid resources");
 float*lse=workspace+4U*4U*16U*256U;void*no_scratch=nullptr;
 void*args[]={&q,&k,&v,&sk,&sv,&ks,&vs,&control,&workspace,&lse,&no_scratch,&no_scratch};
 auto e=cudaLaunchKernel(reinterpret_cast<const void*>(kernel_),dim3(8,4),dim3(128),args,d3_local::kShared,stream);
 if(e!=cudaSuccess)return Failure("launch D3 local FP8",e);
 MergeD3Local4Kernel<<<dim3(16,4),128,0,stream>>>(workspace,lse,out);
 e=cudaGetLastError();return e==cudaSuccess?Status::Ok():Failure("merge D3 local FP8",e);
}
}
