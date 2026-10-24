// Offline test over immutable diagnostic snapshots. No model/runtime preload.
#include "cuda/layer/reference.h"
#include <cuda_runtime.h>
#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
using Control=gem16::internal::DecodeControl;
#define GLOBAL_ARGS const float*, const std::uint8_t*, const std::uint8_t*, const std::uint16_t*, const std::uint16_t*, float*, float*, const Control*, int, dim3, dim3, cudaStream_t
#define LOCAL_ARGS const float*, const std::uint8_t*, const std::uint8_t*, const std::uint8_t*, const std::uint8_t*, const std::uint16_t*, const std::uint16_t*, float*, float*, const Control*, int, dim3, dim3, cudaStream_t
extern "C" cudaError_t gem16_lab_launch_t3_global_parent(GLOBAL_ARGS);
extern "C" cudaError_t gem16_lab_launch_t3_global_warp_norm(GLOBAL_ARGS);
extern "C" cudaError_t gem16_lab_launch_t3_local_parent(LOCAL_ARGS);
extern "C" cudaError_t gem16_lab_launch_t3_local_warp_norm(LOCAL_ARGS);
#undef GLOBAL_ARGS
#undef LOCAL_ARGS
namespace {
void Check(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct Device {
  void* pointer=nullptr; explicit Device(std::size_t bytes) { Check(cudaMalloc(&pointer,bytes)); Check(cudaMemset(pointer,0,bytes)); }
  ~Device() { if(pointer) (void)cudaFree(pointer); }
  Device(const Device&)=delete; Device& operator=(const Device&)=delete;
  template<class T> T* As() const { return static_cast<T*>(pointer); }
};
std::vector<std::byte> Read(const std::filesystem::path& p,std::size_t bytes) {
  if(!std::filesystem::is_regular_file(std::filesystem::symlink_status(p)) || std::filesystem::file_size(p)!=bytes)
    throw std::runtime_error("snapshot extent/type mismatch: "+p.string());
  std::vector<std::byte> result(bytes); std::ifstream f(p,std::ios::binary);
  f.read(reinterpret_cast<char*>(result.data()),static_cast<std::streamsize>(bytes));
  if(!f || f.peek()!=std::char_traits<char>::eof()) throw std::runtime_error("snapshot changed/read failed");
  return result;
}
void Upload(const std::filesystem::path& p,Device& d,std::size_t n) {
  auto bytes=Read(p,n); Check(cudaMemcpy(d.pointer,bytes.data(),n,cudaMemcpyHostToDevice));
}
__global__ void CompareSnapshotAttentionKernel(const float* p,const float* q,const float* pl,const float* ql,
    unsigned active,unsigned splits,unsigned dim,unsigned* bad) {
  const std::size_t stride=std::size_t(active)*16U*dim;
  for(std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<3U*stride;i+=std::size_t(gridDim.x)*blockDim.x) {
    const auto index=(i/stride)*splits*16U*dim+i%stride;
    if(__float_as_uint(p[index])!=__float_as_uint(q[index])) {
      if(atomicAdd(bad,1U)==0U) printf("partial mismatch index=%llu expected=%a (%08x) actual=%a (%08x)\n",
          static_cast<unsigned long long>(index),p[index],__float_as_uint(p[index]),q[index],__float_as_uint(q[index]));
    }
  }
  for(unsigned i=blockIdx.x*blockDim.x+threadIdx.x;i<3U*active*16U;i+=gridDim.x*blockDim.x) {
    const auto index=(i/(active*16U))*splits*16U+i%(active*16U);
    if(__float_as_uint(pl[index])!=__float_as_uint(ql[index])) {
      if(atomicAdd(bad,1U)==0U) printf("LSE mismatch index=%u expected=%a (%08x) actual=%a (%08x)\n",
          index,pl[index],__float_as_uint(pl[index]),ql[index],__float_as_uint(ql[index]));
    }
  }
}
unsigned Run(const std::filesystem::path& root,bool global) {
  const unsigned dim=global?512U:256U,splits=global?137U:8U;
  const std::size_t cache=global?17519U*2U*512U:1024U*8U*256U;
  const std::size_t partial_bytes=3U*splits*16U*dim*4U,lse_bytes=3U*splits*16U*4U;
  Device query(3U*16U*dim*4U),keys(cache),values(cache),sk(3U*8U*256U),sv(3U*8U*256U),ks(2U),vs(2U),controls(3U*sizeof(Control));
  Device parent(partial_bytes),child(partial_bytes),parent_lse(lse_bytes),child_lse(lse_bytes),bad(sizeof(unsigned));
  const dim3 grid(global?splits*2U:splits*8U),block(256U);
  auto launch=[&](bool candidate) {
    if(global) {
      auto f=candidate?gem16_lab_launch_t3_global_warp_norm:gem16_lab_launch_t3_global_parent;
      Check(f(query.As<float>(),keys.As<std::uint8_t>(),values.As<std::uint8_t>(),ks.As<std::uint16_t>(),vs.As<std::uint16_t>(),
        (candidate?child:parent).As<float>(),(candidate?child_lse:parent_lse).As<float>(),controls.As<Control>(),splits,grid,block,nullptr));
    } else {
      auto f=candidate?gem16_lab_launch_t3_local_warp_norm:gem16_lab_launch_t3_local_parent;
      Check(f(query.As<float>(),keys.As<std::uint8_t>(),values.As<std::uint8_t>(),sk.As<std::uint8_t>(),sv.As<std::uint8_t>(),ks.As<std::uint16_t>(),vs.As<std::uint16_t>(),
        (candidate?child:parent).As<float>(),(candidate?child_lse:parent_lse).As<float>(),controls.As<Control>(),splits,grid,block,nullptr));
    }
  };
  auto compare=[&](unsigned position,const char* stage) {
    Check(cudaMemset(bad.pointer,0,sizeof(unsigned)));
    CompareSnapshotAttentionKernel<<<256U,256U>>>(parent.As<float>(),child.As<float>(),parent_lse.As<float>(),child_lse.As<float>(),
      global?(position+3U+127U)/128U:8U,splits,dim,bad.As<unsigned>()); Check(cudaGetLastError());
    unsigned errors=0; Check(cudaMemcpy(&errors,bad.pointer,sizeof(errors),cudaMemcpyDeviceToHost));
    if(errors) throw std::runtime_error(std::string(global?"global ":"local ")+stage+": "+std::to_string(errors)+" partial/LSE mismatches at "+std::to_string(position));
  };
  unsigned count=0;
  for(unsigned phase=0;phase<3;++phase) {
    const auto stem=std::string(global?"global":"local")+"-phase"+std::to_string(phase);
    auto p=[&](const char* suffix){return root/(stem+"-"+suffix+".bin");};
    Upload(p("query"),query,3U*16U*dim*4U); Upload(p("keys"),keys,cache); Upload(p("values"),values,cache);
    Upload(p("key-scale"),ks,2U); Upload(p("value-scale"),vs,2U);
    if(!global) {Upload(p("spec-keys"),sk,3U*8U*256U); Upload(p("spec-values"),sv,3U*8U*256U);}
    Upload(p("controls"),controls,3U*sizeof(Control));
    std::array<Control,3> state{}; Check(cudaMemcpy(state.data(),controls.pointer,sizeof(state),cudaMemcpyDeviceToHost));
    const auto position=std::array<unsigned,3>{16384U,16951U,17512U}[phase];
    for(unsigned row=0;row<3;++row) if(state[row].position!=position+row) throw std::runtime_error("unexpected control rows");
    // Both independently compiled original and candidate must first reproduce
    // the actual production capture. Only active prefixes are ever compared.
    Upload(p("partial"),parent,partial_bytes); Upload(p("lse"),parent_lse,lse_bytes);
    launch(true); compare(position,"candidate vs actual capture");
    Check(cudaMemcpy(child.pointer,parent.pointer,partial_bytes,cudaMemcpyDeviceToDevice));
    Check(cudaMemcpy(child_lse.pointer,parent_lse.pointer,lse_bytes,cudaMemcpyDeviceToDevice));
    launch(false); compare(position,"original vs actual capture"); ++count;
    if(phase!=0U) continue;
    const auto bases=global?std::array<unsigned,3>{0U,16384U,17516U}:std::array<unsigned,3>{0U,896U,1024U};
    for(unsigned base:bases) for(unsigned tail=0;tail<(global&&base==17516U?1U:128U);++tail) {
      auto changed=state; for(unsigned row=0;row<3;++row) changed[row].position=base+tail+row;
      Check(cudaMemcpy(controls.pointer,changed.data(),sizeof(changed),cudaMemcpyHostToDevice));
      launch(false); launch(true); compare(base+tail,"boundary candidate vs original"); ++count;
    }
  }
  std::printf("%s: %u cases exact (3 real, remaining counterfactual positions on captured bytes)\n",global?"global":"local",count);
  return count;
}
}
int main(int argc,char** argv) {
  if(argc!=2) return 2;
  try {const auto local=Run(argv[1],false),global=Run(argv[1],true);
    if(local!=387U || global!=260U) throw std::runtime_error("case count mismatch");
    std::puts("647 cases: original/candidate partials and LSE bit exact");return 0;
  } catch(const std::exception& e) {std::fprintf(stderr,"attention boundary test: %s\n",e.what());return 1;}
}
