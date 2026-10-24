// Explicit offline observer for actual 26B M1/T3 FP8 QKV/O. Immutable resident
// weight pointers are borrowed; every mutable activation/output is copied.
// Do not preload into a primary throughput run.
#include "fp8_projection_lab_abi.h"
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <dlfcn.h>
#include <array>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
using Launch=cudaError_t (*)(const void*,dim3,dim3,void**,std::size_t,cudaStream_t);
using Begin=cudaError_t (*)(cudaStream_t,cudaStreamCaptureMode);
Launch real_launch=nullptr; Begin real_begin=nullptr;
bool enabled=false,ready=false,injecting=false,t3=false;
unsigned qkv_calls=0;
std::array<const void*,32> debug_functions{};unsigned debug_function_count=0;
std::byte* arena=nullptr;
constexpr std::size_t kStride=384U*1024U,kArenaBytes=32U*kStride;
constexpr std::array<unsigned,4> kLayers{0U,5U,15U,29U};
struct Record {
  const void* function=nullptr; dim3 grid{},block{}; std::size_t shared=0;
  std::array<Fp8LabBinding,3> bindings{}; std::uint64_t k=0,tokens=0;
  unsigned count=0,kind=0; bool valid=false;
};
std::array<Record,32> records{}; std::array<bool,3> saved{};
struct Buffers {std::uint8_t* input;float *scales,*expected,*temporary;};
Buffers Buffer(unsigned i) {
  auto* p=arena+i*kStride;
  return {reinterpret_cast<std::uint8_t*>(p),reinterpret_cast<float*>(p+24576U),
      reinterpret_cast<float*>(p+24832U),reinterpret_cast<float*>(p+24832U+128U*1024U)};
}
void Check(cudaError_t e) {if(e!=cudaSuccess){std::fprintf(stderr,"FP8 lab: %s\n",cudaGetErrorString(e));std::abort();}}
template<class T>T Arg(void** args,unsigned i){T value{};std::memcpy(&value,args[i],sizeof(value));return value;}
__global__ void DiagnosticFp8CopyKernel(const std::byte* from,std::byte* to,std::size_t n){
  for(std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=std::size_t(gridDim.x)*blockDim.x)to[i]=from[i];
}
void Copy(const void* from,void* to,std::size_t n,cudaStream_t stream){
  DiagnosticFp8CopyKernel<<<static_cast<unsigned>((n+255U)/256U),256U,0U,stream>>>(
      static_cast<const std::byte*>(from),static_cast<std::byte*>(to),n);Check(cudaGetLastError());
}
int Slot(unsigned layer){for(unsigned i=0;i<kLayers.size();++i)if(layer==kLayers[i])return static_cast<int>(i);return -1;}
__attribute__((constructor))void Initialize(){
  real_launch=reinterpret_cast<Launch>(dlsym(RTLD_NEXT,"cudaLaunchKernel"));
  real_begin=reinterpret_cast<Begin>(dlsym(RTLD_NEXT,"cudaStreamBeginCapture"));
  const char* e=std::getenv("GEM16_FP8_OPERATOR_LAB");enabled=e&&std::strcmp(e,"1")==0;
  const char* m=std::getenv("GEM16_FP8_LAB_MODE");t3=m&&std::strcmp(m,"t3")==0;
}
void Export(const std::filesystem::path& path,const void* from,std::size_t n){
  std::vector<std::byte> host(n);Check(cudaMemcpy(host.data(),from,n,cudaMemcpyDeviceToHost));
  if(std::filesystem::exists(path))throw std::runtime_error("FP8 export exists");
  std::ofstream f(path,std::ios::binary);f.write(reinterpret_cast<const char*>(host.data()),n);
  if(!f)throw std::runtime_error("FP8 export failed");
}
template<class F>cudaGraphExec_t Graph(cudaStream_t stream,F launch){
  cudaGraph_t graph{};Check(real_begin(stream,cudaStreamCaptureModeThreadLocal));
  for(unsigned i=0;i<16;++i)launch();Check(cudaStreamEndCapture(stream,&graph));
  cudaGraphExec_t exec{};Check(cudaGraphInstantiate(&exec,graph,0ULL));Check(cudaGraphDestroy(graph));return exec;
}
}
extern "C" cudaError_t cudaStreamBeginCapture(cudaStream_t stream,cudaStreamCaptureMode mode){
  if(enabled&&!ready){injecting=true;Check(cudaMalloc(&arena,kArenaBytes));Check(cudaMemset(arena,0,kArenaBytes));ready=true;injecting=false;}
  return real_begin(stream,mode);
}
extern "C" cudaError_t cudaLaunchKernel(const void* function,dim3 grid,dim3 block,void** args,std::size_t shared,cudaStream_t stream){
  const auto result=real_launch(function,grid,block,args,shared,stream);
  if(!enabled||!ready||injecting||result!=cudaSuccess)return result;
  const char* name=nullptr;Check(cudaFuncGetName(&name,function));if(!name)return result;
  if(std::getenv("GEM16_FP8_LAB_DEBUG")&&std::strstr(name,"Sm120")){
    bool seen=false;for(unsigned i=0;i<debug_function_count;++i)seen=seen||debug_functions[i]==function;
    if(!seen&&debug_function_count<debug_functions.size()){
      debug_functions[debug_function_count++]=function;std::fprintf(stderr,"FP8 observed: %s\n",name);
    }
  }
  Record r{};bool qkv=false,output=false;
  if(!t3&&std::strstr(name,"Sm120LocalGroupedQkvProjectionKernel")){
    r.kind=0;r.k=Arg<std::uint64_t>(args,5);r.count=3;r.tokens=1;qkv=true;
  }else if(!t3&&std::strstr(name,"Sm120DirectProjectionKernel")&&std::strstr(name,"Fp8MatrixBinding")&&
      (Arg<unsigned>(args,5)==2U||Arg<unsigned>(args,5)==3U)&&Arg<std::uint64_t>(args,6)==1U){
    r.kind=1;r.k=Arg<std::uint64_t>(args,7);r.count=Arg<unsigned>(args,5);r.tokens=1;qkv=true;
  }else if(t3&&(std::strstr(name,"Sm120DirectProjectionFixedBatchKernel")||std::strstr(name,"Sm120DirectProjectionPackedT3Kernel"))&&
      (Arg<unsigned>(args,5)==2U||Arg<unsigned>(args,5)==3U)){
    r.kind=2;r.k=Arg<std::uint64_t>(args,6);r.count=Arg<unsigned>(args,5);r.tokens=3;qkv=true;
  }else if(std::strstr(name,t3?"Sm120DirectProjectionSplitK2FixedT3Kernel":"Sm120DirectProjectionSplitK2Kernel")||
      (t3&&std::strstr(name,"Sm120DirectProjectionSplitK2PackedT3Kernel"))){
    r.kind=t3?4:3;r.k=Arg<std::uint64_t>(args,6);r.tokens=t3?3:1;r.count=1;
    const auto rows=Arg<std::uint64_t>(args,5);
    output=rows==2816U&&(r.k==4096U||r.k==8192U);
    r.bindings[0]={Arg<const std::uint8_t*>(args,2),Arg<const std::uint16_t*>(args,3),Arg<float*>(args,4),rows};
  }
  if(qkv){
    if(r.k!=2816U)return result;
    for(unsigned i=0;i<3;++i)r.bindings[i]=Arg<Fp8LabBinding>(args,2U+i);
    const bool local=r.bindings[0].rows==4096U&&r.bindings[1].rows==2048U&&r.bindings[2].rows==2048U;
    const bool global=r.bindings[0].rows==8192U&&r.bindings[1].rows==1024U&&r.count==2U&&r.bindings[2].rows==0U;
    if(!local&&!global)std::abort();
  }
  if(!qkv&&!output)return result;
  const unsigned layer=qkv?qkv_calls++%30U:(qkv_calls+29U)%30U;
  const int slot=Slot(layer);if(slot<0)return result;
  injecting=true;const unsigned index=2U*static_cast<unsigned>(slot)+(output?1U:0U);const auto b=Buffer(index);
  r.function=function;r.grid=grid;r.block=block;r.shared=shared;r.valid=true;
  Copy(Arg<const void*>(args,0),b.input,r.tokens*r.k,stream);Copy(Arg<const void*>(args,1),b.scales,r.tokens*4U,stream);
  std::size_t offset=0;
  for(unsigned i=0;i<r.count;++i){
    const auto bytes=r.tokens*r.bindings[i].rows*4U;if(offset+bytes>128U*1024U)std::abort();
    Copy(r.bindings[i].output,reinterpret_cast<std::byte*>(b.expected)+offset,bytes,stream);offset+=bytes;
  }
  records[index]=r;injecting=false;return result;
}
extern "C" cudaError_t __cudaLaunchKernel(cudaKernel_t k,dim3 g,dim3 b,void** a,std::size_t s,cudaStream_t t){
  return cudaLaunchKernel(reinterpret_cast<const void*>(k),g,b,a,s,t);
}
extern "C" int gem16_lab_save_phase(unsigned phase){
  if(!enabled||!ready||phase>=3U||saved[phase])return 1;injecting=true;
  for(unsigned i=0;i<8;++i){if(!records[i].valid){std::fprintf(stderr,"missing FP8 record %u, qkv_calls=%u\n",i,qkv_calls);throw std::runtime_error("missing FP8 record");}
    const unsigned destination=8U+phase*8U+i;Copy(arena+i*kStride,arena+destination*kStride,kStride,nullptr);records[destination]=records[i];}
  Check(cudaDeviceSynchronize());saved[phase]=true;injecting=false;return 0;
}
extern "C" int gem16_lab_finish(const char* directory){
  if(!enabled||!ready||!directory)return 1;
  try{
    injecting=true;const std::filesystem::path root(directory);if(!std::filesystem::create_directory(root))throw std::runtime_error("FP8 directory exists");
    const char* choice=std::getenv("GEM16_FP8_LAB_CANDIDATE");const bool test=choice&&std::strcmp(choice,"none")!=0;
    const unsigned variant=choice&&std::strcmp(choice,"pad")==0?1U:0U;
    const auto qkv_candidate=reinterpret_cast<Fp8LabQkvCandidate>(dlsym(RTLD_DEFAULT,t3?"gem16_lab_launch_fp8_qkv_pack3":"gem16_lab_launch_fp8_qkv_m1_pipeline"));
    const auto o_candidate=reinterpret_cast<Fp8LabOCandidate>(dlsym(RTLD_DEFAULT,t3?"gem16_lab_launch_fp8_o_pack3":"gem16_lab_launch_fp8_o_m1_pipeline"));
    if(test&&(!qkv_candidate||!o_candidate))throw std::runtime_error("FP8 candidate symbol absent");
    cudaStream_t stream{};Check(cudaStreamCreate(&stream));
    std::ofstream summary(root/"captures.json");summary.precision(9);
    summary<<"{\"diagnostic_only\":true,\"weights_copied\":false,\"mutable_inputs_snapshotted\":true,\"extra_device_bytes\":"<<kArenaBytes<<",\"cases\":[";
    unsigned count=0;
    for(unsigned phase=0;phase<3;++phase)for(unsigned i=0;i<8;++i){
      if(!saved[phase])throw std::runtime_error("missing FP8 phase");
      const auto& r=records[8U+phase*8U+i];const auto b=Buffer(8U+phase*8U+i);
      const auto stem="phase"+std::to_string(phase)+"-layer"+std::to_string(kLayers[i/2U])+(i%2U?"-o":"-qkv");
      auto bindings=r.bindings;std::size_t output_bytes=0;
      for(unsigned matrix=0;matrix<r.count;++matrix){bindings[matrix].output=b.temporary+output_bytes/4U;output_bytes+=r.tokens*bindings[matrix].rows*4U;}
      const auto* input=b.input;const auto* scales=b.scales;unsigned bindings_count=r.count;auto k=r.k,tokens=r.tokens;
      auto* output=bindings[0].output;const auto* weight=bindings[0].weight;const auto* ws=bindings[0].weight_scales;auto rows=bindings[0].rows;
      void* local[]{&input,&scales,&bindings[0],&bindings[1],&bindings[2],&k};
      void* generic[]{&input,&scales,&bindings[0],&bindings[1],&bindings[2],&bindings_count,&tokens,&k};
      void* fixed[]{&input,&scales,&bindings[0],&bindings[1],&bindings[2],&bindings_count,&k};
      void* out[]{&input,&scales,&weight,&ws,&output,&rows,&k};
      void** args=r.kind==0?local:r.kind==1?generic:r.kind==2?fixed:out;
      auto parent=[&]{Check(real_launch(r.function,r.grid,r.block,args,r.shared,stream));};
      auto candidate=[&]{if(r.kind<=2)Check(qkv_candidate(input,scales,bindings[0],bindings[1],bindings[2],r.count,k,t3?variant:unsigned(r.kind==0),r.grid,r.block,stream));
        else Check(o_candidate(input,scales,bindings[0],k,r.grid,r.block,stream));};
      const bool compare_naive=test&&t3&&r.kind==2U&&variant==1U;
      auto naive=[&]{Check(qkv_candidate(input,scales,bindings[0],bindings[1],bindings[2],r.count,k,0U,r.grid,r.block,stream));};
      std::vector<std::byte> expected(output_bytes),actual(output_bytes);Check(cudaMemcpy(expected.data(),b.expected,output_bytes,cudaMemcpyDeviceToHost));
      auto equal=[&]{Check(cudaStreamSynchronize(stream));Check(cudaMemcpy(actual.data(),b.temporary,output_bytes,cudaMemcpyDeviceToHost));return actual==expected;};
      for(unsigned warm=0;warm<3;++warm)parent();if(!equal())throw std::runtime_error("FP8 actual-parent replay mismatch: "+stem);
      const char* pc=std::getenv("GEM16_LAB_PROFILE_CASE");const char* pk=std::getenv("GEM16_LAB_PROFILE_KIND");const bool profile=pc&&stem==pc;
      if(profile&&pk&&std::strcmp(pk,"parent")==0){Check(cudaProfilerStart());parent();Check(cudaStreamSynchronize(stream));Check(cudaProfilerStop());}
      bool exact=false;
      if(test){for(unsigned warm=0;warm<3;++warm)candidate();exact=equal();if(!exact)throw std::runtime_error("FP8 candidate mismatch: "+stem);
        if(profile&&pk&&std::strcmp(pk,"candidate")==0){Check(cudaProfilerStart());candidate();Check(cudaStreamSynchronize(stream));Check(cudaProfilerStop());}}
      if(compare_naive){naive();if(!equal())throw std::runtime_error("naive packed FP8 mismatch");}
      const auto pg=Graph(stream,parent),cg=test?Graph(stream,candidate):nullptr,ng=compare_naive?Graph(stream,naive):nullptr;
      cudaEvent_t begin{},end{};Check(cudaEventCreate(&begin));Check(cudaEventCreate(&end));
      auto measure=[&](cudaGraphExec_t graph){Check(cudaEventRecord(begin,stream));Check(cudaGraphLaunch(graph,stream));Check(cudaEventRecord(end,stream));Check(cudaEventSynchronize(end));float ms=0;Check(cudaEventElapsedTime(&ms,begin,end));return ms*1000.0F/16.0F;};
      for(unsigned warm=0;warm<3;++warm){(void)measure(pg);if(cg)(void)measure(cg);if(ng)(void)measure(ng);}
      std::array<float,10> p{},c{},n{};
      for(unsigned j=0;j<10;++j){
        if(ng){for(unsigned step=0;step<3;++step){const auto index=(j+step)%3U;
          if(index==0U)p[j]=measure(pg);else if(index==1U)c[j]=measure(cg);else n[j]=measure(ng);}}
        else{if(cg&&j%2)c[j]=measure(cg);p[j]=measure(pg);if(cg&&j%2==0)c[j]=measure(cg);}
      }
      if(cg){Check(cudaGraphLaunch(cg,stream));if(!equal())throw std::runtime_error("post-timing FP8 mismatch");}
      if(ng){Check(cudaGraphLaunch(ng,stream));if(!equal())throw std::runtime_error("post-timing naive FP8 mismatch");}
      Export(root/(stem+"-activation.bin"),b.input,r.tokens*k);Export(root/(stem+"-scales.bin"),b.scales,r.tokens*4U);Export(root/(stem+"-expected.bin"),b.expected,output_bytes);
      cudaFuncAttributes attr{};Check(cudaFuncGetAttributes(&attr,r.function));
      summary<<(count++?",":"")<<"{\"stem\":\""<<stem<<"\",\"kind\":"<<r.kind<<",\"tokens\":"<<tokens<<",\"k\":"<<k<<",\"binding_count\":"<<r.count<<",\"rows\":[";
      for(unsigned j=0;j<r.count;++j)summary<<(j?",":"")<<r.bindings[j].rows;
      summary<<"],\"grid\":["<<r.grid.x<<","<<r.grid.y<<","<<r.grid.z<<"],\"block\":["<<r.block.x<<","<<r.block.y<<","<<r.block.z<<"],\"dynamic_shared_bytes\":"<<r.shared
        <<",\"parent_registers\":"<<attr.numRegs<<",\"parent_static_shared_bytes\":"<<attr.sharedSizeBytes<<",\"parent_bit_exact\":true,\"candidate_tested\":"<<(test?"true":"false")<<",\"candidate_bit_exact\":"<<(exact?"true":"false")<<",\"parent_us_samples\":[";
      for(unsigned j=0;j<10;++j)summary<<(j?",":"")<<p[j];summary<<"],\"candidate_us_samples\":[";
      if(cg)for(unsigned j=0;j<10;++j)summary<<(j?",":"")<<c[j];summary<<"],\"naive_packed_us_samples\":[";
      if(ng)for(unsigned j=0;j<10;++j)summary<<(j?",":"")<<n[j];summary<<"],\"naive_packed_bit_exact\":"<<(ng?"true":"false")<<"}";
      Check(cudaGraphExecDestroy(pg));if(cg)Check(cudaGraphExecDestroy(cg));if(ng)Check(cudaGraphExecDestroy(ng));Check(cudaEventDestroy(begin));Check(cudaEventDestroy(end));
    }
    summary<<"]}\n";if(!summary)throw std::runtime_error("FP8 summary write failed");Check(cudaStreamDestroy(stream));injecting=false;return 0;
  }catch(const std::exception& e){std::fprintf(stderr,"FP8 lab: %s\n",e.what());return 1;}
}
