// Offline, explicitly enabled LD_PRELOAD diagnostic. All mutable attention
// inputs, including speculative/ring KV, are snapshotted before later writes.
// Never attach this observer to a primary throughput run.
#include "cuda/layer/reference.h"
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
using Control = gem16::internal::DecodeControl;
using Launch = cudaError_t (*)(const void*, dim3, dim3, void**, std::size_t, cudaStream_t);
using Begin = cudaError_t (*)(cudaStream_t, cudaStreamCaptureMode);
using Candidate = cudaError_t (*)(const float*, const std::uint8_t*, const std::uint8_t*,
    const std::uint16_t*, const std::uint16_t*, float*, float*, const Control*, int,
    dim3, dim3, cudaStream_t);
using LocalCandidate = cudaError_t (*)(const float*, const std::uint8_t*, const std::uint8_t*,
    const std::uint8_t*, const std::uint8_t*, const std::uint16_t*, const std::uint16_t*,
    float*, float*, const Control*, int, dim3, dim3, cudaStream_t);
Launch real_launch = nullptr;
Begin real_begin = nullptr;
bool enabled = false, ready = false, injecting = false;
std::byte* arena = nullptr;
constexpr std::size_t MiB = 1024U * 1024U;
constexpr std::size_t kPairStride = 72U * MiB;
constexpr std::size_t kArenaBytes = 4U * kPairStride + 16U * MiB;
constexpr std::size_t kGlobalCacheBytes = 17519U * 2U * 512U;
constexpr std::size_t kLocalCacheBytes = 1024U * 8U * 256U;
struct Record { const void* function = nullptr; dim3 grid{}, block{}; int splits = 0; bool valid = false; };
std::array<Record, 8> records{};
std::array<bool, 3> saved{};
unsigned local_calls = 0U, global_calls = 0U;
struct Buffers { float *query, *partial, *lse; Control* controls;
  std::uint8_t *keys, *values, *spec_keys, *spec_values; std::uint16_t *ks, *vs; };
Buffers Buffer(unsigned pair, bool global) {
  auto* start = arena + pair * kPairStride + (global ? 0U : 64U * MiB);
  auto* p = start;
  auto reserve = [&](std::size_t bytes) { auto* result = p; p += (bytes + 255U) & ~std::size_t(255U); return result; };
  const std::size_t dim = global ? 512U : 256U, splits = global ? 137U : 8U;
  const auto cache = global ? kGlobalCacheBytes : kLocalCacheBytes;
  Buffers b{};
  b.query = reinterpret_cast<float*>(reserve(3U * 16U * dim * 4U));
  b.controls = reinterpret_cast<Control*>(reserve(3U * sizeof(Control)));
  b.ks = reinterpret_cast<std::uint16_t*>(reserve(2U)); b.vs = reinterpret_cast<std::uint16_t*>(reserve(2U));
  b.keys = reinterpret_cast<std::uint8_t*>(reserve(cache));
  b.values = reinterpret_cast<std::uint8_t*>(reserve(cache));
  b.spec_keys = reinterpret_cast<std::uint8_t*>(reserve(global ? 0U : 3U * 8U * 256U));
  b.spec_values = reinterpret_cast<std::uint8_t*>(reserve(global ? 0U : 3U * 8U * 256U));
  b.partial = reinterpret_cast<float*>(reserve(3U * splits * 16U * dim * 4U));
  b.lse = reinterpret_cast<float*>(reserve(3U * splits * 16U * 4U));
  if (p > start + (global ? 64U : 8U) * MiB) std::abort();
  return b;
}
void Check(cudaError_t e) { if (e != cudaSuccess) { std::fprintf(stderr,"attention lab: %s\n",cudaGetErrorString(e)); std::abort(); } }
template<class T> T Arg(void** args, unsigned i) { T v{}; std::memcpy(&v,args[i],sizeof(v)); return v; }
__global__ void DiagnosticAttentionCopyKernel(const std::byte* from, std::byte* to, std::size_t n) {
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += std::size_t(gridDim.x) * blockDim.x) to[i] = from[i];
}
void Copy(const void* from, void* to, std::size_t n, cudaStream_t stream) {
  DiagnosticAttentionCopyKernel<<<static_cast<unsigned>((n+255U)/256U),256U,0U,stream>>>(
      static_cast<const std::byte*>(from),static_cast<std::byte*>(to),n); Check(cudaGetLastError());
}
__global__ void CompareAttentionBoundaryKernel(const float* expected, const float* actual,
    const float* expected_lse, const float* actual_lse, unsigned active, unsigned splits,
    unsigned dimension, unsigned* mismatches) {
  const std::size_t row_elements=std::size_t(active)*16U*dimension;
  for(std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<3U*row_elements;
      i+=std::size_t(gridDim.x)*blockDim.x) {
    const auto index=(i/row_elements)*splits*16U*dimension+i%row_elements;
    if(__float_as_uint(expected[index])!=__float_as_uint(actual[index])) atomicAdd(mismatches,1U);
  }
  for(unsigned i=blockIdx.x*blockDim.x+threadIdx.x;i<3U*active*16U;i+=gridDim.x*blockDim.x) {
    const auto index=(i/(active*16U))*splits*16U+i%(active*16U);
    if(__float_as_uint(expected_lse[index])!=__float_as_uint(actual_lse[index])) atomicAdd(mismatches,1U);
  }
}
__attribute__((constructor)) void Initialize() {
  real_launch = reinterpret_cast<Launch>(dlsym(RTLD_NEXT,"cudaLaunchKernel"));
  real_begin = reinterpret_cast<Begin>(dlsym(RTLD_NEXT,"cudaStreamBeginCapture"));
  const char* setting = std::getenv("GEM16_ATTENTION_OPERATOR_LAB");
  enabled = setting && std::strcmp(setting,"1") == 0;
}
void Export(const std::filesystem::path& path, const void* ptr, std::size_t n) {
  std::vector<std::byte> host(n); Check(cudaMemcpy(host.data(),ptr,n,cudaMemcpyDeviceToHost));
  if (std::filesystem::exists(path)) throw std::runtime_error("attention export exists");
  std::ofstream f(path,std::ios::binary); f.write(reinterpret_cast<const char*>(host.data()),n);
  if (!f) throw std::runtime_error("attention export failed");
}
template<class F> cudaGraphExec_t Graph(cudaStream_t stream, F launch) {
  cudaGraph_t graph{}; Check(real_begin(stream,cudaStreamCaptureModeThreadLocal));
  for (unsigned i=0;i<16;++i) launch();
  Check(cudaStreamEndCapture(stream,&graph)); cudaGraphExec_t result{};
  Check(cudaGraphInstantiate(&result,graph,0ULL)); Check(cudaGraphDestroy(graph)); return result;
}
}
extern "C" cudaError_t cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode) {
  if (enabled && !ready) { injecting=true; Check(cudaMalloc(&arena,kArenaBytes));
    Check(cudaMemset(arena,0,kArenaBytes)); ready=true; injecting=false; }
  return real_begin(stream,mode);
}
extern "C" cudaError_t cudaLaunchKernel(const void* function, dim3 grid, dim3 block,
    void** args, std::size_t shared, cudaStream_t stream) {
  const auto result = real_launch(function,grid,block,args,shared,stream);
  if (!enabled || !ready || injecting || result != cudaSuccess) return result;
  const char* name=nullptr; Check(cudaFuncGetName(&name,function)); if (!name) return result;
  const bool global=std::strstr(name,"SplitOnlineDecodeAttentionFp8GlobalKvh2FixedRowsKernel")!=nullptr;
  const bool local=std::strstr(name,"SplitOnlineDecodeAttentionFp8LocalFixedSharedKvKernel")!=nullptr;
  if (!global && !local) return result;
  if (global ? global_calls++ % 5U != 4U : local_calls++ % 25U != 0U) return result;
  injecting=true;
  auto& r=records[global?0:1]; const auto b=Buffer(0,global);
  r.function=function; r.grid=grid; r.block=block; r.splits=Arg<int>(args,global?8:10); r.valid=true;
  if (r.splits!=(global?137:8) || block.x!=256U) std::abort();
  Copy(Arg<const void*>(args,0),b.query,3U*16U*(global?512U:256U)*4U,stream);
  Copy(Arg<const void*>(args,1),b.keys,global?kGlobalCacheBytes:kLocalCacheBytes,stream);
  Copy(Arg<const void*>(args,2),b.values,global?kGlobalCacheBytes:kLocalCacheBytes,stream);
  if (local) { Copy(Arg<const void*>(args,3),b.spec_keys,3U*8U*256U,stream);
    Copy(Arg<const void*>(args,4),b.spec_values,3U*8U*256U,stream); }
  Copy(Arg<const void*>(args,global?3:5),b.ks,2U,stream);
  Copy(Arg<const void*>(args,global?4:6),b.vs,2U,stream);
  Copy(Arg<const void*>(args,global?5:7),b.partial,3U*r.splits*16U*(global?512U:256U)*4U,stream);
  Copy(Arg<const void*>(args,global?6:8),b.lse,3U*r.splits*16U*4U,stream);
  Copy(Arg<const void*>(args,global?7:9),b.controls,3U*sizeof(Control),stream);
  injecting=false; return result;
}
extern "C" cudaError_t __cudaLaunchKernel(cudaKernel_t k,dim3 g,dim3 b,void** a,std::size_t s,cudaStream_t t) {
  return cudaLaunchKernel(reinterpret_cast<const void*>(k),g,b,a,s,t);
}
extern "C" int gem16_lab_save_phase(unsigned phase) {
  if (!enabled || !ready || phase>=3U || saved[phase] || !records[0].valid || !records[1].valid) return 1;
  injecting=true; Copy(arena,arena+(phase+1U)*kPairStride,kPairStride,nullptr);
  records[2U*(phase+1U)]=records[0]; records[2U*(phase+1U)+1U]=records[1];
  Check(cudaDeviceSynchronize()); saved[phase]=true; injecting=false; return 0;
}
extern "C" int gem16_lab_finish(const char* directory) {
  if (!enabled || !ready || !directory) return 1;
  try {
    injecting=true; const std::filesystem::path root(directory);
    if (!std::filesystem::create_directory(root)) throw std::runtime_error("attention directory exists");
    const char* setting=std::getenv("GEM16_ATTENTION_CANDIDATE");
    const bool test_global=setting && std::strcmp(setting,"local")!=0;
    const bool test_local=setting && (std::strcmp(setting,"local")==0 || std::strcmp(setting,"both")==0);
    const auto candidate=reinterpret_cast<Candidate>(dlsym(RTLD_DEFAULT,"gem16_lab_launch_t3_global_warp_norm"));
    const auto local_candidate=reinterpret_cast<LocalCandidate>(dlsym(RTLD_DEFAULT,"gem16_lab_launch_t3_local_warp_norm"));
    if ((test_global && !candidate) || (test_local && !local_candidate)) throw std::runtime_error("candidate symbol absent");
    cudaStream_t stream{}; Check(cudaStreamCreate(&stream));
    auto* partial=reinterpret_cast<float*>(arena+4U*kPairStride);
    auto* lse=reinterpret_cast<float*>(arena+4U*kPairStride+14U*MiB);
    std::ofstream summary(root/"attention.json"); summary.precision(9);
    summary<<"{\"diagnostic_only\":true,\"mutable_kv_snapshotted\":true,\"extra_device_bytes\":"<<kArenaBytes<<",\"cases\":[";
    unsigned count=0;
    for (unsigned phase=0;phase<3;++phase) for (bool global:{false,true}) {
      if (!saved[phase]) throw std::runtime_error("phase absent");
      const auto& r=records[2U*(phase+1U)+(global?0U:1U)]; const auto b=Buffer(phase+1U,global);
      std::array<Control,3> controls{}; Check(cudaMemcpy(controls.data(),b.controls,sizeof(controls),cudaMemcpyDeviceToHost));
      if (controls[0].position!=std::array<std::uint64_t,3>{16384U,16951U,17512U}[phase])
        throw std::runtime_error("control position mismatch");
      for (unsigned row=0;row<3;++row) if (controls[row].position!=controls[0].position+row)
        throw std::runtime_error("row controls not consecutive");
      const auto dim=global?512U:256U;
      const bool test=global?test_global:test_local;
      const unsigned active=global?static_cast<unsigned>((controls[0].position+3U+127U)/128U):8U;
      if (active>unsigned(r.splits)) throw std::runtime_error("active split overflow");
      const auto stem=std::string(global?"global":"local")+"-phase"+std::to_string(phase);
      const float* query=b.query; const auto* keys=b.keys; const auto* values=b.values;
      const auto* sk=b.spec_keys; const auto* sv=b.spec_values;
      const auto* ks=b.ks; const auto* vs=b.vs; const auto* control=b.controls; int splits=r.splits;
      void* ga[]{&query,&keys,&values,&ks,&vs,&partial,&lse,&control,&splits};
      void* la[]{&query,&keys,&values,&sk,&sv,&ks,&vs,&partial,&lse,&control,&splits};
      auto parent=[&]{Check(real_launch(r.function,r.grid,r.block,global?ga:la,0U,stream));};
      auto child=[&]{if(global) Check(candidate(query,keys,values,ks,vs,partial,lse,control,splits,r.grid,r.block,stream));
        else Check(local_candidate(query,keys,values,sk,sv,ks,vs,partial,lse,control,splits,r.grid,r.block,stream));};
      auto download=[&](const float* p,const float* l) {
        const auto pb=std::size_t(active)*16U*dim*4U,lb=std::size_t(active)*16U*4U;
        std::vector<std::byte> host(3U*(pb+lb));
        for (unsigned row=0;row<3;++row) {
          Check(cudaMemcpy(host.data()+row*(pb+lb),p+std::size_t(row)*r.splits*16U*dim,pb,cudaMemcpyDeviceToHost));
          Check(cudaMemcpy(host.data()+row*(pb+lb)+pb,l+std::size_t(row)*r.splits*16U,lb,cudaMemcpyDeviceToHost)); }
        return host;
      };
      const auto expected=download(b.partial,b.lse);
      for (unsigned i=0;i<3;++i) parent(); Check(cudaStreamSynchronize(stream));
      if (download(partial,lse)!=expected) throw std::runtime_error("snapshotted parent mismatch: "+stem);
      const char* pc=std::getenv("GEM16_LAB_PROFILE_CASE"); const char* pk=std::getenv("GEM16_LAB_PROFILE_KIND");
      const bool profile=pc && stem==pc;
      if (profile && pk && std::strcmp(pk,"parent")==0) {
        Check(cudaProfilerStart()); parent(); Check(cudaStreamSynchronize(stream)); Check(cudaProfilerStop()); }
      bool exact=false;
      if (test) {
        for (unsigned i=0;i<3;++i) child(); Check(cudaStreamSynchronize(stream));
        exact=download(partial,lse)==expected; if (!exact) throw std::runtime_error("candidate mismatch: "+stem);
        if (profile && pk && std::strcmp(pk,"candidate")==0) {
          Check(cudaProfilerStart()); child(); Check(cudaStreamSynchronize(stream)); Check(cudaProfilerStop()); }
      }
      const auto pg=Graph(stream,parent); const auto cg=test?Graph(stream,child):nullptr;
      cudaEvent_t begin{},end{}; Check(cudaEventCreate(&begin)); Check(cudaEventCreate(&end));
      auto measure=[&](cudaGraphExec_t g) { Check(cudaEventRecord(begin,stream)); Check(cudaGraphLaunch(g,stream));
        Check(cudaEventRecord(end,stream)); Check(cudaEventSynchronize(end)); float ms=0;
        Check(cudaEventElapsedTime(&ms,begin,end)); return ms*1000.0F/16.0F; };
      for (unsigned i=0;i<3;++i) { (void)measure(pg); if (cg) (void)measure(cg); }
      std::array<float,10> p{},c{};
      for (unsigned i=0;i<10;++i) { if (cg && i%2) c[i]=measure(cg); p[i]=measure(pg); if (cg && i%2==0) c[i]=measure(cg); }
      if (cg) { Check(cudaGraphLaunch(cg,stream)); Check(cudaStreamSynchronize(stream));
        if (download(partial,lse)!=expected) throw std::runtime_error("post-timing candidate mismatch"); }
      Export(root/(stem+"-query.bin"),b.query,3U*16U*dim*4U);
      Export(root/(stem+"-controls.bin"),b.controls,3U*sizeof(Control));
      Export(root/(stem+"-keys.bin"),b.keys,global?kGlobalCacheBytes:kLocalCacheBytes);
      Export(root/(stem+"-values.bin"),b.values,global?kGlobalCacheBytes:kLocalCacheBytes);
      Export(root/(stem+"-key-scale.bin"),b.ks,2U); Export(root/(stem+"-value-scale.bin"),b.vs,2U);
      if (!global) { Export(root/(stem+"-spec-keys.bin"),b.spec_keys,3U*8U*256U); Export(root/(stem+"-spec-values.bin"),b.spec_values,3U*8U*256U); }
      Export(root/(stem+"-partial.bin"),b.partial,3U*r.splits*16U*dim*4U);
      Export(root/(stem+"-lse.bin"),b.lse,3U*r.splits*16U*4U);
      if(test && phase==0U && std::getenv("GEM16_LAB_BOUNDARIES")) {
        // Counterfactual positions over the captured Q/K/V bytes. This tests
        // causality, empty splits and all tail lengths; it is not model quality
        // evidence for those synthetic prefixes. Original snapshots above stay
        // immutable on disk. The expected device output is reused only here.
        auto* errors=reinterpret_cast<unsigned*>(arena+kArenaBytes-256U);
        unsigned cases=0U; const auto* temporary_partial=partial; const auto* temporary_lse=lse;
        const std::array<unsigned,3> bases=global?std::array<unsigned,3>{0U,16384U,17516U}:
                                                   std::array<unsigned,3>{0U,896U,1024U};
        for(unsigned base:bases) for(unsigned tail=0;tail<(global&&base==17516U?1U:128U);++tail) {
          auto changed=controls; for(unsigned row=0;row<3;++row) changed[row].position=base+tail+row;
          Check(cudaMemcpy(b.controls,changed.data(),sizeof(changed),cudaMemcpyHostToDevice));
          Check(cudaMemset(errors,0,sizeof(unsigned)));
          partial=b.partial; lse=b.lse; parent();
          partial=const_cast<float*>(temporary_partial); lse=const_cast<float*>(temporary_lse); child();
          const auto active_boundary=global?(base+tail+3U+127U)/128U:8U;
          CompareAttentionBoundaryKernel<<<256U,256U,0U,stream>>>(b.partial,partial,b.lse,lse,
              active_boundary,r.splits,dim,errors); Check(cudaGetLastError()); Check(cudaStreamSynchronize(stream));
          unsigned bad=0; Check(cudaMemcpy(&bad,errors,sizeof(bad),cudaMemcpyDeviceToHost));
          if(bad) throw std::runtime_error("boundary mismatch: "+stem+" position "+std::to_string(base+tail));
          ++cases;
        }
        Check(cudaMemcpy(b.controls,controls.data(),sizeof(controls),cudaMemcpyHostToDevice));
        std::ofstream boundary(root/(stem+"-boundaries.json"));
        boundary<<"{\"counterfactual_positions_on_captured_bytes\":true,\"cases\":"<<cases
                <<",\"partial_and_lse_bit_exact\":true,\"bases\":["<<bases[0]<<","<<bases[1]<<","<<bases[2]<<"]}\n";
        if(!boundary) throw std::runtime_error("boundary result write failed");
      }
      cudaFuncAttributes attr{}; Check(cudaFuncGetAttributes(&attr,r.function));
      summary<<(count++?",":"")<<"{\"stem\":\""<<stem<<"\",\"layer\":"<<(global?29:0)
        <<",\"position\":"<<controls[0].position<<",\"max_splits\":"<<r.splits<<",\"active_splits\":"<<active
        <<",\"grid_x\":"<<r.grid.x<<",\"block_x\":"<<r.block.x<<",\"parent_registers\":"<<attr.numRegs
        <<",\"parent_shared_bytes\":"<<attr.sharedSizeBytes<<",\"parent_replay_bit_exact\":true,\"candidate_tested\":"<<(cg?"true":"false")
        <<",\"candidate_bit_exact\":"<<(exact?"true":"false")<<",\"parent_us_samples\":[";
      for(unsigned i=0;i<10;++i) summary<<(i?",":"")<<p[i]; summary<<"],\"candidate_us_samples\":[";
      if(cg) for(unsigned i=0;i<10;++i) summary<<(i?",":"")<<c[i]; summary<<"]}";
      Check(cudaGraphExecDestroy(pg)); if(cg) Check(cudaGraphExecDestroy(cg));
      Check(cudaEventDestroy(begin)); Check(cudaEventDestroy(end));
    }
    summary<<"]}\n"; if(!summary) throw std::runtime_error("summary write failed");
    Check(cudaStreamDestroy(stream)); injecting=false; return 0;
  } catch(const std::exception& e) { std::fprintf(stderr,"attention lab: %s\n",e.what()); return 1; }
}
