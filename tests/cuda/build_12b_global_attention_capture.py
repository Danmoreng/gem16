"""Parent-only actual12B Target global D2 captures for the preload start filter."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('destination', type=Path)
a = p.parse_args()
out = a.destination
out.mkdir(parents=True, exist_ok=True)
source = ROOT / 'src/cuda/attention/decode_sm120.cu'
parent = source.read_text() + r'''
extern "C" cudaError_t gem16_12b_attention_parent(
 const float*q,const std::uint8_t*k,const std::uint8_t*v,
 const std::uint16_t*ks,const std::uint16_t*vs,float*partial,float*lse,
 const gem16::internal::DecodeControl*controls,cudaStream_t stream){
 using namespace gem16::internal;
 SplitOnlineDecodeAttentionFp8GlobalGqaKernel<false,true,true>
   <<<dim3(35U,3U),256U,0U,stream>>>(q,k,v,ks,vs,partial,lse,nullptr,nullptr,
                                   0U,controls,17519U,35);
 return cudaGetLastError();
}
'''
observer = r'''
#include "cuda/layer/reference.h"
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>
namespace {
using Control=gem16::internal::DecodeControl;
using Launch=cudaError_t(*)(const void*,dim3,dim3,void**,std::size_t,cudaStream_t);
using Begin=cudaError_t(*)(cudaStream_t,cudaStreamCaptureMode);
using Query=cudaError_t(*)(cudaStream_t);
using Parent=cudaError_t(*)(const float*,const std::uint8_t*,const std::uint8_t*,const std::uint16_t*,const std::uint16_t*,float*,float*,const Control*,cudaStream_t);
Launch launch=nullptr;Begin begin=nullptr;Query query=nullptr;Parent parent=nullptr;
bool enabled=false,ready=false,injecting=false,finished=false,directed=false;
unsigned calls=0,ignored=0,directed_captures=0;
constexpr std::size_t stride=24*1024*1024,slots=8,phases=3;
constexpr std::size_t query_bytes=3*16*512*4,cache_bytes=17519*512;
constexpr std::size_t partial_bytes=3*35*16*512*4,lse_bytes=3*35*16*4;
std::uint8_t*arena=nullptr;unsigned*counter=nullptr;
struct Buffers{float*q;Control*c;std::uint16_t*ks,*vs;std::uint8_t*k,*v;float*p,*l;};
Buffers Buffer(unsigned i){
 auto*start=arena+i*stride;auto*p=start;
 auto reserve=[&](std::size_t n){auto*r=p;p+=(n+255U)&~std::size_t(255U);return r;};
 Buffers b{};
 b.q=reinterpret_cast<float*>(reserve(query_bytes));
 b.c=reinterpret_cast<Control*>(reserve(3*sizeof(Control)));
 b.ks=reinterpret_cast<std::uint16_t*>(reserve(2));b.vs=reinterpret_cast<std::uint16_t*>(reserve(2));
 b.k=reserve(cache_bytes);b.v=reserve(cache_bytes);
 b.p=reinterpret_cast<float*>(reserve(partial_bytes));b.l=reinterpret_cast<float*>(reserve(lse_bytes));
 if(p>start+stride)std::abort();return b;
}
void Check(cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"12B global attention capture: %s\n",cudaGetErrorString(e));std::abort();}}
template<class T>T Arg(void**a,unsigned i){T v{};std::memcpy(&v,a[i],sizeof(v));return v;}
__global__ void Snapshot(const std::uint8_t*from,std::uint8_t*to,std::size_t n,const unsigned*counter,bool increment){
 __shared__ unsigned group;
 if(threadIdx.x==0)group=increment?atomicAdd(const_cast<unsigned*>(counter),1U):*counter-1U;
 __syncthreads();
 int phase=group==0?0:(group==250?1:(group==506?2:-1));if(phase<0)return;
 to+=std::size_t(phase)*slots*stride;
 for(std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=std::size_t(gridDim.x)*blockDim.x)to[i]=from[i];
}
// Only the first query snapshot increments the group counter; it uses one CTA.
// Every subsequent multi-CTA snapshot reads the completed same-stream counter.
void Copy(const void*from,void*to,std::size_t n,bool increment,cudaStream_t stream){
 Snapshot<<<increment?1U:256U,256U,0U,stream>>>(static_cast<const std::uint8_t*>(from),static_cast<std::uint8_t*>(to),n,counter,increment);
 Check(cudaGetLastError());
}
std::vector<std::uint8_t> Download(const void*p,std::size_t n){std::vector<std::uint8_t>v(n);Check(cudaMemcpy(v.data(),p,n,cudaMemcpyDeviceToHost));return v;}
void Export(const std::filesystem::path&p,const void*v,std::size_t n){const auto b=Download(v,n);std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(b.data()),n);if(!f)throw std::runtime_error("fixture write failed");}
void Finish(){
 finished=true;injecting=true;
 if(calls!=8||ignored!=8||directed_captures!=2)throw std::runtime_error("unexpected eight global Target calls / standalone / conditional graph topology");
 unsigned groups=0;Check(cudaMemcpy(&groups,counter,4,cudaMemcpyDeviceToHost));
 if(groups!=507)throw std::runtime_error("expected507 executed D2groups");
 const auto dir=std::filesystem::path(std::getenv("GEM16_12B_GLOBAL_ATTENTION_CAPTURE_DIR"));
 if(!std::filesystem::create_directory(dir))throw std::runtime_error("capture directory exists");
 float*p=nullptr,*l=nullptr;Check(cudaMalloc(&p,partial_bytes));Check(cudaMalloc(&l,lse_bytes));
 cudaStream_t stream{};Check(cudaStreamCreate(&stream));
 std::ofstream f(dir/"captures.json");
 f<<"{\"diagnostic_only\":true,\"parent_only\":true,\"captured_static_calls\":"<<calls<<",\"ignored_standalone_calls\":"<<ignored<<",\"directed_captures\":"<<directed_captures<<",\"groups\":"<<groups<<",\"actual_groups\":[0,250,506],\"cache_capacity\":17519,\"max_splits\":35,\"split_tokens\":512,\"rows\":3,\"query_heads\":16,\"kv_heads\":1,\"head_dimension\":512,\"grid\":[35,3,1],\"block\":[256,1,1],\"cases\":[";
 for(unsigned i=0;i<phases*slots;++i){
  const auto b=Buffer(i);std::array<Control,3>controls{};
  Check(cudaMemcpy(controls.data(),b.c,sizeof(controls),cudaMemcpyDeviceToHost));
  if(controls[0].position<16384||controls[0].position+3>17519)throw std::runtime_error("invalid actual control position");
  const auto ep=Download(b.p,partial_bytes),el=Download(b.l,lse_bytes);
  // Seed untouched invalid splits from their actual bytes, then compare the
  // complete allocation: valid outputs must match and invalid outputs stay put.
  Check(cudaMemcpyAsync(p,b.p,partial_bytes,cudaMemcpyDeviceToDevice,stream));
  Check(cudaMemcpyAsync(l,b.l,lse_bytes,cudaMemcpyDeviceToDevice,stream));
  Check(parent(b.q,b.k,b.v,b.ks,b.vs,p,l,b.c,stream));Check(cudaStreamSynchronize(stream));
  if(ep!=Download(p,partial_bytes)||el!=Download(l,lse_bytes))throw std::runtime_error("parent partial/LSE replay differs from actual bytes");
  const auto stem=std::to_string(i);
  Export(dir/("q-"+stem+".bin"),b.q,query_bytes);Export(dir/("c-"+stem+".bin"),b.c,sizeof(controls));
  Export(dir/("k-"+stem+".bin"),b.k,cache_bytes);Export(dir/("v-"+stem+".bin"),b.v,cache_bytes);
  Export(dir/("ks-"+stem+".bin"),b.ks,2);Export(dir/("vs-"+stem+".bin"),b.vs,2);
  Export(dir/("p-"+stem+".bin"),b.p,partial_bytes);Export(dir/("l-"+stem+".bin"),b.l,lse_bytes);
  if(i)f<<',';f<<"{\"slot\":"<<i%slots<<",\"phase\":"<<i/slots<<",\"position\":"<<controls[0].position<<",\"parent_partial_lse_bit_equal\":true}";
 }
 f<<"]}\n";f.close();if(!f)throw std::runtime_error("manifest write failed");
 Check(cudaStreamDestroy(stream));Check(cudaFree(p));Check(cudaFree(l));injecting=false;
}
__attribute__((constructor))void Init(){
 launch=reinterpret_cast<Launch>(dlsym(RTLD_NEXT,"cudaLaunchKernel"));begin=reinterpret_cast<Begin>(dlsym(RTLD_NEXT,"cudaStreamBeginCapture"));
 query=reinterpret_cast<Query>(dlsym(RTLD_NEXT,"cudaStreamQuery"));enabled=std::getenv("GEM16_12B_GLOBAL_ATTENTION_CAPTURE_DIR")!=nullptr;
}
}
extern "C" cudaError_t cudaStreamBeginCapture(cudaStream_t stream,cudaStreamCaptureMode mode){
 if(enabled&&!ready){injecting=true;Check(cudaMalloc(&arena,phases*slots*stride));Check(cudaMemset(arena,0,phases*slots*stride));
  Check(cudaMalloc(&counter,4));Check(cudaMemset(counter,0,4));parent=reinterpret_cast<Parent>(dlsym(RTLD_DEFAULT,"gem16_12b_attention_parent"));
  if(!parent)std::abort();ready=true;injecting=false;}return begin(stream,mode);
}
extern "C" cudaError_t cudaStreamBeginCaptureToGraph(cudaStream_t stream,cudaGraph_t graph,const cudaGraphNode_t*dependencies,const cudaGraphEdgeData*edges,std::size_t count,cudaStreamCaptureMode mode){
 using Fn=cudaError_t(*)(cudaStream_t,cudaGraph_t,const cudaGraphNode_t*,const cudaGraphEdgeData*,std::size_t,cudaStreamCaptureMode);
 static auto real=reinterpret_cast<Fn>(dlsym(RTLD_NEXT,"cudaStreamBeginCaptureToGraph"));const auto e=real(stream,graph,dependencies,edges,count,mode);
 if(enabled&&ready&&!injecting&&!finished&&e==cudaSuccess){directed=true;++directed_captures;}return e;
}
extern "C" cudaError_t cudaStreamEndCapture(cudaStream_t stream,cudaGraph_t*graph){
 using Fn=cudaError_t(*)(cudaStream_t,cudaGraph_t*);static auto real=reinterpret_cast<Fn>(dlsym(RTLD_NEXT,"cudaStreamEndCapture"));
 const auto e=real(stream,graph);directed=false;return e;
}
extern "C" cudaError_t cudaLaunchKernel(const void*fn,dim3 grid,dim3 block,void**args,std::size_t shared,cudaStream_t stream){
 const auto e=launch(fn,grid,block,args,shared,stream);if(!enabled||!ready||injecting||finished||e!=cudaSuccess)return e;
 const char*n=nullptr;Check(cudaFuncGetName(&n,fn));if(!n||!std::strstr(n,"SplitOnlineDecodeAttentionFp8GlobalGqaKernel")||grid.y!=3)return e;
 if(grid.x!=35||grid.z!=1||block.x!=256||block.y!=1||block.z!=1||Arg<std::uint64_t>(args,11)!=17519||Arg<int>(args,12)!=35)std::abort();
 if(!directed){++ignored;return e;}const unsigned slot=calls++;if(slot>=slots)std::abort();injecting=true;const auto b=Buffer(slot);
 Copy(Arg<const void*>(args,0),b.q,query_bytes,slot==0,stream);
 Copy(Arg<const void*>(args,1),b.k,cache_bytes,false,stream);Copy(Arg<const void*>(args,2),b.v,cache_bytes,false,stream);
 Copy(Arg<const void*>(args,3),b.ks,2,false,stream);Copy(Arg<const void*>(args,4),b.vs,2,false,stream);
 Copy(Arg<const void*>(args,5),b.p,partial_bytes,false,stream);Copy(Arg<const void*>(args,6),b.l,lse_bytes,false,stream);
 Copy(Arg<const void*>(args,10),b.c,3*sizeof(Control),false,stream);injecting=false;return e;
}
extern "C" cudaError_t __cudaLaunchKernel(cudaKernel_t k,dim3 g,dim3 b,void**a,std::size_t n,cudaStream_t s){return cudaLaunchKernel(reinterpret_cast<const void*>(k),g,b,a,n,s);}
extern "C" cudaError_t cudaStreamQuery(cudaStream_t stream){
 const auto e=query(stream);if(enabled&&ready&&!injecting&&!finished&&calls>=8&&e==cudaSuccess){unsigned groups=0;
  injecting=true;Check(cudaMemcpy(&groups,counter,4,cudaMemcpyDeviceToHost));injecting=false;
  if(groups>=507){try{Finish();}catch(const std::exception&x){std::fprintf(stderr,"12B global attention capture: %s\n",x.what());std::abort();}}}return e;
}
'''
for name, text in [('parent.cu', parent), ('observer.cu', observer)]:
    with (out / name).open('x') as f:
        f.write(text)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
with (out / 'capture-derivatives.json').open('x') as f:
    json.dump({'scope': 'parent-only actual12B global Target D2 capture/replay; no preload candidate before current-counter decision',
               'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), source]},
               'outputs': {p.name: sha(p) for p in [out / 'parent.cu', out / 'observer.cu']}}, f, indent=2)
