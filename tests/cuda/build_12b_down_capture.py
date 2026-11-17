"""Generate parent-only D2 Down captures before the weight-pipeline filter.

The observer is an explicit offline LD_PRELOAD diagnostic. Its allocations,
snapshots and replay never support a primary inference throughput claim.
"""
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
source = ROOT / 'src/cuda/nvfp4/sm120.cu'
parent = source.read_text() + r'''
extern "C" cudaError_t gem16_12b_down_parent(
 const std::uint8_t*a,const std::uint8_t*sa,const std::uint8_t*b,
 const std::uint8_t*sb,std::uint16_t*y,float divisor,cudaStream_t stream){
 using namespace gem16::internal;
 Sm120MatrixProjectionKernel<false,1U,8U,false,std::uint16_t>
   <<<60U,256U,0U,stream>>>(a,sa,b,sb,nullptr,nullptr,nullptr,nullptr,y,
                           3U,3840U,15360U,divisor,1.0F);
 return cudaGetLastError();
}
'''
observer = r'''
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
using Launch=cudaError_t(*)(const void*,dim3,dim3,void**,std::size_t,cudaStream_t);
using Begin=cudaError_t(*)(cudaStream_t,cudaStreamCaptureMode);
using Query=cudaError_t(*)(cudaStream_t);
using Parent=cudaError_t(*)(const std::uint8_t*,const std::uint8_t*,const std::uint8_t*,const std::uint8_t*,std::uint16_t*,float,cudaStream_t);
Launch launch=nullptr;Begin begin=nullptr;Query query=nullptr;Parent parent=nullptr;
bool enabled=false,ready=false,injecting=false,finished=false,directed=false;
unsigned calls=0,ignored=0,directed_captures=0;
constexpr std::size_t stride=65536,slots=4,phases=3;
constexpr std::size_t activation_bytes=3*15360/2,scale_bytes=3*15360/16;
constexpr std::size_t output_bytes=3*3840*2,weight_bytes=3840*15360/2,weight_scale_bytes=3840*15360/16;
constexpr std::array<unsigned,4> layers{0,15,31,47};
std::uint8_t*arena=nullptr;unsigned*counter=nullptr;
struct Record{const std::uint8_t*weight=nullptr;const std::uint8_t*scales=nullptr;float divisor=0;};
std::array<Record,slots> records{};
std::uint8_t* Base(unsigned i){return arena+i*stride;}
void Check(cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"12B Down capture: %s\n",cudaGetErrorString(e));std::abort();}}
template<class T>T Arg(void**a,unsigned i){T v{};std::memcpy(&v,a[i],sizeof(v));return v;}
__global__ void Snapshot(const std::uint8_t*from,std::uint8_t*to,std::size_t n,const unsigned*counter,bool increment){
 __shared__ unsigned group;
 if(threadIdx.x==0)group=increment?atomicAdd(const_cast<unsigned*>(counter),1U):*counter-1U;
 __syncthreads();
 int phase=group==0?0:(group==250?1:(group==506?2:-1));
 if(phase<0)return;
 to+=std::size_t(phase)*slots*stride;
 for(std::size_t i=threadIdx.x;i<n;i+=blockDim.x)to[i]=from[i];
}
std::vector<std::uint8_t> Download(const void*p,std::size_t n){std::vector<std::uint8_t>v(n);Check(cudaMemcpy(v.data(),p,n,cudaMemcpyDeviceToHost));return v;}
void Export(const std::filesystem::path&p,const void*v,std::size_t n){const auto b=Download(v,n);std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(b.data()),n);if(!f)throw std::runtime_error("fixture write failed");}
void Finish(){
 finished=true;injecting=true;
 if(calls!=48||ignored!=48||directed_captures!=2)throw std::runtime_error("unexpected static48 Down calls / standalone / conditional graph topology");
 unsigned groups=0;Check(cudaMemcpy(&groups,counter,4,cudaMemcpyDeviceToHost));
 if(groups!=507)throw std::runtime_error("expected507 executed D2 groups");
 const auto dir=std::filesystem::path(std::getenv("GEM16_12B_DOWN_CAPTURE_DIR"));
 if(!std::filesystem::create_directory(dir))throw std::runtime_error("capture directory exists");
 std::uint16_t*output=nullptr;Check(cudaMalloc(&output,output_bytes));
 cudaStream_t stream{};Check(cudaStreamCreate(&stream));
 std::ofstream f(dir/"captures.json");f.precision(10);
 f<<"{\"diagnostic_only\":true,\"parent_only\":true,\"captured_static_calls\":"<<calls<<",\"ignored_standalone_calls\":"<<ignored<<",\"directed_captures\":"<<directed_captures<<",\"groups\":"<<groups<<",\"actual_groups\":[0,250,506],\"tokens\":3,\"rows\":3840,\"width\":15360,\"grid\":[60,1,1],\"block\":[256,1,1],\"cases\":[";
 for(unsigned i=0;i<phases*slots;++i){
  const auto&r=records[i%slots];
  if(!r.weight||!r.scales||!(r.divisor>0))throw std::runtime_error("missing or invalid capture");
  const auto expected=Download(Base(i)+32768,output_bytes);
  Check(parent(Base(i),Base(i)+24576,r.weight,r.scales,output,r.divisor,stream));
  Check(cudaStreamSynchronize(stream));
  if(expected!=Download(output,output_bytes))throw std::runtime_error("parent BF16 replay differs from actual output");
  const auto stem=std::to_string(i);
  Export(dir/("a-"+stem+".bin"),Base(i),activation_bytes);
  Export(dir/("s-"+stem+".bin"),Base(i)+24576,scale_bytes);
  Export(dir/("expected-"+stem+".bin"),Base(i)+32768,output_bytes);
  if(i<slots){Export(dir/("w-"+stem+".bin"),r.weight,weight_bytes);Export(dir/("ws-"+stem+".bin"),r.scales,weight_scale_bytes);}
  if(i)f<<',';
  f<<"{\"slot\":"<<i%slots<<",\"phase\":"<<i/slots<<",\"layer\":"<<layers[i%slots]<<",\"output_divisor\":"<<r.divisor<<",\"parent_bf16_bit_equal\":true}";
 }
 f<<"]}\n";f.close();if(!f)throw std::runtime_error("manifest write failed");
 Check(cudaStreamDestroy(stream));Check(cudaFree(output));injecting=false;
}
__attribute__((constructor))void Init(){
 launch=reinterpret_cast<Launch>(dlsym(RTLD_NEXT,"cudaLaunchKernel"));
 begin=reinterpret_cast<Begin>(dlsym(RTLD_NEXT,"cudaStreamBeginCapture"));
 query=reinterpret_cast<Query>(dlsym(RTLD_NEXT,"cudaStreamQuery"));
 enabled=std::getenv("GEM16_12B_DOWN_CAPTURE_DIR")!=nullptr;
}
}
extern "C" cudaError_t cudaStreamBeginCapture(cudaStream_t stream,cudaStreamCaptureMode mode){
 if(enabled&&!ready){
  injecting=true;Check(cudaMalloc(&arena,phases*slots*stride));Check(cudaMemset(arena,0,phases*slots*stride));
  Check(cudaMalloc(&counter,4));Check(cudaMemset(counter,0,4));
  parent=reinterpret_cast<Parent>(dlsym(RTLD_DEFAULT,"gem16_12b_down_parent"));
  if(!parent)std::abort();ready=true;injecting=false;
 }
 return begin(stream,mode);
}
extern "C" cudaError_t cudaStreamBeginCaptureToGraph(cudaStream_t stream,cudaGraph_t graph,const cudaGraphNode_t*dependencies,const cudaGraphEdgeData*edges,std::size_t count,cudaStreamCaptureMode mode){
 using Fn=cudaError_t(*)(cudaStream_t,cudaGraph_t,const cudaGraphNode_t*,const cudaGraphEdgeData*,std::size_t,cudaStreamCaptureMode);
 static auto real=reinterpret_cast<Fn>(dlsym(RTLD_NEXT,"cudaStreamBeginCaptureToGraph"));
 const auto e=real(stream,graph,dependencies,edges,count,mode);
 if(enabled&&ready&&!injecting&&!finished&&e==cudaSuccess){directed=true;++directed_captures;}
 return e;
}
extern "C" cudaError_t cudaStreamEndCapture(cudaStream_t stream,cudaGraph_t*graph){
 using Fn=cudaError_t(*)(cudaStream_t,cudaGraph_t*);
 static auto real=reinterpret_cast<Fn>(dlsym(RTLD_NEXT,"cudaStreamEndCapture"));
 const auto e=real(stream,graph);directed=false;return e;
}
extern "C" cudaError_t cudaLaunchKernel(const void*fn,dim3 grid,dim3 block,void**args,std::size_t shared,cudaStream_t stream){
 const auto e=launch(fn,grid,block,args,shared,stream);
 if(!enabled||!ready||injecting||finished||e!=cudaSuccess)return e;
 const char*n=nullptr;Check(cudaFuncGetName(&n,fn));
 if(!n||!std::strstr(n,"Sm120MatrixProjectionKernel"))return e;
 if(Arg<std::uint64_t>(args,9)!=3||Arg<std::uint64_t>(args,10)!=3840||Arg<std::uint64_t>(args,11)!=15360)return e;
 if(grid.x!=60||grid.y!=1||grid.z!=1||block.x!=256||block.y!=1||block.z!=1)std::abort();
 if(!directed){++ignored;return e;}
 const unsigned layer=calls++;
 int slot=-1;for(unsigned j=0;j<slots;++j)if(layer==layers[j])slot=j;
 if(slot<0)return e;
 injecting=true;
 records[slot]={Arg<const std::uint8_t*>(args,2),Arg<const std::uint8_t*>(args,3),Arg<float>(args,12)};
 Snapshot<<<1,256,0,stream>>>(Arg<const std::uint8_t*>(args,0),Base(slot),activation_bytes,counter,slot==0);
 Snapshot<<<1,256,0,stream>>>(Arg<const std::uint8_t*>(args,1),Base(slot)+24576,scale_bytes,counter,false);
 Snapshot<<<1,256,0,stream>>>(Arg<const std::uint8_t*>(args,8),Base(slot)+32768,output_bytes,counter,false);
 Check(cudaGetLastError());injecting=false;return e;
}
extern "C" cudaError_t __cudaLaunchKernel(cudaKernel_t k,dim3 g,dim3 b,void**a,std::size_t n,cudaStream_t s){return cudaLaunchKernel(reinterpret_cast<const void*>(k),g,b,a,n,s);}
extern "C" cudaError_t cudaStreamQuery(cudaStream_t stream){
 const auto e=query(stream);
 if(enabled&&ready&&!injecting&&!finished&&calls>=48&&e==cudaSuccess){
  unsigned groups=0;injecting=true;Check(cudaMemcpy(&groups,counter,4,cudaMemcpyDeviceToHost));injecting=false;
  if(groups>=507){try{Finish();}catch(const std::exception&x){std::fprintf(stderr,"12B Down capture: %s\n",x.what());std::abort();}}
 }
 return e;
}
'''
for name, text in [('parent.cu', parent), ('observer.cu', observer)]:
    with (out / name).open('x') as f:
        f.write(text)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
with (out / 'capture-derivatives.json').open('x') as f:
    json.dump({'scope': 'parent-only actual12B D2 Down capture/replay; no pipeline candidate before current-counter decision',
               'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), source]},
               'outputs': {p.name: sha(p) for p in [out / 'parent.cu', out / 'observer.cu']}}, f, indent=2)
