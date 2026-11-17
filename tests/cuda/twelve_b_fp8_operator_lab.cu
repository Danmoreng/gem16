// Explicit offline real-input observer. Never preload in primary timing.
#include "fp8_projection_lab_abi.h"
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
using Gate=cudaError_t(*)(unsigned,const std::uint8_t*,const float*,Fp8LabBinding,Fp8LabBinding,Fp8LabBinding,unsigned,std::uint64_t,dim3,dim3,cudaStream_t);
Launch launch=nullptr;Begin begin=nullptr;Query query=nullptr;Gate gate=nullptr;
bool enabled=false,ready=false,injecting=false,finished=false;unsigned calls=0;
constexpr std::size_t stride=512*1024,slots=8,phases=3,evict_bytes=256*1024*1024;
constexpr std::array<unsigned,4> layers{0,15,31,47};std::uint8_t*arena=nullptr,*evict=nullptr;unsigned*counter=nullptr;
struct Record{std::uint64_t width=0;unsigned count=0;dim3 grid{},block{};std::array<Fp8LabBinding,3>b{};};std::array<Record,slots> records{};
std::uint8_t* Base(unsigned i){return arena+i*stride;}
float* Scale(unsigned i){return reinterpret_cast<float*>(Base(i)+40000);}
float* Output(unsigned i,unsigned matrix){return reinterpret_cast<float*>(Base(i)+65536+matrix*131072);}
void Check(cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"12B FP8 lab: %s\n",cudaGetErrorString(e));std::abort();}}
template<class T>T Arg(void**a,unsigned i){T v{};std::memcpy(&v,a[i],sizeof(v));return v;}
__global__ void Snapshot(const std::uint8_t*from,std::uint8_t*to,std::size_t n,const unsigned*counter,bool increment){__shared__ unsigned group;if(threadIdx.x==0)group=increment?atomicAdd(const_cast<unsigned*>(counter),1U):*counter-1U;__syncthreads();int phase=group==0?0:(group==250?1:(group==506?2:-1));if(phase<0)return;to+=std::size_t(phase)*slots*stride;for(std::size_t i=threadIdx.x;i<n;i+=blockDim.x)to[i]=from[i];}
__global__ void Evict(std::uint8_t*p,std::size_t n){for(std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=std::size_t(gridDim.x)*blockDim.x)p[i]+=1;}
std::vector<std::uint8_t> Download(const void*p,std::size_t n){std::vector<std::uint8_t>v(n);Check(cudaMemcpy(v.data(),p,n,cudaMemcpyDeviceToHost));return v;}
void Export(const std::filesystem::path&p,const void*v,std::size_t n){const auto b=Download(v,n);std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(b.data()),n);if(!f)throw std::runtime_error("fixture write");}
std::array<float*,3> outputs{};
void Run(unsigned variant,unsigned i,cudaStream_t s){const auto&r=records[i%slots];auto b=r.b;for(unsigned j=0;j<r.count;++j)b[j].output=outputs[j];Check(gate(variant,Base(i),Scale(i),b[0],b[1],b[2],r.count,r.width,r.grid,r.block,s));}
cudaGraphExec_t Graph(unsigned variant,unsigned i,unsigned n,cudaStream_t s){cudaGraph_t g{};Check(begin(s,cudaStreamCaptureModeThreadLocal));for(unsigned j=0;j<n;++j)Run(variant,i,s);Check(cudaStreamEndCapture(s,&g));cudaGraphExec_t ex{};Check(cudaGraphInstantiate(&ex,g,0ULL));Check(cudaGraphDestroy(g));return ex;}
void Finish(){finished=true;injecting=true;const auto dir=std::filesystem::path(std::getenv("GEM16_12B_FP8_LAB_DIR"));if(!std::filesystem::create_directory(dir))throw std::runtime_error("directory exists");for(auto&v:outputs)Check(cudaMalloc(&v,131072));Check(cudaMalloc(&evict,evict_bytes));Check(cudaMemset(evict,0,evict_bytes));cudaStream_t s{};Check(cudaStreamCreate(&s));cudaEvent_t start{},end{};Check(cudaEventCreate(&start));Check(cudaEventCreate(&end));
std::ofstream f(dir/"screen.json");f.precision(10);f<<"{\"diagnostic_only\":true,\"borrowed_weights\":true,\"actual_groups\":[0,250,506],\"cases\":[";
for(unsigned i=0;i<phases*slots;++i){const auto&r=records[i%slots];if(!r.count||r.width>8192)throw std::runtime_error("missing/invalid capture");Run(0,i,s);Check(cudaStreamSynchronize(s));std::array<std::vector<std::uint8_t>,3>reference;for(unsigned j=0;j<r.count;++j){const auto n=3*r.b[j].rows*sizeof(float);reference[j]=Download(outputs[j],n);if(reference[j]!=Download(Output(i,j),n))throw std::runtime_error("parent actual replay mismatch");}Run(1,i,s);Check(cudaStreamSynchronize(s));for(unsigned j=0;j<r.count;++j)if(reference[j]!=Download(outputs[j],reference[j].size()))throw std::runtime_error("packed T3 differs");
Export(dir/("a-"+std::to_string(i)+".bin"),Base(i),3*r.width);Export(dir/("s-"+std::to_string(i)+".bin"),Scale(i),12);for(unsigned j=0;j<r.count;++j){auto stem=std::to_string(i)+"-"+std::to_string(j);Export(dir/("expected-"+stem+".bin"),Output(i,j),3*r.b[j].rows*sizeof(float));if(i<slots){Export(dir/("w-"+stem+".bin"),r.b[j].weight,r.b[j].rows*r.width);Export(dir/("ws-"+stem+".bin"),r.b[j].weight_scales,2*r.b[j].rows);}}
auto p=Graph(0,i,16,s),c=Graph(1,i,16,s),pe=Graph(0,i,1,s),ce=Graph(1,i,1,s);for(unsigned j=0;j<4;++j){Check(cudaGraphLaunch(p,s));Check(cudaGraphLaunch(c,s));}Check(cudaStreamSynchronize(s));auto time=[&](cudaGraphExec_t g,bool cold){if(cold)Evict<<<512,256,0,s>>>(evict,evict_bytes);Check(cudaEventRecord(start,s));Check(cudaGraphLaunch(g,s));Check(cudaEventRecord(end,s));Check(cudaEventSynchronize(end));float ms=0;Check(cudaEventElapsedTime(&ms,start,end));return 1000.0*ms/(cold?1:16);};
if(i)f<<',';f<<"{\"slot\":"<<i%slots<<",\"phase\":"<<i/slots<<",\"layer\":"<<layers[(i%slots)/2]<<",\"family\":\""<<(i%2?"O":"QKV")<<"\",\"width\":"<<r.width<<",\"count\":"<<r.count<<",\"rows\":[";for(unsigned j=0;j<r.count;++j){if(j)f<<',';f<<r.b[j].rows;}f<<"],\"grid\":["<<r.grid.x<<','<<r.grid.y<<','<<r.grid.z<<"],\"block\":["<<r.block.x<<','<<r.block.y<<','<<r.block.z<<"],\"bit_equal\":true,\"samples\":[";
for(unsigned j=0;j<25;++j){double a,b,d,e;if(j%2==0){a=time(p,false);b=time(c,false);d=time(pe,true);e=time(ce,true);}else{b=time(c,false);a=time(p,false);e=time(ce,true);d=time(pe,true);}if(j)f<<',';f<<'['<<a<<','<<b<<','<<d<<','<<e<<']';}f<<"]}";for(auto g:{p,c,pe,ce})Check(cudaGraphExecDestroy(g));}
f<<"]}\n";f.close();Check(cudaEventDestroy(start));Check(cudaEventDestroy(end));Check(cudaStreamDestroy(s));for(auto v:outputs)Check(cudaFree(v));Check(cudaFree(evict));injecting=false;}
__attribute__((constructor))void Init(){launch=reinterpret_cast<Launch>(dlsym(RTLD_NEXT,"cudaLaunchKernel"));begin=reinterpret_cast<Begin>(dlsym(RTLD_NEXT,"cudaStreamBeginCapture"));query=reinterpret_cast<Query>(dlsym(RTLD_NEXT,"cudaStreamQuery"));enabled=std::getenv("GEM16_12B_FP8_LAB_DIR")!=nullptr;}
}
extern "C" cudaError_t cudaStreamBeginCapture(cudaStream_t s,cudaStreamCaptureMode mode){if(enabled&&!ready){injecting=true;Check(cudaMalloc(&arena,phases*slots*stride));Check(cudaMemset(arena,0,phases*slots*stride));Check(cudaMalloc(&counter,4));Check(cudaMemset(counter,0,4));gate=reinterpret_cast<Gate>(dlsym(RTLD_DEFAULT,"gem16_12b_fp8"));if(!gate)std::abort();ready=true;injecting=false;}return begin(s,mode);}
extern "C" cudaError_t cudaLaunchKernel(const void*f,dim3 grid,dim3 block,void**args,std::size_t shared,cudaStream_t s){const auto e=launch(f,grid,block,args,shared,s);if(!enabled||!ready||injecting||finished||e!=cudaSuccess)return e;const char*n=nullptr;Check(cudaFuncGetName(&n,f));if(!n||!std::strstr(n,"Sm120DirectProjectionFixedBatchKernel"))return e;const unsigned call=calls++%96,layer=call/2;int slot=-1;for(unsigned j=0;j<4;++j)if(layer==layers[j])slot=2*j+call%2;if(slot<0)return e;injecting=true;Record r;r.width=Arg<std::uint64_t>(args,6);r.count=Arg<unsigned>(args,5);r.grid=grid;r.block=block;for(unsigned j=0;j<3;++j)r.b[j]=Arg<Fp8LabBinding>(args,2+j);if(r.count<1||r.count>3||r.width>8192||r.width*3>40000)std::abort();for(unsigned j=0;j<r.count;++j)if(r.b[j].rows*12>131072)std::abort();records[slot]=r;
Snapshot<<<1,256,0,s>>>(Arg<const std::uint8_t*>(args,0),Base(slot),3*r.width,counter,slot==0);Snapshot<<<1,256,0,s>>>(reinterpret_cast<const std::uint8_t*>(Arg<const float*>(args,1)),reinterpret_cast<std::uint8_t*>(Scale(slot)),12,counter,false);for(unsigned j=0;j<r.count;++j)Snapshot<<<1,256,0,s>>>(reinterpret_cast<const std::uint8_t*>(r.b[j].output),reinterpret_cast<std::uint8_t*>(Output(slot,j)),3*r.b[j].rows*sizeof(float),counter,false);Check(cudaGetLastError());injecting=false;return e;}
extern "C" cudaError_t __cudaLaunchKernel(cudaKernel_t k,dim3 g,dim3 b,void**a,std::size_t n,cudaStream_t s){return cudaLaunchKernel(reinterpret_cast<const void*>(k),g,b,a,n,s);}
extern "C" cudaError_t cudaStreamQuery(cudaStream_t s){const auto e=query(s);if(enabled&&ready&&!injecting&&!finished&&calls>=96&&e==cudaSuccess){unsigned count=0;injecting=true;Check(cudaMemcpy(&count,counter,4,cudaMemcpyDeviceToHost));injecting=false;if(count>=507){try{Finish();}catch(const std::exception&x){std::fprintf(stderr,"12B FP8 lab: %s\n",x.what());std::abort();}}}return e;}
