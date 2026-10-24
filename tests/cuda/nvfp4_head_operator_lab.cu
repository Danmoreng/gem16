// Offline actual-head observer. Mutable inputs and all output bytes are copied;
// immutable resident weights are borrowed. Never use in primary throughput runs.
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <dlfcn.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
using Launch=cudaError_t(*)(const void*,dim3,dim3,void**,std::size_t,cudaStream_t);
using Begin=cudaError_t(*)(cudaStream_t,cudaStreamCaptureMode);
using Candidate=cudaError_t(*)(unsigned,const std::uint8_t*,const std::uint8_t*,const std::uint8_t*,const std::uint8_t*,void*,std::uint64_t,std::uint64_t,float,std::uint32_t,std::uint32_t,dim3,dim3,cudaStream_t);
Launch real_launch;Begin real_begin;bool enabled=false,ready=false,injecting=false,t3=false;
unsigned assistant_calls=0;std::byte* arena=nullptr;
constexpr std::size_t kStride=8U*1024U*1024U,kBytes=12U*kStride;
struct Record {const void* function=nullptr;dim3 grid{},block{};std::size_t shared=0;
 const std::uint8_t *weight=nullptr,*weight_scales=nullptr;std::uint64_t rows=0,k=0,tokens=0;
 float divisor=0;std::uint32_t mask0=0,mask1=0;unsigned kind=0;std::size_t output_bytes=0;bool valid=false;};
std::array<Record,12> records{};std::array<bool,3> saved{};
struct Buffer {std::uint8_t *a,*s;std::byte *expected,*temporary;};
Buffer Buff(unsigned i){auto*p=arena+i*kStride;return{reinterpret_cast<std::uint8_t*>(p),reinterpret_cast<std::uint8_t*>(p+65536),p+131072,p+4U*1024U*1024U};}
void Check(cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"head lab: %s\n",cudaGetErrorString(e));std::abort();}}
template<class T>T Arg(void**args,unsigned i){T x{};std::memcpy(&x,args[i],sizeof(x));return x;}
__global__ void DiagnosticHeadCopy(const std::byte*from,std::byte*to,std::size_t n){for(std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=std::size_t(gridDim.x)*blockDim.x)to[i]=from[i];}
void Copy(const void*f,void*t,std::size_t n,cudaStream_t s){DiagnosticHeadCopy<<<static_cast<unsigned>((n+255)/256),256,0,s>>>(static_cast<const std::byte*>(f),static_cast<std::byte*>(t),n);Check(cudaGetLastError());}
__attribute__((constructor))void Init(){real_launch=reinterpret_cast<Launch>(dlsym(RTLD_NEXT,"cudaLaunchKernel"));real_begin=reinterpret_cast<Begin>(dlsym(RTLD_NEXT,"cudaStreamBeginCapture"));const char*e=std::getenv("GEM16_HEAD_LAB");enabled=e&&std::strcmp(e,"1")==0;const char*m=std::getenv("GEM16_HEAD_MODE");t3=m&&std::strcmp(m,"t3")==0;}
void Export(const std::filesystem::path&p,const void*f,std::size_t n){if(std::filesystem::exists(p))throw std::runtime_error("export exists");std::vector<std::byte>h(n);Check(cudaMemcpy(h.data(),f,n,cudaMemcpyDeviceToHost));std::ofstream o(p,std::ios::binary);o.write(reinterpret_cast<const char*>(h.data()),n);if(!o)throw std::runtime_error("export failed");}
template<class F>cudaGraphExec_t Graph(cudaStream_t s,F launch){cudaGraph_t g{};Check(real_begin(s,cudaStreamCaptureModeThreadLocal));for(unsigned i=0;i<16;++i)launch();Check(cudaStreamEndCapture(s,&g));cudaGraphExec_t x{};Check(cudaGraphInstantiate(&x,g,0ULL));Check(cudaGraphDestroy(g));return x;}
}
extern "C" cudaError_t cudaStreamBeginCapture(cudaStream_t s,cudaStreamCaptureMode m){if(enabled&&!ready){injecting=true;Check(cudaMalloc(&arena,kBytes));Check(cudaMemset(arena,0,kBytes));ready=true;injecting=false;}return real_begin(s,m);}
extern "C" cudaError_t cudaLaunchKernel(const void*f,dim3 g,dim3 b,void**args,std::size_t sh,cudaStream_t s){
 const auto result=real_launch(f,g,b,args,sh,s);if(!enabled||!ready||injecting||result!=cudaSuccess)return result;
 const char*n=nullptr;Check(cudaFuncGetName(&n,f));if(!n)return result;Record r{};void*out=nullptr;unsigned slot=0;
 if(std::strstr(n,"Sm120ProjectionArgmaxCandidatesBf16Kernel")){
  if(!t3)return result;r.kind=2;r.rows=Arg<std::uint64_t>(args,5);r.k=Arg<std::uint64_t>(args,6);r.divisor=Arg<float>(args,7);r.mask0=Arg<std::uint32_t>(args,8);r.mask1=Arg<std::uint32_t>(args,9);r.tokens=1;r.output_bytes=g.x*8U;out=Arg<void*>(args,4);slot=1U+(assistant_calls++%2U);
 }else if(t3&&(std::strstr(n,"Sm120DirectProjectionFixedT3Kernel")||std::strstr(n,"Sm120DirectProjectionPackedHeadT3Kernel"))){
  r.kind=1;r.rows=Arg<std::uint64_t>(args,5);r.k=Arg<std::uint64_t>(args,6);r.divisor=Arg<float>(args,7);r.tokens=3;out=Arg<void*>(args,4);
 }else if(!t3&&std::strstr(n,"Sm120DirectProjectionKernel")&&std::strstr(n,"Gemma4MoePrefillAssignment")){
  r.kind=0;r.rows=Arg<std::uint64_t>(args,11);r.k=Arg<std::uint64_t>(args,12);r.tokens=Arg<std::uint64_t>(args,10);r.divisor=Arg<float>(args,13);out=Arg<void*>(args,9);if(r.tokens!=1)return result;
 }else return result;
 if(r.rows!=262144U)return result;if(r.kind!=2)r.output_bytes=r.rows*r.tokens*4U;
 if(r.tokens*r.k/2U>65536U||r.tokens*r.k/16U>65536U||r.output_bytes>3U*1024U*1024U)std::abort();
 if(!records[slot].valid)std::fprintf(stderr,"head kind%u rows%llu K%llu T%llu masks%u,%u grid%u block%u function%s\n",r.kind,(unsigned long long)r.rows,(unsigned long long)r.k,(unsigned long long)r.tokens,r.mask0,r.mask1,g.x,b.x,n);
 injecting=true;r.function=f;r.grid=g;r.block=b;r.shared=sh;r.weight=Arg<const std::uint8_t*>(args,2);r.weight_scales=Arg<const std::uint8_t*>(args,3);r.valid=true;auto buf=Buff(slot);
 Copy(Arg<const void*>(args,0),buf.a,r.tokens*r.k/2U,s);Copy(Arg<const void*>(args,1),buf.s,r.tokens*r.k/16U,s);Copy(out,buf.expected,r.output_bytes,s);records[slot]=r;injecting=false;return result;
}
extern "C" cudaError_t __cudaLaunchKernel(cudaKernel_t k,dim3 g,dim3 b,void**a,std::size_t s,cudaStream_t t){return cudaLaunchKernel(reinterpret_cast<const void*>(k),g,b,a,s,t);}
extern "C" int gem16_lab_save_phase(unsigned phase){if(!enabled||!ready||phase>=3||saved[phase])return 1;injecting=true;for(unsigned i=0;i<(t3?3U:1U);++i){if(!records[i].valid)throw std::runtime_error("missing head record "+std::to_string(i));unsigned d=3U+phase*3U+i;Copy(arena+i*kStride,arena+d*kStride,kStride,nullptr);records[d]=records[i];}Check(cudaDeviceSynchronize());saved[phase]=true;injecting=false;return 0;}
extern "C" int gem16_lab_finish(const char*directory){try{
 if(!enabled||!ready||!directory)return 1;injecting=true;std::filesystem::path root(directory);if(!std::filesystem::create_directory(root))throw std::runtime_error("head directory exists");
 auto cand=reinterpret_cast<Candidate>(dlsym(RTLD_DEFAULT,"gem16_lab_launch_nvfp4_head"));const char*ch=std::getenv("GEM16_HEAD_CANDIDATE");bool test=ch&&std::strcmp(ch,"1")==0;if(test&&!cand)throw std::runtime_error("missing head candidate");
 cudaStream_t stream{};Check(cudaStreamCreate(&stream));std::ofstream summary(root/"captures.json");summary.precision(9);summary<<"{\"diagnostic_only\":true,\"extra_device_bytes\":"<<kBytes<<",\"immutable_weights_borrowed\":true,\"cases\":[";unsigned count=0;
 for(unsigned p=0;p<3;++p)for(unsigned i=0;i<(t3?3U:1U);++i){if(!saved[p])throw std::runtime_error("missing phase");auto r=records[3+p*3+i];auto buf=Buff(3+p*3+i);
  auto stem="phase"+std::to_string(p)+(i==0?"-target":"-assistant"+std::to_string(i));auto*a=buf.a;auto*sc=buf.s;auto*w=r.weight;auto*ws=r.weight_scales;void*out=buf.temporary;
  const void*nil=nullptr;std::uint32_t zero=0;void*ordinary[]{&a,&sc,&w,&ws,&nil,&nil,&nil,&zero,&zero,&out,&r.tokens,&r.rows,&r.k,&r.divisor};
  void*fixed[]{&a,&sc,&w,&ws,&out,&r.rows,&r.k,&r.divisor};void*assistant[]{&a,&sc,&w,&ws,&out,&r.rows,&r.k,&r.divisor,&r.mask0,&r.mask1};void**args=r.kind==0?ordinary:r.kind==1?fixed:assistant;
  auto parent=[&]{Check(real_launch(r.function,r.grid,r.block,args,r.shared,stream));};auto candidate=[&]{Check(cand(r.kind,a,sc,w,ws,out,r.rows,r.k,r.divisor,r.mask0,r.mask1,r.grid,r.block,stream));};
  bool selected=test&&r.kind!=0;std::vector<std::byte>expected(r.output_bytes),actual(r.output_bytes);Check(cudaMemcpy(expected.data(),buf.expected,r.output_bytes,cudaMemcpyDeviceToHost));
  auto equal=[&]{Check(cudaStreamSynchronize(stream));Check(cudaMemcpy(actual.data(),buf.temporary,r.output_bytes,cudaMemcpyDeviceToHost));return actual==expected;};
  for(unsigned j=0;j<3;++j)parent();if(!equal()){Export(root/(stem+"-mismatch-activation.bin"),buf.a,r.tokens*r.k/2U);Export(root/(stem+"-mismatch-scales.bin"),buf.s,r.tokens*r.k/16U);Export(root/(stem+"-mismatch-expected.bin"),buf.expected,r.output_bytes);Export(root/(stem+"-mismatch-actual.bin"),buf.temporary,r.output_bytes);for(unsigned z=0;z<r.output_bytes/4U;++z){std::uint32_t ev=0,av=0;std::memcpy(&ev,expected.data()+z*4U,4);std::memcpy(&av,actual.data()+z*4U,4);if(ev!=av){std::fprintf(stderr,"first mismatch %s word%u expected%08x actual%08x divisor%.9g\n",stem.c_str(),z,ev,av,r.divisor);break;}}throw std::runtime_error("head parent mismatch "+stem);}
  if(selected){for(unsigned j=0;j<3;++j)candidate();if(!equal())throw std::runtime_error("head candidate mismatch "+stem);}
  const char*pc=std::getenv("GEM16_LAB_PROFILE_CASE");const char*pk=std::getenv("GEM16_LAB_PROFILE_KIND");if(pc&&stem==pc){Check(cudaProfilerStart());if(pk&&std::strcmp(pk,"candidate")==0){if(!selected)throw std::runtime_error("unselected profile");candidate();}else parent();Check(cudaStreamSynchronize(stream));Check(cudaProfilerStop());}
  auto pg=Graph(stream,parent),cg=selected?Graph(stream,candidate):nullptr;cudaEvent_t begin{},end{};Check(cudaEventCreate(&begin));Check(cudaEventCreate(&end));
  auto measure=[&](cudaGraphExec_t g){Check(cudaEventRecord(begin,stream));Check(cudaGraphLaunch(g,stream));Check(cudaEventRecord(end,stream));Check(cudaEventSynchronize(end));float ms=0;Check(cudaEventElapsedTime(&ms,begin,end));return ms*1000.F/16.F;};
  for(unsigned j=0;j<3;++j){(void)measure(pg);if(cg)(void)measure(cg);}std::array<float,10>pv{},cv{};
  for(unsigned j=0;j<10;++j){if(cg&&j%2)cv[j]=measure(cg);pv[j]=measure(pg);if(cg&&j%2==0)cv[j]=measure(cg);}parent();if(!equal())throw std::runtime_error("post parent mismatch");if(cg){candidate();if(!equal())throw std::runtime_error("post candidate mismatch");}
  Export(root/(stem+"-activation.bin"),buf.a,r.tokens*r.k/2U);Export(root/(stem+"-scales.bin"),buf.s,r.tokens*r.k/16U);Export(root/(stem+"-expected.bin"),buf.expected,r.output_bytes);
  cudaFuncAttributes attr{};Check(cudaFuncGetAttributes(&attr,r.function));summary<<(count++?",":"")<<"{\"stem\":\""<<stem<<"\",\"kind\":"<<r.kind<<",\"k\":"<<r.k<<",\"tokens\":"<<r.tokens<<",\"rows\":"<<r.rows<<",\"divisor\":"<<r.divisor<<",\"masks\":["<<r.mask0<<","<<r.mask1<<"],\"output_bytes\":"<<r.output_bytes<<",\"parent_registers\":"<<attr.numRegs<<",\"parent_shared_bytes\":"<<attr.sharedSizeBytes<<",\"parent_bit_exact\":true,\"candidate_tested\":"<<(selected?"true":"false")<<",\"candidate_bit_exact\":"<<(selected?"true":"false")<<",\"parent_us_samples\":[";
  for(unsigned j=0;j<10;++j)summary<<(j?",":"")<<pv[j];summary<<"],\"candidate_us_samples\":[";if(cg)for(unsigned j=0;j<10;++j)summary<<(j?",":"")<<cv[j];summary<<"]}";
  Check(cudaGraphExecDestroy(pg));if(cg)Check(cudaGraphExecDestroy(cg));Check(cudaEventDestroy(begin));Check(cudaEventDestroy(end));
 }summary<<"]}\n";if(!summary)throw std::runtime_error("head summary failed");Check(cudaStreamDestroy(stream));injecting=false;return 0;
 }catch(const std::exception&e){std::fprintf(stderr,"head lab: %s\n",e.what());return 1;}}
