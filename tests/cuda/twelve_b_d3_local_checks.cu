// Fixed-T4 ring/speculative masking and experimental FP8 numerical boundaries.
#include "cuda/attention/twelve_b_d3_local.h"
#include "cuda/attention/sm120.h"
#include "cuda/layer/reference.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>
namespace {
using namespace gem16::internal;
void Check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void Check(const gem16::Status&s){if(!s.ok())throw std::runtime_error(s.message());}
template<class T>struct Buffer{
 unsigned char*base=nullptr;T*p=nullptr;std::size_t n;
 explicit Buffer(std::size_t count):n(count){Check(cudaMalloc(&base,n*sizeof(T)+512));p=reinterpret_cast<T*>(base+256);Check(cudaMemset(base,165,n*sizeof(T)+512));}
 ~Buffer(){if(base)(void)cudaFree(base);}
 void Put(const std::vector<T>&v){if(v.size()!=n)throw std::runtime_error("size");Check(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice));}
 std::vector<T>Get(){std::vector<T>x(n);Check(cudaMemcpy(x.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));Guard();return x;}
 void Guard(){std::array<unsigned char,256>a,b;Check(cudaMemcpy(a.data(),base,256,cudaMemcpyDeviceToHost));Check(cudaMemcpy(b.data(),base+256+n*sizeof(T),256,cudaMemcpyDeviceToHost));for(unsigned i=0;i<256;++i)if(a[i]!=165||b[i]!=165)throw std::runtime_error("guard");}
};
std::uint16_t Bf(float x){return __bfloat16_as_ushort(__float2bfloat16_rn(x));}
float Fp(unsigned char x){__nv_fp8_e4m3 a;a.__x=x;return float(a);}
std::vector<unsigned char> Values(unsigned n,unsigned salt){std::vector<unsigned char>x(n);for(unsigned i=0;i<n;++i)x[i]=__nv_fp8_e4m3(float(int((i*37+salt*11)%257)-128)/64.F).__x;return x;}
}
int main(int argc,char**argv){try{
 const bool scoped = argc > 1;
 const bool short_cache = scoped && std::string_view(argv[1]) == "--short-cache";
 constexpr unsigned qsize=4*16*256,specsize=4*8*256;
 const unsigned kvsize = (short_cache ? 5U : 1024U) * 8U * 256U;
 TwelveBD3FastLocalAttention fast;Check(fast.Initialize());
 Buffer<float>q(qsize),out(qsize),reference(qsize),workspace(DecodeAttentionFixedWorkspaceElements(17519,4));Buffer<std::uint8_t>k(kvsize),v(kvsize),sk(specsize),sv(specsize);Buffer<std::uint16_t>ks(1),vs(1);Buffer<DecodeControl>c(4);
 auto keys=Values(kvsize,1),values=Values(kvsize,2),draft_keys=Values(specsize,3),draft_values=Values(specsize,4);k.Put(keys);v.Put(values);sk.Put(draft_keys);sv.Put(draft_values);ks.Put({Bf(.125F)});vs.Put({Bf(.0625F)});
 cudaStream_t stream{};Check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));unsigned cases=0;double worst_error=0,worst_abs=0;
 auto launch=[&](bool candidate){if(candidate)Check(fast.Launch(q.p,k.p,v.p,sk.p,sv.p,ks.p,vs.p,workspace.p,workspace.n,out.p,c.p,stream));else Check(LaunchOnlineAttentionDecodeFp8LocalFixed12BControlledSm120(q.p,k.p,v.p,sk.p,sv.p,ks.p,vs.p,workspace.p,reference.p,c.p,4,1024,stream));};
 std::vector<unsigned> positions{0,1,127,128,1020,1021,1022,1023,1024,2047,16383,17515,262140};
 if (scoped) positions = {0,1022,1023,17515,262140};
 if (short_cache) positions = {0,1};
 for(unsigned position:positions)for(unsigned variant=0;variant<2;++variant){
  std::vector<float>query(qsize);for(unsigned i=0;i<qsize;++i)query[i]=variant==0?0.F:float(__float2bfloat16_rn(float(int((i*13+3)%129)-64)/32.F));q.Put(query);std::vector<DecodeControl>controls(4);for(unsigned row=0;row<4;++row)controls[row].position=position+row;c.Put(controls);
  launch(true);Check(cudaStreamSynchronize(stream));auto actual=out.Get();for(float x:actual)if(!std::isfinite(x))throw std::runtime_error("nonfinite");
  if(variant==0)for(unsigned row=0;row<4;++row)for(unsigned head=0;head<16;++head)for(unsigned dim=0;dim<256;++dim){
   const unsigned end=position+row,first=end>=1023?end-1023:0;double sum=0;
   for(unsigned token=first;token<=end;++token){unsigned offset=token<position?((token%1024)*8+head/2)*256+dim:((token-position)*8+head/2)*256+dim;sum+=Fp(token<position?values[offset]:draft_values[offset])*.0625;}
   if(std::abs(actual[(row*16+head)*256+dim]-float(sum/(end-first+1)))>1.e-6F)throw std::runtime_error("uniform causal/ring/speculative mismatch");
  }
  if(!scoped){launch(false);Check(cudaStreamSynchronize(stream));auto expected=reference.Get();double norm=0,err=0,absolute=0;for(unsigned i=0;i<qsize;++i){err+=std::pow(double(actual[i])-expected[i],2);norm+=double(expected[i])*expected[i];absolute=std::max(absolute,std::abs(double(actual[i])-expected[i]));}double relative=std::sqrt(err/std::max(norm,1.e-30));worst_error=std::max(worst_error,relative);worst_abs=std::max(worst_abs,absolute);if(relative>.15)throw std::runtime_error("local FP8 exceeds predeclared15% L2 bound");std::cout<<"case pos="<<position<<" variant="<<variant<<" relative_l2="<<relative<<" max_abs="<<absolute<<'\n';}
  cudaGraph_t graph{};cudaGraphExec_t exec{};Check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));launch(true);Check(cudaStreamEndCapture(stream,&graph));Check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));for(unsigned repeat=0;repeat<3;++repeat){Check(cudaGraphLaunch(exec,stream));Check(cudaStreamSynchronize(stream));if(out.Get()!=actual)throw std::runtime_error("graph nondeterminism");}Check(cudaGraphExecDestroy(exec));Check(cudaGraphDestroy(graph));
  if(!scoped&&position==16383&&variant==1)for(bool candidate:{false,true}){
   Check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));for(unsigned repeat=0;repeat<64;++repeat)launch(candidate);Check(cudaStreamEndCapture(stream,&graph));Check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));Check(cudaGraphLaunch(exec,stream));Check(cudaStreamSynchronize(stream));cudaEvent_t begin{},end{};Check(cudaEventCreate(&begin));Check(cudaEventCreate(&end));for(unsigned sample=0;sample<3;++sample){Check(cudaEventRecord(begin,stream));Check(cudaGraphLaunch(exec,stream));Check(cudaEventRecord(end,stream));Check(cudaEventSynchronize(end));float ms=0;Check(cudaEventElapsedTime(&ms,begin,end));std::cout<<"timing candidate="<<candidate<<" sample="<<sample<<" us="<<ms*1000/64<<'\n';}Check(cudaEventDestroy(begin));Check(cudaEventDestroy(end));Check(cudaGraphExecDestroy(exec));Check(cudaGraphDestroy(graph));
  }
  q.Guard();k.Guard();v.Guard();sk.Guard();sv.Guard();workspace.Guard();reference.Guard();c.Guard();++cases;
 }
 Check(cudaStreamDestroy(stream));std::cout<<"PASS cases="<<cases<<" worst_relative_l2="<<worst_error<<" worst_max_abs="<<worst_abs<<" finite/causal/window/ring/speculative/guards/graph\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
