// Changed-math boundary checks, actual norm weights, and diagnostic graph timing.
#include "cuda/engine/target_model.h"
#include "cuda/norm/twelve_b_d3_fast.h"
#include "cuda/norm/twelve_b_d3_paired.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
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
std::vector<float> Values(unsigned n,unsigned variant,unsigned salt){std::vector<float>v(n);for(unsigned i=0;i<n;++i)v[i]=variant==0?0.F:float(__float2bfloat16_rn(float(int((i*37+salt*19)%257)-128)*.019731F*(variant==2?8.F:1.F)));return v;}
double Error(const std::vector<float>&a,const std::vector<float>&b){double err=0,norm=0;for(unsigned i=0;i<a.size();++i){if(!std::isfinite(b[i]))throw std::runtime_error("nonfinite");err+=std::pow(double(a[i])-b[i],2);norm+=double(a[i])*a[i];}return std::sqrt(err/std::max(norm,1.e-30));}
std::vector<float> Dequant(Buffer<std::uint8_t>&a,Buffer<std::uint8_t>&s,float divisor){auto x=a.Get(),scale=s.Get();std::vector<float>v(x.size()*2);for(unsigned i=0;i<v.size();++i){__nv_fp4_e2m1 f;f.__x=(x[i/2]>>((i&1)*4))&15;__nv_fp8_e4m3 z;z.__x=scale[i/16];v[i]=float(f)*float(z)/divisor;}return v;}
}
int main(int argc,char**argv){try{
 if(argc<2)throw std::runtime_error("usage: paired-checks pinned-model [--scoped]");
 const bool scoped=argc>2;LoadedTargetModel model;Check(model.Load(argv[1]));
 constexpr unsigned count=4*3840;double worst_hidden=0,worst_quant=0;unsigned cases=0;
 for(unsigned li:scoped?std::vector<unsigned>{47}:std::vector<unsigned>{0,15,31,47})for(unsigned variant=0;variant<3;++variant){
  auto&l=model.layers()[li];Buffer<float>p(count),p2(count),r(count),h(count),h2(count),post(count),post2(count);Buffer<std::uint8_t>a(count/2),a2(count/2),s(count/16),s2(count/16);
  auto projection=Values(count,variant,li),residual=Values(count,variant,li+3);p.Put(projection);p2.Put(projection);r.Put(residual);
  Check(LaunchTwelveBD3FastPostAttention(p.p,l.post_attention_norm,r.p,variant==1?nullptr:post.p,variant==2?r.p:h.p,l.pre_mlp_norm,a.p,s.p,l.gate.input_divisor,nullptr));auto expected=variant==2?r.Get():h.Get();r.Put(residual);
  Check(LaunchTwelveBD3PairedPostAttention(p2.p,l.post_attention_norm,r.p,variant==1?nullptr:post2.p,variant==2?r.p:h2.p,l.pre_mlp_norm,a2.p,s2.p,l.gate.input_divisor,nullptr));auto actual=variant==2?r.Get():h2.Get();
  double eh=Error(expected,actual),eq=Error(Dequant(a,s,l.gate.input_divisor),Dequant(a2,s2,l.gate.input_divisor));worst_hidden=std::max(worst_hidden,eh);worst_quant=std::max(worst_quant,eq);
  // Predeclared changed-tree bounds: hidden0.05%, decoded FP4 output1% L2.
  if(eh>5.e-4||eq>.01)throw std::runtime_error("paired boundary exceeds predeclared error bounds");
  if(p.Get()!=p2.Get())throw std::runtime_error("BF16 projection changed");
  if(variant!=1&&Error(post.Get(),post2.Get())>5.e-4)throw std::runtime_error("post norm error");
  if(variant==0&&(expected!=actual||a.Get()!=a2.Get()||s.Get()!=s2.Get()))throw std::runtime_error("zero boundary must remain exact");
  p.Guard();p2.Guard();r.Guard();h.Guard();h2.Guard();post.Guard();post2.Guard();a.Guard();a2.Guard();s.Guard();s2.Guard();++cases;
  // Deterministic replay and diagnostic timing use immutable residual inputs.
  if(variant==1){
   r.Put(residual);p.Put(projection);p2.Put(projection);
   for(unsigned candidate=0;candidate<2;++candidate){
    cudaStream_t stream{};cudaGraph_t graph{};cudaGraphExec_t exec{};cudaEvent_t begin{},end{};Check(cudaStreamCreate(&stream));Check(cudaEventCreate(&begin));Check(cudaEventCreate(&end));
    Check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    for(unsigned repeat=0;repeat<64;++repeat){auto launch=candidate==0?LaunchTwelveBD3FastPostAttention:LaunchTwelveBD3PairedPostAttention;Check(launch(candidate==0?p.p:p2.p,l.post_attention_norm,r.p,nullptr,candidate==0?h.p:h2.p,l.pre_mlp_norm,candidate==0?a.p:a2.p,candidate==0?s.p:s2.p,l.gate.input_divisor,stream));}
    Check(cudaStreamEndCapture(stream,&graph));Check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));Check(cudaGraphLaunch(exec,stream));Check(cudaStreamSynchronize(stream));auto stable=candidate==0?h.Get():h2.Get();
    for(unsigned sample=0;sample<3;++sample){Check(cudaEventRecord(begin,stream));Check(cudaGraphLaunch(exec,stream));Check(cudaEventRecord(end,stream));Check(cudaEventSynchronize(end));float ms=0;Check(cudaEventElapsedTime(&ms,begin,end));if(stable!=(candidate==0?h.Get():h2.Get()))throw std::runtime_error("graph replay differs");std::cout<<"timing layer="<<li<<" candidate="<<candidate<<" sample="<<sample<<" us="<<ms*1000/64<<'\n';}
    Check(cudaEventDestroy(begin));Check(cudaEventDestroy(end));Check(cudaGraphExecDestroy(exec));Check(cudaGraphDestroy(graph));Check(cudaStreamDestroy(stream));
   }
  }
 }
 Check(cudaDeviceSynchronize());std::cout<<"PASS cases="<<cases<<" worst_hidden_relative_l2="<<worst_hidden<<" worst_quant_relative_l2="<<worst_quant<<" finite/guards/zeros/alias/optional/graph\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
