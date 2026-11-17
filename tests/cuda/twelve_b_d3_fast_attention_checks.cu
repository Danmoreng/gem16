// Experimental operator checks; exact reference remains a separate API/path.
#include "cuda/attention/twelve_b_d3_fast.h"
#include "cuda/attention/sm120.h"
#include "cuda/layer/reference.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace gem16::internal;
void Check(cudaError_t e) { if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e)); }
void Check(const gem16::Status& s) { if(!s.ok())throw std::runtime_error(s.message()); }
template<class T> struct Buffer {
  unsigned char* base=nullptr;T* p=nullptr;std::size_t n;
  explicit Buffer(std::size_t count):n(count){Check(cudaMalloc(&base,count*sizeof(T)+512));p=reinterpret_cast<T*>(base+256);Check(cudaMemset(base,165,count*sizeof(T)+512));}
  ~Buffer(){if(base)(void)cudaFree(base);}
  void Put(const std::vector<T>& x){if(x.size()!=n)throw std::runtime_error("input size");Check(cudaMemcpy(p,x.data(),n*sizeof(T),cudaMemcpyHostToDevice));}
  std::vector<T> Get(){std::vector<T>x(n);Check(cudaMemcpy(x.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));Guard();return x;}
  void Guard(){std::array<unsigned char,256>a,b;Check(cudaMemcpy(a.data(),base,256,cudaMemcpyDeviceToHost));Check(cudaMemcpy(b.data(),base+256+n*sizeof(T),256,cudaMemcpyDeviceToHost));for(unsigned i=0;i<256;++i)if(a[i]!=165||b[i]!=165)throw std::runtime_error("guard modified");}
};
std::uint16_t Bf(float x){return __bfloat16_as_ushort(__float2bfloat16_rn(x));}
float Fp(std::uint8_t x){__nv_fp8_e4m3 f;f.__x=x;return float(f);}
}

int main(int argc,char**argv){try{
  unsigned cap = 17519;
  bool scoped = false, single = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--scoped") == 0) scoped = true;
    else if (std::strcmp(argv[i], "--single") == 0) single = true;
    else if (std::strcmp(argv[i], "--capacity") == 0 && i + 1 < argc) {
      const auto requested = std::stoul(argv[++i]);
      if (requested < 4 || requested > 262144) throw std::runtime_error("invalid capacity");
      cap = static_cast<unsigned>(requested);
    } else throw std::runtime_error("unknown argument");
  }
  const unsigned rows=single?1U:4U,qsize=rows*16*512;
  TwelveBD3FastGlobalAttention fast;Check(fast.Initialize());
  Buffer<float>q(qsize),out(qsize),exact(qsize),workspace(TwelveBD3FastGlobalAttention::WorkspaceElements()),ref_workspace(DecodeAttentionFixedWorkspaceElements(cap,4));
  Buffer<std::uint8_t>k(cap*512),v(cap*512);Buffer<std::uint16_t>ks(1),vs(1);Buffer<DecodeControl>c(rows);
  std::vector<std::uint8_t>keys(k.n),values(v.n);
  for(std::size_t i=0;i<keys.size();++i){keys[i]=__nv_fp8_e4m3(float(int((i*37+11)%257)-128)/64.F).__x;values[i]=__nv_fp8_e4m3(float(int((i*19+7)%257)-128)/64.F + (i / 512 >= 17519 ? 2.F : 0.F)).__x;}
  k.Put(keys);v.Put(values);ks.Put({Bf(.125F)});vs.Put({Bf(.0625F)});
  cudaStream_t stream{};Check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
  std::vector<unsigned>positions=scoped?std::vector<unsigned>{0,cap-rows}:std::vector<unsigned>{0,510,511,512,1023,16383,17518,17519,cap-rows};
  std::erase_if(positions, [=](unsigned p){ return p > cap - rows; });
  unsigned cases=0;double worst_relative=0,worst_absolute=0;
  for(unsigned position:positions)for(unsigned variant=0;variant<2;++variant){
    std::vector<float>query(qsize);for(unsigned i=0;i<qsize;++i)query[i]=variant==0?0.F:float(__float2bfloat16_rn(float(int((i*13+3)%129)-64)/32.F));q.Put(query);
    std::vector<DecodeControl>control(rows);for(unsigned i=0;i<rows;++i)control[i].position=position+i;c.Put(control);
    Check(fast.Launch(q.p,k.p,v.p,ks.p,vs.p,workspace.p,workspace.n,out.p,c.p,cap,stream,rows));Check(cudaStreamSynchronize(stream));auto actual=out.Get();
    for(float x:actual)if(!std::isfinite(x))throw std::runtime_error("nonfinite attention");
    if(variant==0){
      std::array<double, 512> sums{};
      for (unsigned t = 0; t < position + rows; ++t) {
        for (unsigned dim = 0; dim < 512; ++dim) sums[dim] += Fp(values[t*512+dim]) * .0625;
        if (t < position) continue;
        for (unsigned head = 0; head < 16; ++head) for (unsigned dim = 0; dim < 512; ++dim) {
          float expected = static_cast<float>(sums[dim] / (t + 1));
          if (std::abs(actual[((t-position)*16+head)*512+dim]-expected) > 1.e-6F)
            throw std::runtime_error("uniform attention/causal-mask mismatch");
        }
      }
    }
    if(!scoped){
      if (single || cap <= 512) {
        for (unsigned row = 0; row < rows; ++row) {
          Check(LaunchOnlineAttentionDecodeFp8Sm120(q.p + row*16*512, k.p, v.p,
              ks.p, vs.p, ref_workspace.p, exact.p + row*16*512, c.p + row,
              16, 1, 512, cap, false, stream));
        }
      } else {
        Check(LaunchOnlineAttentionDecodeFp8GlobalFixedControlledSm120(q.p, k.p, v.p,
            ks.p, vs.p, ref_workspace.p, exact.p, c.p, 4, 1, cap, stream));
      }
      Check(cudaStreamSynchronize(stream));auto reference=exact.Get();double err=0,norm=0,absolute=0;
      for(unsigned i=0;i<qsize;++i){double d=actual[i]-reference[i];err+=d*d;norm+=double(reference[i])*reference[i];absolute=std::max(absolute,std::abs(d));}
      double relative=std::sqrt(err/std::max(norm,1.e-30));worst_relative=std::max(worst_relative,relative);worst_absolute=std::max(worst_absolute,absolute);
      if(variant==1&&relative>.15)throw std::runtime_error("moderate-query FP8 relative error exceeds predeclared15% bound");
      std::cout<<"case position="<<position<<" variant="<<variant<<" relative_l2="<<relative<<" max_abs="<<absolute<<'\n';
    }
    // Graph replay must work with an already loaded module and be deterministic.
    cudaGraph_t graph{};cudaGraphExec_t exec{};Check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    Check(fast.Launch(q.p,k.p,v.p,ks.p,vs.p,workspace.p,workspace.n,out.p,c.p,cap,stream,rows));Check(cudaStreamEndCapture(stream,&graph));Check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    for(unsigned repeat=0;repeat<3;++repeat){Check(cudaGraphLaunch(exec,stream));Check(cudaStreamSynchronize(stream));if(out.Get()!=actual)throw std::runtime_error("graph nondeterminism");}
    Check(cudaGraphExecDestroy(exec));Check(cudaGraphDestroy(graph));
    q.Guard();k.Guard();v.Guard();ks.Guard();vs.Guard();c.Guard();workspace.Guard();ref_workspace.Guard();++cases;
  }
  Check(cudaStreamDestroy(stream));std::cout<<"PASS rows="<<rows<<" cases="<<cases<<" worst_relative_l2="<<worst_relative<<" worst_max_abs="<<worst_absolute<<" finite/uniform/causal/guards/graph\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
