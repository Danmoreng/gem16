// Bitwise comparison of four-row cached boundaries with the exact parent.
#include "cuda/engine/target_model.h"
#include "cuda/fp8/reference.h"
#include "cuda/layer/reference.h"
#include "cuda/nvfp4/reference.h"
#include "cuda/norm/twelve_b_d3_fast.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace {
using namespace gem16::internal;
unsigned comparisons=0;
void Check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void Check(const gem16::Status&s){if(!s.ok())throw std::runtime_error(s.message());}
template<class T>struct Buffer{
 unsigned char*base=nullptr;T*p=nullptr;std::size_t n;
 explicit Buffer(std::size_t count):n(count){Check(cudaMalloc(&base,n*sizeof(T)+512));p=reinterpret_cast<T*>(base+256);Check(cudaMemset(base,165,n*sizeof(T)+512));}
 ~Buffer(){if(base)(void)cudaFree(base);}
 void Put(const std::vector<T>&v){if(v.size()!=n)throw std::runtime_error("size");Check(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice));}
 std::vector<T>Get(){std::vector<T>x(n);Check(cudaMemcpy(x.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));Guard();return x;}
 void Guard(){std::array<unsigned char,256>a,b;Check(cudaMemcpy(a.data(),base,256,cudaMemcpyDeviceToHost));Check(cudaMemcpy(b.data(),base+256+n*sizeof(T),256,cudaMemcpyDeviceToHost));for(unsigned i=0;i<256;++i)if(a[i]!=165||b[i]!=165)throw std::runtime_error("guard");}
 void Equal(Buffer&b,const char*name){auto a=Get(),v=b.Get();if(std::memcmp(a.data(),v.data(),n*sizeof(T)))throw std::runtime_error(name);++comparisons;}
};
std::vector<float>Values(std::size_t n,unsigned v,unsigned salt,bool round=false){std::vector<float>x(n);for(std::size_t i=0;i<n;++i){float f=v==0?0.F:float(int((i*37+salt*19)%257)-128)*.019731F*(v==2?8.F:1.F);x[i]=round?float(__float2bfloat16_rn(f)):f;}return x;}
__global__ void Round(float*x,std::size_t n){auto i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n)x[i]=float(__float2bfloat16_rn(x[i]));}
}
int main(int argc,char**argv){try{
 if(argc<2)throw std::runtime_error("usage: norm-checks pinned-model [--scoped]");const bool scoped=argc>2;LoadedTargetModel model;Check(model.Load(argv[1]));
 constexpr unsigned t=4,h=3840;for(unsigned li:scoped?std::vector<unsigned>{47}:std::vector<unsigned>{0,15,31,47})for(unsigned v=scoped?1:0;v<(scoped?2U:3U);++v){
  auto&l=model.layers()[li];const auto width=l.o.contracting;
  Buffer<float>x(t*h),residual(t*h),p(t*h),p2(t*h),hidden(t*h),hidden2(t*h),post(t*h),post2(t*h),scale(t),scale2(t),attention(t*width),attention2(t*width);
  Buffer<std::uint8_t>fp8(t*h),fp82(t*h),qa(t*width),qa2(t*width),packed(t*h/2),packed2(t*h/2),s(t*h/16),s2(t*h/16);Buffer<std::uint16_t>bf(t*h);
  x.Put(Values(t*h,v,li));Check(LaunchRmsNormFp8TokenQuantizationBatch(x.p,l.input_norm,fp8.p,scale.p,t,h,1.e-6F,nullptr));Check(LaunchTwelveBD3FastInputNormFp8(x.p,l.input_norm,fp82.p,scale2.p,nullptr));fp8.Equal(fp82,"input norm FP8 mismatch");scale.Equal(scale2,"input scale mismatch");
  attention.Put(Values(t*width,v,li+2));attention2.Put(Values(t*width,v,li+2));Round<<<(t*width+255)/256,256>>>(attention.p,t*width);Check(cudaGetLastError());Check(LaunchFp8ReferenceTokenQuantizationBatch(attention.p,qa.p,scale.p,t,width,nullptr));Check(LaunchTwelveBD3FastAttentionQuantize(attention2.p,qa2.p,scale2.p,width,nullptr));qa.Equal(qa2,"attention payload mismatch");scale.Equal(scale2,"attention scale mismatch");
  p.Put(Values(t*h,v,li+3));p2.Put(Values(t*h,v,li+3));auto rv=Values(t*h,v,li+4,true);residual.Put(rv);float*po=v==1?nullptr:post.p;float*dest=v==2?residual.p:hidden.p;
  Check(LaunchTwelveBPostAttentionNvfp4BoundaryT4(p.p,l.post_attention_norm,residual.p,po,dest,l.pre_mlp_norm,packed.p,s.p,l.gate.input_divisor,nullptr));auto expected=v==2?residual.Get():hidden.Get();residual.Put(rv);
  Check(LaunchTwelveBD3FastPostAttention(p2.p,l.post_attention_norm,residual.p,v==1?nullptr:post2.p,v==2?residual.p:hidden2.p,l.pre_mlp_norm,packed2.p,s2.p,l.gate.input_divisor,nullptr));auto actual=v==2?residual.Get():hidden2.Get();if(std::memcmp(expected.data(),actual.data(),t*h*sizeof(float)))throw std::runtime_error("hidden alias mismatch");++comparisons;
  p.Equal(p2,"projection boundary mismatch");packed.Equal(packed2,"FP4 payload mismatch");s.Equal(s2,"FP4 scales mismatch");if(po)post.Equal(post2,"post-attention norm mismatch");
  auto values=Values(t*h,v,li+5);std::vector<std::uint16_t>physical(t*h);for(unsigned i=0;i<t*h;++i)physical[i]=__bfloat16_as_ushort(__float2bfloat16_rn(values[i]));bf.Put(physical);residual.Put(rv);
  Check(LaunchRmsNormResidualBf16Input(bf.p,l.post_mlp_norm,residual.p,po,hidden.p,t,h,1.e-6F,l.layer_scalar,nullptr));Check(LaunchTwelveBD3FastPostMlp(bf.p,l.post_mlp_norm,residual.p,v==1?nullptr:post2.p,hidden2.p,l.layer_scalar,nullptr));hidden.Equal(hidden2,"post-MLP hidden mismatch");if(po)post.Equal(post2,"post-MLP normalized mismatch");
  x.Guard();residual.Guard();attention.Guard();attention2.Guard();bf.Guard();
 }
 Check(cudaDeviceSynchronize());std::cout<<"PASS "<<comparisons<<" byte-exact actual-weight boundary comparisons, zeros/general/large, alias/optional-output, guards\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
