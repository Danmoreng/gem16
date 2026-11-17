#include "cuda/norm/twelve_b_ordinary.h"
#include "cuda/fp8/reference.h"
#include "cuda/nvfp4/reference.h"
#include "cuda/layer/reference.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>
using namespace gem16::internal;
void Check(cudaError_t e) { if(e!=cudaSuccess){std::cerr<<cudaGetErrorString(e)<<'\n';std::exit(1);} }
void Check(gem16::Status s) {if(!s.ok()){std::cerr<<s.message()<<"\n";std::exit(1);}}
struct Buffer {
 unsigned char* base=nullptr; std::size_t bytes;
 explicit Buffer(std::size_t n):bytes(n) {Check(cudaMalloc(&base,n+512));Check(cudaMemset(base,0xa5,n+512));}
 ~Buffer(){cudaFree(base);}
 template<class T> T* ptr(){return reinterpret_cast<T*>(base+256);}
 void put(const void* p){Check(cudaMemcpy(base+256,p,bytes,cudaMemcpyHostToDevice));}
 std::vector<unsigned char> get(){std::vector<unsigned char> v(bytes);Check(cudaMemcpy(v.data(),base+256,bytes,cudaMemcpyDeviceToHost));return v;}
 void guards(){std::array<unsigned char,256> a,b;Check(cudaMemcpy(a.data(),base,256,cudaMemcpyDeviceToHost));Check(cudaMemcpy(b.data(),base+256+bytes,256,cudaMemcpyDeviceToHost));for(auto x:a)if(x!=0xa5)std::exit(2);for(auto x:b)if(x!=0xa5)std::exit(2);}
};
__global__ void Round(float* p,unsigned n){unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)p[i]=float(__float2bfloat16_rn(p[i]));}
void Equal(Buffer& a,Buffer& b,const char* label,int edge){auto x=a.get(),y=b.get();if(x!=y){auto i=std::mismatch(x.begin(),x.end(),y.begin()).first-x.begin();std::cerr<<label<<" edge="<<edge<<" byte="<<i<<'\n';std::exit(3);}}
int main(){
 cudaStream_t stream;Check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
 std::mt19937 rng(171117);std::uniform_real_distribution<float> uniform(-10,10);
 std::vector<float> input(8192),residual(3840);std::vector<std::uint16_t> weights(3840),pre(3840);
 for(unsigned i=0;i<3840;++i){weights[i]=__bfloat16_as_ushort(__float2bfloat16_rn(uniform(rng)));pre[i]=__bfloat16_as_ushort(__float2bfloat16_rn(uniform(rng)));}
 Buffer x(8192*4),y(8192*4),r(3840*4),rr(3840*4),w(7680),pw(7680),p(3840*4),q(3840*4),h(3840*4),hh(3840*4),fp(8192),fq(8192),s(4),ss(4),pack(1920),pack2(1920),sc(240),sc2(240);
 w.put(weights.data());pw.put(pre.data());unsigned checks=0;
 for(int edge=0;edge<18;++edge){
  for(unsigned i=0;i<input.size();++i){float v=uniform(rng);if(edge==0)v=0;if(edge==1)v=(i%2?-1:1)*1e-30F;if(edge==2)v=(i%2?-1:1)*1e10F;if(edge==3)v=(i%2?-1:1)*1.00390625F;if(edge==4)v=(i%2?-1:1)*0.99609375F;input[i]=v;}
  for(unsigned i=0;i<3840;++i)residual[i]=edge%3==0?-input[i]:uniform(rng);
  x.put(input.data());y.put(input.data());r.put(residual.data());rr.put(residual.data());
  auto* weight=edge%2?w.ptr<std::uint16_t>():nullptr;
  Check(LaunchRmsNormFp8TokenQuantizationBatch(x.ptr<float>(),weight,fp.ptr<std::uint8_t>(),s.ptr<float>(),1,3840,1e-6F,stream));
  Check(LaunchTwelveBOrdinaryInputNormFp8(y.ptr<float>(),weight,fq.ptr<std::uint8_t>(),ss.ptr<float>(),stream));Check(cudaStreamSynchronize(stream));
  // Only the first3840 bytes are written; retain equal initialization elsewhere.
  Equal(fp,fq,"input FP8",edge);Equal(s,ss,"input scale",edge);checks+=2;
  for(unsigned n:{4096U,8192U}){
   x.put(input.data());y.put(input.data());Round<<<(n+255)/256,256,0,stream>>>(x.ptr<float>(),n);
   Check(LaunchFp8ReferenceTokenQuantization(x.ptr<float>(),fp.ptr<std::uint8_t>(),s.ptr<float>(),n,stream));
   Check(LaunchTwelveBOrdinaryAttentionQuantize(y.ptr<float>(),fq.ptr<std::uint8_t>(),ss.ptr<float>(),n,stream));Check(cudaStreamSynchronize(stream));Equal(fp,fq,"attention FP8",edge);Equal(s,ss,"attention scale",edge);checks+=2;
  }
  for(bool alias:{false,true})for(bool null_post:{false,true}){
   x.put(input.data());y.put(input.data());r.put(residual.data());rr.put(residual.data());
   auto* out1=alias?r.ptr<float>():h.ptr<float>();auto* out2=alias?rr.ptr<float>():hh.ptr<float>();
   Check(LaunchTwelveBPostAttentionNvfp4Boundary(x.ptr<float>(),w.ptr<std::uint16_t>(),r.ptr<float>(),p.ptr<float>(),out1,pw.ptr<std::uint16_t>(),pack.ptr<std::uint8_t>(),sc.ptr<std::uint8_t>(),0.5F,stream));
   Check(LaunchTwelveBOrdinaryPostAttention(y.ptr<float>(),w.ptr<std::uint16_t>(),rr.ptr<float>(),null_post?nullptr:q.ptr<float>(),out2,pw.ptr<std::uint16_t>(),pack2.ptr<std::uint8_t>(),sc2.ptr<std::uint8_t>(),0.5F,stream));Check(cudaStreamSynchronize(stream));
   Equal(x,y,"rounded projection",edge);if(!null_post)Equal(p,q,"post-attention norm",edge);if(alias)Equal(r,rr,"alias hidden",edge);else Equal(h,hh,"hidden",edge);Equal(pack,pack2,"FP4",edge);Equal(sc,sc2,"FP4 scale",edge);checks+=5;
   // Post-MLP input is already BF16-rounded by the existing graph node.
   x.put(input.data());y.put(input.data());r.put(residual.data());rr.put(residual.data());Round<<<15,256,0,stream>>>(x.ptr<float>(),3840);Round<<<15,256,0,stream>>>(y.ptr<float>(),3840);
   auto* scalar=edge%2?pw.ptr<std::uint16_t>():nullptr;
   Check(LaunchRmsNormResidualBf16(x.ptr<float>(),weight,r.ptr<float>(),null_post?nullptr:p.ptr<float>(),out1,1,3840,1e-6F,scalar,stream));
   Check(LaunchTwelveBOrdinaryPostMlp(y.ptr<float>(),weight,rr.ptr<float>(),null_post?nullptr:q.ptr<float>(),out2,scalar,stream));Check(cudaStreamSynchronize(stream));if(!null_post)Equal(p,q,"post-MLP norm",edge);if(alias)Equal(r,rr,"post-MLP alias",edge);else Equal(h,hh,"post-MLP",edge);checks+=2;
  }
  for(Buffer* b:{&x,&y,&r,&rr,&w,&pw,&p,&q,&h,&hh,&fp,&fq,&s,&ss,&pack,&pack2,&sc,&sc2})b->guards();
 }
 Check(cudaStreamDestroy(stream));std::cout<<"PASS "<<checks<<" bit-exact boundary/intermediate/alias/scale checks; guards intact\n";
}
