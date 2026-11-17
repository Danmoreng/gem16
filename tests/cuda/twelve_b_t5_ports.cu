// Exact operator comparison with the pinned model's resident weights.
// Build once with the graph-only parent and once with the T5 ports. No timing.
#include "cuda/engine/target_model.h"
#include "cuda/fp8/reference.h"
#include "cuda/fp8/sm120.h"
#include "cuda/layer/reference.h"
#include "cuda/nvfp4/reference.h"
#include "cuda/nvfp4/sm120.h"
#include "cuda/output_head.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using namespace gem16::internal;
void Check(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
void Check(const gem16::Status& s) {
  if (!s.ok()) throw std::runtime_error(s.message());
}
template<class T> struct Buffer {
  std::uint8_t* base = nullptr;
  T* p = nullptr;
  std::size_t n;
  explicit Buffer(std::size_t count): n(count * sizeof(T)) {
    Check(cudaMalloc(&base, n + 512)); p = reinterpret_cast<T*>(base + 256);
    Check(cudaMemset(base, 165, n + 512));
  }
  ~Buffer() { if (base) (void)cudaFree(base); }
  void Put(const std::vector<T>& data) {
    if (data.size() * sizeof(T) != n) throw std::runtime_error("input size");
    Check(cudaMemcpy(p, data.data(), n, cudaMemcpyHostToDevice));
  }
  void Guard() {
    std::array<std::uint8_t, 256> a{}, b{};
    Check(cudaMemcpy(a.data(), base, 256, cudaMemcpyDeviceToHost));
    Check(cudaMemcpy(b.data(), base + 256 + n, 256, cudaMemcpyDeviceToHost));
    if (!std::all_of(a.begin(),a.end(),[](auto x){return x==165;}) ||
        !std::all_of(b.begin(),b.end(),[](auto x){return x==165;}))
      throw std::runtime_error("device guard modified");
  }
  void Save(const std::filesystem::path& path) {
    std::vector<std::uint8_t> bytes(n);
    Check(cudaMemcpy(bytes.data(), p, n, cudaMemcpyDeviceToHost)); Guard();
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(bytes.data()), n);
    if (!f) throw std::runtime_error("output write");
  }
};
std::vector<float> Values(std::size_t n, unsigned variant, unsigned salt,
                          bool round = false) {
  std::vector<float> v(n);
  for (std::size_t i=0; i<n; ++i) {
    float x = variant == 0 ? 0.0F :
      static_cast<float>(int((i*37U+salt*19U)%257U)-128) * 0.019731F *
      (1.0F + float((i/3840U)%5U) * 0.125F);
    if (variant == 2) x *= 8.0F;
    v[i] = round ? static_cast<float>(__float2bfloat16_rn(x)) : x;
  }
  return v;
}
__global__ void Round(float* p, std::size_t n) {
  const auto i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<n)p[i]=static_cast<float>(__float2bfloat16_rn(p[i]));
}
}

int main(int argc, char** argv) {
  try {
    if(argc!=3)throw std::runtime_error("usage: operator-test target output-directory");
    const std::filesystem::path out=argv[2];
    if(!std::filesystem::create_directory(out))throw std::runtime_error("output exists");
    LoadedTargetModel model; Check(model.Load(argv[1]));
    #ifndef GEM16_PORT_ROWS
#define GEM16_PORT_ROWS 5
#endif
    constexpr std::size_t T=GEM16_PORT_ROWS,H=3840,I=15360;
    for(unsigned layer_index: {0U,15U,31U,47U}) {
      const auto& l=model.layers()[layer_index];
      for(unsigned variant=0;variant<3;++variant) {
        const auto stem=out/("layer-"+std::to_string(layer_index)+"-v"+std::to_string(variant));
        const auto path=[&](const char* part){return std::filesystem::path(stem.string()+"-"+part+".bin");};
        Buffer<float> input(T*H), projection(T*H), residual(T*H), hidden(T*H), post(T*H);
        Buffer<std::uint8_t> fp8(T*H), packed(T*H/2), scales(T*H/16);
        Buffer<float> fp8_scales(T), q(T*l.q.rows), k(T*l.k.rows), v(T*(l.global?1:l.v.rows));
        input.Put(Values(T*H,variant,layer_index));
        Check(LaunchFp8ReferenceTokenQuantizationBatch(input.p,fp8.p,fp8_scales.p,T,H,nullptr));
        Check(LaunchFp8Sm120GroupedQkvProjectionBatch(fp8.p,fp8_scales.p,
          l.q.weight,l.q.scales,q.p,l.q.rows,l.k.weight,l.k.scales,k.p,l.k.rows,
          l.global?nullptr:l.v.weight,l.global?nullptr:l.v.scales,l.global?nullptr:v.p,
          l.global?0:l.v.rows,T,H,nullptr));
        q.Save(path("q")); k.Save(path("k")); if(!l.global)v.Save(path("v"));
        Buffer<float> oi(T*l.o.contracting), os(T);
        Buffer<std::uint8_t> oa(T*l.o.contracting);
        oi.Put(Values(T*l.o.contracting,variant,layer_index+3));
        Check(LaunchFp8ReferenceTokenQuantizationBatch(oi.p,oa.p,os.p,T,l.o.contracting,nullptr));
        Check(LaunchFp8Sm120DirectProjectionBatch(oa.p,os.p,l.o.weight,l.o.scales,
          projection.p,T,H,l.o.contracting,nullptr));
        projection.Save(path("o"));
        // Raw/unrounded projection and BF16-container residual exercise the
        // same boundary with null post output and alias/non-alias hidden output.
        projection.Put(Values(T*H,variant,layer_index+7));
        residual.Put(Values(T*H,variant,layer_index+11,true));
        float* h=variant==2?residual.p:hidden.p;
        float* po=variant==1?post.p:nullptr;
#if GEM16_T5_CANDIDATE
        if constexpr (T==3) {
          Check(LaunchTwelveBPostAttentionNvfp4BoundaryT3(projection.p,l.post_attention_norm,
            residual.p,po,h,l.pre_mlp_norm,packed.p,scales.p,l.gate.input_divisor,nullptr));
        } else if constexpr (T==4) {
          Check(LaunchTwelveBPostAttentionNvfp4BoundaryT4(projection.p,l.post_attention_norm,
            residual.p,po,h,l.pre_mlp_norm,packed.p,scales.p,l.gate.input_divisor,nullptr));
        } else {
          Check(LaunchTwelveBPostAttentionNvfp4BoundaryT5(projection.p,l.post_attention_norm,
            residual.p,po,h,l.pre_mlp_norm,packed.p,scales.p,l.gate.input_divisor,nullptr));
        }
#else
        Round<<<(T*H+255)/256,256>>>(projection.p,T*H); Check(cudaGetLastError());
        Check(LaunchRmsNormResidualBf16(projection.p,l.post_attention_norm,residual.p,
          po,h,T,H,1.0e-6F,nullptr,nullptr));
        Check(LaunchRmsNormNvfp4ActivationQuantizationBatch(h,l.pre_mlp_norm,packed.p,
          scales.p,T,H,1.0e-6F,l.gate.input_divisor,nullptr));
#endif
        projection.Save(path("rounded-projection"));
        if(variant==2)residual.Save(path("hidden"));else hidden.Save(path("hidden"));
        if(po)post.Save(path("post"));
        packed.Save(path("packed"));scales.Save(path("scales"));
        Buffer<float> product(T*I);
        Check(LaunchNvfp4Sm120FusedGateUpBatch(packed.p,scales.p,l.gate.packed_weight,
          l.gate.scales,l.up.packed_weight,l.up.scales,nullptr,nullptr,product.p,T,I,H,
          l.gate.input_divisor,l.gate.weight_divisor,l.up.input_divisor,l.up.weight_divisor,nullptr));
        product.Save(path("product"));
        Buffer<std::uint8_t> da(T*I/2), ds(T*I/16);
        Buffer<std::uint16_t> down(T*H);
        Check(LaunchNvfp4ReferenceActivationQuantization(product.p,da.p,ds.p,T*I,l.down.input_divisor,nullptr));
        Check(LaunchNvfp4Sm120DirectProjectionBf16Batch(da.p,ds.p,l.down.packed_weight,
          l.down.scales,down.p,T,H,I,l.down.input_divisor,l.down.weight_divisor,nullptr));
        down.Save(path("down"));
        input.Guard();residual.Guard();fp8.Guard();fp8_scales.Guard();oi.Guard();oa.Guard();os.Guard();da.Guard();ds.Guard();
      }
    }
    Buffer<float> hidden(T*H), logits(T*262144U);
    Buffer<std::uint32_t> suppressed(7), selected(T);
    Buffer<OutputHeadCandidate> candidates(T*kOutputHeadCandidateBlocks);
    suppressed.Put({1,50,106,258882,258883,0,262143});
    for(unsigned variant=0;variant<3;++variant) {
      hidden.Put(Values(T*H,variant,17,true));
      Check(LaunchFusedOutputHeadBatchCandidates(model.embedding(),hidden.p,suppressed.p,
        7,T,candidates.p,logits.p,nullptr));
      Check(LaunchOutputHeadBatchArgmax(candidates.p,T,selected.p,nullptr));
      const auto stem=out/("head-v"+std::to_string(variant));
      logits.Save(stem.string()+"-logits.bin");
      candidates.Save(stem.string()+"-candidates.bin");
      selected.Save(stem.string()+"-selected.bin");
      hidden.Guard();suppressed.Guard();
    }
    std::cout<<"all T5 operator outputs and guards saved; actual pinned model weights\n";
    return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
