#pragma once
#include "gem16/status.h"
#include <cuda_runtime_api.h>
#include <cstdint>
namespace gem16::internal {
struct DecodeControl;
class TwelveBD3FastLocalAttention {
 public:
  TwelveBD3FastLocalAttention() = default;
  TwelveBD3FastLocalAttention(const TwelveBD3FastLocalAttention&) = delete;
  TwelveBD3FastLocalAttention& operator=(const TwelveBD3FastLocalAttention&) = delete;
  ~TwelveBD3FastLocalAttention();
  Status Initialize();
  Status Launch(const float*,const std::uint8_t*,const std::uint8_t*,
      const std::uint8_t*,const std::uint8_t*,const std::uint16_t*,
      const std::uint16_t*,float*,std::uint64_t,float*,const DecodeControl*,
      cudaStream_t) const;
  static constexpr std::uint64_t WorkspaceElements(){return 4U*4U*16U*(256U+1U);}
 private:
  cudaLibrary_t library_=nullptr;
  cudaKernel_t kernel_=nullptr;
};
}
