#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

// Test-only ABI; production keeps its private matrix binding.
struct Fp8LabBinding {
  const std::uint8_t* weight;
  const std::uint16_t* weight_scales;
  float* output;
  std::uint64_t rows;
};
static_assert(sizeof(Fp8LabBinding)==32U && offsetof(Fp8LabBinding,rows)==24U);
using Fp8LabQkvCandidate = cudaError_t (*)(const std::uint8_t*,const float*,
    Fp8LabBinding,Fp8LabBinding,Fp8LabBinding,unsigned,std::uint64_t,unsigned,
    dim3,dim3,cudaStream_t);
using Fp8LabOCandidate = cudaError_t (*)(const std::uint8_t*,const float*,
    Fp8LabBinding,std::uint64_t,dim3,dim3,cudaStream_t);
