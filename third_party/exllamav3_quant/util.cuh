#pragma once

#include <cstdint>
#include <cuda_fp16.h>

union half2_uint32
{
    uint32_t as_uint32;
    half2 as_half2;
    __device__ half2_uint32(uint32_t value) : as_uint32(value) {}
    __device__ half2_uint32(half2 value) : as_half2(value) {}
};

union half_uint16
{
    uint16_t as_uint16;
    half as_half;
    __device__ half_uint16(uint16_t value) : as_uint16(value) {}
    __device__ half_uint16(half value) : as_half(value) {}
};
