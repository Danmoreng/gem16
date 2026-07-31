#include "cuda/attention/sm120.h"

#include "cuda/layer/reference.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstdint>
#include <limits>
#include <string>
#include <utility>

// The BF16 MMA fragment mapping, XOR shared-memory swizzle, and online-softmax
// schedule in this file are adapted from NInfer's Apache-2.0
// gqa_attention_prefill_bf16.cuh. The staging and addressing are rewritten for
// gem16's float Q, physical E4M3 K/V, current-chunk source, and circular cache.

namespace gem16::internal {
namespace {

constexpr unsigned kFullWarpMask = 0xffffffffU;

constexpr int kDecodeThreads = 256;
constexpr int kDecodeWarps = kDecodeThreads / 32;
constexpr int kDecodeQueryHeads = 16;
constexpr int kDecodeLocalKvHeads = 8;
constexpr int kDecodeLocalHeadDimension = 256;
constexpr int kDecodeLocalGroup = 2;
constexpr int kDecodeLocalChunk = 256;
constexpr int kDecodeGlobalKvHeads = 1;
constexpr int kDecodeGlobalHeadDimension = 512;
constexpr int kDecodeGlobalGroup = 4;
constexpr int kDecodeGlobalChunk = 512;
constexpr int kDecodeGlobalGqaChunk = 512;
constexpr int kDecodeGlobalGqaTileTokens = 16;
constexpr int kDecodeGlobalGqaHeadsPerWarp = 2;
constexpr std::uint64_t kDecodeGqaGlobalContext = 16384U;
// Four-wide physical E4M3 loads pay off only once global-cache traffic
// dominates the additional registers. Shorter capacity tiers retain the
// scalar reduction order and lower register footprint.
constexpr std::uint64_t kDecodeVectorizedGlobalContext = 65536U;

Status Invalid(std::string message) {
  return Status(StatusCode::kInvalidArgument, std::move(message));
}

Status CudaFailure(const char* operation, cudaError_t error) {
  return Status(StatusCode::kInternal,
                std::string(operation) + ": " + cudaGetErrorName(error) +
                    ": " + cudaGetErrorString(error));
}

__device__ __forceinline__ unsigned SharedAddress(const void* pointer) {
  return static_cast<unsigned>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void CopyAsync16(
    void* shared_destination, const void* global_source, int source_bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
               :
               : "r"(SharedAddress(shared_destination)), "l"(global_source),
                 "r"(source_bytes));
#else
  (void)shared_destination;
  (void)global_source;
  (void)source_bytes;
#endif
}

__device__ __forceinline__ void CommitAsyncCopies() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  asm volatile("cp.async.commit_group;\n");
#endif
}

__device__ __forceinline__ void WaitForAsyncCopies() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  asm volatile("cp.async.wait_group 0;\n");
#endif
}

__device__ __forceinline__ int Swizzle(int row, int column) {
  return (((column >> 3) ^ (row & 7)) << 3) | (column & 7);
}

__device__ __forceinline__ unsigned SwizzledAddress(
    unsigned lane_base, unsigned contracting_offset, unsigned address_select,
    unsigned row_select) {
  return lane_base + ((contracting_offset | address_select) ^ row_select);
}

__device__ __forceinline__ void LoadMatrixX4(
    unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3,
    unsigned address) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
               "{%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(address));
}

__device__ __forceinline__ void LoadMatrixX4Transposed(
    unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3,
    unsigned address) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
               "{%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(address));
}

__device__ __forceinline__ void MmaBf16(
    float& c0, float& c1, float& c2, float& c3, unsigned a0, unsigned a1,
    unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
               "{%0,%1,%2,%3};\n"
               : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0),
                 "r"(b1));
}

template <int Width>
__device__ __forceinline__ float WarpMaximum(float value) {
#pragma unroll
  for (int offset = Width / 2; offset > 0; offset >>= 1) {
    value = fmaxf(value,
                  __shfl_xor_sync(kFullWarpMask, value, offset, Width));
  }
  return value;
}

template <int Width>
__device__ __forceinline__ float WarpSum(float value) {
#pragma unroll
  for (int offset = Width / 2; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(kFullWarpMask, value, offset, Width);
  }
  return value;
}

__device__ __forceinline__ std::uint32_t PackBf16x2(float low, float high) {
  std::uint32_t output;
  const std::uint32_t low_bits = __float_as_uint(low);
  const std::uint32_t high_bits = __float_as_uint(high);
  asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n"
               : "=r"(output)
               : "r"(high_bits), "r"(low_bits));
  return output;
}

__device__ __forceinline__ float DecodeFp8(std::uint8_t bits,
                                           float scale) {
  __nv_fp8_e4m3 quantized;
  quantized.__x = bits;
  return static_cast<float>(quantized) * scale;
}

// Decode one aligned physical cache word without a scalar byte-load and
// conversion instruction for every element.
__device__ __forceinline__ float4 DecodeFp8x4(std::uint32_t bits,
                                              float scale) {
  __nv_fp8x4_e4m3 quantized;
  quantized.__x = bits;
  float4 values = static_cast<float4>(quantized);
  values.x *= scale;
  values.y *= scale;
  values.z *= scale;
  values.w *= scale;
  return values;
}

__device__ __forceinline__ float DecodeBlockMaximum(float value,
                                                     float* warp_values) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(kFullWarpMask, value, offset));
  }
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  if (lane == 0) warp_values[warp] = value;
  __syncthreads();
  if (warp == 0) {
    value = lane < kDecodeWarps ? warp_values[lane] : -CUDART_INF_F;
    for (int offset = 16; offset > 0; offset >>= 1) {
      value = fmaxf(value, __shfl_down_sync(kFullWarpMask, value, offset));
    }
    if (lane == 0) warp_values[0] = value;
  }
  __syncthreads();
  const float block_value = warp_values[0];
  __syncthreads();
  return block_value;
}

__device__ __forceinline__ float DecodeBlockSum(float value,
                                                 float* warp_values) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(kFullWarpMask, value, offset);
  }
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  if (lane == 0) warp_values[warp] = value;
  __syncthreads();
  if (warp == 0) {
    value = lane < kDecodeWarps ? warp_values[lane] : 0.0F;
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(kFullWarpMask, value, offset);
    }
    if (lane == 0) warp_values[0] = value;
  }
  __syncthreads();
  const float block_value = warp_values[0];
  __syncthreads();
  return block_value;
}

template <int kHeadDimensionValue, int kKvHeadsValue,
          int kQueriesPerKv, int kQueryGroup, int kChunk, bool kDirect,
          bool kVectorizedGlobal>
__launch_bounds__(kDecodeThreads, 1) __global__
void SplitOnlineDecodeAttentionFp8Kernel(
    const float* __restrict__ query,
    const std::uint8_t* __restrict__ key_cache,
    const std::uint8_t* __restrict__ value_cache,
    const std::uint16_t* __restrict__ key_scale_bf16,
    const std::uint16_t* __restrict__ value_scale_bf16,
    float* __restrict__ partial_output, float* __restrict__ partial_lse,
    float* __restrict__ output, const DecodeControl* __restrict__ control,
    std::uint64_t cache_capacity, bool sliding, int max_splits) {
  static_assert(kQueriesPerKv % kQueryGroup == 0);
  static_assert(!kVectorizedGlobal ||
                (kHeadDimensionValue == kDecodeGlobalHeadDimension &&
                 kKvHeadsValue == kDecodeGlobalKvHeads));
  constexpr int kGroupsPerKv = kQueriesPerKv / kQueryGroup;
  constexpr int kQueryGroups = kKvHeadsValue * kGroupsPerKv;
  constexpr int kQueryHeadsValue = kKvHeadsValue * kQueriesPerKv;
  __shared__ float scores[kQueryGroup * kChunk];
  __shared__ float reduction[kDecodeWarps];

  const int linear_block = static_cast<int>(blockIdx.x);
  const int split = linear_block / kQueryGroups;
  const int query_group = linear_block - split * kQueryGroups;
  if (split >= max_splits) return;
  const std::uint64_t token_count_unbounded = control->position + 1U;
  const std::uint64_t tokens =
      sliding && token_count_unbounded > cache_capacity
          ? cache_capacity
          : token_count_unbounded;
  const std::uint64_t split_start =
      static_cast<std::uint64_t>(split) * kChunk;
  if (split_start >= tokens) return;
  const int split_tokens = static_cast<int>(
      min(static_cast<std::uint64_t>(kChunk), tokens - split_start));
  const std::uint64_t first_slot =
      sliding && token_count_unbounded > cache_capacity
          ? token_count_unbounded % cache_capacity
          : 0U;
  const int kv_head = query_group / kGroupsPerKv;
  const int subgroup = query_group - kv_head * kGroupsPerKv;
  const int query_head_base =
      kv_head * kQueriesPerKv + subgroup * kQueryGroup;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const float key_scale =
      static_cast<float>(__ushort_as_bfloat16(key_scale_bf16[0]));
  const float value_scale =
      static_cast<float>(__ushort_as_bfloat16(value_scale_bf16[0]));

  for (int token = warp; token < split_tokens; token += kDecodeWarps) {
    const std::uint64_t unwrapped_slot =
        first_slot + split_start + static_cast<std::uint64_t>(token);
    const std::uint64_t cache_slot =
        unwrapped_slot < cache_capacity ? unwrapped_slot
                                        : unwrapped_slot - cache_capacity;
    const std::uint8_t* key =
        key_cache +
        (cache_slot * kKvHeadsValue + static_cast<std::uint64_t>(kv_head)) *
            kHeadDimensionValue;
    float score[kQueryGroup] = {};
    if constexpr (kVectorizedGlobal) {
      // The long-context tier assigns four adjacent dimensions to each lane,
      // reducing physical K loads and E4M3 conversions fourfold.
      for (int dimension = lane * 4; dimension < kHeadDimensionValue;
           dimension += 32 * 4) {
        const std::uint32_t packed =
            *reinterpret_cast<const std::uint32_t*>(key + dimension);
        const float4 key_values = DecodeFp8x4(packed, key_scale);
#pragma unroll
        for (int group = 0; group < kQueryGroup; ++group) {
          const float* query_values =
              query + (query_head_base + group) * kHeadDimensionValue +
              dimension;
          score[group] = fmaf(query_values[0], key_values.x, score[group]);
          score[group] = fmaf(query_values[1], key_values.y, score[group]);
          score[group] = fmaf(query_values[2], key_values.z, score[group]);
          score[group] = fmaf(query_values[3], key_values.w, score[group]);
        }
      }
    } else {
      for (int dimension = lane; dimension < kHeadDimensionValue;
           dimension += 32) {
        const float key_value = DecodeFp8(key[dimension], key_scale);
#pragma unroll
        for (int group = 0; group < kQueryGroup; ++group) {
          score[group] = fmaf(
              query[(query_head_base + group) * kHeadDimensionValue +
                    dimension],
              key_value, score[group]);
        }
      }
    }
#pragma unroll
    for (int group = 0; group < kQueryGroup; ++group) {
      for (int offset = 16; offset > 0; offset >>= 1) {
        score[group] +=
            __shfl_down_sync(kFullWarpMask, score[group], offset);
      }
      if (lane == 0) scores[group * kChunk + token] = score[group];
    }
  }
  __syncthreads();

  float inverse_sum[kQueryGroup];
#pragma unroll
  for (int group = 0; group < kQueryGroup; ++group) {
    float local_maximum = -CUDART_INF_F;
    for (int token = static_cast<int>(threadIdx.x); token < split_tokens;
         token += kDecodeThreads) {
      local_maximum =
          fmaxf(local_maximum, scores[group * kChunk + token]);
    }
    const float maximum = DecodeBlockMaximum(local_maximum, reduction);
    float local_sum = 0.0F;
    for (int token = static_cast<int>(threadIdx.x); token < split_tokens;
         token += kDecodeThreads) {
      const float probability =
          expf(scores[group * kChunk + token] - maximum);
      scores[group * kChunk + token] = probability;
      local_sum += probability;
    }
    const float denominator = DecodeBlockSum(local_sum, reduction);
    inverse_sum[group] = denominator > 0.0F ? __frcp_rn(denominator) : 0.0F;
    if constexpr (!kDirect) {
      if (threadIdx.x == 0) {
        partial_lse[split * kQueryHeadsValue + query_head_base + group] =
            maximum + logf(denominator);
      }
    }
  }

  if constexpr (kVectorizedGlobal) {
    // Keep each output dimension's token accumulation order while processing
    // four contiguous V dimensions from one aligned cache word.
    for (int dimension = static_cast<int>(threadIdx.x) * 4;
         dimension < kHeadDimensionValue; dimension += kDecodeThreads * 4) {
      float accumulator[kQueryGroup][4] = {};
      for (int token = 0; token < split_tokens; ++token) {
        const std::uint64_t unwrapped_slot =
            first_slot + split_start + static_cast<std::uint64_t>(token);
        const std::uint64_t cache_slot =
            unwrapped_slot < cache_capacity ? unwrapped_slot
                                            : unwrapped_slot - cache_capacity;
        const std::uint64_t value_offset =
            (cache_slot * kKvHeadsValue +
             static_cast<std::uint64_t>(kv_head)) *
                kHeadDimensionValue +
            dimension;
        const std::uint32_t packed =
            *reinterpret_cast<const std::uint32_t*>(value_cache + value_offset);
        const float4 values = DecodeFp8x4(packed, value_scale);
#pragma unroll
        for (int group = 0; group < kQueryGroup; ++group) {
          const float probability = scores[group * kChunk + token];
          accumulator[group][0] =
              fmaf(probability, values.x, accumulator[group][0]);
          accumulator[group][1] =
              fmaf(probability, values.y, accumulator[group][1]);
          accumulator[group][2] =
              fmaf(probability, values.z, accumulator[group][2]);
          accumulator[group][3] =
              fmaf(probability, values.w, accumulator[group][3]);
        }
      }
#pragma unroll
      for (int group = 0; group < kQueryGroup; ++group) {
        const int query_head = query_head_base + group;
        const std::uint64_t output_base =
            static_cast<std::uint64_t>(query_head) * kHeadDimensionValue +
            dimension;
#pragma unroll
        for (int element = 0; element < 4; ++element) {
          const float normalized =
              accumulator[group][element] * inverse_sum[group];
          if constexpr (kDirect) {
            output[output_base + element] = normalized;
          } else {
            partial_output
                [(static_cast<std::uint64_t>(split) * kQueryHeadsValue +
                  query_head) *
                     kHeadDimensionValue +
                 dimension + element] = normalized;
          }
        }
      }
    }
  } else {
    for (int dimension = static_cast<int>(threadIdx.x);
         dimension < kHeadDimensionValue; dimension += kDecodeThreads) {
      float accumulator[kQueryGroup] = {};
      for (int token = 0; token < split_tokens; ++token) {
        const std::uint64_t unwrapped_slot =
            first_slot + split_start + static_cast<std::uint64_t>(token);
        const std::uint64_t cache_slot =
            unwrapped_slot < cache_capacity ? unwrapped_slot
                                            : unwrapped_slot - cache_capacity;
        const std::uint64_t value_offset =
            (cache_slot * kKvHeadsValue +
             static_cast<std::uint64_t>(kv_head)) *
                kHeadDimensionValue +
            dimension;
        const float value = DecodeFp8(value_cache[value_offset], value_scale);
#pragma unroll
        for (int group = 0; group < kQueryGroup; ++group) {
          accumulator[group] =
              fmaf(scores[group * kChunk + token], value, accumulator[group]);
        }
      }
#pragma unroll
      for (int group = 0; group < kQueryGroup; ++group) {
        const int query_head = query_head_base + group;
        const std::uint64_t output_index =
            static_cast<std::uint64_t>(query_head) * kHeadDimensionValue +
            dimension;
        const float normalized = accumulator[group] * inverse_sum[group];
        if constexpr (kDirect) {
          output[output_index] = normalized;
        } else {
          partial_output
              [(static_cast<std::uint64_t>(split) * kQueryHeadsValue +
                query_head) *
                   kHeadDimensionValue +
               dimension] = normalized;
        }
      }
    }
  }
}

// Long-context global attention assigns two query heads to each warp and
// stages each physical K/V tile once for all 16 heads. Scalar and FP8x4
// specializations retain their respective parent dimension, softmax, and
// per-output accumulation orders; only the K/V source moves to shared memory.
template <bool kDirect, bool kBatch = false, bool kVectorized = true>
__launch_bounds__(kDecodeThreads, 1) __global__
void SplitOnlineDecodeAttentionFp8GlobalGqaKernel(
    const float* __restrict__ query,
    const std::uint8_t* __restrict__ key_cache,
    const std::uint8_t* __restrict__ value_cache,
    const std::uint16_t* __restrict__ key_scale_bf16,
    const std::uint16_t* __restrict__ value_scale_bf16,
    float* __restrict__ partial_output, float* __restrict__ partial_lse,
    float* __restrict__ output, const DecodeControl* __restrict__ control,
    std::uint64_t start_position,
    const DecodeControl* __restrict__ row_controls,
    std::uint64_t cache_capacity, int max_splits) {
  static_assert(kDecodeWarps * kDecodeGlobalGqaHeadsPerWarp ==
                kDecodeQueryHeads);
  static_assert(kDecodeGlobalHeadDimension % (32 * 4) == 0);
  static_assert(kDecodeGlobalGqaChunk % kDecodeGlobalGqaTileTokens == 0);
  __shared__ alignas(16)
      std::uint8_t staged_kv[kDecodeGlobalGqaTileTokens *
                             kDecodeGlobalHeadDimension];
  __shared__ float scores[kDecodeQueryHeads * kDecodeGlobalGqaChunk];
  __shared__ float reduction[kDecodeWarps];
  __shared__ float inverse_sum_shared[kDecodeQueryHeads];

  const int split = static_cast<int>(blockIdx.x);
  const int row = kBatch ? static_cast<int>(blockIdx.y) : 0;
  if (split >= max_splits) return;
  std::uint64_t tokens;
  if constexpr (kBatch) {
    if (row_controls != nullptr) start_position = row_controls[0].position;
    tokens = start_position + static_cast<std::uint64_t>(row) + 1U;
    query += static_cast<std::uint64_t>(row) * kDecodeQueryHeads *
             kDecodeGlobalHeadDimension;
  } else {
    tokens = control->position + 1U;
  }
  const std::uint64_t split_start =
      static_cast<std::uint64_t>(split) * kDecodeGlobalGqaChunk;
  if (split_start >= tokens) return;
  const int split_tokens = static_cast<int>(min(
      static_cast<std::uint64_t>(kDecodeGlobalGqaChunk),
      tokens - split_start));
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int query_head_base = warp * kDecodeGlobalGqaHeadsPerWarp;
  const float key_scale =
      static_cast<float>(__ushort_as_bfloat16(key_scale_bf16[0]));
  const float value_scale =
      static_cast<float>(__ushort_as_bfloat16(value_scale_bf16[0]));

  float query_values[kDecodeGlobalGqaHeadsPerWarp][16];
#pragma unroll
  for (int head = 0; head < kDecodeGlobalGqaHeadsPerWarp; ++head) {
#pragma unroll
    for (int element = 0; element < 16; ++element) {
      const int dimension = kVectorized
          ? lane * 4 + (element / 4) * 128 + element % 4
          : lane + element * 32;
      query_values[head][element] =
          query[(query_head_base + head) * kDecodeGlobalHeadDimension +
                dimension];
    }
  }

  for (int tile_start = 0; tile_start < split_tokens;
       tile_start += kDecodeGlobalGqaTileTokens) {
    const int tile_tokens =
        min(kDecodeGlobalGqaTileTokens, split_tokens - tile_start);
    constexpr int kTileBytes =
        kDecodeGlobalGqaTileTokens * kDecodeGlobalHeadDimension;
    for (int byte = static_cast<int>(threadIdx.x) * 16;
         byte < kTileBytes; byte += kDecodeThreads * 16) {
      const bool valid = byte < tile_tokens * kDecodeGlobalHeadDimension;
      const std::uint8_t* source = valid
          ? key_cache +
                (split_start + static_cast<std::uint64_t>(tile_start)) *
                    kDecodeGlobalHeadDimension +
                byte
          : key_cache;
      CopyAsync16(staged_kv + byte, source, valid ? 16 : 0);
    }
    CommitAsyncCopies();
    WaitForAsyncCopies();
    __syncthreads();

    for (int token = 0; token < tile_tokens; ++token) {
      float score[kDecodeGlobalGqaHeadsPerWarp] = {};
#pragma unroll
      if constexpr (kVectorized) {
#pragma unroll
        for (int quarter = 0; quarter < 4; ++quarter) {
          const int dimension = lane * 4 + quarter * 128;
          const std::uint32_t packed =
              *reinterpret_cast<const std::uint32_t*>(
                  staged_kv + token * kDecodeGlobalHeadDimension + dimension);
          const float4 key_values = DecodeFp8x4(packed, key_scale);
          const float values[4] = {key_values.x, key_values.y, key_values.z,
                                   key_values.w};
#pragma unroll
          for (int head = 0; head < kDecodeGlobalGqaHeadsPerWarp; ++head) {
#pragma unroll
            for (int element = 0; element < 4; ++element) {
              score[head] =
                  fmaf(query_values[head][quarter * 4 + element],
                       values[element], score[head]);
            }
          }
        }
      } else {
#pragma unroll
        for (int element = 0; element < 16; ++element) {
          const int dimension = lane + element * 32;
          const float key_value = DecodeFp8(
              staged_kv[token * kDecodeGlobalHeadDimension + dimension],
              key_scale);
#pragma unroll
          for (int head = 0; head < kDecodeGlobalGqaHeadsPerWarp; ++head) {
            score[head] = fmaf(query_values[head][element], key_value,
                               score[head]);
          }
        }
      }
#pragma unroll
      for (int head = 0; head < kDecodeGlobalGqaHeadsPerWarp; ++head) {
        for (int offset = 16; offset > 0; offset >>= 1) {
          score[head] +=
              __shfl_down_sync(kFullWarpMask, score[head], offset);
        }
        if (lane == 0) {
          scores[(query_head_base + head) * kDecodeGlobalGqaChunk +
                 tile_start + token] = score[head];
        }
      }
    }
    __syncthreads();
  }

  for (int query_head = 0; query_head < kDecodeQueryHeads; ++query_head) {
    float local_maximum = -CUDART_INF_F;
    for (int token = static_cast<int>(threadIdx.x); token < split_tokens;
         token += kDecodeThreads) {
      local_maximum = fmaxf(
          local_maximum,
          scores[query_head * kDecodeGlobalGqaChunk + token]);
    }
    const float maximum = DecodeBlockMaximum(local_maximum, reduction);
    float local_sum = 0.0F;
    for (int token = static_cast<int>(threadIdx.x); token < split_tokens;
         token += kDecodeThreads) {
      const std::uint64_t score_index =
          query_head * kDecodeGlobalGqaChunk + token;
      const float probability = expf(scores[score_index] - maximum);
      scores[score_index] = probability;
      local_sum += probability;
    }
    const float denominator = DecodeBlockSum(local_sum, reduction);
    if (threadIdx.x == 0) {
      inverse_sum_shared[query_head] =
          denominator > 0.0F ? __frcp_rn(denominator) : 0.0F;
      if constexpr (!kDirect) {
        partial_lse
            [(static_cast<std::uint64_t>(row) * max_splits + split) *
                 kDecodeQueryHeads +
             query_head] = maximum + logf(denominator);
      }
    }
    __syncthreads();
  }

  float accumulator[kDecodeGlobalGqaHeadsPerWarp][16] = {};
  for (int tile_start = 0; tile_start < split_tokens;
       tile_start += kDecodeGlobalGqaTileTokens) {
    const int tile_tokens =
        min(kDecodeGlobalGqaTileTokens, split_tokens - tile_start);
    constexpr int kTileBytes =
        kDecodeGlobalGqaTileTokens * kDecodeGlobalHeadDimension;
    for (int byte = static_cast<int>(threadIdx.x) * 16;
         byte < kTileBytes; byte += kDecodeThreads * 16) {
      const bool valid = byte < tile_tokens * kDecodeGlobalHeadDimension;
      const std::uint8_t* source = valid
          ? value_cache +
                (split_start + static_cast<std::uint64_t>(tile_start)) *
                    kDecodeGlobalHeadDimension +
                byte
          : value_cache;
      CopyAsync16(staged_kv + byte, source, valid ? 16 : 0);
    }
    CommitAsyncCopies();
    WaitForAsyncCopies();
    __syncthreads();

    for (int token = 0; token < tile_tokens; ++token) {
      float probability[kDecodeGlobalGqaHeadsPerWarp];
#pragma unroll
      for (int head = 0; head < kDecodeGlobalGqaHeadsPerWarp; ++head) {
        probability[head] =
            scores[(query_head_base + head) * kDecodeGlobalGqaChunk +
                   tile_start + token];
      }
      if constexpr (kVectorized) {
#pragma unroll
        for (int quarter = 0; quarter < 4; ++quarter) {
          const int dimension = lane * 4 + quarter * 128;
          const std::uint32_t packed =
              *reinterpret_cast<const std::uint32_t*>(
                  staged_kv + token * kDecodeGlobalHeadDimension + dimension);
          const float4 decoded = DecodeFp8x4(packed, value_scale);
          const float values[4] = {decoded.x, decoded.y, decoded.z,
                                   decoded.w};
#pragma unroll
          for (int head = 0; head < kDecodeGlobalGqaHeadsPerWarp; ++head) {
#pragma unroll
            for (int element = 0; element < 4; ++element) {
              accumulator[head][quarter * 4 + element] =
                  fmaf(probability[head], values[element],
                       accumulator[head][quarter * 4 + element]);
            }
          }
        }
      } else {
#pragma unroll
        for (int element = 0; element < 16; ++element) {
          const int dimension = lane + element * 32;
          const float value = DecodeFp8(
              staged_kv[token * kDecodeGlobalHeadDimension + dimension],
              value_scale);
#pragma unroll
          for (int head = 0; head < kDecodeGlobalGqaHeadsPerWarp; ++head) {
            accumulator[head][element] =
                fmaf(probability[head], value, accumulator[head][element]);
          }
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int head = 0; head < kDecodeGlobalGqaHeadsPerWarp; ++head) {
    const int query_head = query_head_base + head;
    const float inverse_sum = inverse_sum_shared[query_head];
#pragma unroll
    for (int element = 0; element < 16; ++element) {
      const int dimension = kVectorized
          ? lane * 4 + (element / 4) * 128 + element % 4
          : lane + element * 32;
      const float normalized = accumulator[head][element] * inverse_sum;
      const std::uint64_t output_index =
          (static_cast<std::uint64_t>(row) * kDecodeQueryHeads + query_head) *
              kDecodeGlobalHeadDimension +
          dimension;
      if constexpr (kDirect) {
        output[output_index] = normalized;
      } else {
        partial_output
            [((static_cast<std::uint64_t>(row) * max_splits + split) *
                  kDecodeQueryHeads +
              query_head) *
                 kDecodeGlobalHeadDimension +
             dimension] = normalized;
      }
    }
  }
}

template <int kRows, bool kVectorized>
__launch_bounds__(kDecodeThreads, 1) __global__
void SplitOnlineDecodeAttentionFp8GlobalBatchKernel(
    const float* __restrict__ query,
    const std::uint8_t* __restrict__ key_cache,
    const std::uint8_t* __restrict__ value_cache,
    const std::uint16_t* __restrict__ key_scale_bf16,
    const std::uint16_t* __restrict__ value_scale_bf16,
    float* __restrict__ partial_output, float* __restrict__ partial_lse,
    std::uint64_t start_position, const DecodeControl* row_controls,
    std::uint64_t cache_capacity, int max_splits) {
  static_assert(kRows == 3);
  if (row_controls != nullptr) start_position = row_controls[0].position;
  constexpr int kGroups = kDecodeQueryHeads / kDecodeGlobalGroup;
  __shared__ float scores[kRows * kDecodeGlobalGroup * kDecodeGlobalChunk];
  __shared__ float reduction[kDecodeWarps];

  const int linear_block = static_cast<int>(blockIdx.x);
  const int split = linear_block / kGroups;
  const int query_group = linear_block - split * kGroups;
  if (split >= max_splits) return;
  const std::uint64_t split_start =
      static_cast<std::uint64_t>(split) * kDecodeGlobalChunk;
  const std::uint64_t final_tokens = start_position + kRows;
  if (split_start >= final_tokens) return;
  int split_tokens[kRows];
#pragma unroll
  for (int row = 0; row < kRows; ++row) {
    const std::uint64_t row_tokens = start_position + row + 1U;
    split_tokens[row] =
        split_start < row_tokens
            ? static_cast<int>(min(
                  static_cast<std::uint64_t>(kDecodeGlobalChunk),
                  row_tokens - split_start))
            : 0;
  }
  const int maximum_split_tokens = split_tokens[kRows - 1];
  const int query_head_base = query_group * kDecodeGlobalGroup;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const float key_scale =
      static_cast<float>(__ushort_as_bfloat16(key_scale_bf16[0]));
  const float value_scale =
      static_cast<float>(__ushort_as_bfloat16(value_scale_bf16[0]));

  for (int token = warp; token < maximum_split_tokens;
       token += kDecodeWarps) {
    const std::uint64_t cache_slot =
        split_start + static_cast<std::uint64_t>(token);
    const std::uint8_t* key = key_cache + cache_slot * kDecodeGlobalHeadDimension;
    float score[kRows][kDecodeGlobalGroup] = {};
    if constexpr (kVectorized) {
      for (int dimension = lane * 4; dimension < kDecodeGlobalHeadDimension;
           dimension += 32 * 4) {
        const std::uint32_t packed =
            *reinterpret_cast<const std::uint32_t*>(key + dimension);
        const float4 key_values = DecodeFp8x4(packed, key_scale);
#pragma unroll
        for (int row = 0; row < kRows; ++row) {
          if (token < split_tokens[row]) {
#pragma unroll
            for (int group = 0; group < kDecodeGlobalGroup; ++group) {
              const float* query_values =
                  query +
                  (static_cast<std::uint64_t>(row) * kDecodeQueryHeads +
                   query_head_base + group) *
                      kDecodeGlobalHeadDimension +
                  dimension;
              score[row][group] =
                  fmaf(query_values[0], key_values.x, score[row][group]);
              score[row][group] =
                  fmaf(query_values[1], key_values.y, score[row][group]);
              score[row][group] =
                  fmaf(query_values[2], key_values.z, score[row][group]);
              score[row][group] =
                  fmaf(query_values[3], key_values.w, score[row][group]);
            }
          }
        }
      }
    } else {
      for (int dimension = lane; dimension < kDecodeGlobalHeadDimension;
           dimension += 32) {
        const float key_value = DecodeFp8(key[dimension], key_scale);
#pragma unroll
        for (int row = 0; row < kRows; ++row) {
          if (token < split_tokens[row]) {
#pragma unroll
            for (int group = 0; group < kDecodeGlobalGroup; ++group) {
              score[row][group] = fmaf(
                  query[(static_cast<std::uint64_t>(row) * kDecodeQueryHeads +
                         query_head_base + group) *
                            kDecodeGlobalHeadDimension +
                        dimension],
                  key_value, score[row][group]);
            }
          }
        }
      }
    }
#pragma unroll
    for (int row = 0; row < kRows; ++row) {
      if (token < split_tokens[row]) {
#pragma unroll
        for (int group = 0; group < kDecodeGlobalGroup; ++group) {
          for (int offset = 16; offset > 0; offset >>= 1) {
            score[row][group] += __shfl_down_sync(
                kFullWarpMask, score[row][group], offset);
          }
          if (lane == 0) {
            scores[(row * kDecodeGlobalGroup + group) *
                       kDecodeGlobalChunk +
                   token] = score[row][group];
          }
        }
      }
    }
  }
  __syncthreads();

  float inverse_sum[kRows][kDecodeGlobalGroup];
#pragma unroll
  for (int row = 0; row < kRows; ++row) {
#pragma unroll
    for (int group = 0; group < kDecodeGlobalGroup; ++group) {
      float local_maximum = -CUDART_INF_F;
      for (int token = static_cast<int>(threadIdx.x);
           token < split_tokens[row]; token += kDecodeThreads) {
        local_maximum = fmaxf(
            local_maximum,
            scores[(row * kDecodeGlobalGroup + group) *
                       kDecodeGlobalChunk +
                   token]);
      }
      const float maximum = DecodeBlockMaximum(local_maximum, reduction);
      float local_sum = 0.0F;
      for (int token = static_cast<int>(threadIdx.x);
           token < split_tokens[row]; token += kDecodeThreads) {
        const std::uint64_t score_index =
            (row * kDecodeGlobalGroup + group) * kDecodeGlobalChunk + token;
        const float probability = expf(scores[score_index] - maximum);
        scores[score_index] = probability;
        local_sum += probability;
      }
      const float denominator = DecodeBlockSum(local_sum, reduction);
      inverse_sum[row][group] =
          denominator > 0.0F ? __frcp_rn(denominator) : 0.0F;
      if (threadIdx.x == 0) {
        partial_lse[(static_cast<std::uint64_t>(row) * max_splits + split) *
                        kDecodeQueryHeads +
                    query_head_base + group] = maximum + logf(denominator);
      }
    }
  }

  if constexpr (kVectorized) {
    for (int dimension = static_cast<int>(threadIdx.x) * 4;
         dimension < kDecodeGlobalHeadDimension;
         dimension += kDecodeThreads * 4) {
      float accumulator[kRows][kDecodeGlobalGroup][4] = {};
      for (int token = 0; token < maximum_split_tokens; ++token) {
        const std::uint64_t value_offset =
            (split_start + static_cast<std::uint64_t>(token)) *
                kDecodeGlobalHeadDimension +
            dimension;
        const std::uint32_t packed =
            *reinterpret_cast<const std::uint32_t*>(value_cache + value_offset);
        const float4 values = DecodeFp8x4(packed, value_scale);
#pragma unroll
        for (int row = 0; row < kRows; ++row) {
          if (token < split_tokens[row]) {
#pragma unroll
            for (int group = 0; group < kDecodeGlobalGroup; ++group) {
              const float probability =
                  scores[(row * kDecodeGlobalGroup + group) *
                             kDecodeGlobalChunk +
                         token];
              accumulator[row][group][0] =
                  fmaf(probability, values.x, accumulator[row][group][0]);
              accumulator[row][group][1] =
                  fmaf(probability, values.y, accumulator[row][group][1]);
              accumulator[row][group][2] =
                  fmaf(probability, values.z, accumulator[row][group][2]);
              accumulator[row][group][3] =
                  fmaf(probability, values.w, accumulator[row][group][3]);
            }
          }
        }
      }
#pragma unroll
      for (int row = 0; row < kRows; ++row) {
#pragma unroll
        for (int group = 0; group < kDecodeGlobalGroup; ++group) {
          const int query_head = query_head_base + group;
          const std::uint64_t output_base =
              ((static_cast<std::uint64_t>(row) * max_splits + split) *
                   kDecodeQueryHeads +
               query_head) *
                  kDecodeGlobalHeadDimension +
              dimension;
#pragma unroll
          for (int element = 0; element < 4; ++element) {
            partial_output[output_base + element] =
                accumulator[row][group][element] * inverse_sum[row][group];
          }
        }
      }
    }
  } else {
    for (int dimension = static_cast<int>(threadIdx.x);
         dimension < kDecodeGlobalHeadDimension;
         dimension += kDecodeThreads) {
      float accumulator[kRows][kDecodeGlobalGroup] = {};
      for (int token = 0; token < maximum_split_tokens; ++token) {
        const std::uint64_t value_offset =
            (split_start + static_cast<std::uint64_t>(token)) *
                kDecodeGlobalHeadDimension +
            dimension;
        const float value = DecodeFp8(value_cache[value_offset], value_scale);
#pragma unroll
        for (int row = 0; row < kRows; ++row) {
          if (token < split_tokens[row]) {
#pragma unroll
            for (int group = 0; group < kDecodeGlobalGroup; ++group) {
              accumulator[row][group] = fmaf(
                  scores[(row * kDecodeGlobalGroup + group) *
                             kDecodeGlobalChunk +
                         token],
                  value, accumulator[row][group]);
            }
          }
        }
      }
#pragma unroll
      for (int row = 0; row < kRows; ++row) {
#pragma unroll
        for (int group = 0; group < kDecodeGlobalGroup; ++group) {
          const int query_head = query_head_base + group;
          partial_output
              [((static_cast<std::uint64_t>(row) * max_splits + split) *
                    kDecodeQueryHeads +
                query_head) *
                   kDecodeGlobalHeadDimension +
               dimension] = accumulator[row][group] * inverse_sum[row][group];
        }
      }
    }
  }
}

template <int kRows>
__launch_bounds__(kDecodeThreads, 1) __global__
void MergeOnlineDecodeAttentionGlobalBatchKernel(
    const float* __restrict__ partial_output,
    const float* __restrict__ partial_lse, float* __restrict__ output,
    std::uint64_t start_position, const DecodeControl* row_controls,
    int max_splits) {
  if (row_controls != nullptr) start_position = row_controls[0].position;
  __shared__ float reduction[kDecodeWarps];
  const int query_head = static_cast<int>(blockIdx.x);
  const int row = static_cast<int>(blockIdx.y);
  const int valid_splits = static_cast<int>(
      (start_position + static_cast<std::uint64_t>(row) + 1U +
       kDecodeGlobalChunk - 1U) /
      kDecodeGlobalChunk);
  const std::uint64_t lse_base =
      static_cast<std::uint64_t>(row) * max_splits * kDecodeQueryHeads;
  float local_maximum = -CUDART_INF_F;
  for (int split = static_cast<int>(threadIdx.x); split < valid_splits;
       split += kDecodeThreads) {
    local_maximum = fmaxf(
        local_maximum, partial_lse[lse_base + split * kDecodeQueryHeads +
                                   query_head]);
  }
  const float maximum = DecodeBlockMaximum(local_maximum, reduction);
  float local_sum = 0.0F;
  for (int split = static_cast<int>(threadIdx.x); split < valid_splits;
       split += kDecodeThreads) {
    local_sum += expf(partial_lse[lse_base + split * kDecodeQueryHeads +
                                  query_head] -
                      maximum);
  }
  const float denominator = DecodeBlockSum(local_sum, reduction);
  const float inverse_sum =
      denominator > 0.0F ? __frcp_rn(denominator) : 0.0F;
  for (int dimension = static_cast<int>(threadIdx.x);
       dimension < kDecodeGlobalHeadDimension; dimension += kDecodeThreads) {
    float accumulator = 0.0F;
    for (int split = 0; split < valid_splits; ++split) {
      const float weight =
          expf(partial_lse[lse_base + split * kDecodeQueryHeads + query_head] -
               maximum);
      accumulator = fmaf(
          weight,
          partial_output
              [((static_cast<std::uint64_t>(row) * max_splits + split) *
                    kDecodeQueryHeads +
                query_head) *
                   kDecodeGlobalHeadDimension +
               dimension],
          accumulator);
    }
    output[(static_cast<std::uint64_t>(row) * kDecodeQueryHeads + query_head) *
               kDecodeGlobalHeadDimension +
           dimension] = accumulator * inverse_sum;
  }
}

template <int kHeadDimensionValue, int kChunk>
__launch_bounds__(kDecodeThreads, 1) __global__
void MergeOnlineDecodeAttentionKernel(
    const float* __restrict__ partial_output,
    const float* __restrict__ partial_lse, float* __restrict__ output,
    const DecodeControl* __restrict__ control,
    std::uint64_t cache_capacity, bool sliding, int max_splits) {
  __shared__ float reduction[kDecodeWarps];
  const int query_head = static_cast<int>(blockIdx.x);
  const std::uint64_t token_count_unbounded = control->position + 1U;
  const std::uint64_t tokens =
      sliding && token_count_unbounded > cache_capacity
          ? cache_capacity
          : token_count_unbounded;
  const int valid_splits =
      static_cast<int>((tokens + kChunk - 1U) / kChunk);
  float local_maximum = -CUDART_INF_F;
  for (int split = static_cast<int>(threadIdx.x); split < valid_splits;
       split += kDecodeThreads) {
    local_maximum =
        fmaxf(local_maximum,
              partial_lse[split * kDecodeQueryHeads + query_head]);
  }
  const float maximum = DecodeBlockMaximum(local_maximum, reduction);
  float local_sum = 0.0F;
  for (int split = static_cast<int>(threadIdx.x); split < valid_splits;
       split += kDecodeThreads) {
    local_sum +=
        expf(partial_lse[split * kDecodeQueryHeads + query_head] - maximum);
  }
  const float denominator = DecodeBlockSum(local_sum, reduction);
  const float inverse_sum =
      denominator > 0.0F ? __frcp_rn(denominator) : 0.0F;
  for (int dimension = static_cast<int>(threadIdx.x);
       dimension < kHeadDimensionValue; dimension += kDecodeThreads) {
    float accumulator = 0.0F;
    for (int split = 0; split < valid_splits; ++split) {
      const float weight =
          expf(partial_lse[split * kDecodeQueryHeads + query_head] -
               maximum);
      accumulator = fmaf(
          weight,
          partial_output
              [(static_cast<std::uint64_t>(split) * kDecodeQueryHeads +
                query_head) *
                   kHeadDimensionValue +
               dimension],
          accumulator);
    }
    output[static_cast<std::uint64_t>(query_head) * kHeadDimensionValue +
           dimension] = accumulator * inverse_sum;
  }
}


}  // namespace

std::uint64_t DecodeAttentionWorkspaceElements(std::uint64_t max_context) {
  if (max_context == 0U) return 0U;
  const std::uint64_t local_capacity =
      max_context < 1024U ? max_context : 1024U;
  const std::uint64_t local_splits =
      (local_capacity + kDecodeLocalChunk - 1U) / kDecodeLocalChunk;
  const std::uint64_t global_splits =
      (max_context + kDecodeGlobalGqaChunk - 1U) /
      kDecodeGlobalGqaChunk;
  const std::uint64_t local_elements =
      local_splits * kDecodeQueryHeads *
      (kDecodeLocalHeadDimension + 1U);
  const std::uint64_t global_elements =
      global_splits * kDecodeQueryHeads *
      (kDecodeGlobalHeadDimension + 1U);
  return local_elements > global_elements ? local_elements : global_elements;
}

Status LaunchOnlineAttentionDecodeFp8Sm120(
    const float* query, const std::uint8_t* key_cache,
    const std::uint8_t* value_cache,
    const std::uint16_t* key_scale_bf16,
    const std::uint16_t* value_scale_bf16, float* workspace, float* output,
    const DecodeControl* control, std::uint64_t query_heads,
    std::uint64_t kv_heads, std::uint64_t head_dimension,
    std::uint64_t cache_capacity, bool sliding, cudaStream_t stream) {
  const bool local_shape =
      query_heads == kDecodeQueryHeads && kv_heads == kDecodeLocalKvHeads &&
      head_dimension == kDecodeLocalHeadDimension && sliding;
  const bool global_shape =
      query_heads == kDecodeQueryHeads && kv_heads == kDecodeGlobalKvHeads &&
      head_dimension == kDecodeGlobalHeadDimension && !sliding;
  if (query == nullptr || key_cache == nullptr || value_cache == nullptr ||
      key_scale_bf16 == nullptr || value_scale_bf16 == nullptr ||
      workspace == nullptr || output == nullptr || control == nullptr ||
      cache_capacity == 0U || (!local_shape && !global_shape) ||
      cache_capacity >
          static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    return Invalid("online FP8 decode attention arguments are invalid");
  }

  const bool global_gqa =
      global_shape && cache_capacity >= kDecodeGqaGlobalContext;
  const bool vectorized_global =
      global_shape && cache_capacity >= kDecodeVectorizedGlobalContext;
  const int chunk = local_shape
      ? kDecodeLocalChunk
      : (global_gqa ? kDecodeGlobalGqaChunk : kDecodeGlobalChunk);
  const int max_splits = static_cast<int>(
      (cache_capacity + static_cast<std::uint64_t>(chunk) - 1U) /
      static_cast<std::uint64_t>(chunk));
  const int query_groups =
      local_shape
          ? kDecodeLocalKvHeads
          : (global_gqa
                 ? 1
                 : kDecodeGlobalKvHeads *
                       (kDecodeQueryHeads / kDecodeGlobalGroup));
  const std::uint64_t partial_elements =
      static_cast<std::uint64_t>(max_splits) * kDecodeQueryHeads *
      head_dimension;
  float* partial_lse = workspace + partial_elements;
  const unsigned blocks =
      static_cast<unsigned>(max_splits * query_groups);

  if (local_shape) {
    if (max_splits == 1) {
      SplitOnlineDecodeAttentionFp8Kernel<
          kDecodeLocalHeadDimension, kDecodeLocalKvHeads, 2,
          kDecodeLocalGroup, kDecodeLocalChunk, true, false>
          <<<blocks, kDecodeThreads, 0, stream>>>(
              query, key_cache, value_cache, key_scale_bf16,
              value_scale_bf16, workspace, nullptr, output, control,
              cache_capacity, sliding, max_splits);
    } else {
      SplitOnlineDecodeAttentionFp8Kernel<
          kDecodeLocalHeadDimension, kDecodeLocalKvHeads, 2,
          kDecodeLocalGroup, kDecodeLocalChunk, false, false>
          <<<blocks, kDecodeThreads, 0, stream>>>(
              query, key_cache, value_cache, key_scale_bf16,
              value_scale_bf16, workspace, partial_lse, output, control,
              cache_capacity, sliding, max_splits);
    }
  } else {
    if (global_gqa) {
      if (vectorized_global) {
        if (max_splits == 1) {
          SplitOnlineDecodeAttentionFp8GlobalGqaKernel<true>
              <<<blocks, kDecodeThreads, 0, stream>>>(
                  query, key_cache, value_cache, key_scale_bf16,
                  value_scale_bf16, workspace, nullptr, output, control, 0U,
                  nullptr, cache_capacity, max_splits);
        } else {
          SplitOnlineDecodeAttentionFp8GlobalGqaKernel<false>
              <<<blocks, kDecodeThreads, 0, stream>>>(
                  query, key_cache, value_cache, key_scale_bf16,
                  value_scale_bf16, workspace, partial_lse, output, control,
                  0U, nullptr, cache_capacity, max_splits);
        }
      } else if (max_splits == 1) {
        SplitOnlineDecodeAttentionFp8GlobalGqaKernel<true, false, false>
            <<<blocks, kDecodeThreads, 0, stream>>>(
                query, key_cache, value_cache, key_scale_bf16,
                value_scale_bf16, workspace, nullptr, output, control, 0U,
                nullptr, cache_capacity, max_splits);
      } else {
        SplitOnlineDecodeAttentionFp8GlobalGqaKernel<false, false, false>
            <<<blocks, kDecodeThreads, 0, stream>>>(
                query, key_cache, value_cache, key_scale_bf16,
                value_scale_bf16, workspace, partial_lse, output, control, 0U,
                nullptr, cache_capacity, max_splits);
      }
    } else if (max_splits == 1) {
      SplitOnlineDecodeAttentionFp8Kernel<
          kDecodeGlobalHeadDimension, kDecodeGlobalKvHeads, 16,
          kDecodeGlobalGroup, kDecodeGlobalChunk, true, false>
          <<<blocks, kDecodeThreads, 0, stream>>>(
              query, key_cache, value_cache, key_scale_bf16,
              value_scale_bf16, workspace, nullptr, output, control,
              cache_capacity, sliding, max_splits);
    } else {
      SplitOnlineDecodeAttentionFp8Kernel<
          kDecodeGlobalHeadDimension, kDecodeGlobalKvHeads, 16,
          kDecodeGlobalGroup, kDecodeGlobalChunk, false, false>
          <<<blocks, kDecodeThreads, 0, stream>>>(
              query, key_cache, value_cache, key_scale_bf16,
              value_scale_bf16, workspace, partial_lse, output, control,
              cache_capacity, sliding, max_splits);
    }
  }
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaFailure("launch split online FP8 decode attention", error);
  }
  if (max_splits == 1) return Status::Ok();

  if (local_shape) {
    MergeOnlineDecodeAttentionKernel<kDecodeLocalHeadDimension,
                                     kDecodeLocalChunk>
        <<<kDecodeQueryHeads, kDecodeThreads, 0, stream>>>(
            workspace, partial_lse, output, control, cache_capacity, sliding,
            max_splits);
  } else if (global_gqa) {
    MergeOnlineDecodeAttentionKernel<kDecodeGlobalHeadDimension,
                                     kDecodeGlobalGqaChunk>
        <<<kDecodeQueryHeads, kDecodeThreads, 0, stream>>>(
            workspace, partial_lse, output, control, cache_capacity, sliding,
            max_splits);
  } else {
    MergeOnlineDecodeAttentionKernel<kDecodeGlobalHeadDimension,
                                     kDecodeGlobalChunk>
        <<<kDecodeQueryHeads, kDecodeThreads, 0, stream>>>(
            workspace, partial_lse, output, control, cache_capacity, sliding,
            max_splits);
  }
  error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch online FP8 decode attention merge", error);
}

Status LaunchOnlineAttentionDecodeFp8GlobalD2Sm120(
    const float* query, const std::uint8_t* key_cache,
    const std::uint8_t* value_cache,
    const std::uint16_t* key_scale_bf16,
    const std::uint16_t* value_scale_bf16, float* workspace, float* output,
    std::uint64_t start_position, std::uint64_t cache_capacity,
    cudaStream_t stream) {
  constexpr int kRows = 3;
  if (query == nullptr || key_cache == nullptr || value_cache == nullptr ||
      key_scale_bf16 == nullptr || value_scale_bf16 == nullptr ||
      workspace == nullptr || output == nullptr || cache_capacity <= 512U ||
      start_position >= cache_capacity ||
      kRows > cache_capacity - start_position ||
      cache_capacity >
          static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    return Invalid("global D2 FP8 attention arguments are invalid");
  }
  const int max_splits = static_cast<int>(
      (cache_capacity + kDecodeGlobalChunk - 1U) / kDecodeGlobalChunk);
  constexpr int kGroups = kDecodeQueryHeads / kDecodeGlobalGroup;
  const unsigned blocks = static_cast<unsigned>(max_splits * kGroups);
  const std::uint64_t partial_elements_per_row =
      static_cast<std::uint64_t>(max_splits) * kDecodeQueryHeads *
      kDecodeGlobalHeadDimension;
  float* partial_lse = workspace + kRows * partial_elements_per_row;
  if (cache_capacity >= kDecodeGqaGlobalContext) {
    const dim3 gqa_blocks(static_cast<unsigned>(max_splits), kRows);
    if (cache_capacity >= kDecodeVectorizedGlobalContext) {
      SplitOnlineDecodeAttentionFp8GlobalGqaKernel<false, true>
          <<<gqa_blocks, kDecodeThreads, 0, stream>>>(
              query, key_cache, value_cache, key_scale_bf16,
              value_scale_bf16, workspace, partial_lse, nullptr, nullptr,
              start_position, nullptr, cache_capacity, max_splits);
    } else {
      SplitOnlineDecodeAttentionFp8GlobalGqaKernel<false, true, false>
          <<<gqa_blocks, kDecodeThreads, 0, stream>>>(
              query, key_cache, value_cache, key_scale_bf16,
              value_scale_bf16, workspace, partial_lse, nullptr, nullptr,
              start_position, nullptr, cache_capacity, max_splits);
    }
  } else {
    SplitOnlineDecodeAttentionFp8GlobalBatchKernel<kRows, false>
        <<<blocks, kDecodeThreads, 0, stream>>>(
            query, key_cache, value_cache, key_scale_bf16, value_scale_bf16,
            workspace, partial_lse, start_position, nullptr, cache_capacity,
            max_splits);
  }
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaFailure("launch global D2 split FP8 attention", error);
  }
  const dim3 merge_blocks(kDecodeQueryHeads, kRows);
  MergeOnlineDecodeAttentionGlobalBatchKernel<kRows>
      <<<merge_blocks, kDecodeThreads, 0, stream>>>(
          workspace, partial_lse, output, start_position, nullptr, max_splits);
  error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch global D2 FP8 attention merge", error);
}

Status LaunchOnlineAttentionDecodeFp8GlobalD2ControlledSm120(
    const float* query, const std::uint8_t* key_cache,
    const std::uint8_t* value_cache,
    const std::uint16_t* key_scale_bf16,
    const std::uint16_t* value_scale_bf16, float* workspace, float* output,
    const DecodeControl* row_controls, std::uint64_t cache_capacity,
    cudaStream_t stream) {
  constexpr int kRows = 3;
  if (query == nullptr || key_cache == nullptr || value_cache == nullptr ||
      key_scale_bf16 == nullptr || value_scale_bf16 == nullptr ||
      workspace == nullptr || output == nullptr || row_controls == nullptr ||
      cache_capacity <= 512U ||
      cache_capacity >
          static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    return Invalid("controlled global D2 FP8 attention arguments are invalid");
  }
  const int max_splits = static_cast<int>(
      (cache_capacity + kDecodeGlobalChunk - 1U) / kDecodeGlobalChunk);
  constexpr int kGroups = kDecodeQueryHeads / kDecodeGlobalGroup;
  const unsigned blocks = static_cast<unsigned>(max_splits * kGroups);
  const std::uint64_t partial_elements_per_row =
      static_cast<std::uint64_t>(max_splits) * kDecodeQueryHeads *
      kDecodeGlobalHeadDimension;
  float* partial_lse = workspace + kRows * partial_elements_per_row;
  if (cache_capacity >= kDecodeGqaGlobalContext) {
    const dim3 gqa_blocks(static_cast<unsigned>(max_splits), kRows);
    if (cache_capacity >= kDecodeVectorizedGlobalContext) {
      SplitOnlineDecodeAttentionFp8GlobalGqaKernel<false, true>
          <<<gqa_blocks, kDecodeThreads, 0, stream>>>(
              query, key_cache, value_cache, key_scale_bf16,
              value_scale_bf16, workspace, partial_lse, nullptr, nullptr, 0U,
              row_controls, cache_capacity, max_splits);
    } else {
      SplitOnlineDecodeAttentionFp8GlobalGqaKernel<false, true, false>
          <<<gqa_blocks, kDecodeThreads, 0, stream>>>(
              query, key_cache, value_cache, key_scale_bf16,
              value_scale_bf16, workspace, partial_lse, nullptr, nullptr, 0U,
              row_controls, cache_capacity, max_splits);
    }
  } else {
    SplitOnlineDecodeAttentionFp8GlobalBatchKernel<kRows, false>
        <<<blocks, kDecodeThreads, 0, stream>>>(
            query, key_cache, value_cache, key_scale_bf16, value_scale_bf16,
            workspace, partial_lse, 0U, row_controls, cache_capacity,
            max_splits);
  }
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaFailure("launch controlled global D2 split FP8 attention",
                       error);
  }
  const dim3 merge_blocks(kDecodeQueryHeads, kRows);
  MergeOnlineDecodeAttentionGlobalBatchKernel<kRows>
      <<<merge_blocks, kDecodeThreads, 0, stream>>>(
          workspace, partial_lse, output, 0U, row_controls, max_splits);
  error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch controlled global D2 FP8 attention merge",
                           error);
}

}  // namespace gem16::internal
