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
          int kQueriesPerKv, int kQueryGroup, int kChunk, bool kDirect>
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
      (max_context + kDecodeGlobalChunk - 1U) / kDecodeGlobalChunk;
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

  const int chunk = local_shape ? kDecodeLocalChunk : kDecodeGlobalChunk;
  const int max_splits = static_cast<int>(
      (cache_capacity + static_cast<std::uint64_t>(chunk) - 1U) /
      static_cast<std::uint64_t>(chunk));
  const int query_groups =
      local_shape ? kDecodeLocalKvHeads
                  : kDecodeGlobalKvHeads *
                        (kDecodeQueryHeads / kDecodeGlobalGroup);
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
          kDecodeLocalGroup, kDecodeLocalChunk, true>
          <<<blocks, kDecodeThreads, 0, stream>>>(
              query, key_cache, value_cache, key_scale_bf16,
              value_scale_bf16, workspace, nullptr, output, control,
              cache_capacity, sliding, max_splits);
    } else {
      SplitOnlineDecodeAttentionFp8Kernel<
          kDecodeLocalHeadDimension, kDecodeLocalKvHeads, 2,
          kDecodeLocalGroup, kDecodeLocalChunk, false>
          <<<blocks, kDecodeThreads, 0, stream>>>(
              query, key_cache, value_cache, key_scale_bf16,
              value_scale_bf16, workspace, partial_lse, output, control,
              cache_capacity, sliding, max_splits);
    }
  } else {
    if (max_splits == 1) {
      SplitOnlineDecodeAttentionFp8Kernel<
          kDecodeGlobalHeadDimension, kDecodeGlobalKvHeads, 16,
          kDecodeGlobalGroup, kDecodeGlobalChunk, true>
          <<<blocks, kDecodeThreads, 0, stream>>>(
              query, key_cache, value_cache, key_scale_bf16,
              value_scale_bf16, workspace, nullptr, output, control,
              cache_capacity, sliding, max_splits);
    } else {
      SplitOnlineDecodeAttentionFp8Kernel<
          kDecodeGlobalHeadDimension, kDecodeGlobalKvHeads, 16,
          kDecodeGlobalGroup, kDecodeGlobalChunk, false>
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


}  // namespace gem16::internal
