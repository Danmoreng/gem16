#include "cuda/attention/sm120.h"

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
// gem16gb's float Q, physical E4M3 K/V, current-chunk source, and circular cache.

namespace gem16gb::internal {
namespace {

constexpr int kHeadDimension = 256;
constexpr int kQueryRows = 32;
constexpr int kKeyColumns = 32;
constexpr int kThreads = 64;
constexpr int kQueryHeads = 16;
constexpr int kKvHeads = 8;
constexpr int kGroupSize = kQueryHeads / kKvHeads;
constexpr int kScoreTiles = kKeyColumns / 8;
constexpr int kQkSteps = kHeadDimension / 16;
constexpr int kOutputTiles = kHeadDimension / 8;
constexpr int kPvSteps = kKeyColumns / 16;
constexpr int kSharedElements =
    (kQueryRows + 2 * kKeyColumns) * kHeadDimension;
constexpr unsigned kFullWarpMask = 0xffffffffU;

static_assert(kSharedElements * sizeof(__nv_bfloat16) == 48 * 1024);

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

__device__ __forceinline__ int WindowStart(int query_position,
                                            int cache_capacity) {
  return max(0, query_position + 1 - cache_capacity);
}

__device__ __forceinline__ void StageQuery(
    __nv_bfloat16* destination, const float* query, int query_head,
    int query_start, int tokens, int thread) {
  constexpr int kElementsPerVector = 4;
  constexpr int kVectorsPerRow = kHeadDimension / kElementsPerVector;
  for (int chunk = thread; chunk < kQueryRows * kVectorsPerRow;
       chunk += kThreads) {
    const int row = chunk / kVectorsPerRow;
    const int dimension = (chunk % kVectorsPerRow) * kElementsPerVector;
    float values[kElementsPerVector] = {};
    if (query_start + row < tokens) {
      const float4 packed = *reinterpret_cast<const float4*>(
          query + ((query_start + row) * kQueryHeads + query_head) *
                      kHeadDimension +
          dimension);
      values[0] = packed.x;
      values[1] = packed.y;
      values[2] = packed.z;
      values[3] = packed.w;
    }
#pragma unroll
    for (int element = 0; element < kElementsPerVector; ++element) {
      destination[row * kHeadDimension + Swizzle(row, dimension + element)] =
          __float2bfloat16_rn(values[element]);
    }
  }
}

__device__ __forceinline__ void StageFp8Kv(
    __nv_bfloat16* destination, const std::uint8_t* chunk,
    const std::uint8_t* cache, float scale, int kv_head, int key_start,
    int max_query_position, int chunk_start, int kv_heads, int cache_capacity,
    int thread) {
  constexpr int kElementsPerVector = 8;
  constexpr int kVectorsPerRow = kHeadDimension / kElementsPerVector;
  for (int chunk_index = thread;
       chunk_index < kKeyColumns * kVectorsPerRow;
       chunk_index += kThreads) {
    const int row = chunk_index / kVectorsPerRow;
    const int dimension =
        (chunk_index % kVectorsPerRow) * kElementsPerVector;
    const int absolute_key = key_start + row;
    std::uint32_t words[2] = {};
    if (absolute_key <= max_query_position) {
      const bool in_chunk = absolute_key >= chunk_start;
      const int source_token =
          in_chunk ? absolute_key - chunk_start
                   : absolute_key % cache_capacity;
      const std::uint8_t* source = in_chunk ? chunk : cache;
      const uint2 packed = *reinterpret_cast<const uint2*>(
          source + ((source_token * kv_heads + kv_head) * kHeadDimension) +
          dimension);
      words[0] = packed.x;
      words[1] = packed.y;
    }
#pragma unroll
    for (int element = 0; element < kElementsPerVector; ++element) {
      __nv_fp8_e4m3 quantized;
      quantized.__x = static_cast<std::uint8_t>(
          words[element >> 2] >> ((element & 3) * 8));
      destination[row * kHeadDimension + Swizzle(row, dimension + element)] =
          __float2bfloat16_rn(static_cast<float>(quantized) * scale);
    }
  }
}

__launch_bounds__(kThreads, 1) __global__ void OnlineLocalAttentionFp8Kernel(
    const float* __restrict__ query,
    const std::uint8_t* __restrict__ chunk_key,
    const std::uint8_t* __restrict__ chunk_value,
    const std::uint8_t* __restrict__ key_cache,
    const std::uint8_t* __restrict__ value_cache,
    const std::uint16_t* __restrict__ key_scale_bf16,
    const std::uint16_t* __restrict__ value_scale_bf16,
    float* __restrict__ output, int start_position, int tokens,
    int cache_capacity) {
  __shared__ __align__(16) __nv_bfloat16 shared[kSharedElements];
  __nv_bfloat16* query_shared = shared;
  __nv_bfloat16* key_shared = query_shared + kQueryRows * kHeadDimension;
  __nv_bfloat16* value_shared =
      key_shared + kKeyColumns * kHeadDimension;

  const int query_block = static_cast<int>(blockIdx.x);
  const int query_head = static_cast<int>(blockIdx.y);
  const int thread = static_cast<int>(threadIdx.x);
  const int warp = thread >> 5;
  const int lane = thread & 31;
  const int query_start = query_block * kQueryRows;
  const int kv_head = query_head / kGroupSize;
  const int warp_row_start = warp * 16;
  if (query_head >= kQueryHeads || query_start >= tokens) return;

  const int group_lane = lane >> 2;
  const int lane_in_group = lane & 3;
  const int a_matrix = lane >> 3;
  const int a_row_in_matrix = lane & 7;
  const int a_row_offset =
      a_row_in_matrix + ((a_matrix & 1) << 3);
  const int b_row_in_matrix = lane & 7;
  const int b_contracting_offset = ((lane >> 3) & 1) << 3;

  const unsigned query_shared_base = SharedAddress(query_shared);
  const unsigned key_shared_base = SharedAddress(key_shared);
  const unsigned value_shared_base = SharedAddress(value_shared);
  const unsigned query_lane_base =
      query_shared_base +
      static_cast<unsigned>((warp_row_start + a_row_offset) * 512);
  const unsigned query_address_select =
      static_cast<unsigned>((a_matrix >> 1) << 4);
  const unsigned query_row_select =
      static_cast<unsigned>(a_row_in_matrix << 4);
  const unsigned key_lane_base =
      key_shared_base + static_cast<unsigned>(b_row_in_matrix * 512) +
      (static_cast<unsigned>(lane >> 4) << 12);
  const unsigned key_address_select =
      static_cast<unsigned>((b_contracting_offset >> 3) << 4);
  const unsigned key_row_select =
      static_cast<unsigned>(b_row_in_matrix << 4);
  const unsigned value_lane_base =
      value_shared_base +
      static_cast<unsigned>(((lane >> 3) & 1) * 4096) +
      static_cast<unsigned>(b_row_in_matrix * 512);
  const unsigned value_address_select =
      static_cast<unsigned>((lane >> 4) << 4);
  const unsigned value_row_select =
      static_cast<unsigned>(b_row_in_matrix << 4);

  StageQuery(query_shared, query, query_head, query_start, tokens, thread);

  float accumulator[kOutputTiles][4];
#pragma unroll
  for (int output_tile = 0; output_tile < kOutputTiles; ++output_tile) {
#pragma unroll
    for (int element = 0; element < 4; ++element) {
      accumulator[output_tile][element] = 0.0F;
    }
  }
  float maximum0 = -CUDART_INF_F;
  float maximum1 = -CUDART_INF_F;
  float sum0 = 0.0F;
  float sum1 = 0.0F;

  const int query_rows = min(kQueryRows, tokens - query_start);
  const int max_query_position =
      start_position + query_start + query_rows - 1;
  const int first_window =
      WindowStart(start_position + query_start, cache_capacity);
  const int first_key_start =
      (first_window / kKeyColumns) * kKeyColumns;
  const int key_block_count =
      (max_query_position - first_key_start) / kKeyColumns + 1;
  const float key_scale =
      static_cast<float>(__ushort_as_bfloat16(key_scale_bf16[0]));
  const float value_scale =
      static_cast<float>(__ushort_as_bfloat16(value_scale_bf16[0]));

  for (int key_block = 0; key_block < key_block_count; ++key_block) {
    const int key_start = first_key_start + key_block * kKeyColumns;
    StageFp8Kv(key_shared, chunk_key, key_cache, key_scale, kv_head,
               key_start, max_query_position, start_position, kKvHeads,
               cache_capacity, thread);
    __syncthreads();

    float scores[kScoreTiles][4];
#pragma unroll
    for (int score_tile = 0; score_tile < kScoreTiles; ++score_tile) {
      scores[score_tile][0] = 0.0F;
      scores[score_tile][1] = 0.0F;
      scores[score_tile][2] = 0.0F;
      scores[score_tile][3] = 0.0F;
    }
    unsigned query_fragments[2][4];
    unsigned key_fragments[2][kScoreTiles][2];
    LoadMatrixX4(
        query_fragments[0][0], query_fragments[0][1],
        query_fragments[0][2], query_fragments[0][3],
        SwizzledAddress(query_lane_base, 0U, query_address_select,
                         query_row_select));
#pragma unroll
    for (int tile_pair = 0; tile_pair < kScoreTiles; tile_pair += 2) {
      LoadMatrixX4(
          key_fragments[0][tile_pair][0],
          key_fragments[0][tile_pair][1],
          key_fragments[0][tile_pair + 1][0],
          key_fragments[0][tile_pair + 1][1],
          SwizzledAddress(
              key_lane_base + static_cast<unsigned>(tile_pair * 4096), 0U,
              key_address_select, key_row_select));
    }
#pragma unroll
    for (int step = 0; step < kQkSteps; ++step) {
      const int current = step & 1;
      const int next = current ^ 1;
      if (step + 1 < kQkSteps) {
        const unsigned contracting_offset =
            static_cast<unsigned>((step + 1) << 5);
        LoadMatrixX4(
            query_fragments[next][0], query_fragments[next][1],
            query_fragments[next][2], query_fragments[next][3],
            SwizzledAddress(query_lane_base, contracting_offset,
                             query_address_select, query_row_select));
#pragma unroll
        for (int tile_pair = 0; tile_pair < kScoreTiles; tile_pair += 2) {
          LoadMatrixX4(
              key_fragments[next][tile_pair][0],
              key_fragments[next][tile_pair][1],
              key_fragments[next][tile_pair + 1][0],
              key_fragments[next][tile_pair + 1][1],
              SwizzledAddress(
                  key_lane_base +
                      static_cast<unsigned>(tile_pair * 4096),
                  contracting_offset, key_address_select, key_row_select));
        }
      }
#pragma unroll
      for (int score_tile = 0; score_tile < kScoreTiles; ++score_tile) {
        MmaBf16(
            scores[score_tile][0], scores[score_tile][1],
            scores[score_tile][2], scores[score_tile][3],
            query_fragments[current][0], query_fragments[current][1],
            query_fragments[current][2], query_fragments[current][3],
            key_fragments[current][score_tile][0],
            key_fragments[current][score_tile][1]);
      }
    }

    const int row0 = warp_row_start + group_lane;
    const int row1 = row0 + 8;
    const int query_row0 = query_start + row0;
    const int query_row1 = query_start + row1;
    const bool row0_valid = query_row0 < tokens;
    const bool row1_valid = query_row1 < tokens;
    const int query_position0 =
        row0_valid ? start_position + query_row0 : -1;
    const int query_position1 =
        row1_valid ? start_position + query_row1 : -1;
    const int latest_window = WindowStart(
        start_position + query_start + kQueryRows - 1, cache_capacity);
    const bool full_score_tile =
        query_rows == kQueryRows && key_start >= latest_window &&
        key_start + kKeyColumns - 1 <= start_position + query_start;

    float block_maximum0 = -CUDART_INF_F;
    float block_maximum1 = -CUDART_INF_F;
    if (full_score_tile) {
#pragma unroll
      for (int score_tile = 0; score_tile < kScoreTiles; ++score_tile) {
        block_maximum0 =
            fmaxf(block_maximum0,
                  fmaxf(scores[score_tile][0], scores[score_tile][1]));
        block_maximum1 =
            fmaxf(block_maximum1,
                  fmaxf(scores[score_tile][2], scores[score_tile][3]));
      }
    } else {
      const int window0 =
          row0_valid ? WindowStart(query_position0, cache_capacity) : 0;
      const int window1 =
          row1_valid ? WindowStart(query_position1, cache_capacity) : 0;
#pragma unroll
      for (int score_tile = 0; score_tile < kScoreTiles; ++score_tile) {
        const int key0 =
            key_start + score_tile * 8 + 2 * lane_in_group;
        const int key1 = key0 + 1;
        scores[score_tile][0] =
            row0_valid && key0 >= window0 && key0 <= query_position0
                ? scores[score_tile][0]
                : -CUDART_INF_F;
        scores[score_tile][1] =
            row0_valid && key1 >= window0 && key1 <= query_position0
                ? scores[score_tile][1]
                : -CUDART_INF_F;
        scores[score_tile][2] =
            row1_valid && key0 >= window1 && key0 <= query_position1
                ? scores[score_tile][2]
                : -CUDART_INF_F;
        scores[score_tile][3] =
            row1_valid && key1 >= window1 && key1 <= query_position1
                ? scores[score_tile][3]
                : -CUDART_INF_F;
        block_maximum0 =
            fmaxf(block_maximum0,
                  fmaxf(scores[score_tile][0], scores[score_tile][1]));
        block_maximum1 =
            fmaxf(block_maximum1,
                  fmaxf(scores[score_tile][2], scores[score_tile][3]));
      }
    }
    block_maximum0 = WarpMaximum<4>(block_maximum0);
    block_maximum1 = WarpMaximum<4>(block_maximum1);

    const float next_maximum0 = fmaxf(maximum0, block_maximum0);
    const float next_maximum1 = fmaxf(maximum1, block_maximum1);
    const float alpha0 = isfinite(maximum0)
                             ? expf(maximum0 - next_maximum0)
                             : 0.0F;
    const float alpha1 = isfinite(maximum1)
                             ? expf(maximum1 - next_maximum1)
                             : 0.0F;

    float block_sum0 = 0.0F;
    float block_sum1 = 0.0F;
    unsigned probability_fragments[kPvSteps][4];
#pragma unroll
    for (int score_tile = 0; score_tile < kScoreTiles; ++score_tile) {
      const float probability00 = isfinite(scores[score_tile][0])
                                      ? expf(scores[score_tile][0] -
                                             next_maximum0)
                                      : 0.0F;
      const float probability01 = isfinite(scores[score_tile][1])
                                      ? expf(scores[score_tile][1] -
                                             next_maximum0)
                                      : 0.0F;
      const float probability10 = isfinite(scores[score_tile][2])
                                      ? expf(scores[score_tile][2] -
                                             next_maximum1)
                                      : 0.0F;
      const float probability11 = isfinite(scores[score_tile][3])
                                      ? expf(scores[score_tile][3] -
                                             next_maximum1)
                                      : 0.0F;
      block_sum0 += probability00 + probability01;
      block_sum1 += probability10 + probability11;
      const int probability_step = score_tile >> 1;
      if ((score_tile & 1) == 0) {
        probability_fragments[probability_step][0] =
            PackBf16x2(probability00, probability01);
        probability_fragments[probability_step][1] =
            PackBf16x2(probability10, probability11);
      } else {
        probability_fragments[probability_step][2] =
            PackBf16x2(probability00, probability01);
        probability_fragments[probability_step][3] =
            PackBf16x2(probability10, probability11);
      }
    }

    sum0 = fmaf(sum0, alpha0, block_sum0);
    sum1 = fmaf(sum1, alpha1, block_sum1);
    maximum0 = next_maximum0;
    maximum1 = next_maximum1;
#pragma unroll
    for (int output_tile = 0; output_tile < kOutputTiles; ++output_tile) {
      accumulator[output_tile][0] *= alpha0;
      accumulator[output_tile][1] *= alpha0;
      accumulator[output_tile][2] *= alpha1;
      accumulator[output_tile][3] *= alpha1;
    }

    StageFp8Kv(value_shared, chunk_value, value_cache, value_scale, kv_head,
               key_start, max_query_position, start_position, kKvHeads,
               cache_capacity, thread);
    __syncthreads();

    constexpr int kOutputTilePairs = kOutputTiles / 2;
    constexpr int kValueLoads = kPvSteps * kOutputTilePairs;
    unsigned value_fragments[2][4];
    LoadMatrixX4Transposed(
        value_fragments[0][0], value_fragments[0][1],
        value_fragments[0][2], value_fragments[0][3],
        SwizzledAddress(value_lane_base, 0U, value_address_select,
                         value_row_select));
#pragma unroll
    for (int load = 0; load < kValueLoads; ++load) {
      const int probability_step = load / kOutputTilePairs;
      const int output_tile = (load % kOutputTilePairs) * 2;
      const int current = load & 1;
      const int next = current ^ 1;
      if (load + 1 < kValueLoads) {
        const int next_probability_step =
            (load + 1) / kOutputTilePairs;
        const int next_output_tile =
            ((load + 1) % kOutputTilePairs) * 2;
        LoadMatrixX4Transposed(
            value_fragments[next][0], value_fragments[next][1],
            value_fragments[next][2], value_fragments[next][3],
            SwizzledAddress(
                value_lane_base +
                    static_cast<unsigned>(next_probability_step * 8192),
                static_cast<unsigned>(next_output_tile << 4),
                value_address_select, value_row_select));
      }
      MmaBf16(
          accumulator[output_tile][0], accumulator[output_tile][1],
          accumulator[output_tile][2], accumulator[output_tile][3],
          probability_fragments[probability_step][0],
          probability_fragments[probability_step][1],
          probability_fragments[probability_step][2],
          probability_fragments[probability_step][3],
          value_fragments[current][0], value_fragments[current][1]);
      MmaBf16(
          accumulator[output_tile + 1][0],
          accumulator[output_tile + 1][1],
          accumulator[output_tile + 1][2],
          accumulator[output_tile + 1][3],
          probability_fragments[probability_step][0],
          probability_fragments[probability_step][1],
          probability_fragments[probability_step][2],
          probability_fragments[probability_step][3],
          value_fragments[current][2], value_fragments[current][3]);
    }
  }

  sum0 = WarpSum<4>(sum0);
  sum1 = WarpSum<4>(sum1);
  const float inverse_sum0 = sum0 > 0.0F ? __frcp_rn(sum0) : 0.0F;
  const float inverse_sum1 = sum1 > 0.0F ? __frcp_rn(sum1) : 0.0F;
#pragma unroll
  for (int output_tile = 0; output_tile < kOutputTiles; ++output_tile) {
    const int dimension = output_tile * 8 + 2 * lane_in_group;
    const int query_row0 = query_start + warp_row_start + group_lane;
    const int query_row1 = query_row0 + 8;
    if (query_row0 < tokens) {
      float* output_row =
          output + (query_row0 * kQueryHeads + query_head) * kHeadDimension;
      output_row[dimension] =
          accumulator[output_tile][0] * inverse_sum0;
      output_row[dimension + 1] =
          accumulator[output_tile][1] * inverse_sum0;
    }
    if (query_row1 < tokens) {
      float* output_row =
          output + (query_row1 * kQueryHeads + query_head) * kHeadDimension;
      output_row[dimension] =
          accumulator[output_tile][2] * inverse_sum1;
      output_row[dimension + 1] =
          accumulator[output_tile][3] * inverse_sum1;
    }
  }
}

constexpr int kGlobalHeadDimension = 512;
constexpr int kGlobalQueryRows = 16;
constexpr int kGlobalKeyColumns = 16;
constexpr int kGlobalThreads = 64;
constexpr int kGlobalQueryHeads = 16;
constexpr int kGlobalKvHeads = 1;
constexpr int kGlobalOutputHalf = 256;
constexpr int kGlobalScoreTiles = kGlobalKeyColumns / 8;
constexpr int kGlobalQkSteps = kGlobalHeadDimension / 16;
constexpr int kGlobalOutputTiles = kGlobalOutputHalf / 8;
constexpr int kGlobalPvSteps = kGlobalKeyColumns / 16;
constexpr int kGlobalSharedElements =
    (kGlobalQueryRows + kGlobalKeyColumns) * kGlobalHeadDimension +
    kGlobalKeyColumns * kGlobalHeadDimension;

static_assert(kGlobalSharedElements * sizeof(__nv_bfloat16) == 48 * 1024);

__device__ __forceinline__ void StageGlobalQuery(
    __nv_bfloat16* destination, const float* query, int query_head,
    int query_start, int tokens, int thread) {
  constexpr int kElementsPerVector = 4;
  constexpr int kVectorsPerRow =
      kGlobalHeadDimension / kElementsPerVector;
  for (int chunk = thread;
       chunk < kGlobalQueryRows * kVectorsPerRow;
       chunk += kGlobalThreads) {
    const int row = chunk / kVectorsPerRow;
    const int dimension = (chunk % kVectorsPerRow) * kElementsPerVector;
    float values[kElementsPerVector] = {};
    if (query_start + row < tokens) {
      const float4 packed = *reinterpret_cast<const float4*>(
          query +
          ((query_start + row) * kGlobalQueryHeads + query_head) *
              kGlobalHeadDimension +
          dimension);
      values[0] = packed.x;
      values[1] = packed.y;
      values[2] = packed.z;
      values[3] = packed.w;
    }
#pragma unroll
    for (int element = 0; element < kElementsPerVector; ++element) {
      destination[row * kGlobalHeadDimension +
                  Swizzle(row, dimension + element)] =
          __float2bfloat16_rn(values[element]);
    }
  }
}

__device__ __forceinline__ void StageGlobalFp8Key(
    __nv_bfloat16* destination, const std::uint8_t* chunk,
    const std::uint8_t* cache, float scale, int key_start,
    int max_query_position, int chunk_start, int cache_capacity, int thread) {
  constexpr int kElementsPerVector = 8;
  constexpr int kVectorsPerRow =
      kGlobalHeadDimension / kElementsPerVector;
  for (int chunk_index = thread;
       chunk_index < kGlobalKeyColumns * kVectorsPerRow;
       chunk_index += kGlobalThreads) {
    const int row = chunk_index / kVectorsPerRow;
    const int dimension =
        (chunk_index % kVectorsPerRow) * kElementsPerVector;
    const int absolute_key = key_start + row;
    std::uint32_t words[2] = {};
    if (absolute_key <= max_query_position) {
      const bool in_chunk = absolute_key >= chunk_start;
      const int source_token =
          in_chunk ? absolute_key - chunk_start
                   : absolute_key % cache_capacity;
      const std::uint8_t* source = in_chunk ? chunk : cache;
      const uint2 packed = *reinterpret_cast<const uint2*>(
          source + source_token * kGlobalHeadDimension + dimension);
      words[0] = packed.x;
      words[1] = packed.y;
    }
#pragma unroll
    for (int element = 0; element < kElementsPerVector; ++element) {
      __nv_fp8_e4m3 quantized;
      quantized.__x = static_cast<std::uint8_t>(
          words[element >> 2] >> ((element & 3) * 8));
      destination[row * kGlobalHeadDimension +
                  Swizzle(row, dimension + element)] =
          __float2bfloat16_rn(static_cast<float>(quantized) * scale);
    }
  }
}

__device__ __forceinline__ void StageGlobalFp8Value(
    __nv_bfloat16* destination, const std::uint8_t* chunk,
    const std::uint8_t* cache, float scale, int key_start,
    int max_query_position, int chunk_start, int cache_capacity, int thread) {
  constexpr int kElementsPerVector = 8;
  constexpr int kVectorsPerRow =
      kGlobalHeadDimension / kElementsPerVector;
  for (int chunk_index = thread;
       chunk_index < kGlobalKeyColumns * kVectorsPerRow;
       chunk_index += kGlobalThreads) {
    const int row = chunk_index / kVectorsPerRow;
    const int dimension =
        (chunk_index % kVectorsPerRow) * kElementsPerVector;
    const int output_half = dimension / kGlobalOutputHalf;
    const int half_dimension = dimension % kGlobalOutputHalf;
    const int absolute_key = key_start + row;
    std::uint32_t words[2] = {};
    if (absolute_key <= max_query_position) {
      const bool in_chunk = absolute_key >= chunk_start;
      const int source_token =
          in_chunk ? absolute_key - chunk_start
                   : absolute_key % cache_capacity;
      const std::uint8_t* source = in_chunk ? chunk : cache;
      const uint2 packed = *reinterpret_cast<const uint2*>(
          source + source_token * kGlobalHeadDimension + dimension);
      words[0] = packed.x;
      words[1] = packed.y;
    }
#pragma unroll
    for (int element = 0; element < kElementsPerVector; ++element) {
      __nv_fp8_e4m3 quantized;
      quantized.__x = static_cast<std::uint8_t>(
          words[element >> 2] >> ((element & 3) * 8));
      destination[(output_half * kGlobalKeyColumns + row) *
                      kGlobalOutputHalf +
                  Swizzle(row, half_dimension + element)] =
          __float2bfloat16_rn(static_cast<float>(quantized) * scale);
    }
  }
}

__launch_bounds__(kGlobalThreads, 1) __global__
    void OnlineGlobalAttentionFp8Kernel(
        const float* __restrict__ query,
        const std::uint8_t* __restrict__ chunk_key,
        const std::uint8_t* __restrict__ chunk_value,
        const std::uint8_t* __restrict__ key_cache,
        const std::uint8_t* __restrict__ value_cache,
        const std::uint16_t* __restrict__ key_scale_bf16,
        const std::uint16_t* __restrict__ value_scale_bf16,
        float* __restrict__ output, int start_position, int tokens,
        int cache_capacity) {
  __shared__ __align__(16) __nv_bfloat16 shared[kGlobalSharedElements];
  __nv_bfloat16* query_shared = shared;
  __nv_bfloat16* key_shared =
      query_shared + kGlobalQueryRows * kGlobalHeadDimension;
  __nv_bfloat16* value_shared =
      key_shared + kGlobalKeyColumns * kGlobalHeadDimension;

  const int query_block = static_cast<int>(blockIdx.x);
  const int query_head = static_cast<int>(blockIdx.y);
  const int thread = static_cast<int>(threadIdx.x);
  const int output_half = thread >> 5;
  const int lane = thread & 31;
  const int query_start = query_block * kGlobalQueryRows;
  if (query_head >= kGlobalQueryHeads || query_start >= tokens) return;

  const int group_lane = lane >> 2;
  const int lane_in_group = lane & 3;
  const int a_matrix = lane >> 3;
  const int a_row_in_matrix = lane & 7;
  const int a_row_offset =
      a_row_in_matrix + ((a_matrix & 1) << 3);
  const int b_row_in_matrix = lane & 7;
  const int b_contracting_offset = ((lane >> 3) & 1) << 3;

  const unsigned query_shared_base = SharedAddress(query_shared);
  const unsigned key_shared_base = SharedAddress(key_shared);
  const unsigned value_shared_base = SharedAddress(
      value_shared + output_half * kGlobalKeyColumns * kGlobalOutputHalf);
  const unsigned query_lane_base =
      query_shared_base + static_cast<unsigned>(a_row_offset * 1024);
  const unsigned query_address_select =
      static_cast<unsigned>((a_matrix >> 1) << 4);
  const unsigned query_row_select =
      static_cast<unsigned>(a_row_in_matrix << 4);
  const unsigned key_lane_base =
      key_shared_base + static_cast<unsigned>(b_row_in_matrix * 1024) +
      (static_cast<unsigned>(lane >> 4) << 13);
  const unsigned key_address_select =
      static_cast<unsigned>((b_contracting_offset >> 3) << 4);
  const unsigned key_row_select =
      static_cast<unsigned>(b_row_in_matrix << 4);
  const unsigned value_lane_base =
      value_shared_base +
      static_cast<unsigned>(((lane >> 3) & 1) * 4096) +
      static_cast<unsigned>(b_row_in_matrix * 512);
  const unsigned value_address_select =
      static_cast<unsigned>((lane >> 4) << 4);
  const unsigned value_row_select =
      static_cast<unsigned>(b_row_in_matrix << 4);

  StageGlobalQuery(query_shared, query, query_head, query_start, tokens,
                   thread);

  float accumulator[kGlobalOutputTiles][4];
#pragma unroll
  for (int output_tile = 0; output_tile < kGlobalOutputTiles;
       ++output_tile) {
#pragma unroll
    for (int element = 0; element < 4; ++element) {
      accumulator[output_tile][element] = 0.0F;
    }
  }
  float maximum0 = -CUDART_INF_F;
  float maximum1 = -CUDART_INF_F;
  float sum0 = 0.0F;
  float sum1 = 0.0F;

  const int query_rows = min(kGlobalQueryRows, tokens - query_start);
  const int max_query_position =
      start_position + query_start + query_rows - 1;
  const int key_block_count = max_query_position / kGlobalKeyColumns + 1;
  const float key_scale =
      static_cast<float>(__ushort_as_bfloat16(key_scale_bf16[0]));
  const float value_scale =
      static_cast<float>(__ushort_as_bfloat16(value_scale_bf16[0]));

  for (int key_block = 0; key_block < key_block_count; ++key_block) {
    const int key_start = key_block * kGlobalKeyColumns;
    StageGlobalFp8Key(key_shared, chunk_key, key_cache, key_scale, key_start,
                      max_query_position, start_position, cache_capacity,
                      thread);
    __syncthreads();

    float scores[kGlobalScoreTiles][4];
#pragma unroll
    for (int score_tile = 0; score_tile < kGlobalScoreTiles;
         ++score_tile) {
      scores[score_tile][0] = 0.0F;
      scores[score_tile][1] = 0.0F;
      scores[score_tile][2] = 0.0F;
      scores[score_tile][3] = 0.0F;
    }
    unsigned query_fragments[2][4];
    unsigned key_fragments[2][kGlobalScoreTiles][2];
    LoadMatrixX4(
        query_fragments[0][0], query_fragments[0][1],
        query_fragments[0][2], query_fragments[0][3],
        SwizzledAddress(query_lane_base, 0U, query_address_select,
                         query_row_select));
    LoadMatrixX4(
        key_fragments[0][0][0], key_fragments[0][0][1],
        key_fragments[0][1][0], key_fragments[0][1][1],
        SwizzledAddress(key_lane_base, 0U, key_address_select,
                         key_row_select));
#pragma unroll
    for (int step = 0; step < kGlobalQkSteps; ++step) {
      const int current = step & 1;
      const int next = current ^ 1;
      if (step + 1 < kGlobalQkSteps) {
        const unsigned contracting_offset =
            static_cast<unsigned>((step + 1) << 5);
        LoadMatrixX4(
            query_fragments[next][0], query_fragments[next][1],
            query_fragments[next][2], query_fragments[next][3],
            SwizzledAddress(query_lane_base, contracting_offset,
                             query_address_select, query_row_select));
        LoadMatrixX4(
            key_fragments[next][0][0], key_fragments[next][0][1],
            key_fragments[next][1][0], key_fragments[next][1][1],
            SwizzledAddress(key_lane_base, contracting_offset,
                             key_address_select, key_row_select));
      }
#pragma unroll
      for (int score_tile = 0; score_tile < kGlobalScoreTiles;
           ++score_tile) {
        MmaBf16(
            scores[score_tile][0], scores[score_tile][1],
            scores[score_tile][2], scores[score_tile][3],
            query_fragments[current][0], query_fragments[current][1],
            query_fragments[current][2], query_fragments[current][3],
            key_fragments[current][score_tile][0],
            key_fragments[current][score_tile][1]);
      }
    }

    const int row0 = group_lane;
    const int row1 = row0 + 8;
    const int query_row0 = query_start + row0;
    const int query_row1 = query_start + row1;
    const bool row0_valid = query_row0 < tokens;
    const bool row1_valid = query_row1 < tokens;
    const int query_position0 =
        row0_valid ? start_position + query_row0 : -1;
    const int query_position1 =
        row1_valid ? start_position + query_row1 : -1;
    const bool full_score_tile =
        query_rows == kGlobalQueryRows &&
        key_start + kGlobalKeyColumns - 1 <= start_position + query_start;

    float block_maximum0 = -CUDART_INF_F;
    float block_maximum1 = -CUDART_INF_F;
    if (full_score_tile) {
#pragma unroll
      for (int score_tile = 0; score_tile < kGlobalScoreTiles;
           ++score_tile) {
        block_maximum0 =
            fmaxf(block_maximum0,
                  fmaxf(scores[score_tile][0], scores[score_tile][1]));
        block_maximum1 =
            fmaxf(block_maximum1,
                  fmaxf(scores[score_tile][2], scores[score_tile][3]));
      }
    } else {
#pragma unroll
      for (int score_tile = 0; score_tile < kGlobalScoreTiles;
           ++score_tile) {
        const int key0 =
            key_start + score_tile * 8 + 2 * lane_in_group;
        const int key1 = key0 + 1;
        scores[score_tile][0] =
            row0_valid && key0 <= query_position0 ? scores[score_tile][0]
                                                   : -CUDART_INF_F;
        scores[score_tile][1] =
            row0_valid && key1 <= query_position0 ? scores[score_tile][1]
                                                   : -CUDART_INF_F;
        scores[score_tile][2] =
            row1_valid && key0 <= query_position1 ? scores[score_tile][2]
                                                   : -CUDART_INF_F;
        scores[score_tile][3] =
            row1_valid && key1 <= query_position1 ? scores[score_tile][3]
                                                   : -CUDART_INF_F;
        block_maximum0 =
            fmaxf(block_maximum0,
                  fmaxf(scores[score_tile][0], scores[score_tile][1]));
        block_maximum1 =
            fmaxf(block_maximum1,
                  fmaxf(scores[score_tile][2], scores[score_tile][3]));
      }
    }
    block_maximum0 = WarpMaximum<4>(block_maximum0);
    block_maximum1 = WarpMaximum<4>(block_maximum1);

    const float next_maximum0 = fmaxf(maximum0, block_maximum0);
    const float next_maximum1 = fmaxf(maximum1, block_maximum1);
    const float alpha0 = isfinite(maximum0)
                             ? expf(maximum0 - next_maximum0)
                             : 0.0F;
    const float alpha1 = isfinite(maximum1)
                             ? expf(maximum1 - next_maximum1)
                             : 0.0F;

    float block_sum0 = 0.0F;
    float block_sum1 = 0.0F;
    unsigned probability_fragments[kGlobalPvSteps][4];
#pragma unroll
    for (int score_tile = 0; score_tile < kGlobalScoreTiles;
         ++score_tile) {
      const float probability00 = isfinite(scores[score_tile][0])
                                      ? expf(scores[score_tile][0] -
                                             next_maximum0)
                                      : 0.0F;
      const float probability01 = isfinite(scores[score_tile][1])
                                      ? expf(scores[score_tile][1] -
                                             next_maximum0)
                                      : 0.0F;
      const float probability10 = isfinite(scores[score_tile][2])
                                      ? expf(scores[score_tile][2] -
                                             next_maximum1)
                                      : 0.0F;
      const float probability11 = isfinite(scores[score_tile][3])
                                      ? expf(scores[score_tile][3] -
                                             next_maximum1)
                                      : 0.0F;
      block_sum0 += probability00 + probability01;
      block_sum1 += probability10 + probability11;
      if (score_tile == 0) {
        probability_fragments[0][0] =
            PackBf16x2(probability00, probability01);
        probability_fragments[0][1] =
            PackBf16x2(probability10, probability11);
      } else {
        probability_fragments[0][2] =
            PackBf16x2(probability00, probability01);
        probability_fragments[0][3] =
            PackBf16x2(probability10, probability11);
      }
    }

    sum0 = fmaf(sum0, alpha0, block_sum0);
    sum1 = fmaf(sum1, alpha1, block_sum1);
    maximum0 = next_maximum0;
    maximum1 = next_maximum1;
#pragma unroll
    for (int output_tile = 0; output_tile < kGlobalOutputTiles;
         ++output_tile) {
      accumulator[output_tile][0] *= alpha0;
      accumulator[output_tile][1] *= alpha0;
      accumulator[output_tile][2] *= alpha1;
      accumulator[output_tile][3] *= alpha1;
    }

    StageGlobalFp8Value(value_shared, chunk_value, value_cache, value_scale,
                        key_start, max_query_position, start_position,
                        cache_capacity, thread);
    __syncthreads();

    constexpr int kOutputTilePairs = kGlobalOutputTiles / 2;
    constexpr int kValueLoads = kGlobalPvSteps * kOutputTilePairs;
    unsigned value_fragments[2][4];
    LoadMatrixX4Transposed(
        value_fragments[0][0], value_fragments[0][1],
        value_fragments[0][2], value_fragments[0][3],
        SwizzledAddress(value_lane_base, 0U, value_address_select,
                         value_row_select));
#pragma unroll
    for (int load = 0; load < kValueLoads; ++load) {
      const int output_tile = load * 2;
      const int current = load & 1;
      const int next = current ^ 1;
      if (load + 1 < kValueLoads) {
        const int next_output_tile = (load + 1) * 2;
        LoadMatrixX4Transposed(
            value_fragments[next][0], value_fragments[next][1],
            value_fragments[next][2], value_fragments[next][3],
            SwizzledAddress(value_lane_base,
                             static_cast<unsigned>(next_output_tile << 4),
                             value_address_select, value_row_select));
      }
      MmaBf16(
          accumulator[output_tile][0], accumulator[output_tile][1],
          accumulator[output_tile][2], accumulator[output_tile][3],
          probability_fragments[0][0], probability_fragments[0][1],
          probability_fragments[0][2], probability_fragments[0][3],
          value_fragments[current][0], value_fragments[current][1]);
      MmaBf16(
          accumulator[output_tile + 1][0],
          accumulator[output_tile + 1][1],
          accumulator[output_tile + 1][2],
          accumulator[output_tile + 1][3],
          probability_fragments[0][0], probability_fragments[0][1],
          probability_fragments[0][2], probability_fragments[0][3],
          value_fragments[current][2], value_fragments[current][3]);
    }
  }

  sum0 = WarpSum<4>(sum0);
  sum1 = WarpSum<4>(sum1);
  const float inverse_sum0 = sum0 > 0.0F ? __frcp_rn(sum0) : 0.0F;
  const float inverse_sum1 = sum1 > 0.0F ? __frcp_rn(sum1) : 0.0F;
#pragma unroll
  for (int output_tile = 0; output_tile < kGlobalOutputTiles;
       ++output_tile) {
    const int dimension =
        output_half * kGlobalOutputHalf + output_tile * 8 +
        2 * lane_in_group;
    const int query_row0 = query_start + group_lane;
    const int query_row1 = query_row0 + 8;
    if (query_row0 < tokens) {
      float* output_row =
          output +
          (query_row0 * kGlobalQueryHeads + query_head) *
              kGlobalHeadDimension;
      output_row[dimension] =
          accumulator[output_tile][0] * inverse_sum0;
      output_row[dimension + 1] =
          accumulator[output_tile][1] * inverse_sum0;
    }
    if (query_row1 < tokens) {
      float* output_row =
          output +
          (query_row1 * kGlobalQueryHeads + query_head) *
              kGlobalHeadDimension;
      output_row[dimension] =
          accumulator[output_tile][2] * inverse_sum1;
      output_row[dimension + 1] =
          accumulator[output_tile][3] * inverse_sum1;
    }
  }
}

}  // namespace

Status LaunchOnlineCausalAttentionPrefillFp8LocalSm120(
    const float* query, const std::uint8_t* chunk_key,
    const std::uint8_t* chunk_value, const std::uint8_t* key_cache,
    const std::uint8_t* value_cache, const std::uint16_t* key_scale_bf16,
    const std::uint16_t* value_scale_bf16, float* output,
    std::uint64_t start_position, std::uint64_t tokens,
    std::uint64_t query_heads, std::uint64_t kv_heads,
    std::uint64_t head_dimension, std::uint64_t cache_capacity,
    cudaStream_t stream) {
  if (query == nullptr || chunk_key == nullptr || chunk_value == nullptr ||
      key_cache == nullptr || value_cache == nullptr ||
      key_scale_bf16 == nullptr || value_scale_bf16 == nullptr ||
      output == nullptr || tokens == 0U || query_heads != kQueryHeads ||
      kv_heads != kKvHeads || head_dimension != kHeadDimension ||
      cache_capacity == 0U || tokens > cache_capacity ||
      start_position >
          static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
      tokens > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
      cache_capacity >
          static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
      start_position + tokens - 1U >
          static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    return Invalid("online local FP8 prefill attention arguments are invalid");
  }
  const std::uint64_t query_blocks =
      (tokens + static_cast<std::uint64_t>(kQueryRows) - 1U) /
      static_cast<std::uint64_t>(kQueryRows);
  if (query_blocks >
      static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max())) {
    return Invalid("online local FP8 prefill attention grid exceeds CUDA limits");
  }
  const dim3 grid(static_cast<unsigned>(query_blocks), kQueryHeads);
  OnlineLocalAttentionFp8Kernel<<<grid, kThreads, 0, stream>>>(
      query, chunk_key, chunk_value, key_cache, value_cache, key_scale_bf16,
      value_scale_bf16, output, static_cast<int>(start_position),
      static_cast<int>(tokens), static_cast<int>(cache_capacity));
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch online local FP8 prefill attention", error);
}

Status LaunchOnlineCausalAttentionPrefillFp8GlobalSm120(
    const float* query, const std::uint8_t* chunk_key,
    const std::uint8_t* chunk_value, const std::uint8_t* key_cache,
    const std::uint8_t* value_cache, const std::uint16_t* key_scale_bf16,
    const std::uint16_t* value_scale_bf16, float* output,
    std::uint64_t start_position, std::uint64_t tokens,
    std::uint64_t query_heads, std::uint64_t kv_heads,
    std::uint64_t head_dimension, std::uint64_t cache_capacity,
    cudaStream_t stream) {
  if (query == nullptr || chunk_key == nullptr || chunk_value == nullptr ||
      key_cache == nullptr || value_cache == nullptr ||
      key_scale_bf16 == nullptr || value_scale_bf16 == nullptr ||
      output == nullptr || tokens == 0U ||
      query_heads != kGlobalQueryHeads || kv_heads != kGlobalKvHeads ||
      head_dimension != kGlobalHeadDimension || cache_capacity == 0U ||
      tokens > cache_capacity || start_position >= cache_capacity ||
      start_position >
          static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
      tokens > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
      cache_capacity >
          static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
      start_position + tokens > cache_capacity) {
    return Invalid("online global FP8 prefill attention arguments are invalid");
  }
  const std::uint64_t query_blocks =
      (tokens + static_cast<std::uint64_t>(kGlobalQueryRows) - 1U) /
      static_cast<std::uint64_t>(kGlobalQueryRows);
  if (query_blocks >
      static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max())) {
    return Invalid("online global FP8 prefill attention grid exceeds CUDA limits");
  }
  const dim3 grid(static_cast<unsigned>(query_blocks), kGlobalQueryHeads);
  OnlineGlobalAttentionFp8Kernel<<<grid, kGlobalThreads, 0, stream>>>(
      query, chunk_key, chunk_value, key_cache, value_cache,
      key_scale_bf16, value_scale_bf16, output,
      static_cast<int>(start_position), static_cast<int>(tokens),
      static_cast<int>(cache_capacity));
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch online global FP8 prefill attention", error);
}

}  // namespace gem16gb::internal
