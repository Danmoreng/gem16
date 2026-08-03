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

constexpr int kHeadDimension = 256;
constexpr int kQueryRows = 32;
constexpr int kKeyColumns = 32;
constexpr int kQueryHeads = 16;
constexpr int kKvHeads = 8;
constexpr int kGroupSize = kQueryHeads / kKvHeads;
constexpr int kQueryHeadsPerBlock = kGroupSize;
constexpr int kThreads = 128;
constexpr int kScoreTiles = kKeyColumns / 8;
constexpr int kQkSteps = kHeadDimension / 16;
constexpr int kOutputTiles = kHeadDimension / 8;
constexpr int kPvSteps = kKeyColumns / 16;
constexpr int kOperandElements = kKeyColumns * kHeadDimension;
constexpr int kRawBytes = kKeyColumns * kHeadDimension;
constexpr int kSharedBytes =
    (kQueryHeadsPerBlock * kQueryRows * kHeadDimension +
     kOperandElements) * sizeof(__nv_bfloat16) + 2 * kRawBytes;

static_assert(kSharedBytes == 64 * 1024);
constexpr unsigned kFullWarpMask = 0xffffffffU;

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

__device__ __forceinline__ int WindowStart(int query_position,
                                            int cache_capacity) {
  return max(0, query_position + 1 - cache_capacity);
}

__device__ __forceinline__ int QueryLimit(
    int query_position, int vision_begin, int vision_end) {
  return query_position >= vision_begin && query_position < vision_end
             ? vision_end - 1
             : query_position;
}

__device__ __forceinline__ void StageQuery(
    __nv_bfloat16* destination, const float* query, int query_head_base,
    int query_start, int tokens, int thread) {
  constexpr int kElementsPerVector = 4;
  constexpr int kVectorsPerRow = kHeadDimension / kElementsPerVector;
  constexpr int kVectorsPerHead = kQueryRows * kVectorsPerRow;
  for (int chunk = thread;
       chunk < kQueryHeadsPerBlock * kVectorsPerHead;
       chunk += kThreads) {
    const int head = chunk / kVectorsPerHead;
    const int head_chunk = chunk - head * kVectorsPerHead;
    const int row = head_chunk / kVectorsPerRow;
    const int dimension =
        (head_chunk % kVectorsPerRow) * kElementsPerVector;
    float values[kElementsPerVector] = {};
    if (query_start + row < tokens) {
      const float4 packed = *reinterpret_cast<const float4*>(
          query + ((query_start + row) * kQueryHeads +
                   query_head_base + head) *
                      kHeadDimension +
          dimension);
      values[0] = packed.x;
      values[1] = packed.y;
      values[2] = packed.z;
      values[3] = packed.w;
    }
#pragma unroll
    for (int element = 0; element < kElementsPerVector; ++element) {
      destination[(head * kQueryRows + row) * kHeadDimension +
                  Swizzle(row, dimension + element)] =
          __float2bfloat16_rn(values[element]);
    }
  }
}

__device__ __forceinline__ void StageLocalFp8RawAsync(
    std::uint8_t* destination, const std::uint8_t* chunk,
    const std::uint8_t* cache, int kv_head, int key_start,
    int max_query_position, int chunk_start, int cache_capacity, int thread) {
  constexpr int kElementsPerVector = 16;
  constexpr int kVectorsPerRow = kHeadDimension / kElementsPerVector;
  for (int chunk_index = thread;
       chunk_index < kKeyColumns * kVectorsPerRow;
       chunk_index += kThreads) {
    const int row = chunk_index / kVectorsPerRow;
    const int dimension =
        (chunk_index % kVectorsPerRow) * kElementsPerVector;
    const int absolute_key = key_start + row;
    const bool valid = absolute_key <= max_query_position;
    const bool in_chunk = valid && absolute_key >= chunk_start;
    const int source_token =
        in_chunk ? absolute_key - chunk_start
                 : (valid ? absolute_key % cache_capacity : 0);
    const std::uint8_t* source = in_chunk ? chunk : cache;
    CopyAsync16(
        destination + row * kHeadDimension + dimension,
        source + ((source_token * kKvHeads + kv_head) * kHeadDimension) +
            dimension,
        valid ? kElementsPerVector : 0);
  }
}

__device__ __forceinline__ void ConvertLocalFp8(
    __nv_bfloat16* destination, const std::uint8_t* source, float scale,
    int thread) {
  constexpr int kElementsPerVector = 16;
  constexpr int kVectorsPerRow = kHeadDimension / kElementsPerVector;
  for (int chunk_index = thread;
       chunk_index < kKeyColumns * kVectorsPerRow;
       chunk_index += kThreads) {
    const int row = chunk_index / kVectorsPerRow;
    const int dimension =
        (chunk_index % kVectorsPerRow) * kElementsPerVector;
    const uint4 packed = *reinterpret_cast<const uint4*>(
        source + row * kHeadDimension + dimension);
    const std::uint32_t words[4] = {
        packed.x, packed.y, packed.z, packed.w};
#pragma unroll
    for (int word = 0; word < 4; ++word) {
      __nv_fp8x4_e4m3 quantized;
      quantized.__x = words[word];
      float4 values = static_cast<float4>(quantized);
      values.x *= scale;
      values.y *= scale;
      values.z *= scale;
      values.w *= scale;
      auto* output = destination + row * kHeadDimension +
                     Swizzle(row, dimension + word * 4);
      *reinterpret_cast<__nv_bfloat162*>(output) =
          __floats2bfloat162_rn(values.x, values.y);
      *reinterpret_cast<__nv_bfloat162*>(output + 2) =
          __floats2bfloat162_rn(values.z, values.w);
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
    std::uint16_t* __restrict__ output, int start_position, int tokens,
    int cache_capacity, int vision_begin, int vision_end) {
  __shared__ __align__(16) std::uint8_t shared[kSharedBytes];
  __nv_bfloat16* query_shared =
      reinterpret_cast<__nv_bfloat16*>(shared);
  __nv_bfloat16* operand_shared =
      query_shared + kQueryHeadsPerBlock * kQueryRows * kHeadDimension;
  __nv_bfloat16* key_shared = operand_shared;
  __nv_bfloat16* value_shared = operand_shared;
  std::uint8_t* raw_shared = reinterpret_cast<std::uint8_t*>(
      operand_shared + kOperandElements);

  const int query_block = static_cast<int>(blockIdx.x);
  const int query_head_base =
      static_cast<int>(blockIdx.y) * kQueryHeadsPerBlock;
  const int thread = static_cast<int>(threadIdx.x);
  const int head_in_block = thread >> 6;
  const int query_head = query_head_base + head_in_block;
  const int warp_in_head = (thread >> 5) & 1;
  const int lane = thread & 31;
  const int query_start = query_block * kQueryRows;
  const int kv_head = static_cast<int>(blockIdx.y);
  const int warp_row_start = warp_in_head * 16;
  if (query_head >= kQueryHeads || query_start >= tokens) return;

  const int group_lane = lane >> 2;
  const int lane_in_group = lane & 3;
  const int a_matrix = lane >> 3;
  const int a_row_in_matrix = lane & 7;
  const int a_row_offset =
      a_row_in_matrix + ((a_matrix & 1) << 3);
  const int b_row_in_matrix = lane & 7;
  const int b_contracting_offset = ((lane >> 3) & 1) << 3;

  const unsigned query_shared_base = SharedAddress(
      query_shared + head_in_block * kQueryRows * kHeadDimension);
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

  StageQuery(query_shared, query, query_head_base, query_start, tokens,
             thread);

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
  const int block_first_position = start_position + query_start;
  const int block_last_position = block_first_position + query_rows - 1;
  const bool block_has_vision = vision_begin < vision_end &&
      block_first_position < vision_end && block_last_position >= vision_begin;
  const int max_query_position = block_has_vision
      ? max(block_last_position, vision_end - 1)
      : block_last_position;
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

  StageLocalFp8RawAsync(raw_shared, chunk_key, key_cache, kv_head,
                        first_key_start, max_query_position, start_position,
                        cache_capacity, thread);
  CommitAsyncCopies();
  WaitForAsyncCopies();
  __syncthreads();
  ConvertLocalFp8(key_shared, raw_shared, key_scale, thread);
  __syncthreads();

  for (int key_block = 0; key_block < key_block_count; ++key_block) {
    const int key_start = first_key_start + key_block * kKeyColumns;
    const int current_raw = key_block & 1;
    const int next_raw = current_raw ^ 1;
    StageLocalFp8RawAsync(raw_shared + current_raw * kRawBytes,
                          chunk_value, value_cache, kv_head, key_start,
                          max_query_position, start_position, cache_capacity,
                          thread);
    CommitAsyncCopies();

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
    const int query_limit0 = row0_valid
        ? QueryLimit(query_position0, vision_begin, vision_end) : -1;
    const int query_limit1 = row1_valid
        ? QueryLimit(query_position1, vision_begin, vision_end) : -1;
    const int latest_window = WindowStart(
        start_position + query_start + kQueryRows - 1, cache_capacity);
    const bool full_score_tile = !block_has_vision &&
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
            row0_valid && key0 >= window0 && key0 <= query_limit0
                ? scores[score_tile][0]
                : -CUDART_INF_F;
        scores[score_tile][1] =
            row0_valid && key1 >= window0 && key1 <= query_limit0
                ? scores[score_tile][1]
                : -CUDART_INF_F;
        scores[score_tile][2] =
            row1_valid && key0 >= window1 && key0 <= query_limit1
                ? scores[score_tile][2]
                : -CUDART_INF_F;
        scores[score_tile][3] =
            row1_valid && key1 >= window1 && key1 <= query_limit1
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

    WaitForAsyncCopies();
    __syncthreads();
    ConvertLocalFp8(value_shared,
                    raw_shared + current_raw * kRawBytes,
                    value_scale, thread);
    __syncthreads();

    const bool has_next_key = key_block + 1 < key_block_count;
    if (has_next_key) {
      StageLocalFp8RawAsync(
          raw_shared + next_raw * kRawBytes, chunk_key, key_cache, kv_head,
          key_start + kKeyColumns, max_query_position, start_position,
          cache_capacity, thread);
      CommitAsyncCopies();
    }

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
    if (has_next_key) {
      WaitForAsyncCopies();
      __syncthreads();
      ConvertLocalFp8(key_shared, raw_shared + next_raw * kRawBytes,
                      key_scale, thread);
      __syncthreads();
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
      std::uint16_t* output_row =
          output + (query_row0 * kQueryHeads + query_head) * kHeadDimension;
      *reinterpret_cast<__nv_bfloat162*>(output_row + dimension) =
          __floats2bfloat162_rn(accumulator[output_tile][0] * inverse_sum0,
                               accumulator[output_tile][1] * inverse_sum0);
    }
    if (query_row1 < tokens) {
      std::uint16_t* output_row =
          output + (query_row1 * kQueryHeads + query_head) * kHeadDimension;
      *reinterpret_cast<__nv_bfloat162*>(output_row + dimension) =
          __floats2bfloat162_rn(accumulator[output_tile][2] * inverse_sum1,
                               accumulator[output_tile][3] * inverse_sum1);
    }
  }
}


}  // namespace

Status LaunchOnlineCausalAttentionPrefillFp8LocalSm120(
    const float* query, const std::uint8_t* chunk_key,
    const std::uint8_t* chunk_value, const std::uint8_t* key_cache,
    const std::uint8_t* value_cache, const std::uint16_t* key_scale_bf16,
    const std::uint16_t* value_scale_bf16, std::uint16_t* output_bf16,
    std::uint64_t start_position, std::uint64_t tokens,
    std::uint64_t query_heads, std::uint64_t kv_heads,
    std::uint64_t head_dimension, std::uint64_t cache_capacity,
    cudaStream_t stream, std::uint64_t vision_begin,
    std::uint64_t vision_end) {
  if (query == nullptr || chunk_key == nullptr || chunk_value == nullptr ||
      key_cache == nullptr || value_cache == nullptr ||
      key_scale_bf16 == nullptr || value_scale_bf16 == nullptr ||
      output_bf16 == nullptr || tokens == 0U || query_heads != kQueryHeads ||
      kv_heads != kKvHeads || head_dimension != kHeadDimension ||
      cache_capacity == 0U ||
      start_position >
          static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
      tokens > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
      cache_capacity >
          static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
      start_position + tokens - 1U >
          static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
      vision_begin > vision_end || vision_end > start_position + tokens) {
    return Invalid("online local FP8 prefill attention arguments are invalid");
  }
  const std::uint64_t query_blocks =
      (tokens + static_cast<std::uint64_t>(kQueryRows) - 1U) /
      static_cast<std::uint64_t>(kQueryRows);
  if (query_blocks >
      static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max())) {
    return Invalid("online local FP8 prefill attention grid exceeds CUDA limits");
  }
  static_assert(kQueryHeads % kQueryHeadsPerBlock == 0);
  const dim3 grid(static_cast<unsigned>(query_blocks),
                  kQueryHeads / kQueryHeadsPerBlock);
  OnlineLocalAttentionFp8Kernel<<<grid, kThreads, 0, stream>>>(
      query, chunk_key, chunk_value, key_cache, value_cache, key_scale_bf16,
      value_scale_bf16, output_bf16, static_cast<int>(start_position),
      static_cast<int>(tokens), static_cast<int>(cache_capacity),
      static_cast<int>(vision_begin), static_cast<int>(vision_end));
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch online local FP8 prefill attention", error);
}


}  // namespace gem16::internal
