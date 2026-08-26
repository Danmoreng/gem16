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
constexpr int kGlobalHeadDimension = 512;
constexpr int kGlobalQueryRows = 16;
constexpr int kGlobalKeyColumns = 16;
constexpr int kGlobalQueryHeadsPerBlock = 4;
constexpr int kGlobalThreads = 256;
constexpr int kGlobalQueryHeads = 16;
constexpr int kGlobalOutputHalf = 256;
constexpr int kGlobalScoreTiles = kGlobalKeyColumns / 8;
constexpr int kGlobalQkSteps = kGlobalHeadDimension / 16;
constexpr int kGlobalOutputTiles = kGlobalOutputHalf / 8;
constexpr int kGlobalPvSteps = kGlobalKeyColumns / 16;
constexpr int kGlobalOperandElements =
    kGlobalKeyColumns * kGlobalHeadDimension;
constexpr int kGlobalRawBytes = kGlobalKeyColumns * kGlobalHeadDimension;
constexpr int kGlobalSharedBytes =
    (kGlobalQueryHeadsPerBlock * kGlobalQueryRows *
         kGlobalHeadDimension +
     kGlobalOperandElements) *
        sizeof(__nv_bfloat16) +
    2 * kGlobalRawBytes;

static_assert(kGlobalSharedBytes == 96 * 1024);


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
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
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

__device__ __forceinline__ void StageGlobalQuery(
    __nv_bfloat16* destination, const float* query, int query_head_base,
    int query_start, int tokens, int thread) {
  constexpr int kElementsPerVector = 4;
  constexpr int kVectorsPerRow =
      kGlobalHeadDimension / kElementsPerVector;
  constexpr int kVectorsPerHead =
      kGlobalQueryRows * kVectorsPerRow;
  for (int chunk = thread;
       chunk < kGlobalQueryHeadsPerBlock * kVectorsPerHead;
       chunk += kGlobalThreads) {
    const int head = chunk / kVectorsPerHead;
    const int head_chunk = chunk - head * kVectorsPerHead;
    const int row = head_chunk / kVectorsPerRow;
    const int dimension =
        (head_chunk % kVectorsPerRow) * kElementsPerVector;
    float values[kElementsPerVector] = {};
    if (query_start + row < tokens) {
      const float4 packed = *reinterpret_cast<const float4*>(
          query +
          ((query_start + row) * kGlobalQueryHeads +
           query_head_base + head) *
              kGlobalHeadDimension +
          dimension);
      values[0] = packed.x;
      values[1] = packed.y;
      values[2] = packed.z;
      values[3] = packed.w;
    }
#pragma unroll
    for (int element = 0; element < kElementsPerVector; ++element) {
      destination[(head * kGlobalQueryRows + row) *
                      kGlobalHeadDimension +
                  Swizzle(row, dimension + element)] =
          __float2bfloat16_rn(values[element]);
    }
  }
}

template <int KvHeads>
__device__ __forceinline__ void StageGlobalFp8RawAsync(
    std::uint8_t* destination, const std::uint8_t* chunk,
    const std::uint8_t* cache, int key_start,
    int max_query_position, int chunk_start, int kv_head, int thread) {
  constexpr int kElementsPerVector = 16;
  constexpr int kVectorsPerRow =
      kGlobalHeadDimension / kElementsPerVector;
  for (int chunk_index = thread;
       chunk_index < kGlobalKeyColumns * kVectorsPerRow;
       chunk_index += kGlobalThreads) {
    const int row = chunk_index / kVectorsPerRow;
    const int dimension =
        (chunk_index % kVectorsPerRow) * kElementsPerVector;
    const int absolute_key = key_start + row;
    const bool valid = absolute_key <= max_query_position;
    const bool in_chunk = valid && absolute_key >= chunk_start;
    // Global K/V storage is contiguous and the launcher proves every valid
    // absolute key is below cache_capacity; unlike the local ring, no modulo
    // mapping is required here.
    const int source_token =
        in_chunk ? absolute_key - chunk_start : (valid ? absolute_key : 0);
    const std::uint8_t* source = in_chunk ? chunk : cache;
    CopyAsync16(destination + row * kGlobalHeadDimension + dimension,
                source +
                    (source_token * KvHeads + kv_head) *
                        kGlobalHeadDimension +
                    dimension,
                valid ? kElementsPerVector : 0);
  }
}

__device__ __forceinline__ void ConvertGlobalFp8Key(
    __nv_bfloat16* destination, const std::uint8_t* source, float scale,
    int thread) {
  constexpr int kElementsPerVector = 16;
  constexpr int kVectorsPerRow =
      kGlobalHeadDimension / kElementsPerVector;
  for (int chunk_index = thread;
       chunk_index < kGlobalKeyColumns * kVectorsPerRow;
       chunk_index += kGlobalThreads) {
    const int row = chunk_index / kVectorsPerRow;
    const int dimension =
        (chunk_index % kVectorsPerRow) * kElementsPerVector;
    const uint4 packed = *reinterpret_cast<const uint4*>(
        source + row * kGlobalHeadDimension + dimension);
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
      auto* output = destination + row * kGlobalHeadDimension +
                     Swizzle(row, dimension + word * 4);
      *reinterpret_cast<__nv_bfloat162*>(output) =
          __floats2bfloat162_rn(values.x, values.y);
      *reinterpret_cast<__nv_bfloat162*>(output + 2) =
          __floats2bfloat162_rn(values.z, values.w);
    }
  }
}

__device__ __forceinline__ void ConvertGlobalFp8Value(
    __nv_bfloat16* destination, const std::uint8_t* source, float scale,
    int thread) {
  constexpr int kElementsPerVector = 16;
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
    const uint4 packed = *reinterpret_cast<const uint4*>(
        source + row * kGlobalHeadDimension + dimension);
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
      auto* output =
          destination +
          (output_half * kGlobalKeyColumns + row) * kGlobalOutputHalf +
          Swizzle(row, half_dimension + word * 4);
      *reinterpret_cast<__nv_bfloat162*>(output) =
          __floats2bfloat162_rn(values.x, values.y);
      *reinterpret_cast<__nv_bfloat162*>(output + 2) =
          __floats2bfloat162_rn(values.z, values.w);
    }
  }
}

template <int KvHeads>
__global__ void PrepareGlobalBf16KvKernel(
    const std::uint8_t* chunk_key, const std::uint8_t* chunk_value,
    const std::uint8_t* key_cache, const std::uint8_t* value_cache,
    const std::uint16_t* key_scale_bf16,
    const std::uint16_t* value_scale_bf16, __nv_bfloat16* prepared_key,
    __nv_bfloat16* prepared_value, int start_position, int tokens) {
  const std::uint64_t total_tokens =
      static_cast<std::uint64_t>(start_position) + tokens;
  const std::uint64_t elements =
      total_tokens * KvHeads * kGlobalHeadDimension;
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  const std::uint64_t token =
      index / (KvHeads * kGlobalHeadDimension);
  const std::uint64_t row_offset =
      index - token * KvHeads * kGlobalHeadDimension;
  const bool in_chunk = token >= static_cast<std::uint64_t>(start_position);
  const std::uint64_t source_token =
      in_chunk ? token - start_position : token;
  const std::uint64_t source_index =
      source_token * KvHeads * kGlobalHeadDimension + row_offset;
  const std::uint8_t* key_source = in_chunk ? chunk_key : key_cache;
  const std::uint8_t* value_source = in_chunk ? chunk_value : value_cache;
  const float key_scale =
      static_cast<float>(__ushort_as_bfloat16(key_scale_bf16[0]));
  const float value_scale =
      static_cast<float>(__ushort_as_bfloat16(value_scale_bf16[0]));
  prepared_key[index] = __float2bfloat16_rn(
      DecodeFp8(key_source[source_index], key_scale));
  prepared_value[index] = __float2bfloat16_rn(
      DecodeFp8(value_source[source_index], value_scale));
}

template <int KvHeads>
__device__ __forceinline__ void StageGlobalBf16KeyAsync(
    __nv_bfloat16* destination, const __nv_bfloat16* source,
    int key_start, int max_query_position, int kv_head, int thread) {
  constexpr int kElementsPerVector = 8;
  constexpr int kVectorsPerRow = kGlobalHeadDimension / kElementsPerVector;
  for (int chunk_index = thread;
       chunk_index < kGlobalKeyColumns * kVectorsPerRow;
       chunk_index += kGlobalThreads) {
    const int row = chunk_index / kVectorsPerRow;
    const int dimension =
        (chunk_index % kVectorsPerRow) * kElementsPerVector;
    const int absolute_key = key_start + row;
    const bool valid = absolute_key <= max_query_position;
    CopyAsync16(
        destination + row * kGlobalHeadDimension + Swizzle(row, dimension),
        source +
            (static_cast<std::uint64_t>(valid ? absolute_key : 0) * KvHeads +
             kv_head) *
                kGlobalHeadDimension +
            dimension,
        valid ? 16 : 0);
  }
}

template <int KvHeads>
__device__ __forceinline__ void StageGlobalBf16ValueAsync(
    __nv_bfloat16* destination, const __nv_bfloat16* source,
    int key_start, int max_query_position, int kv_head, int thread) {
  constexpr int kElementsPerVector = 8;
  constexpr int kVectorsPerRow = kGlobalHeadDimension / kElementsPerVector;
  for (int chunk_index = thread;
       chunk_index < kGlobalKeyColumns * kVectorsPerRow;
       chunk_index += kGlobalThreads) {
    const int row = chunk_index / kVectorsPerRow;
    const int dimension =
        (chunk_index % kVectorsPerRow) * kElementsPerVector;
    const int output_half = dimension / kGlobalOutputHalf;
    const int half_dimension = dimension % kGlobalOutputHalf;
    const int absolute_key = key_start + row;
    const bool valid = absolute_key <= max_query_position;
    CopyAsync16(
        destination +
            (output_half * kGlobalKeyColumns + row) * kGlobalOutputHalf +
            Swizzle(row, half_dimension),
        source +
            (static_cast<std::uint64_t>(valid ? absolute_key : 0) * KvHeads +
             kv_head) *
                kGlobalHeadDimension +
            dimension,
        valid ? 16 : 0);
  }
}

template <int KvHeads, bool kPrepared>
__launch_bounds__(kGlobalThreads, 1) __global__
    void OnlineGlobalAttentionFp8Kernel(
        const float* __restrict__ query,
        const std::uint8_t* __restrict__ chunk_key,
        const std::uint8_t* __restrict__ chunk_value,
        const std::uint8_t* __restrict__ key_cache,
        const std::uint8_t* __restrict__ value_cache,
        const std::uint16_t* __restrict__ key_scale_bf16,
        const std::uint16_t* __restrict__ value_scale_bf16,
        std::uint16_t* __restrict__ output, int start_position, int tokens,
        int cache_capacity) {
  __shared__ __align__(16) std::uint8_t shared[kGlobalSharedBytes];
  __nv_bfloat16* query_shared =
      reinterpret_cast<__nv_bfloat16*>(shared);
  __nv_bfloat16* operand_shared =
      query_shared + kGlobalQueryHeadsPerBlock * kGlobalQueryRows *
                         kGlobalHeadDimension;
  __nv_bfloat16* key_shared = operand_shared;
  __nv_bfloat16* value_shared =
      operand_shared + (kPrepared ? kGlobalOperandElements : 0);
  std::uint8_t* raw_shared = reinterpret_cast<std::uint8_t*>(
      operand_shared + (kPrepared ? 2 * kGlobalOperandElements
                                  : kGlobalOperandElements));

  const int query_block = static_cast<int>(blockIdx.x);
  const int query_head_base =
      static_cast<int>(blockIdx.y) * kGlobalQueryHeadsPerBlock;
  static_assert(kGlobalQueryHeads % KvHeads == 0);
  const int kv_head = query_head_base / (kGlobalQueryHeads / KvHeads);
  const int thread = static_cast<int>(threadIdx.x);
  const int head_in_block = thread >> 6;
  const int query_head = query_head_base + head_in_block;
  const int output_half = (thread >> 5) & 1;
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

  const unsigned query_shared_base = SharedAddress(
      query_shared + head_in_block * kGlobalQueryRows *
                         kGlobalHeadDimension);
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

  StageGlobalQuery(query_shared, query, query_head_base, query_start, tokens,
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
  const int key_block_count =
      max_query_position / kGlobalKeyColumns + 1;
  const float key_scale =
      static_cast<float>(__ushort_as_bfloat16(key_scale_bf16[0]));
  const float value_scale =
      static_cast<float>(__ushort_as_bfloat16(value_scale_bf16[0]));

  if constexpr (kPrepared) {
    StageGlobalBf16KeyAsync<KvHeads>(
        key_shared, reinterpret_cast<const __nv_bfloat16*>(key_cache), 0,
        max_query_position, kv_head, thread);
  } else {
    StageGlobalFp8RawAsync<KvHeads>(
        raw_shared, chunk_key, key_cache, 0, max_query_position,
        start_position, kv_head, thread);
  }
  CommitAsyncCopies();
  WaitForAsyncCopies();
  __syncthreads();
  if constexpr (!kPrepared) {
    ConvertGlobalFp8Key(key_shared, raw_shared, key_scale, thread);
    __syncthreads();
  }

  for (int key_block = 0; key_block < key_block_count; ++key_block) {
    const int key_start = key_block * kGlobalKeyColumns;
    const int current_raw = key_block & 1;
    const int next_raw = current_raw ^ 1;
    if constexpr (kPrepared) {
      StageGlobalBf16ValueAsync<KvHeads>(
          value_shared, reinterpret_cast<const __nv_bfloat16*>(value_cache),
          key_start, max_query_position, kv_head, thread);
    } else {
      StageGlobalFp8RawAsync<KvHeads>(
          raw_shared + current_raw * kGlobalRawBytes, chunk_value, value_cache,
          key_start, max_query_position, start_position, kv_head, thread);
    }
    CommitAsyncCopies();

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

    WaitForAsyncCopies();
    __syncthreads();
    if constexpr (!kPrepared) {
      ConvertGlobalFp8Value(
          value_shared, raw_shared + current_raw * kGlobalRawBytes,
          value_scale, thread);
      __syncthreads();
    }

    const bool has_next_key = key_block + 1 < key_block_count;
    if (has_next_key) {
      if constexpr (kPrepared) {
        StageGlobalBf16KeyAsync<KvHeads>(
            key_shared, reinterpret_cast<const __nv_bfloat16*>(key_cache),
            key_start + kGlobalKeyColumns, max_query_position, kv_head,
            thread);
      } else {
        StageGlobalFp8RawAsync<KvHeads>(
            raw_shared + next_raw * kGlobalRawBytes, chunk_key, key_cache,
            key_start + kGlobalKeyColumns, max_query_position, start_position,
            kv_head, thread);
      }
      CommitAsyncCopies();
    }

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
    if (has_next_key) {
      WaitForAsyncCopies();
      __syncthreads();
      if constexpr (!kPrepared) {
        ConvertGlobalFp8Key(
            key_shared, raw_shared + next_raw * kGlobalRawBytes, key_scale,
            thread);
        __syncthreads();
      }
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
      std::uint16_t* output_row =
          output +
          (query_row0 * kGlobalQueryHeads + query_head) *
              kGlobalHeadDimension;
      *reinterpret_cast<__nv_bfloat162*>(output_row + dimension) =
          __floats2bfloat162_rn(accumulator[output_tile][0] * inverse_sum0,
                               accumulator[output_tile][1] * inverse_sum0);
    }
    if (query_row1 < tokens) {
      std::uint16_t* output_row =
          output +
          (query_row1 * kGlobalQueryHeads + query_head) *
              kGlobalHeadDimension;
      *reinterpret_cast<__nv_bfloat162*>(output_row + dimension) =
          __floats2bfloat162_rn(accumulator[output_tile][2] * inverse_sum1,
                               accumulator[output_tile][3] * inverse_sum1);
    }
  }
}


}  // namespace

Status LaunchOnlineCausalAttentionPrefillFp8GlobalSm120(
    const float* query, const std::uint8_t* chunk_key,
    const std::uint8_t* chunk_value, const std::uint8_t* key_cache,
    const std::uint8_t* value_cache, const std::uint16_t* key_scale_bf16,
    const std::uint16_t* value_scale_bf16, std::uint16_t* output_bf16,
    std::uint64_t start_position, std::uint64_t tokens,
    std::uint64_t query_heads, std::uint64_t kv_heads,
    std::uint64_t head_dimension, std::uint64_t cache_capacity,
    cudaStream_t stream) {
  if (query == nullptr || chunk_key == nullptr || chunk_value == nullptr ||
      key_cache == nullptr || value_cache == nullptr ||
      key_scale_bf16 == nullptr || value_scale_bf16 == nullptr ||
      output_bf16 == nullptr || tokens == 0U ||
      query_heads != kGlobalQueryHeads ||
      (kv_heads != 1U && kv_heads != 2U) ||
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
  static_assert(kGlobalQueryHeads % kGlobalQueryHeadsPerBlock == 0);
  const dim3 grid(static_cast<unsigned>(query_blocks),
                  kGlobalQueryHeads / kGlobalQueryHeadsPerBlock);
  if (kv_heads == 1U) {
    OnlineGlobalAttentionFp8Kernel<1, false><<<grid, kGlobalThreads, 0, stream>>>(
        query, chunk_key, chunk_value, key_cache, value_cache,
        key_scale_bf16, value_scale_bf16, output_bf16,
        static_cast<int>(start_position), static_cast<int>(tokens),
        static_cast<int>(cache_capacity));
  } else {
    OnlineGlobalAttentionFp8Kernel<2, false><<<grid, kGlobalThreads, 0, stream>>>(
        query, chunk_key, chunk_value, key_cache, value_cache,
        key_scale_bf16, value_scale_bf16, output_bf16,
        static_cast<int>(start_position), static_cast<int>(tokens),
        static_cast<int>(cache_capacity));
  }
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch online global FP8 prefill attention", error);
}

Status LaunchOnlineCausalAttentionPrefillFp8GlobalPreparedSm120(
    const float* query, const std::uint8_t* chunk_key,
    const std::uint8_t* chunk_value, const std::uint8_t* key_cache,
    const std::uint8_t* value_cache, const std::uint16_t* key_scale_bf16,
    const std::uint16_t* value_scale_bf16,
    std::uint16_t* prepared_key_bf16,
    std::uint16_t* prepared_value_bf16, std::uint16_t* output_bf16,
    std::uint64_t start_position, std::uint64_t tokens,
    std::uint64_t query_heads, std::uint64_t kv_heads,
    std::uint64_t head_dimension, std::uint64_t cache_capacity,
    std::uint64_t prepared_capacity, cudaStream_t stream) {
  if (query == nullptr || chunk_key == nullptr || chunk_value == nullptr ||
      key_cache == nullptr || value_cache == nullptr ||
      key_scale_bf16 == nullptr || value_scale_bf16 == nullptr ||
      prepared_key_bf16 == nullptr || prepared_value_bf16 == nullptr ||
      output_bf16 == nullptr || tokens == 0U ||
      query_heads != kGlobalQueryHeads || kv_heads != 2U ||
      head_dimension != kGlobalHeadDimension || cache_capacity == 0U ||
      tokens > std::numeric_limits<std::uint64_t>::max() - start_position) {
    return Invalid("prepared global FP8 prefill attention arguments are invalid");
  }
  const std::uint64_t total_tokens = start_position + tokens;
  if (total_tokens > cache_capacity || total_tokens > prepared_capacity ||
      total_tokens > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    return Invalid("prepared global FP8 prefill attention arguments are invalid");
  }
  const std::uint64_t kv_elements =
      total_tokens * kv_heads * head_dimension;
  constexpr unsigned kPrepareThreads = 256U;
  const std::uint64_t prepare_blocks =
      (kv_elements + kPrepareThreads - 1U) / kPrepareThreads;
  if (prepare_blocks >
      static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max())) {
    return Invalid("prepared global FP8 prefill grid exceeds CUDA limits");
  }
  PrepareGlobalBf16KvKernel<2><<<static_cast<unsigned>(prepare_blocks),
                                      kPrepareThreads, 0, stream>>>(
      chunk_key, chunk_value, key_cache, value_cache, key_scale_bf16,
      value_scale_bf16,
      reinterpret_cast<__nv_bfloat16*>(prepared_key_bf16),
      reinterpret_cast<__nv_bfloat16*>(prepared_value_bf16),
      static_cast<int>(start_position), static_cast<int>(tokens));
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaFailure("prepare global BF16 K/V staging", error);
  }
  const std::uint64_t query_blocks =
      (tokens + static_cast<std::uint64_t>(kGlobalQueryRows) - 1U) /
      static_cast<std::uint64_t>(kGlobalQueryRows);
  const dim3 grid(static_cast<unsigned>(query_blocks),
                  kGlobalQueryHeads / kGlobalQueryHeadsPerBlock);
  OnlineGlobalAttentionFp8Kernel<2, true><<<grid, kGlobalThreads, 0, stream>>>(
      query, nullptr, nullptr,
      reinterpret_cast<const std::uint8_t*>(prepared_key_bf16),
      reinterpret_cast<const std::uint8_t*>(prepared_value_bf16),
      key_scale_bf16, value_scale_bf16, output_bf16,
      static_cast<int>(start_position), static_cast<int>(tokens),
      static_cast<int>(cache_capacity));
  error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch prepared global FP8 prefill attention",
                           error);
}


}  // namespace gem16::internal
