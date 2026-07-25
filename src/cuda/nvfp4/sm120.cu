#include "cuda/nvfp4/sm120.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

namespace gem16gb::internal {
namespace {

constexpr std::uint64_t kElementsPerKBlock = 64;
constexpr std::uint64_t kRowsPerWarp = 8;
constexpr std::uint64_t kTokensPerMma = 16;
constexpr std::uint64_t kPrefillTokenTilesPerWarp = 8;
constexpr std::uint64_t kFusedGateUpTokenTilesPerWarp = 2;
constexpr unsigned kWarpSize = 32;
constexpr unsigned kWarpsPerBlock = 4;
constexpr unsigned kThreadsPerBlock = kWarpSize * kWarpsPerBlock;
constexpr unsigned kPrefillWarpsPerBlock = 8;
constexpr unsigned kPrefillThreadsPerBlock =
    kWarpSize * kPrefillWarpsPerBlock;
constexpr float kSqrtTwoOverPi = 0.7978845608028654F;
constexpr float kGeluCubic = 0.044715F;

Status Invalid(std::string message) {
  return Status(StatusCode::kInvalidArgument, std::move(message));
}

Status CudaFailure(const char* operation, cudaError_t error) {
  return Status(StatusCode::kInternal,
                std::string(operation) + ": " + cudaGetErrorName(error) + ": " +
                    cudaGetErrorString(error));
}

bool PositiveFinite(float value) {
  return std::isfinite(value) && value > 0.0F;
}

__device__ __forceinline__ std::uint32_t LoadU32(const std::uint8_t* source) {
  return *reinterpret_cast<const std::uint32_t*>(source);
}

__device__ __forceinline__ unsigned SharedAddress(const void* pointer) {
  return static_cast<unsigned>(__cvta_generic_to_shared(pointer));
}

template <unsigned kBytes>
__device__ __forceinline__ void CopyAsyncZeroFill(
    void* shared_destination, const void* global_source, int source_bytes) {
  static_assert(kBytes == 4U || kBytes == 16U);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n"
               :
               : "r"(SharedAddress(shared_destination)), "l"(global_source),
                 "n"(kBytes), "r"(source_bytes));
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

__device__ __forceinline__ void StageNvfp4ActivationAsync(
    std::uint32_t* staged_activation,
    std::uint32_t* staged_activation_scales,
    const std::uint8_t* packed_activation_e2m1,
    const std::uint8_t* activation_scales_e4m3fn,
    std::uint64_t token_base, std::uint64_t k_block,
    std::uint64_t packed_row_bytes, std::uint64_t scale_row_bytes,
    std::uint64_t tokens, unsigned staged_tokens) {
  constexpr unsigned kPackedWordsPerKBlock =
      static_cast<unsigned>(kElementsPerKBlock / 8U);
  constexpr unsigned kAsyncWords = 4U;
  constexpr unsigned kCopiesPerToken =
      kPackedWordsPerKBlock / kAsyncWords;
  for (unsigned copy = threadIdx.x;
       copy < staged_tokens * kCopiesPerToken; copy += blockDim.x) {
    const unsigned token_offset = copy / kCopiesPerToken;
    const unsigned half = copy % kCopiesPerToken;
    const std::uint64_t token = token_base + token_offset;
    const bool valid = token < tokens;
    const std::uint8_t* source = valid
        ? packed_activation_e2m1 + token * packed_row_bytes +
              k_block * 32U + static_cast<std::uint64_t>(half) * 16U
        : packed_activation_e2m1;
    CopyAsyncZeroFill<16U>(
        staged_activation + token_offset * kPackedWordsPerKBlock +
            half * kAsyncWords,
        source, valid ? 16 : 0);
  }
  for (unsigned token_offset = threadIdx.x; token_offset < staged_tokens;
       token_offset += blockDim.x) {
    const std::uint64_t token = token_base + token_offset;
    const bool valid = token < tokens;
    const std::uint8_t* source = valid
        ? activation_scales_e4m3fn + token * scale_row_bytes +
              k_block * 4U
        : activation_scales_e4m3fn;
    CopyAsyncZeroFill<4U>(staged_activation_scales + token_offset,
                          source, valid ? 4 : 0);
  }
}

__device__ __forceinline__ void MmaNvfp4(
    float& d0, float& d1, float& d2, float& d3, std::uint32_t a0,
    std::uint32_t a1, std::uint32_t a2, std::uint32_t a3,
    std::uint32_t b0, std::uint32_t b1, std::uint32_t scale_a,
    std::uint32_t scale_b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  constexpr std::uint16_t instruction_byte_id = 0;
  constexpr std::uint16_t instruction_thread_id = 0;
  asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, "
      "{%8, %9}, "
      "{%10, %11, %12, %13}, "
      "%14, {%16, %17}, "
      "%15, {%16, %17};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
        "f"(d0), "f"(d1), "f"(d2), "f"(d3), "r"(scale_a), "r"(scale_b),
        "h"(instruction_byte_id), "h"(instruction_thread_id));
#else
  (void)d0;
  (void)d1;
  (void)d2;
  (void)d3;
  (void)a0;
  (void)a1;
  (void)a2;
  (void)a3;
  (void)b0;
  (void)b1;
  (void)scale_a;
  (void)scale_b;
#endif
}

__global__ void Sm120DirectProjectionKernel(const std::uint8_t* packed_activation_e2m1,
                                            const std::uint8_t* activation_scales_e4m3fn,
                                            const std::uint8_t* packed_weight_e2m1,
                                            const std::uint8_t* weight_scales_e4m3fn,
                                            float* output,
                                            std::uint64_t tokens,
                                            std::uint64_t rows,
                                            std::uint64_t contracting_elements,
                                            float output_divisor) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  const std::uint64_t token = blockIdx.y;
  if (token >= tokens) return;
  const std::uint64_t packed_token_bytes = contracting_elements / 2U;
  const std::uint64_t scale_token_bytes = contracting_elements / 16U;
  packed_activation_e2m1 += token * packed_token_bytes;
  activation_scales_e4m3fn += token * scale_token_bytes;
  output += token * rows;
  const unsigned warp = threadIdx.x / kWarpSize;
  const unsigned lane = threadIdx.x & (kWarpSize - 1U);
  const std::uint64_t global_warp =
      static_cast<std::uint64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  const std::uint64_t row_tiles = (rows + kRowsPerWarp - 1U) / kRowsPerWarp;
  if (global_warp >= row_tiles) return;

  const unsigned row_in_tile = lane >> 2U;
  const unsigned k_quarter = lane & 3U;
  const std::uint64_t source_row = global_warp * kRowsPerWarp + row_in_tile;
  const std::uint64_t packed_row_bytes = contracting_elements / 2U;
  const std::uint64_t scale_row_bytes = contracting_elements / 16U;
  const std::uint64_t k_blocks = contracting_elements / kElementsPerKBlock;

  float d0 = 0.0F;
  float d1 = 0.0F;
  float d2 = 0.0F;
  float d3 = 0.0F;
  constexpr std::uint16_t instruction_block_id = 0;
  constexpr std::uint16_t instruction_thread_id = 0;

  for (std::uint64_t k_block = 0; k_block < k_blocks; ++k_block) {
    const std::uint64_t activation_byte = k_block * 32U + k_quarter * 4U;
    const std::uint32_t a_first = LoadU32(packed_activation_e2m1 + activation_byte);
    const std::uint32_t a_second = LoadU32(packed_activation_e2m1 + activation_byte + 16U);

    std::uint32_t b_first = 0;
    std::uint32_t b_second = 0;
    std::uint32_t scale_b = 0;
    if (source_row < rows) {
      const std::uint64_t weight_byte = source_row * packed_row_bytes + k_block * 32U +
                                        static_cast<std::uint64_t>(k_quarter) * 4U;
      b_first = LoadU32(packed_weight_e2m1 + weight_byte);
      b_second = LoadU32(packed_weight_e2m1 + weight_byte + 16U);
      scale_b = LoadU32(weight_scales_e4m3fn + source_row * scale_row_bytes + k_block * 4U);
    }
    const std::uint32_t scale_a = LoadU32(activation_scales_e4m3fn + k_block * 4U);

    float next0 = 0.0F;
    float next1 = 0.0F;
    float next2 = 0.0F;
    float next3 = 0.0F;
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13}, "
        "%14, {%16, %17}, "
        "%15, {%16, %17};\n"
        : "=f"(next0), "=f"(next1), "=f"(next2), "=f"(next3)
        : "r"(a_first), "r"(a_first), "r"(a_second), "r"(a_second), "r"(b_first),
          "r"(b_second), "f"(d0), "f"(d1), "f"(d2), "f"(d3), "r"(scale_a),
          "r"(scale_b), "h"(instruction_block_id), "h"(instruction_thread_id));
    d0 = next0;
    d1 = next1;
    d2 = next2;
    d3 = next3;
  }

  if (lane < 4U) {
    const std::uint64_t output_row = global_warp * kRowsPerWarp + lane * 2U;
    if (output_row < rows) output[output_row] = d0 / output_divisor;
    if (output_row + 1U < rows) output[output_row + 1U] = d1 / output_divisor;
  }
#else
  (void)packed_activation_e2m1;
  (void)activation_scales_e4m3fn;
  (void)packed_weight_e2m1;
  (void)weight_scales_e4m3fn;
  (void)output;
  (void)tokens;
  (void)rows;
  (void)contracting_elements;
  (void)output_divisor;
#endif
}

__global__ void Sm120FusedGateUpKernel(
    const std::uint8_t* packed_activation_e2m1,
    const std::uint8_t* activation_scales_e4m3fn,
    const std::uint8_t* packed_gate_weight_e2m1,
    const std::uint8_t* gate_weight_scales_e4m3fn,
    const std::uint8_t* packed_up_weight_e2m1,
    const std::uint8_t* up_weight_scales_e4m3fn,
    float* gate_output,
    float* up_output,
    float* product_output,
    std::uint64_t tokens,
    std::uint64_t rows,
    std::uint64_t contracting_elements,
    float gate_output_divisor,
    float up_output_divisor) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  const std::uint64_t token = blockIdx.y;
  if (token >= tokens) return;
  const std::uint64_t packed_token_bytes = contracting_elements / 2U;
  const std::uint64_t scale_token_bytes = contracting_elements / 16U;
  packed_activation_e2m1 += token * packed_token_bytes;
  activation_scales_e4m3fn += token * scale_token_bytes;
  if (gate_output != nullptr) {
    gate_output += token * rows;
    up_output += token * rows;
  }
  product_output += token * rows;
  const unsigned warp = threadIdx.x / kWarpSize;
  const unsigned lane = threadIdx.x & (kWarpSize - 1U);
  const std::uint64_t global_warp =
      static_cast<std::uint64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  const std::uint64_t row_tiles = (rows + kRowsPerWarp - 1U) / kRowsPerWarp;
  if (global_warp >= row_tiles) return;

  const unsigned row_in_tile = lane >> 2U;
  const unsigned k_quarter = lane & 3U;
  const std::uint64_t source_row = global_warp * kRowsPerWarp + row_in_tile;
  const std::uint64_t packed_row_bytes = contracting_elements / 2U;
  const std::uint64_t scale_row_bytes = contracting_elements / 16U;
  const std::uint64_t k_blocks = contracting_elements / kElementsPerKBlock;

  float gate0 = 0.0F;
  float gate1 = 0.0F;
  float gate2 = 0.0F;
  float gate3 = 0.0F;
  float up0 = 0.0F;
  float up1 = 0.0F;
  float up2 = 0.0F;
  float up3 = 0.0F;
  constexpr std::uint16_t instruction_block_id = 0;
  constexpr std::uint16_t instruction_thread_id = 0;

  for (std::uint64_t k_block = 0; k_block < k_blocks; ++k_block) {
    const std::uint64_t activation_byte = k_block * 32U + k_quarter * 4U;
    const std::uint32_t a_first = LoadU32(packed_activation_e2m1 + activation_byte);
    const std::uint32_t a_second = LoadU32(packed_activation_e2m1 + activation_byte + 16U);
    const std::uint32_t scale_a = LoadU32(activation_scales_e4m3fn + k_block * 4U);

    std::uint32_t gate_b_first = 0;
    std::uint32_t gate_b_second = 0;
    std::uint32_t gate_scale_b = 0;
    std::uint32_t up_b_first = 0;
    std::uint32_t up_b_second = 0;
    std::uint32_t up_scale_b = 0;
    if (source_row < rows) {
      const std::uint64_t weight_byte = source_row * packed_row_bytes + k_block * 32U +
                                        static_cast<std::uint64_t>(k_quarter) * 4U;
      const std::uint64_t scale_byte = source_row * scale_row_bytes + k_block * 4U;
      gate_b_first = LoadU32(packed_gate_weight_e2m1 + weight_byte);
      gate_b_second = LoadU32(packed_gate_weight_e2m1 + weight_byte + 16U);
      gate_scale_b = LoadU32(gate_weight_scales_e4m3fn + scale_byte);
      up_b_first = LoadU32(packed_up_weight_e2m1 + weight_byte);
      up_b_second = LoadU32(packed_up_weight_e2m1 + weight_byte + 16U);
      up_scale_b = LoadU32(up_weight_scales_e4m3fn + scale_byte);
    }

    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13}, "
        "%14, {%16, %17}, "
        "%15, {%16, %17};\n"
        : "+f"(gate0), "+f"(gate1), "+f"(gate2), "+f"(gate3)
        : "r"(a_first), "r"(a_first), "r"(a_second), "r"(a_second), "r"(gate_b_first),
          "r"(gate_b_second), "f"(gate0), "f"(gate1), "f"(gate2), "f"(gate3),
          "r"(scale_a), "r"(gate_scale_b), "h"(instruction_block_id),
          "h"(instruction_thread_id));
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13}, "
        "%14, {%16, %17}, "
        "%15, {%16, %17};\n"
        : "+f"(up0), "+f"(up1), "+f"(up2), "+f"(up3)
        : "r"(a_first), "r"(a_first), "r"(a_second), "r"(a_second), "r"(up_b_first),
          "r"(up_b_second), "f"(up0), "f"(up1), "f"(up2), "f"(up3), "r"(scale_a),
          "r"(up_scale_b), "h"(instruction_block_id), "h"(instruction_thread_id));
  }

  if (lane < 4U) {
    const std::uint64_t output_row = global_warp * kRowsPerWarp + lane * 2U;
    const float gate_values[2] = {gate0, gate1};
    const float up_values[2] = {up0, up1};
#pragma unroll
    for (unsigned pair = 0; pair < 2U; ++pair) {
      const std::uint64_t row = output_row + pair;
      if (row >= rows) continue;
      const float gate = static_cast<float>(
          __float2bfloat16_rn(gate_values[pair] / gate_output_divisor));
      const float up =
          static_cast<float>(__float2bfloat16_rn(up_values[pair] / up_output_divisor));
      const float inner = kSqrtTwoOverPi * (gate + kGeluCubic * gate * gate * gate);
      const float gelu = static_cast<float>(
          __float2bfloat16_rn(0.5F * gate * (1.0F + tanhf(inner))));
      if (gate_output != nullptr) {
        gate_output[row] = gate;
        up_output[row] = up;
      }
      product_output[row] = static_cast<float>(__float2bfloat16_rn(gelu * up));
    }
  }
#else
  (void)packed_activation_e2m1;
  (void)activation_scales_e4m3fn;
  (void)packed_gate_weight_e2m1;
  (void)gate_weight_scales_e4m3fn;
  (void)packed_up_weight_e2m1;
  (void)up_weight_scales_e4m3fn;
  (void)gate_output;
  (void)up_output;
  (void)product_output;
  (void)tokens;
  (void)rows;
  (void)contracting_elements;
  (void)gate_output_divisor;
  (void)up_output_divisor;
#endif
}

template <bool kFusedGateUp, std::uint64_t kTokenTiles,
          unsigned kBlockWarps, bool kStageActivation>
__global__ void Sm120MatrixProjectionKernel(
    const std::uint8_t* packed_activation_e2m1,
    const std::uint8_t* activation_scales_e4m3fn,
    const std::uint8_t* packed_weight_e2m1,
    const std::uint8_t* weight_scales_e4m3fn,
    const std::uint8_t* packed_up_weight_e2m1,
    const std::uint8_t* up_weight_scales_e4m3fn, float* gate_output,
    float* up_output, float* output, std::uint64_t tokens,
    std::uint64_t rows, std::uint64_t contracting_elements,
    float output_divisor, float up_output_divisor) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  static_assert(kTokenTiles >= 1U);
  static_assert(kBlockWarps >= 1U);
  constexpr unsigned kStagedTokens =
      static_cast<unsigned>(kTokensPerMma * kTokenTiles);
  constexpr unsigned kPackedWordsPerKBlock =
      static_cast<unsigned>(kElementsPerKBlock / 8U);
  __shared__ alignas(16) std::uint32_t staged_activation
      [kStageActivation ? 2U : 1U]
      [kStageActivation ? kStagedTokens * kPackedWordsPerKBlock : 1U];
  __shared__ alignas(16) std::uint32_t staged_activation_scales
      [kStageActivation ? 2U : 1U][kStageActivation ? kStagedTokens : 1U];
  const unsigned warp = threadIdx.x / kWarpSize;
  const unsigned lane = threadIdx.x & (kWarpSize - 1U);
  const unsigned group = lane >> 2U;
  const unsigned thread_in_group = lane & 3U;
  const std::uint64_t global_warp =
      static_cast<std::uint64_t>(blockIdx.x) * kBlockWarps + warp;
  const std::uint64_t row_tiles = (rows + kRowsPerWarp - 1U) / kRowsPerWarp;
  const bool warp_active = global_warp < row_tiles;

  const std::uint64_t token_base = static_cast<std::uint64_t>(blockIdx.y) *
                                   kTokensPerMma * kTokenTiles;
  const std::uint64_t weight_column =
      warp_active ? global_warp * kRowsPerWarp + group : rows;
  const std::uint64_t packed_row_bytes = contracting_elements / 2U;
  const std::uint64_t scale_row_bytes = contracting_elements / 16U;
  const std::uint64_t k_blocks = contracting_elements / kElementsPerKBlock;

  float accumulator[kTokenTiles][4] = {};
  float up_accumulator[kTokenTiles][4] = {};

  if constexpr (kStageActivation) {
    StageNvfp4ActivationAsync(
        staged_activation[0], staged_activation_scales[0],
        packed_activation_e2m1, activation_scales_e4m3fn, token_base, 0U,
        packed_row_bytes, scale_row_bytes, tokens, kStagedTokens);
    CommitAsyncCopies();
    WaitForAsyncCopies();
    __syncthreads();
  }

  for (std::uint64_t k_block = 0; k_block < k_blocks; ++k_block) {
    const unsigned current_stage = static_cast<unsigned>(k_block & 1U);
    const bool has_next_stage = k_block + 1U < k_blocks;
    if constexpr (kStageActivation) {
      if (has_next_stage) {
        const unsigned next_stage = current_stage ^ 1U;
        StageNvfp4ActivationAsync(
            staged_activation[next_stage],
            staged_activation_scales[next_stage], packed_activation_e2m1,
            activation_scales_e4m3fn, token_base, k_block + 1U,
            packed_row_bytes, scale_row_bytes, tokens, kStagedTokens);
        CommitAsyncCopies();
      }
    }
    const std::uint64_t k_offset =
        k_block * 32U + static_cast<std::uint64_t>(thread_in_group) * 4U;
    std::uint32_t b0 = 0U;
    std::uint32_t b1 = 0U;
    std::uint32_t scale_b = 0U;
    std::uint32_t up_b0 = 0U;
    std::uint32_t up_b1 = 0U;
    std::uint32_t up_scale_b = 0U;
    if (weight_column < rows) {
      const std::uint64_t weight_offset =
          weight_column * packed_row_bytes + k_offset;
      const std::uint64_t scale_offset =
          weight_column * scale_row_bytes + k_block * 4U;
      b0 = LoadU32(packed_weight_e2m1 + weight_offset);
      b1 = LoadU32(packed_weight_e2m1 + weight_offset + 16U);
      scale_b = LoadU32(weight_scales_e4m3fn + scale_offset);
      if constexpr (kFusedGateUp) {
        up_b0 = LoadU32(packed_up_weight_e2m1 + weight_offset);
        up_b1 = LoadU32(packed_up_weight_e2m1 + weight_offset + 16U);
        up_scale_b = LoadU32(up_weight_scales_e4m3fn + scale_offset);
      }
    }
    // Retain each 8-column weight fragment while a warp advances through a
    // large stack of independent 16-token MMA tiles. This is the primary
    // prompt-specific reuse boundary: increasing kTokenTiles reduces weight
    // and weight-scale traffic without changing any tile's K accumulation.
#pragma unroll
    for (std::uint64_t tile = 0; tile < kTokenTiles; ++tile) {
      const std::uint64_t tile_token_base =
          token_base + tile * kTokensPerMma;
      const std::uint64_t token_low = tile_token_base + group;
      const std::uint64_t token_high = token_low + 8U;
      const unsigned staged_low =
          static_cast<unsigned>(tile * kTokensPerMma) + group;
      const unsigned staged_high = staged_low + 8U;
      const std::uint32_t a0 = kStageActivation
          ? staged_activation[current_stage]
                             [staged_low * kPackedWordsPerKBlock +
                              thread_in_group]
          : (token_low < tokens
                 ? LoadU32(packed_activation_e2m1 +
                           token_low * packed_row_bytes + k_offset)
                 : 0U);
      const std::uint32_t a1 = kStageActivation
          ? staged_activation[current_stage]
                             [staged_high * kPackedWordsPerKBlock +
                              thread_in_group]
          : (token_high < tokens
                 ? LoadU32(packed_activation_e2m1 +
                           token_high * packed_row_bytes + k_offset)
                 : 0U);
      const std::uint32_t a2 = kStageActivation
          ? staged_activation[current_stage]
                             [staged_low * kPackedWordsPerKBlock +
                              thread_in_group + 4U]
          : (token_low < tokens
                 ? LoadU32(packed_activation_e2m1 +
                           token_low * packed_row_bytes + k_offset + 16U)
                 : 0U);
      const std::uint32_t a3 = kStageActivation
          ? staged_activation[current_stage]
                             [staged_high * kPackedWordsPerKBlock +
                              thread_in_group + 4U]
          : (token_high < tokens
                 ? LoadU32(packed_activation_e2m1 +
                           token_high * packed_row_bytes + k_offset + 16U)
                 : 0U);
      std::uint32_t scale_a = 0U;
      // With thread-id-a=0, the lower two lanes in each quad supply the four
      // block scales for rows group and group+8 respectively.
      if (thread_in_group < 2U) {
        const std::uint64_t scale_token =
            token_low + static_cast<std::uint64_t>(thread_in_group) * 8U;
        if (scale_token < tokens) {
          scale_a = kStageActivation
              ? staged_activation_scales[current_stage]
                    [staged_low + thread_in_group * 8U]
              : LoadU32(activation_scales_e4m3fn +
                        scale_token * scale_row_bytes + k_block * 4U);
        }
      }
      MmaNvfp4(accumulator[tile][0], accumulator[tile][1],
                accumulator[tile][2], accumulator[tile][3], a0, a1, a2, a3,
                b0, b1, scale_a, scale_b);
      if constexpr (kFusedGateUp) {
        MmaNvfp4(up_accumulator[tile][0], up_accumulator[tile][1],
                  up_accumulator[tile][2], up_accumulator[tile][3], a0, a1,
                  a2, a3, up_b0, up_b1, scale_a, up_scale_b);
      }
    }
    if constexpr (kStageActivation) {
      if (has_next_stage) {
        WaitForAsyncCopies();
        __syncthreads();
      }
    }
  }

  const std::uint64_t output_column =
      global_warp * kRowsPerWarp + thread_in_group * 2U;
#pragma unroll
  for (std::uint64_t tile = 0; tile < kTokenTiles; ++tile) {
#pragma unroll
    for (unsigned pair = 0; pair < 4U; ++pair) {
      const std::uint64_t token =
          token_base + tile * kTokensPerMma + group +
          ((pair & 2U) == 0U ? 0U : 8U);
      const std::uint64_t column = output_column + (pair & 1U);
      if (token >= tokens || column >= rows) continue;
      const float gate = accumulator[tile][pair] / output_divisor;
      if constexpr (kFusedGateUp) {
        const float rounded_gate =
            static_cast<float>(__float2bfloat16_rn(gate));
        const float rounded_up = static_cast<float>(__float2bfloat16_rn(
            up_accumulator[tile][pair] / up_output_divisor));
        const float inner =
            kSqrtTwoOverPi *
            (rounded_gate + kGeluCubic * rounded_gate * rounded_gate *
                                rounded_gate);
        const float gelu = static_cast<float>(__float2bfloat16_rn(
            0.5F * rounded_gate * (1.0F + tanhf(inner))));
        if (gate_output != nullptr) {
          gate_output[token * rows + column] = rounded_gate;
          up_output[token * rows + column] = rounded_up;
        }
        output[token * rows + column] =
            static_cast<float>(__float2bfloat16_rn(gelu * rounded_up));
      } else {
        output[token * rows + column] = gate;
      }
    }
  }
#else
  (void)packed_activation_e2m1;
  (void)activation_scales_e4m3fn;
  (void)packed_weight_e2m1;
  (void)weight_scales_e4m3fn;
  (void)packed_up_weight_e2m1;
  (void)up_weight_scales_e4m3fn;
  (void)gate_output;
  (void)up_output;
  (void)output;
  (void)tokens;
  (void)rows;
  (void)contracting_elements;
  (void)output_divisor;
  (void)up_output_divisor;
#endif
}

}  // namespace

Status LaunchNvfp4Sm120DirectProjection(const std::uint8_t* packed_activation_e2m1,
                                        const std::uint8_t* activation_scales_e4m3fn,
                                        const std::uint8_t* packed_weight_e2m1,
                                        const std::uint8_t* weight_scales_e4m3fn,
                                        float* output,
                                        std::uint64_t rows,
                                        std::uint64_t contracting_elements,
                                        float activation_global_divisor,
                                        float weight_global_divisor,
                                        cudaStream_t stream) {
  if (packed_activation_e2m1 == nullptr || activation_scales_e4m3fn == nullptr ||
      packed_weight_e2m1 == nullptr || weight_scales_e4m3fn == nullptr || output == nullptr) {
    return Invalid("SM120 NVFP4 direct projection requires non-null device pointers");
  }
  if (rows == 0U || contracting_elements == 0U ||
      contracting_elements % kElementsPerKBlock != 0U) {
    return Invalid("SM120 NVFP4 direct projection requires positive dimensions and K divisible by 64");
  }
  if (!PositiveFinite(activation_global_divisor) || !PositiveFinite(weight_global_divisor)) {
    return Invalid("SM120 NVFP4 direct projection global divisors must be positive and finite");
  }
  const float output_divisor = activation_global_divisor * weight_global_divisor;
  if (!PositiveFinite(output_divisor)) {
    return Invalid("SM120 NVFP4 direct projection global-divisor product overflowed");
  }
  const std::uint64_t row_tiles = (rows + kRowsPerWarp - 1U) / kRowsPerWarp;
  const std::uint64_t blocks = (row_tiles + kWarpsPerBlock - 1U) / kWarpsPerBlock;
  if (blocks > static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max())) {
    return Invalid("SM120 NVFP4 direct projection grid exceeds CUDA limits");
  }
  Sm120DirectProjectionKernel<<<dim3(static_cast<unsigned>(blocks), 1U),
                                kThreadsPerBlock, 0, stream>>>(
      packed_activation_e2m1, activation_scales_e4m3fn, packed_weight_e2m1,
      weight_scales_e4m3fn, output, 1U, rows, contracting_elements,
      output_divisor);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch direct-source SM120 NVFP4 projection", error);
}

Status LaunchNvfp4Sm120DirectProjectionBatch(
    const std::uint8_t* packed_activation_e2m1,
    const std::uint8_t* activation_scales_e4m3fn,
    const std::uint8_t* packed_weight_e2m1,
    const std::uint8_t* weight_scales_e4m3fn, float* output,
    std::uint64_t tokens, std::uint64_t rows,
    std::uint64_t contracting_elements, float activation_global_divisor,
    float weight_global_divisor, cudaStream_t stream) {
  if (packed_activation_e2m1 == nullptr || activation_scales_e4m3fn == nullptr ||
      packed_weight_e2m1 == nullptr || weight_scales_e4m3fn == nullptr ||
      output == nullptr) {
    return Invalid("batched SM120 NVFP4 projection requires non-null device pointers");
  }
  if (tokens == 0U || tokens > 65535U || rows == 0U ||
      contracting_elements == 0U ||
      contracting_elements % kElementsPerKBlock != 0U ||
      !PositiveFinite(activation_global_divisor) ||
      !PositiveFinite(weight_global_divisor)) {
    return Invalid("batched SM120 NVFP4 projection geometry or divisors are invalid");
  }
  const float output_divisor = activation_global_divisor * weight_global_divisor;
  if (!PositiveFinite(output_divisor)) {
    return Invalid("batched SM120 NVFP4 projection divisor product overflowed");
  }
  const std::uint64_t row_tiles = (rows + kRowsPerWarp - 1U) / kRowsPerWarp;
  const std::uint64_t blocks =
      (row_tiles + kPrefillWarpsPerBlock - 1U) /
      kPrefillWarpsPerBlock;
  if (blocks > static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max())) {
    return Invalid("batched SM120 NVFP4 projection grid exceeds CUDA limits");
  }
  const std::uint64_t token_tiles =
      (tokens + kTokensPerMma - 1U) / kTokensPerMma;
  const std::uint64_t grouped_token_tiles =
      (token_tiles + kPrefillTokenTilesPerWarp - 1U) /
      kPrefillTokenTilesPerWarp;
  Sm120MatrixProjectionKernel<false, kPrefillTokenTilesPerWarp,
                              kPrefillWarpsPerBlock, true><<<
      dim3(static_cast<unsigned>(blocks), static_cast<unsigned>(grouped_token_tiles)),
      kPrefillThreadsPerBlock, 0, stream>>>(
      packed_activation_e2m1, activation_scales_e4m3fn, packed_weight_e2m1,
      weight_scales_e4m3fn, nullptr, nullptr, nullptr, nullptr, output, tokens,
      rows, contracting_elements, output_divisor, 1.0F);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch batched direct-source SM120 NVFP4 projection", error);
}

Status LaunchNvfp4Sm120FusedGateUp(
    const std::uint8_t* packed_activation_e2m1,
    const std::uint8_t* activation_scales_e4m3fn,
    const std::uint8_t* packed_gate_weight_e2m1,
    const std::uint8_t* gate_weight_scales_e4m3fn,
    const std::uint8_t* packed_up_weight_e2m1,
    const std::uint8_t* up_weight_scales_e4m3fn,
    float* gate_output,
    float* up_output,
    float* product_output,
    std::uint64_t rows,
    std::uint64_t contracting_elements,
    float gate_activation_global_divisor,
    float gate_weight_global_divisor,
    float up_activation_global_divisor,
    float up_weight_global_divisor,
    cudaStream_t stream) {
  if (packed_activation_e2m1 == nullptr || activation_scales_e4m3fn == nullptr ||
      packed_gate_weight_e2m1 == nullptr || gate_weight_scales_e4m3fn == nullptr ||
      packed_up_weight_e2m1 == nullptr || up_weight_scales_e4m3fn == nullptr ||
      product_output == nullptr || (gate_output == nullptr) != (up_output == nullptr)) {
    return Invalid("SM120 fused Gate/Up requires input/product pointers and paired diagnostics");
  }
  if (rows == 0U || contracting_elements == 0U ||
      contracting_elements % kElementsPerKBlock != 0U) {
    return Invalid("SM120 fused Gate/Up requires positive dimensions and K divisible by 64");
  }
  if (!PositiveFinite(gate_activation_global_divisor) ||
      !PositiveFinite(gate_weight_global_divisor) ||
      !PositiveFinite(up_activation_global_divisor) ||
      !PositiveFinite(up_weight_global_divisor)) {
    return Invalid("SM120 fused Gate/Up global divisors must be positive and finite");
  }
  const float gate_output_divisor =
      gate_activation_global_divisor * gate_weight_global_divisor;
  const float up_output_divisor = up_activation_global_divisor * up_weight_global_divisor;
  if (!PositiveFinite(gate_output_divisor) || !PositiveFinite(up_output_divisor)) {
    return Invalid("SM120 fused Gate/Up global-divisor product overflowed");
  }

  const std::uint64_t row_tiles = (rows + kRowsPerWarp - 1U) / kRowsPerWarp;
  const std::uint64_t blocks = (row_tiles + kWarpsPerBlock - 1U) / kWarpsPerBlock;
  if (blocks > static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max())) {
    return Invalid("SM120 fused Gate/Up grid exceeds CUDA limits");
  }
  Sm120FusedGateUpKernel<<<dim3(static_cast<unsigned>(blocks), 1U),
                           kThreadsPerBlock, 0, stream>>>(
      packed_activation_e2m1, activation_scales_e4m3fn, packed_gate_weight_e2m1,
      gate_weight_scales_e4m3fn, packed_up_weight_e2m1, up_weight_scales_e4m3fn,
      gate_output, up_output, product_output, 1U, rows, contracting_elements,
      gate_output_divisor, up_output_divisor);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess ? Status::Ok()
                              : CudaFailure("launch fused SM120 NVFP4 Gate/Up", error);
}

Status LaunchNvfp4Sm120FusedGateUpBatch(
    const std::uint8_t* packed_activation_e2m1,
    const std::uint8_t* activation_scales_e4m3fn,
    const std::uint8_t* packed_gate_weight_e2m1,
    const std::uint8_t* gate_weight_scales_e4m3fn,
    const std::uint8_t* packed_up_weight_e2m1,
    const std::uint8_t* up_weight_scales_e4m3fn, float* gate_output,
    float* up_output, float* product_output, std::uint64_t tokens,
    std::uint64_t rows, std::uint64_t contracting_elements,
    float gate_activation_global_divisor, float gate_weight_global_divisor,
    float up_activation_global_divisor, float up_weight_global_divisor,
    cudaStream_t stream) {
  if (packed_activation_e2m1 == nullptr || activation_scales_e4m3fn == nullptr ||
      packed_gate_weight_e2m1 == nullptr || gate_weight_scales_e4m3fn == nullptr ||
      packed_up_weight_e2m1 == nullptr || up_weight_scales_e4m3fn == nullptr ||
      product_output == nullptr || (gate_output == nullptr) != (up_output == nullptr)) {
    return Invalid("batched SM120 fused Gate/Up pointer contract is invalid");
  }
  if (tokens == 0U || tokens > 65535U || rows == 0U ||
      contracting_elements == 0U ||
      contracting_elements % kElementsPerKBlock != 0U ||
      !PositiveFinite(gate_activation_global_divisor) ||
      !PositiveFinite(gate_weight_global_divisor) ||
      !PositiveFinite(up_activation_global_divisor) ||
      !PositiveFinite(up_weight_global_divisor)) {
    return Invalid("batched SM120 fused Gate/Up geometry or divisors are invalid");
  }
  const float gate_output_divisor =
      gate_activation_global_divisor * gate_weight_global_divisor;
  const float up_output_divisor =
      up_activation_global_divisor * up_weight_global_divisor;
  if (!PositiveFinite(gate_output_divisor) || !PositiveFinite(up_output_divisor)) {
    return Invalid("batched SM120 fused Gate/Up divisor product overflowed");
  }
  const std::uint64_t row_tiles = (rows + kRowsPerWarp - 1U) / kRowsPerWarp;
  const std::uint64_t blocks = (row_tiles + kWarpsPerBlock - 1U) / kWarpsPerBlock;
  if (blocks > static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max())) {
    return Invalid("batched SM120 fused Gate/Up grid exceeds CUDA limits");
  }
  const std::uint64_t token_tiles =
      (tokens + kTokensPerMma - 1U) / kTokensPerMma;
  const std::uint64_t grouped_token_tiles =
      (token_tiles + kFusedGateUpTokenTilesPerWarp - 1U) /
      kFusedGateUpTokenTilesPerWarp;
  Sm120MatrixProjectionKernel<true, kFusedGateUpTokenTilesPerWarp,
                              kWarpsPerBlock, false><<<
      dim3(static_cast<unsigned>(blocks), static_cast<unsigned>(grouped_token_tiles)),
      kThreadsPerBlock, 0, stream>>>(
      packed_activation_e2m1, activation_scales_e4m3fn,
      packed_gate_weight_e2m1, gate_weight_scales_e4m3fn,
      packed_up_weight_e2m1, up_weight_scales_e4m3fn, gate_output, up_output,
      product_output, tokens, rows, contracting_elements,
      gate_output_divisor, up_output_divisor);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch batched fused SM120 NVFP4 Gate/Up", error);
}

}  // namespace gem16gb::internal
