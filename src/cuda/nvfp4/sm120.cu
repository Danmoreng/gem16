#include "cuda/nvfp4/sm120.h"
#include "cuda/moe/prefill_plan.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>

namespace gem16::internal {
namespace {

constexpr std::uint64_t kElementsPerKBlock = 64;
constexpr std::uint64_t kRowsPerWarp = 8;
constexpr std::uint64_t kTokensPerMma = 16;
constexpr std::uint64_t kPrefillTokenTilesPerWarp = 8;
constexpr std::uint64_t kFusedGateUpTokenTilesPerWarp = 2;
constexpr std::uint64_t kGroupedRowTilesPerWarp = 2;
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

bool Aligned16(const void* pointer) {
  return reinterpret_cast<std::uintptr_t>(pointer) % alignof(uint4) == 0U;
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

template <bool kRoundBf16, int kSelectionMode>
__global__ void Sm120DirectProjectionKernel(const std::uint8_t* packed_activation_e2m1,
                                            const std::uint8_t* activation_scales_e4m3fn,
                                            const std::uint8_t* packed_weight_e2m1,
                                            const std::uint8_t* weight_scales_e4m3fn,
                                            const std::uint32_t* selected_ids,
                                            const Gemma4MoePrefillAssignment* assignments,
                                            const std::uint32_t* permutation,
                                            std::uint32_t slot,
                                            std::uint32_t experts,
                                            float* output,
                                            std::uint64_t tokens,
                                            std::uint64_t rows,
                                            std::uint64_t contracting_elements,
                                            float output_divisor) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  const std::uint64_t token = blockIdx.y;
  if (token >= tokens) return;
  std::uint64_t activation_token = token;
  std::uint64_t output_token = token;
  std::uint32_t expert = 0U;
  if constexpr (kSelectionMode == 1) {
    expert = selected_ids[slot];
  } else if constexpr (kSelectionMode == 3) {
    expert = selected_ids[token];
    activation_token = 0U;
  } else if constexpr (kSelectionMode == 4) {
    expert = selected_ids[token];
  } else if constexpr (kSelectionMode == 2) {
    const std::uint32_t original = permutation[token];
    const Gemma4MoePrefillAssignment assignment = assignments[original];
    expert = assignment.expert_id;
    activation_token = token;
    output_token = original;
  }
  if constexpr (kSelectionMode != 0) {
    if (expert >= experts) return;
    const std::uint64_t k_blocks = contracting_elements / kElementsPerKBlock;
    packed_weight_e2m1 +=
        static_cast<std::uint64_t>(expert) * rows * k_blocks * 32U;
    weight_scales_e4m3fn +=
        static_cast<std::uint64_t>(expert) * rows * k_blocks * 4U;
  }
  const std::uint64_t packed_token_bytes = contracting_elements / 2U;
  const std::uint64_t scale_token_bytes = contracting_elements / 16U;
  packed_activation_e2m1 += activation_token * packed_token_bytes;
  activation_scales_e4m3fn += activation_token * scale_token_bytes;
  output += output_token * rows;
  const unsigned warp = threadIdx.x / kWarpSize;
  const unsigned lane = threadIdx.x & (kWarpSize - 1U);
  const std::uint64_t global_warp =
      static_cast<std::uint64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  const std::uint64_t row_tiles = (rows + kRowsPerWarp - 1U) / kRowsPerWarp;
  if (global_warp >= row_tiles) return;

  const unsigned row_in_tile = lane >> 2U;
  const unsigned k_quarter = lane & 3U;
  const std::uint64_t first_row = global_warp * kRowsPerWarp;
  const std::uint64_t source_row = first_row + row_in_tile;
  const std::uint64_t tile_rows =
      min(static_cast<std::uint64_t>(kRowsPerWarp), rows - first_row);
  const std::uint64_t k_blocks = contracting_elements / kElementsPerKBlock;
  const std::uint64_t weight_tile_offset = first_row * k_blocks * 32U;
  const std::uint64_t scale_tile_offset = first_row * k_blocks * 4U;

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
      const std::uint64_t weight_byte =
          weight_tile_offset + (k_block * tile_rows + row_in_tile) * 32U +
          static_cast<std::uint64_t>(k_quarter) * 4U;
      b_first = LoadU32(packed_weight_e2m1 + weight_byte);
      b_second = LoadU32(packed_weight_e2m1 + weight_byte + 16U);
      const std::uint64_t scale_byte =
          scale_tile_offset + (k_block * tile_rows + row_in_tile) * 4U;
      scale_b = LoadU32(weight_scales_e4m3fn + scale_byte);
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
    const float value0 = d0 / output_divisor;
    const float value1 = d1 / output_divisor;
    if (output_row < rows) {
      output[output_row] = kRoundBf16
                               ? static_cast<float>(__float2bfloat16_rn(value0))
                               : value0;
    }
    if (output_row + 1U < rows) {
      output[output_row + 1U] =
          kRoundBf16 ? static_cast<float>(__float2bfloat16_rn(value1))
                     : value1;
    }
  }
#else
  (void)packed_activation_e2m1;
  (void)activation_scales_e4m3fn;
  (void)packed_weight_e2m1;
  (void)weight_scales_e4m3fn;
  (void)selected_ids;
  (void)assignments;
  (void)permutation;
  (void)slot;
  (void)experts;
  (void)output;
  (void)tokens;
  (void)rows;
  (void)contracting_elements;
  (void)output_divisor;
#endif
}

template <int kSelectionMode>
__global__ void Sm120FusedGateUpKernel(
    const std::uint8_t* packed_activation_e2m1,
    const std::uint8_t* activation_scales_e4m3fn,
    const std::uint8_t* packed_gate_weight_e2m1,
    const std::uint8_t* gate_weight_scales_e4m3fn,
    const std::uint8_t* packed_up_weight_e2m1,
    const std::uint8_t* up_weight_scales_e4m3fn,
    const std::uint32_t* selected_ids,
    const Gemma4MoePrefillAssignment* assignments,
    const std::uint32_t* permutation,
    std::uint32_t slot,
    std::uint32_t experts,
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
  std::uint64_t activation_token = token;
  std::uint32_t expert = 0U;
  if constexpr (kSelectionMode == 1) {
    expert = selected_ids[slot];
  } else if constexpr (kSelectionMode == 3) {
    expert = selected_ids[token];
    activation_token = 0U;
  } else if constexpr (kSelectionMode == 2) {
    const std::uint32_t original = permutation[token];
    const Gemma4MoePrefillAssignment assignment = assignments[original];
    expert = assignment.expert_id;
    activation_token = assignment.token_id;
  }
  if constexpr (kSelectionMode != 0) {
    if (expert >= experts) return;
    const std::uint64_t k_blocks = contracting_elements / kElementsPerKBlock;
    const std::uint64_t expert_rows = 2U * rows;
    const std::uint64_t gate_row =
        static_cast<std::uint64_t>(expert) * expert_rows;
    const std::uint64_t up_row = gate_row + rows;
    packed_gate_weight_e2m1 += gate_row * k_blocks * 32U;
    gate_weight_scales_e4m3fn += gate_row * k_blocks * 4U;
    packed_up_weight_e2m1 += up_row * k_blocks * 32U;
    up_weight_scales_e4m3fn += up_row * k_blocks * 4U;
  }
  const std::uint64_t packed_token_bytes = contracting_elements / 2U;
  const std::uint64_t scale_token_bytes = contracting_elements / 16U;
  packed_activation_e2m1 += activation_token * packed_token_bytes;
  activation_scales_e4m3fn += activation_token * scale_token_bytes;
  if (gate_output != nullptr) {
    const std::uint64_t output_stride =
        kSelectionMode == 3 ? 2U * rows : rows;
    gate_output += token * output_stride;
    up_output += token * output_stride;
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
  const std::uint64_t first_row = global_warp * kRowsPerWarp;
  const std::uint64_t source_row = first_row + row_in_tile;
  const std::uint64_t tile_rows =
      min(static_cast<std::uint64_t>(kRowsPerWarp), rows - first_row);
  const std::uint64_t k_blocks = contracting_elements / kElementsPerKBlock;
  const std::uint64_t weight_tile_offset = first_row * k_blocks * 32U;
  const std::uint64_t scale_tile_offset = first_row * k_blocks * 4U;

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
      const std::uint64_t weight_byte =
          weight_tile_offset + (k_block * tile_rows + row_in_tile) * 32U +
          static_cast<std::uint64_t>(k_quarter) * 4U;
      const std::uint64_t scale_byte =
          scale_tile_offset + (k_block * tile_rows + row_in_tile) * 4U;
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
  (void)selected_ids;
  (void)assignments;
  (void)permutation;
  (void)slot;
  (void)experts;
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
          unsigned kBlockWarps, bool kStageActivation,
          typename Output = float>
__global__ void Sm120MatrixProjectionKernel(
    const std::uint8_t* packed_activation_e2m1,
    const std::uint8_t* activation_scales_e4m3fn,
    const std::uint8_t* packed_weight_e2m1,
    const std::uint8_t* weight_scales_e4m3fn,
    const std::uint8_t* packed_up_weight_e2m1,
    const std::uint8_t* up_weight_scales_e4m3fn, float* gate_output,
    float* up_output, Output* output, std::uint64_t tokens,
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
  const std::uint64_t activation_scale_row_bytes = contracting_elements / 16U;
  const std::uint64_t k_blocks = contracting_elements / kElementsPerKBlock;
  const std::uint64_t first_weight_row = global_warp * kRowsPerWarp;
  const std::uint64_t weight_tile_rows =
      warp_active
          ? min(static_cast<std::uint64_t>(kRowsPerWarp), rows - first_weight_row)
          : 0U;
  const std::uint64_t weight_tile_offset =
      first_weight_row * k_blocks * 32U;
  const std::uint64_t scale_tile_offset =
      first_weight_row * k_blocks * 4U;

  float accumulator[kTokenTiles][4] = {};
  float up_accumulator[kTokenTiles][4] = {};

  if constexpr (kStageActivation) {
    StageNvfp4ActivationAsync(
        staged_activation[0], staged_activation_scales[0],
        packed_activation_e2m1, activation_scales_e4m3fn, token_base, 0U,
        packed_row_bytes, activation_scale_row_bytes, tokens, kStagedTokens);
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
            packed_row_bytes, activation_scale_row_bytes, tokens, kStagedTokens);
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
          weight_tile_offset + (k_block * weight_tile_rows + group) * 32U +
          static_cast<std::uint64_t>(thread_in_group) * 4U;
      const std::uint64_t scale_offset =
          scale_tile_offset + (k_block * weight_tile_rows + group) * 4U;
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
                        scale_token * activation_scale_row_bytes + k_block * 4U);
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
    if constexpr (std::is_same_v<Output, std::uint16_t>) {
#pragma unroll
      for (unsigned token_half = 0; token_half < 2U; ++token_half) {
        const std::uint64_t token =
            token_base + tile * kTokensPerMma + group + token_half * 8U;
        if (token >= tokens || output_column + 1U >= rows) continue;
        const unsigned pair = token_half * 2U;
        const std::uint32_t low = __bfloat16_as_ushort(__float2bfloat16_rn(
            accumulator[tile][pair] / output_divisor));
        const std::uint32_t high = __bfloat16_as_ushort(__float2bfloat16_rn(
            accumulator[tile][pair + 1U] / output_divisor));
        *reinterpret_cast<std::uint32_t*>(
            output + token * rows + output_column) = low | (high << 16U);
      }
    } else {
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

// Expert-major prefill tile. Stable routing provides compact descriptors for
// up to 16 assignments of one expert. Filling the complete MMA M dimension
// lets those assignments share every W13/W2 weight fragment while preserving
// each assignment's original K64 accumulation and BF16 epilogue.
template <typename Output>
__device__ __forceinline__ void StorePhysicalOrContainerBf16(
    Output* output, std::uint64_t index, __nv_bfloat16 value) {
  if constexpr (std::is_same_v<Output, std::uint16_t>) {
    output[index] = __bfloat16_as_ushort(value);
  } else {
    static_assert(std::is_same_v<Output, float>);
    output[index] = static_cast<float>(value);
  }
}

template <bool kFusedGateUp, typename Output>
__global__ void Sm120GroupedExpertMatrixKernel(
    const std::uint8_t* packed_activation_e2m1,
    const std::uint8_t* activation_scales_e4m3fn,
    const std::uint8_t* packed_weight_e2m1,
    const std::uint8_t* weight_scales_e4m3fn,
    const Gemma4MoePrefillAssignment* assignments,
    const std::uint32_t* permutation, const std::uint32_t* prefix,
    const std::uint32_t* expert_tiles,
    const std::uint32_t* expert_tile_count, Output* output,
    std::uint64_t assignment_count, std::uint64_t rows,
    std::uint64_t contracting_elements, std::uint32_t experts,
    float output_divisor) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  const std::uint32_t tile_index = blockIdx.y;
  if (tile_index >= expert_tile_count[0]) return;
  const std::uint32_t descriptor = expert_tiles[tile_index];
  const std::uint32_t expert = descriptor >> 16U;
  const std::uint32_t grouped_base = descriptor & 0xffffU;
  if (expert >= experts || grouped_base >= assignment_count) return;
  const std::uint32_t grouped_end = prefix[expert + 1U];

  const unsigned warp = threadIdx.x / kWarpSize;
  const unsigned lane = threadIdx.x & (kWarpSize - 1U);
  const unsigned group = lane >> 2U;
  const unsigned thread_in_group = lane & 3U;
  const std::uint64_t logical_warp =
      static_cast<std::uint64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  const std::uint64_t row_tiles = rows / kRowsPerWarp;
  const std::uint64_t first_row_tile =
      logical_warp * kGroupedRowTilesPerWarp;
  const bool active_warp = first_row_tile < row_tiles;

  const std::uint64_t k_blocks =
      contracting_elements / kElementsPerKBlock;
  const std::uint64_t packed_row_bytes = contracting_elements / 2U;
  const std::uint64_t scale_row_bytes = contracting_elements / 16U;
  const std::uint64_t expert_row_base =
      static_cast<std::uint64_t>(expert) *
      (kFusedGateUp ? 2U * rows : rows);
  const std::uint64_t first_weight_row = expert_row_base +
                                         first_row_tile * kRowsPerWarp;

  float accumulator[kGroupedRowTilesPerWarp][4] = {};
  float up_accumulator[kGroupedRowTilesPerWarp][4] = {};
  const std::uint32_t grouped_low = grouped_base + group;
  const std::uint32_t grouped_high = grouped_low + 8U;

  __shared__ alignas(16) uint4 staged_activation[16][2];
  __shared__ std::uint32_t staged_activation_scales[16];
  __shared__ std::uint32_t staged_activation_rows[16];
  __shared__ std::uint32_t staged_activation_valid[16];
  if (threadIdx.x < 16U) {
    const std::uint32_t grouped = grouped_base + threadIdx.x;
    std::uint32_t activation_row = grouped;
    bool valid = grouped < grouped_end;
    if constexpr (kFusedGateUp) {
      if (valid) {
        const std::uint32_t original = permutation[grouped];
        if (original >= assignment_count ||
            assignments[original].expert_id != expert) {
          valid = false;
        } else {
          activation_row = assignments[original].token_id;
        }
      }
    }
    staged_activation_rows[threadIdx.x] = activation_row;
    staged_activation_valid[threadIdx.x] = valid ? 1U : 0U;
  }
  __syncthreads();

  for (std::uint64_t k_block = 0; k_block < k_blocks; ++k_block) {
    if (threadIdx.x < 32U) {
      const unsigned activation_index = threadIdx.x >> 1U;
      const unsigned half = threadIdx.x & 1U;
      uint4 value = make_uint4(0U, 0U, 0U, 0U);
      if (staged_activation_valid[activation_index] != 0U) {
        const std::uint64_t activation_row =
            staged_activation_rows[activation_index];
        // Grouped launch validation guarantees a 16-byte base; K64 makes
        // both the packed row stride and every k_block offset 16-byte aligned.
        value = reinterpret_cast<const uint4*>(
            packed_activation_e2m1 + activation_row * packed_row_bytes +
            k_block * 32U)[half];
      }
      staged_activation[activation_index][half] = value;
    }
    if (threadIdx.x < 16U) {
      std::uint32_t scale = 0U;
      if (staged_activation_valid[threadIdx.x] != 0U) {
        scale = LoadU32(
            activation_scales_e4m3fn +
            static_cast<std::uint64_t>(
                staged_activation_rows[threadIdx.x]) *
                scale_row_bytes +
            k_block * 4U);
      }
      staged_activation_scales[threadIdx.x] = scale;
    }
    __syncthreads();

    const auto* low_words = reinterpret_cast<const std::uint32_t*>(
        staged_activation[group]);
    const auto* high_words = reinterpret_cast<const std::uint32_t*>(
        staged_activation[group + 8U]);
    const std::uint32_t a0 = low_words[thread_in_group];
    const std::uint32_t a1 = high_words[thread_in_group];
    const std::uint32_t a2 = low_words[thread_in_group + 4U];
    const std::uint32_t a3 = high_words[thread_in_group + 4U];
    const std::uint32_t scale_a = thread_in_group < 2U
        ? staged_activation_scales[group + thread_in_group * 8U]
        : 0U;

#pragma unroll
    for (std::uint64_t row_tile = 0U;
         row_tile < kGroupedRowTilesPerWarp; ++row_tile) {
      if (!active_warp || first_row_tile + row_tile >= row_tiles) continue;
      const std::uint64_t tile_first_weight_row =
          first_weight_row + row_tile * kRowsPerWarp;
      const std::uint64_t weight_offset =
          tile_first_weight_row * k_blocks * 32U +
          (k_block * kRowsPerWarp + group) * 32U +
          static_cast<std::uint64_t>(thread_in_group) * 4U;
      const std::uint64_t scale_offset =
          tile_first_weight_row * k_blocks * 4U +
          (k_block * kRowsPerWarp + group) * 4U;
      const std::uint32_t b0 = LoadU32(packed_weight_e2m1 + weight_offset);
      const std::uint32_t b1 =
          LoadU32(packed_weight_e2m1 + weight_offset + 16U);
      const std::uint32_t scale_b =
          LoadU32(weight_scales_e4m3fn + scale_offset);
      MmaNvfp4(accumulator[row_tile][0], accumulator[row_tile][1],
                accumulator[row_tile][2], accumulator[row_tile][3], a0, a1,
                a2, a3, b0, b1, scale_a, scale_b);
      if constexpr (kFusedGateUp) {
        const std::uint64_t up_weight_offset =
            (tile_first_weight_row + rows) * k_blocks * 32U +
            (k_block * kRowsPerWarp + group) * 32U +
            static_cast<std::uint64_t>(thread_in_group) * 4U;
        const std::uint64_t up_scale_offset =
            (tile_first_weight_row + rows) * k_blocks * 4U +
            (k_block * kRowsPerWarp + group) * 4U;
        const std::uint32_t up_b0 =
            LoadU32(packed_weight_e2m1 + up_weight_offset);
        const std::uint32_t up_b1 =
            LoadU32(packed_weight_e2m1 + up_weight_offset + 16U);
        const std::uint32_t up_scale_b =
            LoadU32(weight_scales_e4m3fn + up_scale_offset);
        MmaNvfp4(up_accumulator[row_tile][0],
                  up_accumulator[row_tile][1],
                  up_accumulator[row_tile][2],
                  up_accumulator[row_tile][3], a0, a1, a2, a3, up_b0,
                  up_b1, scale_a, up_scale_b);
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (std::uint64_t row_tile = 0U;
       row_tile < kGroupedRowTilesPerWarp; ++row_tile) {
    const std::uint64_t output_column =
        (first_row_tile + row_tile) * kRowsPerWarp + thread_in_group * 2U;
#pragma unroll
    for (unsigned pair = 0; pair < 4U; ++pair) {
      const bool high = (pair & 2U) != 0U;
      const std::uint32_t grouped = high ? grouped_high : grouped_low;
      const std::uint64_t column = output_column + (pair & 1U);
      if (grouped >= grouped_end || column >= rows) continue;
      std::uint32_t output_row = grouped;
      if constexpr (!kFusedGateUp) {
        output_row = permutation[grouped];
        if (output_row >= assignment_count) continue;
      }
      if constexpr (kFusedGateUp) {
        const float gate = static_cast<float>(__float2bfloat16_rn(
            accumulator[row_tile][pair] / output_divisor));
        const float up = static_cast<float>(__float2bfloat16_rn(
            up_accumulator[row_tile][pair] / output_divisor));
        const float inner =
            kSqrtTwoOverPi * (gate + kGeluCubic * gate * gate * gate);
        const float gelu = static_cast<float>(
            __float2bfloat16_rn(0.5F * gate * (1.0F + tanhf(inner))));
        StorePhysicalOrContainerBf16(
            output,
            static_cast<std::uint64_t>(output_row) * rows + column,
            __float2bfloat16_rn(gelu * up));
      } else {
        StorePhysicalOrContainerBf16(
            output,
            static_cast<std::uint64_t>(output_row) * rows + column,
            __float2bfloat16_rn(accumulator[row_tile][pair] /
                                output_divisor));
      }
    }
  }
#else
  (void)packed_activation_e2m1;
  (void)activation_scales_e4m3fn;
  (void)packed_weight_e2m1;
  (void)weight_scales_e4m3fn;
  (void)assignments;
  (void)permutation;
  (void)prefix;
  (void)expert_tiles;
  (void)expert_tile_count;
  (void)output;
  (void)assignment_count;
  (void)rows;
  (void)contracting_elements;
  (void)experts;
  (void)output_divisor;
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
  if (rows == 0U || rows % kRowsPerWarp != 0U || contracting_elements == 0U ||
      contracting_elements % kElementsPerKBlock != 0U) {
    return Invalid("SM120 NVFP4 direct projection requires rows divisible by 8 and K divisible by 64");
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
  Sm120DirectProjectionKernel<false, 0><<<
      dim3(static_cast<unsigned>(blocks), 1U), kThreadsPerBlock, 0, stream>>>(
      packed_activation_e2m1, activation_scales_e4m3fn, packed_weight_e2m1,
      weight_scales_e4m3fn, nullptr, nullptr, nullptr, 0U, 0U, output, 1U, rows,
      contracting_elements, output_divisor);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch scale-tiled SM120 NVFP4 projection", error);
}

Status LaunchNvfp4Sm120DirectProjectionBf16Float(
    const std::uint8_t* packed_activation_e2m1,
    const std::uint8_t* activation_scales_e4m3fn,
    const std::uint8_t* packed_weight_e2m1,
    const std::uint8_t* weight_scales_e4m3fn, float* output,
    std::uint64_t rows, std::uint64_t contracting_elements,
    float activation_global_divisor, float weight_global_divisor,
    cudaStream_t stream) {
  if (packed_activation_e2m1 == nullptr ||
      activation_scales_e4m3fn == nullptr || packed_weight_e2m1 == nullptr ||
      weight_scales_e4m3fn == nullptr || output == nullptr || rows == 0U ||
      rows % kRowsPerWarp != 0U || contracting_elements == 0U ||
      contracting_elements % kElementsPerKBlock != 0U ||
      !PositiveFinite(activation_global_divisor) ||
      !PositiveFinite(weight_global_divisor)) {
    return Invalid("SM120 NVFP4 BF16-float projection contract is invalid");
  }
  const float output_divisor =
      activation_global_divisor * weight_global_divisor;
  if (!PositiveFinite(output_divisor)) {
    return Invalid("SM120 NVFP4 BF16-float projection divisor overflowed");
  }
  const std::uint64_t row_tiles = rows / kRowsPerWarp;
  const std::uint64_t blocks =
      (row_tiles + kWarpsPerBlock - 1U) / kWarpsPerBlock;
  if (blocks > static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max())) {
    return Invalid("SM120 NVFP4 BF16-float projection grid exceeds CUDA limits");
  }
  Sm120DirectProjectionKernel<true, 0><<<
      dim3(static_cast<unsigned>(blocks), 1U), kThreadsPerBlock, 0, stream>>>(
      packed_activation_e2m1, activation_scales_e4m3fn,
      packed_weight_e2m1, weight_scales_e4m3fn, nullptr, nullptr, nullptr,
      0U, 0U, output,
      1U, rows, contracting_elements, output_divisor);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch SM120 NVFP4 BF16-float projection", error);
}

Status LaunchNvfp4Sm120DirectProjectionBf16FloatExactBatch(
    const std::uint8_t* packed_activation_e2m1,
    const std::uint8_t* activation_scales_e4m3fn,
    const std::uint8_t* packed_weight_e2m1,
    const std::uint8_t* weight_scales_e4m3fn, float* output,
    std::uint64_t tokens, std::uint64_t rows,
    std::uint64_t contracting_elements, float activation_global_divisor,
    float weight_global_divisor, cudaStream_t stream) {
  if (packed_activation_e2m1 == nullptr ||
      activation_scales_e4m3fn == nullptr || packed_weight_e2m1 == nullptr ||
      weight_scales_e4m3fn == nullptr || output == nullptr || tokens == 0U ||
      tokens > 65535U || rows == 0U || rows % kRowsPerWarp != 0U ||
      contracting_elements == 0U ||
      contracting_elements % kElementsPerKBlock != 0U ||
      !PositiveFinite(activation_global_divisor) ||
      !PositiveFinite(weight_global_divisor)) {
    return Invalid("exact-batch SM120 NVFP4 projection contract is invalid");
  }
  const float divisor = activation_global_divisor * weight_global_divisor;
  if (!PositiveFinite(divisor)) {
    return Invalid("exact-batch SM120 NVFP4 projection divisor overflowed");
  }
  const std::uint64_t row_tiles = rows / kRowsPerWarp;
  const std::uint64_t blocks =
      (row_tiles + kWarpsPerBlock - 1U) / kWarpsPerBlock;
  if (blocks > static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max())) {
    return Invalid("exact-batch SM120 NVFP4 projection grid exceeds CUDA limits");
  }
  Sm120DirectProjectionKernel<true, 0><<<
      dim3(static_cast<unsigned>(blocks), static_cast<unsigned>(tokens)),
      kThreadsPerBlock, 0, stream>>>(
      packed_activation_e2m1, activation_scales_e4m3fn,
      packed_weight_e2m1, weight_scales_e4m3fn, nullptr, nullptr, nullptr,
      0U, 0U, output, tokens, rows, contracting_elements, divisor);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch exact-batch SM120 NVFP4 projection", error);
}

Status LaunchNvfp4Sm120SelectedDirectProjectionBf16Float(
    const std::uint8_t* packed_activation_e2m1,
    const std::uint8_t* activation_scales_e4m3fn,
    const std::uint8_t* packed_expert_weight_e2m1,
    const std::uint8_t* expert_weight_scales_e4m3fn,
    const std::uint32_t* selected_ids, std::uint32_t slot, float* output,
    std::uint64_t rows_per_expert, std::uint64_t contracting_elements,
    std::uint32_t experts, float activation_global_divisor,
    float weight_global_divisor, cudaStream_t stream) {
  if (packed_activation_e2m1 == nullptr ||
      activation_scales_e4m3fn == nullptr ||
      packed_expert_weight_e2m1 == nullptr ||
      expert_weight_scales_e4m3fn == nullptr || selected_ids == nullptr ||
      output == nullptr || experts == 0U || rows_per_expert == 0U ||
      rows_per_expert % kRowsPerWarp != 0U || contracting_elements == 0U ||
      contracting_elements % kElementsPerKBlock != 0U ||
      !PositiveFinite(activation_global_divisor) ||
      !PositiveFinite(weight_global_divisor)) {
    return Invalid("device-selected SM120 NVFP4 projection contract is invalid");
  }
  const float output_divisor =
      activation_global_divisor * weight_global_divisor;
  if (!PositiveFinite(output_divisor)) {
    return Invalid("device-selected SM120 NVFP4 projection divisor overflowed");
  }
  const std::uint64_t row_tiles = rows_per_expert / kRowsPerWarp;
  const std::uint64_t blocks =
      (row_tiles + kWarpsPerBlock - 1U) / kWarpsPerBlock;
  if (blocks > static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max())) {
    return Invalid("device-selected SM120 NVFP4 projection grid exceeds CUDA limits");
  }
  Sm120DirectProjectionKernel<true, 1><<<
      dim3(static_cast<unsigned>(blocks), 1U), kThreadsPerBlock, 0, stream>>>(
      packed_activation_e2m1, activation_scales_e4m3fn,
      packed_expert_weight_e2m1, expert_weight_scales_e4m3fn, selected_ids,
      nullptr, nullptr, slot, experts, output, 1U, rows_per_expert, contracting_elements,
      output_divisor);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch device-selected SM120 NVFP4 projection",
                           error);
}

Status LaunchNvfp4Sm120SelectedDirectProjectionBf16FloatBatch(
    const std::uint8_t* packed_activation_e2m1,
    const std::uint8_t* activation_scales_e4m3fn,
    const std::uint8_t* packed_expert_weight_e2m1,
    const std::uint8_t* expert_weight_scales_e4m3fn,
    const std::uint32_t* selected_ids, std::uint32_t top_k, float* output,
    std::uint64_t rows_per_expert, std::uint64_t contracting_elements,
    std::uint32_t experts, float activation_global_divisor,
    float weight_global_divisor, cudaStream_t stream) {
  if (packed_activation_e2m1 == nullptr ||
      activation_scales_e4m3fn == nullptr ||
      packed_expert_weight_e2m1 == nullptr ||
      expert_weight_scales_e4m3fn == nullptr || selected_ids == nullptr ||
      output == nullptr || top_k == 0U || top_k > experts || experts == 0U ||
      rows_per_expert == 0U || rows_per_expert % kRowsPerWarp != 0U ||
      contracting_elements == 0U ||
      contracting_elements % kElementsPerKBlock != 0U ||
      !PositiveFinite(activation_global_divisor) ||
      !PositiveFinite(weight_global_divisor)) {
    return Invalid("batched selected SM120 NVFP4 projection contract is invalid");
  }
  const float output_divisor =
      activation_global_divisor * weight_global_divisor;
  if (!PositiveFinite(output_divisor)) {
    return Invalid("batched selected SM120 NVFP4 divisor overflowed");
  }
  const std::uint64_t row_tiles = rows_per_expert / kRowsPerWarp;
  const std::uint64_t blocks =
      (row_tiles + kWarpsPerBlock - 1U) / kWarpsPerBlock;
  if (blocks >
      static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max())) {
    return Invalid("batched selected SM120 NVFP4 grid exceeds CUDA limits");
  }
  Sm120DirectProjectionKernel<true, 4><<<
      dim3(static_cast<unsigned>(blocks), top_k), kThreadsPerBlock, 0,
      stream>>>(packed_activation_e2m1, activation_scales_e4m3fn,
                packed_expert_weight_e2m1, expert_weight_scales_e4m3fn,
                selected_ids, nullptr, nullptr, 0U, experts, output, top_k,
                rows_per_expert, contracting_elements, output_divisor);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch batched selected SM120 NVFP4 projection",
                           error);
}

Status LaunchNvfp4Sm120GroupedExpertDown(
    const std::uint8_t* grouped_product_e2m1,
    const std::uint8_t* grouped_product_scales_e4m3fn,
    const std::uint8_t* packed_expert_weight_e2m1,
    const std::uint8_t* expert_weight_scales_e4m3fn,
    const Gemma4MoePrefillAssignment* assignments,
    const std::uint32_t* permutation, const std::uint32_t* prefix,
    const std::uint32_t* expert_tiles,
    const std::uint32_t* expert_tile_count, float* assignment_order_output,
    std::uint64_t assignment_count, std::uint64_t rows_per_expert,
    std::uint64_t contracting_elements, std::uint32_t experts,
    float activation_global_divisor, float weight_global_divisor,
    cudaStream_t stream) {
  if (grouped_product_e2m1 == nullptr ||
      !Aligned16(grouped_product_e2m1) ||
      grouped_product_scales_e4m3fn == nullptr ||
      packed_expert_weight_e2m1 == nullptr ||
      expert_weight_scales_e4m3fn == nullptr || assignments == nullptr ||
      permutation == nullptr || prefix == nullptr || expert_tiles == nullptr ||
      expert_tile_count == nullptr || assignment_order_output == nullptr ||
      assignment_count == 0U || assignment_count > 65535U || experts == 0U ||
      rows_per_expert == 0U || rows_per_expert % kRowsPerWarp != 0U ||
      contracting_elements == 0U ||
      contracting_elements % kElementsPerKBlock != 0U ||
      !PositiveFinite(activation_global_divisor) ||
      !PositiveFinite(weight_global_divisor)) {
    return Invalid("grouped SM120 expert Down contract is invalid");
  }
  const float divisor = activation_global_divisor * weight_global_divisor;
  if (!PositiveFinite(divisor)) {
    return Invalid("grouped SM120 expert Down divisor overflowed");
  }
  const std::uint64_t row_tiles = rows_per_expert / kRowsPerWarp;
  const std::uint64_t row_tile_groups =
      (row_tiles + kGroupedRowTilesPerWarp - 1U) /
      kGroupedRowTilesPerWarp;
  const std::uint64_t blocks =
      (row_tile_groups + kWarpsPerBlock - 1U) / kWarpsPerBlock;
  if (blocks > static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max())) {
    return Invalid("grouped SM120 expert Down grid exceeds CUDA limits");
  }
  const std::uint64_t tile_grid =
      (assignment_count + kTokensPerMma - 1U) / kTokensPerMma + experts;
  Sm120GroupedExpertMatrixKernel<false, float><<<
      dim3(static_cast<unsigned>(blocks), static_cast<unsigned>(tile_grid)),
      kThreadsPerBlock, 0, stream>>>(
      grouped_product_e2m1, grouped_product_scales_e4m3fn,
      packed_expert_weight_e2m1, expert_weight_scales_e4m3fn, assignments,
      permutation, prefix, expert_tiles, expert_tile_count,
      assignment_order_output, assignment_count, rows_per_expert,
      contracting_elements, experts, divisor);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch grouped SM120 expert Down", error);
}

Status LaunchNvfp4Sm120GroupedExpertDownBf16(
    const std::uint8_t* grouped_product_e2m1,
    const std::uint8_t* grouped_product_scales_e4m3fn,
    const std::uint8_t* packed_expert_weight_e2m1,
    const std::uint8_t* expert_weight_scales_e4m3fn,
    const Gemma4MoePrefillAssignment* assignments,
    const std::uint32_t* permutation, const std::uint32_t* prefix,
    const std::uint32_t* expert_tiles,
    const std::uint32_t* expert_tile_count,
    std::uint16_t* assignment_order_output_bf16,
    std::uint64_t assignment_count, std::uint64_t rows_per_expert,
    std::uint64_t contracting_elements, std::uint32_t experts,
    float activation_global_divisor, float weight_global_divisor,
    cudaStream_t stream) {
  if (grouped_product_e2m1 == nullptr ||
      !Aligned16(grouped_product_e2m1) ||
      grouped_product_scales_e4m3fn == nullptr ||
      packed_expert_weight_e2m1 == nullptr ||
      expert_weight_scales_e4m3fn == nullptr || assignments == nullptr ||
      permutation == nullptr || prefix == nullptr || expert_tiles == nullptr ||
      expert_tile_count == nullptr ||
      assignment_order_output_bf16 == nullptr || assignment_count == 0U ||
      assignment_count > 65535U || experts == 0U ||
      rows_per_expert == 0U || rows_per_expert % kRowsPerWarp != 0U ||
      contracting_elements == 0U ||
      contracting_elements % kElementsPerKBlock != 0U ||
      !PositiveFinite(activation_global_divisor) ||
      !PositiveFinite(weight_global_divisor)) {
    return Invalid("grouped physical-BF16 SM120 expert Down contract is invalid");
  }
  const float divisor = activation_global_divisor * weight_global_divisor;
  if (!PositiveFinite(divisor)) {
    return Invalid(
        "grouped physical-BF16 SM120 expert Down divisor overflowed");
  }
  const std::uint64_t row_tiles = rows_per_expert / kRowsPerWarp;
  const std::uint64_t row_tile_groups =
      (row_tiles + kGroupedRowTilesPerWarp - 1U) /
      kGroupedRowTilesPerWarp;
  const std::uint64_t blocks =
      (row_tile_groups + kWarpsPerBlock - 1U) / kWarpsPerBlock;
  if (blocks >
      static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max())) {
    return Invalid(
        "grouped physical-BF16 SM120 expert Down grid exceeds CUDA limits");
  }
  const std::uint64_t tile_grid =
      (assignment_count + kTokensPerMma - 1U) / kTokensPerMma + experts;
  Sm120GroupedExpertMatrixKernel<false, std::uint16_t><<<
      dim3(static_cast<unsigned>(blocks), static_cast<unsigned>(tile_grid)),
      kThreadsPerBlock, 0, stream>>>(
      grouped_product_e2m1, grouped_product_scales_e4m3fn,
      packed_expert_weight_e2m1, expert_weight_scales_e4m3fn, assignments,
      permutation, prefix, expert_tiles, expert_tile_count,
      assignment_order_output_bf16, assignment_count, rows_per_expert,
      contracting_elements, experts, divisor);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure(
                   "launch grouped physical-BF16 SM120 expert Down", error);
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
      rows % kRowsPerWarp != 0U ||
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
             : CudaFailure("launch batched scale-tiled SM120 NVFP4 projection", error);
}

Status LaunchNvfp4Sm120DirectProjectionBf16Batch(
    const std::uint8_t* packed_activation_e2m1,
    const std::uint8_t* activation_scales_e4m3fn,
    const std::uint8_t* packed_weight_e2m1,
    const std::uint8_t* weight_scales_e4m3fn, std::uint16_t* output_bf16,
    std::uint64_t tokens, std::uint64_t rows,
    std::uint64_t contracting_elements, float activation_global_divisor,
    float weight_global_divisor, cudaStream_t stream) {
  if (packed_activation_e2m1 == nullptr ||
      activation_scales_e4m3fn == nullptr || packed_weight_e2m1 == nullptr ||
      weight_scales_e4m3fn == nullptr || output_bf16 == nullptr) {
    return Invalid("batched SM120 NVFP4 BF16 projection requires non-null device pointers");
  }
  if (tokens == 0U || tokens > 65535U || rows == 0U ||
      rows % kRowsPerWarp != 0U || contracting_elements == 0U ||
      contracting_elements % kElementsPerKBlock != 0U ||
      !PositiveFinite(activation_global_divisor) ||
      !PositiveFinite(weight_global_divisor)) {
    return Invalid("batched SM120 NVFP4 BF16 projection geometry or divisors are invalid");
  }
  const float output_divisor =
      activation_global_divisor * weight_global_divisor;
  if (!PositiveFinite(output_divisor)) {
    return Invalid("batched SM120 NVFP4 BF16 projection divisor product overflowed");
  }
  const std::uint64_t row_tiles =
      (rows + kRowsPerWarp - 1U) / kRowsPerWarp;
  const bool short_batch = tokens <= 5U;
  const bool d2_batch = tokens == 3U;
  const unsigned block_warps =
      short_batch && !d2_batch ? kWarpsPerBlock : kPrefillWarpsPerBlock;
  const std::uint64_t blocks =
      (row_tiles + block_warps - 1U) / block_warps;
  if (blocks >
      static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max())) {
    return Invalid("batched SM120 NVFP4 BF16 projection grid exceeds CUDA limits");
  }
  const std::uint64_t token_tiles =
      (tokens + kTokensPerMma - 1U) / kTokensPerMma;
  if (short_batch) {
    if (d2_batch) {
      Sm120MatrixProjectionKernel<false, 1U, kPrefillWarpsPerBlock, false,
                                  std::uint16_t><<<
          dim3(static_cast<unsigned>(blocks),
               static_cast<unsigned>(token_tiles)),
          kPrefillThreadsPerBlock, 0, stream>>>(
          packed_activation_e2m1, activation_scales_e4m3fn,
          packed_weight_e2m1, weight_scales_e4m3fn, nullptr, nullptr,
          nullptr, nullptr, output_bf16, tokens, rows,
          contracting_elements, output_divisor, 1.0F);
    } else {
      Sm120MatrixProjectionKernel<false, 1U, kWarpsPerBlock, false,
                                  std::uint16_t><<<
          dim3(static_cast<unsigned>(blocks),
               static_cast<unsigned>(token_tiles)),
          kThreadsPerBlock, 0, stream>>>(
          packed_activation_e2m1, activation_scales_e4m3fn,
          packed_weight_e2m1, weight_scales_e4m3fn, nullptr, nullptr,
          nullptr, nullptr, output_bf16, tokens, rows,
          contracting_elements, output_divisor, 1.0F);
    }
  } else {
    const std::uint64_t grouped_token_tiles =
        (token_tiles + kPrefillTokenTilesPerWarp - 1U) /
        kPrefillTokenTilesPerWarp;
    Sm120MatrixProjectionKernel<false, kPrefillTokenTilesPerWarp,
                                kPrefillWarpsPerBlock, true,
                                std::uint16_t><<<
        dim3(static_cast<unsigned>(blocks),
             static_cast<unsigned>(grouped_token_tiles)),
        kPrefillThreadsPerBlock, 0, stream>>>(
        packed_activation_e2m1, activation_scales_e4m3fn,
        packed_weight_e2m1, weight_scales_e4m3fn, nullptr, nullptr, nullptr,
        nullptr, output_bf16, tokens, rows, contracting_elements,
        output_divisor, 1.0F);
  }
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch batched SM120 NVFP4 BF16 projection", error);
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
  if (rows == 0U || rows % kRowsPerWarp != 0U || contracting_elements == 0U ||
      contracting_elements % kElementsPerKBlock != 0U) {
    return Invalid("SM120 fused Gate/Up requires rows divisible by 8 and K divisible by 64");
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
  Sm120FusedGateUpKernel<0><<<dim3(static_cast<unsigned>(blocks), 1U),
                                  kThreadsPerBlock, 0, stream>>>(
      packed_activation_e2m1, activation_scales_e4m3fn, packed_gate_weight_e2m1,
      gate_weight_scales_e4m3fn, packed_up_weight_e2m1, up_weight_scales_e4m3fn,
      nullptr, nullptr, nullptr, 0U, 0U, gate_output, up_output,
      product_output, 1U, rows,
      contracting_elements, gate_output_divisor, up_output_divisor);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess ? Status::Ok()
                              : CudaFailure("launch fused SM120 NVFP4 Gate/Up", error);
}

Status LaunchNvfp4Sm120FusedGateUpExactBatch(
    const std::uint8_t* packed_activation_e2m1,
    const std::uint8_t* activation_scales_e4m3fn,
    const std::uint8_t* packed_gate_weight_e2m1,
    const std::uint8_t* gate_weight_scales_e4m3fn,
    const std::uint8_t* packed_up_weight_e2m1,
    const std::uint8_t* up_weight_scales_e4m3fn, float* product_output,
    std::uint64_t tokens, std::uint64_t rows,
    std::uint64_t contracting_elements,
    float gate_activation_global_divisor, float gate_weight_global_divisor,
    float up_activation_global_divisor, float up_weight_global_divisor,
    cudaStream_t stream) {
  if (packed_activation_e2m1 == nullptr ||
      activation_scales_e4m3fn == nullptr ||
      packed_gate_weight_e2m1 == nullptr ||
      gate_weight_scales_e4m3fn == nullptr ||
      packed_up_weight_e2m1 == nullptr || up_weight_scales_e4m3fn == nullptr ||
      product_output == nullptr || tokens == 0U || tokens > 65535U ||
      rows == 0U || rows % kRowsPerWarp != 0U ||
      contracting_elements == 0U ||
      contracting_elements % kElementsPerKBlock != 0U ||
      !PositiveFinite(gate_activation_global_divisor) ||
      !PositiveFinite(gate_weight_global_divisor) ||
      !PositiveFinite(up_activation_global_divisor) ||
      !PositiveFinite(up_weight_global_divisor)) {
    return Invalid("exact-batch SM120 fused Gate/Up contract is invalid");
  }
  const float gate_divisor =
      gate_activation_global_divisor * gate_weight_global_divisor;
  const float up_divisor =
      up_activation_global_divisor * up_weight_global_divisor;
  if (!PositiveFinite(gate_divisor) || !PositiveFinite(up_divisor)) {
    return Invalid("exact-batch SM120 fused Gate/Up divisor overflowed");
  }
  const std::uint64_t row_tiles = rows / kRowsPerWarp;
  const std::uint64_t blocks =
      (row_tiles + kWarpsPerBlock - 1U) / kWarpsPerBlock;
  if (blocks > static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max())) {
    return Invalid("exact-batch SM120 fused Gate/Up grid exceeds CUDA limits");
  }
  Sm120FusedGateUpKernel<0><<<
      dim3(static_cast<unsigned>(blocks), static_cast<unsigned>(tokens)),
      kThreadsPerBlock, 0, stream>>>(
      packed_activation_e2m1, activation_scales_e4m3fn,
      packed_gate_weight_e2m1, gate_weight_scales_e4m3fn,
      packed_up_weight_e2m1, up_weight_scales_e4m3fn, nullptr, nullptr,
      nullptr, 0U, 0U, nullptr, nullptr, product_output, tokens, rows,
      contracting_elements, gate_divisor, up_divisor);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch exact-batch SM120 fused Gate/Up", error);
}

Status LaunchNvfp4Sm120SelectedFusedGateUp(
    const std::uint8_t* packed_activation_e2m1,
    const std::uint8_t* activation_scales_e4m3fn,
    const std::uint8_t* packed_expert_gate_up_weight_e2m1,
    const std::uint8_t* expert_gate_up_weight_scales_e4m3fn,
    const std::uint32_t* selected_ids, std::uint32_t slot,
    float* gate_output, float* up_output, float* product_output,
    std::uint64_t rows, std::uint64_t contracting_elements,
    std::uint32_t experts, float activation_global_divisor,
    float weight_global_divisor, cudaStream_t stream) {
  if (packed_activation_e2m1 == nullptr ||
      activation_scales_e4m3fn == nullptr ||
      packed_expert_gate_up_weight_e2m1 == nullptr ||
      expert_gate_up_weight_scales_e4m3fn == nullptr ||
      selected_ids == nullptr || product_output == nullptr ||
      (gate_output == nullptr) != (up_output == nullptr) || experts == 0U ||
      rows == 0U || rows % kRowsPerWarp != 0U ||
      contracting_elements == 0U ||
      contracting_elements % kElementsPerKBlock != 0U ||
      !PositiveFinite(activation_global_divisor) ||
      !PositiveFinite(weight_global_divisor)) {
    return Invalid("device-selected SM120 fused Gate/Up contract is invalid");
  }
  const float output_divisor =
      activation_global_divisor * weight_global_divisor;
  if (!PositiveFinite(output_divisor)) {
    return Invalid("device-selected SM120 fused Gate/Up divisor overflowed");
  }
  const std::uint64_t row_tiles = rows / kRowsPerWarp;
  const std::uint64_t blocks =
      (row_tiles + kWarpsPerBlock - 1U) / kWarpsPerBlock;
  if (blocks > static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max())) {
    return Invalid("device-selected SM120 fused Gate/Up grid exceeds CUDA limits");
  }
  Sm120FusedGateUpKernel<1><<<dim3(static_cast<unsigned>(blocks), 1U),
                                 kThreadsPerBlock, 0, stream>>>(
      packed_activation_e2m1, activation_scales_e4m3fn,
      packed_expert_gate_up_weight_e2m1,
      expert_gate_up_weight_scales_e4m3fn,
      packed_expert_gate_up_weight_e2m1,
      expert_gate_up_weight_scales_e4m3fn, selected_ids, nullptr, nullptr,
      slot, experts,
      gate_output, up_output, product_output, 1U, rows,
      contracting_elements, output_divisor, output_divisor);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch device-selected SM120 fused Gate/Up",
                           error);
}

Status LaunchNvfp4Sm120SelectedFusedGateUpBatch(
    const std::uint8_t* packed_activation_e2m1,
    const std::uint8_t* activation_scales_e4m3fn,
    const std::uint8_t* packed_expert_gate_up_weight_e2m1,
    const std::uint8_t* expert_gate_up_weight_scales_e4m3fn,
    const std::uint32_t* selected_ids, std::uint32_t top_k,
    float* gate_output, float* up_output, float* product_output,
    std::uint64_t rows, std::uint64_t contracting_elements,
    std::uint32_t experts, float activation_global_divisor,
    float weight_global_divisor, cudaStream_t stream) {
  if (packed_activation_e2m1 == nullptr ||
      activation_scales_e4m3fn == nullptr ||
      packed_expert_gate_up_weight_e2m1 == nullptr ||
      expert_gate_up_weight_scales_e4m3fn == nullptr ||
      selected_ids == nullptr || product_output == nullptr || top_k == 0U ||
      top_k > experts || (gate_output == nullptr) != (up_output == nullptr) ||
      experts == 0U || rows == 0U || rows % kRowsPerWarp != 0U ||
      contracting_elements == 0U ||
      contracting_elements % kElementsPerKBlock != 0U ||
      !PositiveFinite(activation_global_divisor) ||
      !PositiveFinite(weight_global_divisor)) {
    return Invalid("batched selected SM120 fused Gate/Up contract is invalid");
  }
  const float output_divisor =
      activation_global_divisor * weight_global_divisor;
  if (!PositiveFinite(output_divisor)) {
    return Invalid("batched selected SM120 fused Gate/Up divisor overflowed");
  }
  const std::uint64_t row_tiles = rows / kRowsPerWarp;
  const std::uint64_t blocks =
      (row_tiles + kWarpsPerBlock - 1U) / kWarpsPerBlock;
  if (blocks >
      static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max())) {
    return Invalid("batched selected SM120 fused Gate/Up grid exceeds CUDA limits");
  }
  Sm120FusedGateUpKernel<3><<<dim3(static_cast<unsigned>(blocks), top_k),
                                 kThreadsPerBlock, 0, stream>>>(
      packed_activation_e2m1, activation_scales_e4m3fn,
      packed_expert_gate_up_weight_e2m1,
      expert_gate_up_weight_scales_e4m3fn,
      packed_expert_gate_up_weight_e2m1,
      expert_gate_up_weight_scales_e4m3fn, selected_ids, nullptr, nullptr, 0U,
      experts, gate_output, up_output, product_output, top_k, rows,
      contracting_elements, output_divisor, output_divisor);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch batched selected SM120 fused Gate/Up",
                           error);
}

Status LaunchNvfp4Sm120GroupedExpertFusedGateUp(
    const std::uint8_t* token_activation_e2m1,
    const std::uint8_t* token_activation_scales_e4m3fn,
    const std::uint8_t* packed_expert_gate_up_weight_e2m1,
    const std::uint8_t* expert_gate_up_weight_scales_e4m3fn,
    const Gemma4MoePrefillAssignment* assignments,
    const std::uint32_t* permutation, const std::uint32_t* prefix,
    const std::uint32_t* expert_tiles,
    const std::uint32_t* expert_tile_count, float* grouped_product_output,
    std::uint64_t assignment_count, std::uint64_t rows,
    std::uint64_t contracting_elements, std::uint32_t experts,
    float activation_global_divisor, float weight_global_divisor,
    cudaStream_t stream) {
  if (token_activation_e2m1 == nullptr ||
      !Aligned16(token_activation_e2m1) ||
      token_activation_scales_e4m3fn == nullptr ||
      packed_expert_gate_up_weight_e2m1 == nullptr ||
      expert_gate_up_weight_scales_e4m3fn == nullptr ||
      assignments == nullptr || permutation == nullptr ||
      prefix == nullptr || expert_tiles == nullptr ||
      expert_tile_count == nullptr || grouped_product_output == nullptr ||
      assignment_count == 0U ||
      assignment_count > 65535U || experts == 0U || rows == 0U ||
      rows % kRowsPerWarp != 0U || contracting_elements == 0U ||
      contracting_elements % kElementsPerKBlock != 0U ||
      !PositiveFinite(activation_global_divisor) ||
      !PositiveFinite(weight_global_divisor)) {
    return Invalid("grouped SM120 expert Gate/Up contract is invalid");
  }
  const float divisor = activation_global_divisor * weight_global_divisor;
  if (!PositiveFinite(divisor)) {
    return Invalid("grouped SM120 expert Gate/Up divisor overflowed");
  }
  const std::uint64_t row_tiles = rows / kRowsPerWarp;
  const std::uint64_t row_tile_groups =
      (row_tiles + kGroupedRowTilesPerWarp - 1U) /
      kGroupedRowTilesPerWarp;
  const std::uint64_t blocks =
      (row_tile_groups + kWarpsPerBlock - 1U) / kWarpsPerBlock;
  if (blocks > static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max())) {
    return Invalid("grouped SM120 expert Gate/Up grid exceeds CUDA limits");
  }
  const std::uint64_t tile_grid =
      (assignment_count + kTokensPerMma - 1U) / kTokensPerMma + experts;
  Sm120GroupedExpertMatrixKernel<true, float><<<
      dim3(static_cast<unsigned>(blocks), static_cast<unsigned>(tile_grid)),
      kThreadsPerBlock, 0, stream>>>(
      token_activation_e2m1, token_activation_scales_e4m3fn,
      packed_expert_gate_up_weight_e2m1,
      expert_gate_up_weight_scales_e4m3fn, assignments, permutation, prefix,
      expert_tiles, expert_tile_count, grouped_product_output,
      assignment_count, rows, contracting_elements, experts, divisor);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch grouped SM120 expert Gate/Up", error);
}

Status LaunchNvfp4Sm120GroupedExpertFusedGateUpBf16(
    const std::uint8_t* token_activation_e2m1,
    const std::uint8_t* token_activation_scales_e4m3fn,
    const std::uint8_t* packed_expert_gate_up_weight_e2m1,
    const std::uint8_t* expert_gate_up_weight_scales_e4m3fn,
    const Gemma4MoePrefillAssignment* assignments,
    const std::uint32_t* permutation, const std::uint32_t* prefix,
    const std::uint32_t* expert_tiles,
    const std::uint32_t* expert_tile_count,
    std::uint16_t* grouped_product_output_bf16,
    std::uint64_t assignment_count, std::uint64_t rows,
    std::uint64_t contracting_elements, std::uint32_t experts,
    float activation_global_divisor, float weight_global_divisor,
    cudaStream_t stream) {
  if (token_activation_e2m1 == nullptr ||
      !Aligned16(token_activation_e2m1) ||
      token_activation_scales_e4m3fn == nullptr ||
      packed_expert_gate_up_weight_e2m1 == nullptr ||
      expert_gate_up_weight_scales_e4m3fn == nullptr ||
      assignments == nullptr || permutation == nullptr || prefix == nullptr ||
      expert_tiles == nullptr || expert_tile_count == nullptr ||
      grouped_product_output_bf16 == nullptr || assignment_count == 0U ||
      assignment_count > 65535U || experts == 0U || rows == 0U ||
      rows % kRowsPerWarp != 0U || contracting_elements == 0U ||
      contracting_elements % kElementsPerKBlock != 0U ||
      !PositiveFinite(activation_global_divisor) ||
      !PositiveFinite(weight_global_divisor)) {
    return Invalid(
        "grouped physical-BF16 SM120 expert Gate/Up contract is invalid");
  }
  const float divisor = activation_global_divisor * weight_global_divisor;
  if (!PositiveFinite(divisor)) {
    return Invalid(
        "grouped physical-BF16 SM120 expert Gate/Up divisor overflowed");
  }
  const std::uint64_t row_tiles = rows / kRowsPerWarp;
  const std::uint64_t row_tile_groups =
      (row_tiles + kGroupedRowTilesPerWarp - 1U) /
      kGroupedRowTilesPerWarp;
  const std::uint64_t blocks =
      (row_tile_groups + kWarpsPerBlock - 1U) / kWarpsPerBlock;
  if (blocks >
      static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max())) {
    return Invalid(
        "grouped physical-BF16 SM120 expert Gate/Up grid exceeds CUDA limits");
  }
  const std::uint64_t tile_grid =
      (assignment_count + kTokensPerMma - 1U) / kTokensPerMma + experts;
  Sm120GroupedExpertMatrixKernel<true, std::uint16_t><<<
      dim3(static_cast<unsigned>(blocks), static_cast<unsigned>(tile_grid)),
      kThreadsPerBlock, 0, stream>>>(
      token_activation_e2m1, token_activation_scales_e4m3fn,
      packed_expert_gate_up_weight_e2m1,
      expert_gate_up_weight_scales_e4m3fn, assignments, permutation, prefix,
      expert_tiles, expert_tile_count, grouped_product_output_bf16,
      assignment_count, rows, contracting_elements, experts, divisor);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure(
                   "launch grouped physical-BF16 SM120 expert Gate/Up",
                   error);
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
      rows % kRowsPerWarp != 0U ||
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
      dim3(static_cast<unsigned>(blocks),
           static_cast<unsigned>(grouped_token_tiles)),
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

}  // namespace gem16::internal
