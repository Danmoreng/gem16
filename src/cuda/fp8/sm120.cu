#include "cuda/fp8/sm120.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

namespace gem16::internal {
namespace {

constexpr std::uint64_t kElementsPerKBlock = 32;
constexpr std::uint64_t kRowsPerWarp = 8;
constexpr std::uint64_t kTokensPerMma = 16;
constexpr std::uint64_t kTokenTilesPerWarp = 16;
constexpr std::uint64_t kKBlocksPerStage = 2;
constexpr std::uint64_t kElementsPerStage =
    kElementsPerKBlock * kKBlocksPerStage;
constexpr unsigned kWarpSize = 32;
constexpr unsigned kWarpsPerBlock = 4;
constexpr unsigned kThreadsPerBlock = kWarpSize * kWarpsPerBlock;
constexpr unsigned kSplitKWarpsPerBlock = 2U;
constexpr unsigned kSplitKThreadsPerBlock =
    kWarpSize * kSplitKWarpsPerBlock;
constexpr unsigned kMatrixWarpsPerBlock = 8;
constexpr unsigned kMatrixThreadsPerBlock =
    kWarpSize * kMatrixWarpsPerBlock;
constexpr std::uint64_t kGemma4Moe26BHidden = 2816U;
constexpr std::uint64_t kGemma4Moe26BLocalAttentionWidth = 4096U;
constexpr std::uint64_t kGemma4Moe26BLocalKvWidth = 2048U;
constexpr std::uint64_t kGemma4Moe26BGlobalAttentionWidth = 8192U;
constexpr unsigned kGemma4Moe26BLocalGroupedBlocks =
    static_cast<unsigned>((kGemma4Moe26BLocalAttentionWidth +
                           2U * kGemma4Moe26BLocalKvWidth) /
                          (kWarpsPerBlock * kRowsPerWarp));

Status Invalid(std::string message) {
  return Status(StatusCode::kInvalidArgument, std::move(message));
}

Status CudaFailure(const char* operation, cudaError_t error) {
  return Status(StatusCode::kInternal,
                std::string(operation) + ": " + cudaGetErrorName(error) + ": " +
                    cudaGetErrorString(error));
}

__device__ __forceinline__ std::uint32_t LoadU32(const std::uint8_t* source) {
  return *reinterpret_cast<const std::uint32_t*>(source);
}

__device__ __forceinline__ unsigned SharedAddress(const void* pointer) {
  return static_cast<unsigned>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void CopyAsync16(void* shared_destination,
                                            const void* global_source,
                                            int source_bytes) {
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

__device__ __forceinline__ float DecodeBf16(const std::uint16_t* source) {
  const __nv_bfloat16 value = __ushort_as_bfloat16(*source);
  return static_cast<float>(value);
}

struct Fp8Accumulator {
  float x0 = 0.0F;
  float x1 = 0.0F;
  float x2 = 0.0F;
  float x3 = 0.0F;
};

struct Fp8MatrixBinding {
  const std::uint8_t* weight = nullptr;
  const std::uint16_t* weight_scales = nullptr;
  float* output = nullptr;
  std::uint64_t rows = 0U;
};

__device__ __forceinline__ void AccumulateFp8(std::uint32_t a0,
                                               std::uint32_t a1,
                                               std::uint32_t a2,
                                               std::uint32_t a3,
                                               std::uint32_t b0,
                                               std::uint32_t b1,
                                               Fp8Accumulator& accumulator) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, "
      "{%8, %9}, "
      "{%10, %11, %12, %13};\n"
      : "+f"(accumulator.x0), "+f"(accumulator.x1),
        "+f"(accumulator.x2), "+f"(accumulator.x3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
        "f"(accumulator.x0), "f"(accumulator.x1),
        "f"(accumulator.x2), "f"(accumulator.x3));
}

__device__ __forceinline__ void StageFp8OperandsAsync(
    std::uint32_t* staged_activation, std::uint32_t* staged_weight,
    const std::uint8_t* activation, const std::uint8_t* weight,
    std::uint64_t token_base, std::uint64_t row_base,
    std::uint64_t k_stage, std::uint64_t tokens, std::uint64_t rows,
    std::uint64_t contracting_elements) {
  constexpr unsigned kStagedTokens =
      static_cast<unsigned>(kTokensPerMma * kTokenTilesPerWarp);
  constexpr unsigned kStagedRows = kMatrixWarpsPerBlock * kRowsPerWarp;
  constexpr unsigned kCopiesPerRow =
      static_cast<unsigned>(kElementsPerStage / 16U);
  constexpr unsigned kWordsPerStage =
      static_cast<unsigned>(kElementsPerStage / 4U);
  const std::uint64_t k_base = k_stage * kElementsPerStage;

  for (unsigned copy = threadIdx.x;
       copy < kStagedTokens * kCopiesPerRow; copy += blockDim.x) {
    const unsigned token_offset = copy / kCopiesPerRow;
    const unsigned chunk = copy % kCopiesPerRow;
    const std::uint64_t token = token_base + token_offset;
    const bool valid = token < tokens &&
                       k_base + static_cast<std::uint64_t>(chunk) * 16U <
                           contracting_elements;
    const std::uint8_t* source =
        valid ? activation + token * contracting_elements + k_base +
                    static_cast<std::uint64_t>(chunk) * 16U
              : activation;
    CopyAsync16(staged_activation + token_offset * kWordsPerStage + chunk * 4U,
                source, valid ? 16 : 0);
  }
  for (unsigned copy = threadIdx.x;
       copy < kStagedRows * kCopiesPerRow; copy += blockDim.x) {
    const unsigned row_offset = copy / kCopiesPerRow;
    const unsigned chunk = copy % kCopiesPerRow;
    const std::uint64_t row = row_base + row_offset;
    const bool valid = row < rows &&
                       k_base + static_cast<std::uint64_t>(chunk) * 16U <
                           contracting_elements;
    const std::uint8_t* source =
        valid ? weight + row * contracting_elements + k_base +
                    static_cast<std::uint64_t>(chunk) * 16U
              : weight;
    CopyAsync16(staged_weight + row_offset * kWordsPerStage + chunk * 4U,
                source, valid ? 16 : 0);
  }
}

__global__ void Sm120DirectProjectionKernel(const std::uint8_t* activation,
                                            const float* activation_scale,
                                            Fp8MatrixBinding first,
                                            Fp8MatrixBinding second,
                                            Fp8MatrixBinding third,
                                            unsigned binding_count,
                                            std::uint64_t tokens,
                                            std::uint64_t contracting_elements) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  if (blockIdx.z >= binding_count) return;
  const Fp8MatrixBinding binding =
      blockIdx.z == 0U ? first : (blockIdx.z == 1U ? second : third);
  const std::uint8_t* weight = binding.weight;
  const std::uint16_t* weight_scales = binding.weight_scales;
  float* output = binding.output;
  const std::uint64_t rows = binding.rows;
  const std::uint64_t token = blockIdx.y;
  if (token >= tokens) return;
  activation += token * contracting_elements;
  activation_scale += token;
  output += token * rows;
  const unsigned warp = threadIdx.x / kWarpSize;
  const unsigned lane = threadIdx.x & (kWarpSize - 1U);
  const std::uint64_t global_warp =
      static_cast<std::uint64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  const std::uint64_t row_tiles = (rows + kRowsPerWarp - 1U) / kRowsPerWarp;
  if (global_warp >= row_tiles) return;

  const unsigned source_row_in_tile = lane >> 2U;
  const unsigned k_quarter = lane & 3U;
  const std::uint64_t source_row = global_warp * kRowsPerWarp + source_row_in_tile;
  const std::uint64_t k_blocks = contracting_elements / kElementsPerKBlock;

  float d0 = 0.0F;
  float d1 = 0.0F;
  float d2 = 0.0F;
  float d3 = 0.0F;
  for (std::uint64_t k_block = 0; k_block < k_blocks; ++k_block) {
    const std::uint64_t activation_byte =
        k_block * kElementsPerKBlock + static_cast<std::uint64_t>(k_quarter) * 4U;
    const std::uint32_t a_first = LoadU32(activation + activation_byte);
    const std::uint32_t a_second = LoadU32(activation + activation_byte + 16U);

    std::uint32_t b_first = 0;
    std::uint32_t b_second = 0;
    if (source_row < rows) {
      const std::uint64_t weight_byte = source_row * contracting_elements + activation_byte;
      b_first = LoadU32(weight + weight_byte);
      b_second = LoadU32(weight + weight_byte + 16U);
    }

    float next0 = 0.0F;
    float next1 = 0.0F;
    float next2 = 0.0F;
    float next3 = 0.0F;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(next0), "=f"(next1), "=f"(next2), "=f"(next3)
        : "r"(a_first), "r"(a_first), "r"(a_second), "r"(a_second), "r"(b_first),
          "r"(b_second), "f"(d0), "f"(d1), "f"(d2), "f"(d3));
    d0 = next0;
    d1 = next1;
    d2 = next2;
    d3 = next3;
  }

  if (lane < 4U) {
    const std::uint64_t output_row = global_warp * kRowsPerWarp + lane * 2U;
    const float input_scale = activation_scale[0];
    if (output_row < rows) {
      output[output_row] =
          d0 * input_scale * DecodeBf16(weight_scales + output_row);
    }
    if (output_row + 1U < rows) {
      output[output_row + 1U] =
          d1 * input_scale * DecodeBf16(weight_scales + output_row + 1U);
    }
  }
#else
  (void)activation;
  (void)activation_scale;
  (void)first;
  (void)second;
  (void)third;
  (void)binding_count;
  (void)tokens;
  (void)contracting_elements;
#endif
}

// The 25 local 26B layers project Q4096/K2048/V2048 from the same K2816
// activation. The generic grouped launch uses the Q-sized X grid for all three
// Z planes, leaving half of both K/V planes empty. Flatten the three exact
// block ranges into 256 useful CTAs while preserving the legacy per-row warp,
// source layout, MMA order and FP32 scaling boundary.
__launch_bounds__(kThreadsPerBlock, 1) __global__
void Sm120LocalGroupedQkvProjectionKernel(
    const std::uint8_t* activation, const float* activation_scale,
    Fp8MatrixBinding query, Fp8MatrixBinding key, Fp8MatrixBinding value,
    std::uint64_t contracting_elements) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  constexpr unsigned kRowsPerBlock =
      kWarpsPerBlock * static_cast<unsigned>(kRowsPerWarp);
  constexpr unsigned kQueryBlocks =
      static_cast<unsigned>(kGemma4Moe26BLocalAttentionWidth) / kRowsPerBlock;
  constexpr unsigned kKvBlocks =
      static_cast<unsigned>(kGemma4Moe26BLocalKvWidth) / kRowsPerBlock;
  const unsigned block = blockIdx.x;
  const bool is_query = block < kQueryBlocks;
  const bool is_key = !is_query && block < kQueryBlocks + kKvBlocks;
  const Fp8MatrixBinding binding =
      is_query ? query : (is_key ? key : value);
  const unsigned matrix_block =
      is_query ? block
               : (is_key ? block - kQueryBlocks
                         : block - kQueryBlocks - kKvBlocks);
  const unsigned warp = threadIdx.x / kWarpSize;
  const unsigned lane = threadIdx.x & (kWarpSize - 1U);
  const std::uint64_t global_warp =
      static_cast<std::uint64_t>(matrix_block) * kWarpsPerBlock + warp;
  const unsigned source_row_in_tile = lane >> 2U;
  const unsigned k_quarter = lane & 3U;
  const std::uint64_t source_row =
      global_warp * kRowsPerWarp + source_row_in_tile;
  const std::uint64_t k_blocks =
      contracting_elements / kElementsPerKBlock;

  float d0 = 0.0F;
  float d1 = 0.0F;
  float d2 = 0.0F;
  float d3 = 0.0F;
  for (std::uint64_t k_block = 0U; k_block < k_blocks; ++k_block) {
    const std::uint64_t activation_byte =
        k_block * kElementsPerKBlock +
        static_cast<std::uint64_t>(k_quarter) * 4U;
    const std::uint32_t a_first =
        LoadU32(activation + activation_byte);
    const std::uint32_t a_second =
        LoadU32(activation + activation_byte + 16U);
    const std::uint64_t weight_byte =
        source_row * contracting_elements + activation_byte;
    const std::uint32_t b_first = LoadU32(binding.weight + weight_byte);
    const std::uint32_t b_second =
        LoadU32(binding.weight + weight_byte + 16U);

    float next0 = 0.0F;
    float next1 = 0.0F;
    float next2 = 0.0F;
    float next3 = 0.0F;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(next0), "=f"(next1), "=f"(next2), "=f"(next3)
        : "r"(a_first), "r"(a_first), "r"(a_second), "r"(a_second),
          "r"(b_first), "r"(b_second), "f"(d0), "f"(d1), "f"(d2),
          "f"(d3));
    d0 = next0;
    d1 = next1;
    d2 = next2;
    d3 = next3;
  }

  if (lane < 4U) {
    const std::uint64_t output_row =
        global_warp * kRowsPerWarp + lane * 2U;
    const float input_scale = activation_scale[0];
    binding.output[output_row] =
        d0 * input_scale * DecodeBf16(binding.weight_scales + output_row);
    binding.output[output_row + 1U] =
        d1 * input_scale *
        DecodeBf16(binding.weight_scales + output_row + 1U);
  }
#else
  (void)activation;
  (void)activation_scale;
  (void)query;
  (void)key;
  (void)value;
  (void)contracting_elements;
#endif
}

// Gemma 4 26B's T=1 attention O projections expose only 352 output-row
// warps while contracting over 4K or 8K elements. Pair two warps on the same
// eight output rows, accumulate one contiguous K half per warp, and combine the
// two FP32 fragments inside the CTA. The accepted source-row weight layout and
// every weight byte remain unchanged.
__launch_bounds__(kSplitKThreadsPerBlock, 1) __global__
void Sm120DirectProjectionSplitK2Kernel(
    const std::uint8_t* activation, const float* activation_scale,
    const std::uint8_t* weight, const std::uint16_t* weight_scales,
    float* output, std::uint64_t rows,
    std::uint64_t contracting_elements) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  __shared__ float second_half[kRowsPerWarp];
  const std::uint64_t token = blockIdx.y;
  activation += token * contracting_elements;
  activation_scale += token;
  output += token * rows;
  const unsigned warp = threadIdx.x / kWarpSize;
  const unsigned lane = threadIdx.x & (kWarpSize - 1U);
  const unsigned split = warp;
  const std::uint64_t global_warp = blockIdx.x;
  const std::uint64_t row_tiles =
      (rows + kRowsPerWarp - 1U) / kRowsPerWarp;
  const unsigned source_row_in_tile = lane >> 2U;
  const unsigned k_quarter = lane & 3U;
  const std::uint64_t source_row =
      global_warp * kRowsPerWarp + source_row_in_tile;
  const std::uint64_t k_blocks =
      contracting_elements / kElementsPerKBlock;
  const std::uint64_t split_k_blocks = k_blocks / 2U;
  const std::uint64_t first_k_block = split * split_k_blocks;
  const std::uint64_t final_k_block = first_k_block + split_k_blocks;

  float d0 = 0.0F;
  float d1 = 0.0F;
  float d2 = 0.0F;
  float d3 = 0.0F;
  if (global_warp < row_tiles) {
    for (std::uint64_t k_block = first_k_block;
         k_block < final_k_block; ++k_block) {
      const std::uint64_t activation_byte =
          k_block * kElementsPerKBlock +
          static_cast<std::uint64_t>(k_quarter) * 4U;
      const std::uint32_t a_first =
          LoadU32(activation + activation_byte);
      const std::uint32_t a_second =
          LoadU32(activation + activation_byte + 16U);

      std::uint32_t b_first = 0U;
      std::uint32_t b_second = 0U;
      if (source_row < rows) {
        const std::uint64_t weight_byte =
            source_row * contracting_elements + activation_byte;
        b_first = LoadU32(weight + weight_byte);
        b_second = LoadU32(weight + weight_byte + 16U);
      }

      float next0 = 0.0F;
      float next1 = 0.0F;
      float next2 = 0.0F;
      float next3 = 0.0F;
      asm volatile(
          "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
          "{%0, %1, %2, %3}, "
          "{%4, %5, %6, %7}, "
          "{%8, %9}, "
          "{%10, %11, %12, %13};\n"
          : "=f"(next0), "=f"(next1), "=f"(next2), "=f"(next3)
          : "r"(a_first), "r"(a_first), "r"(a_second), "r"(a_second),
            "r"(b_first), "r"(b_second), "f"(d0), "f"(d1), "f"(d2),
            "f"(d3));
      d0 = next0;
      d1 = next1;
      d2 = next2;
      d3 = next3;
    }
  }

  const unsigned partial_row = lane * 2U;
  if (split == 1U && lane < 4U) {
    second_half[partial_row] = d0;
    second_half[partial_row + 1U] = d1;
  }
  __syncthreads();
  if (split == 0U && lane < 4U) {
    const std::uint64_t output_row =
        global_warp * kRowsPerWarp + lane * 2U;
    const float input_scale = activation_scale[0];
    if (output_row < rows) {
      output[output_row] =
          (d0 + second_half[partial_row]) * input_scale *
          DecodeBf16(weight_scales + output_row);
    }
    if (output_row + 1U < rows) {
      output[output_row + 1U] =
          (d1 + second_half[partial_row + 1U]) * input_scale *
          DecodeBf16(weight_scales + output_row + 1U);
    }
  }
#else
  (void)activation;
  (void)activation_scale;
  (void)weight;
  (void)weight_scales;
  (void)output;
  (void)rows;
  (void)contracting_elements;
#endif
}

// Fixed T=3 form of the 26B attention O projection. The two split-K warps
// retain the accepted half-K accumulation and combination order for each row,
// while every FP8 weight fragment feeds all three verifier activations before
// it is discarded.
__launch_bounds__(kSplitKThreadsPerBlock, 1) __global__
void Sm120DirectProjectionSplitK2FixedT3Kernel(
    const std::uint8_t* activation, const float* activation_scale,
    const std::uint8_t* weight, const std::uint16_t* weight_scales,
    float* output, std::uint64_t rows,
    std::uint64_t contracting_elements) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  constexpr unsigned kTokens = 3U;
  __shared__ float second_half[kTokens][kRowsPerWarp];
  const unsigned warp = threadIdx.x / kWarpSize;
  const unsigned lane = threadIdx.x & (kWarpSize - 1U);
  const unsigned split = warp;
  const std::uint64_t global_warp = blockIdx.x;
  const std::uint64_t row_tiles =
      (rows + kRowsPerWarp - 1U) / kRowsPerWarp;
  const unsigned source_row_in_tile = lane >> 2U;
  const unsigned k_quarter = lane & 3U;
  const std::uint64_t source_row =
      global_warp * kRowsPerWarp + source_row_in_tile;
  const std::uint64_t k_blocks =
      contracting_elements / kElementsPerKBlock;
  const std::uint64_t split_k_blocks = k_blocks / 2U;
  const std::uint64_t first_k_block = split * split_k_blocks;
  const std::uint64_t final_k_block = first_k_block + split_k_blocks;

  Fp8Accumulator accumulators[kTokens]{};
  if (global_warp < row_tiles) {
    for (std::uint64_t k_block = first_k_block;
         k_block < final_k_block; ++k_block) {
      const std::uint64_t activation_byte =
          k_block * kElementsPerKBlock +
          static_cast<std::uint64_t>(k_quarter) * 4U;
      std::uint32_t b_first = 0U;
      std::uint32_t b_second = 0U;
      if (source_row < rows) {
        const std::uint64_t weight_byte =
            source_row * contracting_elements + activation_byte;
        b_first = LoadU32(weight + weight_byte);
        b_second = LoadU32(weight + weight_byte + 16U);
      }
#pragma unroll
      for (unsigned token = 0U; token < kTokens; ++token) {
        const std::uint8_t* token_activation =
            activation +
            static_cast<std::uint64_t>(token) * contracting_elements;
        const std::uint32_t a_first =
            LoadU32(token_activation + activation_byte);
        const std::uint32_t a_second =
            LoadU32(token_activation + activation_byte + 16U);
        AccumulateFp8(a_first, a_first, a_second, a_second, b_first, b_second,
                      accumulators[token]);
      }
    }
  }

  const unsigned partial_row = lane * 2U;
  if (split == 1U && lane < 4U) {
#pragma unroll
    for (unsigned token = 0U; token < kTokens; ++token) {
      second_half[token][partial_row] = accumulators[token].x0;
      second_half[token][partial_row + 1U] = accumulators[token].x1;
    }
  }
  __syncthreads();
  if (split == 0U && lane < 4U) {
    const std::uint64_t output_row =
        global_warp * kRowsPerWarp + lane * 2U;
#pragma unroll
    for (unsigned token = 0U; token < kTokens; ++token) {
      const float input_scale = activation_scale[token];
      const std::uint64_t output_offset =
          static_cast<std::uint64_t>(token) * rows + output_row;
      if (output_row < rows) {
        output[output_offset] =
            (accumulators[token].x0 + second_half[token][partial_row]) *
            input_scale * DecodeBf16(weight_scales + output_row);
      }
      if (output_row + 1U < rows) {
        output[output_offset + 1U] =
            (accumulators[token].x1 +
             second_half[token][partial_row + 1U]) *
            input_scale * DecodeBf16(weight_scales + output_row + 1U);
      }
    }
  }
#else
  (void)activation;
  (void)activation_scale;
  (void)weight;
  (void)weight_scales;
  (void)output;
  (void)rows;
  (void)contracting_elements;
#endif
}

template <unsigned kTokens>
__global__ void Sm120DirectProjectionFixedBatchKernel(
    const std::uint8_t* activation, const float* activation_scale,
    Fp8MatrixBinding first, Fp8MatrixBinding second,
    Fp8MatrixBinding third, unsigned binding_count,
    std::uint64_t contracting_elements) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  static_assert(kTokens == 3U);
  if (blockIdx.z >= binding_count) return;
  const Fp8MatrixBinding binding =
      blockIdx.z == 0U ? first : (blockIdx.z == 1U ? second : third);
  const std::uint8_t* weight = binding.weight;
  const std::uint16_t* weight_scales = binding.weight_scales;
  float* output = binding.output;
  const std::uint64_t rows = binding.rows;
  const unsigned warp = threadIdx.x / kWarpSize;
  const unsigned lane = threadIdx.x & (kWarpSize - 1U);
  const std::uint64_t global_warp =
      static_cast<std::uint64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  const std::uint64_t row_tiles =
      (rows + kRowsPerWarp - 1U) / kRowsPerWarp;
  extern __shared__ __align__(16) std::uint8_t staged_activation[];
  const std::uint64_t activation_vectors =
      kTokens * contracting_elements / sizeof(uint4);
  auto* staged_vectors = reinterpret_cast<uint4*>(staged_activation);
  const auto* activation_vectors_source =
      reinterpret_cast<const uint4*>(activation);
  for (std::uint64_t vector = threadIdx.x; vector < activation_vectors;
       vector += blockDim.x) {
    staged_vectors[vector] = activation_vectors_source[vector];
  }
  __syncthreads();
  if (global_warp >= row_tiles) return;

  const unsigned source_row_in_tile = lane >> 2U;
  const unsigned k_quarter = lane & 3U;
  const std::uint64_t source_row =
      global_warp * kRowsPerWarp + source_row_in_tile;
  const std::uint64_t k_blocks =
      contracting_elements / kElementsPerKBlock;
  Fp8Accumulator accumulators[kTokens]{};
  for (std::uint64_t k_block = 0; k_block < k_blocks; ++k_block) {
    const std::uint64_t activation_byte =
        k_block * kElementsPerKBlock +
        static_cast<std::uint64_t>(k_quarter) * 4U;
    std::uint32_t b_first = 0U;
    std::uint32_t b_second = 0U;
    if (source_row < rows) {
      const std::uint64_t weight_byte =
          source_row * contracting_elements + activation_byte;
      b_first = LoadU32(weight + weight_byte);
      b_second = LoadU32(weight + weight_byte + 16U);
    }
#pragma unroll
    for (unsigned token = 0U; token < kTokens; ++token) {
      const std::uint8_t* token_activation =
          staged_activation + static_cast<std::uint64_t>(token) *
                                  contracting_elements;
      const std::uint32_t a_first =
          LoadU32(token_activation + activation_byte);
      const std::uint32_t a_second =
          LoadU32(token_activation + activation_byte + 16U);
      AccumulateFp8(a_first, a_first, a_second, a_second, b_first, b_second,
                    accumulators[token]);
    }
  }

  if (lane < 4U) {
    const std::uint64_t output_row =
        global_warp * kRowsPerWarp + lane * 2U;
    float2 output_scales{};
    if (output_row + 1U < rows) {
      output_scales = __bfloat1622float2(
          *reinterpret_cast<const __nv_bfloat162*>(weight_scales +
                                                    output_row));
    } else if (output_row < rows) {
      output_scales.x = DecodeBf16(weight_scales + output_row);
    }
#pragma unroll
    for (unsigned token = 0U; token < kTokens; ++token) {
      const float input_scale = activation_scale[token];
      const std::uint64_t output_offset =
          static_cast<std::uint64_t>(token) * rows + output_row;
      if (output_row + 1U < rows) {
        const float2 values{
            accumulators[token].x0 * input_scale * output_scales.x,
            accumulators[token].x1 * input_scale * output_scales.y};
        *reinterpret_cast<float2*>(output + output_offset) = values;
      } else if (output_row < rows) {
        output[output_offset] =
            accumulators[token].x0 * input_scale * output_scales.x;
      }
    }
  }
#else
  (void)activation;
  (void)activation_scale;
  (void)first;
  (void)second;
  (void)third;
  (void)binding_count;
  (void)contracting_elements;
#endif
}

// Eight warps form one 64-column by 256-token CTA tile. Two consecutive K32
// fragments of source-layout FP8 activation and weights are double-buffered
// through shared memory, while every weight fragment is reused for sixteen
// independent 16-token MMA tiles. This is the sole matrix path; T=1 remains
// on the latency-oriented direct kernel above.
__global__ void Sm120MatrixProjectionKernel(
    const std::uint8_t* activation, const float* activation_scale,
    Fp8MatrixBinding first, Fp8MatrixBinding second,
    Fp8MatrixBinding third, unsigned binding_count, std::uint64_t tokens,
    std::uint64_t contracting_elements) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  if (blockIdx.z >= binding_count) return;
  const Fp8MatrixBinding binding =
      blockIdx.z == 0U ? first : (blockIdx.z == 1U ? second : third);
  const std::uint8_t* weight = binding.weight;
  const std::uint16_t* weight_scales = binding.weight_scales;
  float* output = binding.output;
  const std::uint64_t rows = binding.rows;
  constexpr unsigned kStagedTokens =
      static_cast<unsigned>(kTokensPerMma * kTokenTilesPerWarp);
  constexpr unsigned kStagedRows = kMatrixWarpsPerBlock * kRowsPerWarp;
  constexpr unsigned kWordsPerStage =
      static_cast<unsigned>(kElementsPerStage / 4U);
  __shared__ alignas(16) std::uint32_t staged_activation[2]
      [kStagedTokens * kWordsPerStage];
  __shared__ alignas(16) std::uint32_t staged_weight[2]
      [kStagedRows * kWordsPerStage];
  const unsigned warp = threadIdx.x / kWarpSize;
  const unsigned lane = threadIdx.x & (kWarpSize - 1U);
  const unsigned group = lane >> 2U;
  const unsigned thread_in_group = lane & 3U;
  const std::uint64_t global_warp =
      static_cast<std::uint64_t>(blockIdx.x) * kMatrixWarpsPerBlock + warp;
  const std::uint64_t row_tiles = (rows + kRowsPerWarp - 1U) / kRowsPerWarp;
  const bool warp_active = global_warp < row_tiles;

  const std::uint64_t token_base = static_cast<std::uint64_t>(blockIdx.y) *
                                   kTokensPerMma * kTokenTilesPerWarp;
  const std::uint64_t row_base =
      static_cast<std::uint64_t>(blockIdx.x) * kStagedRows;
  if (row_base >= rows) return;
  const unsigned staged_weight_row = warp * kRowsPerWarp + group;
  const std::uint64_t k_blocks = contracting_elements / kElementsPerKBlock;
  const std::uint64_t k_stages =
      (k_blocks + kKBlocksPerStage - 1U) / kKBlocksPerStage;

  Fp8Accumulator accumulators[kTokenTilesPerWarp]{};
  StageFp8OperandsAsync(staged_activation[0], staged_weight[0], activation,
                        weight, token_base, row_base, 0U, tokens, rows,
                        contracting_elements);
  CommitAsyncCopies();
  WaitForAsyncCopies();
  __syncthreads();
  for (std::uint64_t k_stage = 0; k_stage < k_stages; ++k_stage) {
    const unsigned current_stage = static_cast<unsigned>(k_stage & 1U);
    const bool has_next_stage = k_stage + 1U < k_stages;
    if (has_next_stage) {
      const unsigned next_stage = current_stage ^ 1U;
      StageFp8OperandsAsync(staged_activation[next_stage],
                            staged_weight[next_stage], activation, weight,
                            token_base, row_base, k_stage + 1U, tokens, rows,
                            contracting_elements);
      CommitAsyncCopies();
    }
#pragma unroll
    for (unsigned k_sub = 0; k_sub < kKBlocksPerStage; ++k_sub) {
      const unsigned k_word = k_sub *
                                  static_cast<unsigned>(kElementsPerKBlock / 4U) +
                              thread_in_group;
      const std::uint32_t b0 =
          staged_weight[current_stage]
                       [staged_weight_row * kWordsPerStage + k_word];
      const std::uint32_t b1 =
          staged_weight[current_stage]
                       [staged_weight_row * kWordsPerStage + k_word + 4U];
#pragma unroll
      for (unsigned tile = 0; tile < kTokenTilesPerWarp; ++tile) {
        const unsigned staged_low = tile * kTokensPerMma + group;
        const unsigned staged_high = staged_low + 8U;
        const std::uint32_t a0 =
            staged_activation[current_stage]
                             [staged_low * kWordsPerStage + k_word];
        const std::uint32_t a1 =
            staged_activation[current_stage]
                             [staged_high * kWordsPerStage + k_word];
        const std::uint32_t a2 =
            staged_activation[current_stage]
                             [staged_low * kWordsPerStage + k_word + 4U];
        const std::uint32_t a3 =
            staged_activation[current_stage]
                             [staged_high * kWordsPerStage + k_word + 4U];
        AccumulateFp8(a0, a1, a2, a3, b0, b1, accumulators[tile]);
      }
    }
    if (has_next_stage) {
      WaitForAsyncCopies();
      __syncthreads();
    }
  }

  const std::uint64_t output_column =
      global_warp * kRowsPerWarp + thread_in_group * 2U;
  const float weight_scale0 = output_column < rows
                                  ? DecodeBf16(weight_scales + output_column)
                                  : 0.0F;
  const float weight_scale1 = output_column + 1U < rows
                                  ? DecodeBf16(weight_scales + output_column + 1U)
                                  : 0.0F;
#pragma unroll
  for (unsigned tile = 0; tile < kTokenTilesPerWarp; ++tile) {
    const float values[4] = {accumulators[tile].x0, accumulators[tile].x1,
                             accumulators[tile].x2, accumulators[tile].x3};
#pragma unroll
    for (unsigned pair = 0; pair < 4U; ++pair) {
      const std::uint64_t token =
          token_base + static_cast<std::uint64_t>(tile) * kTokensPerMma +
          ((pair & 2U) == 0U ? group : group + 8U);
      const std::uint64_t column = output_column + (pair & 1U);
      if (!warp_active || token >= tokens || column >= rows) continue;
      const float input_scale = activation_scale[token];
      output[token * rows + column] =
          values[pair] * input_scale *
          ((pair & 1U) == 0U ? weight_scale0 : weight_scale1);
    }
  }
#else
  (void)activation;
  (void)activation_scale;
  (void)first;
  (void)second;
  (void)third;
  (void)binding_count;
  (void)tokens;
  (void)contracting_elements;
#endif
}

}  // namespace

Status LaunchFp8Sm120DirectProjection(const std::uint8_t* activation_e4m3fn,
                                      const float* activation_scale,
                                      const std::uint8_t* weight_e4m3fn,
                                      const std::uint16_t* weight_scales_bf16, float* output,
                                      std::uint64_t rows, std::uint64_t contracting_elements,
                                      cudaStream_t stream) {
  if (activation_e4m3fn == nullptr || activation_scale == nullptr || weight_e4m3fn == nullptr ||
      weight_scales_bf16 == nullptr || output == nullptr) {
    return Invalid("SM120 FP8 direct projection requires non-null device pointers");
  }
  if (rows == 0U || contracting_elements == 0U ||
      contracting_elements % kElementsPerKBlock != 0U) {
    return Invalid("SM120 FP8 direct projection requires positive dimensions and K divisible by 32");
  }
  const bool gemma4_moe_26b_attention_output =
      rows == kGemma4Moe26BHidden &&
      (contracting_elements == kGemma4Moe26BLocalAttentionWidth ||
       contracting_elements == kGemma4Moe26BGlobalAttentionWidth);
  const std::uint64_t row_tiles = (rows + kRowsPerWarp - 1U) / kRowsPerWarp;
  const std::uint64_t blocks = (row_tiles + kWarpsPerBlock - 1U) / kWarpsPerBlock;
  if (blocks > static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max())) {
    return Invalid("SM120 FP8 direct projection grid exceeds CUDA limits");
  }
  if (gemma4_moe_26b_attention_output) {
    Sm120DirectProjectionSplitK2Kernel<<<
        dim3(static_cast<unsigned>(row_tiles), 1U),
        kSplitKThreadsPerBlock, 0, stream>>>(
        activation_e4m3fn, activation_scale, weight_e4m3fn,
        weight_scales_bf16, output, rows, contracting_elements);
    const cudaError_t split_k_error = cudaGetLastError();
    return split_k_error == cudaSuccess
               ? Status::Ok()
               : CudaFailure(
                     "launch Gemma 4 26B split-K2 SM120 FP8 projection",
                     split_k_error);
  }
  const Fp8MatrixBinding binding{weight_e4m3fn, weight_scales_bf16, output,
                                 rows};
  const Fp8MatrixBinding empty{};
  Sm120DirectProjectionKernel<<<dim3(static_cast<unsigned>(blocks), 1U),
                                kThreadsPerBlock, 0, stream>>>(
      activation_e4m3fn, activation_scale, binding, empty, empty, 1U, 1U,
      contracting_elements);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch direct-source SM120 FP8 projection", error);
}

Status LaunchFp8Sm120DirectProjectionLegacyForTest(
    const std::uint8_t* activation_e4m3fn, const float* activation_scale,
    const std::uint8_t* weight_e4m3fn,
    const std::uint16_t* weight_scales_bf16, float* output,
    std::uint64_t rows, std::uint64_t contracting_elements,
    cudaStream_t stream) {
  if (activation_e4m3fn == nullptr || activation_scale == nullptr ||
      weight_e4m3fn == nullptr || weight_scales_bf16 == nullptr ||
      output == nullptr || rows == 0U || contracting_elements == 0U ||
      contracting_elements % kElementsPerKBlock != 0U) {
    return Invalid("legacy SM120 FP8 differential arguments are invalid");
  }
  const std::uint64_t row_tiles =
      (rows + kRowsPerWarp - 1U) / kRowsPerWarp;
  const std::uint64_t blocks =
      (row_tiles + kWarpsPerBlock - 1U) / kWarpsPerBlock;
  if (blocks >
      static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max())) {
    return Invalid("legacy SM120 FP8 differential grid exceeds CUDA limits");
  }
  const Fp8MatrixBinding binding{weight_e4m3fn, weight_scales_bf16, output,
                                 rows};
  const Fp8MatrixBinding empty{};
  Sm120DirectProjectionKernel<<<dim3(static_cast<unsigned>(blocks), 1U),
                                kThreadsPerBlock, 0, stream>>>(
      activation_e4m3fn, activation_scale, binding, empty, empty, 1U, 1U,
      contracting_elements);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch legacy SM120 FP8 differential", error);
}

Status LaunchFp8Sm120GroupedQkvProjection(
    const std::uint8_t* activation_e4m3fn, const float* activation_scale,
    const std::uint8_t* q_weight_e4m3fn,
    const std::uint16_t* q_weight_scales_bf16, float* q_output,
    std::uint64_t q_rows, const std::uint8_t* k_weight_e4m3fn,
    const std::uint16_t* k_weight_scales_bf16, float* k_output,
    std::uint64_t k_rows, const std::uint8_t* v_weight_e4m3fn,
    const std::uint16_t* v_weight_scales_bf16, float* v_output,
    std::uint64_t v_rows, std::uint64_t contracting_elements,
    cudaStream_t stream) {
  if (activation_e4m3fn == nullptr || activation_scale == nullptr ||
      q_weight_e4m3fn == nullptr || q_weight_scales_bf16 == nullptr ||
      q_output == nullptr || k_weight_e4m3fn == nullptr ||
      k_weight_scales_bf16 == nullptr || k_output == nullptr) {
    return Invalid(
        "grouped direct SM120 FP8 Q/K projections require non-null device "
        "pointers");
  }
  const bool has_v = v_weight_e4m3fn != nullptr ||
                     v_weight_scales_bf16 != nullptr || v_output != nullptr ||
                     v_rows != 0U;
  if (has_v && (v_weight_e4m3fn == nullptr ||
                v_weight_scales_bf16 == nullptr || v_output == nullptr ||
                v_rows == 0U)) {
    return Invalid(
        "grouped direct SM120 FP8 V projection binding is incomplete");
  }
  if (q_rows == 0U || k_rows == 0U || contracting_elements == 0U ||
      contracting_elements % kElementsPerKBlock != 0U) {
    return Invalid("grouped direct SM120 FP8 projection dimensions are invalid");
  }
  const std::uint64_t maximum_rows =
      has_v ? std::max(q_rows, std::max(k_rows, v_rows))
            : std::max(q_rows, k_rows);
  const std::uint64_t row_tiles =
      (maximum_rows + kRowsPerWarp - 1U) / kRowsPerWarp;
  const std::uint64_t blocks =
      (row_tiles + kWarpsPerBlock - 1U) / kWarpsPerBlock;
  if (blocks >
      static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max())) {
    return Invalid(
        "grouped direct SM120 FP8 projection grid exceeds CUDA limits");
  }
  const Fp8MatrixBinding q{q_weight_e4m3fn, q_weight_scales_bf16, q_output,
                           q_rows};
  const Fp8MatrixBinding k{k_weight_e4m3fn, k_weight_scales_bf16, k_output,
                           k_rows};
  const Fp8MatrixBinding v{v_weight_e4m3fn, v_weight_scales_bf16, v_output,
                           v_rows};
  const bool gemma4_moe_26b_local_qkv =
      has_v && contracting_elements == kGemma4Moe26BHidden &&
      q_rows == kGemma4Moe26BLocalAttentionWidth &&
      k_rows == kGemma4Moe26BLocalKvWidth &&
      v_rows == kGemma4Moe26BLocalKvWidth;
  if (gemma4_moe_26b_local_qkv) {
    Sm120LocalGroupedQkvProjectionKernel<<<
        kGemma4Moe26BLocalGroupedBlocks, kThreadsPerBlock, 0, stream>>>(
        activation_e4m3fn, activation_scale, q, k, v,
        contracting_elements);
    const cudaError_t local_qkv_error = cudaGetLastError();
    return local_qkv_error == cudaSuccess
               ? Status::Ok()
               : CudaFailure(
                     "launch Gemma 4 26B compact local SM120 FP8 Q/K/V",
                     local_qkv_error);
  }
  Sm120DirectProjectionKernel<<<
      dim3(static_cast<unsigned>(blocks), 1U, has_v ? 3U : 2U),
      kThreadsPerBlock, 0, stream>>>(
      activation_e4m3fn, activation_scale, q, k, v, has_v ? 3U : 2U, 1U,
      contracting_elements);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure(
                   "launch grouped direct-source SM120 FP8 projection", error);
}

Status LaunchFp8Sm120LocalGroupedQkvProjectionLegacyForTest(
    const std::uint8_t* activation_e4m3fn, const float* activation_scale,
    const std::uint8_t* q_weight_e4m3fn,
    const std::uint16_t* q_weight_scales_bf16, float* q_output,
    const std::uint8_t* k_weight_e4m3fn,
    const std::uint16_t* k_weight_scales_bf16, float* k_output,
    const std::uint8_t* v_weight_e4m3fn,
    const std::uint16_t* v_weight_scales_bf16, float* v_output,
    cudaStream_t stream) {
  if (activation_e4m3fn == nullptr || activation_scale == nullptr ||
      q_weight_e4m3fn == nullptr || q_weight_scales_bf16 == nullptr ||
      q_output == nullptr || k_weight_e4m3fn == nullptr ||
      k_weight_scales_bf16 == nullptr || k_output == nullptr ||
      v_weight_e4m3fn == nullptr || v_weight_scales_bf16 == nullptr ||
      v_output == nullptr) {
    return Invalid("legacy local FP8 Q/K/V differential pointers are invalid");
  }
  const Fp8MatrixBinding q{q_weight_e4m3fn, q_weight_scales_bf16, q_output,
                           kGemma4Moe26BLocalAttentionWidth};
  const Fp8MatrixBinding k{k_weight_e4m3fn, k_weight_scales_bf16, k_output,
                           kGemma4Moe26BLocalKvWidth};
  const Fp8MatrixBinding v{v_weight_e4m3fn, v_weight_scales_bf16, v_output,
                           kGemma4Moe26BLocalKvWidth};
  constexpr unsigned kLegacyBlocks =
      static_cast<unsigned>(kGemma4Moe26BLocalAttentionWidth /
                            (kWarpsPerBlock * kRowsPerWarp));
  Sm120DirectProjectionKernel<<<dim3(kLegacyBlocks, 1U, 3U),
                                kThreadsPerBlock, 0, stream>>>(
      activation_e4m3fn, activation_scale, q, k, v, 3U, 1U,
      kGemma4Moe26BHidden);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch legacy local SM120 FP8 Q/K/V differential",
                           error);
}

Status LaunchFp8Sm120DirectProjectionBatch(
    const std::uint8_t* activation_e4m3fn, const float* activation_scales,
    const std::uint8_t* weight_e4m3fn,
    const std::uint16_t* weight_scales_bf16, float* output,
    std::uint64_t tokens, std::uint64_t rows,
    std::uint64_t contracting_elements, cudaStream_t stream) {
  if (activation_e4m3fn == nullptr || activation_scales == nullptr ||
      weight_e4m3fn == nullptr || weight_scales_bf16 == nullptr ||
      output == nullptr) {
    return Invalid("batched SM120 FP8 direct projection requires non-null device pointers");
  }
  if (tokens == 0U || tokens > 65535U || rows == 0U ||
      contracting_elements == 0U ||
      contracting_elements % kElementsPerKBlock != 0U) {
    return Invalid("batched SM120 FP8 projection dimensions are invalid");
  }
  const std::uint64_t row_tiles = (rows + kRowsPerWarp - 1U) / kRowsPerWarp;
  const bool gemma4_moe_26b_attention_output =
      rows == kGemma4Moe26BHidden &&
      (contracting_elements == kGemma4Moe26BLocalAttentionWidth ||
       contracting_elements == kGemma4Moe26BGlobalAttentionWidth);
  if (gemma4_moe_26b_attention_output && tokens <= 5U) {
    if (row_tiles >
        static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max())) {
      return Invalid("batched split-K2 SM120 FP8 projection grid exceeds CUDA limits");
    }
    if (tokens == 3U) {
      Sm120DirectProjectionSplitK2FixedT3Kernel<<<
          static_cast<unsigned>(row_tiles), kSplitKThreadsPerBlock, 0,
          stream>>>(activation_e4m3fn, activation_scales, weight_e4m3fn,
                    weight_scales_bf16, output, rows,
                    contracting_elements);
    } else {
      Sm120DirectProjectionSplitK2Kernel<<<
          dim3(static_cast<unsigned>(row_tiles),
               static_cast<unsigned>(tokens)),
          kSplitKThreadsPerBlock, 0, stream>>>(
          activation_e4m3fn, activation_scales, weight_e4m3fn,
          weight_scales_bf16, output, rows, contracting_elements);
    }
    const cudaError_t split_k_error = cudaGetLastError();
    return split_k_error == cudaSuccess
               ? Status::Ok()
               : CudaFailure(
                     "launch batched Gemma 4 26B split-K2 SM120 FP8 projection",
                     split_k_error);
  }
  if (tokens <= 5U) {
    const std::uint64_t direct_blocks =
        (row_tiles + kWarpsPerBlock - 1U) / kWarpsPerBlock;
    if (direct_blocks >
        static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max())) {
      return Invalid("short-batch direct SM120 FP8 projection grid exceeds CUDA limits");
    }
    const Fp8MatrixBinding binding{weight_e4m3fn, weight_scales_bf16,
                                   output, rows};
    const Fp8MatrixBinding empty{};
    if (tokens == 3U) {
      const std::size_t shared_bytes =
          static_cast<std::size_t>(tokens * contracting_elements);
      Sm120DirectProjectionFixedBatchKernel<3U><<<
          dim3(static_cast<unsigned>(direct_blocks), 1U), kThreadsPerBlock,
          shared_bytes, stream>>>(activation_e4m3fn, activation_scales,
                                  binding, empty, empty, 1U,
                                  contracting_elements);
    } else {
      Sm120DirectProjectionKernel<<<
          dim3(static_cast<unsigned>(direct_blocks),
               static_cast<unsigned>(tokens)),
          kThreadsPerBlock, 0, stream>>>(
          activation_e4m3fn, activation_scales, binding, empty, empty, 1U,
          tokens, contracting_elements);
    }
    const cudaError_t direct_error = cudaGetLastError();
    return direct_error == cudaSuccess
               ? Status::Ok()
               : CudaFailure(
                     "launch short-batch direct-source SM120 FP8 projection",
                     direct_error);
  }
  const std::uint64_t blocks =
      (row_tiles + kMatrixWarpsPerBlock - 1U) / kMatrixWarpsPerBlock;
  if (blocks > static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max())) {
    return Invalid("batched SM120 FP8 projection grid exceeds CUDA limits");
  }
  const std::uint64_t token_tiles =
      (tokens + kTokensPerMma - 1U) / kTokensPerMma;
  const std::uint64_t grouped_token_tiles =
      (token_tiles + kTokenTilesPerWarp - 1U) / kTokenTilesPerWarp;
  const Fp8MatrixBinding binding{weight_e4m3fn, weight_scales_bf16, output,
                                 rows};
  const Fp8MatrixBinding empty{};
  Sm120MatrixProjectionKernel<<<dim3(static_cast<unsigned>(blocks),
                                     static_cast<unsigned>(grouped_token_tiles)),
                                kMatrixThreadsPerBlock, 0, stream>>>(
      activation_e4m3fn, activation_scales, binding, empty, empty, 1U, tokens,
      contracting_elements);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch batched direct-source SM120 FP8 projection", error);
}

Status LaunchFp8Sm120DirectProjectionBatchLegacyForTest(
    const std::uint8_t* activation_e4m3fn, const float* activation_scales,
    const std::uint8_t* weight_e4m3fn,
    const std::uint16_t* weight_scales_bf16, float* output,
    std::uint64_t tokens, std::uint64_t rows,
    std::uint64_t contracting_elements, cudaStream_t stream) {
  if (activation_e4m3fn == nullptr || activation_scales == nullptr ||
      weight_e4m3fn == nullptr || weight_scales_bf16 == nullptr ||
      output == nullptr || tokens == 0U || tokens > 65535U || rows == 0U ||
      contracting_elements == 0U ||
      contracting_elements % kElementsPerKBlock != 0U) {
    return Invalid("legacy batched split-K2 SM120 FP8 contract is invalid");
  }
  const std::uint64_t row_tiles =
      (rows + kRowsPerWarp - 1U) / kRowsPerWarp;
  if (row_tiles >
      static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max())) {
    return Invalid("legacy batched split-K2 SM120 FP8 grid exceeds CUDA limits");
  }
  Sm120DirectProjectionSplitK2Kernel<<<
      dim3(static_cast<unsigned>(row_tiles), static_cast<unsigned>(tokens)),
      kSplitKThreadsPerBlock, 0, stream>>>(
      activation_e4m3fn, activation_scales, weight_e4m3fn,
      weight_scales_bf16, output, rows, contracting_elements);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch legacy batched split-K2 SM120 FP8",
                           error);
}

Status LaunchFp8Sm120GroupedQkvProjectionBatch(
    const std::uint8_t* activation_e4m3fn, const float* activation_scales,
    const std::uint8_t* q_weight_e4m3fn,
    const std::uint16_t* q_weight_scales_bf16, float* q_output,
    std::uint64_t q_rows, const std::uint8_t* k_weight_e4m3fn,
    const std::uint16_t* k_weight_scales_bf16, float* k_output,
    std::uint64_t k_rows, const std::uint8_t* v_weight_e4m3fn,
    const std::uint16_t* v_weight_scales_bf16, float* v_output,
    std::uint64_t v_rows, std::uint64_t tokens,
    std::uint64_t contracting_elements, cudaStream_t stream) {
  if (activation_e4m3fn == nullptr || activation_scales == nullptr ||
      q_weight_e4m3fn == nullptr || q_weight_scales_bf16 == nullptr ||
      q_output == nullptr || k_weight_e4m3fn == nullptr ||
      k_weight_scales_bf16 == nullptr || k_output == nullptr) {
    return Invalid("grouped SM120 FP8 Q/K projections require non-null device pointers");
  }
  const bool has_v = v_weight_e4m3fn != nullptr ||
                     v_weight_scales_bf16 != nullptr || v_output != nullptr ||
                     v_rows != 0U;
  if (has_v && (v_weight_e4m3fn == nullptr ||
                v_weight_scales_bf16 == nullptr || v_output == nullptr ||
                v_rows == 0U)) {
    return Invalid("grouped SM120 FP8 V projection binding is incomplete");
  }
  if (tokens == 0U || tokens > 65535U || q_rows == 0U || k_rows == 0U ||
      contracting_elements == 0U ||
      contracting_elements % kElementsPerKBlock != 0U) {
    return Invalid("grouped SM120 FP8 Q/K/V dimensions are invalid");
  }
  const std::uint64_t maximum_rows =
      has_v ? std::max(q_rows, std::max(k_rows, v_rows))
            : std::max(q_rows, k_rows);
  const std::uint64_t row_tiles =
      (maximum_rows + kRowsPerWarp - 1U) / kRowsPerWarp;
  if (tokens <= 5U) {
    const std::uint64_t direct_blocks =
        (row_tiles + kWarpsPerBlock - 1U) / kWarpsPerBlock;
    if (direct_blocks >
        static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max())) {
      return Invalid("grouped direct SM120 FP8 Q/K/V grid exceeds CUDA limits");
    }
    const Fp8MatrixBinding q{q_weight_e4m3fn, q_weight_scales_bf16, q_output,
                             q_rows};
    const Fp8MatrixBinding k{k_weight_e4m3fn, k_weight_scales_bf16, k_output,
                             k_rows};
    const Fp8MatrixBinding v{v_weight_e4m3fn, v_weight_scales_bf16, v_output,
                             v_rows};
    if (tokens == 3U) {
      const std::size_t shared_bytes =
          static_cast<std::size_t>(tokens * contracting_elements);
      Sm120DirectProjectionFixedBatchKernel<3U><<<
          dim3(static_cast<unsigned>(direct_blocks), 1U,
               has_v ? 3U : 2U),
          kThreadsPerBlock, shared_bytes, stream>>>(
          activation_e4m3fn, activation_scales, q, k, v,
          has_v ? 3U : 2U, contracting_elements);
    } else {
      Sm120DirectProjectionKernel<<<
          dim3(static_cast<unsigned>(direct_blocks),
               static_cast<unsigned>(tokens), has_v ? 3U : 2U),
          kThreadsPerBlock, 0, stream>>>(
          activation_e4m3fn, activation_scales, q, k, v,
          has_v ? 3U : 2U, tokens, contracting_elements);
    }
    const cudaError_t direct_error = cudaGetLastError();
    return direct_error == cudaSuccess
               ? Status::Ok()
               : CudaFailure(
                     "launch grouped short-batch direct SM120 FP8 Q/K/V projection",
                     direct_error);
  }
  const std::uint64_t blocks =
      (row_tiles + kMatrixWarpsPerBlock - 1U) / kMatrixWarpsPerBlock;
  if (blocks > static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max())) {
    return Invalid("grouped SM120 FP8 Q/K/V grid exceeds CUDA limits");
  }
  const std::uint64_t token_tiles =
      (tokens + kTokensPerMma - 1U) / kTokensPerMma;
  const std::uint64_t grouped_token_tiles =
      (token_tiles + kTokenTilesPerWarp - 1U) / kTokenTilesPerWarp;
  const Fp8MatrixBinding q{q_weight_e4m3fn, q_weight_scales_bf16, q_output,
                           q_rows};
  const Fp8MatrixBinding k{k_weight_e4m3fn, k_weight_scales_bf16, k_output,
                           k_rows};
  const Fp8MatrixBinding v{v_weight_e4m3fn, v_weight_scales_bf16, v_output,
                           v_rows};
  Sm120MatrixProjectionKernel<<<
      dim3(static_cast<unsigned>(blocks),
           static_cast<unsigned>(grouped_token_tiles), has_v ? 3U : 2U),
      kMatrixThreadsPerBlock, 0, stream>>>(
      activation_e4m3fn, activation_scales, q, k, v, has_v ? 3U : 2U,
      tokens, contracting_elements);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch grouped SM120 FP8 Q/K/V projection", error);
}

}  // namespace gem16::internal
