#include "cuda/fp8/sm120.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <string>
#include <utility>

namespace gem16gb::internal {
namespace {

constexpr std::uint64_t kElementsPerKBlock = 32;
constexpr std::uint64_t kRowsPerWarp = 8;
constexpr std::uint64_t kTokensPerMma = 16;
constexpr unsigned kWarpSize = 32;
constexpr unsigned kWarpsPerBlock = 4;
constexpr unsigned kThreadsPerBlock = kWarpSize * kWarpsPerBlock;

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

__device__ __forceinline__ float DecodeBf16(const std::uint16_t* source) {
  const __nv_bfloat16 value = __ushort_as_bfloat16(*source);
  return static_cast<float>(value);
}

__global__ void Sm120DirectProjectionKernel(const std::uint8_t* activation,
                                            const float* activation_scale,
                                            const std::uint8_t* weight,
                                            const std::uint16_t* weight_scales,
                                            float* output,
                                            std::uint64_t tokens,
                                            std::uint64_t rows,
                                            std::uint64_t contracting_elements) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
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
  (void)weight;
  (void)weight_scales;
  (void)output;
  (void)tokens;
  (void)rows;
  (void)contracting_elements;
#endif
}

// Each warp forms a real 16x8 output tile. The checkpoint weight fragment is
// loaded once and reused for 16 prompt rows instead of launching 16 GEMVs.
__global__ void Sm120MatrixProjectionKernel(
    const std::uint8_t* activation, const float* activation_scale,
    const std::uint8_t* weight, const std::uint16_t* weight_scales,
    float* output, std::uint64_t tokens, std::uint64_t rows,
    std::uint64_t contracting_elements) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  const unsigned warp = threadIdx.x / kWarpSize;
  const unsigned lane = threadIdx.x & (kWarpSize - 1U);
  const unsigned group = lane >> 2U;
  const unsigned thread_in_group = lane & 3U;
  const std::uint64_t global_warp =
      static_cast<std::uint64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  const std::uint64_t row_tiles = (rows + kRowsPerWarp - 1U) / kRowsPerWarp;
  if (global_warp >= row_tiles) return;

  const std::uint64_t token_base =
      static_cast<std::uint64_t>(blockIdx.y) * kTokensPerMma;
  const std::uint64_t token_low = token_base + group;
  const std::uint64_t token_high = token_low + 8U;
  const std::uint64_t weight_column = global_warp * kRowsPerWarp + group;
  const std::uint64_t k_blocks = contracting_elements / kElementsPerKBlock;

  float d0 = 0.0F;
  float d1 = 0.0F;
  float d2 = 0.0F;
  float d3 = 0.0F;
  for (std::uint64_t k_block = 0; k_block < k_blocks; ++k_block) {
    const std::uint64_t k_offset =
        k_block * kElementsPerKBlock +
        static_cast<std::uint64_t>(thread_in_group) * 4U;
    const std::uint32_t a0 = token_low < tokens
                                 ? LoadU32(activation + token_low * contracting_elements +
                                           k_offset)
                                 : 0U;
    const std::uint32_t a1 = token_high < tokens
                                 ? LoadU32(activation + token_high * contracting_elements +
                                           k_offset)
                                 : 0U;
    const std::uint32_t a2 = token_low < tokens
                                 ? LoadU32(activation + token_low * contracting_elements +
                                           k_offset + 16U)
                                 : 0U;
    const std::uint32_t a3 = token_high < tokens
                                 ? LoadU32(activation + token_high * contracting_elements +
                                           k_offset + 16U)
                                 : 0U;

    std::uint32_t b0 = 0U;
    std::uint32_t b1 = 0U;
    if (weight_column < rows) {
      const std::uint64_t weight_offset =
          weight_column * contracting_elements + k_offset;
      b0 = LoadU32(weight + weight_offset);
      b1 = LoadU32(weight + weight_offset + 16U);
    }
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(d0), "f"(d1), "f"(d2), "f"(d3));
  }

  const std::uint64_t output_column =
      global_warp * kRowsPerWarp + thread_in_group * 2U;
  if (token_low < tokens) {
    const float input_scale = activation_scale[token_low];
    if (output_column < rows) {
      output[token_low * rows + output_column] =
          d0 * input_scale * DecodeBf16(weight_scales + output_column);
    }
    if (output_column + 1U < rows) {
      output[token_low * rows + output_column + 1U] =
          d1 * input_scale * DecodeBf16(weight_scales + output_column + 1U);
    }
  }
  if (token_high < tokens) {
    const float input_scale = activation_scale[token_high];
    if (output_column < rows) {
      output[token_high * rows + output_column] =
          d2 * input_scale * DecodeBf16(weight_scales + output_column);
    }
    if (output_column + 1U < rows) {
      output[token_high * rows + output_column + 1U] =
          d3 * input_scale * DecodeBf16(weight_scales + output_column + 1U);
    }
  }
#else
  (void)activation;
  (void)activation_scale;
  (void)weight;
  (void)weight_scales;
  (void)output;
  (void)tokens;
  (void)rows;
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
  const std::uint64_t row_tiles = (rows + kRowsPerWarp - 1U) / kRowsPerWarp;
  const std::uint64_t blocks = (row_tiles + kWarpsPerBlock - 1U) / kWarpsPerBlock;
  if (blocks > static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max())) {
    return Invalid("SM120 FP8 direct projection grid exceeds CUDA limits");
  }
  Sm120DirectProjectionKernel<<<dim3(static_cast<unsigned>(blocks), 1U),
                                kThreadsPerBlock, 0, stream>>>(
      activation_e4m3fn, activation_scale, weight_e4m3fn, weight_scales_bf16, output, 1U,
      rows, contracting_elements);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch direct-source SM120 FP8 projection", error);
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
  const std::uint64_t blocks = (row_tiles + kWarpsPerBlock - 1U) / kWarpsPerBlock;
  if (blocks > static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max())) {
    return Invalid("batched SM120 FP8 projection grid exceeds CUDA limits");
  }
  const std::uint64_t token_tiles =
      (tokens + kTokensPerMma - 1U) / kTokensPerMma;
  Sm120MatrixProjectionKernel<<<dim3(static_cast<unsigned>(blocks),
                                     static_cast<unsigned>(token_tiles)),
                                kThreadsPerBlock, 0, stream>>>(
      activation_e4m3fn, activation_scales, weight_e4m3fn,
      weight_scales_bf16, output, tokens, rows, contracting_elements);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch batched direct-source SM120 FP8 projection", error);
}

}  // namespace gem16gb::internal
