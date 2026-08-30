#include "cuda/moe/reference.h"

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

#include "cuda/layer/reference.h"
#include "cuda/moe/prefill.h"
#include "cuda/trellis35/reference.h"
#include "cuda/nvfp4/reference.h"
#include "cuda/nvfp4/sm120.h"

namespace gem16::internal {
namespace {

constexpr unsigned kThreads = 128;
constexpr unsigned kNormThreads = 256U;
constexpr unsigned kFusedPostNormThreads = 2U * kNormThreads;
constexpr unsigned kWarpSize = 32U;
constexpr unsigned kRouterWarpsPerBlock = 4U;
constexpr std::uint64_t kNvfp4Block = 16;
constexpr std::uint64_t kSm120KBlock = 64;
constexpr std::uint64_t kRowsPerTile = 8;
constexpr float kSqrtTwoOverPi = 0.7978845608028654F;
constexpr float kGeluCubic = 0.044715F;

class NvtxRange {
 public:
  explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
  NvtxRange(const NvtxRange&) = delete;
  NvtxRange& operator=(const NvtxRange&) = delete;
  ~NvtxRange() { nvtxRangePop(); }
};

Status Invalid(std::string message) {
  return Status(StatusCode::kInvalidArgument, std::move(message));
}

Status CudaFailure(const char* operation, cudaError_t error) {
  return Status(StatusCode::kInternal,
                std::string(operation) + ": " + cudaGetErrorName(error) +
                    ": " + cudaGetErrorString(error));
}

bool PositiveFinite(float value) { return std::isfinite(value) && value > 0.0F; }

bool MatrixValid(const Gemma4MoeNvfp4Matrix& matrix,
                 std::uint64_t rows, std::uint64_t columns) {
  return matrix.packed_e2m1 != nullptr && matrix.scales_e4m3fn != nullptr &&
         matrix.rows == rows && matrix.columns == columns &&
         columns != 0U && columns % kSm120KBlock == 0U && rows != 0U &&
         PositiveFinite(matrix.activation_global_divisor) &&
         PositiveFinite(matrix.weight_global_divisor) &&
         PositiveFinite(matrix.activation_global_divisor *
                        matrix.weight_global_divisor);
}

__device__ __forceinline__ float Bf16(std::uint16_t value) {
  return static_cast<float>(__ushort_as_bfloat16(value));
}

__device__ __forceinline__ float RoundBf16(float value) {
  return static_cast<float>(__float2bfloat16_rn(value));
}

__device__ __forceinline__ float ReciprocalApproximateFtz(float value) {
  float result;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
  return result;
}

__device__ __forceinline__ std::uint8_t Nibble(const std::uint8_t* bytes,
                                                std::uint64_t index) {
  const std::uint8_t byte = bytes[index / 2U];
  return static_cast<std::uint8_t>(
      (byte >> ((index & 1U) == 0U ? 0U : 4U)) & 0x0FU);
}

__device__ __forceinline__ std::uint64_t TiledPackedOffset(
    std::uint64_t row, std::uint64_t column, std::uint64_t k_blocks) {
  return (((row / kRowsPerTile) * k_blocks + column / kSm120KBlock) *
              kRowsPerTile +
          row % kRowsPerTile) *
             (kSm120KBlock / 2U) +
         (column % kSm120KBlock) / 2U;
}

__device__ __forceinline__ std::uint64_t TiledScaleOffset(
    std::uint64_t row, std::uint64_t column, std::uint64_t k_blocks) {
  return (((row / kRowsPerTile) * k_blocks + column / kSm120KBlock) *
              kRowsPerTile +
          row % kRowsPerTile) *
             (kSm120KBlock / kNvfp4Block) +
         (column % kSm120KBlock) / kNvfp4Block;
}

__global__ void TiledProjectionKernel(
    const std::uint8_t* activation, const std::uint8_t* activation_scales,
    const std::uint8_t* weights, const std::uint8_t* weight_scales,
    float* output, std::uint64_t local_rows, std::uint64_t columns,
    std::uint64_t row_base, float output_divisor) {
  const std::uint64_t local_row =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (local_row >= local_rows) return;
  const std::uint64_t row = row_base + local_row;
  const std::uint64_t k_blocks = columns / kSm120KBlock;
  float accumulator = 0.0F;
  for (std::uint64_t column = 0; column < columns; ++column) {
    __nv_fp4_e2m1 left;
    left.__x = Nibble(activation, column);
    __nv_fp4_e2m1 right;
    const std::uint64_t packed_offset =
        TiledPackedOffset(row, column, k_blocks);
    const std::uint8_t packed_byte = weights[packed_offset];
    right.__x = static_cast<std::uint8_t>(
        (packed_byte >> ((column & 1U) == 0U ? 0U : 4U)) & 0x0FU);
    __nv_fp8_e4m3 left_scale;
    left_scale.__x = activation_scales[column / kNvfp4Block];
    __nv_fp8_e4m3 right_scale;
    right_scale.__x =
        weight_scales[TiledScaleOffset(row, column, k_blocks)];
    accumulator = fmaf(static_cast<float>(left) *
                           static_cast<float>(left_scale),
                       static_cast<float>(right) *
                           static_cast<float>(right_scale),
                       accumulator);
  }
  output[local_row] = RoundBf16(accumulator / output_divisor);
}

__global__ void SelectedTiledProjectionKernel(
    const std::uint8_t* activation, const std::uint8_t* activation_scales,
    const std::uint8_t* weights, const std::uint8_t* weight_scales,
    const std::uint32_t* selected_ids, std::uint32_t slot, float* output,
    std::uint64_t rows_per_expert, std::uint64_t columns,
    std::uint32_t experts, float output_divisor) {
  const std::uint64_t local_row =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (local_row >= rows_per_expert) return;
  const std::uint32_t expert = selected_ids[slot];
  if (expert >= experts) return;
  const std::uint64_t row =
      static_cast<std::uint64_t>(expert) * rows_per_expert + local_row;
  const std::uint64_t k_blocks = columns / kSm120KBlock;
  float accumulator = 0.0F;
  for (std::uint64_t column = 0; column < columns; ++column) {
    __nv_fp4_e2m1 left;
    left.__x = Nibble(activation, column);
    __nv_fp4_e2m1 right;
    const std::uint64_t packed_offset =
        TiledPackedOffset(row, column, k_blocks);
    const std::uint8_t packed_byte = weights[packed_offset];
    right.__x = static_cast<std::uint8_t>(
        (packed_byte >> ((column & 1U) == 0U ? 0U : 4U)) & 0x0FU);
    __nv_fp8_e4m3 left_scale;
    left_scale.__x = activation_scales[column / kNvfp4Block];
    __nv_fp8_e4m3 right_scale;
    right_scale.__x =
        weight_scales[TiledScaleOffset(row, column, k_blocks)];
    accumulator = fmaf(static_cast<float>(left) *
                           static_cast<float>(left_scale),
                       static_cast<float>(right) *
                           static_cast<float>(right_scale),
                       accumulator);
  }
  output[static_cast<std::uint64_t>(slot) * rows_per_expert + local_row] =
      RoundBf16(accumulator / output_divisor);
}

__device__ float MoeBlockSum(float value) {
  __shared__ float scratch[kNormThreads];
  scratch[threadIdx.x] = value;
  __syncthreads();
  // Preserve the accepted binary-tree addition order while avoiding four
  // block-wide barriers once the live reduction fits in warp zero.
  for (unsigned stride = kNormThreads / 2U; stride >= kWarpSize;
       stride >>= 1U) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x < kWarpSize) {
    for (unsigned stride = kWarpSize / 2U; stride != 0U; stride >>= 1U) {
      if (threadIdx.x < stride) {
        scratch[threadIdx.x] += scratch[threadIdx.x + stride];
      }
      __syncwarp();
    }
  }
  __syncthreads();
  return scratch[0];
}

__device__ float MoeDualGroupSum(float value, unsigned group,
                                 unsigned local_thread) {
  __shared__ float scratch[2U][kNormThreads];
  scratch[group][local_thread] = value;
  __syncthreads();
  // Each 256-thread group reproduces MoeBlockSum's accepted binary tree.
  // Both groups reach every block barrier together.
  for (unsigned stride = kNormThreads / 2U; stride >= kWarpSize;
       stride >>= 1U) {
    if (local_thread < stride) {
      scratch[group][local_thread] +=
          scratch[group][local_thread + stride];
    }
    __syncthreads();
  }
  if (local_thread < kWarpSize) {
    for (unsigned stride = kWarpSize / 2U; stride != 0U; stride >>= 1U) {
      if (local_thread < stride) {
        scratch[group][local_thread] +=
            scratch[group][local_thread + stride];
      }
      __syncwarp();
    }
  }
  __syncthreads();
  return scratch[group][0];
}

__global__ void MoeInputNormsRouterTransformKernel(
    const float* hidden, const std::uint16_t* pre_shared_norm,
    const std::uint16_t* pre_expert_norm,
    const std::uint16_t* router_scale, float* shared_input,
    float* expert_input, float* router_normalized,
    float* router_transformed, std::uint64_t width, float epsilon) {
  float squared_sum = 0.0F;
  for (std::uint64_t index = threadIdx.x; index < width;
       index += blockDim.x) {
    const float value = hidden[index];
    squared_sum = fmaf(value, value, squared_sum);
  }
  const float inverse_rms = rsqrtf(
      MoeBlockSum(squared_sum) / static_cast<float>(width) + epsilon);
  const float router_divisor = rsqrtf(static_cast<float>(width));
  for (std::uint64_t index = threadIdx.x; index < width;
       index += blockDim.x) {
    const float normalized = RoundBf16(hidden[index] * inverse_rms);
    shared_input[index] = RoundBf16(
        hidden[index] * inverse_rms * Bf16(pre_shared_norm[index]));
    expert_input[index] = RoundBf16(
        hidden[index] * inverse_rms * Bf16(pre_expert_norm[index]));
    router_normalized[index] = normalized;
    router_transformed[index] = RoundBf16(
        normalized * Bf16(router_scale[index]) * router_divisor);
  }
}

template <bool kMaterializeRouterNormalized>
__global__ void MoeInputNormsRouterTransformNvfp4Kernel(
    const float* hidden, const std::uint16_t* pre_shared_norm,
    const std::uint16_t* pre_expert_norm,
    const std::uint16_t* router_scale, std::uint8_t* shared_packed,
    std::uint8_t* shared_scales, std::uint8_t* expert_packed,
    std::uint8_t* expert_scales, float* router_normalized,
    float* router_transformed, std::uint64_t width, float epsilon,
    float shared_global_divisor, float expert_global_divisor) {
  const std::uint64_t row = blockIdx.x;
  hidden += row * width;
  shared_packed += row * width / 2U;
  shared_scales += row * width / kNvfp4Block;
  expert_packed += row * width / 2U;
  expert_scales += row * width / kNvfp4Block;
  router_normalized += row * width;
  router_transformed += row * width;
  float squared_sum = 0.0F;
  for (std::uint64_t index = threadIdx.x; index < width;
       index += blockDim.x) {
    const float value = hidden[index];
    squared_sum = fmaf(value, value, squared_sum);
  }
  const float inverse_rms = rsqrtf(
      MoeBlockSum(squared_sum) / static_cast<float>(width) + epsilon);
  const float router_divisor = rsqrtf(static_cast<float>(width));
  const std::uint64_t blocks = width / kNvfp4Block;
  for (std::uint64_t block = threadIdx.x; block < blocks;
       block += blockDim.x) {
    const std::uint64_t begin = block * kNvfp4Block;
    float shared_values[kNvfp4Block];
    float expert_values[kNvfp4Block];
    float shared_amax = 0.0F;
    float expert_amax = 0.0F;
#pragma unroll
    for (std::uint64_t local = 0U; local < kNvfp4Block; ++local) {
      const std::uint64_t index = begin + local;
      const float normalized = RoundBf16(hidden[index] * inverse_rms);
      const float shared_value = RoundBf16(
          hidden[index] * inverse_rms * Bf16(pre_shared_norm[index]));
      const float expert_value = RoundBf16(
          hidden[index] * inverse_rms * Bf16(pre_expert_norm[index]));
      shared_values[local] = shared_value;
      expert_values[local] = expert_value;
      shared_amax = fmaxf(shared_amax, fabsf(shared_value));
      expert_amax = fmaxf(expert_amax, fabsf(expert_value));
      if constexpr (kMaterializeRouterNormalized) {
        router_normalized[index] = normalized;
      }
      router_transformed[index] = RoundBf16(
          normalized * Bf16(router_scale[index]) * router_divisor);
    }
    const __nv_fp8_e4m3 shared_scale(
        (shared_amax * ReciprocalApproximateFtz(6.0F)) *
        shared_global_divisor);
    const __nv_fp8_e4m3 expert_scale(
        (expert_amax * ReciprocalApproximateFtz(6.0F)) *
        expert_global_divisor);
    shared_scales[block] = shared_scale.__x;
    expert_scales[block] = expert_scale.__x;
    const float decoded_shared_scale = static_cast<float>(shared_scale);
    const float decoded_expert_scale = static_cast<float>(expert_scale);
    const float shared_output_scale =
        decoded_shared_scale == 0.0F
            ? 0.0F
            : ReciprocalApproximateFtz(
                  decoded_shared_scale *
                  ReciprocalApproximateFtz(shared_global_divisor));
    const float expert_output_scale =
        decoded_expert_scale == 0.0F
            ? 0.0F
            : ReciprocalApproximateFtz(
                  decoded_expert_scale *
                  ReciprocalApproximateFtz(expert_global_divisor));
#pragma unroll
    for (std::uint64_t pair = 0U; pair < kNvfp4Block / 2U; ++pair) {
      const std::uint64_t low = pair * 2U;
      const __nv_fp4_e2m1 shared_low(
          shared_values[low] * shared_output_scale);
      const __nv_fp4_e2m1 shared_high(
          shared_values[low + 1U] * shared_output_scale);
      const __nv_fp4_e2m1 expert_low(
          expert_values[low] * expert_output_scale);
      const __nv_fp4_e2m1 expert_high(
          expert_values[low + 1U] * expert_output_scale);
      shared_packed[begin / 2U + pair] = static_cast<std::uint8_t>(
          (shared_low.__x & 0x0FU) | ((shared_high.__x & 0x0FU) << 4U));
      expert_packed[begin / 2U + pair] = static_cast<std::uint8_t>(
          (expert_low.__x & 0x0FU) | ((expert_high.__x & 0x0FU) << 4U));
    }
  }
}

// Trellis35 keeps the shared MLP in NVFP4 but consumes the routed input at an
// exact physical BF16 boundary. This sibling preserves the reduction and
// rounding order of MoeInputNormsRouterTransformNvfp4Kernel while omitting the
// routed FP4 bytes/scales that have no Trellis consumer.
template <bool kMaterializeRouterNormalized>
__global__ void MoeInputNormsRouterTransformTrellis35Kernel(
    const float* hidden, const std::uint16_t* pre_shared_norm,
    const std::uint16_t* pre_expert_norm,
    const std::uint16_t* router_scale, std::uint8_t* shared_packed,
    std::uint8_t* shared_scales, float* expert_input,
    float* router_normalized, float* router_transformed, std::uint64_t width,
    float epsilon, float shared_global_divisor) {
  const std::uint64_t row = blockIdx.x;
  hidden += row * width;
  shared_packed += row * width / 2U;
  shared_scales += row * width / kNvfp4Block;
  expert_input += row * width;
  router_normalized += row * width;
  router_transformed += row * width;
  float squared_sum = 0.0F;
  for (std::uint64_t index = threadIdx.x; index < width;
       index += blockDim.x) {
    const float value = hidden[index];
    squared_sum = fmaf(value, value, squared_sum);
  }
  const float inverse_rms = rsqrtf(
      MoeBlockSum(squared_sum) / static_cast<float>(width) + epsilon);
  const float router_divisor = rsqrtf(static_cast<float>(width));
  const std::uint64_t blocks = width / kNvfp4Block;
  for (std::uint64_t block = threadIdx.x; block < blocks;
       block += blockDim.x) {
    const std::uint64_t begin = block * kNvfp4Block;
    float shared_values[kNvfp4Block];
    float shared_amax = 0.0F;
#pragma unroll
    for (std::uint64_t local = 0U; local < kNvfp4Block; ++local) {
      const std::uint64_t index = begin + local;
      const float normalized = RoundBf16(hidden[index] * inverse_rms);
      const float shared_value = RoundBf16(
          hidden[index] * inverse_rms * Bf16(pre_shared_norm[index]));
      const float expert_value = RoundBf16(
          hidden[index] * inverse_rms * Bf16(pre_expert_norm[index]));
      shared_values[local] = shared_value;
      shared_amax = fmaxf(shared_amax, fabsf(shared_value));
      expert_input[index] = expert_value;
      if constexpr (kMaterializeRouterNormalized) {
        router_normalized[index] = normalized;
      }
      router_transformed[index] = RoundBf16(
          normalized * Bf16(router_scale[index]) * router_divisor);
    }
    const __nv_fp8_e4m3 shared_scale(
        (shared_amax * ReciprocalApproximateFtz(6.0F)) *
        shared_global_divisor);
    shared_scales[block] = shared_scale.__x;
    const float decoded_shared_scale = static_cast<float>(shared_scale);
    const float shared_output_scale =
        decoded_shared_scale == 0.0F
            ? 0.0F
            : ReciprocalApproximateFtz(
                  decoded_shared_scale *
                  ReciprocalApproximateFtz(shared_global_divisor));
#pragma unroll
    for (std::uint64_t pair = 0U; pair < kNvfp4Block / 2U; ++pair) {
      const std::uint64_t low = pair * 2U;
      const __nv_fp4_e2m1 shared_low(
          shared_values[low] * shared_output_scale);
      const __nv_fp4_e2m1 shared_high(
          shared_values[low + 1U] * shared_output_scale);
      shared_packed[begin / 2U + pair] = static_cast<std::uint8_t>(
          (shared_low.__x & 0x0FU) | ((shared_high.__x & 0x0FU) << 4U));
    }
  }
}

__global__ void RouterProjectionWarpKernel(const float* input,
                                           const std::uint16_t* weights,
                                           float* logits,
                                           std::uint32_t experts,
                                           std::uint64_t width) {
  const std::uint64_t token = blockIdx.y;
  input += token * width;
  logits += token * experts;
  const unsigned warp = threadIdx.x / kWarpSize;
  const unsigned lane = threadIdx.x & (kWarpSize - 1U);
  const std::uint32_t expert =
      blockIdx.x * kRouterWarpsPerBlock + warp;
  if (expert >= experts) return;
  float accumulator = 0.0F;
  const std::uint64_t base = static_cast<std::uint64_t>(expert) * width;
  // Width is K64-aligned. Each lane consumes adjacent BF16/FP32 pairs in a
  // fixed stride, then the repository-standard warp butterfly combines the
  // 32 partials deterministically. The sole BF16 boundary remains the final
  // logit write.
  for (std::uint64_t index = static_cast<std::uint64_t>(lane) * 2U;
       index < width; index += kWarpSize * 2U) {
    const std::uint32_t weight_pair =
        *reinterpret_cast<const std::uint32_t*>(weights + base + index);
    const float2 input_pair =
        *reinterpret_cast<const float2*>(input + index);
    accumulator =
        fmaf(Bf16(static_cast<std::uint16_t>(weight_pair)), input_pair.x,
             accumulator);
    accumulator = fmaf(
        Bf16(static_cast<std::uint16_t>(weight_pair >> 16U)), input_pair.y,
        accumulator);
  }
#pragma unroll
  for (unsigned offset = kWarpSize / 2U; offset != 0U; offset >>= 1U) {
    accumulator += __shfl_down_sync(0xffffffffU, accumulator, offset);
  }
  if (lane == 0U) logits[expert] = RoundBf16(accumulator);
}

template <bool kParallelExact, bool kMaterializeProbabilities = true>
__global__ void RouterTopKKernel(
    const float* logits, const std::uint16_t* per_expert_scale,
    float* probabilities, std::uint32_t* top_ids, float* top_weights,
    std::uint32_t experts, std::uint32_t top_k, int* routing_finite) {
  const std::uint64_t row = blockIdx.x;
  logits += row * experts;
  probabilities += row * experts;
  top_ids += row * top_k;
  top_weights += row * top_k;
  __shared__ float maximum;
  __shared__ float total;
  __shared__ int valid;
  __shared__ float warp_probability[
      (kThreads / kWarpSize) * 8U];
  __shared__ std::uint32_t warp_id[
      (kThreads / kWarpSize) * 8U];
  __shared__ float local_probability[kThreads];
  const unsigned lane = threadIdx.x & (kWarpSize - 1U);
  const unsigned warp = threadIdx.x / kWarpSize;
  if (threadIdx.x == 0U) valid = 1;
  __syncthreads();
  float local_maximum = -3.402823466e+38F;
  for (std::uint32_t expert = threadIdx.x; expert < experts;
       expert += blockDim.x) {
    const float logit = logits[expert];
    if (!isfinite(logit)) atomicExch(&valid, 0);
    local_maximum = fmaxf(local_maximum, logit);
  }
#pragma unroll
  for (unsigned offset = kWarpSize / 2U; offset != 0U; offset >>= 1U) {
    local_maximum = fmaxf(
        local_maximum,
        __shfl_down_sync(0xffffffffU, local_maximum, offset));
  }
  if (lane == 0U) warp_probability[warp] = local_maximum;
  __syncthreads();
  if (warp == 0U) {
    local_maximum = lane < kThreads / kWarpSize
                        ? warp_probability[lane]
                        : -3.402823466e+38F;
#pragma unroll
    for (unsigned offset = kWarpSize / 2U; offset != 0U; offset >>= 1U) {
      local_maximum = fmaxf(
          local_maximum,
          __shfl_down_sync(0xffffffffU, local_maximum, offset));
    }
    if (lane == 0U) maximum = local_maximum;
  }
  __syncthreads();
  if (!valid) {
    if constexpr (kMaterializeProbabilities) {
      for (std::uint32_t expert = threadIdx.x; expert < experts;
           expert += blockDim.x) {
        probabilities[expert] = 0.0F;
      }
    }
    for (std::uint32_t slot = threadIdx.x; slot < top_k;
         slot += blockDim.x) {
      top_ids[slot] = 0U;
      top_weights[slot] = 0.0F;
    }
    if (threadIdx.x == 0U && routing_finite != nullptr) {
      atomicExch(routing_finite, 0);
    }
    return;
  }
  for (std::uint32_t expert = threadIdx.x; expert < experts;
       expert += blockDim.x) {
    const float probability = expf(logits[expert] - maximum);
    if constexpr (kMaterializeProbabilities) {
      probabilities[expert] = probability;
    } else {
      local_probability[expert] = probability;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0U) {
    total = 0.0F;
    for (std::uint32_t expert = 0; expert < experts; ++expert) {
      if constexpr (kMaterializeProbabilities) {
        total += probabilities[expert];
      } else {
        total += local_probability[expert];
      }
    }
    if (!isfinite(total) || total <= 0.0F) valid = 0;
  }
  __syncthreads();
  if (!valid) {
    if constexpr (kMaterializeProbabilities) {
      for (std::uint32_t expert = threadIdx.x; expert < experts;
           expert += blockDim.x) {
        probabilities[expert] = 0.0F;
      }
    }
    for (std::uint32_t slot = threadIdx.x; slot < top_k;
         slot += blockDim.x) {
      top_ids[slot] = 0U;
      top_weights[slot] = 0.0F;
    }
    if (threadIdx.x == 0U && routing_finite != nullptr) {
      atomicExch(routing_finite, 0);
    }
    return;
  }
  for (std::uint32_t expert = threadIdx.x; expert < experts;
       expert += blockDim.x) {
    if constexpr (kMaterializeProbabilities) {
      probabilities[expert] /= total;
    } else {
      local_probability[expert] /= total;
    }
  }
  __syncthreads();
  if (kParallelExact && experts == 128U && top_k == 8U &&
      blockDim.x == kThreads) {
    float candidate_probability;
    if constexpr (kMaterializeProbabilities) {
      candidate_probability = probabilities[threadIdx.x];
    } else {
      candidate_probability = local_probability[threadIdx.x];
    }
    std::uint32_t candidate_id = threadIdx.x;
#pragma unroll
    for (unsigned size = 2U; size <= kWarpSize; size <<= 1U) {
#pragma unroll
      for (unsigned stride = size >> 1U; stride != 0U; stride >>= 1U) {
        const float other_probability = __shfl_xor_sync(
            0xFFFFFFFFU, candidate_probability, stride);
        const std::uint32_t other_id =
            __shfl_xor_sync(0xFFFFFFFFU, candidate_id, stride);
        const bool other_better =
            other_probability > candidate_probability ||
            (other_probability == candidate_probability &&
             other_id < candidate_id);
        const float better_probability =
            other_better ? other_probability : candidate_probability;
        const std::uint32_t better_id =
            other_better ? other_id : candidate_id;
        const float worse_probability =
            other_better ? candidate_probability : other_probability;
        const std::uint32_t worse_id =
            other_better ? candidate_id : other_id;
        const bool descending_segment = (lane & size) == 0U;
        const bool lower_lane = (lane & stride) == 0U;
        const bool take_better = descending_segment == lower_lane;
        candidate_probability =
            take_better ? better_probability : worse_probability;
        candidate_id = take_better ? better_id : worse_id;
      }
    }
    if (lane < 8U) {
      const unsigned index = warp * 8U + lane;
      warp_probability[index] = candidate_probability;
      warp_id[index] = candidate_id;
    }
    __syncthreads();
    if (warp == 0U) {
      candidate_probability = warp_probability[lane];
      candidate_id = warp_id[lane];
#pragma unroll
      for (unsigned size = 2U; size <= kWarpSize; size <<= 1U) {
#pragma unroll
        for (unsigned stride = size >> 1U; stride != 0U; stride >>= 1U) {
          const float other_probability = __shfl_xor_sync(
              0xFFFFFFFFU, candidate_probability, stride);
          const std::uint32_t other_id =
              __shfl_xor_sync(0xFFFFFFFFU, candidate_id, stride);
          const bool other_better =
              other_probability > candidate_probability ||
              (other_probability == candidate_probability &&
               other_id < candidate_id);
          const float better_probability =
              other_better ? other_probability : candidate_probability;
          const std::uint32_t better_id =
              other_better ? other_id : candidate_id;
          const float worse_probability =
              other_better ? candidate_probability : other_probability;
          const std::uint32_t worse_id =
              other_better ? candidate_id : other_id;
          const bool descending_segment = (lane & size) == 0U;
          const bool lower_lane = (lane & stride) == 0U;
          const bool take_better = descending_segment == lower_lane;
          candidate_probability =
              take_better ? better_probability : worse_probability;
          candidate_id = take_better ? better_id : worse_id;
        }
      }
      if (lane < 8U) {
        top_ids[lane] = candidate_id;
        top_weights[lane] = candidate_probability;
      }
    }
    __syncthreads();
  } else {
    for (std::uint32_t slot = 0; slot < top_k; ++slot) {
      std::uint32_t best_id = experts;
      float best_probability = -1.0F;
      for (std::uint32_t expert = threadIdx.x; expert < experts;
           expert += blockDim.x) {
        bool already_selected = false;
        for (std::uint32_t previous = 0; previous < slot; ++previous) {
          already_selected = already_selected || top_ids[previous] == expert;
        }
        const float probability = probabilities[expert];
        if (!already_selected &&
            (probability > best_probability ||
             (probability == best_probability && expert < best_id))) {
          best_probability = probability;
          best_id = expert;
        }
      }
      for (unsigned offset = kWarpSize / 2U; offset != 0U; offset >>= 1U) {
        const float right_probability =
            __shfl_down_sync(0xffffffffU, best_probability, offset);
        const std::uint32_t right_id =
            __shfl_down_sync(0xffffffffU, best_id, offset);
        if (right_probability > best_probability ||
            (right_probability == best_probability && right_id < best_id)) {
          best_probability = right_probability;
          best_id = right_id;
        }
      }
      if (lane == 0U) {
        warp_probability[warp] = best_probability;
        warp_id[warp] = best_id;
      }
      __syncthreads();
      if (warp == 0U) {
        best_probability =
            lane < kThreads / kWarpSize ? warp_probability[lane] : -1.0F;
        best_id = lane < kThreads / kWarpSize ? warp_id[lane] : experts;
        for (unsigned offset = kWarpSize / 2U; offset != 0U;
             offset >>= 1U) {
          const float right_probability =
              __shfl_down_sync(0xffffffffU, best_probability, offset);
          const std::uint32_t right_id =
              __shfl_down_sync(0xffffffffU, best_id, offset);
          if (right_probability > best_probability ||
              (right_probability == best_probability &&
               right_id < best_id)) {
            best_probability = right_probability;
            best_id = right_id;
          }
        }
      }
      if (threadIdx.x == 0U) {
        if (best_id >= experts) {
          valid = 0;
        } else {
          top_ids[slot] = best_id;
          top_weights[slot] = best_probability;
        }
      }
      __syncthreads();
    }
  }
  if (threadIdx.x == 0U) {
    if (valid) {
      float selected_total = 0.0F;
      for (std::uint32_t slot = 0; slot < top_k; ++slot) {
        selected_total += top_weights[slot];
      }
      if (!isfinite(selected_total) || selected_total <= 0.0F) valid = 0;
      if (valid) {
        for (std::uint32_t slot = 0; slot < top_k; ++slot) {
          const float scale = Bf16(per_expert_scale[top_ids[slot]]);
          if (!isfinite(scale)) {
            valid = 0;
            break;
          }
          top_weights[slot] = (top_weights[slot] / selected_total) * scale;
        }
      }
    }
    if (!valid) {
      if constexpr (kMaterializeProbabilities) {
        for (std::uint32_t expert = 0; expert < experts; ++expert) {
          probabilities[expert] = 0.0F;
        }
      }
      for (std::uint32_t slot = 0; slot < top_k; ++slot) {
        top_ids[slot] = 0U;
        top_weights[slot] = 0.0F;
      }
      if (routing_finite != nullptr) atomicExch(routing_finite, 0);
    }
  }
}

__global__ void GatedGeluProductKernel(const float* gate_up, float* product,
                                       std::uint32_t slot,
                                       std::uint64_t intermediate) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= intermediate) return;
  const std::uint64_t gate_up_base =
      static_cast<std::uint64_t>(slot) * 2U * intermediate;
  const float gate = gate_up[gate_up_base + index];
  const float up = gate_up[gate_up_base + intermediate + index];
  const float inner =
      kSqrtTwoOverPi * (gate + kGeluCubic * gate * gate * gate);
  const float gelu = RoundBf16(0.5F * gate * (1.0F + tanhf(inner)));
  product[static_cast<std::uint64_t>(slot) * intermediate + index] =
      RoundBf16(gelu * up);
}

__global__ void GatedGeluSeparateProductKernel(
    const float* gate, const float* up, float* product,
    std::uint64_t intermediate) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= intermediate) return;
  const float gate_value = gate[index];
  const float inner = kSqrtTwoOverPi *
                      (gate_value + kGeluCubic * gate_value * gate_value *
                                        gate_value);
  const float gelu =
      RoundBf16(0.5F * gate_value * (1.0F + tanhf(inner)));
  product[index] = RoundBf16(gelu * up[index]);
}

__global__ void WeightedReductionKernel(
    const float* expert_down, const float* top_weights,
    float* contributions, float* routed_sum, std::uint64_t width,
    std::uint32_t top_k) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= width) return;
  float sum = 0.0F;
  for (std::uint32_t slot = 0; slot < top_k; ++slot) {
    const std::uint64_t offset = static_cast<std::uint64_t>(slot) * width + index;
    const float weighted = RoundBf16(expert_down[offset] * top_weights[slot]);
    // The exact diagnostic path materializes every slot contribution. Native
    // decode consumes only the fixed-order sum and avoids the unused 8xwidth
    // write by passing null without changing any arithmetic boundary.
    if (contributions != nullptr) contributions[offset] = weighted;
    sum += weighted;
  }
  routed_sum[index] = RoundBf16(sum);
}

__global__ void MoeBranchPostNormKernel(
    const float* shared_output, const std::uint16_t* post_shared_norm,
    const float* routed_sum, const std::uint16_t* post_expert_norm,
    float* shared_post, float* routed_post,
    std::uint64_t width, float epsilon) {
  const bool routed = blockIdx.x != 0U;
  const float* input = routed ? routed_sum : shared_output;
  const std::uint16_t* norm =
      routed ? post_expert_norm : post_shared_norm;
  float* post = routed ? routed_post : shared_post;
  float squared_sum = 0.0F;
  for (std::uint64_t index = threadIdx.x; index < width;
       index += blockDim.x) {
    const float value = input[index];
    squared_sum = fmaf(value, value, squared_sum);
  }
  const float inverse_rms = rsqrtf(
      MoeBlockSum(squared_sum) / static_cast<float>(width) + epsilon);
  for (std::uint64_t index = threadIdx.x; index < width;
       index += blockDim.x) {
    post[index] =
        RoundBf16(input[index] * inverse_rms * Bf16(norm[index]));
  }
}

__global__ void MoeCombinedPostNormResidualKernel(
    const float* shared_post, const float* routed_post,
    const std::uint16_t* post_combined_norm, const float* residual,
    const std::uint16_t* layer_scalar, float* combined, float* feed_forward,
    float* output, std::uint64_t width, float epsilon) {
  float squared_sum = 0.0F;
  for (std::uint64_t index = threadIdx.x; index < width;
       index += blockDim.x) {
    const float value = RoundBf16(shared_post[index] + routed_post[index]);
    combined[index] = value;
    squared_sum = fmaf(value, value, squared_sum);
  }
  const float combined_inverse_rms = rsqrtf(
      MoeBlockSum(squared_sum) / static_cast<float>(width) + epsilon);
  const float scalar = Bf16(layer_scalar[0]);
  for (std::uint64_t index = threadIdx.x; index < width;
       index += blockDim.x) {
    const float normalized = RoundBf16(
        combined[index] * combined_inverse_rms *
        Bf16(post_combined_norm[index]));
    feed_forward[index] = normalized;
    const float residual_sum = RoundBf16(normalized + residual[index]);
    output[index] = RoundBf16(residual_sum * scalar);
  }
}

__global__ void MoeFusedPostNormResidualKernel(
    const float* shared_output, const std::uint16_t* post_shared_norm,
    const float* routed_sum, const std::uint16_t* post_expert_norm,
    const std::uint16_t* post_combined_norm, const float* residual,
    const std::uint16_t* layer_scalar, float* output, std::uint64_t width,
    float epsilon) {
  const std::uint64_t row = blockIdx.x;
  shared_output += row * width;
  routed_sum += row * width;
  residual += row * width;
  output += row * width;
  extern __shared__ float post_values[];
  const unsigned group = threadIdx.x / kNormThreads;
  const unsigned local_thread = threadIdx.x % kNormThreads;
  const bool routed = group != 0U;
  const float* branch_input = routed ? routed_sum : shared_output;
  const std::uint16_t* branch_norm =
      routed ? post_expert_norm : post_shared_norm;
  float* branch_post = post_values + static_cast<std::uint64_t>(group) * width;

  float branch_squared_sum = 0.0F;
  for (std::uint64_t index = local_thread; index < width;
       index += kNormThreads) {
    const float value = branch_input[index];
    branch_squared_sum = fmaf(value, value, branch_squared_sum);
  }
  const float branch_inverse_rms =
      rsqrtf(MoeDualGroupSum(branch_squared_sum, group, local_thread) /
                 static_cast<float>(width) +
             epsilon);
  for (std::uint64_t index = local_thread; index < width;
       index += kNormThreads) {
    branch_post[index] = RoundBf16(branch_input[index] * branch_inverse_rms *
                                   Bf16(branch_norm[index]));
  }
  __syncthreads();

  float combined_squared_sum = 0.0F;
  if (group == 0U) {
    for (std::uint64_t index = local_thread; index < width;
         index += kNormThreads) {
      const float value =
          RoundBf16(post_values[index] + post_values[width + index]);
      post_values[2U * width + index] = value;
      combined_squared_sum = fmaf(value, value, combined_squared_sum);
    }
  }
  const float combined_sum =
      MoeDualGroupSum(combined_squared_sum, group, local_thread);
  if (group == 0U) {
    const float combined_inverse_rms =
        rsqrtf(combined_sum / static_cast<float>(width) + epsilon);
    const float scalar = Bf16(layer_scalar[0]);
    for (std::uint64_t index = local_thread; index < width;
         index += kNormThreads) {
      const float normalized = RoundBf16(
          post_values[2U * width + index] * combined_inverse_rms *
          Bf16(post_combined_norm[index]));
      const float residual_sum = RoundBf16(normalized + residual[index]);
      output[index] = RoundBf16(residual_sum * scalar);
    }
  }
}

std::uint64_t Blocks(std::uint64_t elements) {
  return (elements + kThreads - 1U) / kThreads;
}

Status CheckLaunch(const char* name) {
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess ? Status::Ok() : CudaFailure(name, error);
}

Status LaunchTiled(const Gemma4MoeNvfp4Matrix& matrix,
                   const std::uint8_t* activation,
                   const std::uint8_t* activation_scales, float* output,
                   cudaStream_t stream) {
  TiledProjectionKernel<<<static_cast<unsigned>(Blocks(matrix.rows)), kThreads,
                          0, stream>>>(
      activation, activation_scales, matrix.packed_e2m1,
      matrix.scales_e4m3fn, output, matrix.rows, matrix.columns, 0U,
      matrix.activation_global_divisor * matrix.weight_global_divisor);
  return CheckLaunch("launch M11 tiled NVFP4 projection");
}

Status LaunchSelectedTiled(const Gemma4MoeNvfp4Matrix& matrix,
                           const std::uint8_t* activation,
                           const std::uint8_t* activation_scales,
                           const std::uint32_t* ids, std::uint32_t slot,
                           float* output, std::uint64_t rows_per_expert,
                           std::uint32_t experts, cudaStream_t stream) {
  SelectedTiledProjectionKernel<<<
      static_cast<unsigned>(Blocks(rows_per_expert)), kThreads, 0, stream>>>(
      activation, activation_scales, matrix.packed_e2m1,
      matrix.scales_e4m3fn, ids, slot, output, rows_per_expert,
      matrix.columns, experts,
      matrix.activation_global_divisor * matrix.weight_global_divisor);
  return CheckLaunch("launch M11 device-selected NVFP4 projection");
}

}  // namespace

Status LaunchGemma4MoeLayerImpl(
    const float* hidden, float* output,
    const Gemma4MoeReferenceConfig& config,
    const Gemma4MoeReferenceWeights& weights,
    const Gemma4MoeReferenceWorkspace& workspace, bool native_sm120,
    cudaStream_t stream, cudaStream_t shared_branch_stream = nullptr,
    cudaEvent_t fork_event = nullptr, cudaEvent_t join_event = nullptr,
    const Trellis35DeviceLayerBinding* trellis_layer = nullptr,
    const Trellis35M1Workspace* trellis_workspace = nullptr) {
  const auto& c = config;
  const auto& w = weights;
  const auto& x = workspace;
  if (hidden == nullptr || output == nullptr || hidden == output ||
      c.width == 0U ||
      c.width % kSm120KBlock != 0U || c.shared_intermediate == 0U ||
      c.shared_intermediate % kSm120KBlock != 0U ||
      c.expert_intermediate == 0U ||
      c.expert_intermediate % kSm120KBlock != 0U || c.experts == 0U ||
      c.top_k == 0U || c.top_k > c.experts || !PositiveFinite(c.epsilon)) {
    return Invalid("M11 MoE reference geometry is invalid");
  }
  const bool trellis35 = trellis_layer != nullptr || trellis_workspace != nullptr;
  if (trellis35 &&
      (!native_sm120 || trellis_layer == nullptr ||
       trellis_workspace == nullptr)) {
    return Invalid("Trellis35 decode requires complete native bindings");
  }
  if (w.pre_shared_norm_bf16 == nullptr ||
      w.post_shared_norm_bf16 == nullptr ||
      w.pre_expert_norm_bf16 == nullptr ||
      w.post_expert_norm_bf16 == nullptr ||
      w.post_combined_norm_bf16 == nullptr ||
      w.router_scale_bf16 == nullptr ||
      w.router_projection_bf16 == nullptr ||
      w.per_expert_scale_bf16 == nullptr || w.layer_scalar_bf16 == nullptr ||
      !MatrixValid(w.shared_gate, c.shared_intermediate, c.width) ||
      !MatrixValid(w.shared_up, c.shared_intermediate, c.width) ||
      !MatrixValid(w.shared_down, c.width, c.shared_intermediate) ||
      (!trellis35 &&
       (!MatrixValid(w.expert_gate_up,
                     static_cast<std::uint64_t>(c.experts) * 2U *
                         c.expert_intermediate,
                     c.width) ||
        !MatrixValid(w.expert_down,
                     static_cast<std::uint64_t>(c.experts) * c.width,
                     c.expert_intermediate)))) {
    return Invalid("M11 MoE reference weight contract is invalid");
  }
  const bool workspace_valid =
      x.shared_input != nullptr && x.shared_input_packed != nullptr &&
      x.shared_input_scales != nullptr && x.shared_gate != nullptr &&
      x.shared_up != nullptr && x.shared_product != nullptr &&
      x.shared_product_packed != nullptr &&
      x.shared_product_scales != nullptr && x.shared_output != nullptr &&
      x.shared_post != nullptr && x.router_normalized != nullptr &&
      x.router_transformed != nullptr && x.router_logits != nullptr &&
      x.router_probabilities != nullptr && x.top_ids != nullptr &&
      x.top_weights != nullptr && x.expert_input != nullptr &&
      x.expert_input_packed != nullptr && x.expert_input_scales != nullptr &&
      x.expert_gate_up != nullptr && x.expert_product != nullptr &&
      x.expert_product_packed != nullptr &&
      x.expert_product_scales != nullptr && x.expert_down != nullptr &&
      x.expert_contributions != nullptr && x.routed_sum != nullptr &&
      x.routed_post != nullptr && x.combined != nullptr;
  if (!workspace_valid) return Invalid("M11 MoE reference workspace is incomplete");
  if (w.shared_gate.activation_global_divisor !=
          w.shared_up.activation_global_divisor ||
      w.expert_gate_up.activation_global_divisor <= 0.0F) {
    return Invalid("M11 Gate/Up activation divisors are incompatible");
  }
  const bool concurrent_shared = shared_branch_stream != nullptr ||
                                 fork_event != nullptr || join_event != nullptr;
  if (concurrent_shared &&
      (!native_sm120 || shared_branch_stream == nullptr ||
       fork_event == nullptr || join_event == nullptr ||
       shared_branch_stream == stream)) {
    return Invalid("M14 concurrent shared branch contract is invalid");
  }

  Status status;
  if (native_sm120) {
    if (trellis35 && c.materialize_native_router_normalized) {
      MoeInputNormsRouterTransformTrellis35Kernel<true><<<
          1, kNormThreads, 0, stream>>>(
          hidden, w.pre_shared_norm_bf16, w.pre_expert_norm_bf16,
          w.router_scale_bf16, x.shared_input_packed, x.shared_input_scales,
          trellis_workspace->gate_up_output, x.router_normalized,
          x.router_transformed, c.width, c.epsilon,
          w.shared_gate.activation_global_divisor);
    } else if (trellis35) {
      MoeInputNormsRouterTransformTrellis35Kernel<false><<<
          1, kNormThreads, 0, stream>>>(
          hidden, w.pre_shared_norm_bf16, w.pre_expert_norm_bf16,
          w.router_scale_bf16, x.shared_input_packed, x.shared_input_scales,
          trellis_workspace->gate_up_output, x.router_normalized,
          x.router_transformed, c.width, c.epsilon,
          w.shared_gate.activation_global_divisor);
    } else if (c.materialize_native_router_normalized) {
      MoeInputNormsRouterTransformNvfp4Kernel<true><<<
          1, kNormThreads, 0, stream>>>(
          hidden, w.pre_shared_norm_bf16, w.pre_expert_norm_bf16,
          w.router_scale_bf16, x.shared_input_packed, x.shared_input_scales,
          x.expert_input_packed, x.expert_input_scales, x.router_normalized,
          x.router_transformed, c.width, c.epsilon,
          w.shared_gate.activation_global_divisor,
          w.expert_gate_up.activation_global_divisor);
    } else {
      MoeInputNormsRouterTransformNvfp4Kernel<false><<<
          1, kNormThreads, 0, stream>>>(
          hidden, w.pre_shared_norm_bf16, w.pre_expert_norm_bf16,
          w.router_scale_bf16, x.shared_input_packed, x.shared_input_scales,
          x.expert_input_packed, x.expert_input_scales, x.router_normalized,
          x.router_transformed, c.width, c.epsilon,
          w.shared_gate.activation_global_divisor,
          w.expert_gate_up.activation_global_divisor);
    }
    status = CheckLaunch(trellis35
                             ? "launch Trellis35 M1 input boundary"
                             : "launch fused M14 input norms, router transform, and NVFP4 quantization");
    if (!status.ok()) return status;
  } else {
    MoeInputNormsRouterTransformKernel<<<1, kNormThreads, 0, stream>>>(
        hidden, w.pre_shared_norm_bf16, w.pre_expert_norm_bf16,
        w.router_scale_bf16, x.shared_input, x.expert_input,
        x.router_normalized, x.router_transformed, c.width, c.epsilon);
    status =
        CheckLaunch("launch fused M11 input norms and router transform");
    if (!status.ok()) return status;
    status = LaunchNvfp4ReferenceActivationQuantization(
        x.shared_input, x.shared_input_packed, x.shared_input_scales, c.width,
        w.shared_gate.activation_global_divisor, stream);
    if (!status.ok()) return status;
  }
  cudaStream_t shared_stream = stream;
  if (concurrent_shared) {
    cudaError_t error = cudaEventRecord(fork_event, stream);
    if (error == cudaSuccess) {
      error = cudaStreamWaitEvent(shared_branch_stream, fork_event, 0U);
    }
    if (error != cudaSuccess) {
      return CudaFailure("fork M14 shared branch", error);
    }
    shared_stream = shared_branch_stream;
  }
  if (native_sm120) {
    status = LaunchNvfp4Sm120FusedGateUp(
        x.shared_input_packed, x.shared_input_scales,
        w.shared_gate.packed_e2m1, w.shared_gate.scales_e4m3fn,
        w.shared_up.packed_e2m1, w.shared_up.scales_e4m3fn, x.shared_gate,
        x.shared_up, x.shared_product, c.shared_intermediate, c.width,
        w.shared_gate.activation_global_divisor,
        w.shared_gate.weight_global_divisor,
        w.shared_up.activation_global_divisor,
        w.shared_up.weight_global_divisor, shared_stream);
    if (!status.ok()) return status;
  } else {
    status = LaunchTiled(w.shared_gate, x.shared_input_packed,
                         x.shared_input_scales, x.shared_gate, shared_stream);
    if (!status.ok()) return status;
    status = LaunchTiled(w.shared_up, x.shared_input_packed,
                         x.shared_input_scales, x.shared_up, shared_stream);
    if (!status.ok()) return status;
    GatedGeluSeparateProductKernel<<<
        static_cast<unsigned>(Blocks(c.shared_intermediate)), kThreads, 0,
        shared_stream>>>(x.shared_gate, x.shared_up, x.shared_product,
                  c.shared_intermediate);
    status = CheckLaunch("launch M11 shared gated GELU");
    if (!status.ok()) return status;
  }
  status = LaunchNvfp4ReferenceActivationQuantization(
      x.shared_product, x.shared_product_packed, x.shared_product_scales,
      c.shared_intermediate, w.shared_down.activation_global_divisor,
      shared_stream);
  if (!status.ok()) return status;
  status = native_sm120
               ? LaunchNvfp4Sm120DirectProjectionBf16Float(
                     x.shared_product_packed, x.shared_product_scales,
                     w.shared_down.packed_e2m1,
                     w.shared_down.scales_e4m3fn, x.shared_output, c.width,
                     c.shared_intermediate,
                     w.shared_down.activation_global_divisor,
                     w.shared_down.weight_global_divisor, shared_stream)
               : LaunchTiled(w.shared_down, x.shared_product_packed,
                             x.shared_product_scales, x.shared_output,
                             shared_stream);
  if (!status.ok()) return status;
  const unsigned router_blocks =
      (c.experts + kRouterWarpsPerBlock - 1U) / kRouterWarpsPerBlock;
  RouterProjectionWarpKernel<<<router_blocks, kThreads, 0, stream>>>(
      x.router_transformed, w.router_projection_bf16, x.router_logits,
      c.experts, c.width);
  status = CheckLaunch("launch M11 router projection");
  if (!status.ok()) return status;
  RouterTopKKernel<true><<<1, kThreads, 0, stream>>>(
      x.router_logits, w.per_expert_scale_bf16, x.router_probabilities,
      x.top_ids, x.top_weights, c.experts, c.top_k, x.routing_finite);
  status = CheckLaunch("launch M11 deterministic router top-k");
  if (!status.ok()) return status;

  if (!native_sm120) {
    status = LaunchNvfp4ReferenceActivationQuantization(
        x.expert_input, x.expert_input_packed, x.expert_input_scales, c.width,
        w.expert_gate_up.activation_global_divisor, stream);
    if (!status.ok()) return status;
  }
  if (trellis35) {
    status = LaunchTrellis35SelectedExpertsM1(
        trellis_workspace->gate_up_output, x.top_ids, x.top_weights,
        *trellis_layer,
        *trellis_workspace, x.routed_sum, stream);
    if (!status.ok()) return status;
  } else if (native_sm120) {
    status = LaunchNvfp4Sm120SelectedSplitGateUpBatch(
        x.expert_input_packed, x.expert_input_scales,
        w.expert_gate_up.packed_e2m1, w.expert_gate_up.scales_e4m3fn,
        x.top_ids, c.top_k, x.expert_gate_up,
        x.expert_gate_up + c.expert_intermediate, c.expert_intermediate,
        c.width, c.experts,
        w.expert_gate_up.activation_global_divisor,
        w.expert_gate_up.weight_global_divisor, stream);
    if (!status.ok()) return status;
    status = LaunchStridedGatedGeluNvfp4ActivationQuantization(
        x.expert_gate_up, x.expert_gate_up + c.expert_intermediate,
        x.expert_product, x.expert_product_packed,
        x.expert_product_scales, c.top_k, c.expert_intermediate,
        w.expert_down.activation_global_divisor, stream);
    if (!status.ok()) return status;
    status = LaunchNvfp4Sm120SelectedDirectProjectionReduceBf16FloatBatch(
        x.expert_product_packed, x.expert_product_scales,
        w.expert_down.packed_e2m1, w.expert_down.scales_e4m3fn, x.top_ids,
        x.top_weights, c.top_k, x.routed_sum, c.width,
        c.expert_intermediate, c.experts, w.expert_down.activation_global_divisor,
        w.expert_down.weight_global_divisor, stream);
    if (!status.ok()) return status;
  } else {
    for (std::uint32_t slot = 0; slot < c.top_k; ++slot) {
      const std::uint64_t product_offset =
          static_cast<std::uint64_t>(slot) * c.expert_intermediate;
      status = LaunchSelectedTiled(
          w.expert_gate_up, x.expert_input_packed, x.expert_input_scales,
          x.top_ids, slot, x.expert_gate_up, 2U * c.expert_intermediate,
          c.experts, stream);
      if (!status.ok()) return status;
      GatedGeluProductKernel<<<
          static_cast<unsigned>(Blocks(c.expert_intermediate)), kThreads, 0,
          stream>>>(x.expert_gate_up, x.expert_product, slot,
                    c.expert_intermediate);
      status = CheckLaunch("launch M11 selected expert gated GELU");
      if (!status.ok()) return status;
      const std::uint64_t packed_offset = product_offset / 2U;
      const std::uint64_t scale_offset = product_offset / kNvfp4Block;
      status = LaunchNvfp4ReferenceActivationQuantization(
          x.expert_product + product_offset,
          x.expert_product_packed + packed_offset,
          x.expert_product_scales + scale_offset, c.expert_intermediate,
          w.expert_down.activation_global_divisor, stream);
      if (!status.ok()) return status;
      status = LaunchSelectedTiled(
          w.expert_down, x.expert_product_packed + packed_offset,
          x.expert_product_scales + scale_offset, x.top_ids, slot,
          x.expert_down, c.width, c.experts, stream);
      if (!status.ok()) return status;
    }
  }
  if (concurrent_shared) {
    cudaError_t error = cudaEventRecord(join_event, shared_stream);
    if (error == cudaSuccess) {
      error = cudaStreamWaitEvent(stream, join_event, 0U);
    }
    if (error != cudaSuccess) {
      return CudaFailure("join M14 shared branch", error);
    }
  }
  if (!native_sm120) {
    WeightedReductionKernel<<<static_cast<unsigned>(Blocks(c.width)), kThreads,
                              0, stream>>>(
        x.expert_down, x.top_weights, x.expert_contributions, x.routed_sum,
        c.width, c.top_k);
    status = CheckLaunch("launch M11 slot-order expert reduction");
    if (!status.ok()) return status;
  }
  if (native_sm120) {
    const std::size_t post_norm_shared_bytes =
        3U * static_cast<std::size_t>(c.width) * sizeof(float);
    MoeFusedPostNormResidualKernel<<<1, kFusedPostNormThreads,
                                     post_norm_shared_bytes, stream>>>(
        x.shared_output, w.post_shared_norm_bf16, x.routed_sum,
        w.post_expert_norm_bf16, w.post_combined_norm_bf16, hidden,
        w.layer_scalar_bf16, output, c.width, c.epsilon);
    return CheckLaunch("launch fused native MoE post norms and residual");
  }
  MoeBranchPostNormKernel<<<2, kNormThreads, 0, stream>>>(
      x.shared_output, w.post_shared_norm_bf16, x.routed_sum,
      w.post_expert_norm_bf16, x.shared_post, x.routed_post, c.width,
      c.epsilon);
  status = CheckLaunch("launch parallel M11 branch post norms");
  if (!status.ok()) return status;
  MoeCombinedPostNormResidualKernel<<<1, kNormThreads, 0, stream>>>(
      x.shared_post, x.routed_post, w.post_combined_norm_bf16, hidden,
      w.layer_scalar_bf16, x.combined, x.feed_forward, output, c.width,
      c.epsilon);
  return CheckLaunch("launch M11 combined post norm and residual");
}

Status LaunchGemma4MoeReferenceLayer(
    const float* hidden, float* output,
    const Gemma4MoeReferenceConfig& config,
    const Gemma4MoeReferenceWeights& weights,
    const Gemma4MoeReferenceWorkspace& workspace, cudaStream_t stream) {
  return LaunchGemma4MoeLayerImpl(hidden, output, config, weights, workspace,
                                  false, stream);
}

Status LaunchGemma4MoeSm120Layer(
    const float* hidden, float* output,
    const Gemma4MoeReferenceConfig& config,
    const Gemma4MoeReferenceWeights& weights,
    const Gemma4MoeReferenceWorkspace& workspace, cudaStream_t stream,
    cudaStream_t shared_branch_stream, cudaEvent_t fork_event,
    cudaEvent_t join_event) {
  return LaunchGemma4MoeLayerImpl(hidden, output, config, weights, workspace,
                                  true, stream, shared_branch_stream,
                                  fork_event, join_event);
}

Status LaunchGemma4MoeSm120Trellis35Layer(
    const float* hidden, float* output,
    const Gemma4MoeReferenceConfig& config,
    const Gemma4MoeReferenceWeights& weights,
    const Trellis35DeviceLayerBinding& trellis_layer,
    const Trellis35M1Workspace& trellis_workspace,
    const Gemma4MoeReferenceWorkspace& workspace, cudaStream_t stream,
    cudaStream_t shared_branch_stream, cudaEvent_t fork_event,
    cudaEvent_t join_event) {
  return LaunchGemma4MoeLayerImpl(
      hidden, output, config, weights, workspace, true, stream,
      shared_branch_stream, fork_event, join_event, &trellis_layer,
      &trellis_workspace);
}

__global__ void RecordMtpRouterOverlapKernel(
    const std::uint32_t* top_ids, std::uint32_t experts,
    std::uint32_t top_k, MtpRouterOverlapCounters* counters) {
  if (blockIdx.x != 0U || threadIdx.x != 0U || top_ids == nullptr ||
      counters == nullptr || experts > 128U || top_k == 0U || top_k > 64U) {
    return;
  }
  unsigned long long low[3]{};
  unsigned long long high[3]{};
  for (std::uint32_t row = 0U; row < 3U; ++row) {
    for (std::uint32_t slot = 0U; slot < top_k; ++slot) {
      const std::uint32_t expert = top_ids[row * top_k + slot];
      if (expert >= experts) return;
      if (expert < 64U) {
        low[row] |= 1ULL << expert;
      } else {
        high[row] |= 1ULL << (expert - 64U);
      }
    }
  }
  const auto cardinality = [](unsigned long long lo,
                              unsigned long long hi) {
    return static_cast<unsigned long long>(__popcll(lo) + __popcll(hi));
  };
  const unsigned long long union_size = cardinality(
      low[0] | low[1] | low[2], high[0] | high[1] | high[2]);
  const unsigned long long row01 =
      cardinality(low[0] & low[1], high[0] & high[1]);
  const unsigned long long row02 =
      cardinality(low[0] & low[2], high[0] & high[2]);
  const unsigned long long row12 =
      cardinality(low[1] & low[2], high[1] & high[2]);
  const unsigned long long triple = cardinality(
      low[0] & low[1] & low[2], high[0] & high[1] & high[2]);
  ++counters->verifier_layer_samples;
  counters->routed_assignments += 3ULL * top_k;
  counters->unique_experts_sum += union_size;
  counters->row01_intersection_sum += row01;
  counters->row02_intersection_sum += row02;
  counters->row12_intersection_sum += row12;
  counters->triple_intersection_sum += triple;
  if (union_size < 25U) {
    ++counters->union_size_histogram[union_size];
  }
}

Status LaunchGemma4MoeSm120MtpSharedBatchLayerImpl(
    const float* hidden, float* output, std::uint64_t tokens,
    const Gemma4MoeReferenceConfig& c,
    const Gemma4MoeReferenceWeights& w,
    const Gemma4MoePrefillWorkspace& batch,
    const Gemma4MoeReferenceWorkspace& decode, cudaStream_t stream,
    MtpRouterOverlapCounters* router_overlap,
    const Trellis35DeviceLayerBinding* trellis_layer = nullptr,
    const Trellis35T3Workspace* trellis_workspace = nullptr) {
  const bool trellis35 = trellis_layer != nullptr || trellis_workspace != nullptr;
  if (hidden == nullptr || output == nullptr || hidden == output ||
      tokens == 0U || tokens > 5U || c.width == 0U ||
      c.width % kSm120KBlock != 0U || c.shared_intermediate == 0U ||
      c.expert_intermediate == 0U || c.experts == 0U || c.top_k == 0U ||
      batch.token_hidden == nullptr || batch.token_packed == nullptr ||
      batch.token_scales == nullptr || batch.router_logits == nullptr ||
      batch.router_probabilities == nullptr ||
      batch.shared_product == nullptr ||
      batch.shared_product_packed == nullptr ||
      batch.shared_product_scales == nullptr ||
      batch.shared_output == nullptr || batch.reduced_output == nullptr ||
      batch.permutation == nullptr ||
      batch.expert_product_packed == nullptr ||
      batch.expert_product_scales == nullptr ||
      (!trellis35 && batch.expert_down_bf16 == nullptr) ||
      batch.routing_finite == nullptr ||
      (trellis35 && (tokens != kTrellis35T3Rows ||
                     trellis_layer == nullptr ||
                     trellis_workspace == nullptr))) {
    return Invalid("M25 exact shared-batch MoE contract is invalid");
  }
  (void)decode;
  if (trellis35) {
    MoeInputNormsRouterTransformTrellis35Kernel<false><<<
        static_cast<unsigned>(tokens), kNormThreads, 0, stream>>>(
        hidden, w.pre_shared_norm_bf16, w.pre_expert_norm_bf16,
        w.router_scale_bf16, batch.token_packed, batch.token_scales,
        trellis_workspace->gate_up_output, batch.shared_output,
        batch.token_hidden, c.width, c.epsilon,
        w.shared_gate.activation_global_divisor);
  } else {
    MoeInputNormsRouterTransformNvfp4Kernel<false><<<
        static_cast<unsigned>(tokens), kNormThreads, 0, stream>>>(
        hidden, w.pre_shared_norm_bf16, w.pre_expert_norm_bf16,
        w.router_scale_bf16, batch.token_packed, batch.token_scales,
        batch.expert_product_packed, batch.expert_product_scales,
        batch.shared_output, batch.token_hidden, c.width, c.epsilon,
        w.shared_gate.activation_global_divisor,
        w.expert_gate_up.activation_global_divisor);
  }
  Status status = CheckLaunch(trellis35
                                  ? "launch Trellis35 T3 input boundary"
                                  : "launch M25 exact batched MoE input boundary");
  if (!status.ok()) return status;
  {
    const NvtxRange range("gem16.mtp.target_shared_moe");
    status = LaunchNvfp4Sm120FusedGateUpExactBatch(
        batch.token_packed, batch.token_scales, w.shared_gate.packed_e2m1,
        w.shared_gate.scales_e4m3fn, w.shared_up.packed_e2m1,
        w.shared_up.scales_e4m3fn, batch.shared_product, tokens,
        c.shared_intermediate, c.width,
        w.shared_gate.activation_global_divisor,
        w.shared_gate.weight_global_divisor,
        w.shared_up.activation_global_divisor,
        w.shared_up.weight_global_divisor, stream);
    if (!status.ok()) return status;
    status = LaunchNvfp4ReferenceActivationQuantization(
        batch.shared_product, batch.shared_product_packed,
        batch.shared_product_scales, tokens * c.shared_intermediate,
        w.shared_down.activation_global_divisor, stream);
    if (!status.ok()) return status;
    status = LaunchNvfp4Sm120DirectProjectionBf16FloatExactBatch(
        batch.shared_product_packed, batch.shared_product_scales,
        w.shared_down.packed_e2m1, w.shared_down.scales_e4m3fn,
        batch.shared_output, tokens, c.width, c.shared_intermediate,
        w.shared_down.activation_global_divisor,
        w.shared_down.weight_global_divisor, stream);
  }
  if (!status.ok()) return status;

  const unsigned router_blocks =
      (c.experts + kRouterWarpsPerBlock - 1U) / kRouterWarpsPerBlock;
  RouterProjectionWarpKernel<<<
      dim3(router_blocks, static_cast<unsigned>(tokens)), kThreads, 0,
      stream>>>(batch.token_hidden, w.router_projection_bf16,
                batch.router_logits, c.experts, c.width);
  status = CheckLaunch("launch M25 exact router projection");
  if (!status.ok()) return status;
  RouterTopKKernel<true, false><<<static_cast<unsigned>(tokens), kThreads, 0,
                                  stream>>>(
      batch.router_logits, w.per_expert_scale_bf16,
      batch.router_probabilities, batch.permutation, batch.reduced_output,
      c.experts, c.top_k, batch.routing_finite);
  status = CheckLaunch("launch M25 exact router Top-K");
  if (!status.ok()) return status;
  if (router_overlap != nullptr) {
    if (tokens != 3U || c.top_k != 8U || c.experts > 128U) {
      return Invalid("M25 router-overlap diagnostic requires T3 Top-8");
    }
    RecordMtpRouterOverlapKernel<<<1U, 1U, 0, stream>>>(
        batch.permutation, c.experts, c.top_k, router_overlap);
    status = CheckLaunch("record M25 T3 router overlap");
    if (!status.ok()) return status;
  }

  {
    const NvtxRange range("gem16.mtp.target_routed_moe");
    if (trellis35) {
      status = LaunchTrellis35SelectedExpertsT3(
          trellis_workspace->gate_up_output, batch.permutation,
          batch.reduced_output, *trellis_layer, *trellis_workspace,
          batch.token_hidden, stream);
      if (!status.ok()) return status;
    } else {
      // The physical-BF16 prefill W2 region is dead after routing and has
      // exactly enough bytes for assignment-major FP32 Gate/Up. The
      // shared-product region is likewise dead after shared W2 and holds the
      // smaller routed GELU product. These lifetime aliases stay inside the
      // fixed M17 arena and add no M25 allocation or persistent representation.
      float* expert_gate_up =
          reinterpret_cast<float*>(batch.expert_down_bf16);
      float* expert_product = batch.shared_product;
      status = LaunchNvfp4Sm120SelectedSplitGateUpMtpBatch(
          batch.expert_product_packed, batch.expert_product_scales,
          w.expert_gate_up.packed_e2m1, w.expert_gate_up.scales_e4m3fn,
          batch.permutation, tokens, c.top_k, expert_gate_up,
          expert_gate_up + c.expert_intermediate, c.expert_intermediate,
          c.width, c.experts, w.expert_gate_up.activation_global_divisor,
          w.expert_gate_up.weight_global_divisor, stream);
      if (!status.ok()) return status;
      status = LaunchStridedGatedGeluNvfp4ActivationQuantization(
          expert_gate_up, expert_gate_up + c.expert_intermediate,
          expert_product, batch.expert_product_packed,
          batch.expert_product_scales, tokens * c.top_k,
          c.expert_intermediate, w.expert_down.activation_global_divisor,
          stream);
      if (!status.ok()) return status;
      status =
          LaunchNvfp4Sm120SelectedDirectProjectionReduceBf16FloatMtpBatch(
              batch.expert_product_packed, batch.expert_product_scales,
              w.expert_down.packed_e2m1, w.expert_down.scales_e4m3fn,
              batch.permutation, batch.reduced_output, tokens, c.top_k,
              batch.token_hidden, c.width, c.expert_intermediate, c.experts,
              w.expert_down.activation_global_divisor,
              w.expert_down.weight_global_divisor, stream);
      if (!status.ok()) return status;
    }
  }

  const std::size_t post_norm_shared_bytes =
      3U * static_cast<std::size_t>(c.width) * sizeof(float);
  MoeFusedPostNormResidualKernel<<<
      static_cast<unsigned>(tokens), kFusedPostNormThreads,
      post_norm_shared_bytes, stream>>>(
      batch.shared_output, w.post_shared_norm_bf16, batch.token_hidden,
      w.post_expert_norm_bf16, w.post_combined_norm_bf16, hidden,
      w.layer_scalar_bf16, output, c.width, c.epsilon);
  status = CheckLaunch("launch M25 exact shared-batch MoE post boundary");
  if (!status.ok()) return status;
  return Status::Ok();
}

Status LaunchGemma4MoeSm120MtpSharedBatchLayer(
    const float* hidden, float* output, std::uint64_t tokens,
    const Gemma4MoeReferenceConfig& config,
    const Gemma4MoeReferenceWeights& weights,
    const Gemma4MoePrefillWorkspace& batch_workspace,
    const Gemma4MoeReferenceWorkspace& decode_workspace,
    cudaStream_t stream, MtpRouterOverlapCounters* router_overlap) {
  return LaunchGemma4MoeSm120MtpSharedBatchLayerImpl(
      hidden, output, tokens, config, weights, batch_workspace,
      decode_workspace, stream, router_overlap);
}

Status LaunchGemma4MoeSm120Trellis35T3Layer(
    const float* hidden, float* output, std::uint64_t tokens,
    const Gemma4MoeReferenceConfig& config,
    const Gemma4MoeReferenceWeights& weights,
    const Trellis35DeviceLayerBinding& trellis_layer,
    const Trellis35T3Workspace& trellis_workspace,
    const Gemma4MoePrefillWorkspace& batch_workspace,
    cudaStream_t stream, MtpRouterOverlapCounters* router_overlap) {
  Gemma4MoeReferenceWorkspace unused;
  return LaunchGemma4MoeSm120MtpSharedBatchLayerImpl(
      hidden, output, tokens, config, weights, batch_workspace, unused,
      stream, router_overlap, &trellis_layer, &trellis_workspace);
}

Status LaunchGemma4MoeDecodeTopKDiagnostic(
    const float* logits, const std::uint16_t* per_expert_scale_bf16,
    float* probabilities, std::uint32_t* top_ids, float* top_weights,
    std::uint32_t experts, std::uint32_t top_k,
    Gemma4MoeDecodeTopK implementation, int* routing_finite,
    cudaStream_t stream) {
  if (logits == nullptr || per_expert_scale_bf16 == nullptr ||
      probabilities == nullptr || top_ids == nullptr ||
      top_weights == nullptr || experts == 0U || top_k == 0U ||
      top_k > experts) {
    return Invalid("decode router Top-K diagnostic contract is invalid");
  }
  if (implementation == Gemma4MoeDecodeTopK::kParallelExact) {
    RouterTopKKernel<true><<<1, kThreads, 0, stream>>>(
        logits, per_expert_scale_bf16, probabilities, top_ids, top_weights,
        experts, top_k, routing_finite);
  } else {
    RouterTopKKernel<false><<<1, kThreads, 0, stream>>>(
        logits, per_expert_scale_bf16, probabilities, top_ids, top_weights,
        experts, top_k, routing_finite);
  }
  return CheckLaunch("launch decode router Top-K diagnostic");
}

Status LaunchGemma4MoeTiledNvfp4ReferenceProjection(
    const Gemma4MoeNvfp4Matrix& matrix,
    const std::uint8_t* packed_activation_e2m1,
    const std::uint8_t* activation_scales_e4m3fn, float* output,
    cudaStream_t stream) {
  if (!MatrixValid(matrix, matrix.rows, matrix.columns) ||
      matrix.rows == 0U || matrix.columns == 0U ||
      packed_activation_e2m1 == nullptr ||
      activation_scales_e4m3fn == nullptr || output == nullptr) {
    return Invalid("M13 tiled NVFP4 projection contract is invalid");
  }
  return LaunchTiled(matrix, packed_activation_e2m1,
                     activation_scales_e4m3fn, output, stream);
}

}  // namespace gem16::internal
