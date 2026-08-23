#include "cuda/moe/reference.h"

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

#include "cuda/layer/reference.h"
#include "cuda/nvfp4/reference.h"
#include "cuda/nvfp4/sm120.h"

namespace gem16::internal {
namespace {

constexpr unsigned kThreads = 128;
constexpr unsigned kWarpSize = 32U;
constexpr unsigned kRouterWarpsPerBlock = 4U;
constexpr std::uint64_t kNvfp4Block = 16;
constexpr std::uint64_t kSm120KBlock = 64;
constexpr std::uint64_t kRowsPerTile = 8;
constexpr float kSqrtTwoOverPi = 0.7978845608028654F;
constexpr float kGeluCubic = 0.044715F;

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

__global__ void RouterTransformKernel(const float* hidden,
                                      const float* normalized,
                                      const std::uint16_t* scale,
                                      float* transformed,
                                      std::uint64_t width) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < width) {
    (void)hidden;
    transformed[index] = RoundBf16(
        normalized[index] * Bf16(scale[index]) * rsqrtf(static_cast<float>(width)));
  }
}

__global__ void RouterProjectionWarpKernel(const float* input,
                                           const std::uint16_t* weights,
                                           float* logits,
                                           std::uint32_t experts,
                                           std::uint64_t width) {
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

__global__ void RouterTopKKernel(
    const float* logits, const std::uint16_t* per_expert_scale,
    float* probabilities, std::uint32_t* top_ids, float* top_weights,
    std::uint32_t experts, std::uint32_t top_k, int* routing_finite) {
  if (blockIdx.x != 0U) return;
  __shared__ float maximum;
  __shared__ float total;
  __shared__ int valid;
  __shared__ float candidate_probability[kThreads];
  __shared__ std::uint32_t candidate_id[kThreads];
  if (threadIdx.x == 0U) {
    maximum = -3.402823466e+38F;
    valid = 1;
    for (std::uint32_t expert = 0; expert < experts; ++expert) {
      const float logit = logits[expert];
      if (!isfinite(logit)) valid = 0;
      maximum = fmaxf(maximum, logit);
    }
  }
  __syncthreads();
  if (!valid) {
    for (std::uint32_t expert = threadIdx.x; expert < experts;
         expert += blockDim.x) {
      probabilities[expert] = 0.0F;
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
    probabilities[expert] = expf(logits[expert] - maximum);
  }
  __syncthreads();
  if (threadIdx.x == 0U) {
    total = 0.0F;
    for (std::uint32_t expert = 0; expert < experts; ++expert) {
      total += probabilities[expert];
    }
    if (!isfinite(total) || total <= 0.0F) valid = 0;
  }
  __syncthreads();
  if (!valid) {
    for (std::uint32_t expert = threadIdx.x; expert < experts;
         expert += blockDim.x) {
      probabilities[expert] = 0.0F;
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
    probabilities[expert] /= total;
  }
  __syncthreads();
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
    candidate_probability[threadIdx.x] = best_probability;
    candidate_id[threadIdx.x] = best_id;
    __syncthreads();
    for (unsigned stride = kThreads / 2U; stride != 0U; stride /= 2U) {
      if (threadIdx.x < stride) {
        const float right_probability =
            candidate_probability[threadIdx.x + stride];
        const std::uint32_t right_id = candidate_id[threadIdx.x + stride];
        if (right_probability > candidate_probability[threadIdx.x] ||
            (right_probability == candidate_probability[threadIdx.x] &&
             right_id < candidate_id[threadIdx.x])) {
          candidate_probability[threadIdx.x] = right_probability;
          candidate_id[threadIdx.x] = right_id;
        }
      }
      __syncthreads();
    }
    if (threadIdx.x == 0U) {
      if (candidate_id[0] >= experts) {
        valid = 0;
      } else {
        top_ids[slot] = candidate_id[0];
        top_weights[slot] = candidate_probability[0];
      }
    }
    __syncthreads();
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
      for (std::uint32_t expert = 0; expert < experts; ++expert) {
        probabilities[expert] = 0.0F;
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
    contributions[offset] = weighted;
    sum += weighted;
  }
  routed_sum[index] = RoundBf16(sum);
}

__global__ void CombineKernel(const float* left, const float* right,
                              float* output, std::uint64_t width) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < width) output[index] = RoundBf16(left[index] + right[index]);
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
    cudaStream_t stream) {
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
      !MatrixValid(w.expert_gate_up,
                   static_cast<std::uint64_t>(c.experts) * 2U *
                       c.expert_intermediate,
                   c.width) ||
      !MatrixValid(w.expert_down,
                   static_cast<std::uint64_t>(c.experts) * c.width,
                   c.expert_intermediate)) {
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

  Status status = LaunchRmsNormBf16(hidden, w.pre_shared_norm_bf16,
                                    x.shared_input, 1U, c.width, c.epsilon,
                                    stream);
  if (!status.ok()) return status;
  status = LaunchNvfp4ReferenceActivationQuantization(
      x.shared_input, x.shared_input_packed, x.shared_input_scales, c.width,
      w.shared_gate.activation_global_divisor, stream);
  if (!status.ok()) return status;
  if (native_sm120) {
    status = LaunchNvfp4Sm120FusedGateUp(
        x.shared_input_packed, x.shared_input_scales,
        w.shared_gate.packed_e2m1, w.shared_gate.scales_e4m3fn,
        w.shared_up.packed_e2m1, w.shared_up.scales_e4m3fn, x.shared_gate,
        x.shared_up, x.shared_product, c.shared_intermediate, c.width,
        w.shared_gate.activation_global_divisor,
        w.shared_gate.weight_global_divisor,
        w.shared_up.activation_global_divisor,
        w.shared_up.weight_global_divisor, stream);
    if (!status.ok()) return status;
  } else {
    status = LaunchTiled(w.shared_gate, x.shared_input_packed,
                         x.shared_input_scales, x.shared_gate, stream);
    if (!status.ok()) return status;
    status = LaunchTiled(w.shared_up, x.shared_input_packed,
                         x.shared_input_scales, x.shared_up, stream);
    if (!status.ok()) return status;
    GatedGeluSeparateProductKernel<<<
        static_cast<unsigned>(Blocks(c.shared_intermediate)), kThreads, 0,
        stream>>>(x.shared_gate, x.shared_up, x.shared_product,
                  c.shared_intermediate);
    status = CheckLaunch("launch M11 shared gated GELU");
    if (!status.ok()) return status;
  }
  status = LaunchNvfp4ReferenceActivationQuantization(
      x.shared_product, x.shared_product_packed, x.shared_product_scales,
      c.shared_intermediate, w.shared_down.activation_global_divisor, stream);
  if (!status.ok()) return status;
  status = native_sm120
               ? LaunchNvfp4Sm120DirectProjectionBf16Float(
                     x.shared_product_packed, x.shared_product_scales,
                     w.shared_down.packed_e2m1,
                     w.shared_down.scales_e4m3fn, x.shared_output, c.width,
                     c.shared_intermediate,
                     w.shared_down.activation_global_divisor,
                     w.shared_down.weight_global_divisor, stream)
               : LaunchTiled(w.shared_down, x.shared_product_packed,
                             x.shared_product_scales, x.shared_output, stream);
  if (!status.ok()) return status;
  status = LaunchRmsNormBf16(x.shared_output, w.post_shared_norm_bf16,
                             x.shared_post, 1U, c.width, c.epsilon, stream);
  if (!status.ok()) return status;

  status = LaunchRmsNormBf16(hidden, nullptr, x.router_normalized, 1U,
                             c.width, c.epsilon, stream);
  if (!status.ok()) return status;
  RouterTransformKernel<<<static_cast<unsigned>(Blocks(c.width)), kThreads, 0,
                          stream>>>(hidden, x.router_normalized,
                                    w.router_scale_bf16,
                                    x.router_transformed, c.width);
  status = CheckLaunch("launch M11 router transform");
  if (!status.ok()) return status;
  const unsigned router_blocks =
      (c.experts + kRouterWarpsPerBlock - 1U) / kRouterWarpsPerBlock;
  RouterProjectionWarpKernel<<<router_blocks, kThreads, 0, stream>>>(
      x.router_transformed, w.router_projection_bf16, x.router_logits,
      c.experts, c.width);
  status = CheckLaunch("launch M11 router projection");
  if (!status.ok()) return status;
  RouterTopKKernel<<<1, kThreads, 0, stream>>>(
      x.router_logits, w.per_expert_scale_bf16, x.router_probabilities,
      x.top_ids, x.top_weights, c.experts, c.top_k, x.routing_finite);
  status = CheckLaunch("launch M11 deterministic router top-k");
  if (!status.ok()) return status;

  status = LaunchRmsNormBf16(hidden, w.pre_expert_norm_bf16,
                             x.expert_input, 1U, c.width, c.epsilon, stream);
  if (!status.ok()) return status;
  status = LaunchNvfp4ReferenceActivationQuantization(
      x.expert_input, x.expert_input_packed, x.expert_input_scales, c.width,
      w.expert_gate_up.activation_global_divisor, stream);
  if (!status.ok()) return status;
  if (native_sm120) {
    status = LaunchNvfp4Sm120SelectedFusedGateUpBatch(
        x.expert_input_packed, x.expert_input_scales,
        w.expert_gate_up.packed_e2m1, w.expert_gate_up.scales_e4m3fn,
        x.top_ids, c.top_k, x.expert_gate_up,
        x.expert_gate_up + c.expert_intermediate, x.expert_product,
        c.expert_intermediate, c.width, c.experts,
        w.expert_gate_up.activation_global_divisor,
        w.expert_gate_up.weight_global_divisor, stream);
    if (!status.ok()) return status;
    status = LaunchNvfp4ReferenceActivationQuantization(
        x.expert_product, x.expert_product_packed, x.expert_product_scales,
        static_cast<std::uint64_t>(c.top_k) * c.expert_intermediate,
        w.expert_down.activation_global_divisor, stream);
    if (!status.ok()) return status;
    status = LaunchNvfp4Sm120SelectedDirectProjectionBf16FloatBatch(
        x.expert_product_packed, x.expert_product_scales,
        w.expert_down.packed_e2m1, w.expert_down.scales_e4m3fn, x.top_ids,
        c.top_k, x.expert_down, c.width, c.expert_intermediate, c.experts,
        w.expert_down.activation_global_divisor,
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
  WeightedReductionKernel<<<static_cast<unsigned>(Blocks(c.width)), kThreads,
                            0, stream>>>(
      x.expert_down, x.top_weights, x.expert_contributions, x.routed_sum,
      c.width, c.top_k);
  status = CheckLaunch("launch M11 slot-order expert reduction");
  if (!status.ok()) return status;
  status = LaunchRmsNormBf16(x.routed_sum, w.post_expert_norm_bf16,
                             x.routed_post, 1U, c.width, c.epsilon, stream);
  if (!status.ok()) return status;
  CombineKernel<<<static_cast<unsigned>(Blocks(c.width)), kThreads, 0,
                  stream>>>(x.shared_post, x.routed_post, x.combined, c.width);
  status = CheckLaunch("launch M11 shared and routed combination");
  if (!status.ok()) return status;
  return LaunchRmsNormResidualBf16(
      x.combined, w.post_combined_norm_bf16, hidden, x.feed_forward, output,
      1U, c.width, c.epsilon, w.layer_scalar_bf16, stream);
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
    const Gemma4MoeReferenceWorkspace& workspace, cudaStream_t stream) {
  return LaunchGemma4MoeLayerImpl(hidden, output, config, weights, workspace,
                                  true, stream);
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
