#include "cuda/trellis35/reference.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <utility>

#include "cuda/fp8/reference.h"
#include "cuda/moe/prefill.h"
#include "exllamav3_quant/util.cuh"
#include "exllamav3_quant/quant/codebook.cuh"

namespace gem16::internal {
namespace {

constexpr unsigned kThreads = 256U;
constexpr unsigned kMmaWarps = 4U;
constexpr unsigned kMmaThreads = 32U * kMmaWarps;
constexpr float kHadamardScale = 0.08838834764831845F;
constexpr float kGeluScale = 0.7978845608028654F;
constexpr float kGeluCubic = 0.044715F;
constexpr float kE4M3Maximum = 448.0F;
constexpr unsigned kPrefillRowsPerTile = 4U;
constexpr unsigned kPrefillOutputBlock = 128U;

Status Invalid(std::string message) {
  return Status(StatusCode::kInvalidArgument, std::move(message));
}

Status CudaFailure(const char* operation, cudaError_t error) {
  return Status(StatusCode::kInternal,
                std::string(operation) + ": " + cudaGetErrorName(error) +
                    ": " + cudaGetErrorString(error));
}

Status CheckLaunch(const char* operation) {
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess ? Status::Ok() : CudaFailure(operation, error);
}

__device__ __forceinline__ float F16(const std::uint16_t* value) {
  return __half2float(__ushort_as_half(*value));
}

__device__ __forceinline__ unsigned TensorCoreIndex(unsigned row,
                                                     unsigned column) {
  const unsigned thread = (column & 7U) * 4U + ((row & 7U) >> 1U);
  const unsigned offset = (column >= 8U ? 4U : 0U) +
                          (row >= 8U ? 2U : 0U) + (row & 1U);
  return thread * 8U + offset;
}

__device__ __forceinline__ std::uint16_t PackedWord(
    const std::uint16_t* payload, unsigned original_word) {
  return payload[original_word ^ 1U];
}

template <int Rate>
__device__ __forceinline__ unsigned BranchAt(const std::uint16_t* payload,
                                              unsigned position) {
  const unsigned span = position >> 4U;
  const unsigned element = position & 15U;
  std::uint64_t bits = 0U;
#pragma unroll
  for (unsigned word = 0U; word < Rate; ++word) {
    bits = (bits << 16U) |
           static_cast<std::uint64_t>(
               PackedWord(payload, span * Rate + word));
  }
  return static_cast<unsigned>(
      (bits >> (Rate * (15U - element))) & ((1U << Rate) - 1U));
}

template <int Rate>
__device__ __forceinline__ half DecodeWeight(
    const std::byte* pool, std::uint32_t pool_offset,
    std::uint64_t input_elements, std::uint64_t output_elements,
    std::uint64_t input, std::uint64_t output) {
  const std::uint64_t tile_columns = output_elements / 16U;
  const std::uint64_t tile = (input / 16U) * tile_columns + output / 16U;
  const auto* payload = reinterpret_cast<const std::uint16_t*>(
      pool + pool_offset + tile * 32U * Rate);
  const unsigned position =
      TensorCoreIndex(static_cast<unsigned>(input & 15U),
                      static_cast<unsigned>(output & 15U));
  std::uint32_t state = 0U;
#pragma unroll
  for (unsigned history = 0U; history < (16U + Rate - 1U) / Rate;
       ++history) {
    const unsigned prior = (position + 256U - history) & 255U;
    state |= BranchAt<Rate>(payload, prior) << (Rate * history);
  }
  return decode_3inst<2>(state & 0xffffU);
}

template <int Rate>
__device__ __forceinline__ std::uint32_t DecodeWeightE4M3x4(
    const std::byte* pool, std::uint32_t pool_offset,
    std::uint64_t input_elements, std::uint64_t output_elements,
    std::uint64_t input, std::uint64_t output) {
  std::uint32_t packed = 0U;
#pragma unroll
  for (unsigned index = 0U; index < 4U; ++index) {
    const half decoded = DecodeWeight<Rate>(
        pool, pool_offset, input_elements, output_elements, input + index,
        output);
    const __nv_fp8_e4m3 value(__half2float(decoded));
    packed |= static_cast<std::uint32_t>(value.__x) << (index * 8U);
  }
  return packed;
}

struct Fp8Accumulator {
  float x0 = 0.0F;
  float x1 = 0.0F;
  float x2 = 0.0F;
  float x3 = 0.0F;
};

__device__ __forceinline__ void AccumulateFp8(
    std::uint32_t a0, std::uint32_t a1, std::uint32_t b0,
    std::uint32_t b1, Fp8Accumulator& accumulator) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0, %1, %2, %3}, "
      "{%4, %4, %5, %5}, "
      "{%6, %7}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(accumulator.x0), "+f"(accumulator.x1),
        "+f"(accumulator.x2), "+f"(accumulator.x3)
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
#else
  (void)a0;
  (void)a1;
  (void)b0;
  (void)b1;
  (void)accumulator;
#endif
}

__global__ void InputTransformKernel(const float* input,
                                     const std::uint16_t* suh,
                                     float* output,
                                     std::uint64_t logical_elements,
                                     std::uint64_t physical_elements) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= physical_elements) return;
  const std::uint64_t block = index & ~std::uint64_t{127U};
  const unsigned column = static_cast<unsigned>(index & 127U);
  float accumulator = 0.0F;
#pragma unroll
  for (unsigned row = 0U; row < 128U; ++row) {
    const std::uint64_t source = block + row;
    const float value = source < logical_elements ? input[source] : 0.0F;
    const float sign = (__popc(row & column) & 1U) == 0U ? 1.0F : -1.0F;
    accumulator = fmaf(value * F16(suh + source), sign, accumulator);
  }
  output[index] = accumulator * kHadamardScale;
}

__global__ void ReferenceW4A8ProjectionKernel(
    const std::uint8_t* activation, const float* activation_scale,
    Trellis35DeviceFamilyBinding family, std::uint32_t expert, float* output,
    std::uint64_t input_elements, std::uint64_t output_elements) {
  const std::uint64_t column =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (column >= output_elements) return;
  const Trellis35ExpertDescriptor descriptor = family.descriptors[expert];
  float accumulator = 0.0F;
  for (std::uint64_t row = 0U; row < input_elements; ++row) {
    __nv_fp8_e4m3 activation_value;
    activation_value.__x = activation[row];
    const half decoded =
        descriptor.rate_bits == 3U
            ? DecodeWeight<3>(family.k3_payload_pool, descriptor.pool_offset,
                              input_elements, output_elements, row, column)
            : DecodeWeight<4>(family.k4_payload_pool, descriptor.pool_offset,
                              input_elements, output_elements, row, column);
    const __nv_fp8_e4m3 weight_value(__half2float(decoded));
    accumulator = fmaf(static_cast<float>(activation_value),
                       static_cast<float>(weight_value), accumulator);
  }
  output[column] = accumulator * activation_scale[0];
}

__global__ void MmaW4A8ProjectionSelectedKernel(
    const std::uint8_t* activation, const float* activation_scales,
    Trellis35DeviceFamilyBinding family,
    const std::uint32_t* selected_experts, float* output,
    std::uint64_t input_elements, std::uint64_t output_elements) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  const unsigned slot = blockIdx.y;
  const unsigned warp = threadIdx.x >> 5U;
  const unsigned lane = threadIdx.x & 31U;
  const std::uint64_t output_base =
      (static_cast<std::uint64_t>(blockIdx.x) * kMmaWarps + warp) * 8U;
  const std::uint64_t source_output = output_base + (lane >> 2U);
  if (source_output >= output_elements) return;

  const std::uint32_t expert = selected_experts[slot];
  if (expert >= kTrellis35ExpertCount) return;
  const Trellis35ExpertDescriptor descriptor = family.descriptors[expert];
  const std::byte* pool = descriptor.rate_bits == 3U
                              ? family.k3_payload_pool
                              : family.k4_payload_pool;
  activation += static_cast<std::uint64_t>(slot) * input_elements;
  output += static_cast<std::uint64_t>(slot) * output_elements;
  const std::uint64_t k_quarter = lane & 3U;

  Fp8Accumulator accumulator;
  for (std::uint64_t k = 0U; k < input_elements; k += 32U) {
    const std::uint64_t first = k + k_quarter * 4U;
    const std::uint32_t a0 =
        *reinterpret_cast<const std::uint32_t*>(activation + first);
    const std::uint32_t a1 =
        *reinterpret_cast<const std::uint32_t*>(activation + first + 16U);
    const std::uint32_t b0 =
        descriptor.rate_bits == 3U
            ? DecodeWeightE4M3x4<3>(
                  pool, descriptor.pool_offset, input_elements,
                  output_elements, first, source_output)
            : DecodeWeightE4M3x4<4>(
                  pool, descriptor.pool_offset, input_elements,
                  output_elements, first, source_output);
    const std::uint32_t b1 =
        descriptor.rate_bits == 3U
            ? DecodeWeightE4M3x4<3>(
                  pool, descriptor.pool_offset, input_elements,
                  output_elements, first + 16U, source_output)
            : DecodeWeightE4M3x4<4>(
                  pool, descriptor.pool_offset, input_elements,
                  output_elements, first + 16U, source_output);
    AccumulateFp8(a0, a1, b0, b1, accumulator);
  }

  if (lane < 4U) {
    const std::uint64_t index = output_base + lane * 2U;
    const float scale = activation_scales[slot];
    if (index < output_elements) output[index] = accumulator.x0 * scale;
    if (index + 1U < output_elements) {
      output[index + 1U] = accumulator.x1 * scale;
    }
  }
#else
  (void)activation;
  (void)activation_scales;
  (void)family;
  (void)selected_experts;
  (void)output;
  (void)input_elements;
  (void)output_elements;
#endif
}

__global__ void MmaW4A8ProjectionGroupedT3Kernel(
    const std::uint8_t* activation, const float* activation_scales,
    Trellis35DeviceFamilyBinding family,
    const std::uint32_t* selected_experts, float* output,
    std::uint64_t input_elements, std::uint64_t output_elements) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  const unsigned group_candidate = blockIdx.y;
  const std::uint32_t expert = selected_experts[group_candidate];
  if (expert >= kTrellis35ExpertCount) return;
  for (unsigned prior = 0U; prior < group_candidate; ++prior) {
    if (selected_experts[prior] == expert) return;
  }

  unsigned assignments[kTrellis35T3Rows]{};
  unsigned assignment_count = 0U;
#pragma unroll
  for (unsigned row = 0U; row < kTrellis35T3Rows; ++row) {
#pragma unroll
    for (unsigned slot = 0U; slot < kTrellis35M1TopK; ++slot) {
      const unsigned assignment = row * kTrellis35M1TopK + slot;
      if (selected_experts[assignment] == expert) {
        assignments[assignment_count++] = assignment;
        break;
      }
    }
  }
  if (assignment_count == 0U) return;

  const unsigned warp = threadIdx.x >> 5U;
  const unsigned lane = threadIdx.x & 31U;
  const std::uint64_t output_base =
      (static_cast<std::uint64_t>(blockIdx.x) * kMmaWarps + warp) * 8U;
  const std::uint64_t source_output = output_base + (lane >> 2U);
  if (source_output >= output_elements) return;
  const Trellis35ExpertDescriptor descriptor = family.descriptors[expert];
  const std::byte* pool = descriptor.rate_bits == 3U
                              ? family.k3_payload_pool
                              : family.k4_payload_pool;
  const std::uint64_t k_quarter = lane & 3U;
  Fp8Accumulator accumulators[kTrellis35T3Rows]{};

  for (std::uint64_t k = 0U; k < input_elements; k += 32U) {
    const std::uint64_t first = k + k_quarter * 4U;
    const std::uint32_t b0 =
        descriptor.rate_bits == 3U
            ? DecodeWeightE4M3x4<3>(
                  pool, descriptor.pool_offset, input_elements,
                  output_elements, first, source_output)
            : DecodeWeightE4M3x4<4>(
                  pool, descriptor.pool_offset, input_elements,
                  output_elements, first, source_output);
    const std::uint32_t b1 =
        descriptor.rate_bits == 3U
            ? DecodeWeightE4M3x4<3>(
                  pool, descriptor.pool_offset, input_elements,
                  output_elements, first + 16U, source_output)
            : DecodeWeightE4M3x4<4>(
                  pool, descriptor.pool_offset, input_elements,
                  output_elements, first + 16U, source_output);
#pragma unroll
    for (unsigned index = 0U; index < kTrellis35T3Rows; ++index) {
      if (index < assignment_count) {
        const std::uint8_t* assignment_activation =
            activation +
            static_cast<std::uint64_t>(assignments[index]) * input_elements;
        const std::uint32_t a0 = *reinterpret_cast<const std::uint32_t*>(
            assignment_activation + first);
        const std::uint32_t a1 = *reinterpret_cast<const std::uint32_t*>(
            assignment_activation + first + 16U);
        AccumulateFp8(a0, a1, b0, b1, accumulators[index]);
      }
    }
  }

  if (lane < 4U) {
    const std::uint64_t column = output_base + lane * 2U;
#pragma unroll
    for (unsigned index = 0U; index < kTrellis35T3Rows; ++index) {
      if (index < assignment_count) {
        const unsigned assignment = assignments[index];
        float* assignment_output =
            output + static_cast<std::uint64_t>(assignment) * output_elements;
        const float scale = activation_scales[assignment];
        if (column < output_elements) {
          assignment_output[column] = accumulators[index].x0 * scale;
        }
        if (column + 1U < output_elements) {
          assignment_output[column + 1U] = accumulators[index].x1 * scale;
        }
      }
    }
  }
#else
  (void)activation;
  (void)activation_scales;
  (void)family;
  (void)selected_experts;
  (void)output;
  (void)input_elements;
  (void)output_elements;
#endif
}

__global__ void T3InputTransformKernel(
    const float* input, const std::uint16_t* all_suh,
    const std::uint32_t* selected_experts, float* output,
    std::uint64_t input_stride, unsigned assignments_per_input,
    std::uint64_t logical_elements, std::uint64_t physical_elements) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= physical_elements) return;
  const unsigned assignment = blockIdx.y;
  const std::uint32_t expert = selected_experts[assignment];
  if (expert >= kTrellis35ExpertCount) return;
  input += static_cast<std::uint64_t>(assignment / assignments_per_input) *
           input_stride;
  const std::uint16_t* suh =
      all_suh + static_cast<std::uint64_t>(expert) * physical_elements;
  output += static_cast<std::uint64_t>(assignment) * physical_elements;
  const std::uint64_t block = index & ~std::uint64_t{127U};
  const unsigned column = static_cast<unsigned>(index & 127U);
  float accumulator = 0.0F;
#pragma unroll
  for (unsigned row = 0U; row < 128U; ++row) {
    const std::uint64_t source = block + row;
    const float value = source < logical_elements ? input[source] : 0.0F;
    const float sign = (__popc(row & column) & 1U) == 0U ? 1.0F : -1.0F;
    accumulator = fmaf(value * F16(suh + source), sign, accumulator);
  }
  output[index] = accumulator * kHadamardScale;
}

__device__ __forceinline__ float PrefillTransformedValue(
    const float* input, const std::uint16_t* suh, std::uint64_t index,
    std::uint64_t logical_elements) {
  const std::uint64_t block = index & ~std::uint64_t{127U};
  const unsigned column = static_cast<unsigned>(index & 127U);
  float accumulator = 0.0F;
#pragma unroll
  for (unsigned row = 0U; row < 128U; ++row) {
    const std::uint64_t source = block + row;
    const float value = source < logical_elements ? input[source] : 0.0F;
    const float sign = (__popc(row & column) & 1U) == 0U ? 1.0F : -1.0F;
    accumulator = fmaf(value * F16(suh + source), sign, accumulator);
  }
  return accumulator * kHadamardScale;
}

__device__ __forceinline__ const float* PrefillAssignmentInput(
    const float* input, const Gemma4MoePrefillAssignment& assignment,
    unsigned assignment_index, std::uint64_t input_stride,
    bool token_major_input, std::uint64_t tokens) {
  if (assignment.expert_id >= kTrellis35ExpertCount ||
      (token_major_input && assignment.token_id >= tokens)) {
    return nullptr;
  }
  const std::uint64_t row =
      token_major_input ? assignment.token_id : assignment_index;
  return input + row * input_stride;
}

__global__ void PrefillTransformScaleKernel(
    const float* input, const std::uint16_t* all_suh,
    const Gemma4MoePrefillAssignment* assignments, float* scales,
    std::uint64_t assignment_count, std::uint64_t tokens,
    std::uint64_t input_stride, std::uint64_t logical_elements,
    std::uint64_t physical_elements, bool token_major_input) {
  const unsigned assignment_index = blockIdx.x;
  if (assignment_index >= assignment_count) return;
  const auto assignment = assignments[assignment_index];
  const float* assignment_input = PrefillAssignmentInput(
      input, assignment, assignment_index, input_stride, token_major_input,
      tokens);
  __shared__ float maxima[kThreads];
  float local_maximum = 0.0F;
  if (assignment_input != nullptr) {
    const std::uint16_t* suh =
        all_suh + static_cast<std::uint64_t>(assignment.expert_id) *
                      physical_elements;
    for (std::uint64_t index = threadIdx.x; index < physical_elements;
         index += blockDim.x) {
      local_maximum =
          fmaxf(local_maximum,
                fabsf(PrefillTransformedValue(assignment_input, suh, index,
                                              logical_elements)));
    }
  }
  maxima[threadIdx.x] = local_maximum;
  __syncthreads();
  for (unsigned stride = kThreads / 2U; stride != 0U; stride >>= 1U) {
    if (threadIdx.x < stride) {
      maxima[threadIdx.x] =
          fmaxf(maxima[threadIdx.x], maxima[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0U) {
    scales[assignment_index] =
        maxima[0] == 0.0F ? 1.0F : maxima[0] / kE4M3Maximum;
  }
}

__global__ void PrefillTransformQuantizeKernel(
    const float* input, const std::uint16_t* all_suh,
    const Gemma4MoePrefillAssignment* assignments, const float* scales,
    std::uint8_t* output, std::uint64_t assignment_count,
    std::uint64_t tokens, std::uint64_t input_stride,
    std::uint64_t logical_elements, std::uint64_t physical_elements,
    bool token_major_input) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const unsigned assignment_index = blockIdx.y;
  if (index >= physical_elements || assignment_index >= assignment_count) {
    return;
  }
  const auto assignment = assignments[assignment_index];
  const float* assignment_input = PrefillAssignmentInput(
      input, assignment, assignment_index, input_stride, token_major_input,
      tokens);
  if (assignment_input == nullptr) return;
  const std::uint16_t* suh =
      all_suh + static_cast<std::uint64_t>(assignment.expert_id) *
                    physical_elements;
  const float transformed = PrefillTransformedValue(
      assignment_input, suh, index, logical_elements);
  const __nv_fp8_e4m3 encoded(transformed / scales[assignment_index]);
  output[static_cast<std::uint64_t>(assignment_index) * physical_elements +
         index] = encoded.__x;
}

__global__ void MmaW4A8ProjectionGroupedPrefillTileKernel(
    const std::uint8_t* activation, const float* activation_scales,
    Trellis35DeviceFamilyBinding family,
    const std::uint32_t* expert_prefix, const std::uint32_t* schedule_count,
    const std::uint32_t* schedule, const std::uint32_t* permutation,
    float* output_tile, std::uint64_t input_elements,
    std::uint64_t output_elements,
    std::uint64_t output_offset) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  const unsigned schedule_index = blockIdx.y;
  if (schedule_index >= schedule_count[0]) return;
  const std::uint32_t packed_tile = schedule[schedule_index];
  const unsigned expert = packed_tile >> 16U;
  const std::uint32_t grouped_begin = packed_tile & 0xffffU;
  const std::uint32_t grouped_end = expert_prefix[expert + 1U];
  if (grouped_begin >= grouped_end) return;
  const unsigned assignment_count = static_cast<unsigned>(
      min(grouped_end - grouped_begin, kPrefillRowsPerTile));
  unsigned assignments[kPrefillRowsPerTile]{};
#pragma unroll
  for (unsigned index = 0U; index < kPrefillRowsPerTile; ++index) {
    if (index < assignment_count) {
      assignments[index] = permutation[grouped_begin + index];
    }
  }

  const unsigned warp = threadIdx.x >> 5U;
  const unsigned lane = threadIdx.x & 31U;
  const std::uint64_t tile_output =
      (static_cast<std::uint64_t>(blockIdx.x) * kMmaWarps + warp) * 8U +
      (lane >> 2U);
  const std::uint64_t source_output = output_offset + tile_output;
  if (tile_output >= kPrefillOutputBlock ||
      source_output >= output_elements) {
    return;
  }
  const Trellis35ExpertDescriptor descriptor = family.descriptors[expert];
  const std::byte* pool = descriptor.rate_bits == 3U
                              ? family.k3_payload_pool
                              : family.k4_payload_pool;
  const std::uint64_t k_quarter = lane & 3U;
  Fp8Accumulator accumulators[kPrefillRowsPerTile]{};
  for (std::uint64_t k = 0U; k < input_elements; k += 32U) {
    const std::uint64_t first = k + k_quarter * 4U;
    const std::uint32_t b0 =
        descriptor.rate_bits == 3U
            ? DecodeWeightE4M3x4<3>(
                  pool, descriptor.pool_offset, input_elements,
                  output_elements, first, source_output)
            : DecodeWeightE4M3x4<4>(
                  pool, descriptor.pool_offset, input_elements,
                  output_elements, first, source_output);
    const std::uint32_t b1 =
        descriptor.rate_bits == 3U
            ? DecodeWeightE4M3x4<3>(
                  pool, descriptor.pool_offset, input_elements,
                  output_elements, first + 16U, source_output)
            : DecodeWeightE4M3x4<4>(
                  pool, descriptor.pool_offset, input_elements,
                  output_elements, first + 16U, source_output);
#pragma unroll
    for (unsigned index = 0U; index < kPrefillRowsPerTile; ++index) {
      if (index < assignment_count) {
        const std::uint8_t* assignment_activation =
            activation +
            static_cast<std::uint64_t>(assignments[index]) * input_elements;
        const std::uint32_t a0 = *reinterpret_cast<const std::uint32_t*>(
            assignment_activation + first);
        const std::uint32_t a1 = *reinterpret_cast<const std::uint32_t*>(
            assignment_activation + first + 16U);
        AccumulateFp8(a0, a1, b0, b1, accumulators[index]);
      }
    }
  }
  if (lane < 4U) {
    const std::uint64_t column =
        (static_cast<std::uint64_t>(blockIdx.x) * kMmaWarps + warp) * 8U +
        lane * 2U;
#pragma unroll
    for (unsigned index = 0U; index < kPrefillRowsPerTile; ++index) {
      if (index < assignment_count) {
        const unsigned assignment = assignments[index];
        float* assignment_output =
            output_tile + static_cast<std::uint64_t>(assignment) *
                              kPrefillOutputBlock;
        const float scale = activation_scales[assignment];
        if (column < kPrefillOutputBlock) {
          assignment_output[column] = accumulators[index].x0 * scale;
        }
        if (column + 1U < kPrefillOutputBlock) {
          assignment_output[column + 1U] =
              accumulators[index].x1 * scale;
        }
      }
    }
  }
#else
  (void)activation;
  (void)activation_scales;
  (void)family;
  (void)expert_prefix;
  (void)schedule_count;
  (void)schedule;
  (void)permutation;
  (void)output_tile;
  (void)input_elements;
  (void)output_elements;
  (void)output_offset;
#endif
}

__global__ void BuildTrellis35PrefillTileScheduleKernel(
    const std::uint32_t* prefix, std::uint32_t* tile_count,
    std::uint32_t* schedule) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) return;
  std::uint32_t count = 0U;
  for (std::uint32_t expert = 0U; expert < kTrellis35ExpertCount; ++expert) {
    for (std::uint32_t grouped = prefix[expert];
         grouped < prefix[expert + 1U]; grouped += kPrefillRowsPerTile) {
      schedule[count++] = (expert << 16U) | grouped;
    }
  }
  tile_count[0] = count;
}

__global__ void RestoreTrellis35PrefillHistogramZeroKernel(
    std::uint32_t* histogram, const std::uint32_t* prefix) {
  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    histogram[0] = prefix[1U] - prefix[0U];
  }
}

__global__ void PrefillOutputTransformTileKernel(
    const float* input_tile, const std::uint16_t* all_svh,
    const Gemma4MoePrefillAssignment* assignments, float* output,
    std::uint64_t assignment_count, std::uint64_t output_elements,
    std::uint64_t output_offset) {
  const unsigned assignment_index = blockIdx.x;
  const unsigned column = threadIdx.x;
  if (assignment_index >= assignment_count ||
      column >= kPrefillOutputBlock) {
    return;
  }
  const std::uint32_t expert = assignments[assignment_index].expert_id;
  if (expert >= kTrellis35ExpertCount) return;
  input_tile += static_cast<std::uint64_t>(assignment_index) *
                kPrefillOutputBlock;
  const std::uint16_t* svh =
      all_svh + static_cast<std::uint64_t>(expert) * output_elements +
      output_offset;
  float accumulator = 0.0F;
#pragma unroll
  for (unsigned row = 0U; row < kPrefillOutputBlock; ++row) {
    const float sign = (__popc(row & column) & 1U) == 0U ? 1.0F : -1.0F;
    accumulator = fmaf(input_tile[row], sign, accumulator);
  }
  output[static_cast<std::uint64_t>(assignment_index) * output_elements +
         output_offset + column] =
      accumulator * kHadamardScale * F16(svh + column);
}

__global__ void SelectedInputTransformKernel(
    const float* input, const std::uint16_t* all_suh,
    const std::uint32_t* selected_experts, float* output,
    std::uint64_t input_stride, std::uint64_t logical_elements,
    std::uint64_t physical_elements) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= physical_elements) return;
  const unsigned slot = blockIdx.y;
  const std::uint32_t expert = selected_experts[slot];
  if (expert >= kTrellis35ExpertCount) return;
  input += static_cast<std::uint64_t>(slot) * input_stride;
  const std::uint16_t* suh =
      all_suh + static_cast<std::uint64_t>(expert) * physical_elements;
  output += static_cast<std::uint64_t>(slot) * physical_elements;
  const std::uint64_t block = index & ~std::uint64_t{127U};
  const unsigned column = static_cast<unsigned>(index & 127U);
  float accumulator = 0.0F;
#pragma unroll
  for (unsigned row = 0U; row < 128U; ++row) {
    const std::uint64_t source = block + row;
    const float value = source < logical_elements ? input[source] : 0.0F;
    const float sign = (__popc(row & column) & 1U) == 0U ? 1.0F : -1.0F;
    accumulator = fmaf(value * F16(suh + source), sign, accumulator);
  }
  output[index] = accumulator * kHadamardScale;
}

__global__ void SelectedOutputTransformKernel(
    const float* input, const std::uint16_t* all_svh,
    const std::uint32_t* selected_experts, float* output,
    std::uint64_t elements) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  const unsigned slot = blockIdx.y;
  const std::uint32_t expert = selected_experts[slot];
  if (expert >= kTrellis35ExpertCount) return;
  input += static_cast<std::uint64_t>(slot) * elements;
  output += static_cast<std::uint64_t>(slot) * elements;
  const std::uint16_t* svh =
      all_svh + static_cast<std::uint64_t>(expert) * elements;
  const std::uint64_t block = index & ~std::uint64_t{127U};
  const unsigned column = static_cast<unsigned>(index & 127U);
  float accumulator = 0.0F;
#pragma unroll
  for (unsigned row = 0U; row < 128U; ++row) {
    const float sign = (__popc(row & column) & 1U) == 0U ? 1.0F : -1.0F;
    accumulator = fmaf(input[block + row], sign, accumulator);
  }
  output[index] = accumulator * kHadamardScale * F16(svh + index);
}

__global__ void GatedGeluKernel(const float* gate_up, float* product) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= kTrellis35ExpertIntermediate) return;
  const unsigned slot = blockIdx.y;
  gate_up += static_cast<std::uint64_t>(slot) * kTrellis35GateUpOutput;
  product +=
      static_cast<std::uint64_t>(slot) * kTrellis35ExpertIntermediate;
  const float gate = gate_up[index];
  const float up = gate_up[index + kTrellis35ExpertIntermediate];
  const float gelu =
      0.5F * gate *
      (1.0F + tanhf(kGeluScale * (gate + kGeluCubic * gate * gate * gate)));
  product[index] = gelu * up;
}

__global__ void SlotOrderedReductionKernel(const float* expert_output,
                                           const float* route_weights,
                                           float* output) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= kTrellis35DownOutput) return;
  float accumulator = 0.0F;
#pragma unroll
  for (unsigned slot = 0U; slot < kTrellis35M1TopK; ++slot) {
    accumulator = fmaf(
        route_weights[slot],
        expert_output[static_cast<std::uint64_t>(slot) *
                          kTrellis35DownOutput +
                      index],
        accumulator);
  }
  output[index] = accumulator;
}

__global__ void SlotOrderedReductionT3Kernel(const float* expert_output,
                                             const float* route_weights,
                                             float* output) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= kTrellis35DownOutput) return;
  const unsigned row = blockIdx.y;
  float accumulator = 0.0F;
#pragma unroll
  for (unsigned slot = 0U; slot < kTrellis35M1TopK; ++slot) {
    const unsigned assignment = row * kTrellis35M1TopK + slot;
    accumulator = fmaf(
        route_weights[assignment],
        expert_output[static_cast<std::uint64_t>(assignment) *
                          kTrellis35DownOutput +
                      index],
        accumulator);
  }
  output[static_cast<std::uint64_t>(row) * kTrellis35DownOutput + index] =
      accumulator;
}

__global__ void OutputTransformKernel(const float* input,
                                      const std::uint16_t* svh,
                                      float* output,
                                      std::uint64_t elements) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  const std::uint64_t block = index & ~std::uint64_t{127U};
  const unsigned column = static_cast<unsigned>(index & 127U);
  float accumulator = 0.0F;
#pragma unroll
  for (unsigned row = 0U; row < 128U; ++row) {
    const float sign = (__popc(row & column) & 1U) == 0U ? 1.0F : -1.0F;
    accumulator = fmaf(input[block + row], sign, accumulator);
  }
  output[index] = accumulator * kHadamardScale * F16(svh + index);
}

Status TransformArguments(const float* input, const std::uint16_t* sidecar,
                          float* output, std::uint64_t elements,
                          std::string_view description) {
  if (input == nullptr || sidecar == nullptr || output == nullptr) {
    return Invalid(std::string(description) + " requires non-null pointers");
  }
  if (elements == 0U || elements % 128U != 0U ||
      elements > static_cast<std::uint64_t>(
                     std::numeric_limits<unsigned>::max()) * kThreads) {
    return Invalid(std::string(description) + " extent is invalid");
  }
  return Status::Ok();
}

Status LaunchGroupedT3Projection(
    const std::uint8_t* activation_e4m3, const float* activation_scales,
    const Trellis35DeviceFamilyBinding& family,
    const std::uint32_t* selected_experts, float* transformed_output,
    std::uint64_t input_elements, std::uint64_t output_elements,
    cudaStream_t stream) {
  if (activation_e4m3 == nullptr || activation_scales == nullptr ||
      family.k3_payload_pool == nullptr || family.k4_payload_pool == nullptr ||
      family.descriptors == nullptr || selected_experts == nullptr ||
      transformed_output == nullptr) {
    return Invalid("Trellis35 grouped T3 projection requires non-null pointers");
  }
  if (input_elements == 0U || output_elements == 0U ||
      input_elements % 32U != 0U || output_elements % 8U != 0U ||
      output_elements >
          static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max()) *
              kMmaWarps * 8U) {
    return Invalid("Trellis35 grouped T3 projection geometry is invalid");
  }
  const unsigned blocks = static_cast<unsigned>(
      (output_elements + kMmaWarps * 8U - 1U) / (kMmaWarps * 8U));
  MmaW4A8ProjectionGroupedT3Kernel<<<
      dim3(blocks, kTrellis35T3Assignments), kMmaThreads, 0, stream>>>(
      activation_e4m3, activation_scales, family, selected_experts,
      transformed_output, input_elements, output_elements);
  return CheckLaunch("launch Trellis35 grouped T3 W4A8 projection");
}

Status LaunchPrefillTransformQuantize(
    const float* input, const Trellis35DeviceFamilyBinding& family,
    const Gemma4MoePrefillAssignment* assignments, std::uint8_t* output,
    float* scales, std::uint64_t assignment_count, std::uint64_t tokens,
    std::uint64_t input_stride, std::uint64_t logical_elements,
    std::uint64_t physical_elements, bool token_major_input,
    cudaStream_t stream) {
  if (input == nullptr || family.suh_f16 == nullptr || assignments == nullptr ||
      output == nullptr || scales == nullptr || assignment_count == 0U ||
      tokens == 0U || input_stride == 0U || logical_elements == 0U ||
      logical_elements > physical_elements || physical_elements % 128U != 0U ||
      assignment_count >
          static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max()) ||
      physical_elements >
          static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max()) *
              kThreads) {
    return Invalid("Trellis35 prefill transform/quantize contract is invalid");
  }
  PrefillTransformScaleKernel<<<static_cast<unsigned>(assignment_count),
                                kThreads, 0, stream>>>(
      input, family.suh_f16, assignments, scales, assignment_count, tokens,
      input_stride, logical_elements, physical_elements, token_major_input);
  Status status = CheckLaunch("launch Trellis35 prefill transform scales");
  if (!status.ok()) return status;
  const unsigned blocks = static_cast<unsigned>(
      (physical_elements + kThreads - 1U) / kThreads);
  PrefillTransformQuantizeKernel<<<
      dim3(blocks, static_cast<unsigned>(assignment_count)), kThreads, 0,
      stream>>>(input, family.suh_f16, assignments, scales, output,
                assignment_count, tokens, input_stride, logical_elements,
                physical_elements, token_major_input);
  return CheckLaunch("launch direct Trellis35 prefill E4M3 quantization");
}

Status LaunchPrefillProjectionBlocks(
    const std::uint8_t* activation, const float* activation_scales,
    const Trellis35DeviceFamilyBinding& family,
    const Gemma4MoePrefillAssignment* assignments,
    const std::uint32_t* expert_prefix, const std::uint32_t* schedule_count,
    const std::uint32_t* schedule, const std::uint32_t* permutation,
    float* projection_tile, float* output, std::uint64_t tokens,
    std::uint64_t assignment_count, std::uint64_t input_elements,
    std::uint64_t output_elements, cudaStream_t stream) {
  if (activation == nullptr || activation_scales == nullptr ||
      family.k3_payload_pool == nullptr || family.k4_payload_pool == nullptr ||
      family.descriptors == nullptr || family.svh_f16 == nullptr ||
      assignments == nullptr || expert_prefix == nullptr ||
      schedule_count == nullptr || schedule == nullptr ||
      permutation == nullptr || projection_tile == nullptr ||
      output == nullptr || tokens == 0U || assignment_count == 0U ||
      input_elements == 0U || input_elements % 32U != 0U ||
      output_elements == 0U ||
      output_elements % kPrefillOutputBlock != 0U || tokens > 1024U ||
      assignment_count > 65535U) {
    return Invalid("Trellis35 grouped prefill projection contract is invalid");
  }
  const unsigned schedule_blocks = static_cast<unsigned>(assignment_count);
  const unsigned output_blocks =
      kPrefillOutputBlock / (kMmaWarps * 8U);
  for (std::uint64_t output_offset = 0U; output_offset < output_elements;
       output_offset += kPrefillOutputBlock) {
    MmaW4A8ProjectionGroupedPrefillTileKernel<<<
        dim3(output_blocks, schedule_blocks), kMmaThreads, 0, stream>>>(
        activation, activation_scales, family, expert_prefix, schedule_count,
        schedule, permutation, projection_tile, input_elements,
        output_elements, output_offset);
    Status status =
        CheckLaunch("launch grouped Trellis35 prefill W4A8 output tile");
    if (!status.ok()) return status;
    PrefillOutputTransformTileKernel<<<
        static_cast<unsigned>(assignment_count), kPrefillOutputBlock, 0,
        stream>>>(projection_tile, family.svh_f16, assignments, output,
                  assignment_count, output_elements, output_offset);
    status = CheckLaunch("launch Trellis35 prefill inverse output tile");
    if (!status.ok()) return status;
  }
  return Status::Ok();
}

}  // namespace

Status LaunchTrellis35InputTransformM1(
    const float* input, const std::uint16_t* suh_f16,
    float* transformed_output, std::uint64_t logical_elements,
    std::uint64_t physical_elements, cudaStream_t stream) {
  Status status = TransformArguments(input, suh_f16, transformed_output,
                                     physical_elements,
                                     "Trellis35 input transform");
  if (!status.ok()) return status;
  if (logical_elements == 0U || logical_elements > physical_elements) {
    return Invalid("Trellis35 logical input extent is invalid");
  }
  const unsigned blocks =
      static_cast<unsigned>((physical_elements + kThreads - 1U) / kThreads);
  InputTransformKernel<<<blocks, kThreads, 0, stream>>>(
      input, suh_f16, transformed_output, logical_elements, physical_elements);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch Trellis35 input transform", error);
}

Status LaunchTrellis35ReferenceW4A8ProjectionM1(
    const std::uint8_t* activation_e4m3, const float* activation_scale,
    const Trellis35DeviceFamilyBinding& family, std::uint32_t expert,
    float* transformed_output, std::uint64_t input_elements,
    std::uint64_t output_elements, cudaStream_t stream) {
  if (activation_e4m3 == nullptr || activation_scale == nullptr ||
      family.k3_payload_pool == nullptr || family.k4_payload_pool == nullptr ||
      family.descriptors == nullptr || transformed_output == nullptr) {
    return Invalid("Trellis35 W4A8 projection requires non-null pointers");
  }
  if (expert >= kTrellis35ExpertCount || input_elements == 0U ||
      output_elements == 0U || input_elements % 16U != 0U ||
      output_elements % 16U != 0U ||
      output_elements > static_cast<std::uint64_t>(
                            std::numeric_limits<unsigned>::max()) * kThreads) {
    return Invalid("Trellis35 W4A8 projection geometry is invalid");
  }
  const unsigned blocks =
      static_cast<unsigned>((output_elements + kThreads - 1U) / kThreads);
  ReferenceW4A8ProjectionKernel<<<blocks, kThreads, 0, stream>>>(
      activation_e4m3, activation_scale, family, expert, transformed_output,
      input_elements, output_elements);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch Trellis35 reference W4A8 projection",
                           error);
}

Status LaunchTrellis35MmaW4A8ProjectionM1(
    const std::uint8_t* activation_e4m3, const float* activation_scales,
    const Trellis35DeviceFamilyBinding& family,
    const std::uint32_t* selected_experts, float* transformed_output,
    std::uint64_t input_elements, std::uint64_t output_elements,
    cudaStream_t stream) {
  if (activation_e4m3 == nullptr || activation_scales == nullptr ||
      family.k3_payload_pool == nullptr || family.k4_payload_pool == nullptr ||
      family.descriptors == nullptr || selected_experts == nullptr ||
      transformed_output == nullptr) {
    return Invalid("Trellis35 MMA W4A8 projection requires non-null pointers");
  }
  if (input_elements == 0U || output_elements == 0U ||
      input_elements % 32U != 0U || output_elements % 8U != 0U ||
      output_elements >
          static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max()) *
              kMmaWarps * 8U) {
    return Invalid("Trellis35 MMA W4A8 projection geometry is invalid");
  }
  const unsigned blocks = static_cast<unsigned>(
      (output_elements + kMmaWarps * 8U - 1U) / (kMmaWarps * 8U));
  MmaW4A8ProjectionSelectedKernel<<<
      dim3(blocks, kTrellis35M1TopK), kMmaThreads, 0, stream>>>(
      activation_e4m3, activation_scales, family, selected_experts,
      transformed_output, input_elements, output_elements);
  return CheckLaunch("launch Trellis35 SM120 MMA W4A8 projection");
}

Status LaunchTrellis35OutputTransformM1(
    const float* transformed_input, const std::uint16_t* svh_f16,
    float* output, std::uint64_t elements, cudaStream_t stream) {
  Status status = TransformArguments(transformed_input, svh_f16, output,
                                     elements,
                                     "Trellis35 output transform");
  if (!status.ok()) return status;
  const unsigned blocks =
      static_cast<unsigned>((elements + kThreads - 1U) / kThreads);
  OutputTransformKernel<<<blocks, kThreads, 0, stream>>>(
      transformed_input, svh_f16, output, elements);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch Trellis35 output transform", error);
}

Status LaunchTrellis35SelectedExpertsM1(
    const float* input, const std::uint32_t* selected_experts,
    const float* route_weights, const Trellis35DeviceLayerBinding& layer,
    const Trellis35M1Workspace& workspace, float* output,
    cudaStream_t stream) {
  if (input == nullptr || selected_experts == nullptr ||
      route_weights == nullptr || output == nullptr ||
      layer.gate_up.k3_payload_pool == nullptr ||
      layer.gate_up.k4_payload_pool == nullptr ||
      layer.gate_up.descriptors == nullptr || layer.gate_up.suh_f16 == nullptr ||
      layer.gate_up.svh_f16 == nullptr ||
      layer.down.k3_payload_pool == nullptr ||
      layer.down.k4_payload_pool == nullptr ||
      layer.down.descriptors == nullptr || layer.down.suh_f16 == nullptr ||
      layer.down.svh_f16 == nullptr ||
      workspace.gate_up_input_transformed == nullptr ||
      workspace.gate_up_input_e4m3 == nullptr ||
      workspace.gate_up_input_scales == nullptr ||
      workspace.gate_up_transformed_output == nullptr ||
      workspace.gate_up_output == nullptr || workspace.product == nullptr ||
      workspace.down_input_transformed == nullptr ||
      workspace.down_input_e4m3 == nullptr ||
      workspace.down_input_scales == nullptr ||
      workspace.down_transformed_output == nullptr ||
      workspace.down_output == nullptr) {
    return Invalid("Trellis35 selected-expert M1 requires complete bindings");
  }

  const dim3 gate_input_blocks(
      static_cast<unsigned>((kTrellis35GateUpInput + kThreads - 1U) /
                            kThreads),
      kTrellis35M1TopK);
  SelectedInputTransformKernel<<<gate_input_blocks, kThreads, 0, stream>>>(
      input, layer.gate_up.suh_f16, selected_experts,
      workspace.gate_up_input_transformed, 0U, kTrellis35GateUpInput,
      kTrellis35GateUpInput);
  Status status = CheckLaunch("launch Trellis35 selected Gate+Up input transform");
  if (!status.ok()) return status;

  status = LaunchFp8ReferenceTokenQuantizationBatch(
      workspace.gate_up_input_transformed, workspace.gate_up_input_e4m3,
      workspace.gate_up_input_scales, kTrellis35M1TopK,
      kTrellis35GateUpInput, stream);
  if (!status.ok()) return status;
  status = LaunchTrellis35MmaW4A8ProjectionM1(
      workspace.gate_up_input_e4m3, workspace.gate_up_input_scales,
      layer.gate_up, selected_experts,
      workspace.gate_up_transformed_output, kTrellis35GateUpInput,
      kTrellis35GateUpOutput, stream);
  if (!status.ok()) return status;

  const dim3 gate_output_blocks(
      static_cast<unsigned>((kTrellis35GateUpOutput + kThreads - 1U) /
                            kThreads),
      kTrellis35M1TopK);
  SelectedOutputTransformKernel<<<gate_output_blocks, kThreads, 0, stream>>>(
      workspace.gate_up_transformed_output, layer.gate_up.svh_f16,
      selected_experts, workspace.gate_up_output, kTrellis35GateUpOutput);
  status = CheckLaunch("launch Trellis35 selected Gate+Up output transform");
  if (!status.ok()) return status;

  const dim3 product_blocks(
      static_cast<unsigned>((kTrellis35ExpertIntermediate + kThreads - 1U) /
                            kThreads),
      kTrellis35M1TopK);
  GatedGeluKernel<<<product_blocks, kThreads, 0, stream>>>(
      workspace.gate_up_output, workspace.product);
  status = CheckLaunch("launch Trellis35 selected gated GELU");
  if (!status.ok()) return status;

  const dim3 down_input_blocks(
      static_cast<unsigned>((kTrellis35DownInput + kThreads - 1U) / kThreads),
      kTrellis35M1TopK);
  SelectedInputTransformKernel<<<down_input_blocks, kThreads, 0, stream>>>(
      workspace.product, layer.down.suh_f16, selected_experts,
      workspace.down_input_transformed, kTrellis35ExpertIntermediate,
      kTrellis35ExpertIntermediate,
      kTrellis35DownInput);
  status = CheckLaunch("launch Trellis35 selected Down input transform");
  if (!status.ok()) return status;

  status = LaunchFp8ReferenceTokenQuantizationBatch(
      workspace.down_input_transformed, workspace.down_input_e4m3,
      workspace.down_input_scales, kTrellis35M1TopK, kTrellis35DownInput,
      stream);
  if (!status.ok()) return status;
  status = LaunchTrellis35MmaW4A8ProjectionM1(
      workspace.down_input_e4m3, workspace.down_input_scales, layer.down,
      selected_experts, workspace.down_transformed_output,
      kTrellis35DownInput, kTrellis35DownOutput, stream);
  if (!status.ok()) return status;

  const dim3 down_output_blocks(
      static_cast<unsigned>((kTrellis35DownOutput + kThreads - 1U) /
                            kThreads),
      kTrellis35M1TopK);
  SelectedOutputTransformKernel<<<down_output_blocks, kThreads, 0, stream>>>(
      workspace.down_transformed_output, layer.down.svh_f16,
      selected_experts, workspace.down_output, kTrellis35DownOutput);
  status = CheckLaunch("launch Trellis35 selected Down output transform");
  if (!status.ok()) return status;

  const unsigned reduction_blocks = static_cast<unsigned>(
      (kTrellis35DownOutput + kThreads - 1U) / kThreads);
  SlotOrderedReductionKernel<<<reduction_blocks, kThreads, 0, stream>>>(
      workspace.down_output, route_weights, output);
  return CheckLaunch("launch Trellis35 slot-ordered reduction");
}

Status LaunchTrellis35SelectedExpertsT3(
    const float* input_rows, const std::uint32_t* selected_experts,
    const float* route_weights, const Trellis35DeviceLayerBinding& layer,
    const Trellis35T3Workspace& workspace, float* output_rows,
    cudaStream_t stream) {
  if (input_rows == nullptr || selected_experts == nullptr ||
      route_weights == nullptr || output_rows == nullptr ||
      layer.gate_up.k3_payload_pool == nullptr ||
      layer.gate_up.k4_payload_pool == nullptr ||
      layer.gate_up.descriptors == nullptr || layer.gate_up.suh_f16 == nullptr ||
      layer.gate_up.svh_f16 == nullptr ||
      layer.down.k3_payload_pool == nullptr ||
      layer.down.k4_payload_pool == nullptr ||
      layer.down.descriptors == nullptr || layer.down.suh_f16 == nullptr ||
      layer.down.svh_f16 == nullptr ||
      workspace.gate_up_input_transformed == nullptr ||
      workspace.gate_up_input_e4m3 == nullptr ||
      workspace.gate_up_input_scales == nullptr ||
      workspace.gate_up_transformed_output == nullptr ||
      workspace.gate_up_output == nullptr || workspace.product == nullptr ||
      workspace.down_input_transformed == nullptr ||
      workspace.down_input_e4m3 == nullptr ||
      workspace.down_input_scales == nullptr ||
      workspace.down_transformed_output == nullptr ||
      workspace.down_output == nullptr) {
    return Invalid("Trellis35 selected-expert T3 requires complete bindings");
  }

  const dim3 gate_input_blocks(
      static_cast<unsigned>((kTrellis35GateUpInput + kThreads - 1U) /
                            kThreads),
      kTrellis35T3Assignments);
  T3InputTransformKernel<<<gate_input_blocks, kThreads, 0, stream>>>(
      input_rows, layer.gate_up.suh_f16, selected_experts,
      workspace.gate_up_input_transformed, kTrellis35GateUpInput,
      kTrellis35M1TopK, kTrellis35GateUpInput, kTrellis35GateUpInput);
  Status status =
      CheckLaunch("launch Trellis35 T3 Gate+Up input transform");
  if (!status.ok()) return status;
  status = LaunchFp8ReferenceTokenQuantizationBatch(
      workspace.gate_up_input_transformed, workspace.gate_up_input_e4m3,
      workspace.gate_up_input_scales, kTrellis35T3Assignments,
      kTrellis35GateUpInput, stream);
  if (!status.ok()) return status;
  status = LaunchGroupedT3Projection(
      workspace.gate_up_input_e4m3, workspace.gate_up_input_scales,
      layer.gate_up, selected_experts,
      workspace.gate_up_transformed_output, kTrellis35GateUpInput,
      kTrellis35GateUpOutput, stream);
  if (!status.ok()) return status;

  const dim3 gate_output_blocks(
      static_cast<unsigned>((kTrellis35GateUpOutput + kThreads - 1U) /
                            kThreads),
      kTrellis35T3Assignments);
  SelectedOutputTransformKernel<<<gate_output_blocks, kThreads, 0, stream>>>(
      workspace.gate_up_transformed_output, layer.gate_up.svh_f16,
      selected_experts, workspace.gate_up_output, kTrellis35GateUpOutput);
  status = CheckLaunch("launch Trellis35 T3 Gate+Up output transform");
  if (!status.ok()) return status;

  const dim3 product_blocks(
      static_cast<unsigned>((kTrellis35ExpertIntermediate + kThreads - 1U) /
                            kThreads),
      kTrellis35T3Assignments);
  GatedGeluKernel<<<product_blocks, kThreads, 0, stream>>>(
      workspace.gate_up_output, workspace.product);
  status = CheckLaunch("launch Trellis35 T3 gated GELU");
  if (!status.ok()) return status;

  const dim3 down_input_blocks(
      static_cast<unsigned>((kTrellis35DownInput + kThreads - 1U) / kThreads),
      kTrellis35T3Assignments);
  T3InputTransformKernel<<<down_input_blocks, kThreads, 0, stream>>>(
      workspace.product, layer.down.suh_f16, selected_experts,
      workspace.down_input_transformed, kTrellis35ExpertIntermediate, 1U,
      kTrellis35ExpertIntermediate, kTrellis35DownInput);
  status = CheckLaunch("launch Trellis35 T3 Down input transform");
  if (!status.ok()) return status;
  status = LaunchFp8ReferenceTokenQuantizationBatch(
      workspace.down_input_transformed, workspace.down_input_e4m3,
      workspace.down_input_scales, kTrellis35T3Assignments,
      kTrellis35DownInput, stream);
  if (!status.ok()) return status;
  status = LaunchGroupedT3Projection(
      workspace.down_input_e4m3, workspace.down_input_scales, layer.down,
      selected_experts, workspace.down_transformed_output,
      kTrellis35DownInput, kTrellis35DownOutput, stream);
  if (!status.ok()) return status;

  const dim3 down_output_blocks(
      static_cast<unsigned>((kTrellis35DownOutput + kThreads - 1U) /
                            kThreads),
      kTrellis35T3Assignments);
  SelectedOutputTransformKernel<<<down_output_blocks, kThreads, 0, stream>>>(
      workspace.down_transformed_output, layer.down.svh_f16,
      selected_experts, workspace.down_output, kTrellis35DownOutput);
  status = CheckLaunch("launch Trellis35 T3 Down output transform");
  if (!status.ok()) return status;

  const unsigned reduction_blocks = static_cast<unsigned>(
      (kTrellis35DownOutput + kThreads - 1U) / kThreads);
  SlotOrderedReductionT3Kernel<<<
      dim3(reduction_blocks, kTrellis35T3Rows), kThreads, 0, stream>>>(
      workspace.down_output, route_weights, output_rows);
  return CheckLaunch("launch Trellis35 T3 slot-ordered reduction");
}

Status LaunchTrellis35PrefillExpertsW4A8(
    const float* normalized_hidden, std::uint64_t tokens,
    const Trellis35DeviceLayerBinding& layer,
    const Gemma4MoePrefillWorkspace& workspace, cudaStream_t stream) {
  const std::uint64_t assignment_count = tokens * kTrellis35M1TopK;
  void* product_backing =
      workspace.expert_product != nullptr
          ? static_cast<void*>(workspace.expert_product)
          : static_cast<void*>(workspace.expert_product_bf16);
  void* down_backing =
      workspace.expert_down != nullptr
          ? static_cast<void*>(workspace.expert_down)
          : static_cast<void*>(workspace.expert_down_bf16);
  if (normalized_hidden == nullptr || tokens == 0U || tokens > 1024U ||
      assignment_count > 65535U || product_backing == nullptr ||
      down_backing == nullptr || workspace.token_scales == nullptr ||
      workspace.shared_product == nullptr || workspace.shared_output == nullptr ||
      workspace.token_hidden == nullptr || workspace.assignments == nullptr ||
      workspace.prefix == nullptr || workspace.permutation == nullptr ||
      workspace.router_logits == nullptr || workspace.histogram == nullptr) {
    return Invalid("Trellis35 prefill expert workspace is incomplete");
  }
  auto* gate_activation = static_cast<std::uint8_t*>(product_backing);
  auto* activation_scales =
      reinterpret_cast<float*>(workspace.token_scales);
  auto* projection_tile = workspace.shared_product;
  auto* gate_up_output = static_cast<float*>(down_backing);
  auto* product = static_cast<float*>(product_backing);
  auto* down_activation =
      reinterpret_cast<std::uint8_t*>(workspace.shared_output);
  auto* expert_down = static_cast<float*>(down_backing);
  auto* schedule = reinterpret_cast<std::uint32_t*>(workspace.router_logits);

  BuildTrellis35PrefillTileScheduleKernel<<<1, 1, 0, stream>>>(
      workspace.prefix, workspace.histogram, schedule);
  Status status = CheckLaunch("build Trellis35 prefill expert tile schedule");
  if (!status.ok()) return status;

  status = LaunchPrefillTransformQuantize(
      normalized_hidden, layer.gate_up, workspace.assignments,
      gate_activation, activation_scales, assignment_count, tokens,
      kTrellis35GateUpInput, kTrellis35GateUpInput,
      kTrellis35GateUpInput, true, stream);
  if (!status.ok()) return status;
  status = LaunchPrefillProjectionBlocks(
      gate_activation, activation_scales, layer.gate_up,
      workspace.assignments, workspace.prefix, workspace.histogram, schedule,
      workspace.permutation, projection_tile, gate_up_output, tokens,
      assignment_count,
      kTrellis35GateUpInput, kTrellis35GateUpOutput, stream);
  if (!status.ok()) return status;
  const dim3 product_blocks(
      static_cast<unsigned>((kTrellis35ExpertIntermediate + kThreads - 1U) /
                            kThreads),
      static_cast<unsigned>(assignment_count));
  GatedGeluKernel<<<product_blocks, kThreads, 0, stream>>>(gate_up_output,
                                                           product);
  status = CheckLaunch("launch Trellis35 prefill gated GELU");
  if (!status.ok()) return status;

  status = LaunchPrefillTransformQuantize(
      product, layer.down, workspace.assignments, down_activation,
      activation_scales, assignment_count, tokens,
      kTrellis35ExpertIntermediate, kTrellis35ExpertIntermediate,
      kTrellis35DownInput, false, stream);
  if (!status.ok()) return status;
  status = LaunchPrefillProjectionBlocks(
      down_activation, activation_scales, layer.down, workspace.assignments,
      workspace.prefix, workspace.histogram, schedule, workspace.permutation,
      projection_tile, expert_down, tokens, assignment_count, kTrellis35DownInput,
      kTrellis35DownOutput, stream);
  if (!status.ok()) return status;
  status = LaunchGemma4MoeReduceAssignments(
      expert_down, workspace.assignments, workspace.token_hidden,
      kTrellis35DownOutput, kTrellis35M1TopK, tokens, stream);
  if (!status.ok()) return status;
  RestoreTrellis35PrefillHistogramZeroKernel<<<1, 1, 0, stream>>>(
      workspace.histogram, workspace.prefix);
  return CheckLaunch("restore Trellis35 prefill expert-zero histogram");
}

}  // namespace gem16::internal
