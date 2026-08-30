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

}  // namespace gem16::internal
