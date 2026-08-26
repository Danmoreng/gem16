#include "cuda/moe/router_diagnostic.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

namespace gem16::internal {
namespace {

constexpr unsigned kThreads = 256U;
constexpr unsigned kTensorTokenRows = 16U;
constexpr unsigned kTensorTokenTiles = 2U;
constexpr unsigned kTensorTokens = kTensorTokenRows * kTensorTokenTiles;
constexpr unsigned kTensorExperts = 128U;
constexpr unsigned kTensorKTile = 64U;
constexpr unsigned kTensorWarps = 8U;
constexpr unsigned kTensorThreads = 32U * kTensorWarps;
bool comparison_enabled = false;

struct DeviceComparisonSummary {
  unsigned long long cases;
  unsigned long long top8_set_matches;
  unsigned long long top8_order_matches;
  unsigned long long changed_top8_slots;
  unsigned long long flip_cases;
  double flip_margin_sum;
  double gating_l1_sum;
  unsigned maximum_logit_absolute_delta_bits;
  unsigned maximum_flip_margin_8_9_bits;
  unsigned maximum_tensor_flip_margin_8_9_bits;
  unsigned maximum_gating_l1_bits;
  unsigned long long flip_margin_histogram[8];
  unsigned long long tensor_flip_margin_histogram[8];
  unsigned long long serial_margin_histogram[8];
  unsigned long long tensor_margin_histogram[8];
};

__device__ DeviceComparisonSummary comparison_summary;

Status Invalid(std::string message) {
  return Status(StatusCode::kInvalidArgument, std::move(message));
}

Status CudaFailure(const char* operation, cudaError_t error) {
  return Status(StatusCode::kInternal,
                std::string(operation) + ": " + cudaGetErrorName(error) +
                    ": " + cudaGetErrorString(error));
}

__device__ __forceinline__ float Bf16(std::uint16_t value) {
  return static_cast<float>(__ushort_as_bfloat16(value));
}

__device__ __forceinline__ float RoundBf16(float value) {
  return static_cast<float>(__float2bfloat16_rn(value));
}

__global__ void TransformKernel(const float* normalized,
                                const std::uint16_t* scale,
                                float* transformed, std::uint64_t elements,
                                std::uint64_t width) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  const std::uint64_t column = index % width;
  transformed[index] = RoundBf16(normalized[index] * Bf16(scale[column]) *
                                 rsqrtf(static_cast<float>(width)));
}

__global__ void SerialProjectionKernel(
    const float* input, const std::uint16_t* weights, float* raw_logits,
    float* bf16_logits, std::uint64_t width, std::uint32_t experts,
    std::uint64_t total) {
  const std::uint64_t output =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (output >= total) return;
  const std::uint64_t token = output / experts;
  const std::uint32_t expert =
      static_cast<std::uint32_t>(output - token * experts);
  float accumulator = 0.0F;
  const std::uint64_t input_base = token * width;
  const std::uint64_t weight_base =
      static_cast<std::uint64_t>(expert) * width;
#pragma unroll 1
  for (std::uint64_t column = 0U; column < width; ++column) {
    accumulator = fmaf(Bf16(weights[weight_base + column]),
                       input[input_base + column], accumulator);
  }
  raw_logits[output] = accumulator;
  bf16_logits[output] = RoundBf16(accumulator);
}

__device__ __forceinline__ unsigned SharedAddress(const void* pointer) {
  return static_cast<unsigned>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ unsigned Swizzle(unsigned row, unsigned column) {
  return (((column >> 3U) ^ (row & 7U)) << 3U) | (column & 7U);
}

__device__ __forceinline__ unsigned SwizzledAddress(
    unsigned lane_base, unsigned contracting_offset, unsigned address_select,
    unsigned row_select) {
  return lane_base + ((contracting_offset | address_select) ^ row_select);
}

__device__ __forceinline__ void LoadMatrixX4(
    unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3,
    unsigned address) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
               "{%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(address));
#else
  (void)r0;
  (void)r1;
  (void)r2;
  (void)r3;
  (void)address;
#endif
}

__device__ __forceinline__ void MmaBf16(
    float& c0, float& c1, float& c2, float& c3, unsigned a0, unsigned a1,
    unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
               "{%0,%1,%2,%3};\n"
               : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0),
                 "r"(b1));
#else
  (void)c0;
  (void)c1;
  (void)c2;
  (void)c3;
  (void)a0;
  (void)a1;
  (void)a2;
  (void)a3;
  (void)b0;
  (void)b1;
#endif
}

__global__ void TensorProjectionKernel(
    const float* input, const std::uint16_t* weights, float* raw_logits,
    float* bf16_logits, std::uint64_t width, std::uint64_t tokens) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  __shared__ alignas(16)
      std::uint16_t staged_input[kTensorTokens][kTensorKTile];
  __shared__ alignas(16)
      std::uint16_t staged_weights[kTensorExperts][kTensorKTile];
  const unsigned thread = threadIdx.x;
  const unsigned warp = thread >> 5U;
  const unsigned lane = thread & 31U;
  const unsigned group_lane = lane >> 2U;
  const unsigned lane_in_group = lane & 3U;
  const unsigned a_matrix = lane >> 3U;
  const unsigned a_row_in_matrix = lane & 7U;
  const unsigned a_row_offset =
      a_row_in_matrix + ((a_matrix & 1U) << 3U);
  const unsigned b_row_in_matrix = lane & 7U;
  const unsigned b_contracting_offset = ((lane >> 3U) & 1U) << 3U;
  const unsigned input_base = SharedAddress(staged_input);
  const unsigned input_address_select = (a_matrix >> 1U) << 4U;
  const unsigned input_row_select = a_row_in_matrix << 4U;
  const unsigned weight_base =
      SharedAddress(staged_weights + warp * 16U);
  const unsigned weight_lane_base =
      weight_base + b_row_in_matrix * kTensorKTile * sizeof(std::uint16_t) +
      (lane >> 4U) * 8U * kTensorKTile * sizeof(std::uint16_t);
  const unsigned weight_address_select =
      (b_contracting_offset >> 3U) << 4U;
  const unsigned weight_row_select = b_row_in_matrix << 4U;
  float accumulators[kTensorTokenTiles][2][4] = {};
  const std::uint64_t token_base =
      static_cast<std::uint64_t>(blockIdx.x) * kTensorTokens;

  for (std::uint64_t k_base = 0U; k_base < width;
       k_base += kTensorKTile) {
    for (unsigned index = thread; index < kTensorTokens * kTensorKTile;
         index += blockDim.x) {
      const unsigned token_in_block = index / kTensorKTile;
      const unsigned column = index % kTensorKTile;
      const std::uint64_t token = token_base + token_in_block;
      const float value = token < tokens
                              ? input[token * width + k_base + column]
                              : 0.0F;
      staged_input[token_in_block]
                  [Swizzle(token_in_block & (kTensorTokenRows - 1U),
                           column)] =
          __bfloat16_as_ushort(__float2bfloat16_rn(value));
    }
    for (unsigned index = thread; index < kTensorExperts * kTensorKTile;
         index += blockDim.x) {
      const unsigned expert = index / kTensorKTile;
      const unsigned column = index % kTensorKTile;
      staged_weights[expert][Swizzle(expert & 15U, column)] =
          weights[static_cast<std::uint64_t>(expert) * width + k_base +
                  column];
    }
    __syncthreads();

#pragma unroll
    for (unsigned step = 0U; step < kTensorKTile / 16U; ++step) {
      const unsigned contracting_offset = step << 5U;
      unsigned weight_fragments[2][2];
      LoadMatrixX4(weight_fragments[0][0], weight_fragments[0][1],
                   weight_fragments[1][0], weight_fragments[1][1],
                   SwizzledAddress(weight_lane_base, contracting_offset,
                                   weight_address_select,
                                   weight_row_select));
#pragma unroll
      for (unsigned token_tile = 0U; token_tile < kTensorTokenTiles;
           ++token_tile) {
        const unsigned input_lane_base =
            input_base +
            (token_tile * kTensorTokenRows + a_row_offset) *
                kTensorKTile * sizeof(std::uint16_t);
        unsigned input_fragments[4];
        LoadMatrixX4(input_fragments[0], input_fragments[1],
                     input_fragments[2], input_fragments[3],
                     SwizzledAddress(input_lane_base, contracting_offset,
                                     input_address_select, input_row_select));
#pragma unroll
        for (unsigned expert_tile = 0U; expert_tile < 2U; ++expert_tile) {
          MmaBf16(accumulators[token_tile][expert_tile][0],
                  accumulators[token_tile][expert_tile][1],
                  accumulators[token_tile][expert_tile][2],
                  accumulators[token_tile][expert_tile][3],
                  input_fragments[0], input_fragments[1],
                  input_fragments[2], input_fragments[3],
                  weight_fragments[expert_tile][0],
                  weight_fragments[expert_tile][1]);
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (unsigned token_tile = 0U; token_tile < kTensorTokenTiles;
       ++token_tile) {
    const std::uint64_t token0 =
        token_base + token_tile * kTensorTokenRows + group_lane;
    const std::uint64_t token1 = token0 + 8U;
#pragma unroll
    for (unsigned expert_tile = 0U; expert_tile < 2U; ++expert_tile) {
      const unsigned expert0 =
          warp * 16U + expert_tile * 8U + lane_in_group * 2U;
      const float raw0 = accumulators[token_tile][expert_tile][0];
      const float raw1 = accumulators[token_tile][expert_tile][1];
      const float raw2 = accumulators[token_tile][expert_tile][2];
      const float raw3 = accumulators[token_tile][expert_tile][3];
      if (token0 < tokens) {
        const std::uint64_t output = token0 * kTensorExperts + expert0;
        raw_logits[output] = raw0;
        raw_logits[output + 1U] = raw1;
        bf16_logits[output] = RoundBf16(raw0);
        bf16_logits[output + 1U] = RoundBf16(raw1);
      }
      if (token1 < tokens) {
        const std::uint64_t output = token1 * kTensorExperts + expert0;
        raw_logits[output] = raw2;
        raw_logits[output + 1U] = raw3;
        bf16_logits[output] = RoundBf16(raw2);
        bf16_logits[output + 1U] = RoundBf16(raw3);
      }
    }
  }
#else
  (void)input;
  (void)weights;
  (void)raw_logits;
  (void)bf16_logits;
  (void)width;
  (void)tokens;
#endif
}

__device__ __forceinline__ void InsertTop9(
    float value, unsigned id, float (&values)[9], unsigned (&ids)[9]) {
  unsigned position = 9U;
#pragma unroll
  for (unsigned index = 0U; index < 9U; ++index) {
    if (value > values[index] ||
        (value == values[index] && id < ids[index])) {
      position = index;
      break;
    }
  }
  if (position == 9U) return;
#pragma unroll
  for (unsigned index = 8U; index > position; --index) {
    values[index] = values[index - 1U];
    ids[index] = ids[index - 1U];
  }
  values[position] = value;
  ids[position] = id;
}

__device__ __forceinline__ bool Contains(const unsigned (&ids)[9],
                                         unsigned value) {
#pragma unroll
  for (unsigned index = 0U; index < 8U; ++index) {
    if (ids[index] == value) return true;
  }
  return false;
}

__device__ __forceinline__ float SelectedWeight(
    const float* logits, const unsigned (&ids)[9], unsigned expert,
    const std::uint16_t* scales) {
  if (!Contains(ids, expert)) return 0.0F;
  float maximum = logits[ids[0]];
  float denominator = 0.0F;
#pragma unroll
  for (unsigned index = 0U; index < 8U; ++index) {
    denominator += expf(logits[ids[index]] - maximum);
  }
  return expf(logits[expert] - maximum) / denominator * Bf16(scales[expert]);
}

__device__ __forceinline__ unsigned MarginBin(float margin) {
  constexpr float thresholds[7] = {1.0F / 512.0F, 1.0F / 256.0F,
                                   1.0F / 128.0F, 1.0F / 64.0F,
                                   1.0F / 32.0F,  1.0F / 16.0F,
                                   1.0F / 8.0F};
#pragma unroll
  for (unsigned index = 0U; index < 7U; ++index) {
    if (margin <= thresholds[index]) return index;
  }
  return 7U;
}

__global__ void CompareProjectionKernel(
    const float* serial, const float* tensor,
    const std::uint16_t* per_expert_scale, std::uint64_t tokens) {
  const std::uint64_t token =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (token >= tokens) return;
  serial += token * kTensorExperts;
  tensor += token * kTensorExperts;
  float serial_values[9];
  float tensor_values[9];
  unsigned serial_ids[9];
  unsigned tensor_ids[9];
#pragma unroll
  for (unsigned index = 0U; index < 9U; ++index) {
    serial_values[index] = -CUDART_INF_F;
    tensor_values[index] = -CUDART_INF_F;
    serial_ids[index] = kTensorExperts;
    tensor_ids[index] = kTensorExperts;
  }
  float maximum_logit_delta = 0.0F;
#pragma unroll 1
  for (unsigned expert = 0U; expert < kTensorExperts; ++expert) {
    InsertTop9(serial[expert], expert, serial_values, serial_ids);
    InsertTop9(tensor[expert], expert, tensor_values, tensor_ids);
    maximum_logit_delta =
        fmaxf(maximum_logit_delta, fabsf(serial[expert] - tensor[expert]));
  }
  unsigned changed_slots = 0U;
  bool order_match = true;
  bool set_match = true;
#pragma unroll
  for (unsigned index = 0U; index < 8U; ++index) {
    order_match = order_match && serial_ids[index] == tensor_ids[index];
    set_match = set_match && Contains(tensor_ids, serial_ids[index]);
    changed_slots += !Contains(tensor_ids, serial_ids[index]);
  }
  float gating_l1 = 0.0F;
#pragma unroll 1
  for (unsigned expert = 0U; expert < kTensorExperts; ++expert) {
    gating_l1 += fabsf(SelectedWeight(serial, serial_ids, expert,
                                     per_expert_scale) -
                       SelectedWeight(tensor, tensor_ids, expert,
                                     per_expert_scale));
  }
  const float margin = serial_values[7] - serial_values[8];
  const float tensor_margin = tensor_values[7] - tensor_values[8];
  atomicAdd(&comparison_summary.cases, 1ULL);
  atomicAdd(&comparison_summary.top8_set_matches,
            static_cast<unsigned long long>(set_match));
  atomicAdd(&comparison_summary.top8_order_matches,
            static_cast<unsigned long long>(order_match));
  atomicAdd(&comparison_summary.changed_top8_slots,
            static_cast<unsigned long long>(changed_slots));
  atomicAdd(&comparison_summary.gating_l1_sum,
            static_cast<double>(gating_l1));
  atomicMax(&comparison_summary.maximum_logit_absolute_delta_bits,
            __float_as_uint(maximum_logit_delta));
  atomicMax(&comparison_summary.maximum_gating_l1_bits,
            __float_as_uint(gating_l1));
  atomicAdd(&comparison_summary.serial_margin_histogram[MarginBin(margin)],
            1ULL);
  atomicAdd(
      &comparison_summary.tensor_margin_histogram[MarginBin(tensor_margin)],
      1ULL);
  if (!set_match) {
    atomicAdd(&comparison_summary.flip_cases, 1ULL);
    atomicAdd(&comparison_summary.flip_margin_sum,
              static_cast<double>(margin));
    atomicMax(&comparison_summary.maximum_flip_margin_8_9_bits,
              __float_as_uint(margin));
    atomicMax(&comparison_summary.maximum_tensor_flip_margin_8_9_bits,
              __float_as_uint(tensor_margin));
    atomicAdd(&comparison_summary.flip_margin_histogram[MarginBin(margin)],
              1ULL);
    atomicAdd(&comparison_summary
                   .tensor_flip_margin_histogram[MarginBin(tensor_margin)],
              1ULL);
  }
}

}  // namespace

Status LaunchGemma4RouterProjectionDiagnostic(
    const float* normalized_input, const std::uint16_t* router_scale_bf16,
    const std::uint16_t* router_projection_bf16,
    const Gemma4RouterDiagnosticWorkspace& workspace, std::uint64_t tokens,
    std::uint64_t width, std::uint32_t experts, cudaStream_t stream) {
  if (normalized_input == nullptr || router_scale_bf16 == nullptr ||
      router_projection_bf16 == nullptr || workspace.transformed == nullptr ||
      workspace.serial_raw_logits == nullptr ||
      workspace.serial_bf16_logits == nullptr ||
      workspace.tensor_raw_logits == nullptr ||
      workspace.tensor_bf16_logits == nullptr || tokens == 0U ||
      width == 0U || width % kTensorKTile != 0U ||
      experts != kTensorExperts) {
    return Invalid("router diagnostic arguments are invalid");
  }
  const std::uint64_t transformed_elements = tokens * width;
  const std::uint64_t logits_elements = tokens * experts;
  TransformKernel<<<static_cast<unsigned>((transformed_elements + kThreads - 1U) /
                                          kThreads),
                    kThreads, 0, stream>>>(normalized_input,
                                           router_scale_bf16,
                                           workspace.transformed,
                                           transformed_elements, width);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaFailure("launch router diagnostic transform", error);
  }
  SerialProjectionKernel<<<
      static_cast<unsigned>((logits_elements + kThreads - 1U) / kThreads),
      kThreads, 0, stream>>>(
      workspace.transformed, router_projection_bf16,
      workspace.serial_raw_logits, workspace.serial_bf16_logits, width,
      experts, logits_elements);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaFailure("launch serial router diagnostic", error);
  }
  TensorProjectionKernel<<<
      static_cast<unsigned>((tokens + kTensorTokens - 1U) / kTensorTokens),
      kTensorThreads, 0, stream>>>(
      workspace.transformed, router_projection_bf16,
      workspace.tensor_raw_logits, workspace.tensor_bf16_logits, width,
      tokens);
  error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch tensor-core router diagnostic", error);
}

void SetGemma4RouterComparisonEnabled(bool enabled) {
  comparison_enabled = enabled;
}

bool Gemma4RouterComparisonEnabled() { return comparison_enabled; }

Status ResetGemma4RouterComparison(cudaStream_t stream) {
  const DeviceComparisonSummary zero{};
  cudaError_t error = cudaMemcpyToSymbolAsync(
      comparison_summary, &zero, sizeof(zero), 0U, cudaMemcpyHostToDevice,
      stream);
  if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("reset router comparison summary", error);
}

Result<Gemma4RouterComparisonSummary> CopyGemma4RouterComparison(
    cudaStream_t stream) {
  DeviceComparisonSummary device{};
  cudaError_t error = cudaMemcpyFromSymbolAsync(
      &device, comparison_summary, sizeof(device), 0U, cudaMemcpyDeviceToHost,
      stream);
  if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
  if (error != cudaSuccess) {
    return CudaFailure("copy router comparison summary", error);
  }
  Gemma4RouterComparisonSummary result;
  result.cases = device.cases;
  result.top8_set_matches = device.top8_set_matches;
  result.top8_order_matches = device.top8_order_matches;
  result.changed_top8_slots = device.changed_top8_slots;
  result.flip_cases = device.flip_cases;
  result.flip_margin_sum = device.flip_margin_sum;
  result.gating_l1_sum = device.gating_l1_sum;
  result.maximum_logit_absolute_delta =
      std::bit_cast<float>(device.maximum_logit_absolute_delta_bits);
  result.maximum_flip_margin_8_9 =
      std::bit_cast<float>(device.maximum_flip_margin_8_9_bits);
  result.maximum_tensor_flip_margin_8_9 =
      std::bit_cast<float>(device.maximum_tensor_flip_margin_8_9_bits);
  result.maximum_gating_l1 =
      std::bit_cast<float>(device.maximum_gating_l1_bits);
  for (std::size_t index = 0U; index < result.flip_margin_histogram.size();
       ++index) {
    result.flip_margin_histogram[index] =
        device.flip_margin_histogram[index];
    result.tensor_flip_margin_histogram[index] =
        device.tensor_flip_margin_histogram[index];
    result.serial_margin_histogram[index] =
        device.serial_margin_histogram[index];
    result.tensor_margin_histogram[index] =
        device.tensor_margin_histogram[index];
  }
  return result;
}

Status LaunchGemma4Sm120TensorRouterProjection(
    const float* transformed, const std::uint16_t* router_projection_bf16,
    float* tensor_bf16_logits, std::uint64_t tokens, std::uint64_t width,
    std::uint32_t experts, cudaStream_t stream) {
  if (transformed == nullptr || router_projection_bf16 == nullptr ||
      tensor_bf16_logits == nullptr || tokens == 0U ||
      width == 0U || width % kTensorKTile != 0U ||
      experts != kTensorExperts) {
    return Invalid("SM120 tensor router arguments are invalid");
  }
  // The comparison does not consume raw accumulators. Reuse the same scratch
  // for both outputs; each element receives the raw write followed by the
  // BF16-rounded write from its owning thread.
  TensorProjectionKernel<<<
      static_cast<unsigned>((tokens + kTensorTokens - 1U) / kTensorTokens),
      kTensorThreads, 0, stream>>>(
      transformed, router_projection_bf16, tensor_bf16_logits,
      tensor_bf16_logits, width, tokens);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch SM120 tensor router projection", error);
}

Status LaunchGemma4RouterComparisonDiagnostic(
    const float* serial_bf16_logits, const float* tensor_bf16_logits,
    const std::uint16_t* per_expert_scale_bf16, std::uint64_t tokens,
    std::uint32_t experts, std::uint32_t top_k, cudaStream_t stream) {
  if (serial_bf16_logits == nullptr || tensor_bf16_logits == nullptr ||
      per_expert_scale_bf16 == nullptr || tokens == 0U ||
      experts != kTensorExperts || top_k != 8U) {
    return Invalid("router comparison arguments are invalid");
  }
  CompareProjectionKernel<<<
      static_cast<unsigned>((tokens + kThreads - 1U) / kThreads), kThreads, 0,
      stream>>>(serial_bf16_logits, tensor_bf16_logits,
                per_expert_scale_bf16, tokens);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch router comparison summary", error);
}

}  // namespace gem16::internal
