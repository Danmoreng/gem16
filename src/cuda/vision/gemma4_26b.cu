#include "cuda/vision/gemma4_26b.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "cuda/fp8/cutlass_sm120.h"
#include "cuda/fp8/reference.h"
#include "cuda/layer/reference.h"
#include "model/gemma4_26b_vision_contract.h"
#include "util/environment.h"

namespace gem16::internal {
namespace {

constexpr std::uint32_t kPatchElements = 768U;
constexpr std::uint32_t kHidden = 1152U;
constexpr std::uint32_t kTextHidden = 2816U;
constexpr std::uint32_t kIntermediate = 4304U;
constexpr std::uint32_t kHeads = 16U;
constexpr std::uint32_t kHeadDimension = 72U;
constexpr std::uint32_t kRopeSpatialDimensions = 2U;
constexpr std::uint32_t kRopeFrequencies = 18U;
constexpr std::uint32_t kRopeEntriesPerToken =
    kRopeSpatialDimensions * kRopeFrequencies;
constexpr std::uint32_t kLayers = 27U;
constexpr std::size_t kLayerTimingStages = 10U;
constexpr std::size_t kTimingBoundaries =
    1U + 3U + kLayers * kLayerTimingStages + 2U;
constexpr std::size_t kCutlassWorkspaceBytes = 8U * 1024U * 1024U;
constexpr float kEpsilon = 1.0e-6F;

Status Invalid(std::string message) {
  return Status(StatusCode::kInvalidArgument, std::move(message));
}

Status CudaFailure(const char* operation, cudaError_t error) {
  return Status(error == cudaErrorMemoryAllocation
                    ? StatusCode::kResourceExhausted
                    : StatusCode::kInternal,
                std::string(operation) + ": " + cudaGetErrorName(error) +
                    ": " + cudaGetErrorString(error));
}

class NvtxRange {
 public:
  explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
  NvtxRange(const NvtxRange&) = delete;
  NvtxRange& operator=(const NvtxRange&) = delete;
  ~NvtxRange() { nvtxRangePop(); }
};

std::uint64_t Align(std::uint64_t value, std::uint64_t alignment) {
  if (value > std::numeric_limits<std::uint64_t>::max() - alignment + 1U) {
    return std::numeric_limits<std::uint64_t>::max();
  }
  return (value + alignment - 1U) & ~(alignment - 1U);
}

struct Layout {
  std::uint64_t bytes = 0U;
  template <typename T>
  std::uint64_t Add(std::uint64_t count) {
    bytes = Align(bytes, std::max<std::uint64_t>(256U, alignof(T)));
    if (count > (std::numeric_limits<std::uint64_t>::max() - bytes) /
                    sizeof(T)) {
      bytes = std::numeric_limits<std::uint64_t>::max();
      return bytes;
    }
    const std::uint64_t result = bytes;
    bytes += count * sizeof(T);
    return result;
  }
};

struct Linear {
  const std::uint8_t* weight = nullptr;
  const std::uint16_t* scale = nullptr;
  std::uint32_t rows = 0U;
  std::uint32_t columns = 0U;
};

struct Layer {
  const std::uint16_t* input_norm = nullptr;
  const std::uint16_t* post_attention_norm = nullptr;
  const std::uint16_t* pre_feedforward_norm = nullptr;
  const std::uint16_t* post_feedforward_norm = nullptr;
  const std::uint16_t* q_norm = nullptr;
  const std::uint16_t* k_norm = nullptr;
  Linear q;
  Linear k;
  Linear v;
  Linear o;
  Linear gate;
  Linear up;
  Linear down;
};

template <typename T>
Result<const T*> Tensor(const Gemma4Moe26BVisionDeviceArtifact& artifact,
                        const std::string& name) {
  auto pointer = artifact.Pointer(name);
  if (!pointer.ok()) return pointer.status();
  return reinterpret_cast<const T*>(pointer.value());
}

Result<Linear> BindLinear(
    const Gemma4Moe26BVisionDeviceArtifact& artifact,
    const std::string& name, std::uint32_t rows, std::uint32_t columns) {
  auto weight = Tensor<std::uint8_t>(artifact, name);
  const std::string scale_name =
      name.substr(0U, name.size() - std::string_view(".weight").size()) +
      ".weight_scale";
  auto scale = Tensor<std::uint16_t>(artifact, scale_name);
  if (!weight.ok()) return weight.status();
  if (!scale.ok()) return scale.status();
  return Linear{weight.value(), scale.value(), rows, columns};
}

__device__ __forceinline__ float Bf16(const std::uint16_t* value,
                                      std::uint64_t index) {
  return static_cast<float>(__ushort_as_bfloat16(value[index]));
}

__device__ __forceinline__ std::uint16_t ToBf16(float value) {
  return __bfloat16_as_ushort(__float2bfloat16_rn(value));
}

__global__ void PreparePatchesKernel(const float* input, float* output,
                                     std::uint64_t elements) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    output[index] = static_cast<float>(
        __float2bfloat16_rn(2.0F * (input[index] - 0.5F)));
  }
}

__global__ void AddPositionEmbeddingKernel(
    const std::uint16_t* projected, const std::int32_t* positions,
    const std::uint16_t* table, float* output, std::uint64_t elements) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  const std::uint64_t patch = index / kHidden;
  const std::uint64_t channel = index % kHidden;
  const std::int32_t x = positions[patch * 2U];
  const std::int32_t y = positions[patch * 2U + 1U];
  const float position =
      x < 0 || y < 0
          ? 0.0F
          : static_cast<float>(__float2bfloat16_rn(
                Bf16(table, static_cast<std::uint64_t>(x) * kHidden +
                                channel) +
                Bf16(table, (10240U + static_cast<std::uint64_t>(y)) *
                                    kHidden +
                                channel)));
  output[index] = static_cast<float>(__float2bfloat16_rn(
      Bf16(projected, index) + position));
}

__device__ float WarpSum(float value) {
  for (unsigned offset = 16U; offset != 0U; offset >>= 1U) {
    value += __shfl_down_sync(0xffffffffU, value, offset);
  }
  return __shfl_sync(0xffffffffU, value, 0U);
}

__global__ void BuildRopeTableKernel(const std::int32_t* positions,
                                     std::uint16_t* table,
                                     std::uint32_t tokens) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t entries =
      static_cast<std::uint64_t>(tokens) * kRopeEntriesPerToken;
  if (index >= entries) return;
  const std::uint64_t token = index / kRopeEntriesPerToken;
  const unsigned entry = static_cast<unsigned>(index % kRopeEntriesPerToken);
  const unsigned spatial = entry / kRopeFrequencies;
  const unsigned frequency = entry % kRopeFrequencies;
  const float position = static_cast<float>(positions[token * 2U + spatial]);
  const float inverse_frequency =
      powf(100.0F, -2.0F * static_cast<float>(frequency) / 36.0F);
  table[index * 2U] =
      ToBf16(cosf(position * inverse_frequency));
  table[index * 2U + 1U] =
      ToBf16(sinf(position * inverse_frequency));
}

template <bool kApplyRope, bool kUseTable>
__global__ void HeadNormRopeKernel(
    const std::uint16_t* input, const std::uint16_t* norm_weight,
    const std::int32_t* positions, const std::uint16_t* rope_table,
    std::uint16_t* output, std::uint64_t head_rows) {
  const unsigned warp_in_block = threadIdx.x / 32U;
  const unsigned lane = threadIdx.x % 32U;
  const std::uint64_t row =
      static_cast<std::uint64_t>(blockIdx.x) * (blockDim.x / 32U) +
      warp_in_block;
  if (row >= head_rows) return;
  const std::uint64_t token = row / kHeads;
  const std::uint64_t base = row * kHeadDimension;
  float squared = 0.0F;
  for (unsigned channel = lane; channel < kHeadDimension; channel += 32U) {
    const float value = Bf16(input, base + channel);
    squared = fmaf(value, value, squared);
  }
  const float inverse =
      rsqrtf(WarpSum(squared) / static_cast<float>(kHeadDimension) + kEpsilon);
  float values[3] = {};
#pragma unroll
  for (unsigned item = 0U; item < 3U; ++item) {
    const unsigned channel = lane + item * 32U;
    if (channel < kHeadDimension) {
      const float scale = norm_weight == nullptr ? 1.0F
                                                  : Bf16(norm_weight, channel);
      values[item] = static_cast<float>(__float2bfloat16_rn(
          Bf16(input, base + channel) * inverse * scale));
    }
  }
  if constexpr (kApplyRope) {
    float rotated_values[3] = {};
#pragma unroll
    for (unsigned item = 0U; item < 3U; ++item) {
      const unsigned channel = lane + item * 32U;
      if (channel >= kHeadDimension) continue;
      const unsigned spatial = channel / 36U;
      const unsigned local = channel % 36U;
      const unsigned paired_channel =
          spatial * 36U + (local < 18U ? local + 18U : local - 18U);
      const unsigned paired_lane = paired_channel % 32U;
      const float paired_0 =
          __shfl_sync(0xffffffffU, values[0], paired_lane);
      const float paired_1 =
          __shfl_sync(0xffffffffU, values[1], paired_lane);
      const float paired_2 =
          __shfl_sync(0xffffffffU, values[2], paired_lane);
      const float paired = paired_channel < 32U
                               ? paired_0
                               : paired_channel < 64U ? paired_1 : paired_2;
      const unsigned frequency = local % 18U;
      float cosine = 0.0F;
      float sine = 0.0F;
      if constexpr (kUseTable) {
        const std::uint64_t table_index =
            (token * kRopeEntriesPerToken +
             spatial * kRopeFrequencies + frequency) * 2U;
        cosine = Bf16(rope_table, table_index);
        sine = Bf16(rope_table, table_index + 1U);
      } else {
        const float position =
            static_cast<float>(positions[token * 2U + spatial]);
        const float inverse_frequency =
            powf(100.0F, -2.0F * static_cast<float>(frequency) / 36.0F);
        cosine = static_cast<float>(
            __float2bfloat16_rn(cosf(position * inverse_frequency)));
        sine = static_cast<float>(
            __float2bfloat16_rn(sinf(position * inverse_frequency)));
      }
      const float rotated = local < 18U ? -paired : paired;
      rotated_values[item] = static_cast<float>(
          __float2bfloat16_rn(values[item] * cosine + rotated * sine));
    }
#pragma unroll
    for (unsigned item = 0U; item < 3U; ++item) {
      values[item] = rotated_values[item];
    }
  }
#pragma unroll
  for (unsigned item = 0U; item < 3U; ++item) {
    const unsigned channel = lane + item * 32U;
    if (channel < kHeadDimension) output[base + channel] = ToBf16(values[item]);
  }
}

__global__ void FullAttentionKernel(
    const std::uint16_t* query, const std::uint16_t* key,
    const std::uint16_t* value, std::uint16_t* output,
    std::uint32_t tokens, std::uint32_t valid_tokens) {
  const unsigned warp_in_block = threadIdx.x / 32U;
  const unsigned lane = threadIdx.x % 32U;
  const std::uint64_t row =
      static_cast<std::uint64_t>(blockIdx.x) * (blockDim.x / 32U) +
      warp_in_block;
  const std::uint64_t rows = static_cast<std::uint64_t>(tokens) * kHeads;
  if (row >= rows) return;
  const std::uint64_t token = row / kHeads;
  const std::uint64_t head = row % kHeads;
  const std::uint64_t q_base = token * kHidden + head * kHeadDimension;
  float running_max = -__int_as_float(0x7f800000);
  float denominator = 0.0F;
  for (std::uint32_t source = 0U; source < valid_tokens; ++source) {
    const std::uint64_t kv_base =
        static_cast<std::uint64_t>(source) * kHidden +
        head * kHeadDimension;
    float partial = 0.0F;
#pragma unroll
    for (unsigned item = 0U; item < 3U; ++item) {
      const unsigned channel = lane + item * 32U;
      if (channel < kHeadDimension) {
        partial = fmaf(Bf16(query, q_base + channel),
                       Bf16(key, kv_base + channel), partial);
      }
    }
    const float score = WarpSum(partial);
    const float next_max = fmaxf(running_max, score);
    const float previous_scale = expf(running_max - next_max);
    const float source_scale = expf(score - next_max);
    denominator = denominator * previous_scale + source_scale;
    running_max = next_max;
  }

  // The pinned oracle computes the softmax in FP32, casts the probabilities
  // back to BF16, and only then performs the value projection. Recompute the
  // scores so this correctness-first kernel preserves that boundary without
  // allocating a quadratic attention-score slab.
  float accumulator[3] = {};
  for (std::uint32_t source = 0U; source < valid_tokens; ++source) {
    const std::uint64_t kv_base =
        static_cast<std::uint64_t>(source) * kHidden +
        head * kHeadDimension;
    float partial = 0.0F;
#pragma unroll
    for (unsigned item = 0U; item < 3U; ++item) {
      const unsigned channel = lane + item * 32U;
      if (channel < kHeadDimension) {
        partial = fmaf(Bf16(query, q_base + channel),
                       Bf16(key, kv_base + channel), partial);
      }
    }
    const float score = WarpSum(partial);
    const float probability = static_cast<float>(
        __float2bfloat16_rn(expf(score - running_max) / denominator));
#pragma unroll
    for (unsigned item = 0U; item < 3U; ++item) {
      const unsigned channel = lane + item * 32U;
      if (channel < kHeadDimension) {
        accumulator[item] =
            fmaf(probability, Bf16(value, kv_base + channel),
                 accumulator[item]);
      }
    }
  }
#pragma unroll
  for (unsigned item = 0U; item < 3U; ++item) {
    const unsigned channel = lane + item * 32U;
    if (channel < kHeadDimension) {
      output[q_base + channel] = ToBf16(accumulator[item]);
    }
  }
}

__global__ void GeluProductKernel(const std::uint16_t* gate,
                                  const std::uint16_t* up,
                                  std::uint16_t* product,
                                  std::uint64_t elements) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  const float x = Bf16(gate, index);
  constexpr float kAlpha = 0.7978845608028654F;
  const float activated =
      0.5F * x * (1.0F + tanhf(kAlpha * (x + 0.044715F * x * x * x)));
  product[index] = ToBf16(activated * Bf16(up, index));
}

__global__ void GeluProductQuantizeKernel(
    const std::uint16_t* gate, const std::uint16_t* up,
    std::uint8_t* output, float* output_scale) {
  constexpr unsigned kThreads = 256U;
  constexpr float kE4M3Maximum = 448.0F;
  constexpr float kAlpha = 0.7978845608028654F;
  const std::uint64_t token = blockIdx.x;
  gate += token * kIntermediate;
  up += token * kIntermediate;
  output += token * kIntermediate;
  output_scale += token;
  __shared__ std::uint16_t products[kIntermediate];
  __shared__ float maxima[kThreads];
  float local_maximum = 0.0F;
  for (std::uint32_t index = threadIdx.x; index < kIntermediate;
       index += blockDim.x) {
    const float x = Bf16(gate, index);
    const float activated =
        0.5F * x *
        (1.0F + tanhf(kAlpha * (x + 0.044715F * x * x * x)));
    const std::uint16_t product_bits =
        ToBf16(activated * Bf16(up, index));
    products[index] = product_bits;
    local_maximum = fmaxf(
        local_maximum,
        fabsf(static_cast<float>(__ushort_as_bfloat16(product_bits))));
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
    output_scale[0] =
        maxima[0] == 0.0F ? 1.0F : maxima[0] / kE4M3Maximum;
  }
  __syncthreads();
  const float scale = output_scale[0];
  for (std::uint32_t index = threadIdx.x; index < kIntermediate;
       index += blockDim.x) {
    const __nv_fp8_e4m3 encoded(
        static_cast<float>(__ushort_as_bfloat16(products[index])) / scale);
    output[index] = encoded.__x;
  }
}

__global__ void PoolStandardizeKernel(
    const float* hidden, const std::int32_t* positions,
    const std::uint16_t* bias, const std::uint16_t* scale, float* output,
    std::uint32_t raw_tokens, std::uint32_t soft_tokens,
    std::uint32_t pooled_width) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t elements =
      static_cast<std::uint64_t>(soft_tokens) * kHidden;
  if (index >= elements) return;
  const std::uint32_t pooled = static_cast<std::uint32_t>(index / kHidden);
  const std::uint32_t channel = static_cast<std::uint32_t>(index % kHidden);
  float sum = 0.0F;
  for (std::uint32_t raw = 0U; raw < raw_tokens; ++raw) {
    const std::int32_t x = positions[raw * 2U];
    const std::int32_t y = positions[raw * 2U + 1U];
    if (x >= 0 && y >= 0 &&
        static_cast<std::uint32_t>(x) / 3U +
                pooled_width * (static_cast<std::uint32_t>(y) / 3U) ==
            pooled) {
      sum += hidden[static_cast<std::uint64_t>(raw) * kHidden + channel];
    }
  }
  const float pooled_bf16 = static_cast<float>(__float2bfloat16_rn(sum / 9.0F));
  const float standardized =
      (pooled_bf16 * sqrtf(static_cast<float>(kHidden)) -
       Bf16(bias, channel)) * Bf16(scale, channel);
  output[index] = static_cast<float>(__float2bfloat16_rn(standardized));
}

__global__ void PoolStandardizeDirectKernel(
    const float* hidden, const std::uint16_t* bias,
    const std::uint16_t* scale, float* output, std::uint32_t soft_tokens,
    std::uint32_t pooled_width) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t elements =
      static_cast<std::uint64_t>(soft_tokens) * kHidden;
  if (index >= elements) return;
  const std::uint32_t pooled = static_cast<std::uint32_t>(index / kHidden);
  const std::uint32_t channel = static_cast<std::uint32_t>(index % kHidden);
  const std::uint32_t pooled_x = pooled % pooled_width;
  const std::uint32_t pooled_y = pooled / pooled_width;
  const std::uint32_t raw_width = pooled_width * 3U;
  float sum = 0.0F;
#pragma unroll
  for (std::uint32_t y = 0U; y < 3U; ++y) {
#pragma unroll
    for (std::uint32_t x = 0U; x < 3U; ++x) {
      const std::uint32_t raw =
          (pooled_y * 3U + y) * raw_width + pooled_x * 3U + x;
      sum += hidden[static_cast<std::uint64_t>(raw) * kHidden + channel];
    }
  }
  const float pooled_bf16 = static_cast<float>(__float2bfloat16_rn(sum / 9.0F));
  const float standardized =
      (pooled_bf16 * sqrtf(static_cast<float>(kHidden)) -
       Bf16(bias, channel)) * Bf16(scale, channel);
  output[index] = static_cast<float>(__float2bfloat16_rn(standardized));
}

__global__ void ExpandBf16Kernel(const std::uint16_t* input, float* output,
                                 std::uint64_t elements) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) output[index] = Bf16(input, index);
}

std::uint64_t Blocks(std::uint64_t elements, unsigned threads = 256U) {
  return (elements + threads - 1U) / threads;
}

Result<bool> DirectPoolFromEnvironment() {
  const char* selected = GetEnvironmentVariable("GEM16_VISION_POOL");
  if (selected == nullptr || selected[0] == '\0' ||
      std::string_view(selected) == "direct") {
    return true;
  }
  if (std::string_view(selected) == "scan") return false;
  return Invalid(std::string("invalid GEM16_VISION_POOL mode: ") + selected);
}

Result<bool> RopeTableFromEnvironment() {
  const char* selected = GetEnvironmentVariable("GEM16_VISION_ROPE");
  if (selected == nullptr || selected[0] == '\0' ||
      std::string_view(selected) == "table") {
    return true;
  }
  if (std::string_view(selected) == "transcendental") return false;
  return Invalid(std::string("invalid GEM16_VISION_ROPE mode: ") + selected);
}

Result<bool> FusedFfnQuantFromEnvironment() {
  const char* selected = GetEnvironmentVariable("GEM16_VISION_FFN_QUANT");
  if (selected == nullptr || selected[0] == '\0' ||
      std::string_view(selected) == "fused") {
    return true;
  }
  if (std::string_view(selected) == "split") return false;
  return Invalid(std::string("invalid GEM16_VISION_FFN_QUANT mode: ") +
                 selected);
}

}  // namespace

Status LaunchGemma4Moe26BVisionFusedGeluProductQuantization(
    const std::uint16_t* gate_bf16, const std::uint16_t* up_bf16,
    std::uint8_t* output_e4m3, float* output_scales, std::uint64_t tokens,
    cudaStream_t stream) {
  if (gate_bf16 == nullptr || up_bf16 == nullptr || output_e4m3 == nullptr ||
      output_scales == nullptr) {
    return Invalid("fused Vision GELU product quantization requires non-null "
                   "device pointers");
  }
  if (tokens == 0U ||
      tokens > static_cast<std::uint64_t>(
                   std::numeric_limits<unsigned>::max())) {
    return Invalid("fused Vision GELU product quantization extent is invalid");
  }
  GeluProductQuantizeKernel<<<static_cast<unsigned>(tokens), 256U, 0,
                              stream>>>(gate_bf16, up_bf16, output_e4m3,
                                        output_scales);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch fused Vision GELU product quantization",
                           error);
}

struct Gemma4Moe26BVisionTimingRecorder::Impl {
  std::array<cudaEvent_t, kTimingBoundaries> events{};
  std::size_t boundary_count = 0U;
  Gemma4Moe26BVisionRuntimeTimings metadata;

  ~Impl() {
    for (cudaEvent_t event : events) {
      if (event != nullptr) (void)cudaEventDestroy(event);
    }
  }
};

Gemma4Moe26BVisionTimingRecorder::Gemma4Moe26BVisionTimingRecorder() =
    default;
Gemma4Moe26BVisionTimingRecorder::~Gemma4Moe26BVisionTimingRecorder() =
    default;
Gemma4Moe26BVisionTimingRecorder::Gemma4Moe26BVisionTimingRecorder(
    Gemma4Moe26BVisionTimingRecorder&&) noexcept = default;
Gemma4Moe26BVisionTimingRecorder&
Gemma4Moe26BVisionTimingRecorder::operator=(
    Gemma4Moe26BVisionTimingRecorder&&) noexcept = default;
Gemma4Moe26BVisionTimingRecorder::Gemma4Moe26BVisionTimingRecorder(
    std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

Result<Gemma4Moe26BVisionTimingRecorder>
Gemma4Moe26BVisionTimingRecorder::Create() {
  auto impl = std::make_unique<Impl>();
  for (cudaEvent_t& event : impl->events) {
    const cudaError_t error = cudaEventCreate(&event);
    if (error != cudaSuccess) {
      return CudaFailure("create Gemma 4 26B Vision timing event", error);
    }
  }
  return Gemma4Moe26BVisionTimingRecorder(std::move(impl));
}

Status Gemma4Moe26BVisionTimingRecorder::Begin(
    cudaStream_t stream, const Gemma4Moe26BVisionInputSegment& segment) {
  if (!impl_ || stream == nullptr) {
    return Invalid("Gemma 4 26B Vision timing recorder is invalid");
  }
  impl_->boundary_count = 0U;
  impl_->metadata = Gemma4Moe26BVisionRuntimeTimings{};
  impl_->metadata.budget = segment.soft_token_budget;
  impl_->metadata.raw_patch_count = segment.raw_patch_count;
  impl_->metadata.soft_token_count = segment.soft_token_count;
  return Boundary(stream);
}

Status Gemma4Moe26BVisionTimingRecorder::Boundary(cudaStream_t stream) {
  if (!impl_ || stream == nullptr ||
      impl_->boundary_count >= impl_->events.size()) {
    return Invalid("Gemma 4 26B Vision timing boundary is invalid");
  }
  const cudaError_t error =
      cudaEventRecord(impl_->events[impl_->boundary_count++], stream);
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("record Gemma 4 26B Vision timing event", error);
}

Status Gemma4Moe26BVisionTimingRecorder::Resolve(
    Gemma4Moe26BVisionRuntimeTimings* timings) {
  if (!impl_ || timings == nullptr ||
      impl_->boundary_count != impl_->events.size()) {
    return Invalid("Gemma 4 26B Vision timing result is incomplete");
  }
  cudaError_t error = cudaEventSynchronize(impl_->events.back());
  if (error != cudaSuccess) {
    return CudaFailure("synchronize Gemma 4 26B Vision timing", error);
  }
  auto elapsed = [&](std::size_t begin, std::size_t end,
                     float* output) -> Status {
    const cudaError_t elapsed_error =
        cudaEventElapsedTime(output, impl_->events[begin], impl_->events[end]);
    return elapsed_error == cudaSuccess
               ? Status::Ok()
               : CudaFailure("resolve Gemma 4 26B Vision timing",
                             elapsed_error);
  };
  std::size_t boundary = 0U;
  Status status = elapsed(boundary, boundary + 1U,
                          &impl_->metadata.upload_milliseconds);
  if (!status.ok()) return status;
  ++boundary;
  status = elapsed(boundary, boundary + 1U,
                   &impl_->metadata.patch_project_milliseconds);
  if (!status.ok()) return status;
  ++boundary;
  status = elapsed(boundary, boundary + 1U,
                   &impl_->metadata.position_add_milliseconds);
  if (!status.ok()) return status;
  ++boundary;
  for (auto& layer : impl_->metadata.layers) {
    for (float* output : std::array<float*, kLayerTimingStages>{
             &layer.input_norm_quant_milliseconds,
             &layer.qkv_projection_milliseconds,
             &layer.qkv_norm_rope_milliseconds,
             &layer.attention_milliseconds,
             &layer.output_projection_residual_milliseconds,
             &layer.ffn_norm_quant_milliseconds,
             &layer.gate_up_milliseconds,
             &layer.gelu_milliseconds,
             &layer.product_quant_milliseconds,
             &layer.down_residual_milliseconds}) {
      status = elapsed(boundary, boundary + 1U, output);
      if (!status.ok()) return status;
      ++boundary;
    }
  }
  status = elapsed(boundary, boundary + 1U,
                   &impl_->metadata.pool_standardize_milliseconds);
  if (!status.ok()) return status;
  ++boundary;
  status = elapsed(boundary, boundary + 1U,
                   &impl_->metadata.final_norm_project_milliseconds);
  if (!status.ok()) return status;
  status = elapsed(0U, impl_->events.size() - 1U,
                   &impl_->metadata.total_gpu_milliseconds);
  if (!status.ok()) return status;
  *timings = impl_->metadata;
  return Status::Ok();
}

struct Gemma4Moe26BVisionRuntime::Impl {
  bool direct_pool = false;
  bool rope_table_enabled = false;
  bool fused_ffn_quant = false;
  std::uint32_t maximum_soft_token_budget = 0U;
  std::uint32_t maximum_raw_patches = 0U;
  Linear patch;
  Linear projector;
  const std::uint16_t* position_table = nullptr;
  const std::uint16_t* std_bias = nullptr;
  const std::uint16_t* std_scale = nullptr;
  std::array<Layer, kLayers> layers{};
  void* allocation = nullptr;
  std::uint64_t allocation_bytes = 0U;
  float* host_patches = nullptr;
  std::int32_t* host_positions = nullptr;
  float* device_patches = nullptr;
  std::int32_t* positions = nullptr;
  std::uint16_t* rope_table = nullptr;
  float* state_a = nullptr;
  float* state_b = nullptr;
  std::uint8_t* activation = nullptr;
  float* activation_scales = nullptr;
  std::uint16_t* q = nullptr;
  std::uint16_t* k = nullptr;
  std::uint16_t* v = nullptr;
  std::uint16_t* attention = nullptr;
  std::uint16_t* linear_a = nullptr;
  std::uint16_t* linear_b = nullptr;
  void* cutlass = nullptr;
  float* final_output = nullptr;
  std::uint32_t output_tokens = 0U;

  ~Impl() {
    if (host_patches != nullptr) (void)cudaFreeHost(host_patches);
    if (host_positions != nullptr) (void)cudaFreeHost(host_positions);
    if (allocation != nullptr) (void)cudaFree(allocation);
  }
};

Gemma4Moe26BVisionRuntime::Gemma4Moe26BVisionRuntime() = default;
Gemma4Moe26BVisionRuntime::~Gemma4Moe26BVisionRuntime() = default;
Gemma4Moe26BVisionRuntime::Gemma4Moe26BVisionRuntime(
    Gemma4Moe26BVisionRuntime&&) noexcept = default;
Gemma4Moe26BVisionRuntime& Gemma4Moe26BVisionRuntime::operator=(
    Gemma4Moe26BVisionRuntime&&) noexcept = default;
Gemma4Moe26BVisionRuntime::Gemma4Moe26BVisionRuntime(
    std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

Result<Gemma4Moe26BVisionRuntime> Gemma4Moe26BVisionRuntime::Create(
    const Gemma4Moe26BVisionDeviceArtifact& artifact,
    std::uint32_t maximum_soft_token_budget) {
  if (artifact.arena() == nullptr) {
    return Invalid("Gemma 4 26B Vision artifact is not resident");
  }
  if (maximum_soft_token_budget != 70U &&
      maximum_soft_token_budget != 140U &&
      maximum_soft_token_budget != 280U) {
    return Invalid("Gemma 4 26B Vision maximum soft-token budget must be 70, "
                   "140, or 280");
  }
  auto direct_pool = DirectPoolFromEnvironment();
  if (!direct_pool.ok()) return direct_pool.status();
  auto use_rope_table = RopeTableFromEnvironment();
  if (!use_rope_table.ok()) return use_rope_table.status();
  auto fused_ffn_quant = FusedFfnQuantFromEnvironment();
  if (!fused_ffn_quant.ok()) return fused_ffn_quant.status();
  auto impl = std::make_unique<Impl>();
  impl->direct_pool = direct_pool.value();
  impl->rope_table_enabled = use_rope_table.value();
  impl->fused_ffn_quant = fused_ffn_quant.value();
  impl->maximum_soft_token_budget = maximum_soft_token_budget;
  impl->maximum_raw_patches = maximum_soft_token_budget * 9U;
  auto patch = BindLinear(
      artifact, "model.vision_tower.patch_embedder.input_proj.weight",
      kHidden, kPatchElements);
  auto projector = BindLinear(
      artifact, "model.embed_vision.embedding_projection.weight",
      kTextHidden, kHidden);
  auto position = Tensor<std::uint16_t>(
      artifact, "model.vision_tower.patch_embedder.position_embedding_table");
  auto bias = Tensor<std::uint16_t>(artifact, "model.vision_tower.std_bias");
  auto scale = Tensor<std::uint16_t>(artifact, "model.vision_tower.std_scale");
  if (!patch.ok()) return patch.status();
  if (!projector.ok()) return projector.status();
  if (!position.ok()) return position.status();
  if (!bias.ok()) return bias.status();
  if (!scale.ok()) return scale.status();
  impl->patch = patch.value();
  impl->projector = projector.value();
  impl->position_table = position.value();
  impl->std_bias = bias.value();
  impl->std_scale = scale.value();

  for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
    const std::string prefix = "model.vision_tower.encoder.layers." +
                               std::to_string(layer) + ".";
    auto input_norm = Tensor<std::uint16_t>(artifact, prefix + "input_layernorm.weight");
    auto post_attention = Tensor<std::uint16_t>(artifact, prefix + "post_attention_layernorm.weight");
    auto pre_feedforward = Tensor<std::uint16_t>(artifact, prefix + "pre_feedforward_layernorm.weight");
    auto post_feedforward = Tensor<std::uint16_t>(artifact, prefix + "post_feedforward_layernorm.weight");
    auto q_norm = Tensor<std::uint16_t>(artifact, prefix + "self_attn.q_norm.weight");
    auto k_norm = Tensor<std::uint16_t>(artifact, prefix + "self_attn.k_norm.weight");
    auto q = BindLinear(artifact, prefix + "self_attn.q_proj.linear.weight", kHidden, kHidden);
    auto k = BindLinear(artifact, prefix + "self_attn.k_proj.linear.weight", kHidden, kHidden);
    auto v = BindLinear(artifact, prefix + "self_attn.v_proj.linear.weight", kHidden, kHidden);
    auto o = BindLinear(artifact, prefix + "self_attn.o_proj.linear.weight", kHidden, kHidden);
    auto gate = BindLinear(artifact, prefix + "mlp.gate_proj.linear.weight", kIntermediate, kHidden);
    auto up = BindLinear(artifact, prefix + "mlp.up_proj.linear.weight", kIntermediate, kHidden);
    auto down = BindLinear(artifact, prefix + "mlp.down_proj.linear.weight", kHidden, kIntermediate);
    if (!input_norm.ok()) return input_norm.status();
    if (!post_attention.ok()) return post_attention.status();
    if (!pre_feedforward.ok()) return pre_feedforward.status();
    if (!post_feedforward.ok()) return post_feedforward.status();
    if (!q_norm.ok()) return q_norm.status();
    if (!k_norm.ok()) return k_norm.status();
    if (!q.ok()) return q.status();
    if (!k.ok()) return k.status();
    if (!v.ok()) return v.status();
    if (!o.ok()) return o.status();
    if (!gate.ok()) return gate.status();
    if (!up.ok()) return up.status();
    if (!down.ok()) return down.status();
    impl->layers[layer] = {
        input_norm.value(), post_attention.value(), pre_feedforward.value(),
        post_feedforward.value(), q_norm.value(), k_norm.value(), q.value(),
        k.value(), v.value(), o.value(), gate.value(), up.value(), down.value()};
  }

  Layout layout;
  const std::uint64_t maximum_raw_patches = impl->maximum_raw_patches;
  const auto patches = layout.Add<float>(maximum_raw_patches * kPatchElements);
  const auto positions = layout.Add<std::int32_t>(maximum_raw_patches * 2U);
  const auto state_a = layout.Add<float>(maximum_raw_patches * kHidden);
  const auto state_b = layout.Add<float>(maximum_raw_patches * kHidden);
  const auto activation =
      layout.Add<std::uint8_t>(maximum_raw_patches * kIntermediate);
  const auto activation_scales = layout.Add<float>(maximum_raw_patches);
  const auto q =
      layout.Add<std::uint16_t>(maximum_raw_patches * kHidden);
  const auto k =
      layout.Add<std::uint16_t>(maximum_raw_patches * kHidden);
  const auto v =
      layout.Add<std::uint16_t>(maximum_raw_patches * kHidden);
  const auto attention =
      layout.Add<std::uint16_t>(maximum_raw_patches * kHidden);
  const auto linear_a =
      layout.Add<std::uint16_t>(maximum_raw_patches * kIntermediate);
  const auto linear_b =
      layout.Add<std::uint16_t>(maximum_raw_patches * kIntermediate);
  const auto cutlass = layout.Add<std::byte>(kCutlassWorkspaceBytes);
  const auto final_output = layout.Add<float>(
      static_cast<std::uint64_t>(maximum_soft_token_budget) * kTextHidden);
  const auto rope_table = layout.Add<std::uint16_t>(
      maximum_raw_patches * kRopeEntriesPerToken * 2U);
  if (layout.bytes == std::numeric_limits<std::uint64_t>::max() ||
      layout.bytes > std::numeric_limits<std::size_t>::max()) {
    return Invalid("Gemma 4 26B Vision workspace layout overflow");
  }
  cudaError_t error = cudaMalloc(&impl->allocation, static_cast<std::size_t>(layout.bytes));
  if (error != cudaSuccess) return CudaFailure("allocate fixed Gemma 4 26B Vision workspace", error);
  impl->allocation_bytes = layout.bytes;
  error = cudaHostAlloc(reinterpret_cast<void**>(&impl->host_patches),
                        static_cast<std::size_t>(maximum_raw_patches) *
                            kPatchElements * sizeof(float),
                        cudaHostAllocDefault);
  if (error == cudaSuccess) {
    error = cudaHostAlloc(reinterpret_cast<void**>(&impl->host_positions),
                          static_cast<std::size_t>(maximum_raw_patches) * 2U *
                              sizeof(std::int32_t),
                          cudaHostAllocDefault);
  }
  if (error != cudaSuccess) return CudaFailure("allocate pinned Gemma 4 26B Vision input staging", error);
  auto ptr = [&](std::uint64_t offset) { return static_cast<std::byte*>(impl->allocation) + offset; };
  impl->device_patches = reinterpret_cast<float*>(ptr(patches));
  impl->positions = reinterpret_cast<std::int32_t*>(ptr(positions));
  impl->rope_table = reinterpret_cast<std::uint16_t*>(ptr(rope_table));
  impl->state_a = reinterpret_cast<float*>(ptr(state_a));
  impl->state_b = reinterpret_cast<float*>(ptr(state_b));
  impl->activation = reinterpret_cast<std::uint8_t*>(ptr(activation));
  impl->activation_scales = reinterpret_cast<float*>(ptr(activation_scales));
  impl->q = reinterpret_cast<std::uint16_t*>(ptr(q));
  impl->k = reinterpret_cast<std::uint16_t*>(ptr(k));
  impl->v = reinterpret_cast<std::uint16_t*>(ptr(v));
  impl->attention = reinterpret_cast<std::uint16_t*>(ptr(attention));
  impl->linear_a = reinterpret_cast<std::uint16_t*>(ptr(linear_a));
  impl->linear_b = reinterpret_cast<std::uint16_t*>(ptr(linear_b));
  impl->cutlass = ptr(cutlass);
  impl->final_output = reinterpret_cast<float*>(ptr(final_output));
  return Gemma4Moe26BVisionRuntime(std::move(impl));
}

Status Gemma4Moe26BVisionRuntime::Encode(
    const Gemma4Moe26BVisionInputSegment& segment, cudaStream_t stream,
    Gemma4Moe26BVisionTimingRecorder* timing,
    std::span<cudaEvent_t> phase_events) {
  if (!impl_ || stream == nullptr || segment.raw_patch_count == 0U ||
      (!phase_events.empty() && phase_events.size() != 4U) ||
      segment.raw_patch_count > impl_->maximum_raw_patches ||
      segment.soft_token_budget > impl_->maximum_soft_token_budget ||
      segment.soft_token_count == 0U ||
      segment.soft_token_count > impl_->maximum_soft_token_budget ||
      (segment.soft_token_budget != 70U &&
       segment.soft_token_budget != 140U &&
       segment.soft_token_budget != 280U) ||
      segment.soft_token_count > segment.soft_token_budget ||
      segment.raw_patch_count != segment.soft_token_count * 9U ||
      segment.patches.size() != static_cast<std::size_t>(segment.raw_patch_count) * kPatchElements ||
      segment.positions.size() != static_cast<std::size_t>(segment.raw_patch_count) * 2U) {
    return Invalid("invalid Gemma 4 26B Vision input segment");
  }
  auto grid = ValidateGemma4Moe26BVisionGrid(
      segment.raw_patch_count, segment.soft_token_count, segment.positions);
  if (!grid.ok()) return grid.status();
  const std::uint32_t grid_width = grid.value().width;
  const std::uint32_t tower_tokens = segment.soft_token_budget * 9U;
  cudaError_t error = cudaStreamSynchronize(stream);
  if (error != cudaSuccess) return CudaFailure("synchronize Gemma 4 26B Vision input staging", error);
  std::copy(segment.patches.begin(), segment.patches.end(), impl_->host_patches);
  std::fill_n(
      impl_->host_patches + segment.patches.size(),
      static_cast<std::size_t>(tower_tokens - segment.raw_patch_count) *
          kPatchElements,
      0.0F);
  std::copy(segment.positions.begin(), segment.positions.end(), impl_->host_positions);
  std::fill_n(
      impl_->host_positions + segment.positions.size(),
      static_cast<std::size_t>(tower_tokens - segment.raw_patch_count) * 2U,
      -1);
  const std::uint64_t patch_elements =
      static_cast<std::uint64_t>(tower_tokens) * kPatchElements;
  Status status = timing == nullptr ? Status::Ok()
                                    : timing->Begin(stream, segment);
  if (!status.ok()) return status;
  if (!phase_events.empty()) {
    error = cudaEventRecord(phase_events[0], stream);
    if (error != cudaSuccess) return CudaFailure("record Vision phase begin", error);
  }
  {
    const NvtxRange range("gem16.vision.upload");
    error = cudaMemcpyAsync(
        impl_->device_patches, impl_->host_patches,
        patch_elements * sizeof(float), cudaMemcpyHostToDevice, stream);
    if (error == cudaSuccess) {
      error = cudaMemcpyAsync(impl_->positions, impl_->host_positions,
                              static_cast<std::size_t>(tower_tokens) * 2U *
                                  sizeof(std::int32_t),
                              cudaMemcpyHostToDevice, stream);
    }
  }
  if (error != cudaSuccess) return CudaFailure("upload Gemma 4 26B Vision input", error);
  if (!phase_events.empty()) {
    error = cudaEventRecord(phase_events[1], stream);
    if (error != cudaSuccess) return CudaFailure("record Vision upload phase", error);
  }
  if (timing != nullptr) {
    status = timing->Boundary(stream);
    if (!status.ok()) return status;
  }
  {
    const NvtxRange range("gem16.vision.patch_project");
    PreparePatchesKernel<<<static_cast<unsigned>(Blocks(patch_elements)), 256U,
                           0, stream>>>(impl_->device_patches, impl_->state_a,
                                       patch_elements);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
      return CudaFailure("prepare Vision patches", error);
    }
    status = LaunchFp8ReferenceTokenQuantizationBatch(
        impl_->state_a, impl_->activation, impl_->activation_scales,
        tower_tokens, kPatchElements, stream);
    if (!status.ok()) return status;
    status = LaunchFp8CutlassProjectionBatch(
        impl_->activation, impl_->activation_scales, impl_->patch.weight,
        impl_->patch.scale, impl_->q, tower_tokens, kHidden, kPatchElements,
        impl_->cutlass, kCutlassWorkspaceBytes, stream);
  }
  if (!status.ok()) return status;
  if (timing != nullptr) {
    status = timing->Boundary(stream);
    if (!status.ok()) return status;
  }
  const std::uint64_t hidden_elements =
      static_cast<std::uint64_t>(tower_tokens) * kHidden;
  if (impl_->rope_table_enabled) {
    const std::uint64_t rope_entries =
        static_cast<std::uint64_t>(tower_tokens) * kRopeEntriesPerToken;
    const NvtxRange range("gem16.vision.rope_table");
    BuildRopeTableKernel<<<static_cast<unsigned>(Blocks(rope_entries)), 256U,
                           0, stream>>>(impl_->positions, impl_->rope_table,
                                       tower_tokens);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
      return CudaFailure("build Vision RoPE table", error);
    }
  }
  {
    const NvtxRange range("gem16.vision.position_add");
    AddPositionEmbeddingKernel<<<static_cast<unsigned>(Blocks(hidden_elements)),
                                 256U, 0, stream>>>(
        impl_->q, impl_->positions, impl_->position_table, impl_->state_a,
        hidden_elements);
    error = cudaGetLastError();
  }
  if (error != cudaSuccess) return CudaFailure("add Vision position embeddings", error);
  if (timing != nullptr) {
    status = timing->Boundary(stream);
    if (!status.ok()) return status;
  }

  constexpr unsigned kWarpThreads = 256U;
  const std::uint64_t head_rows =
      static_cast<std::uint64_t>(tower_tokens) * kHeads;
  const unsigned head_blocks = static_cast<unsigned>((head_rows + kWarpThreads / 32U - 1U) / (kWarpThreads / 32U));
  auto mark_timing_boundary = [&]() -> Status {
    return timing == nullptr ? Status::Ok() : timing->Boundary(stream);
  };
  for (std::uint32_t layer_index = 0U; layer_index < kLayers; ++layer_index) {
    const Layer& layer = impl_->layers[layer_index];
    {
      const NvtxRange range("gem16.vision.layer.input_norm_quant");
      status = LaunchRmsNormFp8TokenQuantizationBatch(
          impl_->state_a, layer.input_norm, impl_->activation,
          impl_->activation_scales, tower_tokens, kHidden, kEpsilon, stream);
    }
    if (!status.ok()) return status;
    status = mark_timing_boundary();
    if (!status.ok()) return status;
    {
      const NvtxRange range("gem16.vision.layer.qkv_projection");
      for (const auto& [linear, output] :
           std::array<std::pair<const Linear*, std::uint16_t*>, 3>{
               std::pair{&layer.q, impl_->q}, std::pair{&layer.k, impl_->k},
               std::pair{&layer.v, impl_->v}}) {
        status = LaunchFp8CutlassProjectionBatch(
            impl_->activation, impl_->activation_scales, linear->weight,
            linear->scale, output, tower_tokens, linear->rows,
            linear->columns, impl_->cutlass, kCutlassWorkspaceBytes, stream);
        if (!status.ok()) return status;
      }
    }
    status = mark_timing_boundary();
    if (!status.ok()) return status;
    {
      const NvtxRange range("gem16.vision.layer.qkv_norm_rope");
      if (impl_->rope_table_enabled) {
        HeadNormRopeKernel<true, true>
            <<<head_blocks, kWarpThreads, 0, stream>>>(
                impl_->q, layer.q_norm, impl_->positions, impl_->rope_table,
                impl_->q, head_rows);
        HeadNormRopeKernel<true, true>
            <<<head_blocks, kWarpThreads, 0, stream>>>(
                impl_->k, layer.k_norm, impl_->positions, impl_->rope_table,
                impl_->k, head_rows);
      } else {
        HeadNormRopeKernel<true, false>
            <<<head_blocks, kWarpThreads, 0, stream>>>(
                impl_->q, layer.q_norm, impl_->positions, nullptr, impl_->q,
                head_rows);
        HeadNormRopeKernel<true, false>
            <<<head_blocks, kWarpThreads, 0, stream>>>(
                impl_->k, layer.k_norm, impl_->positions, nullptr, impl_->k,
                head_rows);
      }
      HeadNormRopeKernel<false, false>
          <<<head_blocks, kWarpThreads, 0, stream>>>(
              impl_->v, nullptr, impl_->positions, nullptr, impl_->v,
              head_rows);
      error = cudaGetLastError();
    }
    if (error != cudaSuccess) return CudaFailure("launch Vision QKV norm/RoPE", error);
    status = mark_timing_boundary();
    if (!status.ok()) return status;
    {
      const NvtxRange range("gem16.vision.layer.attention");
      FullAttentionKernel<<<head_blocks, kWarpThreads, 0, stream>>>(
          impl_->q, impl_->k, impl_->v, impl_->attention, tower_tokens,
          segment.raw_patch_count);
      error = cudaGetLastError();
    }
    if (error != cudaSuccess) return CudaFailure("launch Vision full attention", error);
    status = mark_timing_boundary();
    if (!status.ok()) return status;
    {
      const NvtxRange range("gem16.vision.layer.output_projection_residual");
      status = LaunchFp8ReferenceTokenQuantizationBf16Batch(
          impl_->attention, impl_->activation, impl_->activation_scales,
          tower_tokens, kHidden, stream);
      if (!status.ok()) return status;
      status = LaunchFp8CutlassProjectionBatch(
          impl_->activation, impl_->activation_scales, layer.o.weight,
          layer.o.scale, impl_->q, tower_tokens, kHidden, kHidden,
          impl_->cutlass, kCutlassWorkspaceBytes, stream);
      if (!status.ok()) return status;
      status = LaunchRmsNormResidualBf16Input(
          impl_->q, layer.post_attention_norm, impl_->state_a, nullptr,
          impl_->state_b, tower_tokens, kHidden, kEpsilon, nullptr, stream);
    }
    if (!status.ok()) return status;
    status = mark_timing_boundary();
    if (!status.ok()) return status;
    {
      const NvtxRange range("gem16.vision.layer.ffn_norm_quant");
      status = LaunchRmsNormFp8TokenQuantizationBatch(
          impl_->state_b, layer.pre_feedforward_norm, impl_->activation,
          impl_->activation_scales, tower_tokens, kHidden, kEpsilon, stream);
    }
    if (!status.ok()) return status;
    status = mark_timing_boundary();
    if (!status.ok()) return status;
    {
      const NvtxRange range("gem16.vision.layer.gate_up");
      status = LaunchFp8CutlassProjectionBatch(
          impl_->activation, impl_->activation_scales, layer.gate.weight,
          layer.gate.scale, impl_->linear_a, tower_tokens, kIntermediate,
          kHidden, impl_->cutlass, kCutlassWorkspaceBytes, stream);
      if (!status.ok()) return status;
      status = LaunchFp8CutlassProjectionBatch(
          impl_->activation, impl_->activation_scales, layer.up.weight,
          layer.up.scale, impl_->linear_b, tower_tokens, kIntermediate, kHidden,
          impl_->cutlass, kCutlassWorkspaceBytes, stream);
    }
    if (!status.ok()) return status;
    status = mark_timing_boundary();
    if (!status.ok()) return status;
    const std::uint64_t intermediate_elements =
        static_cast<std::uint64_t>(tower_tokens) * kIntermediate;
    if (impl_->fused_ffn_quant) {
      {
        const NvtxRange range("gem16.vision.layer.gelu_product_quant");
        status = LaunchGemma4Moe26BVisionFusedGeluProductQuantization(
            impl_->linear_a, impl_->linear_b, impl_->activation,
            impl_->activation_scales, tower_tokens, stream);
      }
      if (!status.ok()) return status;
      status = mark_timing_boundary();
      if (!status.ok()) return status;
      status = mark_timing_boundary();
      if (!status.ok()) return status;
    } else {
      {
        const NvtxRange range("gem16.vision.layer.gelu");
        GeluProductKernel<<<
            static_cast<unsigned>(Blocks(intermediate_elements)), 256U, 0,
            stream>>>(impl_->linear_a, impl_->linear_b, impl_->linear_a,
                      intermediate_elements);
        error = cudaGetLastError();
      }
      if (error != cudaSuccess) {
        return CudaFailure("launch Vision GELU product", error);
      }
      status = mark_timing_boundary();
      if (!status.ok()) return status;
      {
        const NvtxRange range("gem16.vision.layer.product_quant");
        status = LaunchFp8ReferenceTokenQuantizationBf16Batch(
            impl_->linear_a, impl_->activation, impl_->activation_scales,
            tower_tokens, kIntermediate, stream);
      }
      if (!status.ok()) return status;
      status = mark_timing_boundary();
      if (!status.ok()) return status;
    }
    {
      const NvtxRange range("gem16.vision.layer.down_residual");
      status = LaunchFp8CutlassProjectionBatch(
          impl_->activation, impl_->activation_scales, layer.down.weight,
          layer.down.scale, impl_->q, tower_tokens, kHidden, kIntermediate,
          impl_->cutlass, kCutlassWorkspaceBytes, stream);
      if (!status.ok()) return status;
      status = LaunchRmsNormResidualBf16Input(
          impl_->q, layer.post_feedforward_norm, impl_->state_b, nullptr,
          impl_->state_a, tower_tokens, kHidden, kEpsilon, nullptr, stream);
    }
    if (!status.ok()) return status;
    status = mark_timing_boundary();
    if (!status.ok()) return status;
  }

  if (!phase_events.empty()) {
    error = cudaEventRecord(phase_events[2], stream);
    if (error != cudaSuccess) return CudaFailure("record Vision tower phase", error);
  }
  const std::uint64_t pooled_elements = static_cast<std::uint64_t>(segment.soft_token_count) * kHidden;
  {
    const NvtxRange range("gem16.vision.pool_standardize");
    if (impl_->direct_pool) {
      PoolStandardizeDirectKernel<<<
          static_cast<unsigned>(Blocks(pooled_elements)), 256U, 0, stream>>>(
          impl_->state_a, impl_->std_bias, impl_->std_scale, impl_->state_b,
          segment.soft_token_count, grid_width / 3U);
    } else {
      PoolStandardizeKernel<<<static_cast<unsigned>(Blocks(pooled_elements)),
                              256U, 0, stream>>>(
          impl_->state_a, impl_->positions, impl_->std_bias, impl_->std_scale,
          impl_->state_b, tower_tokens, segment.soft_token_count,
          grid_width / 3U);
    }
    error = cudaGetLastError();
  }
  if (error != cudaSuccess) return CudaFailure("launch Vision pooling and standardization", error);
  status = mark_timing_boundary();
  if (!status.ok()) return status;
  {
    const NvtxRange range("gem16.vision.final_norm_project");
    status = LaunchRmsNormFp8TokenQuantizationBatch(
        impl_->state_b, nullptr, impl_->activation, impl_->activation_scales,
        segment.soft_token_count, kHidden, kEpsilon, stream);
    if (!status.ok()) return status;
    status = LaunchFp8CutlassProjectionBatch(
        impl_->activation, impl_->activation_scales, impl_->projector.weight,
        impl_->projector.scale, impl_->linear_a, segment.soft_token_count,
        kTextHidden, kHidden, impl_->cutlass, kCutlassWorkspaceBytes, stream);
    if (!status.ok()) return status;
    const std::uint64_t output_elements =
        static_cast<std::uint64_t>(segment.soft_token_count) * kTextHidden;
    ExpandBf16Kernel<<<static_cast<unsigned>(Blocks(output_elements)), 256U, 0,
                       stream>>>(impl_->linear_a, impl_->final_output,
                                 output_elements);
    error = cudaGetLastError();
  }
  if (error != cudaSuccess) return CudaFailure("materialize Vision text embeddings", error);
  status = mark_timing_boundary();
  if (!status.ok()) return status;
  impl_->output_tokens = segment.soft_token_count;
  if (!phase_events.empty()) {
    error = cudaEventRecord(phase_events[3], stream);
    if (error != cudaSuccess) return CudaFailure("record Vision pool/project phase", error);
  }
  return Status::Ok();
}

const float* Gemma4Moe26BVisionRuntime::output() const {
  return impl_ == nullptr ? nullptr : impl_->final_output;
}

std::uint32_t Gemma4Moe26BVisionRuntime::output_tokens() const {
  return impl_ == nullptr ? 0U : impl_->output_tokens;
}

std::uint64_t Gemma4Moe26BVisionRuntime::workspace_bytes() const {
  return impl_ == nullptr ? 0U : impl_->allocation_bytes;
}

std::uint64_t Gemma4Moe26BVisionRuntime::host_pinned_bytes() const {
  return impl_ == nullptr
             ? 0U
             : static_cast<std::uint64_t>(impl_->maximum_raw_patches) *
                       kPatchElements * sizeof(float) +
                   static_cast<std::uint64_t>(impl_->maximum_raw_patches) * 2U *
                       sizeof(std::int32_t);
}

std::uint32_t Gemma4Moe26BVisionRuntime::maximum_soft_token_budget() const {
  return impl_ == nullptr ? 0U : impl_->maximum_soft_token_budget;
}

}  // namespace gem16::internal
