#include "gem16/engine.h"

#include "cuda/attention/sm120.h"
#include "cuda/engine/target_model.h"
#include "cuda/fp8/cutlass_sm120.h"
#include "cuda/fp8/reference.h"
#include "cuda/fp8/sm120.h"
#include "cuda/layer/reference.h"
#include "cuda/mtp/assistant.h"
#include "cuda/mtp/scheduler.h"
#include "cuda/mtp/verify.h"
#include "cuda/nvfp4/mlp.h"
#include "cuda/nvfp4/reference.h"
#include "cuda/nvfp4/cutlass_sm120.h"
#include "cuda/nvfp4/sm120.h"
#include "cuda/nvfp4/sm120_layout.h"
#include "cuda/output_head.h"
#include "cuda/sampling/sampling.h"
#include "gem16/model.h"
#include "platform/mapped_file.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <numeric>
#include <ostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace gem16 {
namespace {

constexpr std::uint64_t kHidden = 3840;
constexpr std::uint64_t kIntermediate = 15360;
constexpr std::uint64_t kVocabulary = 262144;
constexpr std::uint64_t kQueryHeads = 16;
constexpr std::uint64_t kLayers = 48;
constexpr std::uint64_t kSlidingWindow = 1024;
constexpr std::uint64_t kMaximumContext = 262144;
constexpr std::uint64_t kAlignment = 256;
constexpr std::uint64_t kMaximumSuppressedTokens = 16;
constexpr std::uint64_t kRepetitionMaskWords = (kVocabulary + 31U) / 32U;
constexpr float kEpsilon = 1.0e-6F;
constexpr unsigned kThreads = 256;
constexpr unsigned kFusedOutputHeadBlocks =
    internal::kOutputHeadCandidateBlocks;
using ArgmaxValue = internal::OutputHeadCandidate;
constexpr std::uint64_t kMaximumMtpDraftTokens = 4U;
constexpr std::uint64_t kMaximumMtpVerifyTokens =
    kMaximumMtpDraftTokens + 1U;
constexpr std::uint64_t kDefaultPrefillChunkTokens = 2048;
constexpr std::uint64_t kMinimumPrefillChunkTokens = 32;
constexpr std::uint64_t kPrefillChunkQuantum = 32;
constexpr std::uint64_t kPrefillScoreBudgetBytes = 512ULL * 1024ULL * 1024ULL;

class NvtxRange {
 public:
  explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
  NvtxRange(const NvtxRange&) = delete;
  NvtxRange& operator=(const NvtxRange&) = delete;
  ~NvtxRange() { nvtxRangePop(); }
};

Status Error(StatusCode code, std::string message) {
  return Status(code, std::move(message));
}

Status CudaFailure(const char* operation, cudaError_t error) {
  return Error(StatusCode::kInternal,
               std::string(operation) + ": " + cudaGetErrorName(error) + ": " +
                   cudaGetErrorString(error));
}

std::uint64_t PrefillChunkTokensForContext(
    std::uint64_t max_context, KvCacheMode kv_cache_mode) {
  if (kv_cache_mode == KvCacheMode::kCheckpointFp8) {
    return kDefaultPrefillChunkTokens;
  }
  const std::uint64_t score_bytes_per_token =
      kQueryHeads * max_context * sizeof(float);
  const std::uint64_t budget_tokens =
      score_bytes_per_token == 0U ? kDefaultPrefillChunkTokens
                                  : kPrefillScoreBudgetBytes / score_bytes_per_token;
  const std::uint64_t bounded =
      std::min({kDefaultPrefillChunkTokens, kSlidingWindow, budget_tokens});
  const std::uint64_t quantized =
      (bounded / kPrefillChunkQuantum) * kPrefillChunkQuantum;
  return std::max(kMinimumPrefillChunkTokens, quantized);
}

Result<std::uint64_t> AlignUp(std::uint64_t value, std::uint64_t alignment) {
  if (alignment == 0U || (alignment & (alignment - 1U)) != 0U) {
    return Error(StatusCode::kInternal, "arena alignment is not a power of two");
  }
  const std::uint64_t mask = alignment - 1U;
  if (value > std::numeric_limits<std::uint64_t>::max() - mask) {
    return Error(StatusCode::kInternal, "arena offset overflow");
  }
  return (value + mask) & ~mask;
}

class DeviceAllocation {
 public:
  DeviceAllocation() = default;
  DeviceAllocation(const DeviceAllocation&) = delete;
  DeviceAllocation& operator=(const DeviceAllocation&) = delete;
  ~DeviceAllocation() {
    if (data_ != nullptr) (void)cudaFree(data_);
  }

  [[nodiscard]] Status Allocate(std::uint64_t bytes, const char* label) {
    if (data_ != nullptr || bytes == 0U || bytes > std::numeric_limits<std::size_t>::max()) {
      return Error(StatusCode::kInvalidArgument, std::string(label) + " size is invalid");
    }
    const cudaError_t error = cudaMalloc(&data_, static_cast<std::size_t>(bytes));
    if (error != cudaSuccess) return CudaFailure(label, error);
    bytes_ = bytes;
    return Status::Ok();
  }

  [[nodiscard]] std::byte* data() const { return static_cast<std::byte*>(data_); }
  [[nodiscard]] std::uint64_t bytes() const { return bytes_; }

 private:
  void* data_ = nullptr;
  std::uint64_t bytes_ = 0;
};

class GraphExecutable {
 public:
  GraphExecutable() = default;
  GraphExecutable(const GraphExecutable&) = delete;
  GraphExecutable& operator=(const GraphExecutable&) = delete;
  ~GraphExecutable() {
    if (executable_ != nullptr) (void)cudaGraphExecDestroy(executable_);
  }

  [[nodiscard]] cudaGraphExec_t get() const { return executable_; }
  void Adopt(cudaGraphExec_t executable) { executable_ = executable; }

 private:
  cudaGraphExec_t executable_ = nullptr;
};

class PinnedHostAllocation {
 public:
  PinnedHostAllocation() = default;
  PinnedHostAllocation(const PinnedHostAllocation&) = delete;
  PinnedHostAllocation& operator=(const PinnedHostAllocation&) = delete;
  ~PinnedHostAllocation() {
    if (data_ != nullptr) (void)cudaFreeHost(data_);
  }

  [[nodiscard]] Status Allocate(std::size_t elements, const char* label) {
    if (data_ != nullptr || elements == 0U ||
        elements > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
      return Error(StatusCode::kInvalidArgument,
                   std::string("pinned ") + label + " size is invalid");
    }
    const cudaError_t error =
        cudaHostAlloc(&data_, elements * sizeof(float), cudaHostAllocDefault);
    if (error != cudaSuccess) return CudaFailure(label, error);
    elements_ = elements;
    return Status::Ok();
  }

  [[nodiscard]] std::span<float> span() const {
    return {static_cast<float*>(data_), elements_};
  }

 private:
  void* data_ = nullptr;
  std::size_t elements_ = 0;
};

struct LayerStateCapture {
  std::size_t attention_context = 0;
  std::size_t attention_elements = 0;
  std::size_t attention_output = 0;
  std::size_t post_attention_norm = 0;
  std::size_t post_attention_residual = 0;
  std::size_t pre_feedforward_norm = 0;
  std::size_t gate = 0;
  std::size_t up = 0;
  std::size_t gelu_product = 0;
  std::size_t mlp_output = 0;
  std::size_t post_feedforward_norm = 0;
  std::size_t hidden = 0;
  std::size_t key = 0;
  std::size_t value = 0;
  std::size_t kv_elements = 0;
};

struct StateCaptureLayout {
  std::array<LayerStateCapture, kLayers> layers{};
  std::size_t elements = 0;
};

StateCaptureLayout MakeStateCaptureLayout() {
  StateCaptureLayout layout;
  for (std::size_t layer = 0; layer < kLayers; ++layer) {
    const bool global = layer % 6U == 5U;
    const std::size_t kv_elements =
        global ? 512U : static_cast<std::size_t>(8U * 256U);
    LayerStateCapture& capture = layout.layers[layer];
    capture.attention_context = layout.elements;
    capture.attention_elements =
        global ? static_cast<std::size_t>(16U * 512U)
               : static_cast<std::size_t>(16U * 256U);
    capture.attention_output =
        capture.attention_context + capture.attention_elements;
    capture.post_attention_norm = capture.attention_output + kHidden;
    capture.post_attention_residual = capture.post_attention_norm + kHidden;
    capture.pre_feedforward_norm = capture.post_attention_residual + kHidden;
    capture.gate = capture.pre_feedforward_norm + kHidden;
    capture.up = capture.gate + kIntermediate;
    capture.gelu_product = capture.up + kIntermediate;
    capture.mlp_output = capture.gelu_product + kIntermediate;
    capture.post_feedforward_norm = capture.mlp_output + kHidden;
    capture.hidden = capture.post_feedforward_norm + kHidden;
    capture.key = capture.hidden + kHidden;
    capture.value = capture.key + kv_elements;
    capture.kv_elements = kv_elements;
    layout.elements = capture.value + kv_elements;
  }
  return layout;
}

using internal::Fp8Binding;
using internal::LayerBinding;
using internal::Nvfp4Binding;

struct WorkspaceOffsets {
  std::uint64_t decode_control = 0;
  std::uint64_t hidden_a = 0;
  std::uint64_t hidden_b = 0;
  std::uint64_t normalized = 0;
  std::uint64_t fp8_activation = 0;
  std::uint64_t fp8_scale = 0;
  std::uint64_t q = 0;
  std::uint64_t k = 0;
  std::uint64_t v = 0;
  std::uint64_t q_norm = 0;
  std::uint64_t k_norm = 0;
  std::uint64_t v_norm = 0;
  std::uint64_t scores = 0;
  std::uint64_t attention = 0;
  std::uint64_t o_activation = 0;
  std::uint64_t o_scale = 0;
  std::uint64_t projection = 0;
  std::uint64_t post_norm = 0;
  std::uint64_t mlp_packed = 0;
  std::uint64_t mlp_scales = 0;
  std::uint64_t gate = 0;
  std::uint64_t up = 0;
  std::uint64_t product = 0;
  std::uint64_t down_packed = 0;
  std::uint64_t down_scales = 0;
  std::uint64_t logits = 0;
  std::uint64_t sampling_logits = 0;
  std::uint64_t sampling_cumulative = 0;
  std::uint64_t sampling_token_ids = 0;
  std::uint64_t sorted_token_ids = 0;
  std::uint64_t sampling_sort_workspace = 0;
  std::uint64_t repetition_mask = 0;
  std::uint64_t output_candidates = 0;
  std::uint64_t selected = 0;
  std::uint64_t suppressed = 0;
  std::uint64_t total = 0;
};

struct MtpWorkspaceOffsets {
  struct LayerKv {
    std::uint64_t key = 0;
    std::uint64_t value = 0;
    std::uint64_t backup_key = 0;
    std::uint64_t backup_value = 0;
  };
  std::array<LayerKv, kLayers> layers{};
  std::uint64_t output_candidates = 0;
  std::uint64_t selected = 0;
  std::uint64_t stop_tokens = 0;
  std::uint64_t group_result = 0;
  std::uint64_t committed_hidden = 0;
  std::uint64_t total = 0;
};

using MtpGroupResult = internal::MtpGroupResult;
static_assert(kMaximumMtpDraftTokens == internal::kMaximumMtpDraftTokens);
static_assert(kMaximumMtpVerifyTokens == internal::kMaximumMtpVerifyTokens);

struct HostDecodeState {
  internal::DecodeControl control{};
  std::uint32_t selected_token = 0;
};

struct PrefillOffsets {
  std::uint64_t token_ids = 0;
  std::uint64_t hidden_a = 0, hidden_b = 0, normalized = 0;
  std::uint64_t fp8_activation = 0, fp8_scales = 0;
  std::uint64_t q = 0, k = 0, v = 0, q_norm = 0, k_norm = 0, v_norm = 0;
  std::uint64_t k_fp8 = 0, v_fp8 = 0, scores = 0, attention = 0;
  std::uint64_t o_activation = 0, o_scales = 0, projection = 0, post_norm = 0;
  std::uint64_t mlp_packed = 0, mlp_scales = 0, gate = 0, up = 0;
  std::uint64_t down_packed = 0, down_scales = 0;
  std::uint64_t cutlass_activation_scales = 0;
  std::uint64_t cutlass_weight = 0, cutlass_weight_scales = 0;
  std::uint64_t cutlass_workspace = 0;
  std::uint64_t local_rope_cosine = 0, local_rope_sine = 0;
  std::uint64_t global_rope_cosine = 0, global_rope_sine = 0;
};

class LayoutBuilder {
 public:
  template <typename T>
  [[nodiscard]] Result<std::uint64_t> Add(std::uint64_t elements) {
    auto aligned = AlignUp(offset_, std::max<std::uint64_t>(alignof(T), 16U));
    if (!aligned.ok()) return aligned.status();
    if (elements > std::numeric_limits<std::uint64_t>::max() / sizeof(T) ||
        elements * sizeof(T) > std::numeric_limits<std::uint64_t>::max() - aligned.value()) {
      return Error(StatusCode::kInternal, "workspace size overflow");
    }
    offset_ = aligned.value() + elements * sizeof(T);
    return aligned.value();
  }
  [[nodiscard]] std::uint64_t size() const { return offset_; }

 private:
  std::uint64_t offset_ = 0;
};

template <typename T>
T* Pointer(DeviceAllocation& arena, std::uint64_t offset) {
  return reinterpret_cast<T*>(arena.data() + offset);
}

__global__ void RoundBf16Kernel(float* values, std::uint64_t elements) {
  const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) values[index] = static_cast<float>(__float2bfloat16_rn(values[index]));
}

__global__ void EmbeddingKernel(const std::uint16_t* weights, std::uint32_t token, float* output) {
  const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= kHidden) return;
  const float weight = static_cast<float>(__ushort_as_bfloat16(weights[
      static_cast<std::uint64_t>(token) * kHidden + index]));
  const float scale = static_cast<float>(__float2bfloat16_rn(sqrtf(static_cast<float>(kHidden))));
  output[index] = static_cast<float>(__float2bfloat16_rn(weight * scale));
}

__global__ void ControlledEmbeddingKernel(
    const std::uint16_t* weights, const internal::DecodeControl* control,
    float* output) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= kHidden) return;
  const float weight = static_cast<float>(__ushort_as_bfloat16(
      weights[static_cast<std::uint64_t>(control->token) * kHidden + index]));
  const float scale = static_cast<float>(
      __float2bfloat16_rn(sqrtf(static_cast<float>(kHidden))));
  output[index] = static_cast<float>(__float2bfloat16_rn(weight * scale));
}

__global__ void EmbeddingBatchKernel(const std::uint16_t* weights,
                                     const std::uint32_t* tokens, float* output,
                                     std::uint64_t total_elements) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total_elements) return;
  const std::uint64_t token_index = index / kHidden;
  const std::uint64_t hidden_index = index % kHidden;
  const float weight = static_cast<float>(__ushort_as_bfloat16(
      weights[static_cast<std::uint64_t>(tokens[token_index]) * kHidden + hidden_index]));
  const float scale = static_cast<float>(
      __float2bfloat16_rn(sqrtf(static_cast<float>(kHidden))));
  output[index] = static_cast<float>(__float2bfloat16_rn(weight * scale));
}

Status LaunchRoundBf16(float* values, std::uint64_t elements, cudaStream_t stream) {
  const std::uint64_t blocks = (elements + kThreads - 1U) / kThreads;
  RoundBf16Kernel<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(values, elements);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess ? Status::Ok() : CudaFailure("launch BF16 rounding", error);
}

Status LaunchFp8Projection(const std::uint8_t* activation, const float* scale,
                           const Fp8Binding& binding, float* output,
                           cudaStream_t stream) {
  return internal::LaunchFp8Sm120DirectProjection(
      activation, scale, binding.weight, binding.scales, output, binding.rows,
      binding.contracting, stream);
}

Status LaunchFp8QkvProjection(
    const std::uint8_t* activation, const float* scale,
    const Fp8Binding& q_binding, float* q_output,
    const Fp8Binding& k_binding, float* k_output,
    const Fp8Binding* v_binding, float* v_output, cudaStream_t stream) {
  if (q_binding.contracting != k_binding.contracting ||
      (v_binding != nullptr &&
       q_binding.contracting != v_binding->contracting)) {
    return Status(StatusCode::kInvalidArgument,
                  "grouped decode FP8 Q/K/V projections require one "
                  "contracting dimension");
  }
  return internal::LaunchFp8Sm120GroupedQkvProjection(
      activation, scale, q_binding.weight, q_binding.scales, q_output,
      q_binding.rows, k_binding.weight, k_binding.scales, k_output,
      k_binding.rows, v_binding == nullptr ? nullptr : v_binding->weight,
      v_binding == nullptr ? nullptr : v_binding->scales,
      v_binding == nullptr ? nullptr : v_output,
      v_binding == nullptr ? 0U : v_binding->rows, q_binding.contracting,
      stream);
}

Status LaunchNvfp4Projection(const std::uint8_t* activation, const std::uint8_t* scales,
                             const Nvfp4Binding& binding, float* output,
                             cudaStream_t stream) {
  return internal::LaunchNvfp4Sm120DirectProjection(
      activation, scales, binding.packed_weight, binding.scales, output, binding.rows,
      binding.contracting, binding.input_divisor, binding.weight_divisor, stream);
}

Status LaunchFp8ProjectionBatch(const std::uint8_t* activation, const float* scales,
                                const Fp8Binding& binding, float* output,
                                std::uint64_t tokens, void* cutlass_workspace,
                                std::size_t cutlass_workspace_bytes,
                                cudaStream_t stream) {
  return internal::LaunchFp8CutlassProjectionBatch(
      activation, scales, binding.weight, binding.scales, output, tokens,
      binding.rows, binding.contracting, cutlass_workspace,
      cutlass_workspace_bytes, stream);
}

Status LaunchFp8QkvProjectionBatch(
    const std::uint8_t* activation, const float* scales,
    const Fp8Binding& q_binding, float* q_output,
    const Fp8Binding& k_binding, float* k_output,
    const Fp8Binding* v_binding, float* v_output, std::uint64_t tokens,
    void* cutlass_workspace, std::size_t cutlass_workspace_bytes,
    cudaStream_t stream) {
  if (q_binding.contracting != k_binding.contracting ||
      (v_binding != nullptr &&
       q_binding.contracting != v_binding->contracting)) {
    return Status(StatusCode::kInvalidArgument,
                  "grouped FP8 Q/K/V projections require one contracting dimension");
  }
  Status status = internal::LaunchFp8CutlassProjectionBatch(
      activation, scales, q_binding.weight, q_binding.scales, q_output, tokens,
      q_binding.rows, q_binding.contracting, cutlass_workspace,
      cutlass_workspace_bytes, stream);
  if (!status.ok()) return status;
  status = internal::LaunchFp8CutlassProjectionBatch(
      activation, scales, k_binding.weight, k_binding.scales, k_output, tokens,
      k_binding.rows, k_binding.contracting, cutlass_workspace,
      cutlass_workspace_bytes, stream);
  if (!status.ok() || v_binding == nullptr) return status;
  return internal::LaunchFp8CutlassProjectionBatch(
      activation, scales, v_binding->weight, v_binding->scales, v_output,
      tokens, v_binding->rows, v_binding->contracting, cutlass_workspace,
      cutlass_workspace_bytes, stream);
}

class InferenceEngine {
 public:
  InferenceEngine() = default;
  InferenceEngine(const InferenceEngine&) = delete;
  InferenceEngine& operator=(const InferenceEngine&) = delete;
  ~InferenceEngine() {
    if (stream_ != nullptr) (void)cudaStreamDestroy(stream_);
  }

  [[nodiscard]] Status Initialize(
      const std::filesystem::path& model_directory,
      std::uint64_t max_context, KvCacheMode kv_cache_mode,
      const SamplingOptions& sampling = {},
      const std::filesystem::path& assistant_model_directory = {},
      std::uint32_t mtp_draft_tokens = 0U) {
    const NvtxRange range("gem16.initialize");
    Status sampling_status = SetSampling(sampling);
    if (!sampling_status.ok()) return sampling_status;
    kv_cache_mode_ = kv_cache_mode;
    max_context_ = max_context;
    prefill_chunk_tokens_ =
        PrefillChunkTokensForContext(max_context_, kv_cache_mode_);
    cudaDeviceProp properties{};
    cudaError_t error = cudaGetDeviceProperties(&properties, 0);
    if (error != cudaSuccess) return CudaFailure("cudaGetDeviceProperties", error);
    if (properties.major != 12 || properties.minor != 0) {
      return Error(StatusCode::kUnsupported, "greedy characterization requires SM120");
    }
    error = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
    if (error != cudaSuccess) return CudaFailure("create inference stream", error);

    Status status = model_.Load(model_directory);
    if (!status.ok()) return status;
    if (!assistant_model_directory.empty()) {
      const NvtxRange assistant_range("gem16.initialize.mtp_assistant");
      std::size_t free_before_assistant = 0U;
      std::size_t total_before_assistant = 0U;
      error = cudaMemGetInfo(&free_before_assistant, &total_before_assistant);
      if (error != cudaSuccess) {
        return CudaFailure("measure memory before assistant load", error);
      }
      status = assistant_.Load(assistant_model_directory);
      if (!status.ok()) return status;
      std::size_t free_after_assistant = 0U;
      std::size_t total_after_assistant = 0U;
      error = cudaMemGetInfo(&free_after_assistant, &total_after_assistant);
      if (error != cudaSuccess) {
        return CudaFailure("measure memory after assistant load", error);
      }
      if (total_before_assistant != total_after_assistant) {
        return Error(StatusCode::kInternal,
                     "device total memory changed during assistant load");
      }
      assistant_device_memory_delta_bytes_ =
          free_before_assistant > free_after_assistant
              ? free_before_assistant - free_after_assistant
              : 0U;
      if (mtp_draft_tokens != 0U) {
        status = assistant_.Prepare(max_context_);
        if (!status.ok()) return status;
      }
    }
    mtp_draft_tokens_ = mtp_draft_tokens;
    status = AllocateCache();
    if (!status.ok()) return status;
    status = AllocateWorkspace();
    if (!status.ok()) return status;
    status = AllocatePrefillWorkspace();
    if (!status.ok()) return status;
    if (mtp_draft_tokens_ != 0U) {
      status = AllocateMtpWorkspace();
      if (!status.ok()) return status;
      constexpr std::size_t kHostResultFloats =
          (sizeof(MtpGroupResult) + sizeof(float) - 1U) / sizeof(float);
      status = mtp_host_result_.Allocate(kHostResultFloats,
                                         "MTP group host result");
      if (!status.ok()) return status;
    }
    status = internal::LaunchRotaryEmbeddingTableBatch(
        Pointer<float>(prefill_workspace_, prefill_offsets_.local_rope_cosine),
        Pointer<float>(prefill_workspace_, prefill_offsets_.local_rope_sine),
        max_context_, 128U, 256U, 0U, 10000.0, 1.0, stream_);
    if (!status.ok()) return status;
    status = internal::LaunchRotaryEmbeddingTableBatch(
        Pointer<float>(prefill_workspace_, prefill_offsets_.global_rope_cosine),
        Pointer<float>(prefill_workspace_, prefill_offsets_.global_rope_sine),
        max_context_, 64U, 512U, 0U, 1000000.0, 1.0, stream_);
    if (!status.ok()) return status;
    error = cudaStreamSynchronize(stream_);
    if (error != cudaSuccess) {
      return CudaFailure("prepare persistent prefill RoPE tables", error);
    }
    status = decode_host_state_.Allocate(
        (sizeof(HostDecodeState) + sizeof(float) - 1U) / sizeof(float),
        "allocate decode graph host control");
    if (!status.ok()) return status;
    std::size_t free_before = 0U;
    std::size_t total_before = 0U;
    cudaError_t memory_error = cudaMemGetInfo(&free_before, &total_before);
    if (memory_error != cudaSuccess) {
      return CudaFailure("measure memory before decode graph capture",
                         memory_error);
    }
    status = PrepareDecodeGraphs();
    if (!status.ok()) return status;
    std::size_t free_after = 0U;
    std::size_t total_after = 0U;
    memory_error = cudaMemGetInfo(&free_after, &total_after);
    if (memory_error != cudaSuccess) {
      return CudaFailure("measure memory after decode graph capture",
                         memory_error);
    }
    if (total_before != total_after) {
      return Error(StatusCode::kInternal,
                   "device total memory changed during decode graph capture");
    }
    decode_graph_device_bytes_ =
        free_before > free_after ? free_before - free_after : 0U;
    return ResetCache();
  }

  [[nodiscard]] Result<std::uint32_t> Forward(
      std::uint32_t token, std::uint64_t position, bool select_token,
      std::span<float> host_logits = {}, std::span<float> host_state = {}) {
    const NvtxRange range("gem16.decode.forward");
    if (token >= kVocabulary || position >= max_context_) {
      return Error(StatusCode::kInvalidArgument, "token or position exceeds inference plan");
    }
    const StateCaptureLayout state_layout = MakeStateCaptureLayout();
    if (!host_state.empty() && host_state.size() != state_layout.elements) {
      return Error(StatusCode::kInternal, "host state capture span has invalid size");
    }
    if (select_token && host_logits.empty() && host_state.empty()) {
      HostDecodeState* host = host_decode_state();
      host->control.token = token;
      host->control.position = position;
      if (sampling_.enabled) host->control.sampling_step = sampling_step_++;
      const cudaError_t launch_error =
          cudaGraphLaunch(full_decode_graph_.get(), stream_);
      if (launch_error != cudaSuccess) {
        return CudaFailure("launch full decode graph", launch_error);
      }
      const cudaError_t sync_error = cudaStreamSynchronize(stream_);
      if (sync_error != cudaSuccess) {
        return CudaFailure("synchronize full decode graph", sync_error);
      }
      latest_target_hidden_ = Pointer<float>(workspace_, offsets_.normalized);
      return host->selected_token;
    }
    Status mark_status = MarkRepetitionToken(token);
    if (!mark_status.ok()) return mark_status;
    float* hidden_a = Pointer<float>(workspace_, offsets_.hidden_a);
    EmbeddingKernel<<<static_cast<unsigned>((kHidden + kThreads - 1U) / kThreads), kThreads,
                      0, stream_>>>(model_.embedding(), token, hidden_a);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return CudaFailure("launch embedding", error);

    for (std::size_t layer_index = 0; layer_index < model_.layers().size();
         ++layer_index) {
      const auto& layer = model_.layers()[layer_index];
      const LayerStateCapture* layer_capture =
          host_state.empty() ? nullptr : &state_layout.layers[layer_index];
      Status status =
          RunLayer(layer_index, layer, position, layer_capture, host_state.data());
      if (!status.ok()) return status;
    }
    float* normalized = Pointer<float>(workspace_, offsets_.normalized);
    Status status = internal::LaunchRmsNormBf16(hidden_a, model_.final_norm(), normalized, 1U,
                                            kHidden, kEpsilon, stream_);
    if (!status.ok()) return status;
    latest_target_hidden_ = normalized;
    if (!select_token) {
      if (!host_state.empty()) {
        error = cudaStreamSynchronize(stream_);
        if (error != cudaSuccess) {
          return CudaFailure("synchronize layer state capture", error);
        }
      }
      return 0U;
    }

    auto* selected = Pointer<std::uint32_t>(workspace_, offsets_.selected);
    float* diagnostic_logits = nullptr;
    if (!host_logits.empty()) {
      if (host_logits.size() != kVocabulary) {
        return Error(StatusCode::kInternal,
                     "host logit capture span has invalid size");
      }
      diagnostic_logits = Pointer<float>(workspace_, offsets_.logits);
    }
    if (sampling_.enabled) {
      diagnostic_logits = Pointer<float>(workspace_, offsets_.logits);
      status = internal::LaunchFusedOutputHeadCandidates(
          model_.embedding(), normalized,
          Pointer<std::uint32_t>(workspace_, offsets_.suppressed),
          suppressed_token_count_, nullptr,
          Pointer<ArgmaxValue>(workspace_, offsets_.output_candidates),
          diagnostic_logits, stream_);
      if (!status.ok()) return status;
      if (!host_logits.empty()) {
        error = cudaMemcpyAsync(host_logits.data(), diagnostic_logits,
                                host_logits.size_bytes(),
                                cudaMemcpyDeviceToHost, stream_);
        if (error != cudaSuccess) {
          return CudaFailure("copy sampled full logits", error);
        }
      }
      status = SelectSampledToken(diagnostic_logits, selected);
      if (!status.ok()) return status;
    } else {
      status = internal::LaunchFusedOutputHeadCandidates(
          model_.embedding(), normalized,
          Pointer<std::uint32_t>(workspace_, offsets_.suppressed),
          suppressed_token_count_, nullptr,
          Pointer<ArgmaxValue>(workspace_, offsets_.output_candidates),
          diagnostic_logits, stream_);
      if (!status.ok()) return status;
      if (diagnostic_logits != nullptr) {
        error = cudaMemcpyAsync(host_logits.data(), diagnostic_logits,
                                host_logits.size_bytes(),
                                cudaMemcpyDeviceToHost, stream_);
        if (error != cudaSuccess) {
          return CudaFailure("copy fused full logits", error);
        }
      }
      status = internal::LaunchOutputHeadCandidateArgmax(
          Pointer<ArgmaxValue>(workspace_, offsets_.output_candidates),
          selected, stream_);
      if (!status.ok()) return status;
    }
    std::uint32_t host_token = 0;
    error = cudaMemcpyAsync(&host_token, selected, sizeof(host_token), cudaMemcpyDeviceToHost,
                            stream_);
    if (error != cudaSuccess) return CudaFailure("copy selected token", error);
    error = cudaStreamSynchronize(stream_);
    if (error != cudaSuccess) return CudaFailure("synchronize selected token", error);
    return host_token;
  }

  [[nodiscard]] std::uint64_t weight_bytes() const { return model_.weight_bytes(); }
  [[nodiscard]] bool assistant_loaded() const { return assistant_.loaded(); }
  [[nodiscard]] std::uint64_t assistant_source_bytes() const {
    return assistant_.source_bytes();
  }
  [[nodiscard]] std::uint64_t assistant_weight_bytes() const {
    return assistant_.arena_bytes();
  }
  [[nodiscard]] std::uint64_t assistant_device_memory_delta_bytes() const {
    return assistant_device_memory_delta_bytes_;
  }
  [[nodiscard]] std::uint64_t assistant_tensor_count() const {
    return assistant_.tensor_count();
  }
  [[nodiscard]] std::uint64_t assistant_workspace_bytes() const {
    return assistant_.workspace_bytes() + mtp_workspace_.bytes();
  }
  [[nodiscard]] Status GenerateAssistantDraftsDevice(
      std::uint32_t input_token, std::uint64_t processed_position,
      std::uint32_t draft_count) {
    const NvtxRange range("gem16.mtp.propose");
    if (mtp_draft_tokens_ == 0U || !assistant_.prepared() ||
        latest_target_hidden_ == nullptr || draft_count == 0U ||
        draft_count > mtp_draft_tokens_ || processed_position >= max_context_) {
      return Error(StatusCode::kInvalidArgument,
                   "active MTP proposal state is invalid");
    }
    const auto make_view = [this, processed_position](
                               const LayerBinding& layer) {
      internal::AssistantSharedKvView view;
      view.mode = kv_cache_mode_ == KvCacheMode::kCheckpointFp8
                      ? internal::AssistantKvCacheMode::kCheckpointFp8
                      : internal::AssistantKvCacheMode::kBf16;
      view.key_fp8 = layer.key_cache_fp8;
      view.value_fp8 = layer.value_cache_fp8;
      view.key_bf16 = layer.key_cache_bf16;
      view.value_bf16 = layer.value_cache_bf16;
      view.key_scale_bf16 = layer.k_cache_scale;
      view.value_scale_bf16 = layer.v_cache_scale;
      view.capacity = layer.global
                          ? max_context_
                          : std::min(max_context_, kSlidingWindow);
      view.tokens = layer.global
                        ? processed_position + 1U
                        : std::min(processed_position + 1U, view.capacity);
      view.first_slot =
          layer.global || processed_position + 1U <= view.capacity
              ? 0U
              : (processed_position + 1U) % view.capacity;
      view.kv_heads = layer.kv_heads;
      view.head_dimension = layer.head_dimension;
      return view;
    };
    const LayerBinding& sliding = model_.layers()[46U];
    const LayerBinding& full = model_.layers()[47U];
    if (sliding.global || !full.global) {
      return Error(StatusCode::kInternal,
                   "target MTP shared-KV layer mapping is invalid");
    }
    internal::AssistantProposalContext context;
    context.target_embedding = model_.embedding();
    context.target_hidden = latest_target_hidden_;
    context.sliding_kv = make_view(sliding);
    context.full_kv = make_view(full);
    context.input_token = input_token;
    context.position = processed_position;
    return assistant_.GenerateDraftsDevice(context, draft_count, stream_);
  }

  [[nodiscard]] Status VerifyAcceptCommitAssistantBatch(
      std::uint32_t input_token, std::uint64_t start_position,
      std::uint32_t proposal_count, MtpGroupResult* host_result) {
    const NvtxRange range("gem16.mtp.verify_accept_commit");
    const std::uint64_t tokens = proposal_count + 1U;
    if (mtp_draft_tokens_ == 0U || proposal_count == 0U ||
        proposal_count > mtp_draft_tokens_ ||
        tokens > kMaximumMtpVerifyTokens || host_result == nullptr ||
        start_position >= max_context_ ||
        tokens > max_context_ - start_position) {
      return Error(StatusCode::kInvalidArgument,
                   "batched MTP verification extent is invalid");
    }
    const std::uint32_t* device_drafts = assistant_.device_draft_tokens();
    if (device_drafts == nullptr) {
      return Error(StatusCode::kInternal,
                   "assistant device draft storage is unavailable");
    }
    auto* device_tokens =
        Pointer<std::uint32_t>(prefill_workspace_, prefill_offsets_.token_ids);
    Status status = internal::LaunchBuildMtpVerificationInputs(
        input_token, device_drafts, proposal_count, device_tokens, stream_);
    if (!status.ok()) return status;
    cudaError_t error = cudaSuccess;
    float* hidden =
        Pointer<float>(prefill_workspace_, prefill_offsets_.hidden_a);
    const std::uint64_t hidden_elements = tokens * kHidden;
    EmbeddingBatchKernel<<<
        static_cast<unsigned>((hidden_elements + kThreads - 1U) / kThreads),
        kThreads, 0, stream_>>>(model_.embedding(), device_tokens, hidden,
                                hidden_elements);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
      return CudaFailure("launch MTP verification embedding", error);
    }
    for (std::size_t index = 0; index < model_.layers().size(); ++index) {
      status = RunLayerBatch(model_.layers()[index], start_position, tokens,
                             index);
      if (!status.ok()) return status;
    }
    float* normalized =
        Pointer<float>(prefill_workspace_, prefill_offsets_.normalized);
    status = internal::LaunchRmsNormBf16(
        hidden, model_.final_norm(), normalized, tokens, kHidden, kEpsilon,
        stream_);
    if (!status.ok()) return status;
    auto* candidates = Pointer<ArgmaxValue>(
        mtp_workspace_, mtp_offsets_.output_candidates);
    auto* selected =
        Pointer<std::uint32_t>(mtp_workspace_, mtp_offsets_.selected);
    status = internal::LaunchFusedOutputHeadBatchCandidates(
        model_.embedding(), normalized,
        Pointer<std::uint32_t>(workspace_, offsets_.suppressed),
        suppressed_token_count_, tokens, candidates, stream_);
    if (!status.ok()) return status;
    status = internal::LaunchOutputHeadBatchArgmax(candidates, tokens,
                                                   selected, stream_);
    if (!status.ok()) return status;
    auto* device_result = Pointer<MtpGroupResult>(
        mtp_workspace_, mtp_offsets_.group_result);
    status = internal::LaunchAcceptMtpGroup(
        device_drafts, selected, proposal_count,
        Pointer<std::uint32_t>(mtp_workspace_, mtp_offsets_.stop_tokens),
        mtp_stop_token_count_, device_result, stream_);
    if (!status.ok()) return status;
    for (std::size_t index = 0; index < model_.layers().size(); ++index) {
      const LayerBinding& layer = model_.layers()[index];
      const std::uint64_t capacity =
          layer.global ? max_context_ : std::min(max_context_, kSlidingWindow);
      if (kv_cache_mode_ == KvCacheMode::kCheckpointFp8) {
        status = internal::LaunchCommitMtpKvFp8(
            Pointer<std::uint8_t>(mtp_workspace_, mtp_offsets_.layers[index].key),
            Pointer<std::uint8_t>(mtp_workspace_, mtp_offsets_.layers[index].value),
            layer.key_cache_fp8, layer.value_cache_fp8, start_position,
            layer.kv_elements, capacity, device_result, stream_);
      } else {
        status = internal::LaunchCommitMtpKvBf16(
            Pointer<float>(mtp_workspace_, mtp_offsets_.layers[index].key),
            Pointer<float>(mtp_workspace_, mtp_offsets_.layers[index].value),
            layer.key_cache_bf16, layer.value_cache_bf16, start_position,
            layer.kv_elements, capacity, device_result, stream_);
      }
      if (!status.ok()) return status;
    }
    float* committed_hidden =
        Pointer<float>(mtp_workspace_, mtp_offsets_.committed_hidden);
    status = internal::LaunchCommitMtpHidden(
        normalized, committed_hidden, kHidden, device_result, stream_);
    if (!status.ok()) return status;
    latest_target_hidden_ = committed_hidden;
    auto* pinned_result = reinterpret_cast<MtpGroupResult*>(
        mtp_host_result_.span().data());
    error = cudaMemcpyAsync(pinned_result, device_result,
                            sizeof(MtpGroupResult), cudaMemcpyDeviceToHost,
                            stream_);
    if (error != cudaSuccess) {
      return CudaFailure("copy GPU MTP group result", error);
    }
    error = cudaStreamSynchronize(stream_);
    if (error != cudaSuccess) {
      return CudaFailure("synchronize GPU MTP group", error);
    }
    *host_result = *pinned_result;
    return Status::Ok();
  }

  [[nodiscard]] std::uint64_t cache_bytes() const { return cache_.bytes(); }
  [[nodiscard]] std::uint64_t workspace_bytes() const {
    return workspace_.bytes() + prefill_workspace_.bytes() +
           mtp_workspace_.bytes();
  }
  [[nodiscard]] std::uint64_t decode_graph_device_bytes() const {
    return decode_graph_device_bytes_;
  }
  [[nodiscard]] std::uint64_t prefill_chunk_tokens() const {
    return prefill_chunk_tokens_;
  }

  [[nodiscard]] Result<std::uint32_t> Prefill(
      std::span<const std::uint32_t> token_ids,
      std::span<float> host_logits = {}) {
    return PrefillAt(token_ids, 0U, host_logits);
  }

  [[nodiscard]] Result<std::uint32_t> PrefillAt(
      std::span<const std::uint32_t> token_ids, std::uint64_t start_position,
      std::span<float> host_logits = {}) {
    const NvtxRange range("gem16.prefill");
    if (token_ids.empty() || start_position > max_context_ ||
        token_ids.size() > max_context_ - start_position) {
      return Error(StatusCode::kInvalidArgument, "prefill token extent is invalid");
    }
    if (!host_logits.empty() && host_logits.size() != kVocabulary) {
      return Error(StatusCode::kInternal,
                   "host prefill logit capture span has invalid size");
    }
    std::uint32_t selected_token = 0U;
    for (std::size_t begin = 0; begin < token_ids.size(); begin += prefill_chunk_tokens_) {
      const std::uint64_t tokens = std::min<std::size_t>(
          prefill_chunk_tokens_, token_ids.size() - begin);
      auto* device_tokens = Pointer<std::uint32_t>(prefill_workspace_, prefill_offsets_.token_ids);
      cudaError_t error = cudaMemcpyAsync(
          device_tokens, token_ids.data() + begin,
          static_cast<std::size_t>(tokens * sizeof(std::uint32_t)),
          cudaMemcpyHostToDevice, stream_);
      if (error != cudaSuccess) return CudaFailure("copy prefill token IDs", error);
      Status status;
      if (sampling_.enabled && sampling_.repetition_penalty != 1.0F) {
        status = internal::LaunchMarkRepetitionTokens(
            device_tokens, tokens,
            Pointer<std::uint32_t>(workspace_, offsets_.repetition_mask),
            stream_);
        if (!status.ok()) return status;
      }
      float* hidden = Pointer<float>(prefill_workspace_, prefill_offsets_.hidden_a);
      const std::uint64_t hidden_elements = tokens * kHidden;
      EmbeddingBatchKernel<<<static_cast<unsigned>((hidden_elements + kThreads - 1U) /
                                                   kThreads),
                             kThreads, 0, stream_>>>(
          model_.embedding(), device_tokens, hidden, hidden_elements);
      error = cudaGetLastError();
      if (error != cudaSuccess) return CudaFailure("launch prefill embedding", error);
      for (const auto& layer : model_.layers()) {
        status = RunLayerBatch(layer, start_position + begin, tokens);
        if (!status.ok()) return status;
      }
      float* normalized = Pointer<float>(prefill_workspace_, prefill_offsets_.normalized);
      status = internal::LaunchRmsNormBf16(
          hidden, model_.final_norm(), normalized, tokens, kHidden, kEpsilon, stream_);
      if (!status.ok()) return status;
      if (begin + tokens == token_ids.size()) {
        float* last = normalized + (tokens - 1U) * kHidden;
        latest_target_hidden_ = last;
        float* logits = Pointer<float>(workspace_, offsets_.logits);
        auto* selected = Pointer<std::uint32_t>(workspace_, offsets_.selected);
        if (sampling_.enabled) {
          status = internal::LaunchFusedOutputHeadCandidates(
              model_.embedding(), last,
              Pointer<std::uint32_t>(workspace_, offsets_.suppressed),
              suppressed_token_count_, nullptr,
              Pointer<ArgmaxValue>(workspace_, offsets_.output_candidates),
              logits, stream_);
        } else {
          status = internal::LaunchOutputHeadLogits(model_.embedding(), last,
                                                    logits, stream_);
        }
        if (!status.ok()) return status;
        if (!host_logits.empty()) {
          error = cudaMemcpyAsync(host_logits.data(), logits,
                                  host_logits.size_bytes(),
                                  cudaMemcpyDeviceToHost, stream_);
          if (error != cudaSuccess) {
            return CudaFailure("copy prefill full logits", error);
          }
        }
        if (sampling_.enabled) {
          status = SelectSampledToken(logits, selected);
          if (!status.ok()) return status;
        } else {
          status = internal::LaunchLogitArgmax(
              logits, Pointer<std::uint32_t>(workspace_, offsets_.suppressed),
              suppressed_token_count_, selected, stream_);
          if (!status.ok()) return status;
        }
        error = cudaMemcpyAsync(&selected_token, selected, sizeof(selected_token),
                                cudaMemcpyDeviceToHost, stream_);
        if (error != cudaSuccess) return CudaFailure("copy prefill token", error);
      }
    }
    const cudaError_t error = cudaStreamSynchronize(stream_);
    if (error != cudaSuccess) return CudaFailure("synchronize prefill", error);
    return selected_token;
  }

  [[nodiscard]] Status ResetCache() {
    cudaError_t error = cudaMemsetAsync(
        cache_.data(), 0, static_cast<std::size_t>(cache_.bytes()), stream_);
    if (error != cudaSuccess) return CudaFailure("clear KV cache", error);
    if (sampling_.enabled) {
      error = cudaMemsetAsync(
          Pointer<std::uint32_t>(workspace_, offsets_.repetition_mask), 0,
          static_cast<std::size_t>(kRepetitionMaskWords * sizeof(std::uint32_t)),
          stream_);
      if (error != cudaSuccess) {
        return CudaFailure("clear repetition mask", error);
      }
    }
    sampling_step_ = 0U;
    error = cudaStreamSynchronize(stream_);
    return error == cudaSuccess ? Status::Ok() : CudaFailure("reset KV cache", error);
  }

  [[nodiscard]] Status SetSampling(const SamplingOptions& options) {
    Status status = ValidateSamplingOptions(
        options, static_cast<std::uint32_t>(kVocabulary));
    if (!status.ok()) return status;
    sampling_ = options;
    sampling_step_ = 0U;
    return Status::Ok();
  }

  [[nodiscard]] Status SetSuppressedTokens(std::span<const std::uint32_t> tokens) {
    if (tokens.size() > kMaximumSuppressedTokens) {
      return Error(StatusCode::kUnsupported,
                   "the initial greedy path supports at most 16 suppressed tokens");
    }
    suppressed_token_count_ = static_cast<std::uint32_t>(tokens.size());
    host_decode_state()->control.suppressed_token_count = suppressed_token_count_;
    if (tokens.empty()) return Status::Ok();
    const cudaError_t error = cudaMemcpyAsync(
        Pointer<std::uint32_t>(workspace_, offsets_.suppressed), tokens.data(),
        tokens.size_bytes(), cudaMemcpyHostToDevice, stream_);
    if (error != cudaSuccess) return CudaFailure("copy suppressed token IDs", error);
    const cudaError_t sync_error = cudaStreamSynchronize(stream_);
    return sync_error == cudaSuccess ? Status::Ok()
                                    : CudaFailure("configure suppressed token IDs", sync_error);
  }

  [[nodiscard]] Status SetMtpStopTokens(
      std::span<const std::uint32_t> tokens) {
    if (mtp_draft_tokens_ == 0U) return Status::Ok();
    if (tokens.size() > kMaximumSuppressedTokens) {
      return Error(StatusCode::kUnsupported,
                   "active MTP supports at most 16 stop-token IDs");
    }
    mtp_stop_token_count_ = static_cast<std::uint32_t>(tokens.size());
    if (tokens.empty()) return Status::Ok();
    const cudaError_t error = cudaMemcpyAsync(
        Pointer<std::uint32_t>(mtp_workspace_, mtp_offsets_.stop_tokens),
        tokens.data(), tokens.size_bytes(), cudaMemcpyHostToDevice, stream_);
    if (error != cudaSuccess) return CudaFailure("copy MTP stop-token IDs", error);
    const cudaError_t sync_error = cudaStreamSynchronize(stream_);
    return sync_error == cudaSuccess
               ? Status::Ok()
               : CudaFailure("configure MTP stop-token IDs", sync_error);
  }

 private:
  [[nodiscard]] Status MarkRepetitionToken(std::uint32_t token) {
    if (!sampling_.enabled || sampling_.repetition_penalty == 1.0F) {
      return Status::Ok();
    }
    return internal::LaunchMarkRepetitionToken(
        token, Pointer<std::uint32_t>(workspace_, offsets_.repetition_mask),
        stream_);
  }

  [[nodiscard]] Status SelectSampledToken(
      float* logits, std::uint32_t* selected,
      const internal::DecodeControl* control = nullptr) {
    return internal::LaunchSampleToken(
        logits, Pointer<float>(workspace_, offsets_.sampling_logits),
        Pointer<double>(workspace_, offsets_.sampling_cumulative),
        Pointer<std::uint32_t>(workspace_, offsets_.sampling_token_ids),
        Pointer<std::uint32_t>(workspace_, offsets_.sorted_token_ids),
        Pointer<std::uint32_t>(workspace_, offsets_.repetition_mask),
        Pointer<std::uint32_t>(workspace_, offsets_.suppressed),
        suppressed_token_count_, static_cast<std::uint32_t>(kVocabulary),
        sampling_, control == nullptr ? sampling_step_++ : 0U, control,
        selected,
        Pointer<std::uint8_t>(workspace_, offsets_.sampling_sort_workspace),
        sampling_sort_workspace_bytes_, stream_);
  }

  [[nodiscard]] Status AllocateCache() {
    LayoutBuilder layout;
    struct CacheOffsets { std::uint64_t key; std::uint64_t value; };
    std::array<CacheOffsets, kLayers> offsets{};
    for (std::size_t index = 0; index < kLayers; ++index) {
      const auto& layer = model_.layers()[index];
      const std::uint64_t cache_tokens =
          layer.global ? max_context_ : std::min(max_context_, kSlidingWindow);
      Result<std::uint64_t> key =
          kv_cache_mode_ == KvCacheMode::kCheckpointFp8
              ? layout.Add<std::uint8_t>(cache_tokens * layer.kv_elements)
              : layout.Add<float>(cache_tokens * layer.kv_elements);
      Result<std::uint64_t> value =
          kv_cache_mode_ == KvCacheMode::kCheckpointFp8
              ? layout.Add<std::uint8_t>(cache_tokens * layer.kv_elements)
              : layout.Add<float>(cache_tokens * layer.kv_elements);
      if (!key.ok()) return key.status();
      if (!value.ok()) return value.status();
      offsets[index] = {key.value(), value.value()};
    }
    auto size = AlignUp(layout.size(), kAlignment);
    if (!size.ok()) return size.status();
    Status status = cache_.Allocate(
        size.value(), kv_cache_mode_ == KvCacheMode::kCheckpointFp8
                          ? "allocate checkpoint FP8 KV cache"
                          : "allocate BF16-semantics KV cache");
    if (!status.ok()) return status;
    for (std::size_t index = 0; index < kLayers; ++index) {
      if (kv_cache_mode_ == KvCacheMode::kCheckpointFp8) {
        model_.SetLayerFp8Cache(
            index, Pointer<std::uint8_t>(cache_, offsets[index].key),
            Pointer<std::uint8_t>(cache_, offsets[index].value));
      } else {
        model_.SetLayerBf16Cache(
            index, Pointer<float>(cache_, offsets[index].key),
            Pointer<float>(cache_, offsets[index].value));
      }
    }
    return Status::Ok();
  }

  [[nodiscard]] Status AllocateWorkspace() {
    LayoutBuilder layout;
#define GEM16_ADD(field, type, elements)                 \
    do {                                                   \
      auto next = layout.Add<type>(elements);              \
      if (!next.ok()) return next.status();                 \
      offsets_.field = next.value();                        \
    } while (false)
    GEM16_ADD(decode_control, internal::DecodeControl, 1U);
    GEM16_ADD(hidden_a, float, kHidden);
    GEM16_ADD(hidden_b, float, kHidden);
    GEM16_ADD(normalized, float, kHidden);
    GEM16_ADD(fp8_activation, std::uint8_t, kHidden);
    GEM16_ADD(fp8_scale, float, 1U);
    GEM16_ADD(q, float, kQueryHeads * 512U);
    GEM16_ADD(k, float, 8U * 256U);
    GEM16_ADD(v, float, 8U * 256U);
    GEM16_ADD(q_norm, float, kQueryHeads * 512U);
    GEM16_ADD(k_norm, float, 8U * 256U);
    GEM16_ADD(v_norm, float, 8U * 256U);
    GEM16_ADD(scores, float,
                internal::DecodeAttentionWorkspaceElements(max_context_));
    GEM16_ADD(attention, float, kQueryHeads * 512U);
    GEM16_ADD(o_activation, std::uint8_t, kQueryHeads * 512U);
    GEM16_ADD(o_scale, float, 1U);
    GEM16_ADD(projection, float, kHidden);
    GEM16_ADD(post_norm, float, kHidden);
    GEM16_ADD(mlp_packed, std::uint8_t, kHidden / 2U);
    GEM16_ADD(mlp_scales, std::uint8_t, kHidden / 16U);
    GEM16_ADD(gate, float, kIntermediate);
    GEM16_ADD(up, float, kIntermediate);
    GEM16_ADD(product, float, kIntermediate);
    GEM16_ADD(down_packed, std::uint8_t, kIntermediate / 2U);
    GEM16_ADD(down_scales, std::uint8_t, kIntermediate / 16U);
    GEM16_ADD(logits, float, kVocabulary);
    if (sampling_.enabled) {
      GEM16_ADD(sampling_logits, float, kVocabulary);
      GEM16_ADD(sampling_cumulative, double, kVocabulary);
      GEM16_ADD(sampling_token_ids, std::uint32_t, kVocabulary);
      GEM16_ADD(sorted_token_ids, std::uint32_t, kVocabulary);
      auto sort_bytes = internal::SamplingWorkspaceBytes(
          static_cast<std::uint32_t>(kVocabulary), stream_);
      if (!sort_bytes.ok()) return sort_bytes.status();
      sampling_sort_workspace_bytes_ = sort_bytes.value();
      GEM16_ADD(sampling_sort_workspace, std::uint8_t,
                sampling_sort_workspace_bytes_);
      GEM16_ADD(repetition_mask, std::uint32_t, kRepetitionMaskWords);
    }
    GEM16_ADD(output_candidates, ArgmaxValue, kFusedOutputHeadBlocks);
    GEM16_ADD(selected, std::uint32_t, 1U);
    GEM16_ADD(suppressed, std::uint32_t, kMaximumSuppressedTokens);
#undef GEM16_ADD
    auto size = AlignUp(layout.size(), kAlignment);
    if (!size.ok()) return size.status();
    offsets_.total = size.value();
    return workspace_.Allocate(size.value(), "allocate inference workspace arena");
  }

  [[nodiscard]] Status AllocateMtpWorkspace() {
    LayoutBuilder layout;
    for (std::size_t index = 0; index < model_.layers().size(); ++index) {
      const std::uint64_t elements =
          kMaximumMtpVerifyTokens * model_.layers()[index].kv_elements;
      Result<std::uint64_t> key =
          kv_cache_mode_ == KvCacheMode::kCheckpointFp8
              ? layout.Add<std::uint8_t>(elements)
              : layout.Add<float>(elements);
      Result<std::uint64_t> value =
          kv_cache_mode_ == KvCacheMode::kCheckpointFp8
              ? layout.Add<std::uint8_t>(elements)
              : layout.Add<float>(elements);
      Result<std::uint64_t> backup_key =
          kv_cache_mode_ == KvCacheMode::kCheckpointFp8
              ? layout.Add<std::uint8_t>(elements)
              : layout.Add<float>(elements);
      Result<std::uint64_t> backup_value =
          kv_cache_mode_ == KvCacheMode::kCheckpointFp8
              ? layout.Add<std::uint8_t>(elements)
              : layout.Add<float>(elements);
      if (!key.ok()) return key.status();
      if (!value.ok()) return value.status();
      if (!backup_key.ok()) return backup_key.status();
      if (!backup_value.ok()) return backup_value.status();
      mtp_offsets_.layers[index] = {
          key.value(), value.value(), backup_key.value(), backup_value.value()};
    }
    auto candidates = layout.Add<ArgmaxValue>(
        kMaximumMtpVerifyTokens * kFusedOutputHeadBlocks);
    auto selected = layout.Add<std::uint32_t>(kMaximumMtpVerifyTokens);
    auto stop_tokens = layout.Add<std::uint32_t>(kMaximumSuppressedTokens);
    auto group_result = layout.Add<MtpGroupResult>(1U);
    auto committed_hidden = layout.Add<float>(kHidden);
    if (!candidates.ok()) return candidates.status();
    if (!selected.ok()) return selected.status();
    if (!stop_tokens.ok()) return stop_tokens.status();
    if (!group_result.ok()) return group_result.status();
    if (!committed_hidden.ok()) return committed_hidden.status();
    mtp_offsets_.output_candidates = candidates.value();
    mtp_offsets_.selected = selected.value();
    mtp_offsets_.stop_tokens = stop_tokens.value();
    mtp_offsets_.group_result = group_result.value();
    mtp_offsets_.committed_hidden = committed_hidden.value();
    auto size = AlignUp(layout.size(), kAlignment);
    if (!size.ok()) return size.status();
    mtp_offsets_.total = size.value();
    return mtp_workspace_.Allocate(size.value(),
                                   "allocate MTP verification workspace");
  }

  [[nodiscard]] Status AllocatePrefillWorkspace() {
    LayoutBuilder layout;
#define GEM16_PREFILL_ADD(field, type, elements)          \
    do {                                                     \
      auto next = layout.Add<type>(elements);                \
      if (!next.ok()) return next.status();                   \
      prefill_offsets_.field = next.value();                  \
    } while (false)
    const std::uint64_t tokens = prefill_chunk_tokens_;
    constexpr std::uint64_t max_q = kQueryHeads * 512U;
    constexpr std::uint64_t max_kv = 8U * 256U;
    GEM16_PREFILL_ADD(token_ids, std::uint32_t, tokens);
    GEM16_PREFILL_ADD(hidden_a, float, tokens * kHidden);
    GEM16_PREFILL_ADD(hidden_b, float, tokens * kHidden);
    GEM16_PREFILL_ADD(normalized, float, tokens * kHidden);
    GEM16_PREFILL_ADD(fp8_activation, std::uint8_t, tokens * max_q);
    GEM16_PREFILL_ADD(fp8_scales, float, tokens);
    GEM16_PREFILL_ADD(q, float, tokens * max_q);
    GEM16_PREFILL_ADD(k, float, tokens * max_kv);
    GEM16_PREFILL_ADD(v, float, tokens * max_kv);
    GEM16_PREFILL_ADD(q_norm, float, tokens * max_q);
    GEM16_PREFILL_ADD(k_norm, float, tokens * max_kv);
    GEM16_PREFILL_ADD(v_norm, float, tokens * max_kv);
    GEM16_PREFILL_ADD(k_fp8, std::uint8_t, tokens * max_kv);
    GEM16_PREFILL_ADD(v_fp8, std::uint8_t, tokens * max_kv);
    if (kv_cache_mode_ == KvCacheMode::kBf16Correctness) {
      GEM16_PREFILL_ADD(scores, float,
                          tokens * kQueryHeads * max_context_);
    }
    GEM16_PREFILL_ADD(attention, float, tokens * max_q);
    GEM16_PREFILL_ADD(o_activation, std::uint8_t, tokens * max_q);
    GEM16_PREFILL_ADD(o_scales, float, tokens);
    GEM16_PREFILL_ADD(projection, float, tokens * kHidden);
    GEM16_PREFILL_ADD(post_norm, float, tokens * kHidden);
    GEM16_PREFILL_ADD(mlp_packed, std::uint8_t, tokens * kHidden / 2U);
    GEM16_PREFILL_ADD(mlp_scales, std::uint8_t, tokens * kHidden / 16U);
    GEM16_PREFILL_ADD(gate, std::uint16_t, tokens * kIntermediate);
    GEM16_PREFILL_ADD(up, std::uint16_t, tokens * kIntermediate);
    GEM16_PREFILL_ADD(down_packed, std::uint8_t, tokens * kIntermediate / 2U);
    GEM16_PREFILL_ADD(down_scales, std::uint8_t, tokens * kIntermediate / 16U);
    constexpr std::uint64_t kCutlassScaleRows = 128U;
    constexpr std::uint64_t kCutlassWorkspaceBytes = 8U * 1024U * 1024U;
    const std::uint64_t cutlass_tokens =
        ((tokens + kCutlassScaleRows - 1U) / kCutlassScaleRows) *
        kCutlassScaleRows;
    GEM16_PREFILL_ADD(cutlass_activation_scales, std::uint8_t,
                        cutlass_tokens * kIntermediate / 16U);
    GEM16_PREFILL_ADD(cutlass_weight, std::uint8_t,
                        kIntermediate * kHidden / 2U);
    GEM16_PREFILL_ADD(cutlass_weight_scales, std::uint8_t,
                        kIntermediate * kHidden / 16U);
    GEM16_PREFILL_ADD(cutlass_workspace, std::uint8_t,
                        kCutlassWorkspaceBytes);
    GEM16_PREFILL_ADD(local_rope_cosine, float, max_context_ * 128U);
    GEM16_PREFILL_ADD(local_rope_sine, float, max_context_ * 128U);
    GEM16_PREFILL_ADD(global_rope_cosine, float, max_context_ * 64U);
    GEM16_PREFILL_ADD(global_rope_sine, float, max_context_ * 64U);
#undef GEM16_PREFILL_ADD
    auto size = AlignUp(layout.size(), kAlignment);
    if (!size.ok()) return size.status();
    return prefill_workspace_.Allocate(size.value(), "allocate native prefill workspace");
  }

  [[nodiscard]] Status RunLayerBatch(
      const LayerBinding& layer, std::uint64_t start_position,
      std::uint64_t tokens, std::size_t mtp_layer_index = kLayers) {
    const bool mtp_verification = mtp_layer_index < kLayers;
    const NvtxRange range(mtp_verification ? "gem16.mtp.verify.layer"
                                           : "gem16.prefill.layer");
    float* hidden_a = Pointer<float>(prefill_workspace_, prefill_offsets_.hidden_a);
    float* hidden_b = Pointer<float>(prefill_workspace_, prefill_offsets_.hidden_b);
    auto* fp8 = Pointer<std::uint8_t>(prefill_workspace_, prefill_offsets_.fp8_activation);
    float* fp8_scales = Pointer<float>(prefill_workspace_, prefill_offsets_.fp8_scales);
    float* q = Pointer<float>(prefill_workspace_, prefill_offsets_.q);
    float* k = Pointer<float>(prefill_workspace_, prefill_offsets_.k);
    float* v = Pointer<float>(prefill_workspace_, prefill_offsets_.v);
    float* q_norm = Pointer<float>(prefill_workspace_, prefill_offsets_.q_norm);
    float* k_norm = Pointer<float>(prefill_workspace_, prefill_offsets_.k_norm);
    float* v_norm = Pointer<float>(prefill_workspace_, prefill_offsets_.v_norm);
    float* attention = Pointer<float>(prefill_workspace_, prefill_offsets_.attention);
    float* projection = Pointer<float>(prefill_workspace_, prefill_offsets_.projection);
    auto* cutlass_workspace = Pointer<std::uint8_t>(
        prefill_workspace_, prefill_offsets_.cutlass_workspace);
    constexpr std::size_t kCutlassWorkspaceBytes = 8U * 1024U * 1024U;
    Status status = internal::LaunchRmsNormFp8TokenQuantizationBatch(
        hidden_a, layer.input_norm, fp8, fp8_scales, tokens, kHidden,
        kEpsilon, stream_);
    if (!status.ok()) return status;
    status = mtp_verification
                 ? internal::LaunchFp8Sm120GroupedQkvProjectionBatch(
                       fp8, fp8_scales, layer.q.weight, layer.q.scales, q,
                       layer.q.rows, layer.k.weight, layer.k.scales, k,
                       layer.k.rows,
                       layer.global ? nullptr : layer.v.weight,
                       layer.global ? nullptr : layer.v.scales,
                       layer.global ? nullptr : v,
                       layer.global ? 0U : layer.v.rows, tokens,
                       layer.q.contracting, stream_)
                 : LaunchFp8QkvProjectionBatch(
                       fp8, fp8_scales, layer.q, q, layer.k, k,
                       layer.global ? nullptr : &layer.v, v, tokens,
                       cutlass_workspace, kCutlassWorkspaceBytes, stream_);
    if (!status.ok()) return status;
    if (layer.global) {
      const cudaError_t error = cudaMemcpyAsync(
          v, k, static_cast<std::size_t>(tokens * layer.kv_elements * sizeof(float)),
          cudaMemcpyDeviceToDevice, stream_);
      if (error != cudaSuccess) return CudaFailure("reuse batched global K for V", error);
    }
    for (const Status next : {
             LaunchRoundBf16(v, tokens * layer.kv_elements, stream_),
             internal::LaunchRmsNormBf16(v, nullptr, v_norm,
                                     tokens * layer.kv_heads, layer.head_dimension,
                                     kEpsilon, stream_)}) {
      if (!next.ok()) return next;
    }
    status = internal::LaunchProjectionRmsNormRotaryBf16Batch(
        q, layer.q_norm, q_norm, k, layer.k_norm, k_norm,
        layer.global
            ? Pointer<float>(prefill_workspace_, prefill_offsets_.global_rope_cosine) +
                  start_position * 64U
            : Pointer<float>(prefill_workspace_, prefill_offsets_.local_rope_cosine) +
                  start_position * 128U,
        layer.global
            ? Pointer<float>(prefill_workspace_, prefill_offsets_.global_rope_sine) +
                  start_position * 64U
            : Pointer<float>(prefill_workspace_, prefill_offsets_.local_rope_sine) +
                  start_position * 128U,
        tokens, kQueryHeads, layer.kv_heads, layer.head_dimension,
        layer.global ? 0.25 : 1.0, kEpsilon, stream_);
    if (!status.ok()) return status;
    const std::uint64_t capacity =
        layer.global ? max_context_ : std::min(max_context_, kSlidingWindow);
    if (kv_cache_mode_ == KvCacheMode::kCheckpointFp8) {
      auto* k_fp8 = Pointer<std::uint8_t>(prefill_workspace_, prefill_offsets_.k_fp8);
      auto* v_fp8 = Pointer<std::uint8_t>(prefill_workspace_, prefill_offsets_.v_fp8);
      status = internal::LaunchQuantizeKvFp8Batch(
          k_norm, v_norm, k_fp8, v_fp8, layer.k_cache_scale,
          layer.v_cache_scale, tokens, layer.kv_elements, stream_);
      if (!status.ok()) return status;
      if (mtp_verification) {
        auto* speculative_key = Pointer<std::uint8_t>(
            mtp_workspace_, mtp_offsets_.layers[mtp_layer_index].key);
        auto* speculative_value = Pointer<std::uint8_t>(
            mtp_workspace_, mtp_offsets_.layers[mtp_layer_index].value);
        const std::size_t bytes = static_cast<std::size_t>(
            tokens * layer.kv_elements * sizeof(std::uint8_t));
        cudaError_t copy_error = cudaMemcpyAsync(
            speculative_key, k_fp8, bytes, cudaMemcpyDeviceToDevice, stream_);
        if (copy_error == cudaSuccess) {
          copy_error = cudaMemcpyAsync(speculative_value, v_fp8, bytes,
                                       cudaMemcpyDeviceToDevice, stream_);
        }
        if (copy_error != cudaSuccess) {
          return CudaFailure("retain speculative FP8 KV", copy_error);
        }
        if (!layer.global) {
          status = internal::LaunchCopyCircularMtpKvFp8(
              layer.key_cache_fp8, layer.value_cache_fp8,
              Pointer<std::uint8_t>(
                  mtp_workspace_,
                  mtp_offsets_.layers[mtp_layer_index].backup_key),
              Pointer<std::uint8_t>(
                  mtp_workspace_,
                  mtp_offsets_.layers[mtp_layer_index].backup_value),
              start_position, tokens, layer.kv_elements, capacity, false,
              stream_);
          if (!status.ok()) return status;
        }
        if (layer.global) {
          status = internal::LaunchAppendKvFp8Batch(
              k_fp8, v_fp8, layer.key_cache_fp8, layer.value_cache_fp8,
              start_position, tokens, layer.kv_elements, capacity, stream_);
          if (!status.ok()) return status;
        }
        float* decode_scores = Pointer<float>(workspace_, offsets_.scores);
        auto* control = Pointer<internal::DecodeControl>(
            workspace_, offsets_.decode_control);
        for (std::uint64_t row = 0U; row < tokens; ++row) {
          const std::uint64_t position = start_position + row;
          if (!layer.global) {
            status = internal::LaunchAppendKvFp8Batch(
                k_fp8 + row * layer.kv_elements,
                v_fp8 + row * layer.kv_elements, layer.key_cache_fp8,
                layer.value_cache_fp8, position, 1U, layer.kv_elements,
                capacity, stream_);
            if (!status.ok()) return status;
          }
          const std::uint64_t attention_tokens =
              layer.global ? position + 1U
                           : std::min(position + 1U, capacity);
          const std::uint64_t first_slot =
              layer.global || position + 1U <= capacity
                  ? 0U
                  : (position + 1U) % capacity;
          if (capacity <= 512U) {
            status = internal::LaunchLocalAttentionDecodeFp8(
                q_norm + row * layer.query_elements, layer.key_cache_fp8,
                layer.value_cache_fp8, layer.k_cache_scale,
                layer.v_cache_scale, decode_scores,
                attention + row * layer.query_elements, kQueryHeads,
                layer.kv_heads, layer.head_dimension, attention_tokens,
                stream_, capacity, first_slot);
          } else {
            status = internal::LaunchSetMtpAttentionPosition(
                control, position, stream_);
            if (!status.ok()) return status;
            status = internal::LaunchOnlineAttentionDecodeFp8Sm120(
                q_norm + row * layer.query_elements, layer.key_cache_fp8,
                layer.value_cache_fp8, layer.k_cache_scale,
                layer.v_cache_scale, decode_scores,
                attention + row * layer.query_elements, control, kQueryHeads,
                layer.kv_heads, layer.head_dimension, capacity,
                !layer.global, stream_);
          }
          if (!status.ok()) return status;
        }
        if (!layer.global) {
          status = internal::LaunchCopyCircularMtpKvFp8(
              layer.key_cache_fp8, layer.value_cache_fp8,
              Pointer<std::uint8_t>(
                  mtp_workspace_,
                  mtp_offsets_.layers[mtp_layer_index].backup_key),
              Pointer<std::uint8_t>(
                  mtp_workspace_,
                  mtp_offsets_.layers[mtp_layer_index].backup_value),
              start_position, tokens, layer.kv_elements, capacity, true,
              stream_);
          if (!status.ok()) return status;
        }
      } else {
        status = layer.global
                     ? internal::LaunchOnlineCausalAttentionPrefillFp8GlobalSm120(
                           q_norm, k_fp8, v_fp8, layer.key_cache_fp8,
                           layer.value_cache_fp8, layer.k_cache_scale,
                           layer.v_cache_scale, attention, start_position,
                           tokens, kQueryHeads, layer.kv_heads,
                           layer.head_dimension, capacity, stream_)
                     : internal::LaunchOnlineCausalAttentionPrefillFp8LocalSm120(
                           q_norm, k_fp8, v_fp8, layer.key_cache_fp8,
                           layer.value_cache_fp8, layer.k_cache_scale,
                           layer.v_cache_scale, attention, start_position,
                           tokens, kQueryHeads, layer.kv_heads,
                           layer.head_dimension, capacity, stream_);
        if (!status.ok()) return status;
        const std::uint64_t commit_offset =
            layer.global || tokens <= capacity ? 0U : tokens - capacity;
        status = internal::LaunchAppendKvFp8Batch(
            k_fp8 + commit_offset * layer.kv_elements,
            v_fp8 + commit_offset * layer.kv_elements, layer.key_cache_fp8,
            layer.value_cache_fp8, start_position + commit_offset,
            tokens - commit_offset, layer.kv_elements, capacity, stream_);
      }
    } else if (mtp_verification) {
      auto* speculative_key = Pointer<float>(
          mtp_workspace_, mtp_offsets_.layers[mtp_layer_index].key);
      auto* speculative_value = Pointer<float>(
          mtp_workspace_, mtp_offsets_.layers[mtp_layer_index].value);
      const std::size_t bytes = static_cast<std::size_t>(
          tokens * layer.kv_elements * sizeof(float));
      cudaError_t copy_error = cudaMemcpyAsync(
          speculative_key, k_norm, bytes, cudaMemcpyDeviceToDevice, stream_);
      if (copy_error == cudaSuccess) {
        copy_error = cudaMemcpyAsync(speculative_value, v_norm, bytes,
                                     cudaMemcpyDeviceToDevice, stream_);
      }
      if (copy_error != cudaSuccess) {
        return CudaFailure("retain speculative BF16 KV", copy_error);
      }
      if (!layer.global) {
        status = internal::LaunchCopyCircularMtpKvBf16(
            layer.key_cache_bf16, layer.value_cache_bf16,
            Pointer<float>(mtp_workspace_,
                           mtp_offsets_.layers[mtp_layer_index].backup_key),
            Pointer<float>(mtp_workspace_,
                           mtp_offsets_.layers[mtp_layer_index].backup_value),
            start_position, tokens, layer.kv_elements, capacity, false,
            stream_);
        if (!status.ok()) return status;
      }
      if (layer.global) {
        status = internal::LaunchAppendKvBatch(
            k_norm, v_norm, layer.key_cache_bf16, layer.value_cache_bf16,
            start_position, tokens, layer.kv_elements, capacity, stream_);
        if (!status.ok()) return status;
      }
      float* decode_scores = Pointer<float>(workspace_, offsets_.scores);
      for (std::uint64_t row = 0U; row < tokens; ++row) {
        const std::uint64_t position = start_position + row;
        if (!layer.global) {
          status = internal::LaunchAppendKvBatch(
              k_norm + row * layer.kv_elements,
              v_norm + row * layer.kv_elements, layer.key_cache_bf16,
              layer.value_cache_bf16, position, 1U, layer.kv_elements,
              capacity, stream_);
          if (!status.ok()) return status;
        }
        const std::uint64_t attention_tokens =
            layer.global ? position + 1U : std::min(position + 1U, capacity);
        const std::uint64_t first_slot =
            layer.global || position + 1U <= capacity
                ? 0U
                : (position + 1U) % capacity;
        status = internal::LaunchLocalAttentionDecode(
            q_norm + row * layer.query_elements, layer.key_cache_bf16,
            layer.value_cache_bf16, decode_scores,
            attention + row * layer.query_elements, kQueryHeads,
            layer.kv_heads, layer.head_dimension, attention_tokens, stream_,
            capacity, first_slot);
        if (!status.ok()) return status;
      }
      if (!layer.global) {
        status = internal::LaunchCopyCircularMtpKvBf16(
            layer.key_cache_bf16, layer.value_cache_bf16,
            Pointer<float>(mtp_workspace_,
                           mtp_offsets_.layers[mtp_layer_index].backup_key),
            Pointer<float>(mtp_workspace_,
                           mtp_offsets_.layers[mtp_layer_index].backup_value),
            start_position, tokens, layer.kv_elements, capacity, true,
            stream_);
        if (!status.ok()) return status;
      }
    } else {
      float* scores =
          Pointer<float>(prefill_workspace_, prefill_offsets_.scores);
      status = internal::LaunchFusedCausalAttentionPrefill(
          q_norm, k_norm, v_norm, layer.key_cache_bf16,
          layer.value_cache_bf16, scores, attention, start_position, tokens,
          kQueryHeads, layer.kv_heads, layer.head_dimension, capacity,
          !layer.global, stream_);
      if (!status.ok()) return status;
      status = internal::LaunchAppendKvBatch(
          k_norm, v_norm, layer.key_cache_bf16, layer.value_cache_bf16,
          start_position, tokens, layer.kv_elements, capacity, stream_);
    }
    if (!status.ok()) return status;
    return LaunchLayerBatchSuffix(layer, tokens, mtp_verification);
  }

  [[nodiscard]] Status LaunchLayerBatchSuffix(
      const LayerBinding& layer, std::uint64_t tokens,
      bool mtp_verification) {
    float* hidden_a =
        Pointer<float>(prefill_workspace_, prefill_offsets_.hidden_a);
    float* hidden_b =
        Pointer<float>(prefill_workspace_, prefill_offsets_.hidden_b);
    float* attention =
        Pointer<float>(prefill_workspace_, prefill_offsets_.attention);
    float* projection =
        Pointer<float>(prefill_workspace_, prefill_offsets_.projection);
    auto* cutlass_workspace = Pointer<std::uint8_t>(
        prefill_workspace_, prefill_offsets_.cutlass_workspace);
    constexpr std::size_t kCutlassWorkspaceBytes = 8U * 1024U * 1024U;
    const std::uint64_t hidden_elements = tokens * kHidden;
    Status status =
        LaunchRoundBf16(attention, tokens * layer.query_elements, stream_);
    if (!status.ok()) return status;
    auto* o_activation = Pointer<std::uint8_t>(prefill_workspace_, prefill_offsets_.o_activation);
    float* o_scales = Pointer<float>(prefill_workspace_, prefill_offsets_.o_scales);
    status = internal::LaunchFp8ReferenceTokenQuantizationBatch(
        attention, o_activation, o_scales, tokens, layer.query_elements, stream_);
    if (!status.ok()) return status;
    status = mtp_verification
                 ? internal::LaunchFp8Sm120DirectProjectionBatch(
                       o_activation, o_scales, layer.o.weight, layer.o.scales,
                       projection, tokens, layer.o.rows, layer.o.contracting,
                       stream_)
                 : LaunchFp8ProjectionBatch(
                       o_activation, o_scales, layer.o, projection, tokens,
                       cutlass_workspace, kCutlassWorkspaceBytes, stream_);
    if (!status.ok()) return status;
    status = LaunchRoundBf16(projection, hidden_elements, stream_);
    if (!status.ok()) return status;
    status = internal::LaunchRmsNormResidualBf16(
        projection, layer.post_attention_norm, hidden_a, nullptr, hidden_b,
        tokens, kHidden, kEpsilon, nullptr, stream_);
    if (!status.ok()) return status;

    auto* mlp_packed = Pointer<std::uint8_t>(prefill_workspace_, prefill_offsets_.mlp_packed);
    auto* mlp_scales = Pointer<std::uint8_t>(prefill_workspace_, prefill_offsets_.mlp_scales);
    auto* gate = Pointer<std::uint16_t>(prefill_workspace_, prefill_offsets_.gate);
    auto* up = Pointer<std::uint16_t>(prefill_workspace_, prefill_offsets_.up);
    status = internal::LaunchRmsNormNvfp4ActivationQuantizationBatch(
        hidden_b, layer.pre_mlp_norm, mlp_packed, mlp_scales, tokens, kHidden,
        kEpsilon, layer.gate.input_divisor, stream_);
    if (!status.ok()) return status;
    auto* cutlass_activation_scales = Pointer<std::uint8_t>(
        prefill_workspace_, prefill_offsets_.cutlass_activation_scales);
    auto* cutlass_weight = Pointer<std::uint8_t>(
        prefill_workspace_, prefill_offsets_.cutlass_weight);
    auto* cutlass_weight_scales = Pointer<std::uint8_t>(
        prefill_workspace_, prefill_offsets_.cutlass_weight_scales);
    float* mtp_product = nullptr;
    if (mtp_verification) {
      // The fused native operator preserves the two BF16 projection
      // boundaries and the BF16 GELU/product boundary. Reuse inactive
      // CUTLASS weight scratch only for this transient batch product.
      mtp_product = reinterpret_cast<float*>(cutlass_weight);
      status = internal::LaunchNvfp4Sm120FusedGateUpBatch(
          mlp_packed, mlp_scales, layer.gate.packed_weight,
          layer.gate.scales, layer.up.packed_weight, layer.up.scales, nullptr,
          nullptr, mtp_product, tokens, layer.gate.rows,
          layer.gate.contracting, layer.gate.input_divisor,
          layer.gate.weight_divisor, layer.up.input_divisor,
          layer.up.weight_divisor, stream_);
    } else {
      status = internal::LaunchNvfp4CutlassInterleaveActivationScales(
          mlp_scales, cutlass_activation_scales, tokens, kHidden, stream_);
      if (!status.ok()) return status;
      status = internal::LaunchNvfp4CutlassProjectionBf16Batch(
          mlp_packed, cutlass_activation_scales, layer.gate.packed_weight,
          layer.gate.scales, cutlass_weight, cutlass_weight_scales,
          cutlass_workspace, kCutlassWorkspaceBytes, gate, tokens,
          layer.gate.rows, layer.gate.contracting, layer.gate.input_divisor,
          layer.gate.weight_divisor, stream_);
      if (!status.ok()) return status;
      status = internal::LaunchNvfp4CutlassProjectionBf16Batch(
          mlp_packed, cutlass_activation_scales, layer.up.packed_weight,
          layer.up.scales, cutlass_weight, cutlass_weight_scales,
          cutlass_workspace, kCutlassWorkspaceBytes, up, tokens,
          layer.up.rows, layer.up.contracting, layer.up.input_divisor,
          layer.up.weight_divisor, stream_);
    }
    if (!status.ok()) return status;
    auto* down_packed = Pointer<std::uint8_t>(prefill_workspace_, prefill_offsets_.down_packed);
    auto* down_scales = Pointer<std::uint8_t>(prefill_workspace_, prefill_offsets_.down_scales);
    status = mtp_verification
                 ? internal::LaunchNvfp4ReferenceActivationQuantization(
                       mtp_product, down_packed, down_scales,
                       tokens * kIntermediate, layer.down.input_divisor,
                       stream_)
                 : internal::LaunchGatedGeluNvfp4ActivationQuantizationBf16(
                       gate, up, down_packed, down_scales,
                       tokens * kIntermediate, layer.down.input_divisor,
                       stream_);
    if (!status.ok()) return status;
    auto* down_bf16 = reinterpret_cast<std::uint16_t*>(projection);
    if (mtp_verification) {
      status = internal::LaunchNvfp4Sm120DirectProjectionBf16Batch(
          down_packed, down_scales, layer.down.packed_weight,
          layer.down.scales, down_bf16, tokens, layer.down.rows,
          layer.down.contracting, layer.down.input_divisor,
          layer.down.weight_divisor, stream_);
    } else {
      status = internal::LaunchNvfp4CutlassInterleaveActivationScales(
          down_scales, cutlass_activation_scales, tokens, kIntermediate,
          stream_);
      if (!status.ok()) return status;
      status = internal::LaunchNvfp4CutlassProjectionBf16Batch(
          down_packed, cutlass_activation_scales, layer.down.packed_weight,
          layer.down.scales, cutlass_weight, cutlass_weight_scales,
          cutlass_workspace, kCutlassWorkspaceBytes, down_bf16, tokens,
          layer.down.rows, layer.down.contracting, layer.down.input_divisor,
          layer.down.weight_divisor, stream_);
    }
    if (!status.ok()) return status;
    return internal::LaunchRmsNormResidualBf16Input(
        down_bf16, layer.post_mlp_norm, hidden_b, nullptr, hidden_a, tokens,
        kHidden, kEpsilon, layer.layer_scalar, stream_);
  }

  template <typename Launch>
  [[nodiscard]] Status CaptureDecodeGraph(GraphExecutable& destination,
                                          Launch&& launch,
                                          const char* label) {
    cudaError_t error =
        cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal);
    if (error != cudaSuccess) return CudaFailure(label, error);
    const Status launch_status = launch();
    cudaGraph_t graph = nullptr;
    error = cudaStreamEndCapture(stream_, &graph);
    if (!launch_status.ok()) {
      if (graph != nullptr) (void)cudaGraphDestroy(graph);
      return launch_status;
    }
    if (error != cudaSuccess) {
      if (graph != nullptr) (void)cudaGraphDestroy(graph);
      return CudaFailure(label, error);
    }
    cudaGraphExec_t executable = nullptr;
    error = cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0U);
    const cudaError_t destroy_error = cudaGraphDestroy(graph);
    if (error != cudaSuccess) return CudaFailure(label, error);
    if (destroy_error != cudaSuccess) {
      (void)cudaGraphExecDestroy(executable);
      return CudaFailure(label, destroy_error);
    }
    destination.Adopt(executable);
    return Status::Ok();
  }

  [[nodiscard]] HostDecodeState* host_decode_state() const {
    return reinterpret_cast<HostDecodeState*>(decode_host_state_.span().data());
  }

  [[nodiscard]] Status LaunchControlledDecodeLayer(
      const LayerBinding& layer) {
    Status status = LaunchDecodePrefix(layer, false);
    if (!status.ok()) return status;
    auto* control = Pointer<internal::DecodeControl>(
        workspace_, offsets_.decode_control);
    float* q_norm = Pointer<float>(workspace_, offsets_.q_norm);
    float* k_norm = Pointer<float>(workspace_, offsets_.k_norm);
    float* v_norm = Pointer<float>(workspace_, offsets_.v_norm);
    float* scores = Pointer<float>(workspace_, offsets_.scores);
    float* attention = Pointer<float>(workspace_, offsets_.attention);
    const float* rotary_cosine =
        layer.global
            ? Pointer<float>(prefill_workspace_,
                             prefill_offsets_.global_rope_cosine)
            : Pointer<float>(prefill_workspace_,
                             prefill_offsets_.local_rope_cosine);
    const float* rotary_sine =
        layer.global
            ? Pointer<float>(prefill_workspace_,
                             prefill_offsets_.global_rope_sine)
            : Pointer<float>(prefill_workspace_,
                             prefill_offsets_.local_rope_sine);
    status = internal::LaunchProjectionRmsNormRotaryBf16Controlled(
        Pointer<float>(workspace_, offsets_.q), layer.q_norm, q_norm,
        Pointer<float>(workspace_, offsets_.k), layer.k_norm, k_norm,
        rotary_cosine, rotary_sine, control, kQueryHeads, layer.kv_heads,
        layer.head_dimension, layer.global ? 0.25 : 1.0, kEpsilon, stream_);
    if (!status.ok()) return status;
    const std::uint64_t capacity =
        layer.global ? max_context_ : std::min(max_context_, kSlidingWindow);
    if (kv_cache_mode_ == KvCacheMode::kCheckpointFp8) {
      status = internal::LaunchAppendKvFp8Controlled(
          k_norm, v_norm, layer.key_cache_fp8, layer.value_cache_fp8,
          layer.k_cache_scale, layer.v_cache_scale, control, layer.kv_heads,
          layer.head_dimension, capacity, !layer.global, stream_);
      if (!status.ok()) return status;
      if (capacity <= 512U) {
        status = internal::LaunchLocalAttentionDecodeFp8Controlled(
            q_norm, layer.key_cache_fp8, layer.value_cache_fp8,
            layer.k_cache_scale, layer.v_cache_scale, scores, attention,
            control, kQueryHeads, layer.kv_heads, layer.head_dimension,
            capacity, !layer.global, stream_);
      } else {
        status = internal::LaunchOnlineAttentionDecodeFp8Sm120(
            q_norm, layer.key_cache_fp8, layer.value_cache_fp8,
            layer.k_cache_scale, layer.v_cache_scale, scores, attention,
            control, kQueryHeads, layer.kv_heads, layer.head_dimension,
            capacity, !layer.global, stream_);
      }
    } else {
      status = internal::LaunchAppendKvControlled(
          k_norm, v_norm, layer.key_cache_bf16, layer.value_cache_bf16,
          control, layer.kv_heads, layer.head_dimension, capacity,
          !layer.global, stream_);
      if (!status.ok()) return status;
      status = internal::LaunchLocalAttentionDecodeControlled(
          q_norm, layer.key_cache_bf16, layer.value_cache_bf16, scores,
          attention, control, kQueryHeads, layer.kv_heads,
          layer.head_dimension, capacity, !layer.global, stream_);
    }
    if (!status.ok()) return status;
    return LaunchDecodeSuffix(layer, nullptr, nullptr);
  }

  [[nodiscard]] Status LaunchFullDecodeGraphBody() {
    HostDecodeState* host = host_decode_state();
    auto* control = Pointer<internal::DecodeControl>(
        workspace_, offsets_.decode_control);
    cudaError_t error = cudaMemcpyAsync(
        control, &host->control, sizeof(host->control), cudaMemcpyHostToDevice,
        stream_);
    if (error != cudaSuccess) {
      return CudaFailure("copy decode graph control", error);
    }
    if (sampling_.enabled && sampling_.repetition_penalty != 1.0F) {
      Status mark_status = internal::LaunchMarkControlledRepetitionToken(
          control,
          Pointer<std::uint32_t>(workspace_, offsets_.repetition_mask),
          stream_);
      if (!mark_status.ok()) return mark_status;
    }
    float* hidden = Pointer<float>(workspace_, offsets_.hidden_a);
    ControlledEmbeddingKernel<<<
        static_cast<unsigned>((kHidden + kThreads - 1U) / kThreads), kThreads,
        0, stream_>>>(model_.embedding(), control, hidden);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
      return CudaFailure("launch controlled embedding", error);
    }
    for (const LayerBinding& layer : model_.layers()) {
      Status status = LaunchControlledDecodeLayer(layer);
      if (!status.ok()) return status;
    }
    float* normalized = Pointer<float>(workspace_, offsets_.normalized);
    Status status = internal::LaunchRmsNormBf16(
        hidden, model_.final_norm(), normalized, 1U, kHidden, kEpsilon, stream_);
    if (!status.ok()) return status;
    auto* selected = Pointer<std::uint32_t>(workspace_, offsets_.selected);
    float* sampling_logits = sampling_.enabled
                                 ? Pointer<float>(workspace_, offsets_.logits)
                                 : nullptr;
    status = internal::LaunchFusedOutputHeadCandidates(
        model_.embedding(), normalized,
        Pointer<std::uint32_t>(workspace_, offsets_.suppressed), 0U, control,
        Pointer<ArgmaxValue>(workspace_, offsets_.output_candidates),
        sampling_logits, stream_);
    if (!status.ok()) return status;
    if (sampling_.enabled) {
      status = SelectSampledToken(sampling_logits, selected, control);
      if (!status.ok()) return status;
    } else {
      status = internal::LaunchOutputHeadCandidateArgmax(
          Pointer<ArgmaxValue>(workspace_, offsets_.output_candidates),
          selected, stream_);
      if (!status.ok()) return status;
    }
    error = cudaMemcpyAsync(&host->selected_token, selected,
                            sizeof(host->selected_token),
                            cudaMemcpyDeviceToHost, stream_);
    return error == cudaSuccess
               ? Status::Ok()
               : CudaFailure("copy controlled selected token", error);
  }

  [[nodiscard]] Status PrepareDecodeGraphs() {
    cudaError_t error = cudaStreamSynchronize(stream_);
    if (error != cudaSuccess) {
      return CudaFailure("synchronize before decode graph capture", error);
    }
    for (std::size_t index = 0; index < model_.layers().size(); ++index) {
      const LayerBinding& layer = model_.layers()[index];
      Status status = CaptureDecodeGraph(
          decode_prefix_graphs_[index],
          [this, &layer]() { return LaunchDecodePrefix(layer); },
          "capture decode prefix graph");
      if (!status.ok()) return status;
      status = CaptureDecodeGraph(
          decode_suffix_graphs_[index],
          [this, &layer]() {
            return LaunchDecodeSuffix(layer, nullptr, nullptr);
          },
          "capture decode suffix graph");
      if (!status.ok()) return status;
    }
    *host_decode_state() = HostDecodeState{};
    return CaptureDecodeGraph(
        full_decode_graph_, [this]() { return LaunchFullDecodeGraphBody(); },
        "capture full decode graph");
  }

  [[nodiscard]] Status LaunchDecodePrefix(const LayerBinding& layer,
                                          bool normalize_query_key = true) {
    float* hidden_a = Pointer<float>(workspace_, offsets_.hidden_a);
    auto* fp8_activation =
        Pointer<std::uint8_t>(workspace_, offsets_.fp8_activation);
    float* fp8_scale = Pointer<float>(workspace_, offsets_.fp8_scale);
    float* q = Pointer<float>(workspace_, offsets_.q);
    float* k = Pointer<float>(workspace_, offsets_.k);
    float* v = Pointer<float>(workspace_, offsets_.v);
    float* q_norm = Pointer<float>(workspace_, offsets_.q_norm);
    float* k_norm = Pointer<float>(workspace_, offsets_.k_norm);
    float* v_norm = Pointer<float>(workspace_, offsets_.v_norm);

    Status status = internal::LaunchRmsNormFp8TokenQuantizationBatch(
        hidden_a, layer.input_norm, fp8_activation, fp8_scale, 1U, kHidden,
        kEpsilon, stream_);
    if (!status.ok()) return status;
    status = LaunchFp8QkvProjection(
        fp8_activation, fp8_scale, layer.q, q, layer.k, k,
        layer.global ? nullptr : &layer.v, v, stream_);
    if (!status.ok()) return status;
    if (layer.global) {
      const cudaError_t error = cudaMemcpyAsync(
          v, k, layer.kv_elements * sizeof(float), cudaMemcpyDeviceToDevice,
          stream_);
      if (error != cudaSuccess) {
        return CudaFailure("reuse global K projection for V", error);
      }
    }
    status = LaunchRoundBf16(v, layer.kv_elements, stream_);
    if (!status.ok()) return status;
    status = internal::LaunchRmsNormBf16(
        v, nullptr, v_norm, layer.kv_heads, layer.head_dimension, kEpsilon,
        stream_);
    if (!status.ok() || !normalize_query_key) return status;
    for (const Status next : {
             LaunchRoundBf16(q, layer.query_elements, stream_),
             LaunchRoundBf16(k, layer.kv_elements, stream_),
             internal::LaunchRmsNormBf16(q, layer.q_norm, q_norm, kQueryHeads,
                                         layer.head_dimension, kEpsilon,
                                         stream_),
             internal::LaunchRmsNormBf16(k, layer.k_norm, k_norm,
                                         layer.kv_heads, layer.head_dimension,
                                         kEpsilon, stream_),
         }) {
      if (!next.ok()) return next;
    }
    return Status::Ok();
  }

  [[nodiscard]] Status LaunchDecodeSuffix(
      const LayerBinding& layer, const LayerStateCapture* capture,
      float* host_state) {
    float* hidden_a = Pointer<float>(workspace_, offsets_.hidden_a);
    float* hidden_b = Pointer<float>(workspace_, offsets_.hidden_b);
    float* normalized = Pointer<float>(workspace_, offsets_.normalized);
    float* attention = Pointer<float>(workspace_, offsets_.attention);
    auto* o_activation =
        Pointer<std::uint8_t>(workspace_, offsets_.o_activation);
    float* o_scale = Pointer<float>(workspace_, offsets_.o_scale);
    float* projection = Pointer<float>(workspace_, offsets_.projection);
    float* post_norm = Pointer<float>(workspace_, offsets_.post_norm);
    auto* mlp_packed =
        Pointer<std::uint8_t>(workspace_, offsets_.mlp_packed);
    auto* mlp_scales =
        Pointer<std::uint8_t>(workspace_, offsets_.mlp_scales);
    float* gate = Pointer<float>(workspace_, offsets_.gate);
    float* up = Pointer<float>(workspace_, offsets_.up);
    float* product = Pointer<float>(workspace_, offsets_.product);
    auto* down_packed =
        Pointer<std::uint8_t>(workspace_, offsets_.down_packed);
    auto* down_scales =
        Pointer<std::uint8_t>(workspace_, offsets_.down_scales);
    const auto capture_values =
        [this, capture, host_state](std::size_t offset, const float* source,
                                    std::size_t elements,
                                    const char* label) -> Status {
      if (capture == nullptr) return Status::Ok();
      const cudaError_t error = cudaMemcpyAsync(
          host_state + offset, source, elements * sizeof(float),
          cudaMemcpyDeviceToHost, stream_);
      return error == cudaSuccess ? Status::Ok() : CudaFailure(label, error);
    };
    const auto capture_hidden =
        [&capture_values](std::size_t offset, const float* source,
                          const char* label) -> Status {
      return capture_values(offset, source, static_cast<std::size_t>(kHidden),
                            label);
    };

    Status status = LaunchRoundBf16(attention, layer.query_elements, stream_);
    if (!status.ok()) return status;
    if (capture != nullptr) {
      const cudaError_t error = cudaMemcpyAsync(
          host_state + capture->attention_context, attention,
          capture->attention_elements * sizeof(float), cudaMemcpyDeviceToHost,
          stream_);
      if (error != cudaSuccess) {
        return CudaFailure("copy attention context state", error);
      }
    }
    status = internal::LaunchFp8ReferenceTokenQuantization(
        attention, o_activation, o_scale, layer.query_elements, stream_);
    if (!status.ok()) return status;
    status = LaunchFp8Projection(o_activation, o_scale, layer.o, projection, stream_);
    if (!status.ok()) return status;
    status = LaunchRoundBf16(projection, kHidden, stream_);
    if (!status.ok()) return status;
    if (capture != nullptr) {
      status = capture_hidden(capture->attention_output, projection,
                              "copy attention output state");
      if (!status.ok()) return status;
    }
    status = internal::LaunchRmsNormResidualBf16(
        projection, layer.post_attention_norm, hidden_a, post_norm, hidden_b,
        1U, kHidden, kEpsilon, nullptr, stream_);
    if (!status.ok()) return status;
    if (capture != nullptr) {
      status = capture_hidden(capture->post_attention_norm, post_norm,
                              "copy post-attention norm state");
      if (!status.ok()) return status;
      status = capture_hidden(capture->post_attention_residual, hidden_b,
                              "copy post-attention residual state");
      if (!status.ok()) return status;
    }
    if (capture == nullptr) {
      status = internal::LaunchRmsNormNvfp4ActivationQuantizationBatch(
          hidden_b, layer.pre_mlp_norm, mlp_packed, mlp_scales, 1U, kHidden,
          kEpsilon, layer.gate.input_divisor, stream_);
    } else {
      status = internal::LaunchRmsNormBf16(
          hidden_b, layer.pre_mlp_norm, normalized, 1U, kHidden, kEpsilon,
          stream_);
      if (!status.ok()) return status;
      status = capture_hidden(capture->pre_feedforward_norm, normalized,
                              "copy pre-feedforward norm state");
      if (!status.ok()) return status;
      status = internal::LaunchNvfp4ReferenceActivationQuantization(
          normalized, mlp_packed, mlp_scales, kHidden,
          layer.gate.input_divisor, stream_);
    }
    if (!status.ok()) return status;
    status = LaunchNvfp4Projection(mlp_packed, mlp_scales, layer.gate, gate,
                                   stream_);
    if (!status.ok()) return status;
    status = LaunchNvfp4Projection(mlp_packed, mlp_scales, layer.up, up,
                                   stream_);
    if (!status.ok()) return status;
    if (capture == nullptr) {
      status = internal::LaunchGatedGeluNvfp4ActivationQuantization(
          gate, up, down_packed, down_scales, kIntermediate,
          layer.down.input_divisor, stream_);
    } else {
      status = LaunchRoundBf16(gate, kIntermediate, stream_);
      if (!status.ok()) return status;
      status = LaunchRoundBf16(up, kIntermediate, stream_);
      if (!status.ok()) return status;
      status = internal::LaunchGeluTanhProduct(gate, up, product, kIntermediate,
                                               stream_);
      if (!status.ok()) return status;
      status = LaunchRoundBf16(product, kIntermediate, stream_);
      if (!status.ok()) return status;
      status = capture_values(capture->gate, gate, kIntermediate,
                              "copy MLP gate state");
      if (!status.ok()) return status;
      status = capture_values(capture->up, up, kIntermediate,
                              "copy MLP up state");
      if (!status.ok()) return status;
      status = capture_values(capture->gelu_product, product, kIntermediate,
                              "copy MLP GELU product state");
      if (!status.ok()) return status;
      status = internal::LaunchNvfp4ReferenceActivationQuantization(
          product, down_packed, down_scales, kIntermediate,
          layer.down.input_divisor, stream_);
    }
    if (!status.ok()) return status;
    status = LaunchNvfp4Projection(down_packed, down_scales, layer.down,
                                   projection, stream_);
    if (!status.ok()) return status;
    status = LaunchRoundBf16(projection, kHidden, stream_);
    if (!status.ok()) return status;
    if (capture != nullptr) {
      status = capture_hidden(capture->mlp_output, projection,
                              "copy MLP output state");
      if (!status.ok()) return status;
    }
    status = internal::LaunchRmsNormResidualBf16(
        projection, layer.post_mlp_norm, hidden_b, post_norm, hidden_a, 1U,
        kHidden, kEpsilon, layer.layer_scalar, stream_);
    if (!status.ok()) return status;
    if (capture != nullptr) {
      status = capture_hidden(capture->post_feedforward_norm, post_norm,
                              "copy post-feedforward norm state");
      if (!status.ok()) return status;
    }
    if (capture != nullptr) {
      status = capture_hidden(capture->hidden, hidden_a,
                              "copy layer hidden state");
      if (!status.ok()) return status;
    }
    return Status::Ok();
  }

  [[nodiscard]] Status RunLayer(std::size_t layer_index,
                                const LayerBinding& layer, std::uint64_t position,
                                const LayerStateCapture* capture,
                                float* host_state) {
    const NvtxRange range("gem16.decode.layer");
    float* q_norm = Pointer<float>(workspace_, offsets_.q_norm);
    float* k_norm = Pointer<float>(workspace_, offsets_.k_norm);
    float* v_norm = Pointer<float>(workspace_, offsets_.v_norm);
    float* scores = Pointer<float>(workspace_, offsets_.scores);
    float* attention = Pointer<float>(workspace_, offsets_.attention);

    Status status;
    if (capture == nullptr) {
      const cudaError_t error =
          cudaGraphLaunch(decode_prefix_graphs_[layer_index].get(), stream_);
      if (error != cudaSuccess) return CudaFailure("launch decode prefix graph", error);
    } else {
      status = LaunchDecodePrefix(layer);
      if (!status.ok()) return status;
    }
    if (layer.global) {
      status = internal::LaunchProportionalRotaryEmbedding(
          q_norm, kQueryHeads, layer.head_dimension, 0.25, position, 1000000.0, 1.0, stream_);
      if (!status.ok()) return status;
      status = internal::LaunchProportionalRotaryEmbedding(
          k_norm, layer.kv_heads, layer.head_dimension, 0.25, position, 1000000.0, 1.0,
          stream_);
    } else {
      status = internal::LaunchRotaryEmbedding(q_norm, kQueryHeads, layer.head_dimension,
                                               layer.head_dimension, position, 10000.0, stream_);
      if (!status.ok()) return status;
      status = internal::LaunchRotaryEmbedding(k_norm, layer.kv_heads, layer.head_dimension,
                                               layer.head_dimension, position, 10000.0, stream_);
    }
    if (!status.ok()) return status;
    status = LaunchRoundBf16(q_norm, layer.query_elements, stream_);
    if (!status.ok()) return status;
    status = LaunchRoundBf16(k_norm, layer.kv_elements, stream_);
    if (!status.ok()) return status;
    if (capture != nullptr) {
      const std::size_t kv_bytes = capture->kv_elements * sizeof(float);
      cudaError_t capture_error = cudaMemcpyAsync(
          host_state + capture->key, k_norm, kv_bytes, cudaMemcpyDeviceToHost,
          stream_);
      if (capture_error != cudaSuccess) {
        return CudaFailure("copy layer K input state", capture_error);
      }
      capture_error = cudaMemcpyAsync(
          host_state + capture->value, v_norm, kv_bytes,
          cudaMemcpyDeviceToHost, stream_);
      if (capture_error != cudaSuccess) {
        return CudaFailure("copy layer V input state", capture_error);
      }
    }
    const std::uint64_t cache_capacity =
        layer.global ? max_context_ : std::min(max_context_, kSlidingWindow);
    const std::uint64_t cache_slot = layer.global ? position : position % cache_capacity;
    const std::uint64_t attention_tokens =
        layer.global ? position + 1U : std::min(position + 1U, cache_capacity);
    const std::uint64_t first_slot =
        layer.global || position + 1U <= cache_capacity
            ? 0U
            : (position + 1U) % cache_capacity;
    if (kv_cache_mode_ == KvCacheMode::kCheckpointFp8) {
      status = internal::LaunchAppendKvFp8(
          k_norm, v_norm, layer.key_cache_fp8, layer.value_cache_fp8,
          layer.k_cache_scale, layer.v_cache_scale, cache_slot, layer.kv_heads,
          layer.head_dimension, stream_);
      if (!status.ok()) return status;
      status = internal::LaunchLocalAttentionDecodeFp8(
          q_norm, layer.key_cache_fp8, layer.value_cache_fp8,
          layer.k_cache_scale, layer.v_cache_scale, scores, attention,
          kQueryHeads, layer.kv_heads, layer.head_dimension, attention_tokens,
          stream_, cache_capacity, first_slot);
    } else {
      status = internal::LaunchAppendKv(
          k_norm, v_norm, layer.key_cache_bf16, layer.value_cache_bf16,
          cache_slot, layer.kv_heads, layer.head_dimension, stream_);
      if (!status.ok()) return status;
      status = internal::LaunchLocalAttentionDecode(
          q_norm, layer.key_cache_bf16, layer.value_cache_bf16, scores,
          attention, kQueryHeads, layer.kv_heads, layer.head_dimension,
          attention_tokens, stream_, cache_capacity, first_slot);
    }
    if (!status.ok()) return status;
    if (capture == nullptr) {
      const cudaError_t error =
          cudaGraphLaunch(decode_suffix_graphs_[layer_index].get(), stream_);
      return error == cudaSuccess
                 ? Status::Ok()
                 : CudaFailure("launch decode suffix graph", error);
    }
    return LaunchDecodeSuffix(layer, capture, host_state);
  }

  internal::LoadedTargetModel model_;
  internal::AssistantModel assistant_;
  DeviceAllocation cache_;
  DeviceAllocation workspace_;
  DeviceAllocation prefill_workspace_;
  DeviceAllocation mtp_workspace_;
  PinnedHostAllocation decode_host_state_;
  PinnedHostAllocation mtp_host_result_;
  WorkspaceOffsets offsets_{};
  PrefillOffsets prefill_offsets_{};
  MtpWorkspaceOffsets mtp_offsets_{};
  std::array<GraphExecutable, kLayers> decode_prefix_graphs_{};
  std::array<GraphExecutable, kLayers> decode_suffix_graphs_{};
  GraphExecutable full_decode_graph_;
  cudaStream_t stream_ = nullptr;
  std::uint64_t max_context_ = 0;
  std::uint64_t prefill_chunk_tokens_ = kMinimumPrefillChunkTokens;
  std::uint64_t decode_graph_device_bytes_ = 0;
  std::uint64_t assistant_device_memory_delta_bytes_ = 0;
  std::uint32_t mtp_draft_tokens_ = 0U;
  const float* latest_target_hidden_ = nullptr;
  std::uint64_t sampling_step_ = 0;
  std::size_t sampling_sort_workspace_bytes_ = 0;
  std::uint32_t suppressed_token_count_ = 0;
  std::uint32_t mtp_stop_token_count_ = 0;
  KvCacheMode kv_cache_mode_ = KvCacheMode::kCheckpointFp8;
  SamplingOptions sampling_{};
};

double Milliseconds(std::chrono::steady_clock::duration duration) {
  return std::chrono::duration<double, std::milli>(duration).count();
}

double Percentile(std::vector<double> sorted, double quantile) {
  if (sorted.empty()) return 0.0;
  std::sort(sorted.begin(), sorted.end());
  const double rank = quantile * static_cast<double>(sorted.size() - 1U);
  const std::size_t lower = static_cast<std::size_t>(rank);
  const std::size_t upper = std::min(lower + 1U, sorted.size() - 1U);
  const double fraction = rank - static_cast<double>(lower);
  return sorted[lower] + fraction * (sorted[upper] - sorted[lower]);
}

double StudentTCritical95(std::size_t degrees_of_freedom) {
  constexpr std::array values = {
      0.0, 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365,
      2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131,
      2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069,
      2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042};
  return degrees_of_freedom < values.size() ? values[degrees_of_freedom] : 1.96;
}

BenchmarkDistribution Summarize(std::span<const double> samples) {
  BenchmarkDistribution summary;
  summary.sample_count = static_cast<std::uint64_t>(samples.size());
  if (samples.empty()) return summary;
  summary.minimum = *std::min_element(samples.begin(), samples.end());
  summary.maximum = *std::max_element(samples.begin(), samples.end());
  summary.mean = std::accumulate(samples.begin(), samples.end(), 0.0) /
                 static_cast<double>(samples.size());
  std::vector<double> values(samples.begin(), samples.end());
  summary.median = Percentile(values, 0.5);
  summary.p95 = Percentile(values, 0.95);
  summary.p99 = Percentile(std::move(values), 0.99);
  if (samples.size() > 1U) {
    double squared_deviation = 0.0;
    for (const double value : samples) {
      const double deviation = value - summary.mean;
      squared_deviation += deviation * deviation;
    }
    summary.standard_deviation =
        std::sqrt(squared_deviation / static_cast<double>(samples.size() - 1U));
    const double margin = StudentTCritical95(samples.size() - 1U) *
                          summary.standard_deviation /
                          std::sqrt(static_cast<double>(samples.size()));
    summary.confidence_95_low = summary.mean - margin;
    summary.confidence_95_high = summary.mean + margin;
  } else {
    summary.confidence_95_low = summary.mean;
    summary.confidence_95_high = summary.mean;
  }
  return summary;
}

std::uint64_t UpdateTokenChecksum(std::uint64_t checksum, std::uint32_t token) {
  constexpr std::uint64_t kFnvPrime = 1099511628211ULL;
  for (unsigned shift = 0; shift < 32U; shift += 8U) {
    checksum ^= static_cast<std::uint8_t>(token >> shift);
    checksum *= kFnvPrime;
  }
  return checksum;
}

void WriteDistributionJson(std::ostream& output,
                           const BenchmarkDistribution& distribution) {
  output << "{\"sample_count\":" << distribution.sample_count
         << ",\"mean\":" << distribution.mean
         << ",\"median\":" << distribution.median
         << ",\"standard_deviation\":" << distribution.standard_deviation
         << ",\"minimum\":" << distribution.minimum
         << ",\"maximum\":" << distribution.maximum
         << ",\"p95\":" << distribution.p95
         << ",\"p99\":" << distribution.p99
         << ",\"confidence_95\":[" << distribution.confidence_95_low << ','
         << distribution.confidence_95_high << "]}";
}

Status WriteStateDump(const std::filesystem::path& path, std::uint64_t position,
                      KvCacheMode kv_cache_mode,
                      std::span<const float> captured_state) {
  if constexpr (std::endian::native != std::endian::little) {
    return Error(StatusCode::kUnsupported,
                 "state dumps currently require a little-endian host");
  }
  const StateCaptureLayout layout = MakeStateCaptureLayout();
  if (captured_state.size() != layout.elements) {
    return Error(StatusCode::kInternal, "captured state has invalid size");
  }
  std::ofstream dump(path, std::ios::binary | std::ios::trunc);
  if (!dump) return Error(StatusCode::kIoError, "cannot open layer-state dump");

  constexpr std::array<char, 8> kMagic = {'G', '1', '6', 'S', 'T', '0', '0', '1'};
  const auto write = [&dump](const auto& value) {
    dump.write(reinterpret_cast<const char*>(&value),
               static_cast<std::streamsize>(sizeof(value)));
  };
  dump.write(kMagic.data(), static_cast<std::streamsize>(kMagic.size()));
  const std::uint32_t version = 5U;
  const std::uint32_t layer_count = static_cast<std::uint32_t>(kLayers);
  const std::uint64_t hidden_elements = kHidden;
  const std::uint64_t total_elements =
      static_cast<std::uint64_t>(captured_state.size());
  const std::uint32_t path_id = 0U;
  const std::uint32_t kv_cache_mode_id =
      kv_cache_mode == KvCacheMode::kCheckpointFp8 ? 0U : 1U;
  write(version);
  write(layer_count);
  write(position);
  write(hidden_elements);
  write(total_elements);
  write(path_id);
  write(kv_cache_mode_id);

  for (std::size_t layer = 0; layer < kLayers; ++layer) {
    const LayerStateCapture& capture = layout.layers[layer];
    const std::uint32_t layer_index = static_cast<std::uint32_t>(layer);
    const std::uint32_t flags = layer % 6U == 5U ? 1U : 0U;
    const std::uint64_t kv_elements =
        static_cast<std::uint64_t>(capture.kv_elements);
    write(layer_index);
    write(flags);
    write(kv_elements);
    dump.write(
        reinterpret_cast<const char*>(
            captured_state.data() + capture.attention_context),
        static_cast<std::streamsize>(
            capture.attention_elements * sizeof(float)));
    for (const auto [offset, elements] :
         {std::pair{capture.attention_output,
                    static_cast<std::size_t>(kHidden)},
          std::pair{capture.post_attention_norm,
                    static_cast<std::size_t>(kHidden)},
          std::pair{capture.post_attention_residual,
                    static_cast<std::size_t>(kHidden)},
          std::pair{capture.pre_feedforward_norm,
                    static_cast<std::size_t>(kHidden)},
          std::pair{capture.gate, static_cast<std::size_t>(kIntermediate)},
          std::pair{capture.up, static_cast<std::size_t>(kIntermediate)},
          std::pair{capture.gelu_product,
                    static_cast<std::size_t>(kIntermediate)},
          std::pair{capture.mlp_output, static_cast<std::size_t>(kHidden)},
          std::pair{capture.post_feedforward_norm,
                    static_cast<std::size_t>(kHidden)},
          std::pair{capture.hidden, static_cast<std::size_t>(kHidden)},
          std::pair{capture.key, capture.kv_elements},
          std::pair{capture.value, capture.kv_elements}}) {
      dump.write(reinterpret_cast<const char*>(captured_state.data() + offset),
                 static_cast<std::streamsize>(elements * sizeof(float)));
    }
  }
  return dump.good() ? Status::Ok()
                     : Error(StatusCode::kIoError,
                             "failed to write layer-state dump");
}

}  // namespace

struct ConversationSession::Impl {
  InferenceEngine engine;
  std::vector<std::uint32_t> cached_token_ids;
  std::vector<std::uint32_t> stop_token_ids;
  std::uint64_t max_context_tokens = 0U;
  KvCacheMode kv_cache_mode = KvCacheMode::kCheckpointFp8;
  double model_load_milliseconds = 0.0;
  SamplingOptions sampling;
  bool poisoned = false;
};

ConversationSession::ConversationSession(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

ConversationSession::ConversationSession(ConversationSession&&) noexcept =
    default;
ConversationSession& ConversationSession::operator=(
    ConversationSession&&) noexcept = default;
ConversationSession::~ConversationSession() = default;

Result<ConversationSession> ConversationSession::Create(
    const ConversationSessionOptions& options) {
  if (options.model_directory.empty()) {
    return Error(StatusCode::kInvalidArgument,
                 "conversation session requires --model");
  }
  if (options.max_context_tokens == 0U ||
      options.max_context_tokens > kMaximumContext) {
    return Error(StatusCode::kUnsupported,
                 "the hybrid KV cache supports 1..262144 tokens");
  }
  for (const std::uint32_t token : options.stop_token_ids) {
    if (token >= kVocabulary) {
      return Error(StatusCode::kInvalidArgument,
                   "stop token ID exceeds vocabulary");
    }
  }
  for (const std::uint32_t token : options.suppressed_token_ids) {
    if (token >= kVocabulary) {
      return Error(StatusCode::kInvalidArgument,
                   "suppressed token ID exceeds vocabulary");
    }
  }

  auto impl = std::make_unique<Impl>();
  impl->stop_token_ids = options.stop_token_ids;
  impl->max_context_tokens = options.max_context_tokens;
  impl->kv_cache_mode = options.kv_cache_mode;
  impl->sampling = options.sampling;
  impl->cached_token_ids.reserve(
      static_cast<std::size_t>(options.max_context_tokens));
  const auto load_start = std::chrono::steady_clock::now();
  Status status = impl->engine.Initialize(options.model_directory,
                                         options.max_context_tokens,
                                         options.kv_cache_mode,
                                         options.sampling);
  if (!status.ok()) return status;
  status = impl->engine.SetSuppressedTokens(options.suppressed_token_ids);
  if (!status.ok()) return status;
  impl->model_load_milliseconds =
      Milliseconds(std::chrono::steady_clock::now() - load_start);
  return ConversationSession(std::move(impl));
}

Result<GreedyInferenceResult> ConversationSession::Generate(
    std::span<const std::uint32_t> full_prompt_token_ids,
    std::uint64_t max_generated_tokens,
    GeneratedTokenCallback generated_token_callback,
    void* generated_token_callback_context) {
  if (impl_ == nullptr) {
    return Error(StatusCode::kInternal,
                 "conversation session was moved from");
  }
  if (impl_->poisoned) {
    return Error(StatusCode::kInternal,
                 "conversation session cannot continue after an inference failure");
  }
  if (full_prompt_token_ids.empty()) {
    return Error(StatusCode::kInvalidArgument,
                 "conversation turn requires prompt token IDs");
  }
  if (max_generated_tokens == 0U) {
    return Error(StatusCode::kInvalidArgument,
                 "--max-tokens must be positive");
  }
  if (full_prompt_token_ids.size() > impl_->max_context_tokens ||
      max_generated_tokens - 1U >
          impl_->max_context_tokens - full_prompt_token_ids.size()) {
    return Error(StatusCode::kInvalidArgument,
                 "conversation prompt plus generated positions exceed --max-context");
  }
  for (const std::uint32_t token : full_prompt_token_ids) {
    if (token >= kVocabulary) {
      return Error(StatusCode::kInvalidArgument,
                   "input token ID exceeds vocabulary");
    }
  }
  const std::size_t comparable_tokens = std::min(
      impl_->cached_token_ids.size(), full_prompt_token_ids.size());
  const auto mismatch = std::mismatch(
      impl_->cached_token_ids.begin(),
      impl_->cached_token_ids.begin() + comparable_tokens,
      full_prompt_token_ids.begin());
  if (impl_->cached_token_ids.size() > full_prompt_token_ids.size() ||
      mismatch.first !=
          impl_->cached_token_ids.begin() + comparable_tokens) {
    const std::size_t mismatch_index = static_cast<std::size_t>(
        mismatch.first - impl_->cached_token_ids.begin());
    const std::string cached_id =
        mismatch_index < impl_->cached_token_ids.size()
            ? std::to_string(impl_->cached_token_ids[mismatch_index])
            : "end";
    const std::string rendered_id =
        mismatch_index < full_prompt_token_ids.size()
            ? std::to_string(full_prompt_token_ids[mismatch_index])
            : "end";
    return Error(
        StatusCode::kInvalidArgument,
        "rendered conversation differs from the resident KV-cache prefix at token " +
            std::to_string(mismatch_index) + " (cached " + cached_id +
            ", rendered " + rendered_id + ")");
  }
  const std::size_t prefix_tokens = impl_->cached_token_ids.size();
  const std::span<const std::uint32_t> suffix =
      full_prompt_token_ids.subspan(prefix_tokens);
  if (suffix.empty()) {
    return Error(StatusCode::kInvalidArgument,
                 "conversation turn adds no uncached prompt tokens");
  }

  GreedyInferenceResult result;
  result.output_token_ids.reserve(
      static_cast<std::size_t>(max_generated_tokens));
  result.kv_cache_mode = impl_->kv_cache_mode;
  result.sampling = impl_->sampling;
  result.decode_graphs = true;
  result.model_load_milliseconds = impl_->model_load_milliseconds;
  result.weight_arena_bytes = impl_->engine.weight_bytes();
  result.kv_cache_bytes = impl_->engine.cache_bytes();
  result.workspace_bytes = impl_->engine.workspace_bytes();
  result.decode_graph_device_bytes =
      impl_->engine.decode_graph_device_bytes();
  result.prefill_chunk_tokens = impl_->engine.prefill_chunk_tokens();
  result.max_context_tokens = impl_->max_context_tokens;
  result.packed_weight_source_layout_direct = false;
  result.token_loop_allocations = false;
  result.benchmark_qualified = false;

  const auto prompt_start = std::chrono::steady_clock::now();
  auto prefilled = impl_->engine.PrefillAt(suffix, prefix_tokens);
  if (!prefilled.ok()) {
    impl_->poisoned = true;
    return prefilled.status();
  }
  impl_->cached_token_ids.insert(impl_->cached_token_ids.end(),
                                 suffix.begin(), suffix.end());
  std::uint32_t next_token = prefilled.value();
  result.prompt_milliseconds = Milliseconds(
      std::chrono::steady_clock::now() - prompt_start);
  result.output_token_ids.push_back(next_token);
  if (generated_token_callback != nullptr) {
    Status status = generated_token_callback(
        generated_token_callback_context, next_token);
    if (!status.ok()) {
      impl_->poisoned = true;
      return status;
    }
  }
  if (std::find(impl_->stop_token_ids.begin(), impl_->stop_token_ids.end(),
                next_token) != impl_->stop_token_ids.end()) {
    result.stopped = true;
    result.stop_token_id = next_token;
  }

  const auto decode_start = std::chrono::steady_clock::now();
  for (std::uint64_t generated = 1U;
       generated < max_generated_tokens && !result.stopped; ++generated) {
    const std::uint64_t position = impl_->cached_token_ids.size();
    const std::uint32_t input_token = next_token;
    auto forwarded =
        impl_->engine.Forward(input_token, position, true);
    if (!forwarded.ok()) {
      impl_->poisoned = true;
      return forwarded.status();
    }
    impl_->cached_token_ids.push_back(input_token);
    next_token = forwarded.value();
    result.output_token_ids.push_back(next_token);
    if (generated_token_callback != nullptr) {
      Status status = generated_token_callback(
          generated_token_callback_context, next_token);
      if (!status.ok()) {
        impl_->poisoned = true;
        return status;
      }
    }
    if (std::find(impl_->stop_token_ids.begin(),
                  impl_->stop_token_ids.end(), next_token) !=
        impl_->stop_token_ids.end()) {
      result.stopped = true;
      result.stop_token_id = next_token;
    }
  }
  result.decode_milliseconds = Milliseconds(
      std::chrono::steady_clock::now() - decode_start);
  const std::uint64_t measured_decode_tokens =
      result.output_token_ids.size() - 1U;
  if (measured_decode_tokens != 0U && result.decode_milliseconds > 0.0) {
    result.decode_tokens_per_second =
        static_cast<double>(measured_decode_tokens) * 1000.0 /
        result.decode_milliseconds;
  }
  return result;
}

std::uint64_t ConversationSession::cached_token_count() const {
  return impl_ == nullptr ? 0U : impl_->cached_token_ids.size();
}

Result<GreedyInferenceResult> RunGreedyInference(const GreedyInferenceOptions& options) {
  if (options.model_directory.empty()) {
    return Error(StatusCode::kInvalidArgument, "greedy inference requires --model");
  }
  if (options.input_token_ids.empty()) {
    return Error(StatusCode::kInvalidArgument, "greedy inference requires input token IDs");
  }
  if (options.max_generated_tokens == 0U) {
    return Error(StatusCode::kInvalidArgument, "--max-tokens must be positive");
  }
  const bool teacher_forcing = !options.teacher_forced_token_ids.empty();
  const bool mtp_enabled = options.mtp_draft_tokens != 0U;
  if (mtp_enabled &&
      (options.mtp_draft_tokens != 1U && options.mtp_draft_tokens != 2U &&
       options.mtp_draft_tokens != 4U)) {
    return Error(StatusCode::kInvalidArgument,
                 "active MTP requires draft length 1, 2, or 4");
  }
  if (mtp_enabled && options.assistant_model_directory.empty()) {
    return Error(StatusCode::kInvalidArgument,
                 "active MTP requires --assistant-model");
  }
  if (options.mtp_adaptive && !mtp_enabled) {
    return Error(StatusCode::kInvalidArgument,
                 "--mtp-adaptive requires active MTP");
  }
  if (mtp_enabled &&
      (teacher_forcing || options.sampling.enabled ||
       !options.logits_dump_path.empty() || !options.state_dump_path.empty())) {
    return Error(StatusCode::kUnsupported,
                 "the MTP correctness path currently requires greedy generation without diagnostic dumps");
  }
  if (teacher_forcing && options.sampling.enabled) {
    return Error(StatusCode::kInvalidArgument,
                 "teacher-forced diagnostics require greedy prediction");
  }
  const std::uint64_t generation_steps =
      teacher_forcing
          ? static_cast<std::uint64_t>(options.teacher_forced_token_ids.size())
          : options.max_generated_tokens;
  if (options.max_context_tokens == 0U || options.max_context_tokens > kMaximumContext) {
    return Error(StatusCode::kUnsupported,
                 "the hybrid KV cache supports 1..262144 tokens");
  }
  if (options.input_token_ids.size() > options.max_context_tokens ||
      generation_steps - 1U >
          options.max_context_tokens - options.input_token_ids.size()) {
    return Error(StatusCode::kInvalidArgument,
                 "prompt plus generated decode positions exceed --max-context");
  }
  if (options.state_dump_path.empty() !=
      !options.state_dump_position.has_value()) {
    return Error(StatusCode::kInvalidArgument,
                 "--dump-state and --dump-state-position must be used together");
  }
  if (options.state_dump_position.has_value()) {
    const std::uint64_t maximum_forward_position =
        static_cast<std::uint64_t>(options.input_token_ids.size() - 1U) +
        (generation_steps - 1U);
    if (*options.state_dump_position > maximum_forward_position) {
      return Error(StatusCode::kInvalidArgument,
                   "state dump position is outside the requested inference");
    }
  }
  for (const std::uint32_t token : options.input_token_ids) {
    if (token >= kVocabulary) {
      return Error(StatusCode::kInvalidArgument, "input token ID exceeds vocabulary");
    }
  }
  for (const std::uint32_t token : options.teacher_forced_token_ids) {
    if (token >= kVocabulary) {
      return Error(StatusCode::kInvalidArgument,
                   "teacher-forced token ID exceeds vocabulary");
    }
  }
  for (const std::uint32_t token : options.stop_token_ids) {
    if (token >= kVocabulary) {
      return Error(StatusCode::kInvalidArgument, "stop token ID exceeds vocabulary");
    }
  }
  for (const std::uint32_t token : options.suppressed_token_ids) {
    if (token >= kVocabulary) {
      return Error(StatusCode::kInvalidArgument, "suppressed token ID exceeds vocabulary");
    }
  }

  PinnedHostAllocation captured_logits;
  if (!options.logits_dump_path.empty()) {
    if constexpr (std::endian::native != std::endian::little) {
      return Error(StatusCode::kUnsupported,
                   "raw full-logit dumps currently require a little-endian host");
    }
    if (generation_steps >
        std::numeric_limits<std::size_t>::max() / kVocabulary) {
      return Error(StatusCode::kInvalidArgument, "requested logit capture is too large");
    }
    Status status = captured_logits.Allocate(
        static_cast<std::size_t>(generation_steps * kVocabulary),
        "full-logit capture");
    if (!status.ok()) return status;
  }
  PinnedHostAllocation captured_state;
  if (!options.state_dump_path.empty()) {
    Status status = captured_state.Allocate(MakeStateCaptureLayout().elements,
                                            "layer-state capture");
    if (!status.ok()) return status;
  }

  const auto load_start = std::chrono::steady_clock::now();
  InferenceEngine engine;
  Status status = engine.Initialize(options.model_directory,
                                    options.max_context_tokens,
                                    options.kv_cache_mode,
                                    options.sampling,
                                    options.assistant_model_directory,
                                    options.mtp_draft_tokens);
  if (!status.ok()) return status;
  status = engine.SetSuppressedTokens(options.suppressed_token_ids);
  if (!status.ok()) return status;
  status = engine.SetMtpStopTokens(options.stop_token_ids);
  if (!status.ok()) return status;
  const auto load_end = std::chrono::steady_clock::now();

  GreedyInferenceResult result;
  result.output_token_ids.reserve(static_cast<std::size_t>(generation_steps));
  if (mtp_enabled) {
    result.mtp_proposed_token_ids.reserve(
        static_cast<std::size_t>(generation_steps * options.mtp_draft_tokens));
  }
  result.teacher_forced_token_ids = options.teacher_forced_token_ids;
  result.teacher_forcing = teacher_forcing;
  result.kv_cache_mode = options.kv_cache_mode;
  result.sampling = options.sampling;
  result.decode_graphs =
      options.state_dump_path.empty() && options.logits_dump_path.empty();
  result.model_load_milliseconds = Milliseconds(load_end - load_start);
  result.weight_arena_bytes = engine.weight_bytes();
  result.assistant_loaded = engine.assistant_loaded();
  result.assistant_source_bytes = engine.assistant_source_bytes();
  result.assistant_weight_arena_bytes = engine.assistant_weight_bytes();
  result.assistant_device_memory_delta_bytes =
      engine.assistant_device_memory_delta_bytes();
  result.assistant_tensor_count = engine.assistant_tensor_count();
  result.assistant_workspace_bytes = engine.assistant_workspace_bytes();
  result.mtp_enabled = mtp_enabled;
  result.mtp_adaptive = options.mtp_adaptive;
  result.mtp_draft_tokens = options.mtp_draft_tokens;
  result.kv_cache_bytes = engine.cache_bytes();
  result.workspace_bytes = engine.workspace_bytes();
  result.decode_graph_device_bytes = engine.decode_graph_device_bytes();
  result.prefill_chunk_tokens = engine.prefill_chunk_tokens();
  result.max_context_tokens = options.max_context_tokens;
  result.packed_weight_source_layout_direct = false;
  result.token_loop_allocations = false;
  result.benchmark_qualified = false;

  const auto prompt_start = std::chrono::steady_clock::now();
  std::uint32_t next_token = 0;
  bool state_captured = false;
  if (options.state_dump_path.empty()) {
    const std::span<float> prefill_logits =
        captured_logits.span().empty()
            ? std::span<float>()
            : captured_logits.span().first(static_cast<std::size_t>(kVocabulary));
    auto prefilled = engine.Prefill(options.input_token_ids, prefill_logits);
    if (!prefilled.ok()) return prefilled.status();
    next_token = prefilled.value();
  } else for (std::size_t index = 0; index < options.input_token_ids.size(); ++index) {
    const bool select = index + 1U == options.input_token_ids.size();
    const std::span<float> logit_capture =
        select && !captured_logits.span().empty()
            ? captured_logits.span().first(static_cast<std::size_t>(kVocabulary))
            : std::span<float>();
    const bool capture_state =
        options.state_dump_position.has_value() &&
        *options.state_dump_position == index;
    auto forwarded = engine.Forward(
        options.input_token_ids[index], index, select, logit_capture,
        capture_state ? captured_state.span() : std::span<float>());
    if (!forwarded.ok()) return forwarded.status();
    state_captured = state_captured || capture_state;
    if (select) next_token = forwarded.value();
  }
  const auto prompt_end = std::chrono::steady_clock::now();
  result.prompt_milliseconds = Milliseconds(prompt_end - prompt_start);
  result.output_token_ids.push_back(next_token);
  if (options.generated_token_callback != nullptr) {
    status = options.generated_token_callback(
        options.generated_token_callback_context, next_token);
    if (!status.ok()) return status;
  }
  if (!teacher_forcing &&
      std::find(options.stop_token_ids.begin(), options.stop_token_ids.end(), next_token) !=
      options.stop_token_ids.end()) {
    result.stopped = true;
    result.stop_token_id = next_token;
  }

  const auto decode_start = std::chrono::steady_clock::now();
  if (mtp_enabled) {
    std::uint64_t processed_position = options.input_token_ids.size() - 1U;
    internal::AdaptiveMtpScheduler adaptive_scheduler(
        options.mtp_draft_tokens, processed_position, options.mtp_adaptive);
    const auto emit_ordinary_token = [&](bool adaptive_fallback) -> Status {
      auto forwarded = engine.Forward(next_token, processed_position + 1U,
                                      true);
      if (!forwarded.ok()) return forwarded.status();
      ++processed_position;
      ++result.mtp_target_forwards;
      ++result.mtp_target_batches;
      if (adaptive_fallback) ++result.mtp_ordinary_fallback_tokens;
      next_token = forwarded.value();
      result.output_token_ids.push_back(next_token);
      if (options.generated_token_callback != nullptr) {
        Status callback_status = options.generated_token_callback(
            options.generated_token_callback_context, next_token);
        if (!callback_status.ok()) return callback_status;
      }
      if (std::find(options.stop_token_ids.begin(),
                    options.stop_token_ids.end(), next_token) !=
          options.stop_token_ids.end()) {
        result.stopped = true;
        result.stop_token_id = next_token;
      }
      return Status::Ok();
    };
    while (result.output_token_ids.size() < generation_steps &&
           !result.stopped) {
      const std::size_t remaining = static_cast<std::size_t>(
          generation_steps - result.output_token_ids.size());
      const bool adaptive_fallback =
          adaptive_scheduler.use_ordinary_fallback();
      if (remaining == 1U || adaptive_fallback) {
        status = emit_ordinary_token(adaptive_fallback);
        if (!status.ok()) return status;
        if (adaptive_fallback) adaptive_scheduler.ConsumeOrdinaryFallback();
        continue;
      }
      const std::size_t proposal_count = std::min<std::size_t>(
          adaptive_scheduler.active_drafts(), remaining - 1U);
      status = engine.GenerateAssistantDraftsDevice(
          next_token, processed_position,
          static_cast<std::uint32_t>(proposal_count));
      if (!status.ok()) return status;
      MtpGroupResult group;
      status = engine.VerifyAcceptCommitAssistantBatch(
          next_token, processed_position + 1U,
          static_cast<std::uint32_t>(proposal_count), &group);
      if (!status.ok()) return status;
      ++result.mtp_verification_groups;
      if (proposal_count == 1U) {
        ++result.mtp_d1_groups;
      } else if (proposal_count == 2U) {
        ++result.mtp_d2_groups;
      } else if (proposal_count == 4U) {
        ++result.mtp_d4_groups;
      }
      result.mtp_proposed_tokens += proposal_count;
      result.mtp_proposed_token_ids.insert(
          result.mtp_proposed_token_ids.end(), group.proposed.begin(),
          group.proposed.begin() + static_cast<std::ptrdiff_t>(proposal_count));
      result.mtp_target_forwards += proposal_count + 1U;
      ++result.mtp_target_batches;
      result.mtp_accepted_tokens += group.accepted_count;
      result.mtp_rejected_tokens += proposal_count - group.accepted_count;
      processed_position += group.output_count;
      for (std::uint32_t index = 0U; index < group.output_count; ++index) {
        next_token = group.verified[index];
        result.output_token_ids.push_back(next_token);
        if (options.generated_token_callback != nullptr) {
          status = options.generated_token_callback(
              options.generated_token_callback_context, next_token);
          if (!status.ok()) return status;
        }
        if (std::find(options.stop_token_ids.begin(),
                      options.stop_token_ids.end(), next_token) !=
            options.stop_token_ids.end()) {
          result.stopped = true;
          result.stop_token_id = next_token;
          break;
        }
      }
      if (!result.stopped) {
        adaptive_scheduler.Observe(processed_position,
                                   group.accepted_count);
      }
    }
  } else {
    for (std::uint64_t generated = 1U;
         generated < generation_steps && !result.stopped; ++generated) {
      const std::uint64_t position =
          options.input_token_ids.size() + generated - 1U;
      const std::size_t logit_offset =
          static_cast<std::size_t>(generated * kVocabulary);
      const std::span<float> logit_capture =
          captured_logits.span().empty()
              ? std::span<float>()
              : captured_logits.span().subspan(
                    logit_offset, static_cast<std::size_t>(kVocabulary));
      const bool capture_state =
          options.state_dump_position.has_value() &&
          *options.state_dump_position == position;
      const std::uint32_t input_token =
          teacher_forcing
              ? options.teacher_forced_token_ids[
                    static_cast<std::size_t>(generated - 1U)]
              : next_token;
      auto forwarded = engine.Forward(
          input_token, position, true, logit_capture,
          capture_state ? captured_state.span() : std::span<float>());
      if (!forwarded.ok()) return forwarded.status();
      state_captured = state_captured || capture_state;
      next_token = forwarded.value();
      result.output_token_ids.push_back(next_token);
      if (options.generated_token_callback != nullptr) {
        status = options.generated_token_callback(
            options.generated_token_callback_context, next_token);
        if (!status.ok()) return status;
      }
      if (!teacher_forcing &&
          std::find(options.stop_token_ids.begin(),
                    options.stop_token_ids.end(), next_token) !=
              options.stop_token_ids.end()) {
        result.stopped = true;
        result.stop_token_id = next_token;
      }
    }
  }
  if (teacher_forcing) {
    result.teacher_forced_matches = static_cast<std::uint64_t>(
        std::inner_product(
            result.output_token_ids.begin(), result.output_token_ids.end(),
            result.teacher_forced_token_ids.begin(), std::size_t{0},
            std::plus<>(), std::equal_to<>()));
  }
  const auto decode_end = std::chrono::steady_clock::now();
  result.decode_milliseconds = Milliseconds(decode_end - decode_start);
  const std::uint64_t measured_decode_tokens =
      result.output_token_ids.empty() ? 0U : result.output_token_ids.size() - 1U;
  if (measured_decode_tokens != 0U && result.decode_milliseconds > 0.0) {
    result.decode_tokens_per_second =
        static_cast<double>(measured_decode_tokens) * 1000.0 / result.decode_milliseconds;
  }
  if (!options.logits_dump_path.empty()) {
    result.logits_dump_steps = result.output_token_ids.size();
    const std::size_t dump_elements =
        static_cast<std::size_t>(result.logits_dump_steps * kVocabulary);
    std::ofstream dump(options.logits_dump_path, std::ios::binary | std::ios::trunc);
    if (!dump) {
      return Error(StatusCode::kIoError, "cannot open full-logit dump");
    }
    dump.write(reinterpret_cast<const char*>(captured_logits.span().data()),
               static_cast<std::streamsize>(dump_elements * sizeof(float)));
    if (!dump) {
      return Error(StatusCode::kIoError, "failed to write full-logit dump");
    }
    result.logits_dumped = true;
  }
  if (!options.state_dump_path.empty()) {
    if (!state_captured) {
      return Error(StatusCode::kInvalidArgument,
                   "generation stopped before the requested state dump position");
    }
    status = WriteStateDump(options.state_dump_path,
                            *options.state_dump_position, options.kv_cache_mode,
                            captured_state.span());
    if (!status.ok()) return status;
    result.state_dumped = true;
    result.state_dump_position = *options.state_dump_position;
  }
  return result;
}

Result<DecodeBenchmarkResult> RunDecodeBenchmark(
    const DecodeBenchmarkOptions& options) {
  if (options.model_directory.empty()) {
    return Error(StatusCode::kInvalidArgument, "decode benchmark requires --model");
  }
  if (options.context_tokens == 0U || options.generated_tokens == 0U ||
      options.warmup_runs == 0U || options.measured_runs == 0U) {
    return Error(StatusCode::kInvalidArgument,
                 "context, tokens, warmups, and repetitions must be positive");
  }
  const std::uint64_t planned_context =
      static_cast<std::uint64_t>(options.context_tokens) + options.generated_tokens;
  if (planned_context > kMaximumContext) {
    return Error(StatusCode::kUnsupported,
                 "the hybrid benchmark cache supports prompt plus decode up to 262144 tokens");
  }

  std::vector<std::uint32_t> prompt(options.context_tokens);
  for (std::size_t index = 0; index < prompt.size(); ++index) {
    const std::uint64_t value = static_cast<std::uint64_t>(options.prompt_seed) +
                                static_cast<std::uint64_t>(index) * 7919U;
    prompt[index] = 1000U + static_cast<std::uint32_t>(value % 9000U);
  }

  const auto load_start = std::chrono::steady_clock::now();
  InferenceEngine engine;
  Status status = engine.Initialize(options.model_directory, planned_context,
                                    options.kv_cache_mode, options.sampling);
  if (!status.ok()) return status;
  const auto load_end = std::chrono::steady_clock::now();

  const auto run_once = [&]() -> Result<DecodeBenchmarkRun> {
    Status reset_status = engine.ResetCache();
    if (!reset_status.ok()) return reset_status;

    DecodeBenchmarkRun run;
    run.inter_token_latency_milliseconds.resize(options.generated_tokens);
    const auto prompt_start = std::chrono::steady_clock::now();
    std::uint32_t next_token = 0U;
    auto prefilled = engine.Prefill(prompt);
    if (!prefilled.ok()) return prefilled.status();
    next_token = prefilled.value();
    const auto prompt_end = std::chrono::steady_clock::now();
    run.prompt_milliseconds = Milliseconds(prompt_end - prompt_start);
    run.first_output_token_id = next_token;
    run.output_token_checksum = UpdateTokenChecksum(14695981039346656037ULL, next_token);

    const auto decode_start = std::chrono::steady_clock::now();
    for (std::uint32_t generated = 0U; generated < options.generated_tokens; ++generated) {
      const std::uint64_t position =
          static_cast<std::uint64_t>(options.context_tokens) + generated;
      const auto token_start = std::chrono::steady_clock::now();
      auto forwarded = engine.Forward(next_token, position, true);
      if (!forwarded.ok()) return forwarded.status();
      const auto token_end = std::chrono::steady_clock::now();
      next_token = forwarded.value();
      run.inter_token_latency_milliseconds[generated] =
          Milliseconds(token_end - token_start);
      run.output_token_checksum = UpdateTokenChecksum(run.output_token_checksum, next_token);
    }
    const auto decode_end = std::chrono::steady_clock::now();
    run.decode_milliseconds = Milliseconds(decode_end - decode_start);
    run.decode_tokens_per_second =
        static_cast<double>(options.generated_tokens) * 1000.0 /
        run.decode_milliseconds;
    run.last_output_token_id = next_token;
    return run;
  };

  for (std::uint32_t warmup = 0U; warmup < options.warmup_runs; ++warmup) {
    auto discarded = run_once();
    if (!discarded.ok()) return discarded.status();
  }

  DecodeBenchmarkResult result;
  result.options = options;
  result.model_load_milliseconds = Milliseconds(load_end - load_start);
  result.weight_arena_bytes = engine.weight_bytes();
  result.kv_cache_bytes = engine.cache_bytes();
  result.workspace_bytes = engine.workspace_bytes();
  result.decode_graph_device_bytes = engine.decode_graph_device_bytes();
  result.prefill_chunk_tokens = engine.prefill_chunk_tokens();
  result.runs.reserve(options.measured_runs);
  std::vector<double> prompt_samples;
  std::vector<double> throughput_samples;
  std::vector<double> latency_samples;
  prompt_samples.reserve(options.measured_runs);
  throughput_samples.reserve(options.measured_runs);
  latency_samples.reserve(static_cast<std::size_t>(options.measured_runs) *
                          options.generated_tokens);
  for (std::uint32_t repetition = 0U; repetition < options.measured_runs; ++repetition) {
    auto measured = run_once();
    if (!measured.ok()) return measured.status();
    prompt_samples.push_back(measured.value().prompt_milliseconds);
    throughput_samples.push_back(measured.value().decode_tokens_per_second);
    latency_samples.insert(latency_samples.end(),
                           measured.value().inter_token_latency_milliseconds.begin(),
                           measured.value().inter_token_latency_milliseconds.end());
    result.runs.push_back(std::move(measured.value()));
  }
  result.prompt_milliseconds = Summarize(prompt_samples);
  result.decode_tokens_per_second = Summarize(throughput_samples);
  result.inter_token_latency_milliseconds = Summarize(latency_samples);
  result.deterministic_outputs = std::all_of(
      result.runs.begin(), result.runs.end(),
      [&result](const DecodeBenchmarkRun& run) {
        return run.output_token_checksum == result.runs.front().output_token_checksum;
      });
  return result;
}

Status WriteGreedyInferenceJson(const GreedyInferenceResult& result, std::ostream& output) {
  output << "{\n  \"schema_version\": 1,\n"
         << "  \"status\": \"characterization\",\n"
         << "  \"benchmark_qualified\": false,\n"
         << "  \"precision\": \"bf16_state_fp8_attention_nvfp4_mlp\",\n"
         << "  \"projection_path\": \"native_sm120\",\n"
         << "  \"decode_attention_path\": \""
         << (result.kv_cache_mode == KvCacheMode::kCheckpointFp8 &&
                     result.max_context_tokens > 512U
                 ? "fp8_online_split_gqa"
                 : "score_softmax_value_reference")
         << "\",\n"
         << "  \"kv_cache_mode\": \""
         << (result.kv_cache_mode == KvCacheMode::kCheckpointFp8
                 ? "checkpoint_fp8"
                 : "bf16_correctness")
         << "\",\n"
         << "  \"kv_cache_storage\": \""
         << (result.kv_cache_mode == KvCacheMode::kCheckpointFp8
                 ? "uint8_e4m3fn"
                 : "float32_bf16_semantics")
         << "\",\n"
         << "  \"kv_cache_layout\": \"hybrid_local_ring_global_contiguous\",\n"
         << "  \"local_attention_window\": " << kSlidingWindow << ",\n"
         << "  \"decoding_mode\": \""
         << (result.teacher_forcing
                 ? "teacher_forced"
                 : (result.sampling.enabled ? "sampled" : "greedy"))
         << "\",\n"
         << "  \"sampling\": {\"enabled\":"
         << (result.sampling.enabled ? "true" : "false")
         << ",\"temperature\":" << result.sampling.temperature
         << ",\"top_k\":" << result.sampling.top_k
         << ",\"top_p\":" << result.sampling.top_p
         << ",\"min_p\":" << result.sampling.min_p
         << ",\"repetition_penalty\":"
         << result.sampling.repetition_penalty
         << ",\"seed\":" << result.sampling.seed << "},\n"
         << "  \"fallbacks\": " << result.fallback_count << ",\n"
         << "  \"packed_weight_source_layout_direct\": "
         << (result.packed_weight_source_layout_direct ? "true" : "false") << ",\n"
         << "  \"weight_layout\": \"sm120_row8_k64\",\n"
         << "  \"weight_scale_layout\": \"sm120_row8_k64\",\n"
         << "  \"load_time_weight_swizzle\": true,\n"
         << "  \"load_time_scale_swizzle\": true,\n"
         << "  \"persistent_repack_bytes\": 0,\n"
         << "  \"token_loop_allocations\": "
         << (result.token_loop_allocations ? "true" : "false") << ",\n"
         << "  \"fused_gate_up\": false,\n"
         << "  \"fused_prefill_attention\": true,\n"
         << "  \"fp8_prefill_tile\": \"cutlass_m128n128k64\",\n"
         << "  \"nvfp4_gate_up_prefill_tile\": \"cutlass_m128n128k128\",\n"
         << "  \"nvfp4_gate_up_prefill_weight_scratch\": true,\n"
         << "  \"nvfp4_down_prefill_tile\": \"cutlass_m128n128k128\",\n"
         << "  \"fp8_prefill_pipeline_stages\": 0,\n"
         << "  \"fp8_prefill_schedule\": \"cutlass_auto\",\n"
         << "  \"local_prefill_query_heads_per_cta\": 2,\n"
         << "  \"global_prefill_query_heads_per_cta\": 4,\n"
         << "  \"local_prefill_fp8_staging\": \"fp8x16_fp8x4_bf16x2\",\n"
         << "  \"global_prefill_fp8_staging\": "
            "\"async_contiguous_fp8x16_fp8x4_bf16x2\",\n"
         << "  \"grouped_qkv_prefill\": false,\n"
         << "  \"grouped_qkv_decode\": true,\n"
         << "  \"fused_rmsnorm_boundaries\": true,\n"
         << "  \"fused_prefill_rmsnorm_fp8_quantization\": true,\n"
         << "  \"fused_prefill_rmsnorm_nvfp4_quantization\": true,\n"
         << "  \"fused_prefill_gated_gelu_nvfp4_quantization\": true,\n"
         << "  \"fused_prefill_qk_rmsnorm_rope\": true,\n"
         << "  \"prefill_rope_table\": \"precomputed_exact_max_context\",\n"
         << "  \"fused_output_head\": true,\n"
         << "  \"decode_graphs\": "
         << (result.decode_graphs ? "true" : "false") << ",\n"
         << "  \"model_load_ms\": " << result.model_load_milliseconds << ",\n"
         << "  \"prompt_ms\": " << result.prompt_milliseconds << ",\n"
         << "  \"decode_ms\": " << result.decode_milliseconds << ",\n"
         << "  \"decode_tokens_per_second\": " << result.decode_tokens_per_second << ",\n"
         << "  \"weight_arena_bytes\": " << result.weight_arena_bytes << ",\n"
         << "  \"assistant\": {\"loaded\":"
         << (result.assistant_loaded ? "true" : "false")
         << ",\"execution_enabled\":"
         << (result.mtp_enabled ? "true" : "false")
         << ",\"tensor_count\":" << result.assistant_tensor_count
         << ",\"source_bytes\":" << result.assistant_source_bytes
         << ",\"arena_bytes\":" << result.assistant_weight_arena_bytes
         << ",\"workspace_bytes\":" << result.assistant_workspace_bytes
         << ",\"attention_path\":\""
         << (!result.mtp_enabled
                 ? "disabled"
                 : result.kv_cache_mode == KvCacheMode::kCheckpointFp8
                       ? "fp8_online_split_long_reference_short"
                       : "bf16_score_softmax_value_reference")
         << "\",\"device_memory_delta_bytes\":"
         << result.assistant_device_memory_delta_bytes << "},\n"
         << "  \"mtp\": {\"enabled\":"
         << (result.mtp_enabled ? "true" : "false")
         << ",\"verification_mode\":\""
         << (result.mtp_enabled ? "batched_exact_target" : "disabled")
         << "\",\"acceptance_path\":\""
         << (result.mtp_enabled ? "gpu_accept_commit" : "disabled")
         << "\",\"host_synchronizations_per_group\":"
         << (result.mtp_enabled ? 1 : 0)
         << ",\"short_batch_projection_path\":\""
         << (result.mtp_enabled
                 ? "decode_order_fp8_qkv_nvfp4_down_t_le_5"
                 : "disabled")
         << "\",\"adaptive\":"
         << (result.mtp_adaptive ? "true" : "false")
         << ",\"draft_tokens\":" << result.mtp_draft_tokens
         << ",\"d1_groups\":" << result.mtp_d1_groups
         << ",\"d2_groups\":" << result.mtp_d2_groups
         << ",\"d4_groups\":" << result.mtp_d4_groups
         << ",\"ordinary_fallback_tokens\":"
         << result.mtp_ordinary_fallback_tokens
         << ",\"proposed_tokens\":" << result.mtp_proposed_tokens
         << ",\"accepted_tokens\":" << result.mtp_accepted_tokens
         << ",\"rejected_tokens\":" << result.mtp_rejected_tokens
         << ",\"verification_groups\":"
         << result.mtp_verification_groups
         << ",\"target_forwards\":" << result.mtp_target_forwards
         << ",\"target_batches\":" << result.mtp_target_batches
         << ",\"mean_accepted_length\":"
         << (result.mtp_verification_groups == 0U
                 ? 0.0
                 : static_cast<double>(result.mtp_accepted_tokens) /
                       static_cast<double>(result.mtp_verification_groups))
         << ",\"proposed_token_ids\":[";
  for (std::size_t index = 0; index < result.mtp_proposed_token_ids.size();
       ++index) {
    if (index != 0U) output << ',';
    output << result.mtp_proposed_token_ids[index];
  }
  output << "]},\n"
         << "  \"kv_cache_bytes\": " << result.kv_cache_bytes << ",\n"
         << "  \"workspace_bytes\": " << result.workspace_bytes << ",\n"
         << "  \"prefill_chunk_tokens\": " << result.prefill_chunk_tokens << ",\n"
         << "  \"decode_graph_device_bytes\": "
         << result.decode_graph_device_bytes << ",\n"
         << "  \"logits_dumped\": " << (result.logits_dumped ? "true" : "false") << ",\n"
         << "  \"logits_dump_format\": \"raw_float32_little_endian\",\n"
         << "  \"logits_dump_steps\": " << result.logits_dump_steps << ",\n"
         << "  \"logits_dump_vocabulary\": " << kVocabulary << ",\n"
         << "  \"state_dumped\": "
         << (result.state_dumped ? "true" : "false") << ",\n"
         << "  \"state_dump_format\": \"gem16_layer_state_v5\",\n"
         << "  \"state_dump_position\": ";
  if (result.state_dumped) {
    output << result.state_dump_position;
  } else {
    output << "null";
  }
  output << ",\n"
         << "  \"finish_reason\": \"" << (result.stopped ? "stop" : "length") << "\",\n"
         << "  \"stop_token_id\": ";
  if (result.stopped) {
    output << result.stop_token_id;
  } else {
    output << "null";
  }
  output << ",\n"
         << "  \"output_token_ids\": [";
  for (std::size_t index = 0; index < result.output_token_ids.size(); ++index) {
    if (index != 0U) output << ',';
    output << result.output_token_ids[index];
  }
  output << "],\n"
         << "  \"teacher_forced_token_ids\": [";
  for (std::size_t index = 0; index < result.teacher_forced_token_ids.size(); ++index) {
    if (index != 0U) output << ',';
    output << result.teacher_forced_token_ids[index];
  }
  output << "],\n"
         << "  \"teacher_forced_matches\": "
         << result.teacher_forced_matches << "\n}\n";
  return output.good() ? Status::Ok()
                       : Error(StatusCode::kIoError, "failed to write inference JSON");
}

Status WriteDecodeBenchmarkJson(const DecodeBenchmarkResult& result,
                                std::ostream& output) {
  output << std::setprecision(17)
         << "{\"schema_version\":1,\"status\":\"characterization\","
         << "\"benchmark_qualified\":false,\"mode\":\"decode\",\"batch_size\":1,"
         << "\"precision\":\"bf16_state_fp8_attention_nvfp4_mlp\","
         << "\"projection_path\":\"native_sm120\","
         << "\"decode_attention_path\":\""
         << (result.options.kv_cache_mode == KvCacheMode::kCheckpointFp8 &&
                     static_cast<std::uint64_t>(result.options.context_tokens) +
                             result.options.generated_tokens >
                         512U
                 ? "fp8_online_split_gqa"
                 : "score_softmax_value_reference")
         << "\",\"kv_cache_mode\":\""
         << (result.options.kv_cache_mode == KvCacheMode::kCheckpointFp8
                 ? "checkpoint_fp8" : "bf16_correctness")
         << "\",\"kv_cache_layout\":\"hybrid_local_ring_global_contiguous"
         << "\",\"fused_gate_up\":false"
         << ",\"fused_prefill_attention\":true"
         << ",\"fp8_prefill_tile\":\"cutlass_m128n128k64\""
         << ",\"nvfp4_gate_up_prefill_tile\":\"cutlass_m128n128k128\""
         << ",\"nvfp4_gate_up_prefill_weight_scratch\":true"
         << ",\"nvfp4_down_prefill_tile\":\"cutlass_m128n128k128\""
         << ",\"fp8_prefill_pipeline_stages\":0"
         << ",\"fp8_prefill_schedule\":\"cutlass_auto\""
         << ",\"local_prefill_query_heads_per_cta\":2"
         << ",\"global_prefill_query_heads_per_cta\":4"
         << ",\"local_prefill_fp8_staging\":\"fp8x16_fp8x4_bf16x2\""
         << ",\"global_prefill_fp8_staging\":"
            "\"async_contiguous_fp8x16_fp8x4_bf16x2\""
         << ",\"grouped_qkv_prefill\":false"
         << ",\"grouped_qkv_decode\":true"
         << ",\"fused_rmsnorm_boundaries\":true"
         << ",\"fused_prefill_rmsnorm_fp8_quantization\":true"
         << ",\"fused_prefill_rmsnorm_nvfp4_quantization\":true"
         << ",\"fused_prefill_gated_gelu_nvfp4_quantization\":true"
         << ",\"fused_prefill_qk_rmsnorm_rope\":true"
         << ",\"prefill_rope_table\":\"precomputed_exact_max_context\""
         << ",\"fused_output_head\":true"
         << ",\"decode_graphs\":true"
         << ",\"decoding_mode\":\""
         << (result.options.sampling.enabled ? "sampled" : "greedy") << '"'
         << ",\"sampling\":{\"enabled\":"
         << (result.options.sampling.enabled ? "true" : "false")
         << ",\"temperature\":" << result.options.sampling.temperature
         << ",\"top_k\":" << result.options.sampling.top_k
         << ",\"top_p\":" << result.options.sampling.top_p
         << ",\"min_p\":" << result.options.sampling.min_p
         << ",\"repetition_penalty\":"
         << result.options.sampling.repetition_penalty
         << ",\"seed\":" << result.options.sampling.seed << '}'
         << ",\"context_tokens\":" << result.options.context_tokens
         << ",\"generated_tokens\":" << result.options.generated_tokens
         << ",\"warmup_runs\":" << result.options.warmup_runs
         << ",\"measured_runs\":" << result.options.measured_runs
         << ",\"prompt_seed\":" << result.options.prompt_seed
         << ",\"prompt_token_formula\":\"1000+((seed+index*7919)%9000)\","
         << "\"timing_boundary\":\"host_end_to_end_forward_and_gpu_selection\","
         << "\"first_selected_token_excluded_from_decode\":true,"
         << "\"model_loaded_once\":true,\"cache_reset_outside_timing\":true,"
         << "\"prefill_path\":\"native_chunked_sm120\","
         << "\"packed_weight_source_layout_direct\":"
         << (result.packed_weight_source_layout_direct ? "true" : "false")
         << ",\"weight_layout\":\"sm120_row8_k64\""
         << ",\"weight_scale_layout\":\"sm120_row8_k64\""
         << ",\"load_time_weight_swizzle\":true"
         << ",\"load_time_scale_swizzle\":true"
         << ",\"persistent_repack_bytes\":0"
         << ",\"token_loop_allocations\":" << (result.token_loop_allocations ? "true" : "false")
         << ",\"deterministic_outputs\":" << (result.deterministic_outputs ? "true" : "false")
         << ",\"model_load_ms\":" << result.model_load_milliseconds
         << ",\"memory_bytes\":{\"weights\":" << result.weight_arena_bytes
         << ",\"kv_cache\":" << result.kv_cache_bytes
         << ",\"workspace\":" << result.workspace_bytes << "},"
         << "\"prefill_chunk_tokens\":" << result.prefill_chunk_tokens << ','
         << "\"decode_graph_device_bytes\":"
         << result.decode_graph_device_bytes << ','
         << "\"summary\":{\"time_to_first_token_ms\":";
  WriteDistributionJson(output, result.prompt_milliseconds);
  output << ",\"decode_tokens_per_second\":";
  WriteDistributionJson(output, result.decode_tokens_per_second);
  output << ",\"inter_token_latency_ms\":";
  WriteDistributionJson(output, result.inter_token_latency_milliseconds);
  output << "},\"runs\":[";
  for (std::size_t run_index = 0; run_index < result.runs.size(); ++run_index) {
    if (run_index != 0U) output << ',';
    const DecodeBenchmarkRun& run = result.runs[run_index];
    output << "{\"run\":" << run_index
           << ",\"time_to_first_token_ms\":" << run.prompt_milliseconds
           << ",\"decode_ms\":" << run.decode_milliseconds
           << ",\"decode_tokens_per_second\":" << run.decode_tokens_per_second
           << ",\"first_output_token_id\":" << run.first_output_token_id
           << ",\"last_output_token_id\":" << run.last_output_token_id
           << ",\"output_token_checksum\":" << run.output_token_checksum
           << ",\"inter_token_latency_ms\":[";
    for (std::size_t index = 0; index < run.inter_token_latency_milliseconds.size(); ++index) {
      if (index != 0U) output << ',';
      output << run.inter_token_latency_milliseconds[index];
    }
    output << "]}";
  }
  output << "]}\n";
  return output.good() ? Status::Ok()
                       : Error(StatusCode::kIoError,
                               "failed to write decode benchmark JSON");
}

Status WritePrefillBenchmarkJson(const DecodeBenchmarkResult& result,
                                 std::ostream& output) {
  std::vector<double> throughput;
  throughput.reserve(result.runs.size());
  for (const auto& run : result.runs) {
    throughput.push_back(static_cast<double>(result.options.context_tokens) *
                         1000.0 / run.prompt_milliseconds);
  }
  const BenchmarkDistribution throughput_summary = Summarize(throughput);
  output << std::setprecision(17)
         << "{\"schema_version\":1,\"status\":\"characterization\","
         << "\"benchmark_qualified\":false,\"mode\":\"prefill\",\"batch_size\":1,"
         << "\"prompt_tokens\":" << result.options.context_tokens
         << ",\"warmup_runs\":" << result.options.warmup_runs
         << ",\"measured_runs\":" << result.options.measured_runs
         << ",\"prefill_path\":\"native_chunked_sm120\""
         << ",\"fused_prefill_attention\":true"
         << ",\"fp8_prefill_tile\":\"cutlass_m128n128k64\""
         << ",\"nvfp4_gate_up_prefill_tile\":\"cutlass_m128n128k128\""
         << ",\"nvfp4_gate_up_prefill_weight_scratch\":true"
         << ",\"nvfp4_down_prefill_tile\":\"cutlass_m128n128k128\""
         << ",\"fp8_prefill_pipeline_stages\":0"
         << ",\"fp8_prefill_schedule\":\"cutlass_auto\""
         << ",\"local_prefill_query_heads_per_cta\":2"
         << ",\"global_prefill_query_heads_per_cta\":4"
         << ",\"local_prefill_fp8_staging\":\"fp8x16_fp8x4_bf16x2\""
         << ",\"global_prefill_fp8_staging\":"
            "\"async_contiguous_fp8x16_fp8x4_bf16x2\""
         << ",\"grouped_qkv_prefill\":false"
         << ",\"grouped_qkv_decode\":true"
         << ",\"fused_rmsnorm_boundaries\":true"
         << ",\"fused_prefill_rmsnorm_fp8_quantization\":true"
         << ",\"fused_prefill_rmsnorm_nvfp4_quantization\":true"
         << ",\"fused_prefill_gated_gelu_nvfp4_quantization\":true"
         << ",\"fused_prefill_qk_rmsnorm_rope\":true"
         << ",\"prefill_rope_table\":\"precomputed_exact_max_context\""
         << ",\"decode_graphs\":true"
         << ",\"kv_cache_layout\":\"hybrid_local_ring_global_contiguous\","
         << "\"model_load_ms\":" << result.model_load_milliseconds
         << ",\"memory_bytes\":{\"weights\":" << result.weight_arena_bytes
         << ",\"kv_cache\":" << result.kv_cache_bytes
         << ",\"workspace\":" << result.workspace_bytes << "},"
         << "\"prefill_chunk_tokens\":" << result.prefill_chunk_tokens << ','
         << "\"decode_graph_device_bytes\":"
         << result.decode_graph_device_bytes << ','
         << "\"summary\":{\"prompt_tokens_per_second\":";
  WriteDistributionJson(output, throughput_summary);
  output << ",\"time_to_first_token_ms\":";
  WriteDistributionJson(output, result.prompt_milliseconds);
  output << "},\"runs\":[";
  for (std::size_t index = 0; index < result.runs.size(); ++index) {
    if (index != 0U) output << ',';
    output << "{\"run\":" << index
           << ",\"prompt_ms\":" << result.runs[index].prompt_milliseconds
           << ",\"prompt_tokens_per_second\":" << throughput[index]
           << ",\"first_output_token_id\":"
           << result.runs[index].first_output_token_id << '}';
  }
  output << "]}\n";
  return output.good() ? Status::Ok()
                       : Error(StatusCode::kIoError,
                               "failed to write prefill benchmark JSON");
}

}  // namespace gem16
