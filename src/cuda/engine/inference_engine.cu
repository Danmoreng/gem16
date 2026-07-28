#include "gem16/engine.h"
#include "cuda/engine/inference_engine.h"

#include "cuda/attention/sm120.h"
#include "cuda/engine/target_model.h"
#include "cuda/engine/state_capture.h"
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

using internal::Fp8Binding;
using internal::LayerBinding;
using internal::LayerStateCapture;
using internal::MakeStateCaptureLayout;
using internal::Nvfp4Binding;
using internal::StateCaptureLayout;
using internal::WriteStateDump;



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

}  // namespace

class InferenceEngine::Impl {
 public:
  Impl() = default;
  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  ~Impl() {
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


InferenceEngine::InferenceEngine() : impl_(std::make_unique<Impl>()) {}
InferenceEngine::InferenceEngine(InferenceEngine&&) noexcept = default;
InferenceEngine& InferenceEngine::operator=(InferenceEngine&&) noexcept = default;
InferenceEngine::~InferenceEngine() = default;

Status InferenceEngine::Initialize(const std::filesystem::path& model_directory, std::uint64_t max_context, KvCacheMode kv_cache_mode, const SamplingOptions& sampling, const std::filesystem::path& assistant_model_directory, std::uint32_t mtp_draft_tokens) { return impl_->Initialize(model_directory, max_context, kv_cache_mode, sampling, assistant_model_directory, mtp_draft_tokens); }
Result<std::uint32_t> InferenceEngine::Forward(std::uint32_t token, std::uint64_t position, bool select_token, std::span<float> host_logits, std::span<float> host_state) { return impl_->Forward(token, position, select_token, host_logits, host_state); }
Result<std::uint32_t> InferenceEngine::Prefill(std::span<const std::uint32_t> token_ids, std::span<float> host_logits) { return impl_->Prefill(token_ids, host_logits); }
Result<std::uint32_t> InferenceEngine::PrefillAt(std::span<const std::uint32_t> token_ids, std::uint64_t start_position, std::span<float> host_logits) { return impl_->PrefillAt(token_ids, start_position, host_logits); }
Status InferenceEngine::GenerateAssistantDraftsDevice(std::uint32_t input_token, std::uint64_t processed_position, std::uint32_t draft_count) { return impl_->GenerateAssistantDraftsDevice(input_token, processed_position, draft_count); }
Status InferenceEngine::VerifyAcceptCommitAssistantBatch(std::uint32_t input_token, std::uint64_t start_position, std::uint32_t proposal_count, internal::MtpGroupResult* host_result) { return impl_->VerifyAcceptCommitAssistantBatch(input_token, start_position, proposal_count, host_result); }
Status InferenceEngine::ResetCache() { return impl_->ResetCache(); }
Status InferenceEngine::SetSampling(const SamplingOptions& options) { return impl_->SetSampling(options); }
Status InferenceEngine::SetSuppressedTokens(std::span<const std::uint32_t> tokens) { return impl_->SetSuppressedTokens(tokens); }
Status InferenceEngine::SetMtpStopTokens(std::span<const std::uint32_t> tokens) { return impl_->SetMtpStopTokens(tokens); }
std::uint64_t InferenceEngine::weight_bytes() const { return impl_->weight_bytes(); }
bool InferenceEngine::assistant_loaded() const { return impl_->assistant_loaded(); }
std::uint64_t InferenceEngine::assistant_source_bytes() const { return impl_->assistant_source_bytes(); }
std::uint64_t InferenceEngine::assistant_weight_bytes() const { return impl_->assistant_weight_bytes(); }
std::uint64_t InferenceEngine::assistant_device_memory_delta_bytes() const { return impl_->assistant_device_memory_delta_bytes(); }
std::uint64_t InferenceEngine::assistant_tensor_count() const { return impl_->assistant_tensor_count(); }
std::uint64_t InferenceEngine::assistant_workspace_bytes() const { return impl_->assistant_workspace_bytes(); }
std::uint64_t InferenceEngine::cache_bytes() const { return impl_->cache_bytes(); }
std::uint64_t InferenceEngine::workspace_bytes() const { return impl_->workspace_bytes(); }
std::uint64_t InferenceEngine::decode_graph_device_bytes() const { return impl_->decode_graph_device_bytes(); }
std::uint64_t InferenceEngine::prefill_chunk_tokens() const { return impl_->prefill_chunk_tokens(); }

}  // namespace gem16
