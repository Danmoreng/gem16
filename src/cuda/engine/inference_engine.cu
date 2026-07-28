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

#include "cuda/engine/inference_engine_initialize.cuh"
#include "cuda/engine/inference_engine_forward.cuh"
#include "cuda/engine/inference_engine_mtp.cuh"
#include "cuda/engine/inference_engine_prefill_api.cuh"
#include "cuda/engine/inference_engine_memory.cuh"
#include "cuda/engine/inference_engine_prefill_layers.cuh"
#include "cuda/engine/inference_engine_decode_graphs.cuh"

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
