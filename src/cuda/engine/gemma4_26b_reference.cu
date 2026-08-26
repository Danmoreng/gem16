#include "cuda/engine/gemma4_26b_reference.h"

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cuda/attention/gemma4_26b_reference.h"
#include "cuda/attention/sm120.h"
#include "cuda/engine/gemma4_26b_artifact.h"
#include "cuda/layer/reference.h"
#include "cuda/moe/prefill.h"
#include "cuda/moe/reference.h"
#include "cuda/mtp/gemma4_26b_assistant.h"
#include "cuda/mtp/verify.h"
#include "cuda/nvfp4/reference.h"
#include "cuda/nvfp4/sm120.h"
#include "cuda/output_head.h"
#include "cuda/sampling/sampling.h"
#include "gem16/model.h"
#include "model/config.h"
#include "model/gemma4_26b_attention.h"
#include "model/gemma4_26b_residency.h"

namespace gem16::internal {
namespace {

constexpr std::uint64_t kWidth = 2816U;
constexpr std::uint64_t kVocabulary = 262144U;
constexpr std::uint64_t kShared = 2112U;
constexpr std::uint64_t kExpert = 704U;
constexpr std::uint32_t kExperts = 128U;
constexpr std::uint32_t kTopK = 8U;
constexpr std::uint64_t kLayers = 30U;
constexpr std::uint64_t kMaximumContextTokens = 262144U;
constexpr std::uint64_t kPrefillMaxTokens = 1024U;
constexpr std::uint64_t kPreparedGlobalPrefillTokens = 16384U;
// The native online SM120 attention path never materializes a score slab.
// Keep one validated sentinel element because the shared workspace contract
// still requires a non-null pointer, and spend the reclaimed 64 MiB on larger
// chunks that reuse resident expert weights across more tokens.
constexpr std::uint64_t kPrefillScoreElements = 1U;
constexpr std::uint64_t kNvfp4Block = 16U;
constexpr std::uint64_t kSm120KBlock = 64U;
constexpr std::uint64_t kRowsPerTile = 8U;
constexpr unsigned kArgmaxThreads = 256U;
constexpr unsigned kArgmaxBlocks =
    static_cast<unsigned>(kVocabulary / kArgmaxThreads);
constexpr std::array<std::uint32_t, 4> kCaptureLayers{0U, 5U, 6U, 29U};
constexpr std::uint64_t kMiB = 1024U * 1024U;
constexpr std::uint64_t kPrimaryContextMargin = 700U * kMiB;
constexpr std::uint64_t kLongContextMargin = 400U * kMiB;
constexpr std::uint32_t kMaximumSuppressedTokens = 16U;
constexpr std::uint32_t kRepetitionMaskWords =
    static_cast<std::uint32_t>((kVocabulary + 31U) / 32U);
constexpr std::uint64_t kM25MaximumVerifyTokens = 5U;

Status Invalid(std::string message) {
  return Status(StatusCode::kInvalidArgument, std::move(message));
}

Status CudaFailure(const char* operation, cudaError_t error) {
  return Status(StatusCode::kInternal,
                std::string(operation) + ": " + cudaGetErrorName(error) +
                    ": " + cudaGetErrorString(error));
}

Status CudaAllocationFailure(const char* operation, cudaError_t error) {
  if (error == cudaErrorMemoryAllocation) {
    return Status(StatusCode::kResourceExhausted,
                  std::string(operation) + ": " + cudaGetErrorName(error) +
                      ": " + cudaGetErrorString(error));
  }
  return CudaFailure(operation, error);
}

class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  ~DeviceBuffer() {
    if (pointer_ != nullptr) (void)cudaFree(pointer_);
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : pointer_(std::exchange(other.pointer_, nullptr)),
        bytes_(std::exchange(other.bytes_, 0U)) {}
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this == &other) return *this;
    if (pointer_ != nullptr) (void)cudaFree(pointer_);
    pointer_ = std::exchange(other.pointer_, nullptr);
    bytes_ = std::exchange(other.bytes_, 0U);
    return *this;
  }
  Status Allocate(std::uint64_t bytes, const char* label) {
    if (bytes == 0U || bytes > std::numeric_limits<std::size_t>::max()) {
      return Invalid(std::string(label) + " has an invalid size");
    }
    const cudaError_t error =
        cudaMalloc(&pointer_, static_cast<std::size_t>(bytes));
    if (error != cudaSuccess) return CudaAllocationFailure(label, error);
    bytes_ = bytes;
    return Status::Ok();
  }
  template <typename T>
  T* As(std::uint64_t offset = 0U) const {
    return reinterpret_cast<T*>(static_cast<std::byte*>(pointer_) + offset);
  }
  std::uint64_t bytes() const { return bytes_; }

 private:
  void* pointer_ = nullptr;
  std::uint64_t bytes_ = 0U;
};

struct LayoutBuilder {
  std::uint64_t bytes = 0U;
  template <typename T>
  std::uint64_t Add(std::uint64_t elements) {
    constexpr std::uint64_t alignment = 256U;
    if (bytes > std::numeric_limits<std::uint64_t>::max() - (alignment - 1U)) {
      bytes = std::numeric_limits<std::uint64_t>::max();
      return bytes;
    }
    bytes = (bytes + alignment - 1U) & ~(alignment - 1U);
    const std::uint64_t offset = bytes;
    if (elements > (std::numeric_limits<std::uint64_t>::max() - bytes) /
                       sizeof(T)) {
      bytes = std::numeric_limits<std::uint64_t>::max();
    } else {
      bytes += elements * sizeof(T);
    }
    return offset;
  }
};

__device__ __forceinline__ std::uint64_t DeviceTiledPackedOffset(
    std::uint64_t row, std::uint64_t column, std::uint64_t k_blocks) {
  return (((row / kRowsPerTile) * k_blocks + column / kSm120KBlock) *
               kRowsPerTile +
           row % kRowsPerTile) *
              (kSm120KBlock / 2U) +
          (column % kSm120KBlock) / 2U;
}

__device__ __forceinline__ std::uint64_t DeviceTiledScaleOffset(
    std::uint64_t row, std::uint64_t column, std::uint64_t k_blocks) {
  return (((row / kRowsPerTile) * k_blocks + column / kSm120KBlock) *
               kRowsPerTile +
           row % kRowsPerTile) *
              (kSm120KBlock / kNvfp4Block) +
          (column % kSm120KBlock) / kNvfp4Block;
}

__global__ void TiledEmbeddingLookupKernel(
    const std::uint8_t* packed, const std::uint8_t* scales, float divisor,
    std::uint32_t token, float* output) {
  const std::uint64_t column =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (column >= kWidth) return;
  const std::uint64_t k_blocks = kWidth / kSm120KBlock;
  const std::uint64_t offset =
      DeviceTiledPackedOffset(token, column, k_blocks);
  const std::uint8_t byte = packed[offset];
  __nv_fp4_e2m1 value;
  value.__x = static_cast<std::uint8_t>(
      (byte >> ((column & 1U) == 0U ? 0U : 4U)) & 0x0FU);
  __nv_fp8_e4m3 scale;
  scale.__x = scales[DeviceTiledScaleOffset(token, column, k_blocks)];
  const float embedding_scale = static_cast<float>(
      __float2bfloat16_rn(sqrtf(static_cast<float>(kWidth))));
  output[column] = static_cast<float>(__float2bfloat16_rn(
      static_cast<float>(value) * static_cast<float>(scale) / divisor *
      embedding_scale));
}

__global__ void TiledEmbeddingLookupBatchKernel(
    const std::uint8_t* packed, const std::uint8_t* scales, float divisor,
    const std::uint32_t* tokens, float* output, std::uint64_t elements) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  const std::uint64_t token_index = index / kWidth;
  const std::uint64_t column = index % kWidth;
  const std::uint64_t row = tokens[token_index];
  const std::uint64_t k_blocks = kWidth / kSm120KBlock;
  const std::uint64_t offset = DeviceTiledPackedOffset(row, column, k_blocks);
  const std::uint8_t byte = packed[offset];
  __nv_fp4_e2m1 value;
  value.__x = static_cast<std::uint8_t>(
      (byte >> ((column & 1U) == 0U ? 0U : 4U)) & 0x0FU);
  __nv_fp8_e4m3 scale;
  scale.__x = scales[DeviceTiledScaleOffset(row, column, k_blocks)];
  const float embedding_scale = static_cast<float>(
      __float2bfloat16_rn(sqrtf(static_cast<float>(kWidth))));
  output[index] = static_cast<float>(__float2bfloat16_rn(
      static_cast<float>(value) * static_cast<float>(scale) / divisor *
      embedding_scale));
}

__global__ void TiledEmbeddingLookupControlledKernel(
    const std::uint8_t* packed, const std::uint8_t* scales, float divisor,
    const DecodeControl* control, float* output) {
  const std::uint64_t column =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (column >= kWidth) return;
  const std::uint64_t row = control->token;
  const std::uint64_t k_blocks = kWidth / kSm120KBlock;
  const std::uint64_t offset = DeviceTiledPackedOffset(row, column, k_blocks);
  const std::uint8_t byte = packed[offset];
  __nv_fp4_e2m1 value;
  value.__x = static_cast<std::uint8_t>(
      (byte >> ((column & 1U) == 0U ? 0U : 4U)) & 0x0FU);
  __nv_fp8_e4m3 scale;
  scale.__x = scales[DeviceTiledScaleOffset(row, column, k_blocks)];
  const float embedding_scale = static_cast<float>(
      __float2bfloat16_rn(sqrtf(static_cast<float>(kWidth))));
  output[column] = static_cast<float>(__float2bfloat16_rn(
      static_cast<float>(value) * static_cast<float>(scale) / divisor *
      embedding_scale));
}

__global__ void CapturePrefillRouterIdsKernel(
    const Gemma4MoePrefillAssignment* assignments, std::uint32_t* output,
    std::uint64_t token) {
  const std::uint32_t slot = threadIdx.x;
  if (blockIdx.x == 0U && slot < kTopK) {
    output[slot] = assignments[token * kTopK + slot].expert_id;
  }
}

__global__ void SetDecodeControlKernel(DecodeControl* control,
                                       std::uint32_t token,
                                       std::uint64_t position) {
  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    *control = DecodeControl{token, 0U, position, position};
  }
}

__global__ void BuildMtpDecodeControlsKernel(
    DecodeControl* controls, std::uint64_t start_position,
    std::uint64_t sampling_step, std::uint64_t tokens) {
  const std::uint64_t row = threadIdx.x;
  if (blockIdx.x == 0U && row < tokens) {
    controls[row] = DecodeControl{0U, 0U, start_position + row,
                                  sampling_step + row};
  }
}

struct LogitCandidate {
  float value;
  std::uint32_t id;
};

struct PredictionStatus {
  std::uint32_t token;
  float logit;
  int finite;
  int routing_finite;
};
static_assert(sizeof(PredictionStatus) == 16U);

struct MtpVerifierHostResult {
  MtpGroupTransaction transaction{};
  std::array<int, kM25MaximumVerifyTokens> logits_finite{};
  std::array<float, kM25MaximumVerifyTokens> top_values{};
  std::array<float, kM25MaximumVerifyTokens> second_values{};
  int routing_finite = 0;
};

struct MtpVerifierLayerOffsets {
  std::uint64_t backup_key = 0U;
  std::uint64_t backup_value = 0U;
  std::uint64_t compact_key = 0U;
  std::uint64_t compact_value = 0U;
};

__device__ __forceinline__ LogitCandidate BetterCandidate(
    LogitCandidate left, LogitCandidate right) {
  return right.value > left.value ||
                 (right.value == left.value && right.id < left.id)
             ? right
             : left;
}

__device__ bool MtpTokenSuppressed(std::uint32_t token,
                                   const std::uint32_t* suppressed,
                                   std::uint32_t suppressed_count) {
  for (std::uint32_t index = 0U; index < suppressed_count; ++index) {
    if (suppressed[index] == token) return true;
  }
  return false;
}

__global__ void SoftcapArgmaxBlocksKernel(float* logits, float softcap,
                                          int* all_finite,
                                          LogitCandidate* candidates,
                                          const std::uint32_t* suppressed,
                                          std::uint32_t suppressed_count,
                                          const std::uint32_t* excluded,
                                          bool apply_softcap) {
  __shared__ LogitCandidate partial[kArgmaxThreads];
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  LogitCandidate candidate{-3.402823466e+38F,
                           static_cast<std::uint32_t>(index)};
  if (index < kVocabulary) {
    const float value = apply_softcap
                            ? tanhf(logits[index] / softcap) * softcap
                            : logits[index];
    if (apply_softcap) logits[index] = value;
    if (!isfinite(value)) {
      atomicExch(all_finite, 0);
    } else if ((excluded == nullptr || *excluded != index) &&
               !MtpTokenSuppressed(static_cast<std::uint32_t>(index),
                                   suppressed, suppressed_count)) {
      candidate.value = value;
    }
  }
  partial[threadIdx.x] = candidate;
  __syncthreads();
  for (unsigned stride = kArgmaxThreads / 2U; stride != 0U; stride /= 2U) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] = BetterCandidate(
          partial[threadIdx.x], partial[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0U) candidates[blockIdx.x] = partial[0];
}

__global__ void FinalArgmaxKernel(const LogitCandidate* candidates,
                                  std::uint32_t* token, float* value,
                                  DecodeControl* next_control) {
  __shared__ LogitCandidate partial[kArgmaxThreads];
  LogitCandidate best{-3.402823466e+38F, 0U};
  for (unsigned index = threadIdx.x; index < kArgmaxBlocks;
       index += blockDim.x) {
    best = BetterCandidate(best, candidates[index]);
  }
  partial[threadIdx.x] = best;
  __syncthreads();
  for (unsigned stride = kArgmaxThreads / 2U; stride != 0U; stride /= 2U) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] = BetterCandidate(
          partial[threadIdx.x], partial[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0U) {
    *token = partial[0].id;
    *value = partial[0].value;
    if (next_control != nullptr) {
      // Prepare the ordinary greedy successor inside the captured graph. The
      // host validates token and position before it omits an explicit control
      // update, so teacher forcing and sampled overrides remain exact.
      next_control->token = partial[0].id;
      ++next_control->position;
    }
  }
}

Status LaunchSoftcapArgmax(float* logits, float softcap, int* all_finite,
                           LogitCandidate* candidates,
                           std::uint32_t* token, float* value,
                           cudaStream_t stream,
                           DecodeControl* next_control = nullptr,
                           const std::uint32_t* suppressed = nullptr,
                           std::uint32_t suppressed_count = 0U,
                           const std::uint32_t* excluded = nullptr,
                           bool apply_softcap = true) {
  cudaError_t error = cudaMemsetAsync(all_finite, 1, sizeof(int), stream);
  if (error != cudaSuccess) {
    return CudaFailure("initialize M17 finite flag", error);
  }
  SoftcapArgmaxBlocksKernel<<<kArgmaxBlocks, kArgmaxThreads, 0, stream>>>(
      logits, softcap, all_finite, candidates, suppressed, suppressed_count,
      excluded, apply_softcap);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaFailure("launch M17 fused softcap/argmax blocks", error);
  }
  FinalArgmaxKernel<<<1, kArgmaxThreads, 0, stream>>>(
      candidates, token, value, next_control);
  error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch M17 final argmax", error);
}

template <bool kCommitLogits>
__global__ void CommitMtpPredictionKernel(
    const float* batch_logits, float* committed_logits,
    const MtpGroupResult* result, const int* logits_finite,
    const int* routing_finite, PredictionStatus* status) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint32_t row = result->output_count - 1U;
  if constexpr (kCommitLogits) {
    if (index < kVocabulary) {
      committed_logits[index] = batch_logits[row * kVocabulary + index];
    }
  }
  if (index == 0U) {
    const std::uint32_t token = result->verified[row];
    status->token = token;
    status->logit = batch_logits[row * kVocabulary + token];
    status->finite = logits_finite[row];
    status->routing_finite = *routing_finite;
  }
}

__global__ void LatchMtpGroupNonFiniteKernel(
    const int* logits_finite, std::uint32_t rows,
    const int* routing_finite, MtpChainResult* chain_result) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) return;
  bool non_finite = *routing_finite == 0;
  for (std::uint32_t row = 0U; row < rows; ++row) {
    non_finite = non_finite || logits_finite[row] == 0;
  }
  if (non_finite) ++chain_result->non_finite_step_count;
}

__global__ void LatchMtpOrdinaryNonFiniteKernel(
    const PredictionStatus* status, MtpChainResult* chain_result) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) return;
  if (status->finite == 0 || status->routing_finite == 0) {
    ++chain_result->non_finite_step_count;
  }
}

int CaptureIndex(std::uint32_t layer) {
  for (std::size_t index = 0; index < kCaptureLayers.size(); ++index) {
    if (kCaptureLayers[index] == layer) return static_cast<int>(index);
  }
  return -1;
}

template <typename T>
Result<const T*> ArtifactPointer(const Gemma4Moe26BDeviceArtifact& artifact,
                                 const char* name) {
  auto pointer = artifact.Pointer(name);
  if (!pointer.ok()) return pointer.status();
  return reinterpret_cast<const T*>(pointer.value());
}

}  // namespace

struct Gemma4Moe26BReferenceEngine::Impl {
  int device = 0;
  std::uint64_t context = 0U;
  std::uint64_t position = 0U;
  std::uint64_t sliding_capacity = 0U;
  std::uint64_t prefill_chunks = 0U;
  std::uint64_t minimum_prefill_chunk = 0U;
  std::uint64_t prefill_calls = 0U;
  std::uint64_t decode_graph_launches = 0U;
  std::uint64_t token_selections = 0U;
  std::uint64_t sliding_ring_wraps = 0U;
  std::uint64_t maximum_global_position_exclusive = 0U;
  std::uint64_t fallback_count = 0U;
  std::uint64_t recurring_allocation_count = 0U;
  bool pending_decode_self_feed = false;
  bool decode_self_feed_valid = false;
  bool has_last_input_token = false;
  std::uint32_t last_input_token = 0U;
  std::uint32_t decode_self_feed_token = 0U;
  std::uint64_t decode_self_feed_position = 0U;
  Gemma4Moe26BBackend backend = Gemma4Moe26BBackend::kReference;
  Gemma4Moe26BMtpVerifierBackend mtp_verifier_backend =
      Gemma4Moe26BMtpVerifierBackend::kExactDecodeParent;
  cudaStream_t stream = nullptr;
  cudaStream_t shared_moe_stream = nullptr;
  cudaEvent_t shared_moe_fork = nullptr;
  cudaEvent_t shared_moe_join = nullptr;
  Gemma4Moe26BDeviceArtifact artifact;
  Gemma4Moe26BAssistantModel assistant;
  DeviceBuffer mtp_verifier;
  MtpVerifierHostResult* mtp_verifier_host = nullptr;
  std::array<cudaGraphExec_t, kMaximumMtpDraftTokens + 1U>
      mtp_group_graphs{};
  std::array<cudaGraphExec_t, kMaximumMtpDraftTokens + 1U>
      mtp_chain_graphs{};
  std::uint64_t mtp_group_graph_device_bytes = 0U;
  std::array<std::uint64_t, kMaximumMtpDraftTokens + 1U>
      mtp_chain_graph_device_bytes{};
  std::uint64_t mtp_group_graph_launches = 0U;
  std::array<std::uint64_t, kMaximumMtpDraftTokens + 1U>
      mtp_chain_graph_launches{};
  std::uint64_t mtp_normalized_offset = 0U;
  std::uint64_t mtp_head_activation_offset = 0U;
  std::uint64_t mtp_head_scales_offset = 0U;
  std::uint64_t mtp_logits_offset = 0U;
  std::uint64_t mtp_selected_offset = 0U;
  std::uint64_t mtp_second_selected_offset = 0U;
  std::uint64_t mtp_top_values_offset = 0U;
  std::uint64_t mtp_second_values_offset = 0U;
  std::uint64_t mtp_finite_offset = 0U;
  std::uint64_t mtp_transaction_offset = 0U;
  std::uint64_t mtp_row_controls_offset = 0U;
  std::uint64_t mtp_sampling_repetition_masks_offset = 0U;
  std::uint64_t mtp_stop_tokens_offset = 0U;
  std::uint64_t mtp_chain_result_offset = 0U;
  std::uint64_t mtp_chain_outputs_offset = 0U;
  std::uint64_t mtp_chain_proposals_offset = 0U;
  std::byte* mtp_chain_host = nullptr;
  std::uint32_t mtp_stop_token_count = 0U;
  std::array<MtpVerifierLayerOffsets, kLayers> mtp_layer_offsets{};
  float last_mtp_verification_min_margin = 0.0F;
  Gemma4Moe26BAttentionTraits traits{};
  std::array<Gemma4Moe26BAttentionReferenceWeights, kLayers> attention_weights{};
  std::array<Gemma4MoeReferenceWeights, kLayers> moe_weights{};
  std::array<Gemma4Moe26BKvCacheView, kLayers> caches{};
  Gemma4Moe26BAttentionReferenceWorkspace attention_workspace{};
  Gemma4MoeReferenceWorkspace moe_workspace{};
  Gemma4MoeReferenceConfig moe_config{kWidth, kShared, kExpert, kExperts,
                                      kTopK, 1.0e-6F};
  Gemma4MoeNvfp4Matrix head{};
  const std::uint16_t* final_norm = nullptr;
  float softcap = 30.0F;
  DeviceBuffer kv;
  DeviceBuffer workspace;
  DeviceBuffer prefill_workspace;
  DecodeControl* decode_control = nullptr;
  cudaGraphExec_t decode_graph = nullptr;
  std::uint32_t* prefill_tokens = nullptr;
  std::uint32_t* prefill_host_tokens = nullptr;
  float* prefill_hidden_a = nullptr;
  float* prefill_hidden_b = nullptr;
  Gemma4Moe26BAttentionReferenceWorkspace prefill_attention_workspace{};
  float* prefill_local_rotary_cosine = nullptr;
  float* prefill_local_rotary_sine = nullptr;
  Gemma4MoePrefillWorkspace prefill_moe_workspace{};
  float* hidden_a = nullptr;
  float* hidden_b = nullptr;
  float* final_hidden = nullptr;
  float* local_rotary_cosine = nullptr;
  float* local_rotary_sine = nullptr;
  std::uint8_t* head_activation = nullptr;
  std::uint8_t* head_activation_scales = nullptr;
  float* logits = nullptr;
  PredictionStatus* prediction_device_status = nullptr;
  PredictionStatus* prediction_host_status = nullptr;
  std::uint32_t* prediction_token = nullptr;
  float* prediction_logit = nullptr;
  int* finite = nullptr;
  int* routing_finite = nullptr;
  LogitCandidate* output_candidates = nullptr;
  float* sampling_logits = nullptr;
  double* sampling_cumulative = nullptr;
  std::uint32_t* sampling_token_ids = nullptr;
  std::uint32_t* sampling_sorted_token_ids = nullptr;
  std::uint32_t* repetition_mask = nullptr;
  std::uint32_t* suppressed_token_ids = nullptr;
  std::uint32_t* selected_token = nullptr;
  void* sampling_algorithm_workspace = nullptr;
  std::size_t sampling_algorithm_workspace_bytes = 0U;
  SamplingOptions sampling{};
  std::uint64_t sampling_step = 0U;
  std::uint32_t suppressed_token_count = 0U;
  std::array<float*, kCaptureLayers.size()> layer_captures{};
  std::array<float*, kCaptureLayers.size()> router_probability_captures{};
  std::array<std::uint32_t*, kCaptureLayers.size()> router_id_captures{};

  Status LaunchControlledDecodeBody();
  Status PrepareDecodeGraph();
  Status LaunchFixedMtpGraphBody(std::uint32_t draft_count,
                                 bool copy_transaction);
  Status SelectMtpVerificationTokens(
      float* batch_logits, const std::uint32_t* verification_inputs,
      DecodeControl* row_controls, std::uint32_t tokens,
      std::uint32_t* selected);
  Status PrepareFixedMtpGraph(std::uint32_t draft_count);
  Status LaunchFixedMtpOrdinaryTailBody();
  Status PrepareFixedMtpChainGraph(std::uint32_t draft_count);
  Status ExecuteFixedMtpGraphGroup(std::uint32_t pending_token,
                                   std::uint32_t draft_count,
                                   MtpGroupResult* host_result);

  ~Impl() {
    if (stream != nullptr) {
      (void)cudaStreamSynchronize(stream);
    }
    if (shared_moe_stream != nullptr) {
      (void)cudaStreamSynchronize(shared_moe_stream);
    }
    if (decode_graph != nullptr) (void)cudaGraphExecDestroy(decode_graph);
    for (cudaGraphExec_t graph : mtp_group_graphs) {
      if (graph != nullptr) (void)cudaGraphExecDestroy(graph);
    }
    for (cudaGraphExec_t graph : mtp_chain_graphs) {
      if (graph != nullptr) (void)cudaGraphExecDestroy(graph);
    }
    if (prefill_host_tokens != nullptr) {
      (void)cudaFreeHost(prefill_host_tokens);
    }
    if (prediction_host_status != nullptr) {
      (void)cudaFreeHost(prediction_host_status);
    }
    if (mtp_verifier_host != nullptr) {
      (void)cudaFreeHost(mtp_verifier_host);
    }
    if (mtp_chain_host != nullptr) (void)cudaFreeHost(mtp_chain_host);
    if (stream != nullptr) {
      (void)cudaStreamDestroy(stream);
    }
    if (shared_moe_fork != nullptr) (void)cudaEventDestroy(shared_moe_fork);
    if (shared_moe_join != nullptr) (void)cudaEventDestroy(shared_moe_join);
    if (shared_moe_stream != nullptr) {
      (void)cudaStreamDestroy(shared_moe_stream);
    }
  }
};

Status Gemma4Moe26BReferenceEngine::Impl::LaunchControlledDecodeBody() {
  if (decode_control == nullptr) {
    return Invalid("M17 decode graph control is not initialized");
  }
  cudaError_t error = cudaMemsetAsync(routing_finite, 1, sizeof(int), stream);
  if (error != cudaSuccess) {
    return CudaFailure("initialize M17 router finite flag", error);
  }
  constexpr unsigned threads = 256U;
  TiledEmbeddingLookupControlledKernel<<<
      static_cast<unsigned>((kWidth + threads - 1U) / threads), threads, 0,
      stream>>>(head.packed_e2m1, head.scales_e4m3fn,
                head.weight_global_divisor, decode_control, hidden_a);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaFailure("launch M17 controlled embedding", error);
  }
  const Gemma4Moe26BAttentionLayerTraits* local_trait = nullptr;
  const Gemma4Moe26BAttentionLayerTraits* global_trait = nullptr;
  for (const auto& trait : traits) {
    const bool sliding =
        trait.attention == Gemma4Moe26BAttentionType::kSliding;
    const Gemma4Moe26BAttentionLayerTraits*& profile =
        sliding ? local_trait : global_trait;
    if (profile == nullptr) {
      profile = &trait;
    } else if (profile->head_dimension != trait.head_dimension ||
               profile->rotary_factor != trait.rotary_factor ||
               profile->rope_theta != trait.rope_theta ||
               profile->rope_scaling_factor != trait.rope_scaling_factor) {
      return Invalid("M17 attention classes do not share RoPE profiles");
    }
  }
  if (local_trait == nullptr || global_trait == nullptr) {
    return Invalid("M17 requires local and global RoPE profiles");
  }
  auto prepare_rotary = [&](const Gemma4Moe26BAttentionLayerTraits& trait,
                            float* cosine, float* sine) {
    const std::uint64_t rotating_pairs = static_cast<std::uint64_t>(
        trait.rotary_factor *
        static_cast<double>(trait.head_dimension / 2U));
    return LaunchRotaryEmbeddingTableControlled(
        cosine, sine, rotating_pairs, trait.head_dimension, decode_control,
        trait.rope_theta, trait.rope_scaling_factor, stream);
  };
  Status status = prepare_rotary(*global_trait,
                                 attention_workspace.rotary_cosine,
                                 attention_workspace.rotary_sine);
  if (!status.ok()) return status;
  status = prepare_rotary(*local_trait, local_rotary_cosine,
                          local_rotary_sine);
  if (!status.ok()) return status;
  for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
    auto layer_workspace = attention_workspace;
    if (traits[layer].attention == Gemma4Moe26BAttentionType::kSliding) {
      layer_workspace.rotary_cosine = local_rotary_cosine;
      layer_workspace.rotary_sine = local_rotary_sine;
    }
    status = LaunchGemma4Moe26BAttentionReferenceControlledLayer(
        hidden_a, hidden_b, traits[layer], attention_weights[layer],
        caches[layer], layer_workspace, decode_control, 1.0e-6F, stream,
        true);
    if (!status.ok()) return status;
    status = LaunchGemma4MoeSm120Layer(
        hidden_b, hidden_a, moe_config, moe_weights[layer], moe_workspace,
        stream, shared_moe_stream, shared_moe_fork, shared_moe_join);
    if (!status.ok()) return status;
    const int capture = CaptureIndex(layer);
    if (capture >= 0) {
      const std::size_t index = static_cast<std::size_t>(capture);
      error = cudaMemcpyAsync(layer_captures[index], hidden_a,
                              kWidth * sizeof(float),
                              cudaMemcpyDeviceToDevice, stream);
      if (error == cudaSuccess) {
        error = cudaMemcpyAsync(
            router_probability_captures[index],
            moe_workspace.router_probabilities, kExperts * sizeof(float),
            cudaMemcpyDeviceToDevice, stream);
      }
      if (error == cudaSuccess) {
        error = cudaMemcpyAsync(router_id_captures[index],
                                moe_workspace.top_ids,
                                kTopK * sizeof(std::uint32_t),
                                cudaMemcpyDeviceToDevice, stream);
      }
      if (error != cudaSuccess) {
        return CudaFailure("capture M17 decode layer", error);
      }
    }
  }
  status = LaunchRmsNormNvfp4ActivationQuantizationBatch(
      hidden_a, final_norm, head_activation, head_activation_scales, 1U,
      kWidth, 1.0e-6F, head.activation_global_divisor, stream);
  if (!status.ok()) return status;
  status = LaunchNvfp4Sm120DirectProjectionBf16Float(
      head_activation, head_activation_scales, head.packed_e2m1,
      head.scales_e4m3fn, logits, head.rows, head.columns,
      head.activation_global_divisor, head.weight_global_divisor, stream);
  if (!status.ok()) return status;
  return LaunchSoftcapArgmax(logits, softcap, finite, output_candidates,
                             prediction_token, prediction_logit, stream,
                             decode_control);
}

Status Gemma4Moe26BReferenceEngine::Impl::PrepareDecodeGraph() {
  cudaError_t error = cudaStreamSynchronize(stream);
  if (error != cudaSuccess) {
    return CudaFailure("synchronize before M17 graph capture", error);
  }
  error = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
  if (error != cudaSuccess) return CudaFailure("begin M17 graph capture", error);
  const Status body = LaunchControlledDecodeBody();
  cudaGraph_t graph = nullptr;
  error = cudaStreamEndCapture(stream, &graph);
  if (!body.ok()) {
    if (graph != nullptr) (void)cudaGraphDestroy(graph);
    return body;
  }
  if (error != cudaSuccess) {
    if (graph != nullptr) (void)cudaGraphDestroy(graph);
    return CudaFailure("end M17 graph capture", error);
  }
  error = cudaGraphInstantiate(&decode_graph, graph, nullptr, nullptr, 0U);
  const cudaError_t destroy_error = cudaGraphDestroy(graph);
  if (error != cudaSuccess) return CudaFailure("instantiate M17 graph", error);
  if (destroy_error != cudaSuccess) {
    (void)cudaGraphExecDestroy(decode_graph);
    decode_graph = nullptr;
    return CudaFailure("destroy captured M17 graph", destroy_error);
  }
  return Status::Ok();
}

Status Gemma4Moe26BReferenceEngine::Impl::SelectMtpVerificationTokens(
    float* batch_logits, const std::uint32_t* verification_inputs,
    DecodeControl* row_controls, std::uint32_t tokens,
    std::uint32_t* selected) {
  if (!sampling.enabled) return Status::Ok();
  if (batch_logits == nullptr || verification_inputs == nullptr ||
      row_controls == nullptr || tokens == 0U ||
      tokens > kM25MaximumVerifyTokens || selected == nullptr) {
    return Invalid("M25 sampled verifier buffers are invalid");
  }
  auto* row_masks = mtp_verifier.As<std::uint32_t>(
      mtp_sampling_repetition_masks_offset);
  if (sampling.repetition_penalty != 1.0F) {
    Status status = LaunchBuildSpeculativeRepetitionMasks(
        repetition_mask, verification_inputs, tokens, kRepetitionMaskWords,
        row_masks, stream);
    if (!status.ok()) return status;
  }
  for (std::uint32_t row = 0U; row < tokens; ++row) {
    // LaunchSampleToken uses its logits input as radix-sort output. Stage one
    // row in the ordinary logits buffer so the verifier batch remains intact
    // for commit diagnostics and the accepted-row prediction state.
    cudaError_t error = cudaMemcpyAsync(
        logits, batch_logits + static_cast<std::uint64_t>(row) * kVocabulary,
        kVocabulary * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    if (error != cudaSuccess) {
      return CudaFailure("stage M25 sampled verifier logits", error);
    }
    std::uint32_t* mask = sampling.repetition_penalty == 1.0F
                              ? repetition_mask
                              : row_masks +
                                    static_cast<std::uint64_t>(row) *
                                        kRepetitionMaskWords;
    Status status = LaunchSampleToken(
        logits, sampling_logits, sampling_cumulative, sampling_token_ids,
        sampling_sorted_token_ids, mask, suppressed_token_ids,
        suppressed_token_count, static_cast<std::uint32_t>(kVocabulary),
        sampling, 0U, row_controls + row, selected + row,
        sampling_algorithm_workspace, sampling_algorithm_workspace_bytes,
        stream);
    if (!status.ok()) return status;
  }
  return Status::Ok();
}

Status Gemma4Moe26BReferenceEngine::Impl::LaunchFixedMtpGraphBody(
    std::uint32_t draft_count, bool copy_transaction) {
  const std::uint32_t tokens = draft_count + 1U;
  if ((draft_count != 1U && draft_count != 2U && draft_count != 4U) ||
      mtp_verifier_backend !=
          Gemma4Moe26BMtpVerifierBackend::kExactSharedBatchedMoe ||
      !assistant.prepared() || mtp_verifier.bytes() == 0U ||
      mtp_verifier_host == nullptr) {
    return Invalid("M25 fixed graph body is unavailable for this D/backend");
  }

  auto* transaction =
      mtp_verifier.As<MtpGroupTransaction>(mtp_transaction_offset);
  auto* control = &transaction->control;
  constexpr std::uint32_t kSlidingLayer = 28U;
  constexpr std::uint32_t kFullLayer = 29U;
  const auto make_view = [this](std::uint32_t layer) {
    AssistantSharedKvView view;
    const bool global =
        traits[layer].attention == Gemma4Moe26BAttentionType::kFull;
    view.mode = AssistantKvCacheMode::kCheckpointFp8;
    view.key_fp8 = caches[layer].key;
    view.value_fp8 = caches[layer].value;
    view.key_scale_bf16 = attention_weights[layer].key_cache_scale_bf16;
    view.value_scale_bf16 = attention_weights[layer].value_cache_scale_bf16;
    view.capacity = caches[layer].capacity;
    // Controlled online attention derives the visible prefix/ring slot from
    // MtpDeviceControl. A non-zero fixed extent keeps capture-time validation
    // independent from the first replay position.
    view.tokens = view.capacity;
    view.first_slot = 0U;
    view.kv_heads = traits[layer].kv_heads;
    view.head_dimension = traits[layer].head_dimension;
    return view;
  };
  if (traits[kSlidingLayer].attention !=
          Gemma4Moe26BAttentionType::kSliding ||
      traits[kFullLayer].attention != Gemma4Moe26BAttentionType::kFull) {
    return Status(StatusCode::kInternal,
                  "M25 fixed graph shared-KV layer mapping is invalid");
  }
  Gemma4Moe26BAssistantProposalContext proposal;
  proposal.target_embedding = head;
  proposal.target_hidden = final_hidden;
  proposal.sliding_kv = make_view(kSlidingLayer);
  proposal.full_kv = make_view(kFullLayer);
  Status status = assistant.GenerateDraftsDevice(
      proposal, draft_count, stream, control);
  if (!status.ok()) return status;
  const std::uint32_t* drafts = assistant.device_draft_tokens();
  if (drafts == nullptr) {
    return Status(StatusCode::kInternal,
                  "M25 fixed graph Assistant drafts are unavailable");
  }

  auto* row_controls =
      mtp_verifier.As<DecodeControl>(mtp_row_controls_offset);
  status = LaunchBuildControlledMtpInputs(
      control, drafts, draft_count, prefill_tokens, row_controls,
      suppressed_token_count, stream);
  if (!status.ok()) return status;
  cudaError_t error = cudaMemsetAsync(routing_finite, 1, sizeof(int), stream);
  if (error != cudaSuccess) {
    return CudaFailure("initialize M25 fixed graph router finite flag", error);
  }
  constexpr unsigned threads = 256U;
  const std::uint64_t hidden_elements = tokens * kWidth;
  TiledEmbeddingLookupBatchKernel<<<
      static_cast<unsigned>((hidden_elements + threads - 1U) / threads),
      threads, 0, stream>>>(
      head.packed_e2m1, head.scales_e4m3fn, head.weight_global_divisor,
      prefill_tokens, prefill_hidden_a, hidden_elements);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaFailure("launch M25 fixed graph embedding", error);
  }

  for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
    const auto& trait = traits[layer];
    const std::uint64_t kv_elements =
        trait.kv_heads * trait.head_dimension;
    const auto offsets = mtp_layer_offsets[layer];
    auto* backup_key = mtp_verifier.As<std::uint8_t>(offsets.backup_key);
    auto* backup_value =
        mtp_verifier.As<std::uint8_t>(offsets.backup_value);
    auto* compact_key = mtp_verifier.As<std::uint8_t>(offsets.compact_key);
    auto* compact_value =
        mtp_verifier.As<std::uint8_t>(offsets.compact_value);
    status = LaunchCopyCircularMtpKvFp8Controlled(
        caches[layer].key, caches[layer].value, backup_key, backup_value,
        tokens, kv_elements, caches[layer].capacity, false, control, stream);
    if (!status.ok()) return status;
    // D2 is the selected fixed-depth candidate. Its three consecutive Target
    // rows share the qualified fixed-row attention body instead of replaying
    // one online-attention launch sequence per row. Keep D1/D4 on their
    // established parent until those non-selected depths are qualified
    // independently.
    const bool shared_fixed_attention = tokens == 3U;
    status = LaunchGemma4Moe26BAttentionSm120MtpFixedLayer(
        prefill_hidden_a, prefill_hidden_b, 0U, trait,
        attention_weights[layer], caches[layer], prefill_attention_workspace,
        prefill_moe_workspace.shared_product, row_controls, tokens,
        shared_fixed_attention, true, true, 1.0e-6F, stream);
    if (!status.ok()) return status;
    status = LaunchGemma4MoeSm120MtpSharedBatchLayer(
        prefill_hidden_b, prefill_hidden_a, tokens, moe_config,
        moe_weights[layer], prefill_moe_workspace, moe_workspace, stream);
    if (!status.ok()) return status;
    status = LaunchCopyCircularMtpKvFp8Controlled(
        caches[layer].key, caches[layer].value, compact_key, compact_value,
        tokens, kv_elements, caches[layer].capacity, false, control, stream);
    if (!status.ok()) return status;
    status = LaunchCopyCircularMtpKvFp8Controlled(
        caches[layer].key, caches[layer].value, backup_key, backup_value,
        tokens, kv_elements, caches[layer].capacity, true, control, stream);
    if (!status.ok()) return status;
  }

  float* normalized = mtp_verifier.As<float>(mtp_normalized_offset);
  auto* head_activation =
      mtp_verifier.As<std::uint8_t>(mtp_head_activation_offset);
  auto* head_scales =
      mtp_verifier.As<std::uint8_t>(mtp_head_scales_offset);
  float* batch_logits = mtp_verifier.As<float>(mtp_logits_offset);
  auto* selected =
      mtp_verifier.As<std::uint32_t>(mtp_selected_offset);
  auto* second_selected =
      mtp_verifier.As<std::uint32_t>(mtp_second_selected_offset);
  float* top_values = mtp_verifier.As<float>(mtp_top_values_offset);
  float* second_values = mtp_verifier.As<float>(mtp_second_values_offset);
  int* logits_finite = mtp_verifier.As<int>(mtp_finite_offset);
  status = LaunchRmsNormBf16Nvfp4ActivationQuantizationBatch(
      prefill_hidden_a, final_norm, normalized, head_activation, head_scales,
      tokens, kWidth, 1.0e-6F, head.activation_global_divisor, stream);
  if (!status.ok()) return status;
  status = LaunchNvfp4Sm120DirectProjectionBf16FloatExactBatch(
      head_activation, head_scales, head.packed_e2m1, head.scales_e4m3fn,
      batch_logits, tokens, head.rows, head.columns,
      head.activation_global_divisor, head.weight_global_divisor, stream);
  if (!status.ok()) return status;
  for (std::uint64_t row = 0U; row < tokens; ++row) {
    status = LaunchSoftcapArgmax(
        batch_logits + row * kVocabulary, softcap, logits_finite + row,
        output_candidates, selected + row, top_values + row, stream, nullptr,
        suppressed_token_ids, suppressed_token_count);
    if (!status.ok()) return status;
    if (copy_transaction) {
      status = LaunchSoftcapArgmax(
          batch_logits + row * kVocabulary, softcap, logits_finite + row,
          output_candidates, second_selected + row, second_values + row,
          stream, nullptr, suppressed_token_ids, suppressed_token_count,
          selected + row, false);
      if (!status.ok()) return status;
    }
  }
  status = SelectMtpVerificationTokens(
      batch_logits, prefill_tokens, row_controls, tokens, selected);
  if (!status.ok()) return status;
  status = LaunchAcceptMtpGroup(
      drafts, selected, draft_count,
      mtp_verifier.As<std::uint32_t>(mtp_stop_tokens_offset),
      mtp_stop_token_count,
      &transaction->result, control, stream);
  if (!status.ok()) return status;
  if (sampling.enabled && sampling.repetition_penalty != 1.0F) {
    status = LaunchCommitSpeculativeRepetitionMask(
        mtp_verifier.As<std::uint32_t>(
            mtp_sampling_repetition_masks_offset),
        kRepetitionMaskWords, &transaction->result.output_count,
        repetition_mask, stream);
    if (!status.ok()) return status;
  }
  for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
    const auto& trait = traits[layer];
    const auto offsets = mtp_layer_offsets[layer];
    status = LaunchCommitMtpKvFp8Controlled(
        mtp_verifier.As<std::uint8_t>(offsets.compact_key),
        mtp_verifier.As<std::uint8_t>(offsets.compact_value),
        caches[layer].key, caches[layer].value,
        trait.kv_heads * trait.head_dimension, caches[layer].capacity,
        &transaction->result, control, stream);
    if (!status.ok()) return status;
  }
  status = LaunchCommitMtpHidden(normalized, final_hidden, kWidth,
                                 &transaction->result, stream);
  if (!status.ok()) return status;
  if (copy_transaction) {
    CommitMtpPredictionKernel<true>
        <<<kArgmaxBlocks, kArgmaxThreads, 0, stream>>>(
            batch_logits, logits, &transaction->result, logits_finite,
            routing_finite, prediction_device_status);
  } else {
    CommitMtpPredictionKernel<false><<<1U, 1U, 0, stream>>>(
        batch_logits, logits, &transaction->result, logits_finite,
        routing_finite, prediction_device_status);
  }
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaFailure("commit M25 fixed graph prediction", error);
  }

  if (!copy_transaction) {
    LatchMtpGroupNonFiniteKernel<<<1U, 1U, 0, stream>>>(
        logits_finite, tokens, routing_finite,
        mtp_verifier.As<MtpChainResult>(mtp_chain_result_offset));
    error = cudaGetLastError();
    return error == cudaSuccess
               ? Status::Ok()
               : CudaFailure("latch M25 chain verifier finite state", error);
  }

  error = cudaMemcpyAsync(&mtp_verifier_host->transaction, transaction,
                          sizeof(MtpGroupTransaction),
                          cudaMemcpyDeviceToHost, stream);
  if (error == cudaSuccess) {
    error = cudaMemcpyAsync(mtp_verifier_host->logits_finite.data(),
                            logits_finite, tokens * sizeof(int),
                            cudaMemcpyDeviceToHost, stream);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpyAsync(mtp_verifier_host->top_values.data(), top_values,
                            tokens * sizeof(float), cudaMemcpyDeviceToHost,
                            stream);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpyAsync(mtp_verifier_host->second_values.data(),
                            second_values, tokens * sizeof(float),
                            cudaMemcpyDeviceToHost, stream);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpyAsync(&mtp_verifier_host->routing_finite,
                            routing_finite, sizeof(int),
                            cudaMemcpyDeviceToHost, stream);
  }
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("copy M25 fixed graph transaction", error);
}

Status Gemma4Moe26BReferenceEngine::Impl::PrepareFixedMtpGraph(
    std::uint32_t draft_count) {
  if (draft_count == 0U || draft_count > kMaximumMtpDraftTokens) {
    return Invalid("M25 fixed graph draft count is invalid");
  }
  if (mtp_group_graphs[draft_count] != nullptr) return Status::Ok();
  if (draft_count != 1U && draft_count != 2U && draft_count != 4U) {
    return Status(StatusCode::kUnsupported,
                  "M25 fixed graph kernel body is not implemented for this D");
  }
  cudaError_t error = cudaStreamSynchronize(stream);
  if (error != cudaSuccess) {
    return CudaFailure("synchronize before M25 fixed graph capture", error);
  }
  std::size_t free_before = 0U;
  std::size_t total_bytes = 0U;
  error = cudaMemGetInfo(&free_before, &total_bytes);
  if (error != cudaSuccess) {
    return CudaFailure("measure before M25 fixed graph capture", error);
  }
  error = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
  if (error != cudaSuccess) {
    return CudaFailure("begin M25 fixed graph capture", error);
  }
  const Status body = LaunchFixedMtpGraphBody(draft_count, true);
  cudaGraph_t graph = nullptr;
  error = cudaStreamEndCapture(stream, &graph);
  if (!body.ok()) {
    if (graph != nullptr) (void)cudaGraphDestroy(graph);
    return body;
  }
  if (error != cudaSuccess) {
    if (graph != nullptr) (void)cudaGraphDestroy(graph);
    return CudaFailure("end M25 fixed graph capture", error);
  }
  cudaGraphExec_t executable = nullptr;
  error = cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0U);
  const cudaError_t destroy_error = cudaGraphDestroy(graph);
  if (error != cudaSuccess) {
    return CudaFailure("instantiate M25 fixed graph", error);
  }
  if (destroy_error != cudaSuccess) {
    (void)cudaGraphExecDestroy(executable);
    return CudaFailure("destroy M25 fixed graph source", destroy_error);
  }
  std::size_t free_after = 0U;
  error = cudaMemGetInfo(&free_after, &total_bytes);
  if (error != cudaSuccess) {
    (void)cudaGraphExecDestroy(executable);
    return CudaFailure("measure after M25 fixed graph capture", error);
  }
  const std::uint64_t required_margin =
      context >= 65536U ? 200U * kMiB : 700U * kMiB;
  if (free_after < required_margin) {
    (void)cudaGraphExecDestroy(executable);
    return Status(StatusCode::kResourceExhausted,
                  "M25 fixed graph leaves less than the profile margin");
  }
  mtp_group_graphs[draft_count] = executable;
  if (free_before > free_after) {
    mtp_group_graph_device_bytes += free_before - free_after;
  }
  return Status::Ok();
}

Status Gemma4Moe26BReferenceEngine::Impl::ExecuteFixedMtpGraphGroup(
    std::uint32_t pending_token, std::uint32_t draft_count,
    MtpGroupResult* host_result) {
  if (draft_count == 0U || draft_count > kMaximumMtpDraftTokens ||
      host_result == nullptr || mtp_group_graphs[draft_count] == nullptr) {
    return Invalid("M25 fixed graph execution state is invalid");
  }
  const std::uint64_t tokens = draft_count + 1U;
  *mtp_verifier_host = {};
  auto& host_control = mtp_verifier_host->transaction.control;
  host_control.current.input_token = pending_token;
  host_control.current.processed_position = position - 1U;
  host_control.current.remaining_output_capacity = context - position;
  host_control.current.output_write_position = 0U;
  host_control.current.sampling_step = sampling_step;
  host_control.next = host_control.current;
  host_control.fixed_draft_tokens = draft_count;
  host_control.sampling_enabled = sampling.enabled ? 1U : 0U;
  auto* transaction =
      mtp_verifier.As<MtpGroupTransaction>(mtp_transaction_offset);
  cudaError_t error = cudaMemcpyAsync(
      transaction, &mtp_verifier_host->transaction,
      sizeof(MtpGroupTransaction), cudaMemcpyHostToDevice, stream);
  if (error != cudaSuccess) {
    return CudaFailure("initialize M25 fixed graph transaction", error);
  }
  error = cudaGraphLaunch(mtp_group_graphs[draft_count], stream);
  if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
  if (error != cudaSuccess) {
    return CudaFailure("execute M25 fixed graph group", error);
  }
  ++mtp_group_graph_launches;

  const auto& completed = mtp_verifier_host->transaction;
  if (completed.control.current.input_token != pending_token ||
      completed.control.current.processed_position + 1U != position ||
      completed.control.fixed_draft_tokens != draft_count ||
      completed.control.transition_valid != 1U ||
      completed.control.proposal_count != draft_count ||
      completed.result.output_count == 0U ||
      completed.result.output_count > tokens ||
      completed.control.next.input_token !=
          completed.result.verified[completed.result.output_count - 1U] ||
      completed.control.next.processed_position !=
          completed.control.current.processed_position +
              completed.result.output_count) {
    return Status(StatusCode::kInternal,
                  "M25 fixed graph device transition is inconsistent");
  }
  if (mtp_verifier_host->routing_finite == 0 ||
      std::any_of(mtp_verifier_host->logits_finite.begin(),
                  mtp_verifier_host->logits_finite.begin() + tokens,
                  [](int value) { return value == 0; })) {
    return Status(StatusCode::kInternal,
                  "M25 fixed graph produced non-finite Target values");
  }
  last_mtp_verification_min_margin =
      std::numeric_limits<float>::infinity();
  for (std::uint64_t row = 0U; row < tokens; ++row) {
    last_mtp_verification_min_margin = std::min(
        last_mtp_verification_min_margin,
        mtp_verifier_host->top_values[row] -
            mtp_verifier_host->second_values[row]);
  }

  const std::uint64_t previous_position = position;
  position += completed.result.output_count;
  if (sliding_capacity != 0U) {
    sliding_ring_wraps += position / sliding_capacity -
                          previous_position / sliding_capacity;
  }
  maximum_global_position_exclusive =
      std::max(maximum_global_position_exclusive, position);
  token_selections += completed.result.output_count;
  sampling_step = completed.control.next.sampling_step;
  pending_decode_self_feed = false;
  decode_self_feed_valid = false;
  has_last_input_token = true;
  last_input_token = completed.result.output_count == 1U
                         ? pending_token
                         : completed.result.proposed[
                               completed.result.output_count - 2U];
  *host_result = completed.result;
  return Status::Ok();
}

Status Gemma4Moe26BReferenceEngine::Impl::LaunchFixedMtpOrdinaryTailBody() {
  auto* transaction =
      mtp_verifier.As<MtpGroupTransaction>(mtp_transaction_offset);
  Status status = LaunchInitializeMtpOrdinaryTail(
      transaction, decode_control, suppressed_token_count, stream);
  if (!status.ok()) return status;
  if (sampling.enabled && sampling.repetition_penalty != 1.0F) {
    status = LaunchMarkControlledRepetitionToken(
        decode_control, repetition_mask, stream);
    if (!status.ok()) return status;
  }
  status = LaunchControlledDecodeBody();
  if (!status.ok()) return status;
  // The ordinary graph normally keeps only the fused final activation/head.
  // MTP needs the normalized Target hidden row for the next Assistant input.
  status = LaunchRmsNormBf16(hidden_a, final_norm, final_hidden, 1U, kWidth,
                             1.0e-6F, stream);
  if (!status.ok()) return status;
  LatchMtpOrdinaryNonFiniteKernel<<<1U, 1U, 0, stream>>>(
      prediction_device_status,
      mtp_verifier.As<MtpChainResult>(mtp_chain_result_offset));
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaFailure("latch M25 chain ordinary finite state", error);
  }
  const std::uint32_t* selected = prediction_token;
  if (sampling.enabled) {
    status = LaunchSampleToken(
        logits, sampling_logits, sampling_cumulative, sampling_token_ids,
        sampling_sorted_token_ids, repetition_mask, suppressed_token_ids,
        suppressed_token_count, static_cast<std::uint32_t>(kVocabulary),
        sampling, 0U, decode_control, selected_token,
        sampling_algorithm_workspace, sampling_algorithm_workspace_bytes,
        stream);
    if (!status.ok()) return status;
    selected = selected_token;
  }
  return LaunchFinalizeMtpOrdinaryTail(
      selected, mtp_verifier.As<std::uint32_t>(mtp_stop_tokens_offset),
      mtp_stop_token_count, transaction,
      mtp_verifier.As<MtpChainResult>(mtp_chain_result_offset),
      mtp_verifier.As<std::uint32_t>(mtp_chain_outputs_offset), nullptr,
      stream);
}

Status Gemma4Moe26BReferenceEngine::Impl::PrepareFixedMtpChainGraph(
    std::uint32_t draft_count) {
  if (draft_count == 0U || draft_count > kMaximumMtpDraftTokens) {
    return Invalid("M25 fixed chain graph draft count is invalid");
  }
  if (mtp_chain_graphs[draft_count] != nullptr) return Status::Ok();
  if ((draft_count != 1U && draft_count != 2U && draft_count != 4U) ||
      mtp_group_graphs[draft_count] == nullptr ||
      mtp_verifier_backend !=
          Gemma4Moe26BMtpVerifierBackend::kExactSharedBatchedMoe) {
    return Status(StatusCode::kUnsupported,
                  "M25 fixed chain graph is unavailable for this D/backend");
  }
  cudaError_t error = cudaStreamSynchronize(stream);
  if (error != cudaSuccess) {
    return CudaFailure("synchronize before M25 chain graph capture", error);
  }
  std::size_t free_before = 0U;
  std::size_t total_bytes = 0U;
  error = cudaMemGetInfo(&free_before, &total_bytes);
  if (error != cudaSuccess) {
    return CudaFailure("measure before M25 chain graph capture", error);
  }

  cudaGraph_t root = nullptr;
  cudaGraph_t route_source = nullptr;
  cudaGraph_t d2_source = nullptr;
  cudaGraph_t ordinary_source = nullptr;
  cudaGraph_t continue_source = nullptr;
  cudaGraphExec_t executable = nullptr;
  const auto cleanup = [&]() {
    if (route_source != nullptr) (void)cudaGraphDestroy(route_source);
    if (d2_source != nullptr) (void)cudaGraphDestroy(d2_source);
    if (ordinary_source != nullptr) (void)cudaGraphDestroy(ordinary_source);
    if (continue_source != nullptr) (void)cudaGraphDestroy(continue_source);
    if (root != nullptr) (void)cudaGraphDestroy(root);
    if (executable != nullptr) (void)cudaGraphExecDestroy(executable);
  };
  error = cudaGraphCreate(&root, 0U);
  if (error != cudaSuccess) {
    return CudaFailure("create M25 fixed chain graph", error);
  }
  cudaGraphConditionalHandle loop_condition = 0U;
  error = cudaGraphConditionalHandleCreate(
      &loop_condition, root, 1U, cudaGraphCondAssignDefault);
  if (error != cudaSuccess) {
    cleanup();
    return CudaFailure("create M25 chain loop condition", error);
  }
  cudaGraphConditionalHandle d2_condition = 0U;
  error = cudaGraphConditionalHandleCreate(
      &d2_condition, root, 0U, cudaGraphCondAssignDefault);
  if (error != cudaSuccess) {
    cleanup();
    return CudaFailure("create M25 chain D2 condition", error);
  }
  cudaGraphConditionalHandle ordinary_condition = 0U;
  error = cudaGraphConditionalHandleCreate(
      &ordinary_condition, root, 0U, cudaGraphCondAssignDefault);
  if (error != cudaSuccess) {
    cleanup();
    return CudaFailure("create M25 chain ordinary condition", error);
  }
  cudaGraphNodeParams loop_parameters{};
  loop_parameters.type = cudaGraphNodeTypeConditional;
  loop_parameters.conditional.handle = loop_condition;
  loop_parameters.conditional.type = cudaGraphCondTypeWhile;
  loop_parameters.conditional.size = 1U;
  cudaGraphNode_t loop_node = nullptr;
  error = cudaGraphAddNode(&loop_node, root, nullptr, nullptr, 0U,
                           &loop_parameters);
  if (error != cudaSuccess) {
    cleanup();
    return CudaFailure("add M25 chain loop", error);
  }

  auto* transaction =
      mtp_verifier.As<MtpGroupTransaction>(mtp_transaction_offset);
  auto* chain_result =
      mtp_verifier.As<MtpChainResult>(mtp_chain_result_offset);
  auto* chain_outputs =
      mtp_verifier.As<std::uint32_t>(mtp_chain_outputs_offset);
  auto* chain_proposals =
      mtp_verifier.As<std::uint32_t>(mtp_chain_proposals_offset);

  error = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
  if (error != cudaSuccess) {
    cleanup();
    return CudaFailure("begin M25 chain route capture", error);
  }
  Status status = LaunchSelectMtpChainBranch(
      transaction, d2_condition, ordinary_condition, stream);
  error = cudaStreamEndCapture(stream, &route_source);
  if (!status.ok() || error != cudaSuccess) {
    cleanup();
    return !status.ok() ? status
                        : CudaFailure("end M25 chain route capture", error);
  }

  error = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
  if (error != cudaSuccess) {
    cleanup();
    return CudaFailure("begin M25 chain D2 capture", error);
  }
  status = LaunchFixedMtpGraphBody(draft_count, false);
  if (status.ok()) {
    status = LaunchAdvanceMtpChain(
        transaction, chain_result, chain_outputs, chain_proposals, nullptr,
        stream);
  }
  error = cudaStreamEndCapture(stream, &d2_source);
  if (!status.ok() || error != cudaSuccess) {
    cleanup();
    return !status.ok() ? status
                        : CudaFailure("end M25 chain D2 capture", error);
  }

  error = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
  if (error != cudaSuccess) {
    cleanup();
    return CudaFailure("begin M25 chain ordinary capture", error);
  }
  status = LaunchFixedMtpOrdinaryTailBody();
  error = cudaStreamEndCapture(stream, &ordinary_source);
  if (!status.ok() || error != cudaSuccess) {
    cleanup();
    return !status.ok()
               ? status
               : CudaFailure("end M25 chain ordinary capture", error);
  }

  error = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
  if (error != cudaSuccess) {
    cleanup();
    return CudaFailure("begin M25 chain continuation capture", error);
  }
  status = LaunchContinueMtpChain(
      transaction, nullptr, loop_condition, stream);
  error = cudaStreamEndCapture(stream, &continue_source);
  if (!status.ok() || error != cudaSuccess) {
    cleanup();
    return !status.ok()
               ? status
               : CudaFailure("end M25 chain continuation capture", error);
  }

  cudaGraph_t loop_body = loop_parameters.conditional.phGraph_out[0];
  cudaGraphNode_t route_node = nullptr;
  error = cudaGraphAddChildGraphNode(&route_node, loop_body, nullptr, 0U,
                                     route_source);
  if (error != cudaSuccess) {
    cleanup();
    return CudaFailure("add M25 chain route", error);
  }
  cudaGraphNodeParams d2_parameters{};
  d2_parameters.type = cudaGraphNodeTypeConditional;
  d2_parameters.conditional.handle = d2_condition;
  d2_parameters.conditional.type = cudaGraphCondTypeIf;
  d2_parameters.conditional.size = 1U;
  cudaGraphNode_t d2_node = nullptr;
  error = cudaGraphAddNode(&d2_node, loop_body, &route_node, nullptr, 1U,
                           &d2_parameters);
  if (error != cudaSuccess) {
    cleanup();
    return CudaFailure("add M25 chain D2 branch", error);
  }
  cudaGraphNode_t d2_body_node = nullptr;
  error = cudaGraphAddChildGraphNode(
      &d2_body_node, d2_parameters.conditional.phGraph_out[0], nullptr, 0U,
      d2_source);
  if (error != cudaSuccess) {
    cleanup();
    return CudaFailure("add M25 chain D2 body", error);
  }
  cudaGraphNodeParams ordinary_parameters{};
  ordinary_parameters.type = cudaGraphNodeTypeConditional;
  ordinary_parameters.conditional.handle = ordinary_condition;
  ordinary_parameters.conditional.type = cudaGraphCondTypeIf;
  ordinary_parameters.conditional.size = 1U;
  cudaGraphNode_t ordinary_node = nullptr;
  error = cudaGraphAddNode(&ordinary_node, loop_body, &route_node, nullptr,
                           1U, &ordinary_parameters);
  if (error != cudaSuccess) {
    cleanup();
    return CudaFailure("add M25 chain ordinary branch", error);
  }
  cudaGraphNode_t ordinary_body_node = nullptr;
  error = cudaGraphAddChildGraphNode(
      &ordinary_body_node, ordinary_parameters.conditional.phGraph_out[0],
      nullptr, 0U, ordinary_source);
  if (error != cudaSuccess) {
    cleanup();
    return CudaFailure("add M25 chain ordinary body", error);
  }
  const cudaGraphNode_t branch_nodes[] = {d2_node, ordinary_node};
  cudaGraphNode_t continue_node = nullptr;
  error = cudaGraphAddChildGraphNode(
      &continue_node, loop_body, branch_nodes, 2U, continue_source);
  if (error != cudaSuccess) {
    cleanup();
    return CudaFailure("add M25 chain continuation", error);
  }
  error = cudaGraphInstantiate(&executable, root, nullptr, nullptr, 0U);
  if (error != cudaSuccess) {
    cleanup();
    return CudaFailure("instantiate M25 fixed chain graph", error);
  }
  mtp_chain_graphs[draft_count] = executable;
  executable = nullptr;
  cleanup();

  std::size_t free_after = 0U;
  error = cudaMemGetInfo(&free_after, &total_bytes);
  if (error != cudaSuccess) {
    (void)cudaGraphExecDestroy(mtp_chain_graphs[draft_count]);
    mtp_chain_graphs[draft_count] = nullptr;
    return CudaFailure("measure after M25 chain graph capture", error);
  }
  const std::uint64_t required_margin =
      context >= 65536U ? 200U * kMiB : 700U * kMiB;
  if (free_after < required_margin) {
    (void)cudaGraphExecDestroy(mtp_chain_graphs[draft_count]);
    mtp_chain_graphs[draft_count] = nullptr;
    return Status(StatusCode::kResourceExhausted,
                  "M25 chain graph leaves less than the profile margin");
  }
  if (free_before > free_after) {
    mtp_chain_graph_device_bytes[draft_count] += free_before - free_after;
  }
  return Status::Ok();
}

Gemma4Moe26BReferenceEngine::Gemma4Moe26BReferenceEngine() = default;
Gemma4Moe26BReferenceEngine::~Gemma4Moe26BReferenceEngine() = default;
Gemma4Moe26BReferenceEngine::Gemma4Moe26BReferenceEngine(
    Gemma4Moe26BReferenceEngine&&) noexcept = default;
Gemma4Moe26BReferenceEngine& Gemma4Moe26BReferenceEngine::operator=(
    Gemma4Moe26BReferenceEngine&&) noexcept = default;
Gemma4Moe26BReferenceEngine::Gemma4Moe26BReferenceEngine(
    std::unique_ptr<Impl> implementation)
    : implementation_(std::move(implementation)) {}

Result<Gemma4Moe26BReferenceEngine> Gemma4Moe26BReferenceEngine::Create(
    const std::filesystem::path& model_directory,
    std::uint64_t context_tokens, int device, Gemma4Moe26BBackend backend) {
  if (context_tokens == 0U || context_tokens > kMaximumContextTokens ||
      device < 0) {
    return Invalid("Gemma 4 26B context must be in [1, 262144]");
  }
  cudaError_t error = cudaSetDevice(device);
  if (error != cudaSuccess) return CudaFailure("select M13 CUDA device", error);
  auto config = LoadModelConfig(model_directory / "config.json");
  if (!config.ok()) return config.status();
  if (context_tokens > config.value().max_positions) {
    return Invalid("Gemma 4 26B context exceeds the model maximum");
  }
  Status valid = ValidateGemma4Moe26BContract(config.value());
  if (!valid.ok()) return valid;
  auto manifest = InspectCheckpoint({model_directory, true});
  if (!manifest.ok()) return manifest.status();
  auto traits = BuildGemma4Moe26BAttentionTraits(config.value());
  if (!traits.ok()) return traits.status();
  valid = ValidateGemma4Moe26BAttentionBindings(manifest.value().tensors,
                                                traits.value());
  if (!valid.ok()) return valid;
  auto plan = BuildGemma4Moe26BResidencyPlan(manifest.value(), config.value());
  if (!plan.ok()) return plan.status();
  auto artifact = Gemma4Moe26BDeviceArtifact::Load(
      model_directory, manifest.value(), plan.value());
  if (!artifact.ok()) return artifact.status();

  auto impl = std::make_unique<Impl>();
  impl->device = device;
  impl->context = context_tokens;
  impl->backend = backend;
  if (backend == Gemma4Moe26BBackend::kSm120Integrated) {
    impl->moe_config.prefill_router =
        Gemma4MoePrefillRouter::kSm120TensorCore;
    impl->moe_config.materialize_native_router_normalized = false;
  }
  impl->traits = traits.value();
  for (const auto& trait : impl->traits) {
    if (trait.attention == Gemma4Moe26BAttentionType::kSliding) {
      if (impl->sliding_capacity == 0U) {
        impl->sliding_capacity = trait.cache_capacity;
      } else if (impl->sliding_capacity != trait.cache_capacity) {
        return Status(StatusCode::kDataLoss,
                      "Gemma 4 26B sliding cache capacities disagree");
      }
    }
  }
  if (impl->sliding_capacity == 0U) {
    return Status(StatusCode::kDataLoss,
                  "Gemma 4 26B has no validated sliding cache layer");
  }
  impl->artifact = std::move(artifact).value();
  error = cudaStreamCreateWithFlags(&impl->stream, cudaStreamNonBlocking);
  if (error != cudaSuccess) return CudaFailure("create M13 stream", error);
  if (backend == Gemma4Moe26BBackend::kSm120Integrated) {
    error = cudaStreamCreateWithFlags(&impl->shared_moe_stream,
                                      cudaStreamNonBlocking);
    if (error != cudaSuccess) {
      return CudaFailure("create M20 shared-MoE stream", error);
    }
    error = cudaEventCreateWithFlags(&impl->shared_moe_fork,
                                     cudaEventDisableTiming);
    if (error == cudaSuccess) {
      error = cudaEventCreateWithFlags(&impl->shared_moe_join,
                                       cudaEventDisableTiming);
    }
    if (error != cudaSuccess) {
      return CudaFailure("create M20 shared-MoE events", error);
    }
  }

  for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
    auto attention = BindGemma4Moe26BAttentionReferenceWeights(
        impl->artifact, impl->traits[layer]);
    if (!attention.ok()) return attention.status();
    impl->attention_weights[layer] = attention.value();
    auto moe = BindGemma4Moe26BReferenceWeights(impl->artifact, layer);
    if (!moe.ok()) return moe.status();
    impl->moe_weights[layer] = moe.value();
  }

  auto head_packed = ArtifactPointer<std::uint8_t>(
      impl->artifact, "model.language_model.embed_tokens.weight_packed");
  auto head_scales = ArtifactPointer<std::uint8_t>(
      impl->artifact, "model.language_model.embed_tokens.weight_scale");
  auto final_norm = ArtifactPointer<std::uint16_t>(
      impl->artifact, "model.language_model.norm.weight");
  auto head_activation_divisor = impl->artifact.HostFloat32(
      "model.language_model.embed_tokens.input_global_scale");
  auto head_weight_divisor = impl->artifact.HostFloat32(
      "model.language_model.embed_tokens.weight_global_scale");
  if (!head_packed.ok()) return head_packed.status();
  if (!head_scales.ok()) return head_scales.status();
  if (!final_norm.ok()) return final_norm.status();
  if (!head_activation_divisor.ok()) return head_activation_divisor.status();
  if (!head_weight_divisor.ok()) return head_weight_divisor.status();
  impl->head = {head_packed.value(), head_scales.value(), kVocabulary, kWidth,
                head_activation_divisor.value(), head_weight_divisor.value()};
  impl->final_norm = final_norm.value();
  impl->softcap = static_cast<float>(config.value().final_logit_softcap);

  auto kv_bytes = Gemma4Moe26BFp8KvBytes(impl->traits, context_tokens);
  if (!kv_bytes.ok()) return kv_bytes.status();
  valid = impl->kv.Allocate(kv_bytes.value(), "allocate M13 FP8 K/V arena");
  if (!valid.ok()) return valid;
  std::uint64_t kv_offset = 0U;
  for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
    const auto& trait = impl->traits[layer];
    const std::uint64_t capacity =
        trait.attention == Gemma4Moe26BAttentionType::kSliding
            ? trait.cache_capacity
            : context_tokens;
    const std::uint64_t one = capacity * trait.kv_heads * trait.head_dimension;
    impl->caches[layer] = {impl->kv.As<std::uint8_t>(kv_offset),
                           impl->kv.As<std::uint8_t>(kv_offset + one),
                           capacity};
    kv_offset += 2U * one;
  }
  if (kv_offset != kv_bytes.value()) {
    return Status(StatusCode::kInternal,
                  "M13 K/V partition does not match the accepted byte formula");
  }

  LayoutBuilder layout;
  const auto hidden_a = layout.Add<float>(kWidth);
  const auto hidden_b = layout.Add<float>(kWidth);
  const auto final_hidden = layout.Add<float>(kWidth);
  const auto a_input_fp8 = layout.Add<std::uint8_t>(kWidth);
  const auto a_input_scale = layout.Add<float>(1U);
  const auto q_raw = layout.Add<float>(16U * 512U);
  const auto k_raw = layout.Add<float>(8U * 512U);
  const auto v_raw = layout.Add<float>(8U * 512U);
  const auto q_norm = layout.Add<float>(16U * 512U);
  const auto k_norm = layout.Add<float>(8U * 512U);
  const auto v_norm = layout.Add<float>(8U * 512U);
  const auto cosine = layout.Add<float>(256U);
  const auto sine = layout.Add<float>(256U);
  const auto local_cosine = layout.Add<float>(256U);
  const auto local_sine = layout.Add<float>(256U);
  const auto staged_k = layout.Add<std::uint8_t>(8U * 512U);
  const auto staged_v = layout.Add<std::uint8_t>(8U * 512U);
  const std::uint64_t decode_attention_workspace_elements = std::max(
      std::max(16U * context_tokens,
               DecodeAttentionWorkspaceElements(context_tokens)),
      DecodeAttentionWorkspaceElements(impl->sliding_capacity));
  const auto scores =
      layout.Add<float>(decode_attention_workspace_elements);
  const auto attention = layout.Add<float>(16U * 512U);
  const auto output_fp8 = layout.Add<std::uint8_t>(16U * 512U);
  const auto output_scale = layout.Add<float>(1U);
  const auto output_projection = layout.Add<float>(kWidth);
  const auto post_attention = layout.Add<float>(kWidth);

  const auto shared_input = layout.Add<float>(kWidth);
  const auto shared_input_packed = layout.Add<std::uint8_t>(kWidth / 2U);
  const auto shared_input_scales = layout.Add<std::uint8_t>(kWidth / 16U);
  const auto shared_gate = layout.Add<float>(kShared);
  const auto shared_up = layout.Add<float>(kShared);
  const auto shared_product = layout.Add<float>(kShared);
  const auto shared_product_packed = layout.Add<std::uint8_t>(kShared / 2U);
  const auto shared_product_scales = layout.Add<std::uint8_t>(kShared / 16U);
  const auto shared_output = layout.Add<float>(kWidth);
  const auto shared_post = layout.Add<float>(kWidth);
  const auto router_normalized = layout.Add<float>(kWidth);
  const auto router_transformed = layout.Add<float>(kWidth);
  const auto router_logits = layout.Add<float>(kExperts);
  const auto router_probabilities = layout.Add<float>(kExperts);
  const auto top_ids = layout.Add<std::uint32_t>(kTopK);
  const auto top_weights = layout.Add<float>(kTopK);
  const auto expert_input = layout.Add<float>(kWidth);
  const auto expert_input_packed = layout.Add<std::uint8_t>(kWidth / 2U);
  const auto expert_input_scales = layout.Add<std::uint8_t>(kWidth / 16U);
  const auto expert_gate_up = layout.Add<float>(kTopK * 2U * kExpert);
  const auto expert_product = layout.Add<float>(kTopK * kExpert);
  const auto expert_product_packed =
      layout.Add<std::uint8_t>(kTopK * kExpert / 2U);
  const auto expert_product_scales =
      layout.Add<std::uint8_t>(kTopK * kExpert / 16U);
  const auto expert_down = layout.Add<float>(kTopK * kWidth);
  const auto expert_contributions = layout.Add<float>(kTopK * kWidth);
  const auto routed_sum = layout.Add<float>(kWidth);
  const auto routed_post = layout.Add<float>(kWidth);
  const auto combined = layout.Add<float>(kWidth);
  const auto feed_forward = layout.Add<float>(kWidth);
  const auto head_activation = layout.Add<std::uint8_t>(kWidth / 2U);
  const auto head_activation_scale_offset =
      layout.Add<std::uint8_t>(kWidth / 16U);
  const auto logits = layout.Add<float>(kVocabulary);
  const auto prediction_status = layout.Add<PredictionStatus>(1U);
  const auto output_candidates = layout.Add<LogitCandidate>(kArgmaxBlocks);
  const auto decode_control = layout.Add<DecodeControl>(1U);
  auto sampling_workspace_bytes = SamplingWorkspaceBytes(
      static_cast<std::uint32_t>(kVocabulary), impl->stream);
  if (!sampling_workspace_bytes.ok()) {
    return sampling_workspace_bytes.status();
  }
  const auto sampling_logits = layout.Add<float>(kVocabulary);
  const auto sampling_cumulative = layout.Add<double>(kVocabulary);
  const auto sampling_token_ids = layout.Add<std::uint32_t>(kVocabulary);
  const auto sampling_sorted_token_ids =
      layout.Add<std::uint32_t>(kVocabulary);
  const auto repetition_mask =
      layout.Add<std::uint32_t>(kRepetitionMaskWords);
  const auto suppressed_token_ids =
      layout.Add<std::uint32_t>(kMaximumSuppressedTokens);
  const auto selected_token = layout.Add<std::uint32_t>(1U);
  const auto sampling_algorithm_workspace = layout.Add<std::uint8_t>(
      sampling_workspace_bytes.value());
  std::array<std::uint64_t, kCaptureLayers.size()> capture_outputs{};
  std::array<std::uint64_t, kCaptureLayers.size()> capture_probs{};
  std::array<std::uint64_t, kCaptureLayers.size()> capture_ids{};
  for (std::size_t i = 0; i < kCaptureLayers.size(); ++i) {
    capture_outputs[i] = layout.Add<float>(kWidth);
    capture_probs[i] = layout.Add<float>(kExperts);
    capture_ids[i] = layout.Add<std::uint32_t>(kTopK);
  }
  if (layout.bytes == std::numeric_limits<std::uint64_t>::max()) {
    return Invalid("M13 workspace layout overflow");
  }
  valid = impl->workspace.Allocate(layout.bytes, "allocate M13 fixed workspace");
  if (!valid.ok()) return valid;
  auto ptr = [&](std::uint64_t offset) {
    return impl->workspace.As<std::byte>(offset);
  };
  impl->hidden_a = reinterpret_cast<float*>(ptr(hidden_a));
  impl->hidden_b = reinterpret_cast<float*>(ptr(hidden_b));
  impl->final_hidden = reinterpret_cast<float*>(ptr(final_hidden));
  impl->local_rotary_cosine = reinterpret_cast<float*>(ptr(local_cosine));
  impl->local_rotary_sine = reinterpret_cast<float*>(ptr(local_sine));
  impl->attention_workspace = {
      reinterpret_cast<std::uint8_t*>(ptr(a_input_fp8)),
      reinterpret_cast<float*>(ptr(a_input_scale)),
      reinterpret_cast<float*>(ptr(q_raw)), reinterpret_cast<float*>(ptr(k_raw)),
      reinterpret_cast<float*>(ptr(v_raw)), reinterpret_cast<float*>(ptr(q_norm)),
      reinterpret_cast<float*>(ptr(k_norm)), reinterpret_cast<float*>(ptr(v_norm)),
      reinterpret_cast<float*>(ptr(cosine)), reinterpret_cast<float*>(ptr(sine)),
      reinterpret_cast<std::uint8_t*>(ptr(staged_k)),
      reinterpret_cast<std::uint8_t*>(ptr(staged_v)),
      reinterpret_cast<float*>(ptr(scores)),
      decode_attention_workspace_elements,
      reinterpret_cast<float*>(ptr(attention)),
      reinterpret_cast<std::uint8_t*>(ptr(output_fp8)),
      reinterpret_cast<float*>(ptr(output_scale)),
      reinterpret_cast<float*>(ptr(output_projection)),
      reinterpret_cast<float*>(ptr(post_attention)), nullptr, 0U};
  impl->moe_workspace = {
      reinterpret_cast<float*>(ptr(shared_input)),
      reinterpret_cast<std::uint8_t*>(ptr(shared_input_packed)),
      reinterpret_cast<std::uint8_t*>(ptr(shared_input_scales)),
      reinterpret_cast<float*>(ptr(shared_gate)),
      reinterpret_cast<float*>(ptr(shared_up)),
      reinterpret_cast<float*>(ptr(shared_product)),
      reinterpret_cast<std::uint8_t*>(ptr(shared_product_packed)),
      reinterpret_cast<std::uint8_t*>(ptr(shared_product_scales)),
      reinterpret_cast<float*>(ptr(shared_output)),
      reinterpret_cast<float*>(ptr(shared_post)),
      reinterpret_cast<float*>(ptr(router_normalized)),
      reinterpret_cast<float*>(ptr(router_transformed)),
      reinterpret_cast<float*>(ptr(router_logits)),
      reinterpret_cast<float*>(ptr(router_probabilities)),
      reinterpret_cast<std::uint32_t*>(ptr(top_ids)),
      reinterpret_cast<float*>(ptr(top_weights)),
      reinterpret_cast<float*>(ptr(expert_input)),
      reinterpret_cast<std::uint8_t*>(ptr(expert_input_packed)),
      reinterpret_cast<std::uint8_t*>(ptr(expert_input_scales)),
      reinterpret_cast<float*>(ptr(expert_gate_up)),
      reinterpret_cast<float*>(ptr(expert_product)),
      reinterpret_cast<std::uint8_t*>(ptr(expert_product_packed)),
      reinterpret_cast<std::uint8_t*>(ptr(expert_product_scales)),
      reinterpret_cast<float*>(ptr(expert_down)),
      reinterpret_cast<float*>(ptr(expert_contributions)),
      reinterpret_cast<float*>(ptr(routed_sum)),
      reinterpret_cast<float*>(ptr(routed_post)),
      reinterpret_cast<float*>(ptr(combined)),
      reinterpret_cast<float*>(ptr(feed_forward)),
      reinterpret_cast<int*>(
          ptr(prediction_status) +
          offsetof(PredictionStatus, routing_finite))};
  impl->head_activation = reinterpret_cast<std::uint8_t*>(ptr(head_activation));
  impl->head_activation_scales =
      reinterpret_cast<std::uint8_t*>(ptr(head_activation_scale_offset));
  impl->logits = reinterpret_cast<float*>(ptr(logits));
  impl->prediction_device_status =
      reinterpret_cast<PredictionStatus*>(ptr(prediction_status));
  auto* prediction_status_bytes = reinterpret_cast<std::byte*>(
      impl->prediction_device_status);
  impl->prediction_token = reinterpret_cast<std::uint32_t*>(
      prediction_status_bytes + offsetof(PredictionStatus, token));
  impl->prediction_logit = reinterpret_cast<float*>(
      prediction_status_bytes + offsetof(PredictionStatus, logit));
  impl->finite = reinterpret_cast<int*>(
      prediction_status_bytes + offsetof(PredictionStatus, finite));
  impl->routing_finite = reinterpret_cast<int*>(
      prediction_status_bytes + offsetof(PredictionStatus, routing_finite));
  error = cudaMallocHost(&impl->prediction_host_status,
                         sizeof(PredictionStatus));
  if (error != cudaSuccess) {
    return CudaFailure("allocate pinned M20 prediction status", error);
  }
  impl->output_candidates =
      reinterpret_cast<LogitCandidate*>(ptr(output_candidates));
  impl->decode_control = reinterpret_cast<DecodeControl*>(ptr(decode_control));
  impl->sampling_logits = reinterpret_cast<float*>(ptr(sampling_logits));
  impl->sampling_cumulative =
      reinterpret_cast<double*>(ptr(sampling_cumulative));
  impl->sampling_token_ids =
      reinterpret_cast<std::uint32_t*>(ptr(sampling_token_ids));
  impl->sampling_sorted_token_ids =
      reinterpret_cast<std::uint32_t*>(ptr(sampling_sorted_token_ids));
  impl->repetition_mask =
      reinterpret_cast<std::uint32_t*>(ptr(repetition_mask));
  impl->suppressed_token_ids =
      reinterpret_cast<std::uint32_t*>(ptr(suppressed_token_ids));
  impl->selected_token =
      reinterpret_cast<std::uint32_t*>(ptr(selected_token));
  impl->sampling_algorithm_workspace = ptr(sampling_algorithm_workspace);
  impl->sampling_algorithm_workspace_bytes =
      sampling_workspace_bytes.value();
  for (std::size_t i = 0; i < kCaptureLayers.size(); ++i) {
    impl->layer_captures[i] = reinterpret_cast<float*>(ptr(capture_outputs[i]));
    impl->router_probability_captures[i] =
        reinterpret_cast<float*>(ptr(capture_probs[i]));
    impl->router_id_captures[i] =
        reinterpret_cast<std::uint32_t*>(ptr(capture_ids[i]));
  }

  if (backend == Gemma4Moe26BBackend::kSm120Integrated) {
    LayoutBuilder prefill;
    const auto p_tokens = prefill.Add<std::uint32_t>(kPrefillMaxTokens);
    const auto p_hidden_a = prefill.Add<float>(kPrefillMaxTokens * kWidth);
    const auto p_hidden_b = prefill.Add<float>(kPrefillMaxTokens * kWidth);
    // The two RoPE profiles remain live across every attention/MoE pair in a
    // chunk. Keep them in the persistent prefix shared by both phase layouts;
    // placing them in attention-only scratch would let each intervening MoE
    // layer overwrite the tables before the next attention layer consumes
    // them.
    const auto p_global_cosine =
        prefill.Add<float>(kPrefillMaxTokens * 256U);
    const auto p_global_sine =
        prefill.Add<float>(kPrefillMaxTokens * 256U);
    const auto p_local_cosine =
        prefill.Add<float>(kPrefillMaxTokens * 256U);
    const auto p_local_sine =
        prefill.Add<float>(kPrefillMaxTokens * 256U);

    // Attention and MoE execute in-order on impl->stream. Build both phase
    // layouts from the persistent token/hidden prefix so their temporary
    // regions physically alias without overlapping live values.
    // Query-normalized rows stay live through attention. All projection
    // temporaries are dead by then, while prepared global K/V are dead before
    // output quantization. Model those three phases explicitly so the <=16K
    // BF16 staging does not increase the fixed M09 workspace.
    LayoutBuilder attention_common = prefill;
    const auto p_scores =
        attention_common.Add<float>(kPrefillScoreElements);
    const auto p_q_norm = attention_common.Add<float>(
        kPrefillMaxTokens * 16U * 512U);
    LayoutBuilder attention_projection = attention_common;
    const auto p_input_fp8 =
        attention_projection.Add<std::uint8_t>(kPrefillMaxTokens * kWidth);
    const auto p_input_scale =
        attention_projection.Add<float>(kPrefillMaxTokens);
    const auto p_q_raw =
        attention_projection.Add<std::uint16_t>(kPrefillMaxTokens * 16U * 512U);
    const auto p_k_raw =
        attention_projection.Add<std::uint16_t>(kPrefillMaxTokens * 2048U);
    const auto p_v_raw =
        attention_projection.Add<std::uint16_t>(kPrefillMaxTokens * 2048U);
    const auto p_k_norm =
        attention_projection.Add<float>(kPrefillMaxTokens * 2048U);
    const auto p_v_norm =
        attention_projection.Add<float>(kPrefillMaxTokens * 2048U);
    const auto p_staged_k =
        attention_projection.Add<std::uint8_t>(kPrefillMaxTokens * 2048U);
    const auto p_staged_v =
        attention_projection.Add<std::uint8_t>(kPrefillMaxTokens * 2048U);
    const auto p_cutlass_workspace = attention_projection.Add<std::byte>(
        kGemma4Moe26BAttentionCutlassWorkspaceBytes);

    LayoutBuilder attention_prepared = attention_common;
    const auto p_global_key_bf16 = attention_prepared.Add<std::uint16_t>(
        kPreparedGlobalPrefillTokens * 2U * 512U);
    const auto p_global_value_bf16 = attention_prepared.Add<std::uint16_t>(
        kPreparedGlobalPrefillTokens * 2U * 512U);
    const auto p_attention = attention_prepared.Add<std::uint16_t>(
        kPrefillMaxTokens * 16U * 512U);

    LayoutBuilder attention_post = attention_common;
    const auto p_output_fp8 =
        attention_post.Add<std::uint8_t>(kPrefillMaxTokens * 16U * 512U);
    const auto p_output_scale =
        attention_post.Add<float>(kPrefillMaxTokens);
    const auto p_output_projection =
        attention_post.Add<float>(kPrefillMaxTokens * kWidth);
    LayoutBuilder attention = attention_projection;
    attention.bytes = std::max(
        {attention_projection.bytes, attention_prepared.bytes,
         attention_post.bytes});

    LayoutBuilder moe = prefill;
    const auto p_router_logits =
        moe.Add<float>(kPrefillMaxTokens * kExperts);
    const auto p_router_probabilities =
        moe.Add<float>(kPrefillMaxTokens * kExperts);
    const auto p_token_hidden =
        moe.Add<float>(kPrefillMaxTokens * kWidth);
    const auto p_token_packed =
        moe.Add<std::uint8_t>(kPrefillMaxTokens * kWidth / 2U);
    const auto p_token_scales =
        moe.Add<std::uint8_t>(kPrefillMaxTokens * kWidth / 16U);
    const auto p_expert_product = moe.Add<std::uint16_t>(
        kPrefillMaxTokens * kTopK * kExpert);
    const auto p_expert_product_packed = moe.Add<std::uint8_t>(
        kPrefillMaxTokens * kTopK * kExpert / 2U);
    const auto p_expert_product_scales = moe.Add<std::uint8_t>(
        kPrefillMaxTokens * kTopK * kExpert / 16U);
    const auto p_expert_down = moe.Add<std::uint16_t>(
        kPrefillMaxTokens * kTopK * kWidth);
    const auto p_shared_product =
        moe.Add<float>(kPrefillMaxTokens * kShared);
    const auto p_shared_product_packed =
        moe.Add<std::uint8_t>(kPrefillMaxTokens * kShared / 2U);
    const auto p_shared_product_scales =
        moe.Add<std::uint8_t>(kPrefillMaxTokens * kShared / 16U);
    const auto p_shared_output =
        moe.Add<float>(kPrefillMaxTokens * kWidth);
    const auto p_reduced_output =
        moe.Add<float>(kPrefillMaxTokens * kWidth);
    const auto p_assignments = moe.Add<Gemma4MoePrefillAssignment>(
        kPrefillMaxTokens * kTopK);
    const auto p_histogram = moe.Add<std::uint32_t>(kExperts);
    const auto p_prefix = moe.Add<std::uint32_t>(kExperts + 1U);
    const auto p_permutation =
        moe.Add<std::uint32_t>(kPrefillMaxTokens * kTopK);
    const auto p_inverse =
        moe.Add<std::uint32_t>(kPrefillMaxTokens * kTopK);
    prefill.bytes = std::max(attention.bytes, moe.bytes);
    constexpr std::uint64_t kM09MoePrefillCap = 192U * 1024U * 1024U;
    if (prefill.bytes == std::numeric_limits<std::uint64_t>::max() ||
        prefill.bytes > kM09MoePrefillCap) {
      return Invalid("M17 fixed prefill workspace exceeds the M09 cap");
    }
    valid = impl->prefill_workspace.Allocate(
        prefill.bytes, "allocate M17 fixed prefill workspace");
    if (!valid.ok()) return valid;
    auto pptr = [&](std::uint64_t offset) {
      return impl->prefill_workspace.As<std::byte>(offset);
    };
    impl->prefill_tokens = reinterpret_cast<std::uint32_t*>(pptr(p_tokens));
    error = cudaMallocHost(&impl->prefill_host_tokens,
                           kPrefillMaxTokens * sizeof(std::uint32_t));
    if (error != cudaSuccess) {
      return CudaFailure("allocate M17 pinned prefill tokens", error);
    }
    impl->prefill_hidden_a = reinterpret_cast<float*>(pptr(p_hidden_a));
    impl->prefill_hidden_b = reinterpret_cast<float*>(pptr(p_hidden_b));
    impl->prefill_attention_workspace = {
        reinterpret_cast<std::uint8_t*>(pptr(p_input_fp8)),
        reinterpret_cast<float*>(pptr(p_input_scale)),
        reinterpret_cast<float*>(pptr(p_q_raw)),
        reinterpret_cast<float*>(pptr(p_k_raw)),
        reinterpret_cast<float*>(pptr(p_v_raw)),
        reinterpret_cast<float*>(pptr(p_q_norm)),
        reinterpret_cast<float*>(pptr(p_k_norm)),
        reinterpret_cast<float*>(pptr(p_v_norm)),
        reinterpret_cast<float*>(pptr(p_global_cosine)),
        reinterpret_cast<float*>(pptr(p_global_sine)),
        reinterpret_cast<std::uint8_t*>(pptr(p_staged_k)),
        reinterpret_cast<std::uint8_t*>(pptr(p_staged_v)),
        reinterpret_cast<float*>(pptr(p_scores)), kPrefillScoreElements,
        reinterpret_cast<float*>(pptr(p_attention)),
        reinterpret_cast<std::uint8_t*>(pptr(p_output_fp8)),
        reinterpret_cast<float*>(pptr(p_output_scale)),
        reinterpret_cast<float*>(pptr(p_output_projection)),
        nullptr,
        static_cast<void*>(pptr(p_cutlass_workspace)),
        kGemma4Moe26BAttentionCutlassWorkspaceBytes};
    impl->prefill_attention_workspace.global_key_bf16 =
        reinterpret_cast<std::uint16_t*>(pptr(p_global_key_bf16));
    impl->prefill_attention_workspace.global_value_bf16 =
        reinterpret_cast<std::uint16_t*>(pptr(p_global_value_bf16));
    impl->prefill_attention_workspace.global_bf16_capacity =
        kPreparedGlobalPrefillTokens;
    impl->prefill_local_rotary_cosine =
        reinterpret_cast<float*>(pptr(p_local_cosine));
    impl->prefill_local_rotary_sine =
        reinterpret_cast<float*>(pptr(p_local_sine));
    impl->prefill_moe_workspace = {
        reinterpret_cast<float*>(pptr(p_router_logits)),
        reinterpret_cast<float*>(pptr(p_router_probabilities)),
        reinterpret_cast<float*>(pptr(p_token_hidden)),
        reinterpret_cast<std::uint8_t*>(pptr(p_token_packed)),
        reinterpret_cast<std::uint8_t*>(pptr(p_token_scales)),
        nullptr,
        reinterpret_cast<std::uint8_t*>(pptr(p_expert_product_packed)),
        reinterpret_cast<std::uint8_t*>(pptr(p_expert_product_scales)),
        nullptr,
        reinterpret_cast<float*>(pptr(p_shared_product)),
        reinterpret_cast<std::uint8_t*>(pptr(p_shared_product_packed)),
        reinterpret_cast<std::uint8_t*>(pptr(p_shared_product_scales)),
        reinterpret_cast<float*>(pptr(p_shared_output)),
        reinterpret_cast<float*>(pptr(p_reduced_output)),
        reinterpret_cast<Gemma4MoePrefillAssignment*>(pptr(p_assignments)),
        reinterpret_cast<std::uint32_t*>(pptr(p_histogram)),
        reinterpret_cast<std::uint32_t*>(pptr(p_prefix)),
        reinterpret_cast<std::uint32_t*>(pptr(p_permutation)),
        reinterpret_cast<std::uint32_t*>(pptr(p_inverse)),
        impl->routing_finite};
    impl->prefill_moe_workspace.expert_product_bf16 =
        reinterpret_cast<std::uint16_t*>(pptr(p_expert_product));
    impl->prefill_moe_workspace.expert_down_bf16 =
        reinterpret_cast<std::uint16_t*>(pptr(p_expert_down));
  }

  Gemma4Moe26BReferenceEngine engine(std::move(impl));
  valid = engine.Reset();
  if (!valid.ok()) return valid;
  if (backend == Gemma4Moe26BBackend::kSm120Integrated) {
    valid = engine.implementation_->PrepareDecodeGraph();
    if (!valid.ok()) return valid;
  }
  std::size_t free_bytes = 0U;
  std::size_t total_bytes = 0U;
  error = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (error != cudaSuccess) {
    return CudaFailure("measure M21 initialized memory margin", error);
  }
  const std::uint64_t required_margin =
      context_tokens >= 65536U ? kLongContextMargin : kPrimaryContextMargin;
  if (free_bytes < required_margin) {
    const std::uint64_t required_slot =
        engine.weight_arena_bytes() + engine.kv_cache_bytes() +
        engine.workspace_bytes();
    const std::uint64_t shortfall =
        required_margin - static_cast<std::uint64_t>(free_bytes);
    return Status(
        StatusCode::kResourceExhausted,
        "cannot create Gemma 4 26B execution slot: context=" +
            std::to_string(context_tokens) +
            " free=" + std::to_string(free_bytes) +
            " required_slot=" + std::to_string(required_slot) +
            " probe_resident=" +
            std::to_string(engine.weight_arena_bytes()) +
            " required_margin=" + std::to_string(required_margin) +
            " shortfall=" + std::to_string(shortfall));
  }
  return engine;
}

Status Gemma4Moe26BReferenceEngine::Reset() {
  if (!implementation_) return Invalid("M13 engine is not initialized");
  cudaError_t error = cudaMemsetAsync(implementation_->kv.As<std::byte>(), 0,
                                      implementation_->kv.bytes(),
                                      implementation_->stream);
  if (error != cudaSuccess) return CudaFailure("clear M13 K/V arena", error);
  error = cudaMemsetAsync(implementation_->workspace.As<std::byte>(), 0,
                          implementation_->workspace.bytes(),
                          implementation_->stream);
  if (error != cudaSuccess) return CudaFailure("clear M13 workspace", error);
  if (implementation_->prefill_workspace.bytes() != 0U) {
    error = cudaMemsetAsync(
        implementation_->prefill_workspace.As<std::byte>(), 0,
        implementation_->prefill_workspace.bytes(), implementation_->stream);
    if (error != cudaSuccess) {
      return CudaFailure("clear M17 prefill workspace", error);
    }
  }
  if (implementation_->mtp_verifier.bytes() != 0U) {
    error = cudaMemsetAsync(implementation_->mtp_verifier.As<std::byte>(), 0,
                            implementation_->mtp_verifier.bytes(),
                            implementation_->stream);
    if (error != cudaSuccess) {
      return CudaFailure("clear M25 verifier workspace", error);
    }
  }
  error = cudaStreamSynchronize(implementation_->stream);
  if (error != cudaSuccess) return CudaFailure("synchronize M13 reset", error);
  implementation_->position = 0U;
  implementation_->prefill_chunks = 0U;
  implementation_->minimum_prefill_chunk = 0U;
  implementation_->prefill_calls = 0U;
  implementation_->decode_graph_launches = 0U;
  implementation_->token_selections = 0U;
  implementation_->sliding_ring_wraps = 0U;
  implementation_->maximum_global_position_exclusive = 0U;
  implementation_->fallback_count = 0U;
  implementation_->recurring_allocation_count = 0U;
  implementation_->mtp_group_graph_launches = 0U;
  implementation_->mtp_chain_graph_launches.fill(0U);
  implementation_->last_mtp_verification_min_margin = 0.0F;
  implementation_->sampling_step = 0U;
  // Graph executables remain instantiated across Reset(), so their observed
  // cudaMemGetInfo deltas and captured stop-token count intentionally remain
  // part of the configured runtime. ConversationSession recaptures the graphs
  // when its sampling/stop-token configuration changes.
  implementation_->pending_decode_self_feed = false;
  implementation_->decode_self_feed_valid = false;
  implementation_->has_last_input_token = false;
  implementation_->last_input_token = 0U;
  implementation_->decode_self_feed_token = 0U;
  implementation_->decode_self_feed_position = 0U;
  return Status::Ok();
}

Status Gemma4Moe26BReferenceEngine::ForwardToken(std::uint32_t token) {
  if (!implementation_) return Invalid("M13 engine is not initialized");
  auto& x = *implementation_;
  if (token >= kVocabulary || x.position >= x.context) {
    return Invalid("M13 token or position exceeds the fixed contract");
  }
  if (x.sampling.enabled && x.sampling.repetition_penalty != 1.0F) {
    Status marked = LaunchMarkRepetitionToken(
        token, x.repetition_mask, x.stream);
    if (!marked.ok()) return marked;
  }
  if (x.backend == Gemma4Moe26BBackend::kSm120Integrated) {
    if (x.decode_graph == nullptr) {
      return Invalid("M17 decode graph is not initialized");
    }
    const bool use_device_self_feed =
        x.decode_self_feed_valid && x.decode_self_feed_token == token &&
        x.decode_self_feed_position == x.position;
    x.decode_self_feed_valid = false;
    cudaError_t graph_error = cudaSuccess;
    if (!use_device_self_feed) {
      SetDecodeControlKernel<<<1, 1, 0, x.stream>>>(x.decode_control, token,
                                                    x.position);
      graph_error = cudaGetLastError();
    }
    if (graph_error == cudaSuccess) {
      graph_error = cudaGraphLaunch(x.decode_graph, x.stream);
    }
    if (graph_error != cudaSuccess) {
      return CudaFailure("launch M17 decode graph", graph_error);
    }
    const std::uint64_t previous_position = x.position;
    ++x.position;
    ++x.decode_graph_launches;
    x.pending_decode_self_feed = true;
    if (x.sliding_capacity != 0U && previous_position != 0U &&
        previous_position % x.sliding_capacity == 0U) {
      ++x.sliding_ring_wraps;
    }
    x.maximum_global_position_exclusive =
        std::max(x.maximum_global_position_exclusive, x.position);
    x.has_last_input_token = true;
    x.last_input_token = token;
    return Status::Ok();
  }
  constexpr unsigned threads = 256U;
  cudaError_t error = cudaMemsetAsync(x.routing_finite, 1, sizeof(int),
                                      x.stream);
  if (error != cudaSuccess) {
    return CudaFailure("initialize M13 router finite flag", error);
  }
  TiledEmbeddingLookupKernel<<<static_cast<unsigned>((kWidth + threads - 1U) /
                                                       threads),
                               threads, 0, x.stream>>>(
      x.head.packed_e2m1, x.head.scales_e4m3fn,
      x.head.weight_global_divisor, token, x.hidden_a);
  error = cudaGetLastError();
  if (error != cudaSuccess) return CudaFailure("launch M13 embedding", error);

  for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
    Status status = LaunchGemma4Moe26BAttentionReferenceLayer(
        x.hidden_a, x.hidden_b, x.position, x.traits[layer],
        x.attention_weights[layer], x.caches[layer], x.attention_workspace,
        1.0e-6F, x.stream);
    if (!status.ok()) return status;
    status = x.backend != Gemma4Moe26BBackend::kReference
                 ? LaunchGemma4MoeSm120Layer(
                       x.hidden_b, x.hidden_a, x.moe_config,
                       x.moe_weights[layer], x.moe_workspace, x.stream)
                 : LaunchGemma4MoeReferenceLayer(
                       x.hidden_b, x.hidden_a, x.moe_config,
                       x.moe_weights[layer], x.moe_workspace, x.stream);
    if (!status.ok()) return status;
    const int capture = CaptureIndex(layer);
    if (capture >= 0) {
      error = cudaMemcpyAsync(x.layer_captures[static_cast<std::size_t>(capture)],
                              x.hidden_a, kWidth * sizeof(float),
                              cudaMemcpyDeviceToDevice, x.stream);
      if (error == cudaSuccess) {
        error = cudaMemcpyAsync(
            x.router_probability_captures[static_cast<std::size_t>(capture)],
            x.moe_workspace.router_probabilities, kExperts * sizeof(float),
            cudaMemcpyDeviceToDevice, x.stream);
      }
      if (error == cudaSuccess) {
        error = cudaMemcpyAsync(
            x.router_id_captures[static_cast<std::size_t>(capture)],
            x.moe_workspace.top_ids, kTopK * sizeof(std::uint32_t),
            cudaMemcpyDeviceToDevice, x.stream);
      }
      if (error != cudaSuccess) return CudaFailure("capture M13 layer", error);
    }
  }
  Status status = LaunchRmsNormBf16(x.hidden_a, x.final_norm, x.final_hidden,
                                    1U, kWidth, 1.0e-6F, x.stream);
  if (!status.ok()) return status;
  status = LaunchNvfp4ReferenceActivationQuantization(
      x.final_hidden, x.head_activation, x.head_activation_scales, kWidth,
      x.head.activation_global_divisor, x.stream);
  if (!status.ok()) return status;
  status = x.backend != Gemma4Moe26BBackend::kReference
               ? LaunchNvfp4Sm120DirectProjectionBf16Float(
                     x.head_activation, x.head_activation_scales,
                     x.head.packed_e2m1, x.head.scales_e4m3fn, x.logits,
                     x.head.rows, x.head.columns,
                     x.head.activation_global_divisor,
                     x.head.weight_global_divisor, x.stream)
               : LaunchGemma4MoeTiledNvfp4ReferenceProjection(
                     x.head, x.head_activation, x.head_activation_scales,
                     x.logits, x.stream);
  if (!status.ok()) return status;
  status = LaunchSoftcapArgmax(
      x.logits, x.softcap, x.finite, x.output_candidates,
      x.prediction_token, x.prediction_logit, x.stream);
  if (!status.ok()) return status;
  const std::uint64_t previous_position = x.position;
  ++x.position;
  if (x.sliding_capacity != 0U && previous_position != 0U &&
      previous_position % x.sliding_capacity == 0U) {
    ++x.sliding_ring_wraps;
  }
  x.maximum_global_position_exclusive =
      std::max(x.maximum_global_position_exclusive, x.position);
  x.has_last_input_token = true;
  x.last_input_token = token;
  return Status::Ok();
}

Status Gemma4Moe26BReferenceEngine::PrefillTokens(
    std::span<const std::uint32_t> tokens) {
  if (!implementation_ || tokens.empty() ||
      implementation_->backend != Gemma4Moe26BBackend::kSm120Integrated ||
      implementation_->prefill_workspace.bytes() == 0U ||
      tokens.size() > implementation_->context - implementation_->position) {
    return Invalid("M17 prefill request exceeds the initialized contract");
  }
  for (const std::uint32_t token : tokens) {
    if (token >= kVocabulary) return Invalid("M17 prefill token is invalid");
  }
  auto& x = *implementation_;
  x.pending_decode_self_feed = false;
  x.decode_self_feed_valid = false;
  constexpr unsigned threads = 256U;
  std::size_t consumed = 0U;
  cudaError_t error =
      cudaMemsetAsync(x.routing_finite, 1, sizeof(int), x.stream);
  if (error != cudaSuccess) {
    return CudaFailure("initialize M17 prefill router finite flag", error);
  }
  const Gemma4Moe26BAttentionLayerTraits* local_trait = nullptr;
  const Gemma4Moe26BAttentionLayerTraits* global_trait = nullptr;
  for (const auto& trait : x.traits) {
    const bool sliding =
        trait.attention == Gemma4Moe26BAttentionType::kSliding;
    const Gemma4Moe26BAttentionLayerTraits*& profile =
        sliding ? local_trait : global_trait;
    if (profile == nullptr) {
      profile = &trait;
    } else if (profile->head_dimension != trait.head_dimension ||
               profile->rotary_factor != trait.rotary_factor ||
               profile->rope_theta != trait.rope_theta ||
               profile->rope_scaling_factor != trait.rope_scaling_factor) {
      return Invalid("M17 prefill attention classes do not share RoPE profiles");
    }
  }
  if (local_trait == nullptr || global_trait == nullptr ||
      x.prefill_local_rotary_cosine == nullptr ||
      x.prefill_local_rotary_sine == nullptr) {
    return Invalid("M17 prefill RoPE profiles are not initialized");
  }
  while (consumed < tokens.size()) {
    const std::uint64_t chunk = std::min<std::uint64_t>(
        kPrefillMaxTokens, tokens.size() - consumed);
    const bool is_last_chunk =
        consumed + static_cast<std::size_t>(chunk) == tokens.size();
    error = cudaStreamSynchronize(x.stream);
    if (error != cudaSuccess) {
      return CudaFailure("synchronize M17 prefill token staging", error);
    }
    std::copy_n(tokens.data() + consumed, static_cast<std::size_t>(chunk),
                x.prefill_host_tokens);
    error = cudaMemcpyAsync(
        x.prefill_tokens, x.prefill_host_tokens,
        chunk * sizeof(std::uint32_t), cudaMemcpyHostToDevice, x.stream);
    if (error == cudaSuccess && x.sampling.enabled &&
        x.sampling.repetition_penalty != 1.0F) {
      Status marked = LaunchMarkRepetitionTokens(
          x.prefill_tokens, chunk, x.repetition_mask, x.stream);
      if (!marked.ok()) return marked;
    }
    if (error != cudaSuccess) return CudaFailure("copy M17 prefill tokens", error);
    const std::uint64_t hidden_elements = chunk * kWidth;
    TiledEmbeddingLookupBatchKernel<<<
        static_cast<unsigned>((hidden_elements + threads - 1U) / threads),
        threads, 0, x.stream>>>(
        x.head.packed_e2m1, x.head.scales_e4m3fn,
        x.head.weight_global_divisor, x.prefill_tokens, x.prefill_hidden_a,
        hidden_elements);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
      return CudaFailure("launch M17 prefill embedding", error);
    }
    auto prepare_rotary = [&](const Gemma4Moe26BAttentionLayerTraits& trait,
                              float* cosine, float* sine) {
      const std::uint64_t rotating_pairs = static_cast<std::uint64_t>(
          trait.rotary_factor *
          static_cast<double>(trait.head_dimension / 2U));
      return LaunchRotaryEmbeddingTableBatch(
          cosine, sine, chunk, rotating_pairs, trait.head_dimension,
          x.position, trait.rope_theta, trait.rope_scaling_factor, x.stream);
    };
    Status status = prepare_rotary(
        *global_trait, x.prefill_attention_workspace.rotary_cosine,
        x.prefill_attention_workspace.rotary_sine);
    if (!status.ok()) return status;
    status = prepare_rotary(*local_trait, x.prefill_local_rotary_cosine,
                            x.prefill_local_rotary_sine);
    if (!status.ok()) return status;
    for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
      auto layer_workspace = x.prefill_attention_workspace;
      if (x.traits[layer].attention ==
          Gemma4Moe26BAttentionType::kSliding) {
        layer_workspace.rotary_cosine = x.prefill_local_rotary_cosine;
        layer_workspace.rotary_sine = x.prefill_local_rotary_sine;
      }
      status = LaunchGemma4Moe26BAttentionSm120PrefillLayer(
          x.prefill_hidden_a, x.prefill_hidden_b, x.position, chunk,
          x.traits[layer], x.attention_weights[layer], x.caches[layer],
          layer_workspace, 1.0e-6F, x.stream, true);
      if (!status.ok()) return status;
      status = LaunchGemma4MoeSm120PrefillLayer(
          x.prefill_hidden_b, x.prefill_hidden_a, chunk, x.moe_config,
          x.moe_weights[layer], x.prefill_moe_workspace, x.stream);
      if (!status.ok()) return status;
      const int capture = CaptureIndex(layer);
      if (is_last_chunk && capture >= 0) {
        const std::size_t capture_index = static_cast<std::size_t>(capture);
        error = cudaMemcpyAsync(
            x.layer_captures[capture_index],
            x.prefill_hidden_a + (chunk - 1U) * kWidth,
            kWidth * sizeof(float), cudaMemcpyDeviceToDevice, x.stream);
        if (error == cudaSuccess) {
          error = cudaMemcpyAsync(
              x.router_probability_captures[capture_index],
              x.prefill_moe_workspace.router_probabilities +
                  (chunk - 1U) * kExperts,
              kExperts * sizeof(float), cudaMemcpyDeviceToDevice, x.stream);
        }
        if (error == cudaSuccess) {
          CapturePrefillRouterIdsKernel<<<1, kTopK, 0, x.stream>>>(
              x.prefill_moe_workspace.assignments,
              x.router_id_captures[capture_index], chunk - 1U);
          error = cudaGetLastError();
        }
        if (error != cudaSuccess) {
          return CudaFailure("capture M17 prefill layer", error);
        }
      }
    }
    if (is_last_chunk) {
      float* last_hidden = x.prefill_hidden_a + (chunk - 1U) * kWidth;
      Status status = LaunchRmsNormBf16(
          last_hidden, x.final_norm, x.final_hidden, 1U, kWidth, 1.0e-6F,
          x.stream);
      if (!status.ok()) return status;
      status = LaunchNvfp4ReferenceActivationQuantization(
          x.final_hidden, x.head_activation, x.head_activation_scales, kWidth,
          x.head.activation_global_divisor, x.stream);
      if (!status.ok()) return status;
      status = LaunchNvfp4Sm120DirectProjectionBf16Float(
          x.head_activation, x.head_activation_scales, x.head.packed_e2m1,
          x.head.scales_e4m3fn, x.logits, x.head.rows, x.head.columns,
          x.head.activation_global_divisor, x.head.weight_global_divisor,
          x.stream);
      if (!status.ok()) return status;
      status = LaunchSoftcapArgmax(
          x.logits, x.softcap, x.finite, x.output_candidates,
          x.prediction_token, x.prediction_logit, x.stream);
      if (!status.ok()) return status;
    }
    const std::uint64_t previous_position = x.position;
    x.position += chunk;
    if (x.sliding_capacity != 0U) {
      x.sliding_ring_wraps += x.position / x.sliding_capacity -
                                previous_position / x.sliding_capacity;
    }
    x.maximum_global_position_exclusive =
        std::max(x.maximum_global_position_exclusive, x.position);
    ++x.prefill_chunks;
    x.minimum_prefill_chunk =
        x.minimum_prefill_chunk == 0U
            ? chunk
            : std::min(x.minimum_prefill_chunk, chunk);
    consumed += static_cast<std::size_t>(chunk);
  }
  ++x.prefill_calls;
  x.has_last_input_token = true;
  x.last_input_token = tokens.back();
  return Status::Ok();
}

Result<Gemma4Moe26BReferencePrediction>
Gemma4Moe26BReferenceEngine::Prediction() {
  if (!implementation_) return Invalid("M13 engine is not initialized");
  auto& x = *implementation_;
  cudaError_t error = cudaMemcpyAsync(
      x.prediction_host_status, x.prediction_device_status,
      sizeof(PredictionStatus), cudaMemcpyDeviceToHost, x.stream);
  if (error == cudaSuccess) error = cudaStreamSynchronize(x.stream);
  if (error != cudaSuccess) {
    return CudaFailure("copy compact M20 prediction status", error);
  }
  const PredictionStatus status = *x.prediction_host_status;
  Gemma4Moe26BReferencePrediction result{status.token, status.logit,
                                         status.finite != 0};
  if (status.routing_finite == 0) {
    return Status(StatusCode::kInternal,
                  "Gemma 4 26B router produced a non-finite value");
  }
  if (!result.all_logits_finite) {
    return Status(StatusCode::kInternal,
                  "Gemma 4 26B output logits are non-finite");
  }
  if (x.pending_decode_self_feed) {
    x.pending_decode_self_feed = false;
    x.decode_self_feed_valid = true;
    x.decode_self_feed_token = status.token;
    x.decode_self_feed_position = x.position;
  }
  return result;
}

Status Gemma4Moe26BReferenceEngine::ConfigureTokenSelection(
    const SamplingOptions& options,
    std::span<const std::uint32_t> suppressed_token_ids) {
  if (!implementation_) return Invalid("M22 engine is not initialized");
  Status valid = ValidateSamplingOptions(
      options, static_cast<std::uint32_t>(kVocabulary));
  if (!valid.ok()) return valid;
  if (suppressed_token_ids.size() > kMaximumSuppressedTokens) {
    return Invalid("Gemma 4 26B supports at most 16 suppressed token IDs");
  }
  for (const std::uint32_t token : suppressed_token_ids) {
    if (token >= kVocabulary) {
      return Invalid("Gemma 4 26B suppressed token ID exceeds vocabulary");
    }
  }
  auto& x = *implementation_;
  if (x.position != 0U || x.prefill_calls != 0U ||
      x.decode_graph_launches != 0U) {
    return Invalid("M22 token selection must be configured before execution");
  }
  // Sampling options are captured by value in the fixed MTP graphs. A reused
  // resident runtime may start a new session with different sampling controls,
  // so discard only the MTP executables and recapture them after configuration.
  for (cudaGraphExec_t& graph : x.mtp_group_graphs) {
    if (graph != nullptr) (void)cudaGraphExecDestroy(graph);
    graph = nullptr;
  }
  for (cudaGraphExec_t& graph : x.mtp_chain_graphs) {
    if (graph != nullptr) (void)cudaGraphExecDestroy(graph);
    graph = nullptr;
  }
  x.mtp_group_graph_device_bytes = 0U;
  x.mtp_chain_graph_device_bytes.fill(0U);
  cudaError_t error = cudaMemsetAsync(
      x.repetition_mask, 0,
      static_cast<std::size_t>(kRepetitionMaskWords) * sizeof(std::uint32_t),
      x.stream);
  if (error == cudaSuccess && !suppressed_token_ids.empty()) {
    error = cudaMemcpyAsync(
        x.suppressed_token_ids, suppressed_token_ids.data(),
        suppressed_token_ids.size_bytes(), cudaMemcpyHostToDevice, x.stream);
  }
  if (error == cudaSuccess) error = cudaStreamSynchronize(x.stream);
  if (error != cudaSuccess) {
    return CudaFailure("configure M22 token selection", error);
  }
  x.sampling = options;
  x.sampling_step = 0U;
  x.suppressed_token_count =
      static_cast<std::uint32_t>(suppressed_token_ids.size());
  return Status::Ok();
}

Status Gemma4Moe26BReferenceEngine::ConfigurePrefillRouter(
    Gemma4MoePrefillRouter router) {
  if (!implementation_) return Invalid("M20 engine is not initialized");
  auto& x = *implementation_;
  if (x.position != 0U || x.prefill_calls != 0U ||
      x.decode_graph_launches != 0U) {
    return Invalid("M20 prefill router must be selected before execution");
  }
  if (router != Gemma4MoePrefillRouter::kSerialExact &&
      router != Gemma4MoePrefillRouter::kSm120TensorCore) {
    return Invalid("M20 prefill router selection is invalid");
  }
  if (router == Gemma4MoePrefillRouter::kSm120TensorCore &&
      x.backend != Gemma4Moe26BBackend::kSm120Integrated) {
    return Status(StatusCode::kUnsupported,
                  "SM120 tensor prefill router requires the integrated backend");
  }
  x.moe_config.prefill_router = router;
  return Status::Ok();
}

Status Gemma4Moe26BReferenceEngine::ConfigureMtpVerifierBackend(
    Gemma4Moe26BMtpVerifierBackend backend) {
  if (!implementation_) return Invalid("M25 engine is not initialized");
  auto& x = *implementation_;
  if (!x.assistant.prepared() || x.position != 0U || x.prefill_calls != 0U ||
      x.decode_graph_launches != 0U) {
    return Invalid("M25 verifier backend must be selected before execution");
  }
  x.mtp_verifier_backend = backend;
  if (backend != Gemma4Moe26BMtpVerifierBackend::kExactSharedBatchedMoe) {
    return Status::Ok();
  }
  for (const std::uint32_t draft_count : {1U, 2U, 4U}) {
    Status status = x.PrepareFixedMtpGraph(draft_count);
    if (!status.ok()) return status;
    status = x.PrepareFixedMtpChainGraph(draft_count);
    if (!status.ok()) return status;
  }
  return Status::Ok();
}

Result<std::uint32_t> Gemma4Moe26BReferenceEngine::SelectToken() {
  if (!implementation_) return Invalid("M22 engine is not initialized");
  auto prediction = Prediction();
  if (!prediction.ok()) return prediction.status();
  auto& x = *implementation_;
  ++x.token_selections;
  if (!x.sampling.enabled && x.suppressed_token_count == 0U) {
    return prediction.value().token;
  }
  Status selected;
  if (x.sampling.enabled) {
    selected = LaunchSampleToken(
        x.logits, x.sampling_logits, x.sampling_cumulative,
        x.sampling_token_ids, x.sampling_sorted_token_ids,
        x.repetition_mask, x.suppressed_token_ids,
        x.suppressed_token_count, static_cast<std::uint32_t>(kVocabulary),
        x.sampling, x.sampling_step++, nullptr, x.selected_token,
        x.sampling_algorithm_workspace,
        x.sampling_algorithm_workspace_bytes, x.stream);
  } else {
    selected = LaunchLogitArgmax(
        x.logits, x.suppressed_token_ids, x.suppressed_token_count,
        x.selected_token, x.stream);
  }
  if (!selected.ok()) return selected;
  std::uint32_t token = 0U;
  cudaError_t error = cudaMemcpyAsync(
      &token, x.selected_token, sizeof(token), cudaMemcpyDeviceToHost,
      x.stream);
  if (error == cudaSuccess) error = cudaStreamSynchronize(x.stream);
  if (error != cudaSuccess) {
    return CudaFailure("copy M22 selected token", error);
  }
  return token;
}

Status Gemma4Moe26BReferenceEngine::LoadMtpAssistant(
    const std::filesystem::path& assistant_directory) {
  if (!implementation_ || assistant_directory.empty()) {
    return Invalid("M25 Assistant load request is invalid");
  }
  auto& x = *implementation_;
  if (x.backend != Gemma4Moe26BBackend::kSm120Integrated ||
      x.assistant.loaded()) {
    return Invalid(
        "M25 Assistant requires one unconfigured SM120 integrated engine");
  }
  Gemma4Moe26BAssistantModel candidate;
  Status status = candidate.Load(assistant_directory);
  if (!status.ok()) return status;
  status = candidate.Prepare(x.context);
  if (!status.ok()) return status;

  // The verifier is a separately named fixed region. It holds five Target
  // output rows plus both the overwritten-cache backup and compact
  // speculative K/V rows for every layer. No verifier storage is borrowed
  // from the Assistant, and no allocation occurs after this load boundary.
  LayoutBuilder layout;
  const std::uint64_t normalized =
      layout.Add<float>(kM25MaximumVerifyTokens * kWidth);
  const std::uint64_t head_activation = layout.Add<std::uint8_t>(
      kM25MaximumVerifyTokens * kWidth / 2U);
  const std::uint64_t head_scales = layout.Add<std::uint8_t>(
      kM25MaximumVerifyTokens * kWidth / kNvfp4Block);
  const std::uint64_t verifier_logits =
      layout.Add<float>(kM25MaximumVerifyTokens * kVocabulary);
  const std::uint64_t selected =
      layout.Add<std::uint32_t>(kM25MaximumVerifyTokens);
  const std::uint64_t second_selected =
      layout.Add<std::uint32_t>(kM25MaximumVerifyTokens);
  const std::uint64_t top_values =
      layout.Add<float>(kM25MaximumVerifyTokens);
  const std::uint64_t second_values =
      layout.Add<float>(kM25MaximumVerifyTokens);
  const std::uint64_t finite = layout.Add<int>(kM25MaximumVerifyTokens);
  const std::uint64_t transaction = layout.Add<MtpGroupTransaction>(1U);
  const std::uint64_t row_controls =
      layout.Add<DecodeControl>(kM25MaximumVerifyTokens);
  const std::uint64_t sampling_repetition_masks =
      layout.Add<std::uint32_t>(kM25MaximumVerifyTokens *
                                kRepetitionMaskWords);
  const std::uint64_t stop_tokens =
      layout.Add<std::uint32_t>(kMaximumSuppressedTokens);
  const std::uint64_t chain_result = layout.Add<MtpChainResult>(1U);
  const std::uint64_t chain_outputs =
      layout.Add<std::uint32_t>(x.context);
  const std::uint64_t chain_proposals = layout.Add<std::uint32_t>(
      kMaximumMtpDraftTokens * x.context);
  std::array<MtpVerifierLayerOffsets, kLayers> layer_offsets{};
  for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
    const std::uint64_t elements = kM25MaximumVerifyTokens *
                                   x.traits[layer].kv_heads *
                                   x.traits[layer].head_dimension;
    layer_offsets[layer].backup_key = layout.Add<std::uint8_t>(elements);
    layer_offsets[layer].backup_value = layout.Add<std::uint8_t>(elements);
    layer_offsets[layer].compact_key = layout.Add<std::uint8_t>(elements);
    layer_offsets[layer].compact_value = layout.Add<std::uint8_t>(elements);
  }
  if (layout.bytes == 0U ||
      layout.bytes == std::numeric_limits<std::uint64_t>::max()) {
    return Invalid("M25 verifier workspace layout overflow");
  }
  DeviceBuffer verifier;
  status = verifier.Allocate(layout.bytes, "allocate M25 fixed verifier workspace");
  if (!status.ok()) return status;
  MtpVerifierHostResult* host_result = nullptr;
  auto host_error = cudaMallocHost(&host_result, sizeof(MtpVerifierHostResult));
  if (host_error != cudaSuccess) {
    return CudaFailure("allocate pinned M25 verifier result", host_error);
  }
  std::byte* chain_host = nullptr;
  const std::uint64_t chain_host_bytes =
      sizeof(MtpChainResult) + x.context * sizeof(std::uint32_t);
  host_error = cudaMallocHost(&chain_host,
                              static_cast<std::size_t>(chain_host_bytes));
  if (host_error != cudaSuccess) {
    (void)cudaFreeHost(host_result);
    return CudaFailure("allocate pinned M25 chain result", host_error);
  }
  std::size_t free_bytes = 0U;
  std::size_t total_bytes = 0U;
  const auto measured = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (measured != cudaSuccess) {
    (void)cudaFreeHost(host_result);
    (void)cudaFreeHost(chain_host);
    return CudaFailure("measure M25 post-Assistant margin", measured);
  }
  const std::uint64_t required_margin =
      x.context >= 65536U ? 200U * kMiB : 700U * kMiB;
  if (free_bytes < required_margin) {
    (void)cudaFreeHost(host_result);
    (void)cudaFreeHost(chain_host);
    return Status(
        StatusCode::kResourceExhausted,
        "M25 Assistant and verifier leave less than the profile margin");
  }
  x.assistant = std::move(candidate);
  x.mtp_verifier = std::move(verifier);
  x.mtp_verifier_host = host_result;
  x.mtp_chain_host = chain_host;
  x.mtp_normalized_offset = normalized;
  x.mtp_head_activation_offset = head_activation;
  x.mtp_head_scales_offset = head_scales;
  x.mtp_logits_offset = verifier_logits;
  x.mtp_selected_offset = selected;
  x.mtp_second_selected_offset = second_selected;
  x.mtp_top_values_offset = top_values;
  x.mtp_second_values_offset = second_values;
  x.mtp_finite_offset = finite;
  x.mtp_transaction_offset = transaction;
  x.mtp_row_controls_offset = row_controls;
  x.mtp_sampling_repetition_masks_offset = sampling_repetition_masks;
  x.mtp_stop_tokens_offset = stop_tokens;
  x.mtp_chain_result_offset = chain_result;
  x.mtp_chain_outputs_offset = chain_outputs;
  x.mtp_chain_proposals_offset = chain_proposals;
  x.mtp_layer_offsets = layer_offsets;
  return Status::Ok();
}

Status Gemma4Moe26BReferenceEngine::ConfigureMtpStopTokens(
    std::span<const std::uint32_t> stop_token_ids) {
  if (!implementation_) return Invalid("M25 engine is not initialized");
  auto& x = *implementation_;
  if (!x.assistant.prepared() || x.mtp_verifier.bytes() == 0U ||
      x.position != 0U || x.prefill_calls != 0U ||
      x.decode_graph_launches != 0U) {
    return Invalid("M25 stop tokens must be configured before execution");
  }
  if (stop_token_ids.size() > kMaximumSuppressedTokens) {
    return Invalid("Gemma 4 26B MTP supports at most 16 stop-token IDs");
  }
  for (const std::uint32_t token : stop_token_ids) {
    if (token >= kVocabulary) {
      return Invalid("Gemma 4 26B MTP stop-token ID exceeds vocabulary");
    }
  }
  const bool graphs_prepared = std::any_of(
      x.mtp_chain_graphs.begin(), x.mtp_chain_graphs.end(),
      [](cudaGraphExec_t graph) { return graph != nullptr; });
  if (graphs_prepared && stop_token_ids.size() != x.mtp_stop_token_count) {
    return Invalid("M25 stop-token count cannot change after graph capture");
  }
  cudaError_t error = cudaSuccess;
  if (!stop_token_ids.empty()) {
    error = cudaMemcpyAsync(
        x.mtp_verifier.As<std::uint32_t>(x.mtp_stop_tokens_offset),
        stop_token_ids.data(), stop_token_ids.size_bytes(),
        cudaMemcpyHostToDevice, x.stream);
  }
  if (error == cudaSuccess) error = cudaStreamSynchronize(x.stream);
  if (error != cudaSuccess) {
    return CudaFailure("configure M25 stop-token IDs", error);
  }
  x.mtp_stop_token_count =
      static_cast<std::uint32_t>(stop_token_ids.size());
  return Status::Ok();
}

Status Gemma4Moe26BReferenceEngine::GenerateMtpAssistantDrafts(
    std::span<std::uint32_t> draft_token_ids) {
  if (!implementation_ || draft_token_ids.empty() ||
      draft_token_ids.size() > 4U) {
    return Invalid("M25 Assistant draft request must contain one to four slots");
  }
  auto& x = *implementation_;
  if (!x.assistant.prepared() || !x.has_last_input_token || x.position == 0U) {
    return Invalid("M25 Assistant requires one completed target transaction");
  }
  constexpr std::uint32_t kSlidingLayer = 28U;
  constexpr std::uint32_t kFullLayer = 29U;
  const auto make_view = [&x](std::uint32_t layer) {
    AssistantSharedKvView view;
    const bool global =
        x.traits[layer].attention == Gemma4Moe26BAttentionType::kFull;
    view.mode = AssistantKvCacheMode::kCheckpointFp8;
    view.key_fp8 = x.caches[layer].key;
    view.value_fp8 = x.caches[layer].value;
    view.key_scale_bf16 = x.attention_weights[layer].key_cache_scale_bf16;
    view.value_scale_bf16 =
        x.attention_weights[layer].value_cache_scale_bf16;
    view.capacity = x.caches[layer].capacity;
    view.tokens = global ? x.position : std::min(x.position, view.capacity);
    view.first_slot = global || x.position <= view.capacity
                          ? 0U
                          : x.position % view.capacity;
    view.kv_heads = x.traits[layer].kv_heads;
    view.head_dimension = x.traits[layer].head_dimension;
    return view;
  };
  if (x.traits[kSlidingLayer].attention !=
          Gemma4Moe26BAttentionType::kSliding ||
      x.traits[kFullLayer].attention != Gemma4Moe26BAttentionType::kFull) {
    return Status(StatusCode::kInternal,
                  "M25 target shared-KV layer mapping is invalid");
  }
  Gemma4Moe26BAssistantProposalContext context;
  context.target_embedding = x.head;
  context.target_hidden = x.final_hidden;
  context.sliding_kv = make_view(kSlidingLayer);
  context.full_kv = make_view(kFullLayer);
  context.input_token = x.last_input_token;
  context.position = x.position - 1U;
  return x.assistant.GenerateDrafts(context, draft_token_ids, x.stream);
}

Status Gemma4Moe26BReferenceEngine::GenerateMtpAssistantDraftsForPending(
    std::uint32_t pending_token,
    std::span<std::uint32_t> draft_token_ids) {
  if (!implementation_ || draft_token_ids.empty() ||
      draft_token_ids.size() > 4U || pending_token >= kVocabulary) {
    return Invalid("M25 pending-token Assistant request is invalid");
  }
  auto& x = *implementation_;
  if (!x.assistant.prepared() || x.position == 0U) {
    return Invalid("M25 pending-token Assistant requires committed Target state");
  }
  constexpr std::uint32_t kSlidingLayer = 28U;
  constexpr std::uint32_t kFullLayer = 29U;
  const auto make_view = [&x](std::uint32_t layer) {
    AssistantSharedKvView view;
    const bool global =
        x.traits[layer].attention == Gemma4Moe26BAttentionType::kFull;
    view.mode = AssistantKvCacheMode::kCheckpointFp8;
    view.key_fp8 = x.caches[layer].key;
    view.value_fp8 = x.caches[layer].value;
    view.key_scale_bf16 = x.attention_weights[layer].key_cache_scale_bf16;
    view.value_scale_bf16 = x.attention_weights[layer].value_cache_scale_bf16;
    view.capacity = x.caches[layer].capacity;
    view.tokens = global ? x.position : std::min(x.position, view.capacity);
    view.first_slot = global || x.position <= view.capacity
                          ? 0U
                          : x.position % view.capacity;
    view.kv_heads = x.traits[layer].kv_heads;
    view.head_dimension = x.traits[layer].head_dimension;
    return view;
  };
  Gemma4Moe26BAssistantProposalContext context;
  context.target_embedding = x.head;
  context.target_hidden = x.final_hidden;
  context.sliding_kv = make_view(kSlidingLayer);
  context.full_kv = make_view(kFullLayer);
  context.input_token = pending_token;
  context.position = x.position - 1U;
  return x.assistant.GenerateDrafts(context, draft_token_ids, x.stream);
}

Status Gemma4Moe26BReferenceEngine::RunMtpAssistantGroup(
    std::uint32_t pending_token, std::uint32_t proposal_count,
    MtpGroupResult* host_result) {
  if (!implementation_ || host_result == nullptr ||
      (proposal_count != 1U && proposal_count != 2U && proposal_count != 4U)) {
    return Invalid("M25 Target-verification group request is invalid");
  }
  auto& x = *implementation_;
  const std::uint64_t tokens = proposal_count + 1U;
  if (!x.assistant.prepared() || x.mtp_verifier.bytes() == 0U ||
      x.mtp_verifier_host == nullptr || x.position == 0U ||
      pending_token >= kVocabulary || x.position >= x.context ||
      tokens > x.context - x.position) {
    return Invalid("M25 Target-verification group exceeds resident state");
  }
  if (x.mtp_verifier_backend ==
          Gemma4Moe26BMtpVerifierBackend::kExactSharedBatchedMoe &&
      (proposal_count == 1U || proposal_count == 2U ||
       proposal_count == 4U)) {
    if (x.mtp_group_graphs[proposal_count] == nullptr) {
      Status prepared = x.PrepareFixedMtpGraph(proposal_count);
      if (!prepared.ok()) return prepared;
    }
    return x.ExecuteFixedMtpGraphGroup(pending_token, proposal_count,
                                       host_result);
  }

  constexpr std::uint32_t kSlidingLayer = 28U;
  constexpr std::uint32_t kFullLayer = 29U;
  const auto make_view = [&x](std::uint32_t layer) {
    AssistantSharedKvView view;
    const bool global =
        x.traits[layer].attention == Gemma4Moe26BAttentionType::kFull;
    view.mode = AssistantKvCacheMode::kCheckpointFp8;
    view.key_fp8 = x.caches[layer].key;
    view.value_fp8 = x.caches[layer].value;
    view.key_scale_bf16 = x.attention_weights[layer].key_cache_scale_bf16;
    view.value_scale_bf16 = x.attention_weights[layer].value_cache_scale_bf16;
    view.capacity = x.caches[layer].capacity;
    view.tokens = global ? x.position : std::min(x.position, view.capacity);
    view.first_slot = global || x.position <= view.capacity
                          ? 0U
                          : x.position % view.capacity;
    view.kv_heads = x.traits[layer].kv_heads;
    view.head_dimension = x.traits[layer].head_dimension;
    return view;
  };
  if (x.traits[kSlidingLayer].attention !=
          Gemma4Moe26BAttentionType::kSliding ||
      x.traits[kFullLayer].attention != Gemma4Moe26BAttentionType::kFull) {
    return Status(StatusCode::kInternal,
                  "M25 target shared-KV layer mapping is invalid");
  }
  Gemma4Moe26BAssistantProposalContext proposal;
  proposal.target_embedding = x.head;
  proposal.target_hidden = x.final_hidden;
  proposal.sliding_kv = make_view(kSlidingLayer);
  proposal.full_kv = make_view(kFullLayer);
  proposal.input_token = pending_token;
  // This is the qualified 12B Assistant convention: the pending token is
  // combined with the hidden state/KV through the last committed Target row.
  proposal.position = x.position - 1U;
  Status status = x.assistant.GenerateDraftsDevice(
      proposal, proposal_count, x.stream);
  if (!status.ok()) return status;
  const std::uint32_t* drafts = x.assistant.device_draft_tokens();
  if (drafts == nullptr) {
    return Status(StatusCode::kInternal,
                  "M25 Assistant device proposals are unavailable");
  }

  auto* transaction = x.mtp_verifier.As<MtpGroupTransaction>(
      x.mtp_transaction_offset);
  *x.mtp_verifier_host = {};
  auto& control = x.mtp_verifier_host->transaction.control;
  control.current.input_token = pending_token;
  control.current.processed_position = x.position - 1U;
  control.current.remaining_output_capacity = x.context - x.position;
  control.current.output_write_position = 0U;
  control.current.sampling_step = x.sampling_step;
  control.next = control.current;
  control.fixed_draft_tokens = proposal_count;
  control.sampling_enabled = x.sampling.enabled ? 1U : 0U;
  cudaError_t error = cudaMemcpyAsync(
      transaction, &x.mtp_verifier_host->transaction,
      sizeof(MtpGroupTransaction), cudaMemcpyHostToDevice, x.stream);
  if (error != cudaSuccess) {
    return CudaFailure("initialize M25 verifier transaction", error);
  }

  status = LaunchBuildMtpVerificationInputs(
      pending_token, drafts, proposal_count, x.prefill_tokens, x.stream);
  if (!status.ok()) return status;
  error = cudaMemsetAsync(x.routing_finite, 1, sizeof(int), x.stream);
  if (error != cudaSuccess) {
    return CudaFailure("initialize M25 verifier router finite flag", error);
  }
  constexpr unsigned threads = 256U;
  const std::uint64_t hidden_elements = tokens * kWidth;
  TiledEmbeddingLookupBatchKernel<<<
      static_cast<unsigned>((hidden_elements + threads - 1U) / threads),
      threads, 0, x.stream>>>(
      x.head.packed_e2m1, x.head.scales_e4m3fn,
      x.head.weight_global_divisor, x.prefill_tokens, x.prefill_hidden_a,
      hidden_elements);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaFailure("launch M25 verifier embedding", error);
  }

  const std::uint64_t start_position = x.position;
  auto* row_controls =
      x.mtp_verifier.As<DecodeControl>(x.mtp_row_controls_offset);
  BuildMtpDecodeControlsKernel<<<1U, static_cast<unsigned>(tokens), 0,
                                 x.stream>>>(
      row_controls, start_position, x.sampling_step, tokens);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaFailure("build M25 exact verifier row controls", error);
  }
  const bool batch_attention =
      x.mtp_verifier_backend ==
          Gemma4Moe26BMtpVerifierBackend::kBatchedAttention ||
      x.mtp_verifier_backend ==
          Gemma4Moe26BMtpVerifierBackend::kFullyBatched;
  const bool batch_moe =
      x.mtp_verifier_backend ==
          Gemma4Moe26BMtpVerifierBackend::kBatchedMoe ||
      x.mtp_verifier_backend ==
          Gemma4Moe26BMtpVerifierBackend::kFullyBatched;
  const bool exact_shared_batch_moe =
      x.mtp_verifier_backend ==
      Gemma4Moe26BMtpVerifierBackend::kExactSharedBatchedMoe;
  for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
    const auto& trait = x.traits[layer];
    const std::uint64_t kv_elements =
        trait.kv_heads * trait.head_dimension;
    const auto offsets = x.mtp_layer_offsets[layer];
    auto* backup_key =
        x.mtp_verifier.As<std::uint8_t>(offsets.backup_key);
    auto* backup_value =
        x.mtp_verifier.As<std::uint8_t>(offsets.backup_value);
    auto* compact_key =
        x.mtp_verifier.As<std::uint8_t>(offsets.compact_key);
    auto* compact_value =
        x.mtp_verifier.As<std::uint8_t>(offsets.compact_value);
    status = LaunchCopyCircularMtpKvFp8(
        x.caches[layer].key, x.caches[layer].value, backup_key, backup_value,
        start_position, tokens, kv_elements, x.caches[layer].capacity, false,
        x.stream);
    if (!status.ok()) return status;
    if ((exact_shared_batch_moe || batch_attention) && tokens == 3U) {
      // The fixed-D2 verifier shares the qualified 12B three-row attention
      // machinery. MoE scratch is dead until the attention boundary and is
      // large enough for the split-online partials, so no M25 allocation is
      // introduced here.
      status = LaunchGemma4Moe26BAttentionSm120MtpD2Layer(
          x.prefill_hidden_a, x.prefill_hidden_b, start_position, trait,
          x.attention_weights[layer], x.caches[layer],
          x.prefill_attention_workspace,
          x.prefill_moe_workspace.shared_product, row_controls,
          batch_attention, exact_shared_batch_moe, false, 1.0e-6F,
          x.stream);
      if (!status.ok()) return status;
    } else if (batch_attention) {
      status = LaunchGemma4Moe26BAttentionSm120PrefillLayer(
          x.prefill_hidden_a, x.prefill_hidden_b, start_position, tokens,
          trait, x.attention_weights[layer], x.caches[layer],
          x.prefill_attention_workspace, 1.0e-6F, x.stream);
      if (!status.ok()) return status;
    } else {
      // Layer-major microbatching preserves the frozen ordinary-decode
      // arithmetic for every row. Later rows see earlier speculative K/V at
      // the same layer while all rows reuse one fixed decode workspace.
      for (std::uint64_t row = 0U; row < tokens; ++row) {
        status = LaunchGemma4Moe26BAttentionReferenceControlledLayer(
            x.prefill_hidden_a + row * kWidth,
            x.prefill_hidden_b + row * kWidth, trait,
            x.attention_weights[layer], x.caches[layer],
            x.attention_workspace, row_controls + row, 1.0e-6F, x.stream,
            false);
        if (!status.ok()) return status;
      }
    }
    if (exact_shared_batch_moe) {
      status = LaunchGemma4MoeSm120MtpSharedBatchLayer(
          x.prefill_hidden_b, x.prefill_hidden_a, tokens, x.moe_config,
          x.moe_weights[layer], x.prefill_moe_workspace, x.moe_workspace,
          x.stream);
      if (!status.ok()) return status;
    } else if (batch_moe) {
      status = LaunchGemma4MoeSm120PrefillLayer(
          x.prefill_hidden_b, x.prefill_hidden_a, tokens, x.moe_config,
          x.moe_weights[layer], x.prefill_moe_workspace, x.stream);
      if (!status.ok()) return status;
    } else {
      for (std::uint64_t row = 0U; row < tokens; ++row) {
      status = LaunchGemma4MoeSm120Layer(
          x.prefill_hidden_b + row * kWidth,
          x.prefill_hidden_a + row * kWidth, x.moe_config,
          x.moe_weights[layer], x.moe_workspace, x.stream,
          x.shared_moe_stream, x.shared_moe_fork, x.shared_moe_join);
      if (!status.ok()) return status;
      }
    }
    status = LaunchCopyCircularMtpKvFp8(
        x.caches[layer].key, x.caches[layer].value, compact_key, compact_value,
        start_position, tokens, kv_elements, x.caches[layer].capacity, false,
        x.stream);
    if (!status.ok()) return status;
    status = LaunchCopyCircularMtpKvFp8(
        x.caches[layer].key, x.caches[layer].value, backup_key, backup_value,
        start_position, tokens, kv_elements, x.caches[layer].capacity, true,
        x.stream);
    if (!status.ok()) return status;
  }

  float* normalized =
      x.mtp_verifier.As<float>(x.mtp_normalized_offset);
  auto* head_activation =
      x.mtp_verifier.As<std::uint8_t>(x.mtp_head_activation_offset);
  auto* head_scales =
      x.mtp_verifier.As<std::uint8_t>(x.mtp_head_scales_offset);
  float* batch_logits = x.mtp_verifier.As<float>(x.mtp_logits_offset);
  auto* selected =
      x.mtp_verifier.As<std::uint32_t>(x.mtp_selected_offset);
  auto* second_selected = x.mtp_verifier.As<std::uint32_t>(
      x.mtp_second_selected_offset);
  float* top_values =
      x.mtp_verifier.As<float>(x.mtp_top_values_offset);
  float* second_values =
      x.mtp_verifier.As<float>(x.mtp_second_values_offset);
  int* finite = x.mtp_verifier.As<int>(x.mtp_finite_offset);
  status = LaunchRmsNormBf16(x.prefill_hidden_a, x.final_norm, normalized,
                             tokens, kWidth, 1.0e-6F, x.stream);
  if (!status.ok()) return status;
  status = LaunchRmsNormNvfp4ActivationQuantizationBatch(
      x.prefill_hidden_a, x.final_norm, head_activation, head_scales, tokens,
      kWidth, 1.0e-6F, x.head.activation_global_divisor, x.stream);
  if (!status.ok()) return status;
  status = LaunchNvfp4Sm120DirectProjectionBf16FloatExactBatch(
      head_activation, head_scales, x.head.packed_e2m1,
      x.head.scales_e4m3fn, batch_logits, tokens, x.head.rows, x.head.columns,
      x.head.activation_global_divisor, x.head.weight_global_divisor,
      x.stream);
  if (!status.ok()) return status;
  for (std::uint64_t row = 0U; row < tokens; ++row) {
    status = LaunchSoftcapArgmax(
        batch_logits + row * kVocabulary, x.softcap, finite + row,
        x.output_candidates, selected + row, top_values + row, x.stream,
        nullptr, x.suppressed_token_ids, x.suppressed_token_count);
    if (!status.ok()) return status;
    status = LaunchSoftcapArgmax(
        batch_logits + row * kVocabulary, x.softcap, finite + row,
        x.output_candidates, second_selected + row, second_values + row,
        x.stream, nullptr, x.suppressed_token_ids, x.suppressed_token_count,
        selected + row, false);
    if (!status.ok()) return status;
  }
  status = x.SelectMtpVerificationTokens(
      batch_logits, x.prefill_tokens, row_controls,
      static_cast<std::uint32_t>(tokens), selected);
  if (!status.ok()) return status;
  status = LaunchAcceptMtpGroup(
      drafts, selected, proposal_count,
      x.mtp_verifier.As<std::uint32_t>(x.mtp_stop_tokens_offset),
      x.mtp_stop_token_count,
      &transaction->result, &transaction->control, x.stream);
  if (!status.ok()) return status;
  if (x.sampling.enabled && x.sampling.repetition_penalty != 1.0F) {
    status = LaunchCommitSpeculativeRepetitionMask(
        x.mtp_verifier.As<std::uint32_t>(
            x.mtp_sampling_repetition_masks_offset),
        kRepetitionMaskWords, &transaction->result.output_count,
        x.repetition_mask, x.stream);
    if (!status.ok()) return status;
  }
  for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
    const auto& trait = x.traits[layer];
    const auto offsets = x.mtp_layer_offsets[layer];
    status = LaunchCommitMtpKvFp8(
        x.mtp_verifier.As<std::uint8_t>(offsets.compact_key),
        x.mtp_verifier.As<std::uint8_t>(offsets.compact_value),
        x.caches[layer].key, x.caches[layer].value, start_position,
        trait.kv_heads * trait.head_dimension, x.caches[layer].capacity,
        &transaction->result, x.stream);
    if (!status.ok()) return status;
  }
  status = LaunchCommitMtpHidden(normalized, x.final_hidden, kWidth,
                                 &transaction->result, x.stream);
  if (!status.ok()) return status;
  CommitMtpPredictionKernel<true>
      <<<kArgmaxBlocks, kArgmaxThreads, 0, x.stream>>>(
          batch_logits, x.logits, &transaction->result, finite,
          x.routing_finite, x.prediction_device_status);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaFailure("commit M25 verifier prediction", error);
  }

  error = cudaMemcpyAsync(&x.mtp_verifier_host->transaction, transaction,
                          sizeof(MtpGroupTransaction),
                          cudaMemcpyDeviceToHost, x.stream);
  if (error == cudaSuccess) {
    error = cudaMemcpyAsync(x.mtp_verifier_host->logits_finite.data(), finite,
                            tokens * sizeof(int), cudaMemcpyDeviceToHost,
                            x.stream);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpyAsync(x.mtp_verifier_host->top_values.data(), top_values,
                            tokens * sizeof(float), cudaMemcpyDeviceToHost,
                            x.stream);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpyAsync(x.mtp_verifier_host->second_values.data(),
                            second_values, tokens * sizeof(float),
                            cudaMemcpyDeviceToHost, x.stream);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpyAsync(&x.mtp_verifier_host->routing_finite,
                            x.routing_finite, sizeof(int),
                            cudaMemcpyDeviceToHost, x.stream);
  }
  if (error == cudaSuccess) error = cudaStreamSynchronize(x.stream);
  if (error != cudaSuccess) {
    return CudaFailure("synchronize M25 verifier transaction", error);
  }
  const auto& completed = x.mtp_verifier_host->transaction;
  if (completed.control.transition_valid != 1U ||
      completed.control.proposal_count != proposal_count ||
      completed.result.output_count == 0U ||
      completed.result.output_count > tokens ||
      completed.control.next.input_token !=
          completed.result.verified[completed.result.output_count - 1U] ||
      completed.control.next.processed_position !=
          completed.control.current.processed_position +
              completed.result.output_count) {
    return Status(StatusCode::kInternal,
                  "M25 verifier device transition is inconsistent");
  }
  if (x.mtp_verifier_host->routing_finite == 0 ||
      std::any_of(x.mtp_verifier_host->logits_finite.begin(),
                  x.mtp_verifier_host->logits_finite.begin() + tokens,
                  [](int value) { return value == 0; })) {
    return Status(StatusCode::kInternal,
                  "M25 Target verification produced non-finite values");
  }
  x.last_mtp_verification_min_margin =
      std::numeric_limits<float>::infinity();
  for (std::uint64_t row = 0U; row < tokens; ++row) {
    x.last_mtp_verification_min_margin = std::min(
        x.last_mtp_verification_min_margin,
        x.mtp_verifier_host->top_values[row] -
            x.mtp_verifier_host->second_values[row]);
  }

  const std::uint64_t previous_position = x.position;
  x.position += completed.result.output_count;
  if (x.sliding_capacity != 0U) {
    x.sliding_ring_wraps += x.position / x.sliding_capacity -
                            previous_position / x.sliding_capacity;
  }
  x.maximum_global_position_exclusive =
      std::max(x.maximum_global_position_exclusive, x.position);
  x.token_selections += completed.result.output_count;
  if (x.sampling.enabled) {
    x.sampling_step += completed.result.output_count;
  }
  x.pending_decode_self_feed = false;
  x.decode_self_feed_valid = false;
  x.has_last_input_token = true;
  x.last_input_token = completed.result.output_count == 1U
                           ? pending_token
                           : completed.result.proposed[
                                 completed.result.output_count - 2U];
  *host_result = completed.result;
  return Status::Ok();
}

Status Gemma4Moe26BReferenceEngine::RunFixedMtpGraphChain(
    std::uint32_t pending_token, std::uint32_t draft_count,
    std::span<std::uint32_t> output_token_ids,
    MtpChainResult* host_result) {
  if (!implementation_ || host_result == nullptr ||
      output_token_ids.empty() || pending_token >= kVocabulary ||
      draft_count == 0U || draft_count > kMaximumMtpDraftTokens) {
    return Invalid("M25 fixed graph chain request is invalid");
  }
  auto& x = *implementation_;
  if (x.mtp_chain_graphs[draft_count] == nullptr ||
      !x.assistant.prepared() ||
      x.position == 0U || x.position >= x.context ||
      output_token_ids.size() > x.context - x.position) {
    return Invalid("M25 fixed graph chain exceeds the resident state");
  }
  *x.mtp_verifier_host = {};
  auto& host_control = x.mtp_verifier_host->transaction.control;
  host_control.current.input_token = pending_token;
  host_control.current.processed_position = x.position - 1U;
  host_control.current.remaining_output_capacity = output_token_ids.size();
  host_control.current.output_write_position = 0U;
  host_control.current.sampling_step = x.sampling_step;
  host_control.next = host_control.current;
  host_control.fixed_draft_tokens = draft_count;
  host_control.sampling_enabled = x.sampling.enabled ? 1U : 0U;
  auto* device_transaction =
      x.mtp_verifier.As<MtpGroupTransaction>(x.mtp_transaction_offset);
  auto* device_result =
      x.mtp_verifier.As<MtpChainResult>(x.mtp_chain_result_offset);
  cudaError_t error = cudaMemcpyAsync(
      device_transaction, &x.mtp_verifier_host->transaction,
      sizeof(MtpGroupTransaction), cudaMemcpyHostToDevice, x.stream);
  if (error == cudaSuccess) {
    error = cudaMemsetAsync(device_result, 0, sizeof(MtpChainResult), x.stream);
  }
  if (error == cudaSuccess) {
    error = cudaGraphLaunch(x.mtp_chain_graphs[draft_count], x.stream);
  }
  if (error == cudaSuccess) error = cudaStreamSynchronize(x.stream);
  if (error != cudaSuccess) {
    return CudaFailure("execute M25 fixed graph chain", error);
  }
  ++x.mtp_chain_graph_launches[draft_count];

  auto* pinned_result = reinterpret_cast<MtpChainResult*>(x.mtp_chain_host);
  auto* pinned_outputs = reinterpret_cast<std::uint32_t*>(
      x.mtp_chain_host + sizeof(MtpChainResult));
  error = cudaMemcpy(pinned_result, device_result, sizeof(MtpChainResult),
                     cudaMemcpyDeviceToHost);
  if (error == cudaSuccess) {
    error = cudaMemcpy(
        pinned_outputs,
        x.mtp_verifier.As<std::uint32_t>(x.mtp_chain_outputs_offset),
        output_token_ids.size_bytes(), cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(&x.mtp_verifier_host->transaction, device_transaction,
                       sizeof(MtpGroupTransaction), cudaMemcpyDeviceToHost);
  }
  if (error != cudaSuccess) {
    return CudaFailure("copy M25 fixed graph chain result", error);
  }
  const auto& final_control = x.mtp_verifier_host->transaction.control;
  if (pinned_result->output_count == 0U ||
      pinned_result->output_count > output_token_ids.size() ||
      (pinned_result->stopped == 0U &&
       pinned_result->output_count != output_token_ids.size()) ||
      pinned_result->proposed_count !=
          draft_count * pinned_result->group_count ||
      pinned_result->accepted_count + pinned_result->rejected_count !=
          pinned_result->proposed_count ||
      pinned_result->ordinary_tail_count > draft_count ||
      pinned_result->non_finite_step_count != 0U ||
      final_control.current.input_token !=
          pinned_outputs[pinned_result->output_count - 1U] ||
      final_control.current.processed_position !=
          x.position - 1U + pinned_result->output_count ||
      final_control.current.remaining_output_capacity !=
          output_token_ids.size() - pinned_result->output_count ||
      final_control.current.output_write_position !=
          pinned_result->output_count) {
    return Status(StatusCode::kInternal,
                  "M25 fixed graph chain result is inconsistent");
  }
  std::copy_n(pinned_outputs, pinned_result->output_count,
              output_token_ids.begin());
  const std::uint64_t previous_position = x.position;
  x.position += pinned_result->output_count;
  if (x.sliding_capacity != 0U) {
    x.sliding_ring_wraps += x.position / x.sliding_capacity -
                            previous_position / x.sliding_capacity;
  }
  x.maximum_global_position_exclusive =
      std::max(x.maximum_global_position_exclusive, x.position);
  x.token_selections += pinned_result->output_count;
  x.sampling_step = final_control.current.sampling_step;
  x.pending_decode_self_feed = false;
  x.decode_self_feed_valid = false;
  x.has_last_input_token = true;
  x.last_input_token = pinned_result->output_count == 1U
                           ? pending_token
                           : output_token_ids[pinned_result->output_count - 2U];
  *host_result = *pinned_result;
  return Status::Ok();
}

bool Gemma4Moe26BReferenceEngine::mtp_assistant_loaded() const {
  return implementation_ != nullptr && implementation_->assistant.loaded();
}

std::uint64_t Gemma4Moe26BReferenceEngine::mtp_assistant_weight_bytes() const {
  return implementation_ == nullptr ? 0U
                                    : implementation_->assistant.arena_bytes();
}

std::uint64_t
Gemma4Moe26BReferenceEngine::mtp_assistant_workspace_bytes() const {
  return implementation_ == nullptr
             ? 0U
             : implementation_->assistant.workspace_bytes();
}

bool Gemma4Moe26BReferenceEngine::mtp_group_graph_prepared(
    std::uint32_t draft_count) const {
  return implementation_ != nullptr &&
         draft_count <= kMaximumMtpDraftTokens &&
         implementation_->mtp_group_graphs[draft_count] != nullptr;
}

std::uint64_t
Gemma4Moe26BReferenceEngine::mtp_group_graph_device_bytes() const {
  return implementation_ == nullptr
             ? 0U
             : implementation_->mtp_group_graph_device_bytes;
}

std::uint64_t Gemma4Moe26BReferenceEngine::mtp_group_graph_launches() const {
  return implementation_ == nullptr
             ? 0U
             : implementation_->mtp_group_graph_launches;
}

bool Gemma4Moe26BReferenceEngine::mtp_chain_graph_prepared(
    std::uint32_t draft_count) const {
  return implementation_ != nullptr &&
         draft_count <= kMaximumMtpDraftTokens &&
         implementation_->mtp_chain_graphs[draft_count] != nullptr;
}

std::uint64_t
Gemma4Moe26BReferenceEngine::mtp_chain_graph_device_bytes(
    std::uint32_t draft_count) const {
  return implementation_ == nullptr || draft_count > kMaximumMtpDraftTokens
             ? 0U
             : implementation_->mtp_chain_graph_device_bytes[draft_count];
}

std::uint64_t Gemma4Moe26BReferenceEngine::mtp_chain_graph_launches(
    std::uint32_t draft_count) const {
  return implementation_ == nullptr || draft_count > kMaximumMtpDraftTokens
             ? 0U
             : implementation_->mtp_chain_graph_launches[draft_count];
}

float Gemma4Moe26BReferenceEngine::last_mtp_verification_min_margin() const {
  return implementation_ == nullptr
             ? 0.0F
             : implementation_->last_mtp_verification_min_margin;
}

Status Gemma4Moe26BReferenceEngine::CopyMtpAssistantOracleInputs(
    std::span<float> concatenated_input,
    std::span<float> assistant_logits,
    std::span<std::uint8_t> sliding_key,
    std::span<std::uint8_t> sliding_value,
    std::span<std::uint8_t> full_key,
    std::span<std::uint8_t> full_value,
    std::span<std::uint16_t> kv_scale_bf16_bits) {
  if (!implementation_) return Invalid("M25 oracle engine is unavailable");
  auto& x = *implementation_;
  constexpr std::uint32_t kSlidingLayer = 28U;
  constexpr std::uint32_t kFullLayer = 29U;
  const std::uint64_t sliding_tokens =
      std::min(x.position, x.caches[kSlidingLayer].capacity);
  const std::uint64_t sliding_bytes =
      sliding_tokens * x.traits[kSlidingLayer].kv_heads *
      x.traits[kSlidingLayer].head_dimension;
  const std::uint64_t full_bytes =
      x.position * x.traits[kFullLayer].kv_heads *
      x.traits[kFullLayer].head_dimension;
  if (!x.assistant.prepared() || x.position == 0U ||
      x.position > x.caches[kSlidingLayer].capacity ||
      sliding_key.size() != sliding_bytes ||
      sliding_value.size() != sliding_bytes ||
      full_key.size() != full_bytes || full_value.size() != full_bytes ||
      kv_scale_bf16_bits.size() != 4U) {
    return Invalid("M25 Assistant oracle cache geometry is invalid");
  }
  Status status = x.assistant.CopyLastOracleState(
      concatenated_input, assistant_logits, x.stream);
  if (!status.ok()) return status;
  const std::array copies = {
      std::pair{static_cast<void*>(sliding_key.data()),
                static_cast<const void*>(x.caches[kSlidingLayer].key)},
      std::pair{static_cast<void*>(sliding_value.data()),
                static_cast<const void*>(x.caches[kSlidingLayer].value)},
      std::pair{static_cast<void*>(full_key.data()),
                static_cast<const void*>(x.caches[kFullLayer].key)},
      std::pair{static_cast<void*>(full_value.data()),
                static_cast<const void*>(x.caches[kFullLayer].value)},
  };
  const std::array sizes = {sliding_key.size_bytes(),
                            sliding_value.size_bytes(), full_key.size_bytes(),
                            full_value.size_bytes()};
  cudaError_t error = cudaSuccess;
  for (std::size_t index = 0; index < copies.size() &&
                              error == cudaSuccess;
       ++index) {
    error = cudaMemcpyAsync(copies[index].first, copies[index].second,
                            sizes[index], cudaMemcpyDeviceToHost, x.stream);
  }
  const std::array<const std::uint16_t*, 4> scales = {
      x.attention_weights[kSlidingLayer].key_cache_scale_bf16,
      x.attention_weights[kSlidingLayer].value_cache_scale_bf16,
      x.attention_weights[kFullLayer].key_cache_scale_bf16,
      x.attention_weights[kFullLayer].value_cache_scale_bf16,
  };
  for (std::size_t index = 0; index < scales.size() && error == cudaSuccess;
       ++index) {
    error = cudaMemcpyAsync(kv_scale_bf16_bits.data() + index, scales[index],
                            sizeof(std::uint16_t), cudaMemcpyDeviceToHost,
                            x.stream);
  }
  if (error == cudaSuccess) error = cudaStreamSynchronize(x.stream);
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("copy M25 Assistant oracle cache", error);
}

Status Gemma4Moe26BReferenceEngine::CopyLogits(std::span<float> output) {
  if (!implementation_ || output.size() != kVocabulary) {
    return Invalid("M13 logit destination has the wrong extent");
  }
  cudaError_t error = cudaStreamSynchronize(implementation_->stream);
  if (error == cudaSuccess) {
    error = cudaMemcpy(output.data(), implementation_->logits,
                       output.size_bytes(), cudaMemcpyDeviceToHost);
  }
  return error == cudaSuccess ? Status::Ok()
                              : CudaFailure("copy M13 logits", error);
}

Status Gemma4Moe26BReferenceEngine::CopyLayerOutput(
    std::uint32_t layer, std::span<float> output) {
  const int index = CaptureIndex(layer);
  if (!implementation_ || index < 0 || output.size() != kWidth) {
    return Invalid("M13 layer capture request is invalid");
  }
  cudaError_t error = cudaStreamSynchronize(implementation_->stream);
  if (error == cudaSuccess) {
    error = cudaMemcpy(output.data(),
                       implementation_->layer_captures[static_cast<std::size_t>(index)],
                       output.size_bytes(), cudaMemcpyDeviceToHost);
  }
  return error == cudaSuccess ? Status::Ok()
                              : CudaFailure("copy M13 layer output", error);
}

Status Gemma4Moe26BReferenceEngine::CopyRouterProbabilities(
    std::uint32_t layer, std::span<float> output) {
  const int index = CaptureIndex(layer);
  if (!implementation_ || index < 0 || output.size() != kExperts) {
    return Invalid("M13 router probability capture request is invalid");
  }
  cudaError_t error = cudaStreamSynchronize(implementation_->stream);
  if (error == cudaSuccess) {
    error = cudaMemcpy(
        output.data(),
        implementation_->router_probability_captures[static_cast<std::size_t>(index)],
        output.size_bytes(), cudaMemcpyDeviceToHost);
  }
  return error == cudaSuccess ? Status::Ok()
                              : CudaFailure("copy M13 router probabilities", error);
}

Status Gemma4Moe26BReferenceEngine::CopyRouterTopIds(
    std::uint32_t layer, std::span<std::uint32_t> output) {
  const int index = CaptureIndex(layer);
  if (!implementation_ || index < 0 || output.size() != kTopK) {
    return Invalid("M13 router ID capture request is invalid");
  }
  cudaError_t error = cudaStreamSynchronize(implementation_->stream);
  if (error == cudaSuccess) {
    error = cudaMemcpy(
        output.data(),
        implementation_->router_id_captures[static_cast<std::size_t>(index)],
        output.size_bytes(), cudaMemcpyDeviceToHost);
  }
  return error == cudaSuccess ? Status::Ok()
                              : CudaFailure("copy M13 router IDs", error);
}

std::uint64_t Gemma4Moe26BReferenceEngine::position() const {
  return implementation_ ? implementation_->position : 0U;
}
std::uint64_t Gemma4Moe26BReferenceEngine::context_capacity() const {
  return implementation_ ? implementation_->context : 0U;
}
std::uint64_t Gemma4Moe26BReferenceEngine::weight_arena_bytes() const {
  return implementation_ ? implementation_->artifact.arena_bytes() : 0U;
}
std::uint64_t Gemma4Moe26BReferenceEngine::kv_cache_bytes() const {
  return implementation_ ? implementation_->kv.bytes() : 0U;
}
std::uint64_t Gemma4Moe26BReferenceEngine::workspace_bytes() const {
  return implementation_ ? implementation_->workspace.bytes() +
                               implementation_->prefill_workspace.bytes() +
                               implementation_->mtp_verifier.bytes()
                         : 0U;
}
std::uint64_t Gemma4Moe26BReferenceEngine::sliding_cache_capacity() const {
  return implementation_ ? implementation_->sliding_capacity : 0U;
}
std::uint64_t Gemma4Moe26BReferenceEngine::prefill_chunk_count() const {
  return implementation_ ? implementation_->prefill_chunks : 0U;
}
std::uint64_t
Gemma4Moe26BReferenceEngine::minimum_prefill_chunk_tokens() const {
  return implementation_ ? implementation_->minimum_prefill_chunk : 0U;
}
Gemma4Moe26BExecutionEvidence
Gemma4Moe26BReferenceEngine::execution_evidence() const {
  Gemma4Moe26BExecutionEvidence result;
  if (!implementation_) return result;
  result.integrated_native_backend =
      implementation_->backend == Gemma4Moe26BBackend::kSm120Integrated;
  result.decode_graph_ready = implementation_->decode_graph != nullptr;
  result.tensor_core_prefill_router =
      implementation_->moe_config.prefill_router ==
      Gemma4MoePrefillRouter::kSm120TensorCore;
  result.prefill_calls = implementation_->prefill_calls;
  result.prefill_chunks = implementation_->prefill_chunks;
  result.decode_graph_launches = implementation_->decode_graph_launches;
  result.token_selections = implementation_->token_selections;
  result.sliding_ring_wraps = implementation_->sliding_ring_wraps;
  result.maximum_global_position_exclusive =
      implementation_->maximum_global_position_exclusive;
  result.fallback_count = implementation_->fallback_count;
  result.recurring_allocation_count =
      implementation_->recurring_allocation_count;
  return result;
}

}  // namespace gem16::internal
