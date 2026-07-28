#include "cuda/mtp/assistant.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cuda/attention/sm120.h"
#include "cuda/layer/reference.h"
#include "cuda/nvfp4/mlp.h"
#include "gem16/model.h"
#include "gem16/types.h"
#include "platform/mapped_file.h"

namespace gem16::internal {
namespace {

constexpr std::uint64_t kArenaAlignment = 256U;
constexpr std::uint64_t kAssistantHidden = 1024U;
constexpr std::uint64_t kBackboneHidden = 3840U;
constexpr std::uint64_t kIntermediate = 8192U;
constexpr std::uint64_t kVocabulary = 262144U;
constexpr std::size_t kUploadProbeBytes = 32U;
constexpr std::uint64_t kQueryHeads = 16U;
constexpr unsigned kThreads = 256U;
constexpr unsigned kOutputHeadBlocks = 4096U;
constexpr std::uint32_t kSuppressedTokenA = 258883U;
constexpr std::uint32_t kSuppressedTokenB = 258882U;
constexpr float kEpsilon = 1.0e-6F;

struct ArgmaxValue {
  float value;
  std::uint32_t token;
};

__device__ float BlockSum(float value) {
  __shared__ float scratch[kThreads];
  scratch[threadIdx.x] = value;
  __syncthreads();
  for (unsigned stride = kThreads / 2U; stride != 0U; stride >>= 1U) {
    if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    __syncthreads();
  }
  return scratch[0];
}

__global__ void PreProjectionKernel(
    const std::uint16_t* target_embedding, const std::uint32_t* token,
    const float* backbone_hidden, const std::uint16_t* weight,
    float* output) {
  const std::uint64_t row = blockIdx.x;
  float sum = 0.0F;
  const float embedding_scale = static_cast<float>(
      __float2bfloat16_rn(sqrtf(static_cast<float>(kBackboneHidden))));
  for (std::uint64_t column = threadIdx.x;
       column < 2U * kBackboneHidden; column += blockDim.x) {
    float input = 0.0F;
    if (column < kBackboneHidden) {
      const float raw = static_cast<float>(__ushort_as_bfloat16(
          target_embedding[static_cast<std::uint64_t>(token[0]) *
                               kBackboneHidden +
                           column]));
      input = static_cast<float>(__float2bfloat16_rn(raw * embedding_scale));
    } else {
      input = backbone_hidden[column - kBackboneHidden];
    }
    const float coefficient = static_cast<float>(__ushort_as_bfloat16(
        weight[row * (2U * kBackboneHidden) + column]));
    sum = fmaf(input, coefficient, sum);
  }
  const float reduced = BlockSum(sum);
  if (threadIdx.x == 0U) {
    output[row] = static_cast<float>(__float2bfloat16_rn(reduced));
  }
}

__global__ void Bf16GemvKernel(
    const float* input, const std::uint16_t* weight, float* output,
    std::uint64_t rows, std::uint64_t contracting) {
  const std::uint64_t row = blockIdx.x;
  if (row >= rows) return;
  float sum = 0.0F;
  for (std::uint64_t column = threadIdx.x; column < contracting;
       column += blockDim.x) {
    const float coefficient = static_cast<float>(__ushort_as_bfloat16(
        weight[row * contracting + column]));
    sum = fmaf(input[column], coefficient, sum);
  }
  const float reduced = BlockSum(sum);
  if (threadIdx.x == 0U) {
    output[row] = static_cast<float>(__float2bfloat16_rn(reduced));
  }
}

__global__ void SetAssistantAttentionPositionKernel(
    DecodeControl* control, std::uint64_t position) {
  if (blockIdx.x == 0U && threadIdx.x == 0U) control->position = position;
}

__global__ void RoundBf16Kernel(float* values, std::uint64_t elements) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    values[index] = static_cast<float>(__float2bfloat16_rn(values[index]));
  }
}

__device__ bool Better(ArgmaxValue candidate, ArgmaxValue current) {
  return candidate.value > current.value ||
         (candidate.value == current.value && candidate.token < current.token);
}

__global__ void AssistantOutputCandidatesKernel(
    const std::uint16_t* weights, const float* hidden,
    ArgmaxValue* candidates) {
  constexpr unsigned kWarpSize = 32U;
  constexpr unsigned kWarps = kThreads / kWarpSize;
  __shared__ ArgmaxValue warp_candidates[kWarps];
  const unsigned warp = threadIdx.x / kWarpSize;
  const unsigned lane = threadIdx.x % kWarpSize;
  ArgmaxValue best{-FLT_MAX, 0U};
  for (std::uint64_t token =
           static_cast<std::uint64_t>(blockIdx.x) * kWarps + warp;
       token < kVocabulary;
       token += static_cast<std::uint64_t>(gridDim.x) * kWarps) {
    float sum = 0.0F;
    const std::uint64_t base = token * kAssistantHidden;
    for (std::uint64_t index = lane; index < kAssistantHidden;
         index += kWarpSize) {
      const float weight = static_cast<float>(
          __ushort_as_bfloat16(weights[base + index]));
      sum = fmaf(weight, hidden[index], sum);
    }
    for (unsigned offset = kWarpSize / 2U; offset != 0U; offset >>= 1U) {
      sum += __shfl_down_sync(0xFFFFFFFFU, sum, offset);
    }
    if (lane == 0U && token != kSuppressedTokenA &&
        token != kSuppressedTokenB) {
      const ArgmaxValue candidate{sum, static_cast<std::uint32_t>(token)};
      if (Better(candidate, best)) best = candidate;
    }
  }
  if (lane == 0U) warp_candidates[warp] = best;
  __syncthreads();
  if (warp == 0U) {
    best = lane < kWarps ? warp_candidates[lane]
                         : ArgmaxValue{-FLT_MAX, 0U};
    for (unsigned offset = kWarpSize / 2U; offset != 0U; offset >>= 1U) {
      const ArgmaxValue other{
          __shfl_down_sync(0xFFFFFFFFU, best.value, offset),
          __shfl_down_sync(0xFFFFFFFFU, best.token, offset)};
      if (Better(other, best)) best = other;
    }
    if (lane == 0U) candidates[blockIdx.x] = best;
  }
}

__global__ void AssistantOutputArgmaxKernel(
    const ArgmaxValue* candidates, std::uint32_t* selected,
    std::uint32_t* draft_tokens, std::uint32_t draft_index) {
  __shared__ ArgmaxValue scratch[kThreads];
  ArgmaxValue best{-FLT_MAX, 0U};
  for (std::uint32_t index = threadIdx.x; index < kOutputHeadBlocks;
       index += blockDim.x) {
    if (Better(candidates[index], best)) best = candidates[index];
  }
  scratch[threadIdx.x] = best;
  __syncthreads();
  for (unsigned stride = kThreads / 2U; stride != 0U; stride >>= 1U) {
    if (threadIdx.x < stride &&
        Better(scratch[threadIdx.x + stride], scratch[threadIdx.x])) {
      scratch[threadIdx.x] = scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0U) {
    selected[0] = scratch[0].token;
    draft_tokens[draft_index] = scratch[0].token;
  }
}

Status Error(StatusCode code, std::string message) {
  return Status(code, std::move(message));
}

Status CudaFailure(const char* operation, cudaError_t error) {
  return Error(StatusCode::kInternal,
               std::string(operation) + ": " + cudaGetErrorName(error) +
                   ": " + cudaGetErrorString(error));
}

Result<std::uint64_t> AlignUp(std::uint64_t value) {
  constexpr std::uint64_t mask = kArenaAlignment - 1U;
  if (value > std::numeric_limits<std::uint64_t>::max() - mask) {
    return Error(StatusCode::kInvalidArgument,
                 "assistant arena offset overflows uint64");
  }
  return (value + mask) & ~mask;
}

Status VerifyUploadEdges(const std::byte* device, const std::byte* source,
                         std::uint64_t bytes, const std::string& name) {
  if (device == nullptr || source == nullptr || bytes == 0U) {
    return Error(StatusCode::kDataLoss,
                 "assistant upload probe has an invalid range: " + name);
  }
  const std::size_t probe = static_cast<std::size_t>(
      std::min<std::uint64_t>(bytes, kUploadProbeBytes));
  std::array<std::byte, kUploadProbeBytes> observed{};
  cudaError_t error = cudaMemcpy(observed.data(), device, probe,
                                 cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    return CudaFailure("verify assistant tensor prefix", error);
  }
  if (std::memcmp(observed.data(), source, probe) != 0) {
    return Error(StatusCode::kDataLoss,
                 "assistant device tensor prefix differs from source: " + name);
  }
  if (bytes <= probe) return Status::Ok();
  const std::uint64_t suffix_offset = bytes - probe;
  error = cudaMemcpy(observed.data(), device + suffix_offset, probe,
                     cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    return CudaFailure("verify assistant tensor suffix", error);
  }
  if (std::memcmp(observed.data(), source + suffix_offset, probe) != 0) {
    return Error(StatusCode::kDataLoss,
                 "assistant device tensor suffix differs from source: " + name);
  }
  return Status::Ok();
}

}  // namespace

struct AssistantModel::Impl {
  struct DeviceTensor {
    const TensorInfo* info = nullptr;
    std::byte* data = nullptr;
  };

  struct WorkspaceOffsets {
    std::uint64_t target_hidden = 0U;
    std::uint64_t hidden_a = 0U;
    std::uint64_t hidden_b = 0U;
    std::uint64_t normalized = 0U;
    std::uint64_t query = 0U;
    std::uint64_t normalized_query = 0U;
    std::uint64_t attention = 0U;
    std::uint64_t projection = 0U;
    std::uint64_t post_norm = 0U;
    std::uint64_t gate = 0U;
    std::uint64_t up = 0U;
    std::uint64_t product = 0U;
    std::uint64_t attention_control = 0U;
    std::uint64_t scores = 0U;
    std::uint64_t feedback_hidden = 0U;
    std::uint64_t candidates = 0U;
    std::uint64_t selected = 0U;
    std::uint64_t draft_tokens = 0U;
  };

  ~Impl() {
    if (workspace != nullptr) (void)cudaFree(workspace);
    if (arena != nullptr) (void)cudaFree(arena);
  }

  template <typename T>
  [[nodiscard]] T* Workspace(std::uint64_t offset) const {
    return reinterpret_cast<T*>(static_cast<std::byte*>(workspace) + offset);
  }

  [[nodiscard]] Result<const std::uint16_t*> Bf16(
      const std::string& name, std::vector<std::uint64_t> shape) const {
    const auto found = tensors.find(name);
    if (found == tensors.end()) {
      return Error(StatusCode::kNotFound,
                   "required assistant tensor is missing: " + name);
    }
    const DeviceTensor& tensor = found->second;
    if (tensor.info->storage_dtype != "BF16" || tensor.info->shape != shape) {
      return Error(StatusCode::kDataLoss,
                   "unexpected assistant BF16 tensor geometry: " + name);
    }
    return reinterpret_cast<const std::uint16_t*>(tensor.data);
  }

  [[nodiscard]] Status Bind() {
    auto embedding = Bf16("model.embed_tokens.weight",
                          {kVocabulary, kAssistantHidden});
    auto pre = Bf16("pre_projection.weight",
                    {kAssistantHidden, 2U * kBackboneHidden});
    auto post = Bf16("post_projection.weight",
                     {kBackboneHidden, kAssistantHidden});
    auto norm = Bf16("model.norm.weight", {kAssistantHidden});
    if (!embedding.ok()) return embedding.status();
    if (!pre.ok()) return pre.status();
    if (!post.ok()) return post.status();
    if (!norm.ok()) return norm.status();
    bindings.embedding = embedding.value();
    bindings.pre_projection = pre.value();
    bindings.post_projection = post.value();
    bindings.final_norm = norm.value();

    for (std::size_t index = 0; index < bindings.layers.size(); ++index) {
      AssistantLayerBinding& layer = bindings.layers[index];
      layer.global = index + 1U == bindings.layers.size();
      layer.query_elements = layer.global ? 8192U : 4096U;
      const std::string base = "model.layers." + std::to_string(index) + ".";
      auto input_norm = Bf16(base + "input_layernorm.weight", {kAssistantHidden});
      auto q = Bf16(base + "self_attn.q_proj.weight",
                    {layer.query_elements, kAssistantHidden});
      auto q_norm = Bf16(base + "self_attn.q_norm.weight",
                         {layer.global ? 512U : 256U});
      auto o = Bf16(base + "self_attn.o_proj.weight",
                    {kAssistantHidden, layer.query_elements});
      auto post_attention = Bf16(base + "post_attention_layernorm.weight",
                                 {kAssistantHidden});
      auto pre_feedforward = Bf16(base + "pre_feedforward_layernorm.weight",
                                  {kAssistantHidden});
      auto gate = Bf16(base + "mlp.gate_proj.weight",
                       {kIntermediate, kAssistantHidden});
      auto up = Bf16(base + "mlp.up_proj.weight",
                     {kIntermediate, kAssistantHidden});
      auto down = Bf16(base + "mlp.down_proj.weight",
                       {kAssistantHidden, kIntermediate});
      auto post_feedforward = Bf16(
          base + "post_feedforward_layernorm.weight", {kAssistantHidden});
      auto scalar = Bf16(base + "layer_scalar", {1U});
      for (const auto* result :
           {&input_norm, &q, &q_norm, &o, &post_attention, &pre_feedforward,
            &gate, &up, &down, &post_feedforward, &scalar}) {
        if (!result->ok()) return result->status();
      }
      layer.input_norm = input_norm.value();
      layer.q_projection = q.value();
      layer.q_norm = q_norm.value();
      layer.o_projection = o.value();
      layer.post_attention_norm = post_attention.value();
      layer.pre_feedforward_norm = pre_feedforward.value();
      layer.gate_projection = gate.value();
      layer.up_projection = up.value();
      layer.down_projection = down.value();
      layer.post_feedforward_norm = post_feedforward.value();
      layer.layer_scalar = scalar.value();
    }
    return Status::Ok();
  }

  ModelManifest manifest;
  std::unordered_map<std::string, DeviceTensor> tensors;
  AssistantBindings bindings;
  WorkspaceOffsets offsets;
  void* arena = nullptr;
  void* workspace = nullptr;
  std::uint64_t arena_size = 0U;
  std::uint64_t workspace_size = 0U;
  std::uint64_t max_context = 0U;
};

AssistantModel::AssistantModel() : impl_(std::make_unique<Impl>()) {}
AssistantModel::~AssistantModel() = default;

Status AssistantModel::Load(const std::filesystem::path& directory) {
  if (impl_->arena != nullptr || !impl_->tensors.empty()) {
    return Error(StatusCode::kInvalidArgument,
                 "assistant model is already loaded");
  }
  auto inspected = InspectCheckpoint({directory, true});
  if (!inspected.ok()) return inspected.status();
  if (inspected.value().architecture !=
          "Gemma4UnifiedAssistantForCausalLM" ||
      inspected.value().model_type != "gemma4_unified_assistant") {
    return Error(StatusCode::kUnsupported,
                 "--assistant-model requires the pinned Gemma 4 unified assistant");
  }
  impl_->manifest = std::move(inspected).value();

  std::uint64_t cursor = 0U;
  for (const TensorInfo& tensor : impl_->manifest.tensors) {
    if (!tensor.loaded_in_text_only_mode || tensor.storage_dtype != "BF16") {
      return Error(StatusCode::kDataLoss,
                   "assistant arena accepts only resident BF16 tensors: " +
                       tensor.name);
    }
    auto aligned = AlignUp(cursor);
    if (!aligned.ok()) return aligned.status();
    if (tensor.byte_length >
        std::numeric_limits<std::uint64_t>::max() - aligned.value()) {
      return Error(StatusCode::kInvalidArgument,
                   "assistant arena size overflows uint64");
    }
    cursor = aligned.value() + tensor.byte_length;
  }
  auto arena_size = AlignUp(cursor);
  if (!arena_size.ok()) return arena_size.status();
  if (arena_size.value() == 0U ||
      arena_size.value() > std::numeric_limits<std::size_t>::max()) {
    return Error(StatusCode::kInvalidArgument,
                 "assistant arena size is invalid");
  }
  cudaError_t error =
      cudaMalloc(&impl_->arena, static_cast<std::size_t>(arena_size.value()));
  if (error != cudaSuccess) {
    return CudaFailure("allocate BF16 assistant weight arena", error);
  }
  impl_->arena_size = arena_size.value();

  cursor = 0U;
  for (const TensorInfo& tensor : impl_->manifest.tensors) {
    auto aligned = AlignUp(cursor);
    if (!aligned.ok()) return aligned.status();
    Impl::DeviceTensor view{&tensor,
                            static_cast<std::byte*>(impl_->arena) + aligned.value()};
    if (!impl_->tensors.emplace(tensor.name, view).second) {
      return Error(StatusCode::kDataLoss,
                   "duplicate assistant device tensor: " + tensor.name);
    }
    cursor = aligned.value() + tensor.byte_length;
  }

  std::set<std::string> shards;
  for (const TensorInfo& tensor : impl_->manifest.tensors) {
    shards.insert(tensor.source_shard);
  }
  for (const std::string& shard : shards) {
    auto mapped = MappedFile::Open(directory / shard);
    if (!mapped.ok()) return mapped.status();
    for (const TensorInfo& tensor : impl_->manifest.tensors) {
      if (tensor.source_shard != shard) continue;
      if (tensor.byte_offset > mapped.value().size() ||
          tensor.byte_length > mapped.value().size() - tensor.byte_offset) {
        return Error(StatusCode::kDataLoss,
                     "assistant tensor upload range is invalid: " + tensor.name);
      }
      const auto found = impl_->tensors.find(tensor.name);
      if (found == impl_->tensors.end()) {
        return Error(StatusCode::kInternal,
                     "assistant tensor has no device view: " + tensor.name);
      }
      const std::byte* source = mapped.value().data() + tensor.byte_offset;
      error = cudaMemcpy(found->second.data, source,
                         static_cast<std::size_t>(tensor.byte_length),
                         cudaMemcpyHostToDevice);
      if (error != cudaSuccess) {
        return CudaFailure("upload BF16 assistant tensor", error);
      }
      Status status = VerifyUploadEdges(found->second.data, source,
                                        tensor.byte_length, tensor.name);
      if (!status.ok()) return status;
    }
  }
  return impl_->Bind();
}

Status AssistantModel::Prepare(std::uint64_t max_context) {
  if (!loaded()) {
    return Error(StatusCode::kInvalidArgument,
                 "assistant weights must be loaded before workspace preparation");
  }
  if (impl_->workspace != nullptr || max_context == 0U ||
      max_context > 262144U) {
    return Error(StatusCode::kInvalidArgument,
                 "assistant workspace context is invalid or already prepared");
  }
  std::uint64_t cursor = 0U;
  const auto add = [&cursor](std::uint64_t elements, std::uint64_t element_bytes,
                             std::uint64_t& destination) -> Status {
    constexpr std::uint64_t alignment = 16U;
    const std::uint64_t mask = alignment - 1U;
    if (cursor > std::numeric_limits<std::uint64_t>::max() - mask) {
      return Error(StatusCode::kInvalidArgument,
                   "assistant workspace alignment overflows uint64");
    }
    const std::uint64_t aligned = (cursor + mask) & ~mask;
    if (elements != 0U &&
        element_bytes > std::numeric_limits<std::uint64_t>::max() / elements) {
      return Error(StatusCode::kInvalidArgument,
                   "assistant workspace extent overflows uint64");
    }
    const std::uint64_t bytes = elements * element_bytes;
    if (bytes > std::numeric_limits<std::uint64_t>::max() - aligned) {
      return Error(StatusCode::kInvalidArgument,
                   "assistant workspace size overflows uint64");
    }
    destination = aligned;
    cursor = aligned + bytes;
    return Status::Ok();
  };
  struct Region {
    std::uint64_t elements;
    std::uint64_t element_bytes;
    std::uint64_t* destination;
  };
  const std::uint64_t attention_workspace_elements = std::max(
      kQueryHeads * max_context, DecodeAttentionWorkspaceElements(max_context));
  const std::array regions = {
      Region{kBackboneHidden, sizeof(float), &impl_->offsets.target_hidden},
      Region{kAssistantHidden, sizeof(float), &impl_->offsets.hidden_a},
      Region{kAssistantHidden, sizeof(float), &impl_->offsets.hidden_b},
      Region{kAssistantHidden, sizeof(float), &impl_->offsets.normalized},
      Region{kQueryHeads * 512U, sizeof(float), &impl_->offsets.query},
      Region{kQueryHeads * 512U, sizeof(float),
             &impl_->offsets.normalized_query},
      Region{kQueryHeads * 512U, sizeof(float), &impl_->offsets.attention},
      Region{kAssistantHidden, sizeof(float), &impl_->offsets.projection},
      Region{kAssistantHidden, sizeof(float), &impl_->offsets.post_norm},
      Region{kIntermediate, sizeof(float), &impl_->offsets.gate},
      Region{kIntermediate, sizeof(float), &impl_->offsets.up},
      Region{kIntermediate, sizeof(float), &impl_->offsets.product},
      Region{1U, sizeof(DecodeControl), &impl_->offsets.attention_control},
      Region{attention_workspace_elements, sizeof(float),
             &impl_->offsets.scores},
      Region{kBackboneHidden, sizeof(float), &impl_->offsets.feedback_hidden},
      Region{kOutputHeadBlocks, sizeof(ArgmaxValue), &impl_->offsets.candidates},
      Region{1U, sizeof(std::uint32_t), &impl_->offsets.selected},
      Region{4U, sizeof(std::uint32_t), &impl_->offsets.draft_tokens},
  };
  for (const Region& region : regions) {
    Status status =
        add(region.elements, region.element_bytes, *region.destination);
    if (!status.ok()) return status;
  }
  auto aligned_size = AlignUp(cursor);
  if (!aligned_size.ok()) return aligned_size.status();
  cudaError_t error = cudaMalloc(
      &impl_->workspace, static_cast<std::size_t>(aligned_size.value()));
  if (error != cudaSuccess) {
    return CudaFailure("allocate assistant proposal workspace", error);
  }
  impl_->workspace_size = aligned_size.value();
  impl_->max_context = max_context;
  return Status::Ok();
}

Status AssistantModel::GenerateDraftsDevice(
    const AssistantProposalContext& context, std::uint32_t draft_count,
    cudaStream_t stream) {
  if (!prepared() || stream == nullptr || draft_count == 0U ||
      draft_count > 4U || context.target_embedding == nullptr ||
      context.target_hidden == nullptr || context.input_token >= kVocabulary ||
      context.position >= impl_->max_context) {
    return Error(StatusCode::kInvalidArgument,
                 "assistant proposal context is invalid");
  }
  const auto validate_kv = [this](const AssistantSharedKvView& view,
                                  std::uint64_t expected_heads,
                                  std::uint64_t expected_dimension) -> Status {
    if (view.tokens == 0U || view.tokens > view.capacity ||
        view.capacity > impl_->max_context || view.kv_heads != expected_heads ||
        view.head_dimension != expected_dimension) {
      return Error(StatusCode::kInvalidArgument,
                   "assistant shared-KV geometry is invalid");
    }
    if (view.mode == AssistantKvCacheMode::kCheckpointFp8) {
      if (view.key_fp8 == nullptr || view.value_fp8 == nullptr ||
          view.key_scale_bf16 == nullptr || view.value_scale_bf16 == nullptr) {
        return Error(StatusCode::kInvalidArgument,
                     "assistant FP8 shared-KV pointers are incomplete");
      }
    } else if (view.key_bf16 == nullptr || view.value_bf16 == nullptr) {
      return Error(StatusCode::kInvalidArgument,
                   "assistant BF16 shared-KV pointers are incomplete");
    }
    return Status::Ok();
  };
  Status status = validate_kv(context.sliding_kv, 8U, 256U);
  if (!status.ok()) return status;
  status = validate_kv(context.full_kv, 1U, 512U);
  if (!status.ok()) return status;
  if (context.sliding_kv.mode != context.full_kv.mode) {
    return Error(StatusCode::kInvalidArgument,
                 "assistant shared-KV modes disagree");
  }

  float* target_hidden = impl_->Workspace<float>(impl_->offsets.target_hidden);
  cudaError_t error = cudaMemcpyAsync(
      target_hidden, context.target_hidden,
      static_cast<std::size_t>(kBackboneHidden * sizeof(float)),
      cudaMemcpyDeviceToDevice, stream);
  if (error != cudaSuccess) {
    return CudaFailure("copy target hidden state for assistant", error);
  }

  float* hidden_a = impl_->Workspace<float>(impl_->offsets.hidden_a);
  float* hidden_b = impl_->Workspace<float>(impl_->offsets.hidden_b);
  float* normalized = impl_->Workspace<float>(impl_->offsets.normalized);
  float* query = impl_->Workspace<float>(impl_->offsets.query);
  float* normalized_query =
      impl_->Workspace<float>(impl_->offsets.normalized_query);
  float* attention = impl_->Workspace<float>(impl_->offsets.attention);
  float* projection = impl_->Workspace<float>(impl_->offsets.projection);
  float* post_norm = impl_->Workspace<float>(impl_->offsets.post_norm);
  float* gate = impl_->Workspace<float>(impl_->offsets.gate);
  float* up = impl_->Workspace<float>(impl_->offsets.up);
  float* product = impl_->Workspace<float>(impl_->offsets.product);
  auto* attention_control =
      impl_->Workspace<DecodeControl>(impl_->offsets.attention_control);
  float* scores = impl_->Workspace<float>(impl_->offsets.scores);
  float* feedback = impl_->Workspace<float>(impl_->offsets.feedback_hidden);
  auto* candidates = impl_->Workspace<ArgmaxValue>(impl_->offsets.candidates);
  auto* selected = impl_->Workspace<std::uint32_t>(impl_->offsets.selected);
  auto* device_drafts =
      impl_->Workspace<std::uint32_t>(impl_->offsets.draft_tokens);

  const auto launch_gemv = [stream](
                                const float* input,
                                const std::uint16_t* weight, float* output,
                                std::uint64_t rows,
                                std::uint64_t contracting) -> Status {
    Bf16GemvKernel<<<static_cast<unsigned>(rows), kThreads, 0, stream>>>(
        input, weight, output, rows, contracting);
    const cudaError_t launch_error = cudaGetLastError();
    return launch_error == cudaSuccess
               ? Status::Ok()
               : CudaFailure("launch assistant BF16 GEMV", launch_error);
  };
  const auto round_bf16 = [stream](float* values,
                                   std::uint64_t elements) -> Status {
    const std::uint64_t blocks = (elements + kThreads - 1U) / kThreads;
    RoundBf16Kernel<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
        values, elements);
    const cudaError_t launch_error = cudaGetLastError();
    return launch_error == cudaSuccess
               ? Status::Ok()
               : CudaFailure("launch assistant BF16 rounding", launch_error);
  };

  error = cudaMemcpyAsync(selected, &context.input_token,
                          sizeof(context.input_token), cudaMemcpyHostToDevice,
                          stream);
  if (error != cudaSuccess) {
    return CudaFailure("copy initial assistant draft token", error);
  }
  SetAssistantAttentionPositionKernel<<<1U, 1U, 0, stream>>>(
      attention_control, context.position);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaFailure("set assistant attention position", error);
  }
  const float* backbone_hidden = target_hidden;
  for (std::uint32_t step = 0; step < draft_count; ++step) {
    PreProjectionKernel<<<static_cast<unsigned>(kAssistantHidden), kThreads, 0,
                          stream>>>(
        context.target_embedding, selected, backbone_hidden,
        impl_->bindings.pre_projection, hidden_a);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
      return CudaFailure("launch assistant pre-projection", error);
    }

    for (const AssistantLayerBinding& layer : impl_->bindings.layers) {
      const AssistantSharedKvView& kv =
          layer.global ? context.full_kv : context.sliding_kv;
      status = LaunchRmsNormBf16(hidden_a, layer.input_norm, normalized, 1U,
                                 kAssistantHidden, kEpsilon, stream);
      if (!status.ok()) return status;
      status = launch_gemv(normalized, layer.q_projection, query,
                           layer.query_elements, kAssistantHidden);
      if (!status.ok()) return status;
      status = LaunchRmsNormBf16(
          query, layer.q_norm, normalized_query, kQueryHeads,
          layer.global ? 512U : 256U, kEpsilon, stream);
      if (!status.ok()) return status;
      if (layer.global) {
        status = LaunchProportionalRotaryEmbedding(
            normalized_query, kQueryHeads, 512U, 0.25, context.position,
            1000000.0, 1.0, stream);
      } else {
        status = LaunchRotaryEmbedding(normalized_query, kQueryHeads, 256U,
                                       256U, context.position, 10000.0,
                                       stream);
      }
      if (!status.ok()) return status;
      status = round_bf16(normalized_query, layer.query_elements);
      if (!status.ok()) return status;
      if (kv.mode == AssistantKvCacheMode::kCheckpointFp8 &&
          kv.tokens > 512U) {
        status = LaunchOnlineAttentionDecodeFp8Sm120(
            normalized_query, kv.key_fp8, kv.value_fp8, kv.key_scale_bf16,
            kv.value_scale_bf16, scores, attention, attention_control,
            kQueryHeads, kv.kv_heads, kv.head_dimension, kv.capacity,
            !layer.global, stream);
      } else if (kv.mode == AssistantKvCacheMode::kCheckpointFp8) {
        status = LaunchLocalAttentionDecodeFp8(
            normalized_query, kv.key_fp8, kv.value_fp8,
            kv.key_scale_bf16, kv.value_scale_bf16, scores, attention,
            kQueryHeads, kv.kv_heads, kv.head_dimension, kv.tokens, stream,
            kv.capacity, kv.first_slot);
      } else {
        status = LaunchLocalAttentionDecode(
            normalized_query, kv.key_bf16, kv.value_bf16, scores, attention,
            kQueryHeads, kv.kv_heads, kv.head_dimension, kv.tokens, stream,
            kv.capacity, kv.first_slot);
      }
      if (!status.ok()) return status;
      status = round_bf16(attention, layer.query_elements);
      if (!status.ok()) return status;
      status = launch_gemv(attention, layer.o_projection, projection,
                           kAssistantHidden, layer.query_elements);
      if (!status.ok()) return status;
      status = LaunchRmsNormResidualBf16(
          projection, layer.post_attention_norm, hidden_a, post_norm, hidden_b,
          1U, kAssistantHidden, kEpsilon, nullptr, stream);
      if (!status.ok()) return status;
      status = LaunchRmsNormBf16(hidden_b, layer.pre_feedforward_norm,
                                 normalized, 1U, kAssistantHidden, kEpsilon,
                                 stream);
      if (!status.ok()) return status;
      status = launch_gemv(normalized, layer.gate_projection, gate,
                           kIntermediate, kAssistantHidden);
      if (!status.ok()) return status;
      status = launch_gemv(normalized, layer.up_projection, up, kIntermediate,
                           kAssistantHidden);
      if (!status.ok()) return status;
      status = LaunchGeluTanhProduct(gate, up, product, kIntermediate, stream);
      if (!status.ok()) return status;
      status = round_bf16(product, kIntermediate);
      if (!status.ok()) return status;
      status = launch_gemv(product, layer.down_projection, projection,
                           kAssistantHidden, kIntermediate);
      if (!status.ok()) return status;
      status = LaunchRmsNormResidualBf16(
          projection, layer.post_feedforward_norm, hidden_b, post_norm,
          hidden_a, 1U, kAssistantHidden, kEpsilon, layer.layer_scalar,
          stream);
      if (!status.ok()) return status;
    }

    status = LaunchRmsNormBf16(hidden_a, impl_->bindings.final_norm,
                               normalized, 1U, kAssistantHidden, kEpsilon,
                               stream);
    if (!status.ok()) return status;
    status = launch_gemv(normalized, impl_->bindings.post_projection, feedback,
                         kBackboneHidden, kAssistantHidden);
    if (!status.ok()) return status;
    AssistantOutputCandidatesKernel<<<kOutputHeadBlocks, kThreads, 0, stream>>>(
        impl_->bindings.embedding, normalized, candidates);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
      return CudaFailure("launch assistant output candidates", error);
    }
    AssistantOutputArgmaxKernel<<<1U, kThreads, 0, stream>>>(
        candidates, selected, device_drafts, static_cast<std::uint32_t>(step));
    error = cudaGetLastError();
    if (error != cudaSuccess) {
      return CudaFailure("launch assistant output argmax", error);
    }
    backbone_hidden = feedback;
  }
  return Status::Ok();
}

Status AssistantModel::GenerateDrafts(
    const AssistantProposalContext& context,
    std::span<std::uint32_t> draft_token_ids, cudaStream_t stream) {
  Status status = GenerateDraftsDevice(
      context, static_cast<std::uint32_t>(draft_token_ids.size()), stream);
  if (!status.ok()) return status;
  cudaError_t error = cudaMemcpyAsync(
      draft_token_ids.data(), device_draft_tokens(), draft_token_ids.size_bytes(),
      cudaMemcpyDeviceToHost, stream);
  if (error != cudaSuccess) {
    return CudaFailure("copy assistant draft tokens", error);
  }
  error = cudaStreamSynchronize(stream);
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("synchronize assistant drafts", error);
}

const std::uint32_t* AssistantModel::device_draft_tokens() const {
  return prepared()
             ? impl_->Workspace<std::uint32_t>(impl_->offsets.draft_tokens)
             : nullptr;
}

bool AssistantModel::loaded() const { return impl_->arena != nullptr; }
bool AssistantModel::prepared() const { return impl_->workspace != nullptr; }
std::uint64_t AssistantModel::arena_bytes() const { return impl_->arena_size; }
std::uint64_t AssistantModel::workspace_bytes() const {
  return impl_->workspace_size;
}
std::uint64_t AssistantModel::source_bytes() const {
  return impl_->manifest.total_tensor_bytes;
}
std::uint64_t AssistantModel::tensor_count() const {
  return static_cast<std::uint64_t>(impl_->manifest.tensors.size());
}
const AssistantBindings& AssistantModel::bindings() const {
  return impl_->bindings;
}

}  // namespace gem16::internal
