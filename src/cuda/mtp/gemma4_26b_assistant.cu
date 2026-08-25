#include "cuda/mtp/gemma4_26b_assistant.h"

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

#include "cuda/attention/sm120.h"
#include "cuda/engine/gemma4_26b_artifact.h"
#include "cuda/fp8/reference.h"
#include "cuda/fp8/sm120.h"
#include "cuda/layer/reference.h"
#include "cuda/mtp/verify.h"
#include "cuda/nvfp4/reference.h"
#include "cuda/nvfp4/sm120.h"
#include "gem16/model.h"
#include "model/gemma4_26b_residency.h"

namespace gem16::internal {
namespace {

constexpr std::uint64_t kAssistantHidden = 1024U;
constexpr std::uint64_t kBackboneHidden = 2816U;
constexpr std::uint64_t kIntermediate = 8192U;
constexpr std::uint64_t kVocabulary = 262144U;
constexpr std::uint64_t kQueryHeads = 16U;
constexpr std::uint64_t kRowsPerTile = 8U;
constexpr std::uint64_t kKBlock = 64U;
constexpr std::uint64_t kNvfp4Group = 16U;
constexpr unsigned kThreads = 256U;
constexpr unsigned kArgmaxBlocks = 4096U;
constexpr float kEpsilon = 1.0e-6F;
constexpr std::uint32_t kSuppressedTokenA = 258883U;
constexpr std::uint32_t kSuppressedTokenB = 258882U;

Status Invalid(std::string message) {
  return Status(StatusCode::kInvalidArgument, std::move(message));
}

Status CudaFailure(const char* operation, cudaError_t error) {
  return Status(StatusCode::kInternal,
                std::string(operation) + ": " + cudaGetErrorName(error) +
                    ": " + cudaGetErrorString(error));
}

struct Fp8Matrix {
  const std::uint8_t* weight = nullptr;
  const std::uint16_t* scale = nullptr;
  std::uint64_t rows = 0U;
  std::uint64_t columns = 0U;
};

struct LayerBinding {
  bool global = false;
  std::uint64_t query_elements = 0U;
  const std::uint16_t* input_norm = nullptr;
  Fp8Matrix q;
  const std::uint16_t* q_norm = nullptr;
  Fp8Matrix o;
  const std::uint16_t* post_attention_norm = nullptr;
  const std::uint16_t* pre_feedforward_norm = nullptr;
  Gemma4MoeNvfp4Matrix gate;
  Gemma4MoeNvfp4Matrix up;
  Gemma4MoeNvfp4Matrix down;
  const std::uint16_t* post_feedforward_norm = nullptr;
  const std::uint16_t* layer_scalar = nullptr;
};

struct Bindings {
  Gemma4MoeNvfp4Matrix head;
  Fp8Matrix pre;
  Fp8Matrix post;
  const std::uint16_t* final_norm = nullptr;
  std::array<LayerBinding, 4> layers{};
};

struct LayoutBuilder {
  std::uint64_t bytes = 0U;
  template <typename T>
  std::uint64_t Add(std::uint64_t elements) {
    constexpr std::uint64_t alignment = 256U;
    const std::uint64_t mask = alignment - 1U;
    if (bytes > std::numeric_limits<std::uint64_t>::max() - mask) {
      bytes = std::numeric_limits<std::uint64_t>::max();
      return bytes;
    }
    bytes = (bytes + mask) & ~mask;
    const std::uint64_t offset = bytes;
    if (elements >
        (std::numeric_limits<std::uint64_t>::max() - bytes) / sizeof(T)) {
      bytes = std::numeric_limits<std::uint64_t>::max();
    } else {
      bytes += elements * sizeof(T);
    }
    return offset;
  }
};

struct Candidate {
  float value;
  std::uint32_t token;
};

__global__ void InitializeControlledAssistantKernel(
    const MtpDeviceControl* control, std::uint32_t* selected,
    DecodeControl* attention_control) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) return;
  selected[0] = control->current.input_token;
  *attention_control = {};
  attention_control->token = control->current.input_token;
  attention_control->position = control->current.processed_position;
}

__device__ bool Better(Candidate candidate, Candidate current) {
  return candidate.value > current.value ||
         (candidate.value == current.value && candidate.token < current.token);
}

__device__ std::uint64_t TiledPackedOffset(std::uint64_t row,
                                           std::uint64_t column,
                                           std::uint64_t k_blocks) {
  return (((row / kRowsPerTile) * k_blocks + column / kKBlock) *
               kRowsPerTile +
           row % kRowsPerTile) *
              (kKBlock / 2U) +
          (column % kKBlock) / 2U;
}

__device__ std::uint64_t TiledScaleOffset(std::uint64_t row,
                                          std::uint64_t column,
                                          std::uint64_t k_blocks) {
  return (((row / kRowsPerTile) * k_blocks + column / kKBlock) *
               kRowsPerTile +
           row % kRowsPerTile) *
              (kKBlock / kNvfp4Group) +
          (column % kKBlock) / kNvfp4Group;
}

__global__ void PrepareInputKernel(
    const std::uint8_t* target_packed, const std::uint8_t* target_scales,
    float target_weight_divisor, const std::uint32_t* selected,
    const float* target_hidden, float* concatenated) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= 2U * kBackboneHidden) return;
  if (index >= kBackboneHidden) {
    concatenated[index] = target_hidden[index - kBackboneHidden];
    return;
  }
  const std::uint64_t row = selected[0];
  const std::uint64_t k_blocks = kBackboneHidden / kKBlock;
  const std::uint8_t byte =
      target_packed[TiledPackedOffset(row, index, k_blocks)];
  __nv_fp4_e2m1 value;
  value.__x = static_cast<std::uint8_t>(
      (byte >> ((index & 1U) == 0U ? 0U : 4U)) & 0x0FU);
  __nv_fp8_e4m3 scale;
  scale.__x = target_scales[TiledScaleOffset(row, index, k_blocks)];
  const float embedding_scale = static_cast<float>(
      __float2bfloat16_rn(sqrtf(static_cast<float>(kBackboneHidden))));
  concatenated[index] = static_cast<float>(__float2bfloat16_rn(
      static_cast<float>(value) * static_cast<float>(scale) /
      target_weight_divisor * embedding_scale));
}

__global__ void RoundBf16Kernel(float* values, std::uint64_t elements) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    values[index] = static_cast<float>(__float2bfloat16_rn(values[index]));
  }
}

__global__ void SetControlKernel(DecodeControl* control,
                                 std::uint32_t token,
                                 std::uint64_t position) {
  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    *control = DecodeControl{token, 0U, position, position};
  }
}

__global__ void ArgmaxCandidatesKernel(const float* logits,
                                       Candidate* candidates) {
  __shared__ Candidate scratch[kThreads];
  Candidate best{-FLT_MAX, 0U};
  for (std::uint64_t token =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       token < kVocabulary;
       token += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
    if (token == kSuppressedTokenA || token == kSuppressedTokenB) continue;
    const float value = logits[token];
    if (isfinite(value)) {
      const Candidate candidate{value, static_cast<std::uint32_t>(token)};
      if (Better(candidate, best)) best = candidate;
    }
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
  if (threadIdx.x == 0U) candidates[blockIdx.x] = scratch[0];
}

__global__ void ArgmaxFinalKernel(const Candidate* candidates,
                                  std::uint32_t* selected,
                                  std::uint32_t* drafts,
                                  std::uint32_t draft_index) {
  __shared__ Candidate scratch[kThreads];
  Candidate best{-FLT_MAX, 0U};
  for (std::uint32_t index = threadIdx.x; index < kArgmaxBlocks;
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
    drafts[draft_index] = scratch[0].token;
  }
}

template <typename T>
Result<const T*> Pointer(const Gemma4Moe26BDeviceArtifact& artifact,
                         const std::string& name) {
  auto pointer = artifact.Pointer(name);
  if (!pointer.ok()) return pointer.status();
  return reinterpret_cast<const T*>(pointer.value());
}

Result<Fp8Matrix> BindFp8(const Gemma4Moe26BDeviceArtifact& artifact,
                         const std::string& module, std::uint64_t rows,
                         std::uint64_t columns) {
  auto weight = Pointer<std::uint8_t>(artifact, module + ".weight");
  auto scale = Pointer<std::uint16_t>(artifact, module + ".weight_scale");
  if (!weight.ok()) return weight.status();
  if (!scale.ok()) return scale.status();
  return Fp8Matrix{weight.value(), scale.value(), rows, columns};
}

Result<Gemma4MoeNvfp4Matrix> BindNvfp4(
    const Gemma4Moe26BDeviceArtifact& artifact, const std::string& module,
    std::uint64_t rows, std::uint64_t columns) {
  auto packed = Pointer<std::uint8_t>(artifact, module + ".weight_packed");
  auto scales = Pointer<std::uint8_t>(artifact, module + ".weight_scale");
  auto activation = artifact.HostFloat32(module + ".input_global_scale");
  auto weight = artifact.HostFloat32(module + ".weight_global_scale");
  if (!packed.ok()) return packed.status();
  if (!scales.ok()) return scales.status();
  if (!activation.ok()) return activation.status();
  if (!weight.ok()) return weight.status();
  return Gemma4MoeNvfp4Matrix{packed.value(), scales.value(), rows, columns,
                              activation.value(), weight.value()};
}

}  // namespace

struct Gemma4Moe26BAssistantModel::Impl {
  struct Offsets {
    std::uint64_t concatenated = 0U;
    std::uint64_t hidden_a = 0U;
    std::uint64_t hidden_b = 0U;
    std::uint64_t normalized = 0U;
    std::uint64_t query = 0U;
    std::uint64_t normalized_query = 0U;
    std::uint64_t attention = 0U;
    std::uint64_t projection = 0U;
    std::uint64_t post_norm = 0U;
    std::uint64_t product = 0U;
    std::uint64_t feedback = 0U;
    std::uint64_t fp8_activation = 0U;
    std::uint64_t fp8_scale = 0U;
    std::uint64_t nvfp4_activation = 0U;
    std::uint64_t nvfp4_scales = 0U;
    std::uint64_t attention_control = 0U;
    std::uint64_t scores = 0U;
    std::uint64_t logits = 0U;
    std::uint64_t candidates = 0U;
    std::uint64_t selected = 0U;
    std::uint64_t drafts = 0U;
  } offsets;

  ~Impl() {
    if (workspace != nullptr) (void)cudaFree(workspace);
  }

  template <typename T>
  T* Workspace(std::uint64_t offset) const {
    return reinterpret_cast<T*>(static_cast<std::byte*>(workspace) + offset);
  }

  std::filesystem::path directory;
  ModelManifest manifest;
  Gemma4Moe26BResidencyPlan residency;
  Gemma4Moe26BDeviceArtifact artifact;
  Bindings bindings;
  void* workspace = nullptr;
  std::uint64_t workspace_bytes = 0U;
  std::uint64_t max_context = 0U;
};

Gemma4Moe26BAssistantModel::Gemma4Moe26BAssistantModel()
    : impl_(std::make_unique<Impl>()) {}
Gemma4Moe26BAssistantModel::~Gemma4Moe26BAssistantModel() = default;
Gemma4Moe26BAssistantModel::Gemma4Moe26BAssistantModel(
    Gemma4Moe26BAssistantModel&&) noexcept = default;
Gemma4Moe26BAssistantModel& Gemma4Moe26BAssistantModel::operator=(
    Gemma4Moe26BAssistantModel&&) noexcept = default;

Status Gemma4Moe26BAssistantModel::Load(
    const std::filesystem::path& directory) {
  if (loaded()) return Invalid("M25 Assistant is already loaded");
  auto inspected = InspectCheckpoint({directory, true});
  if (!inspected.ok()) return inspected.status();
  auto residency =
      BuildGemma4Moe26BAssistantResidencyPlan(inspected.value());
  if (!residency.ok()) return residency.status();
  auto artifact = Gemma4Moe26BDeviceArtifact::Load(
      directory, inspected.value(), residency.value());
  if (!artifact.ok()) return artifact.status();

  Bindings bindings;
  auto head = BindNvfp4(artifact.value(), "model.embed_tokens", kVocabulary,
                        kAssistantHidden);
  auto pre = BindFp8(artifact.value(), "pre_projection", kAssistantHidden,
                     2U * kBackboneHidden);
  auto post = BindFp8(artifact.value(), "post_projection", kBackboneHidden,
                      kAssistantHidden);
  auto norm = Pointer<std::uint16_t>(artifact.value(), "model.norm.weight");
  if (!head.ok()) return head.status();
  if (!pre.ok()) return pre.status();
  if (!post.ok()) return post.status();
  if (!norm.ok()) return norm.status();
  bindings.head = head.value();
  bindings.pre = pre.value();
  bindings.post = post.value();
  bindings.final_norm = norm.value();

  for (std::size_t index = 0; index < bindings.layers.size(); ++index) {
    auto& layer = bindings.layers[index];
    layer.global = index + 1U == bindings.layers.size();
    layer.query_elements = layer.global ? 8192U : 4096U;
    const std::string base = "model.layers." + std::to_string(index) + ".";
    auto input = Pointer<std::uint16_t>(
        artifact.value(), base + "input_layernorm.weight");
    auto q = BindFp8(artifact.value(), base + "self_attn.q_proj",
                     layer.query_elements, kAssistantHidden);
    auto q_norm = Pointer<std::uint16_t>(
        artifact.value(), base + "self_attn.q_norm.weight");
    auto o = BindFp8(artifact.value(), base + "self_attn.o_proj",
                     kAssistantHidden, layer.query_elements);
    auto post_attention = Pointer<std::uint16_t>(
        artifact.value(), base + "post_attention_layernorm.weight");
    auto pre_feedforward = Pointer<std::uint16_t>(
        artifact.value(), base + "pre_feedforward_layernorm.weight");
    auto gate = BindNvfp4(artifact.value(), base + "mlp.gate_proj",
                          kIntermediate, kAssistantHidden);
    auto up = BindNvfp4(artifact.value(), base + "mlp.up_proj", kIntermediate,
                        kAssistantHidden);
    auto down = BindNvfp4(artifact.value(), base + "mlp.down_proj",
                          kAssistantHidden, kIntermediate);
    auto post_feedforward = Pointer<std::uint16_t>(
        artifact.value(), base + "post_feedforward_layernorm.weight");
    auto scalar = Pointer<std::uint16_t>(artifact.value(),
                                         base + "layer_scalar");
    if (!input.ok()) return input.status();
    if (!q.ok()) return q.status();
    if (!q_norm.ok()) return q_norm.status();
    if (!o.ok()) return o.status();
    if (!post_attention.ok()) return post_attention.status();
    if (!pre_feedforward.ok()) return pre_feedforward.status();
    if (!gate.ok()) return gate.status();
    if (!up.ok()) return up.status();
    if (!down.ok()) return down.status();
    if (!post_feedforward.ok()) return post_feedforward.status();
    if (!scalar.ok()) return scalar.status();
    layer.input_norm = input.value();
    layer.q = q.value();
    layer.q_norm = q_norm.value();
    layer.o = o.value();
    layer.post_attention_norm = post_attention.value();
    layer.pre_feedforward_norm = pre_feedforward.value();
    layer.gate = gate.value();
    layer.up = up.value();
    layer.down = down.value();
    layer.post_feedforward_norm = post_feedforward.value();
    layer.layer_scalar = scalar.value();
  }
  impl_->directory = directory;
  impl_->manifest = std::move(inspected).value();
  impl_->residency = std::move(residency).value();
  impl_->artifact = std::move(artifact).value();
  impl_->bindings = bindings;
  return Status::Ok();
}

Status Gemma4Moe26BAssistantModel::Prepare(std::uint64_t max_context) {
  if (!loaded() || prepared() || max_context == 0U || max_context > 65536U) {
    return Invalid("M25 Assistant workspace context is invalid");
  }
  LayoutBuilder layout;
  impl_->offsets.concatenated = layout.Add<float>(2U * kBackboneHidden);
  impl_->offsets.hidden_a = layout.Add<float>(kAssistantHidden);
  impl_->offsets.hidden_b = layout.Add<float>(kAssistantHidden);
  impl_->offsets.normalized = layout.Add<float>(kAssistantHidden);
  impl_->offsets.query = layout.Add<float>(kQueryHeads * 512U);
  impl_->offsets.normalized_query = layout.Add<float>(kQueryHeads * 512U);
  impl_->offsets.attention = layout.Add<float>(kQueryHeads * 512U);
  impl_->offsets.projection = layout.Add<float>(kAssistantHidden);
  impl_->offsets.post_norm = layout.Add<float>(kAssistantHidden);
  impl_->offsets.product = layout.Add<float>(kIntermediate);
  impl_->offsets.feedback = layout.Add<float>(kBackboneHidden);
  impl_->offsets.fp8_activation = layout.Add<std::uint8_t>(kIntermediate);
  impl_->offsets.fp8_scale = layout.Add<float>(1U);
  impl_->offsets.nvfp4_activation =
      layout.Add<std::uint8_t>(kIntermediate / 2U);
  impl_->offsets.nvfp4_scales =
      layout.Add<std::uint8_t>(kIntermediate / kNvfp4Group);
  impl_->offsets.attention_control = layout.Add<DecodeControl>(1U);
  const std::uint64_t score_elements = std::max(
      kQueryHeads * max_context, DecodeAttentionWorkspaceElements(max_context));
  impl_->offsets.scores = layout.Add<float>(score_elements);
  impl_->offsets.logits = layout.Add<float>(kVocabulary);
  impl_->offsets.candidates = layout.Add<Candidate>(kArgmaxBlocks);
  impl_->offsets.selected = layout.Add<std::uint32_t>(1U);
  impl_->offsets.drafts = layout.Add<std::uint32_t>(4U);
  if (layout.bytes == 0U ||
      layout.bytes > impl_->residency.fixed_regions.front().bytes ||
      layout.bytes > std::numeric_limits<std::size_t>::max()) {
    return Status(StatusCode::kResourceExhausted,
                  "M25 Assistant workspace exceeds its fixed residency region");
  }
  const auto error = cudaMalloc(&impl_->workspace,
                                static_cast<std::size_t>(layout.bytes));
  if (error != cudaSuccess) {
    return CudaFailure("allocate M25 Assistant workspace", error);
  }
  impl_->workspace_bytes = layout.bytes;
  impl_->max_context = max_context;
  return Status::Ok();
}

Status Gemma4Moe26BAssistantModel::GenerateDraftsDevice(
    const Gemma4Moe26BAssistantProposalContext& context,
    std::uint32_t draft_count, cudaStream_t stream,
    const MtpDeviceControl* control) {
  if (!prepared() || stream == nullptr || draft_count == 0U ||
      draft_count > 4U ||
      (control == nullptr &&
       (context.input_token >= kVocabulary ||
        context.position >= impl_->max_context)) ||
      context.target_hidden == nullptr ||
      context.target_embedding.packed_e2m1 == nullptr ||
      context.target_embedding.scales_e4m3fn == nullptr ||
      context.target_embedding.rows != kVocabulary ||
      context.target_embedding.columns != kBackboneHidden ||
      !std::isfinite(context.target_embedding.weight_global_divisor) ||
      context.target_embedding.weight_global_divisor <= 0.0F) {
    return Invalid("M25 Assistant proposal context is invalid");
  }
  const auto validate_kv = [this](const AssistantSharedKvView& view,
                                  std::uint64_t heads,
                                  std::uint64_t dimension) -> Status {
    if (view.mode != AssistantKvCacheMode::kCheckpointFp8 ||
        view.key_fp8 == nullptr || view.value_fp8 == nullptr ||
        view.key_scale_bf16 == nullptr || view.value_scale_bf16 == nullptr ||
        view.tokens == 0U || view.tokens > view.capacity ||
        view.capacity > impl_->max_context || view.kv_heads != heads ||
        view.head_dimension != dimension) {
      return Invalid("M25 Assistant shared target-KV geometry is invalid");
    }
    return Status::Ok();
  };
  Status status = validate_kv(context.sliding_kv, 8U, 256U);
  if (!status.ok()) return status;
  status = validate_kv(context.full_kv, 2U, 512U);
  if (!status.ok()) return status;

  float* concatenated =
      impl_->Workspace<float>(impl_->offsets.concatenated);
  float* hidden_a = impl_->Workspace<float>(impl_->offsets.hidden_a);
  float* hidden_b = impl_->Workspace<float>(impl_->offsets.hidden_b);
  float* normalized = impl_->Workspace<float>(impl_->offsets.normalized);
  float* query = impl_->Workspace<float>(impl_->offsets.query);
  float* normalized_query =
      impl_->Workspace<float>(impl_->offsets.normalized_query);
  float* attention = impl_->Workspace<float>(impl_->offsets.attention);
  float* projection = impl_->Workspace<float>(impl_->offsets.projection);
  float* post_norm = impl_->Workspace<float>(impl_->offsets.post_norm);
  float* product = impl_->Workspace<float>(impl_->offsets.product);
  float* feedback = impl_->Workspace<float>(impl_->offsets.feedback);
  auto* fp8_activation =
      impl_->Workspace<std::uint8_t>(impl_->offsets.fp8_activation);
  float* fp8_scale = impl_->Workspace<float>(impl_->offsets.fp8_scale);
  auto* nvfp4_activation =
      impl_->Workspace<std::uint8_t>(impl_->offsets.nvfp4_activation);
  auto* nvfp4_scales =
      impl_->Workspace<std::uint8_t>(impl_->offsets.nvfp4_scales);
  auto* attention_control =
      impl_->Workspace<DecodeControl>(impl_->offsets.attention_control);
  float* scores = impl_->Workspace<float>(impl_->offsets.scores);
  float* logits = impl_->Workspace<float>(impl_->offsets.logits);
  auto* candidates = impl_->Workspace<Candidate>(impl_->offsets.candidates);
  auto* selected =
      impl_->Workspace<std::uint32_t>(impl_->offsets.selected);
  auto* drafts = impl_->Workspace<std::uint32_t>(impl_->offsets.drafts);

  const auto round = [stream](float* values,
                              std::uint64_t elements) -> Status {
    const std::uint64_t blocks = (elements + kThreads - 1U) / kThreads;
    RoundBf16Kernel<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
        values, elements);
    const auto error = cudaGetLastError();
    return error == cudaSuccess
               ? Status::Ok()
               : CudaFailure("round M25 Assistant BF16 boundary", error);
  };
  const auto fp8 = [&](const float* input, const Fp8Matrix& matrix,
                       float* output) -> Status {
    Status projected = LaunchFp8ReferenceTokenQuantization(
        input, fp8_activation, fp8_scale, matrix.columns, stream);
    if (!projected.ok()) return projected;
    projected = LaunchFp8Sm120DirectProjection(
        fp8_activation, fp8_scale, matrix.weight, matrix.scale, output,
        matrix.rows, matrix.columns, stream);
    if (!projected.ok()) return projected;
    return round(output, matrix.rows);
  };
  const auto quantize_nvfp4 = [&](const float* input,
                                  const Gemma4MoeNvfp4Matrix& matrix) {
    return LaunchNvfp4ReferenceActivationQuantization(
        input, nvfp4_activation, nvfp4_scales, matrix.columns,
        matrix.activation_global_divisor, stream);
  };
  const auto project_nvfp4 = [&](const Gemma4MoeNvfp4Matrix& matrix,
                                 float* output) {
    return LaunchNvfp4Sm120DirectProjectionBf16Float(
        nvfp4_activation, nvfp4_scales, matrix.packed_e2m1,
        matrix.scales_e4m3fn, output, matrix.rows, matrix.columns,
        matrix.activation_global_divisor, matrix.weight_global_divisor,
        stream);
  };

  cudaError_t error = cudaSuccess;
  if (control == nullptr) {
    error = cudaMemcpyAsync(selected, &context.input_token,
                            sizeof(context.input_token),
                            cudaMemcpyHostToDevice, stream);
    if (error != cudaSuccess) {
      return CudaFailure("initialize M25 Assistant token", error);
    }
    SetControlKernel<<<1U, 1U, 0, stream>>>(
        attention_control, context.input_token, context.position);
  } else {
    InitializeControlledAssistantKernel<<<1U, 1U, 0, stream>>>(
        control, selected, attention_control);
  }
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaFailure("set M25 Assistant constant position", error);
  }

  const float* backbone_hidden = context.target_hidden;
  for (std::uint32_t step = 0; step < draft_count; ++step) {
    constexpr std::uint64_t kInputElements = 2U * kBackboneHidden;
    constexpr unsigned kInputBlocks = static_cast<unsigned>(
        (kInputElements + kThreads - 1U) / kThreads);
    PrepareInputKernel<<<kInputBlocks, kThreads, 0, stream>>>(
        context.target_embedding.packed_e2m1,
        context.target_embedding.scales_e4m3fn,
        context.target_embedding.weight_global_divisor, selected,
        backbone_hidden, concatenated);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
      return CudaFailure("prepare M25 Assistant target input", error);
    }
    status = fp8(concatenated, impl_->bindings.pre, hidden_a);
    if (!status.ok()) return status;

    for (const auto& layer : impl_->bindings.layers) {
      const auto& kv = layer.global ? context.full_kv : context.sliding_kv;
      status = LaunchRmsNormBf16(hidden_a, layer.input_norm, normalized, 1U,
                                 kAssistantHidden, kEpsilon, stream);
      if (!status.ok()) return status;
      status = fp8(normalized, layer.q, query);
      if (!status.ok()) return status;
      status = LaunchRmsNormBf16(
          query, layer.q_norm, normalized_query, kQueryHeads,
          layer.global ? 512U : 256U, kEpsilon, stream);
      if (!status.ok()) return status;
      if (layer.global) {
        status = control == nullptr
                     ? LaunchProportionalRotaryEmbedding(
                           normalized_query, kQueryHeads, 512U, 0.25,
                           context.position, 1000000.0, 1.0, stream)
                     : LaunchProportionalRotaryEmbeddingControlled(
                           normalized_query, kQueryHeads, 512U, 0.25,
                           attention_control, 1000000.0, 1.0, stream);
      } else {
        status = control == nullptr
                     ? LaunchRotaryEmbedding(
                           normalized_query, kQueryHeads, 256U, 256U,
                           context.position, 10000.0, stream)
                     : LaunchRotaryEmbeddingControlled(
                           normalized_query, kQueryHeads, 256U, 256U,
                           attention_control, 10000.0, stream);
      }
      if (!status.ok()) return status;
      status = round(normalized_query, layer.query_elements);
      if (!status.ok()) return status;
      status = LaunchOnlineAttentionDecodeFp8Sm120(
          normalized_query, kv.key_fp8, kv.value_fp8, kv.key_scale_bf16,
          kv.value_scale_bf16, scores, attention, attention_control,
          kQueryHeads, kv.kv_heads, kv.head_dimension, kv.capacity,
          !layer.global, stream);
      if (!status.ok()) return status;
      status = round(attention, layer.query_elements);
      if (!status.ok()) return status;
      status = fp8(attention, layer.o, projection);
      if (!status.ok()) return status;
      status = LaunchRmsNormResidualBf16(
          projection, layer.post_attention_norm, hidden_a, post_norm, hidden_b,
          1U, kAssistantHidden, kEpsilon, nullptr, stream);
      if (!status.ok()) return status;
      status = LaunchRmsNormBf16(hidden_b, layer.pre_feedforward_norm,
                                 normalized, 1U, kAssistantHidden, kEpsilon,
                                 stream);
      if (!status.ok()) return status;
      status = quantize_nvfp4(normalized, layer.gate);
      if (!status.ok()) return status;
      if (layer.gate.activation_global_divisor !=
          layer.up.activation_global_divisor) {
        return Status(StatusCode::kDataLoss,
                      "M25 fused Gate/Up activation divisors disagree");
      }
      status = LaunchNvfp4Sm120FusedGateUp(
          nvfp4_activation, nvfp4_scales, layer.gate.packed_e2m1,
          layer.gate.scales_e4m3fn, layer.up.packed_e2m1,
          layer.up.scales_e4m3fn, nullptr, nullptr, product, kIntermediate,
          kAssistantHidden, layer.gate.activation_global_divisor,
          layer.gate.weight_global_divisor,
          layer.up.activation_global_divisor,
          layer.up.weight_global_divisor, stream);
      if (!status.ok()) return status;
      status = quantize_nvfp4(product, layer.down);
      if (!status.ok()) return status;
      status = project_nvfp4(layer.down, projection);
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
    status = fp8(normalized, impl_->bindings.post, feedback);
    if (!status.ok()) return status;
    status = quantize_nvfp4(normalized, impl_->bindings.head);
    if (!status.ok()) return status;
    status = project_nvfp4(impl_->bindings.head, logits);
    if (!status.ok()) return status;
    ArgmaxCandidatesKernel<<<kArgmaxBlocks, kThreads, 0, stream>>>(logits,
                                                                  candidates);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
      return CudaFailure("launch M25 Assistant argmax candidates", error);
    }
    ArgmaxFinalKernel<<<1U, kThreads, 0, stream>>>(candidates, selected, drafts,
                                                   step);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
      return CudaFailure("launch M25 Assistant argmax final", error);
    }
    backbone_hidden = feedback;
  }
  return Status::Ok();
}

Status Gemma4Moe26BAssistantModel::GenerateDrafts(
    const Gemma4Moe26BAssistantProposalContext& context,
    std::span<std::uint32_t> draft_token_ids, cudaStream_t stream) {
  Status status = GenerateDraftsDevice(
      context, static_cast<std::uint32_t>(draft_token_ids.size()), stream);
  if (!status.ok()) return status;
  const auto error = cudaMemcpyAsync(
      draft_token_ids.data(), device_draft_tokens(),
      draft_token_ids.size_bytes(), cudaMemcpyDeviceToHost, stream);
  if (error != cudaSuccess) {
    return CudaFailure("copy M25 Assistant draft tokens", error);
  }
  const auto synchronized = cudaStreamSynchronize(stream);
  return synchronized == cudaSuccess
             ? Status::Ok()
             : CudaFailure("synchronize M25 Assistant drafts", synchronized);
}

Status Gemma4Moe26BAssistantModel::CopyLastOracleState(
    std::span<float> concatenated_input, std::span<float> logits,
    cudaStream_t stream) const {
  if (!prepared() || stream == nullptr ||
      concatenated_input.size() != 2U * kBackboneHidden ||
      logits.size() != kVocabulary) {
    return Invalid("M25 Assistant oracle copy geometry is invalid");
  }
  auto error = cudaMemcpyAsync(
      concatenated_input.data(),
      impl_->Workspace<float>(impl_->offsets.concatenated),
      concatenated_input.size_bytes(), cudaMemcpyDeviceToHost, stream);
  if (error == cudaSuccess) {
    error = cudaMemcpyAsync(logits.data(),
                            impl_->Workspace<float>(impl_->offsets.logits),
                            logits.size_bytes(), cudaMemcpyDeviceToHost,
                            stream);
  }
  if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("copy M25 Assistant oracle state", error);
}

const std::uint32_t* Gemma4Moe26BAssistantModel::device_draft_tokens() const {
  return prepared()
             ? impl_->Workspace<std::uint32_t>(impl_->offsets.drafts)
             : nullptr;
}

bool Gemma4Moe26BAssistantModel::loaded() const {
  return impl_ != nullptr && impl_->artifact.arena() != nullptr;
}
bool Gemma4Moe26BAssistantModel::prepared() const {
  return impl_ != nullptr && impl_->workspace != nullptr;
}
std::uint64_t Gemma4Moe26BAssistantModel::arena_bytes() const {
  return loaded() ? impl_->artifact.arena_bytes() : 0U;
}
std::uint64_t Gemma4Moe26BAssistantModel::workspace_bytes() const {
  return prepared() ? impl_->workspace_bytes : 0U;
}
std::uint64_t Gemma4Moe26BAssistantModel::source_bytes() const {
  return loaded() ? impl_->manifest.total_tensor_bytes : 0U;
}
std::uint64_t Gemma4Moe26BAssistantModel::tensor_count() const {
  return loaded() ? static_cast<std::uint64_t>(impl_->manifest.tensors.size())
                  : 0U;
}

}  // namespace gem16::internal
