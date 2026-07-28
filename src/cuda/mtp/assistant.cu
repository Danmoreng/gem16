#include "cuda/mtp/assistant.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <limits>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

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

  ~Impl() {
    if (arena != nullptr) (void)cudaFree(arena);
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
  void* arena = nullptr;
  std::uint64_t arena_size = 0U;
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

bool AssistantModel::loaded() const { return impl_->arena != nullptr; }
std::uint64_t AssistantModel::arena_bytes() const { return impl_->arena_size; }
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
