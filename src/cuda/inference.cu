#include "gem16gb/engine.h"

#include "cuda/attention/sm120.h"
#include "cuda/fp8/reference.h"
#include "cuda/fp8/sm120.h"
#include "cuda/layer/reference.h"
#include "cuda/nvfp4/mlp.h"
#include "cuda/nvfp4/reference.h"
#include "cuda/nvfp4/sm120.h"
#include "cuda/nvfp4/sm120_layout.h"
#include "gem16gb/model.h"
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

namespace gem16gb {
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
constexpr float kEpsilon = 1.0e-6F;
constexpr unsigned kThreads = 256;
constexpr unsigned kFusedOutputHeadBlocks = 4096;
constexpr std::uint64_t kDefaultPrefillChunkTokens = 1024;
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
      std::min(kDefaultPrefillChunkTokens, budget_tokens);
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

struct DeviceTensor {
  const TensorInfo* info = nullptr;
  std::byte* data = nullptr;
  float scalar_f32 = 0.0F;
  bool has_scalar_f32 = false;
  bool sm120_scale_tiled = false;
};

struct Fp8Binding {
  const std::uint8_t* weight = nullptr;
  const std::uint16_t* scales = nullptr;
  std::uint64_t rows = 0;
  std::uint64_t contracting = 0;
};

struct Nvfp4Binding {
  const std::uint8_t* packed_weight = nullptr;
  const std::uint8_t* scales = nullptr;
  float input_divisor = 0.0F;
  float weight_divisor = 0.0F;
  std::uint64_t rows = 0;
  std::uint64_t contracting = 0;
};

struct LayerBinding {
  bool global = false;
  std::uint64_t kv_heads = 0;
  std::uint64_t head_dimension = 0;
  std::uint64_t query_elements = 0;
  std::uint64_t kv_elements = 0;
  Fp8Binding q;
  Fp8Binding k;
  Fp8Binding v;
  Fp8Binding o;
  Nvfp4Binding gate;
  Nvfp4Binding up;
  Nvfp4Binding down;
  const std::uint16_t* input_norm = nullptr;
  const std::uint16_t* q_norm = nullptr;
  const std::uint16_t* k_norm = nullptr;
  const std::uint16_t* post_attention_norm = nullptr;
  const std::uint16_t* pre_mlp_norm = nullptr;
  const std::uint16_t* post_mlp_norm = nullptr;
  const std::uint16_t* layer_scalar = nullptr;
  const std::uint16_t* k_cache_scale = nullptr;
  const std::uint16_t* v_cache_scale = nullptr;
  float* key_cache_bf16 = nullptr;
  float* value_cache_bf16 = nullptr;
  std::uint8_t* key_cache_fp8 = nullptr;
  std::uint8_t* value_cache_fp8 = nullptr;
};

class LoadedModel {
 public:
  [[nodiscard]] Status Load(const std::filesystem::path& directory) {
    auto inspected = InspectCheckpoint({directory, true});
    if (!inspected.ok()) return inspected.status();
    manifest_ = std::move(inspected).value();

    std::uint64_t arena_bytes = 0;
    for (const auto& tensor : manifest_.tensors) {
      if (!tensor.loaded_in_text_only_mode) continue;
      auto aligned = AlignUp(arena_bytes, kAlignment);
      if (!aligned.ok()) return aligned.status();
      if (tensor.byte_length > std::numeric_limits<std::uint64_t>::max() - aligned.value()) {
        return Error(StatusCode::kInternal, "weight arena size overflow");
      }
      arena_bytes = aligned.value() + tensor.byte_length;
    }
    auto final_size = AlignUp(arena_bytes, kAlignment);
    if (!final_size.ok()) return final_size.status();
    Status status = weights_.Allocate(final_size.value(), "allocate text-only weight arena");
    if (!status.ok()) return status;

    std::uint64_t offset = 0;
    for (const auto& tensor : manifest_.tensors) {
      if (!tensor.loaded_in_text_only_mode) continue;
      auto aligned = AlignUp(offset, kAlignment);
      if (!aligned.ok()) return aligned.status();
      DeviceTensor view;
      view.info = &tensor;
      view.data = weights_.data() + aligned.value();
      const auto inserted = tensors_.emplace(tensor.name, view);
      if (!inserted.second) {
        return Error(StatusCode::kDataLoss, "duplicate device tensor: " + tensor.name);
      }
      offset = aligned.value() + tensor.byte_length;
    }

    std::unordered_set<std::string> shards;
    for (const auto& tensor : manifest_.tensors) {
      if (tensor.loaded_in_text_only_mode) shards.insert(tensor.source_shard);
    }
    for (const auto& shard : shards) {
      auto mapped = internal::MappedFile::Open(directory / shard);
      if (!mapped.ok()) return mapped.status();
      for (auto& [name, view] : tensors_) {
        (void)name;
        const TensorInfo& tensor = *view.info;
        if (tensor.source_shard != shard) continue;
        if (tensor.byte_offset > mapped.value().size() ||
            tensor.byte_length > mapped.value().size() - tensor.byte_offset) {
          return Error(StatusCode::kDataLoss, "tensor upload range is invalid: " + tensor.name);
        }
        const std::byte* source = mapped.value().data() + tensor.byte_offset;
        std::vector<std::uint8_t> tiled_scales;
        if (tensor.quantization_class == "NVFP4_LOCAL_SCALE_E4M3") {
          if (tensor.shape.size() != 2U ||
              tensor.shape[1] > std::numeric_limits<std::uint64_t>::max() / 16U) {
            return Error(StatusCode::kDataLoss,
                         "invalid NVFP4 local-scale geometry: " + tensor.name);
          }
          const auto layout = internal::PlanSm120Nvfp4SourceLayout(
              tensor.shape[0], tensor.shape[1] * 16U);
          if (!layout.ok()) return layout.status();
          const auto tiled = internal::TileSm120Nvfp4WeightScales(
              layout.value(),
              std::span<const std::uint8_t>(
                  reinterpret_cast<const std::uint8_t*>(source),
                  static_cast<std::size_t>(tensor.byte_length)));
          if (!tiled.ok()) return tiled.status();
          tiled_scales = std::move(tiled).value();
          source = reinterpret_cast<const std::byte*>(tiled_scales.data());
          view.sm120_scale_tiled = true;
        }
        const cudaError_t error = cudaMemcpy(view.data, source,
                                             static_cast<std::size_t>(tensor.byte_length),
                                             cudaMemcpyHostToDevice);
        if (error != cudaSuccess) return CudaFailure("upload checkpoint tensor", error);
        if (tensor.storage_dtype == "F32" && tensor.byte_length == sizeof(float)) {
          std::uint32_t bits = 0;
          std::memcpy(&bits, source, sizeof(bits));
          view.scalar_f32 = std::bit_cast<float>(bits);
          view.has_scalar_f32 = true;
        }
      }
    }
    return Bind();
  }

  [[nodiscard]] const std::array<LayerBinding, kLayers>& layers() const { return layers_; }
  [[nodiscard]] const std::uint16_t* embedding() const { return embedding_; }
  [[nodiscard]] const std::uint16_t* final_norm() const { return final_norm_; }
  [[nodiscard]] std::uint64_t weight_bytes() const { return weights_.bytes(); }

  void SetLayerBf16Cache(std::size_t layer, float* key, float* value) {
    layers_[layer].key_cache_bf16 = key;
    layers_[layer].value_cache_bf16 = value;
  }

  void SetLayerFp8Cache(std::size_t layer, std::uint8_t* key,
                        std::uint8_t* value) {
    layers_[layer].key_cache_fp8 = key;
    layers_[layer].value_cache_fp8 = value;
  }

 private:
  [[nodiscard]] Result<const DeviceTensor*> Tensor(const std::string& name) const {
    const auto found = tensors_.find(name);
    if (found == tensors_.end()) {
      return Error(StatusCode::kNotFound, "required inference tensor is missing: " + name);
    }
    return &found->second;
  }

  [[nodiscard]] Result<const std::uint16_t*> Bf16(const std::string& name,
                                                  std::uint64_t elements) const {
    auto tensor = Tensor(name);
    if (!tensor.ok()) return tensor.status();
    if (tensor.value()->info->storage_dtype != "BF16" ||
        tensor.value()->info->byte_length != elements * sizeof(std::uint16_t)) {
      return Error(StatusCode::kDataLoss, "unexpected BF16 tensor geometry: " + name);
    }
    return reinterpret_cast<const std::uint16_t*>(tensor.value()->data);
  }

  [[nodiscard]] Result<Fp8Binding> Fp8(const std::string& name, std::uint64_t rows,
                                       std::uint64_t contracting) const {
    auto weight = Tensor(name + ".weight");
    auto scales = Tensor(name + ".weight_scale");
    if (!weight.ok()) return weight.status();
    if (!scales.ok()) return scales.status();
    if (weight.value()->info->storage_dtype != "F8_E4M3" ||
        weight.value()->info->shape != std::vector<std::uint64_t>{rows, contracting} ||
        scales.value()->info->storage_dtype != "BF16" ||
        scales.value()->info->shape != std::vector<std::uint64_t>{rows, 1U}) {
      return Error(StatusCode::kDataLoss, "unexpected FP8 tensor geometry: " + name);
    }
    return Fp8Binding{reinterpret_cast<const std::uint8_t*>(weight.value()->data),
                      reinterpret_cast<const std::uint16_t*>(scales.value()->data), rows,
                      contracting};
  }

  [[nodiscard]] Result<Nvfp4Binding> Nvfp4(const std::string& name, std::uint64_t rows,
                                           std::uint64_t contracting) const {
    auto packed = Tensor(name + ".weight_packed");
    auto scales = Tensor(name + ".weight_scale");
    auto input = Tensor(name + ".input_global_scale");
    auto weight = Tensor(name + ".weight_global_scale");
    if (!packed.ok()) return packed.status();
    if (!scales.ok()) return scales.status();
    if (!input.ok()) return input.status();
    if (!weight.ok()) return weight.status();
    if (packed.value()->info->storage_dtype != "U8" ||
        packed.value()->info->logical_shape != std::vector<std::uint64_t>{rows, contracting} ||
        scales.value()->info->storage_dtype != "F8_E4M3" ||
        scales.value()->info->shape != std::vector<std::uint64_t>{rows, contracting / 16U} ||
        !scales.value()->sm120_scale_tiled ||
        !input.value()->has_scalar_f32 || !weight.value()->has_scalar_f32 ||
        !std::isfinite(input.value()->scalar_f32) || input.value()->scalar_f32 <= 0.0F ||
        !std::isfinite(weight.value()->scalar_f32) || weight.value()->scalar_f32 <= 0.0F) {
      return Error(StatusCode::kDataLoss, "unexpected NVFP4 tensor family: " + name);
    }
    return Nvfp4Binding{reinterpret_cast<const std::uint8_t*>(packed.value()->data),
                        reinterpret_cast<const std::uint8_t*>(scales.value()->data),
                        input.value()->scalar_f32, weight.value()->scalar_f32, rows,
                        contracting};
  }

  [[nodiscard]] Status Bind() {
    auto embedding = Bf16("model.language_model.embed_tokens.weight", kVocabulary * kHidden);
    auto final_norm = Bf16("model.language_model.norm.weight", kHidden);
    if (!embedding.ok()) return embedding.status();
    if (!final_norm.ok()) return final_norm.status();
    embedding_ = embedding.value();
    final_norm_ = final_norm.value();

    for (std::size_t index = 0; index < layers_.size(); ++index) {
      LayerBinding& layer = layers_[index];
      layer.global = index % 6U == 5U;
      layer.kv_heads = layer.global ? 1U : 8U;
      layer.head_dimension = layer.global ? 512U : 256U;
      layer.query_elements = kQueryHeads * layer.head_dimension;
      layer.kv_elements = layer.kv_heads * layer.head_dimension;
      const std::string base = "model.language_model.layers." + std::to_string(index) + ".";

      auto q = Fp8(base + "self_attn.q_proj", layer.query_elements, kHidden);
      auto k = Fp8(base + "self_attn.k_proj", layer.kv_elements, kHidden);
      auto o = Fp8(base + "self_attn.o_proj", kHidden, layer.query_elements);
      if (!q.ok()) return q.status();
      if (!k.ok()) return k.status();
      if (!o.ok()) return o.status();
      layer.q = q.value();
      layer.k = k.value();
      layer.o = o.value();
      if (!layer.global) {
        auto v = Fp8(base + "self_attn.v_proj", layer.kv_elements, kHidden);
        if (!v.ok()) return v.status();
        layer.v = v.value();
      }

      auto gate = Nvfp4(base + "mlp.gate_proj", kIntermediate, kHidden);
      auto up = Nvfp4(base + "mlp.up_proj", kIntermediate, kHidden);
      auto down = Nvfp4(base + "mlp.down_proj", kHidden, kIntermediate);
      if (!gate.ok()) return gate.status();
      if (!up.ok()) return up.status();
      if (!down.ok()) return down.status();
      if (std::bit_cast<std::uint32_t>(gate.value().input_divisor) !=
          std::bit_cast<std::uint32_t>(up.value().input_divisor)) {
        return Error(StatusCode::kDataLoss,
                     "Gate and Up input divisors differ in layer " + std::to_string(index));
      }
      layer.gate = gate.value();
      layer.up = up.value();
      layer.down = down.value();

      auto input_norm = Bf16(base + "input_layernorm.weight", kHidden);
      auto q_norm = Bf16(base + "self_attn.q_norm.weight", layer.head_dimension);
      auto k_norm = Bf16(base + "self_attn.k_norm.weight", layer.head_dimension);
      auto post_attention = Bf16(base + "post_attention_layernorm.weight", kHidden);
      auto pre_mlp = Bf16(base + "pre_feedforward_layernorm.weight", kHidden);
      auto post_mlp = Bf16(base + "post_feedforward_layernorm.weight", kHidden);
      auto scalar = Bf16(base + "layer_scalar", 1U);
      auto k_cache_scale = Bf16(base + "self_attn.k_scale", 1U);
      auto v_cache_scale = Bf16(base + "self_attn.v_scale", 1U);
      if (!input_norm.ok()) return input_norm.status();
      if (!q_norm.ok()) return q_norm.status();
      if (!k_norm.ok()) return k_norm.status();
      if (!post_attention.ok()) return post_attention.status();
      if (!pre_mlp.ok()) return pre_mlp.status();
      if (!post_mlp.ok()) return post_mlp.status();
      if (!scalar.ok()) return scalar.status();
      if (!k_cache_scale.ok()) return k_cache_scale.status();
      if (!v_cache_scale.ok()) return v_cache_scale.status();
      layer.input_norm = input_norm.value();
      layer.q_norm = q_norm.value();
      layer.k_norm = k_norm.value();
      layer.post_attention_norm = post_attention.value();
      layer.pre_mlp_norm = pre_mlp.value();
      layer.post_mlp_norm = post_mlp.value();
      layer.layer_scalar = scalar.value();
      layer.k_cache_scale = k_cache_scale.value();
      layer.v_cache_scale = v_cache_scale.value();
    }
    return Status::Ok();
  }

  ModelManifest manifest_;
  DeviceAllocation weights_;
  std::unordered_map<std::string, DeviceTensor> tensors_;
  std::array<LayerBinding, kLayers> layers_{};
  const std::uint16_t* embedding_ = nullptr;
  const std::uint16_t* final_norm_ = nullptr;
};

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
  std::uint64_t output_candidates = 0;
  std::uint64_t selected = 0;
  std::uint64_t suppressed = 0;
  std::uint64_t total = 0;
};

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
  std::uint64_t mlp_packed = 0, mlp_scales = 0, gate = 0, up = 0, product = 0;
  std::uint64_t down_packed = 0, down_scales = 0;
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

__global__ void OutputHeadKernel(const std::uint16_t* weights, const float* hidden,
                                 float* logits) {
  const std::uint64_t token = blockIdx.x;
  float sum = 0.0F;
  const std::uint64_t base = token * kHidden;
  for (std::uint64_t index = threadIdx.x; index < kHidden; index += blockDim.x) {
    const float weight = static_cast<float>(__ushort_as_bfloat16(weights[base + index]));
    sum = fmaf(weight, hidden[index], sum);
  }
  const float logit = BlockSum(sum);
  if (threadIdx.x == 0U) logits[token] = tanhf(logit / 30.0F) * 30.0F;
}

struct ArgmaxValue {
  float value;
  std::uint32_t token;
};

__device__ bool IsSuppressed(
    std::uint64_t token, const std::uint32_t* suppressed,
    std::uint32_t suppressed_count) {
  for (std::uint32_t index = 0; index < suppressed_count; ++index) {
    if (token == suppressed[index]) return true;
  }
  return false;
}

__global__ void FusedOutputHeadCandidatesKernel(
    const std::uint16_t* weights, const float* hidden,
    const std::uint32_t* suppressed, std::uint32_t suppressed_count,
    const internal::DecodeControl* control, ArgmaxValue* candidates,
    float* diagnostic_logits) {
  constexpr unsigned kWarpSize = 32U;
  constexpr unsigned kWarpsPerBlock = kThreads / kWarpSize;
  __shared__ ArgmaxValue warp_candidates[kWarpsPerBlock];
  const unsigned warp = threadIdx.x / kWarpSize;
  const unsigned lane = threadIdx.x % kWarpSize;
  ArgmaxValue best{-FLT_MAX, 0U};
  const std::uint32_t dynamic_suppressed_count =
      control == nullptr ? suppressed_count : control->suppressed_token_count;
  for (std::uint64_t token =
           static_cast<std::uint64_t>(blockIdx.x) * kWarpsPerBlock + warp;
       token < kVocabulary;
       token += static_cast<std::uint64_t>(gridDim.x) * kWarpsPerBlock) {
    float sum = 0.0F;
    const std::uint64_t base = token * kHidden;
    for (std::uint64_t index = lane; index < kHidden;
         index += kWarpSize) {
      const float weight = static_cast<float>(
          __ushort_as_bfloat16(weights[base + index]));
      sum = fmaf(weight, hidden[index], sum);
    }
    for (unsigned offset = kWarpSize / 2U; offset != 0U; offset >>= 1U) {
      sum += __shfl_down_sync(0xFFFFFFFFU, sum, offset);
    }
    if (lane == 0U) {
      const float softcapped = tanhf(sum / 30.0F) * 30.0F;
      if (diagnostic_logits != nullptr) diagnostic_logits[token] = softcapped;
      if (!IsSuppressed(token, suppressed, dynamic_suppressed_count)) {
        if (softcapped > best.value ||
            (softcapped == best.value && token < best.token)) {
          best = {softcapped, static_cast<std::uint32_t>(token)};
        }
      }
    }
  }
  if (lane == 0U) warp_candidates[warp] = best;
  __syncthreads();
  if (warp == 0U) {
    best = lane < kWarpsPerBlock ? warp_candidates[lane]
                                 : ArgmaxValue{-FLT_MAX, 0U};
    for (unsigned offset = kWarpSize / 2U; offset != 0U; offset >>= 1U) {
      const float other_value =
          __shfl_down_sync(0xFFFFFFFFU, best.value, offset);
      const std::uint32_t other_token =
          __shfl_down_sync(0xFFFFFFFFU, best.token, offset);
      if (other_value > best.value ||
          (other_value == best.value && other_token < best.token)) {
        best = {other_value, other_token};
      }
    }
    if (lane == 0U) candidates[blockIdx.x] = best;
  }
}

__global__ void OutputHeadCandidateArgmaxKernel(
    const ArgmaxValue* candidates, std::uint32_t* selected) {
  __shared__ ArgmaxValue scratch[kThreads];
  ArgmaxValue best{-FLT_MAX, 0U};
  for (std::uint32_t index = threadIdx.x; index < kFusedOutputHeadBlocks;
       index += blockDim.x) {
    const ArgmaxValue candidate = candidates[index];
    if (candidate.value > best.value ||
        (candidate.value == best.value && candidate.token < best.token)) {
      best = candidate;
    }
  }
  scratch[threadIdx.x] = best;
  __syncthreads();
  for (unsigned stride = kThreads / 2U; stride != 0U; stride >>= 1U) {
    if (threadIdx.x < stride) {
      const ArgmaxValue other = scratch[threadIdx.x + stride];
      if (other.value > scratch[threadIdx.x].value ||
          (other.value == scratch[threadIdx.x].value &&
           other.token < scratch[threadIdx.x].token)) {
        scratch[threadIdx.x] = other;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0U) selected[0] = scratch[0].token;
}

__global__ void ArgmaxKernel(const float* logits, const std::uint32_t* suppressed,
                             std::uint32_t suppressed_count, std::uint32_t* selected) {
  __shared__ ArgmaxValue scratch[kThreads];
  ArgmaxValue best{-FLT_MAX, 0U};
  for (std::uint64_t index = threadIdx.x; index < kVocabulary; index += blockDim.x) {
    bool skip = false;
    for (std::uint32_t suppressed_index = 0; suppressed_index < suppressed_count;
         ++suppressed_index) {
      if (index == suppressed[suppressed_index]) {
        skip = true;
        break;
      }
    }
    if (skip) continue;
    const float value = logits[index];
    if (value > best.value || (value == best.value && index < best.token)) {
      best = {value, static_cast<std::uint32_t>(index)};
    }
  }
  scratch[threadIdx.x] = best;
  __syncthreads();
  for (unsigned stride = kThreads / 2U; stride != 0U; stride >>= 1U) {
    if (threadIdx.x < stride) {
      const ArgmaxValue other = scratch[threadIdx.x + stride];
      if (other.value > scratch[threadIdx.x].value ||
          (other.value == scratch[threadIdx.x].value &&
           other.token < scratch[threadIdx.x].token)) {
        scratch[threadIdx.x] = other;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0U) selected[0] = scratch[0].token;
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

Status LaunchNvfp4Projection(const std::uint8_t* activation, const std::uint8_t* scales,
                             const Nvfp4Binding& binding, float* output,
                             cudaStream_t stream) {
  return internal::LaunchNvfp4Sm120DirectProjection(
      activation, scales, binding.packed_weight, binding.scales, output, binding.rows,
      binding.contracting, binding.input_divisor, binding.weight_divisor, stream);
}

Status LaunchFp8ProjectionBatch(const std::uint8_t* activation, const float* scales,
                                const Fp8Binding& binding, float* output,
                                std::uint64_t tokens, cudaStream_t stream) {
  return internal::LaunchFp8Sm120DirectProjectionBatch(
      activation, scales, binding.weight, binding.scales, output, tokens,
      binding.rows, binding.contracting, stream);
}

Status LaunchFp8QkvProjectionBatch(
    const std::uint8_t* activation, const float* scales,
    const Fp8Binding& q_binding, float* q_output,
    const Fp8Binding& k_binding, float* k_output,
    const Fp8Binding* v_binding, float* v_output, std::uint64_t tokens,
    cudaStream_t stream) {
  if (q_binding.contracting != k_binding.contracting ||
      (v_binding != nullptr &&
       q_binding.contracting != v_binding->contracting)) {
    return Status(StatusCode::kInvalidArgument,
                  "grouped FP8 Q/K/V projections require one contracting dimension");
  }
  return internal::LaunchFp8Sm120GroupedQkvProjectionBatch(
      activation, scales, q_binding.weight, q_binding.scales, q_output,
      q_binding.rows, k_binding.weight, k_binding.scales, k_output,
      k_binding.rows, v_binding == nullptr ? nullptr : v_binding->weight,
      v_binding == nullptr ? nullptr : v_binding->scales,
      v_binding == nullptr ? nullptr : v_output,
      v_binding == nullptr ? 0U : v_binding->rows, tokens,
      q_binding.contracting, stream);
}

Status LaunchNvfp4ProjectionBatch(
    const std::uint8_t* activation, const std::uint8_t* scales,
    const Nvfp4Binding& binding, float* output, std::uint64_t tokens,
    cudaStream_t stream) {
  return internal::LaunchNvfp4Sm120DirectProjectionBatch(
      activation, scales, binding.packed_weight, binding.scales, output, tokens,
      binding.rows, binding.contracting, binding.input_divisor,
      binding.weight_divisor, stream);
}

class InferenceEngine {
 public:
  InferenceEngine() = default;
  InferenceEngine(const InferenceEngine&) = delete;
  InferenceEngine& operator=(const InferenceEngine&) = delete;
  ~InferenceEngine() {
    if (stream_ != nullptr) (void)cudaStreamDestroy(stream_);
  }

  [[nodiscard]] Status Initialize(const std::filesystem::path& model_directory,
                                  std::uint64_t max_context,
                                  KvCacheMode kv_cache_mode) {
    const NvtxRange range("gem16gb.initialize");
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
    status = AllocateCache();
    if (!status.ok()) return status;
    status = AllocateWorkspace();
    if (!status.ok()) return status;
    status = AllocatePrefillWorkspace();
    if (!status.ok()) return status;
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
    const NvtxRange range("gem16gb.decode.forward");
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
      const cudaError_t launch_error =
          cudaGraphLaunch(full_decode_graph_.get(), stream_);
      if (launch_error != cudaSuccess) {
        return CudaFailure("launch full decode graph", launch_error);
      }
      const cudaError_t sync_error = cudaStreamSynchronize(stream_);
      if (sync_error != cudaSuccess) {
        return CudaFailure("synchronize full decode graph", sync_error);
      }
      return host->selected_token;
    }
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
    FusedOutputHeadCandidatesKernel<<<kFusedOutputHeadBlocks, kThreads, 0,
                                      stream_>>>(
        model_.embedding(), normalized,
        Pointer<std::uint32_t>(workspace_, offsets_.suppressed),
        suppressed_token_count_, nullptr,
        Pointer<ArgmaxValue>(workspace_, offsets_.output_candidates),
        diagnostic_logits);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
      return CudaFailure("launch fused output-head candidates", error);
    }
    if (diagnostic_logits != nullptr) {
      error = cudaMemcpyAsync(host_logits.data(), diagnostic_logits,
                              host_logits.size_bytes(),
                              cudaMemcpyDeviceToHost, stream_);
      if (error != cudaSuccess) {
        return CudaFailure("copy fused full logits", error);
      }
    }
    OutputHeadCandidateArgmaxKernel<<<1U, kThreads, 0, stream_>>>(
        Pointer<ArgmaxValue>(workspace_, offsets_.output_candidates), selected);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
      return CudaFailure("launch fused output-head argmax", error);
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
  [[nodiscard]] std::uint64_t cache_bytes() const { return cache_.bytes(); }
  [[nodiscard]] std::uint64_t workspace_bytes() const {
    return workspace_.bytes() + prefill_workspace_.bytes();
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
    const NvtxRange range("gem16gb.prefill");
    if (token_ids.empty() || token_ids.size() > max_context_) {
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
      float* hidden = Pointer<float>(prefill_workspace_, prefill_offsets_.hidden_a);
      const std::uint64_t hidden_elements = tokens * kHidden;
      EmbeddingBatchKernel<<<static_cast<unsigned>((hidden_elements + kThreads - 1U) /
                                                   kThreads),
                             kThreads, 0, stream_>>>(
          model_.embedding(), device_tokens, hidden, hidden_elements);
      error = cudaGetLastError();
      if (error != cudaSuccess) return CudaFailure("launch prefill embedding", error);
      for (const auto& layer : model_.layers()) {
        Status status = RunLayerBatch(layer, begin, tokens);
        if (!status.ok()) return status;
      }
      float* normalized = Pointer<float>(prefill_workspace_, prefill_offsets_.normalized);
      Status status = internal::LaunchRmsNormBf16(
          hidden, model_.final_norm(), normalized, tokens, kHidden, kEpsilon, stream_);
      if (!status.ok()) return status;
      if (begin + tokens == token_ids.size()) {
        float* last = normalized + (tokens - 1U) * kHidden;
        float* logits = Pointer<float>(workspace_, offsets_.logits);
        auto* selected = Pointer<std::uint32_t>(workspace_, offsets_.selected);
        OutputHeadKernel<<<static_cast<unsigned>(kVocabulary), kThreads, 0, stream_>>>(
            model_.embedding(), last, logits);
        error = cudaGetLastError();
        if (error != cudaSuccess) return CudaFailure("launch prefill output head", error);
        if (!host_logits.empty()) {
          error = cudaMemcpyAsync(host_logits.data(), logits,
                                  host_logits.size_bytes(),
                                  cudaMemcpyDeviceToHost, stream_);
          if (error != cudaSuccess) {
            return CudaFailure("copy prefill full logits", error);
          }
        }
        ArgmaxKernel<<<1U, kThreads, 0, stream_>>>(
            logits, Pointer<std::uint32_t>(workspace_, offsets_.suppressed),
            suppressed_token_count_, selected);
        error = cudaGetLastError();
        if (error != cudaSuccess) return CudaFailure("launch prefill argmax", error);
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
    error = cudaStreamSynchronize(stream_);
    return error == cudaSuccess ? Status::Ok() : CudaFailure("reset KV cache", error);
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

 private:
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
#define GEM16GB_ADD(field, type, elements)                 \
    do {                                                   \
      auto next = layout.Add<type>(elements);              \
      if (!next.ok()) return next.status();                 \
      offsets_.field = next.value();                        \
    } while (false)
    GEM16GB_ADD(decode_control, internal::DecodeControl, 1U);
    GEM16GB_ADD(hidden_a, float, kHidden);
    GEM16GB_ADD(hidden_b, float, kHidden);
    GEM16GB_ADD(normalized, float, kHidden);
    GEM16GB_ADD(fp8_activation, std::uint8_t, kHidden);
    GEM16GB_ADD(fp8_scale, float, 1U);
    GEM16GB_ADD(q, float, kQueryHeads * 512U);
    GEM16GB_ADD(k, float, 8U * 256U);
    GEM16GB_ADD(v, float, 8U * 256U);
    GEM16GB_ADD(q_norm, float, kQueryHeads * 512U);
    GEM16GB_ADD(k_norm, float, 8U * 256U);
    GEM16GB_ADD(v_norm, float, 8U * 256U);
    GEM16GB_ADD(scores, float, kQueryHeads * max_context_);
    GEM16GB_ADD(attention, float, kQueryHeads * 512U);
    GEM16GB_ADD(o_activation, std::uint8_t, kQueryHeads * 512U);
    GEM16GB_ADD(o_scale, float, 1U);
    GEM16GB_ADD(projection, float, kHidden);
    GEM16GB_ADD(post_norm, float, kHidden);
    GEM16GB_ADD(mlp_packed, std::uint8_t, kHidden / 2U);
    GEM16GB_ADD(mlp_scales, std::uint8_t, kHidden / 16U);
    GEM16GB_ADD(gate, float, kIntermediate);
    GEM16GB_ADD(up, float, kIntermediate);
    GEM16GB_ADD(product, float, kIntermediate);
    GEM16GB_ADD(down_packed, std::uint8_t, kIntermediate / 2U);
    GEM16GB_ADD(down_scales, std::uint8_t, kIntermediate / 16U);
    GEM16GB_ADD(logits, float, kVocabulary);
    GEM16GB_ADD(output_candidates, ArgmaxValue, kFusedOutputHeadBlocks);
    GEM16GB_ADD(selected, std::uint32_t, 1U);
    GEM16GB_ADD(suppressed, std::uint32_t, kMaximumSuppressedTokens);
#undef GEM16GB_ADD
    auto size = AlignUp(layout.size(), kAlignment);
    if (!size.ok()) return size.status();
    offsets_.total = size.value();
    return workspace_.Allocate(size.value(), "allocate inference workspace arena");
  }

  [[nodiscard]] Status AllocatePrefillWorkspace() {
    LayoutBuilder layout;
#define GEM16GB_PREFILL_ADD(field, type, elements)          \
    do {                                                     \
      auto next = layout.Add<type>(elements);                \
      if (!next.ok()) return next.status();                   \
      prefill_offsets_.field = next.value();                  \
    } while (false)
    const std::uint64_t tokens = prefill_chunk_tokens_;
    constexpr std::uint64_t max_q = kQueryHeads * 512U;
    constexpr std::uint64_t max_kv = 8U * 256U;
    GEM16GB_PREFILL_ADD(token_ids, std::uint32_t, tokens);
    GEM16GB_PREFILL_ADD(hidden_a, float, tokens * kHidden);
    GEM16GB_PREFILL_ADD(hidden_b, float, tokens * kHidden);
    GEM16GB_PREFILL_ADD(normalized, float, tokens * kHidden);
    GEM16GB_PREFILL_ADD(fp8_activation, std::uint8_t, tokens * max_q);
    GEM16GB_PREFILL_ADD(fp8_scales, float, tokens);
    GEM16GB_PREFILL_ADD(q, float, tokens * max_q);
    GEM16GB_PREFILL_ADD(k, float, tokens * max_kv);
    GEM16GB_PREFILL_ADD(v, float, tokens * max_kv);
    GEM16GB_PREFILL_ADD(q_norm, float, tokens * max_q);
    GEM16GB_PREFILL_ADD(k_norm, float, tokens * max_kv);
    GEM16GB_PREFILL_ADD(v_norm, float, tokens * max_kv);
    GEM16GB_PREFILL_ADD(k_fp8, std::uint8_t, tokens * max_kv);
    GEM16GB_PREFILL_ADD(v_fp8, std::uint8_t, tokens * max_kv);
    if (kv_cache_mode_ == KvCacheMode::kBf16Correctness) {
      GEM16GB_PREFILL_ADD(scores, float,
                          tokens * kQueryHeads * max_context_);
    }
    GEM16GB_PREFILL_ADD(attention, float, tokens * max_q);
    GEM16GB_PREFILL_ADD(o_activation, std::uint8_t, tokens * max_q);
    GEM16GB_PREFILL_ADD(o_scales, float, tokens);
    GEM16GB_PREFILL_ADD(projection, float, tokens * kHidden);
    GEM16GB_PREFILL_ADD(post_norm, float, tokens * kHidden);
    GEM16GB_PREFILL_ADD(mlp_packed, std::uint8_t, tokens * kHidden / 2U);
    GEM16GB_PREFILL_ADD(mlp_scales, std::uint8_t, tokens * kHidden / 16U);
    GEM16GB_PREFILL_ADD(gate, float, tokens * kIntermediate);
    GEM16GB_PREFILL_ADD(up, float, tokens * kIntermediate);
    GEM16GB_PREFILL_ADD(product, float, tokens * kIntermediate);
    GEM16GB_PREFILL_ADD(down_packed, std::uint8_t, tokens * kIntermediate / 2U);
    GEM16GB_PREFILL_ADD(down_scales, std::uint8_t, tokens * kIntermediate / 16U);
#undef GEM16GB_PREFILL_ADD
    auto size = AlignUp(layout.size(), kAlignment);
    if (!size.ok()) return size.status();
    return prefill_workspace_.Allocate(size.value(), "allocate native prefill workspace");
  }

  [[nodiscard]] Status RunLayerBatch(const LayerBinding& layer,
                                     std::uint64_t start_position,
                                     std::uint64_t tokens) {
    const NvtxRange range("gem16gb.prefill.layer");
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
    const std::uint64_t hidden_elements = tokens * kHidden;
    Status status = internal::LaunchRmsNormFp8TokenQuantizationBatch(
        hidden_a, layer.input_norm, fp8, fp8_scales, tokens, kHidden,
        kEpsilon, stream_);
    if (!status.ok()) return status;
    status = LaunchFp8QkvProjectionBatch(
        fp8, fp8_scales, layer.q, q, layer.k, k,
        layer.global ? nullptr : &layer.v, v, tokens, stream_);
    if (!status.ok()) return status;
    if (layer.global) {
      const cudaError_t error = cudaMemcpyAsync(
          v, k, static_cast<std::size_t>(tokens * layer.kv_elements * sizeof(float)),
          cudaMemcpyDeviceToDevice, stream_);
      if (error != cudaSuccess) return CudaFailure("reuse batched global K for V", error);
    }
    for (const Status next : {
             LaunchRoundBf16(q, tokens * layer.query_elements, stream_),
             LaunchRoundBf16(k, tokens * layer.kv_elements, stream_),
             LaunchRoundBf16(v, tokens * layer.kv_elements, stream_),
             internal::LaunchRmsNormBf16(q, layer.q_norm, q_norm,
                                     tokens * kQueryHeads, layer.head_dimension,
                                     kEpsilon, stream_),
             internal::LaunchRmsNormBf16(k, layer.k_norm, k_norm,
                                     tokens * layer.kv_heads, layer.head_dimension,
                                     kEpsilon, stream_),
             internal::LaunchRmsNormBf16(v, nullptr, v_norm,
                                     tokens * layer.kv_heads, layer.head_dimension,
                                     kEpsilon, stream_)}) {
      if (!next.ok()) return next;
    }
    if (layer.global) {
      status = internal::LaunchProportionalRotaryEmbeddingBatch(
          q_norm, tokens, kQueryHeads, layer.head_dimension, 0.25,
          start_position, 1000000.0, 1.0, stream_);
      if (!status.ok()) return status;
      status = internal::LaunchProportionalRotaryEmbeddingBatch(
          k_norm, tokens, layer.kv_heads, layer.head_dimension, 0.25,
          start_position, 1000000.0, 1.0, stream_);
    } else {
      status = internal::LaunchRotaryEmbeddingBatch(
          q_norm, tokens, kQueryHeads, layer.head_dimension,
          layer.head_dimension, start_position, 10000.0, stream_);
      if (!status.ok()) return status;
      status = internal::LaunchRotaryEmbeddingBatch(
          k_norm, tokens, layer.kv_heads, layer.head_dimension,
          layer.head_dimension, start_position, 10000.0, stream_);
    }
    if (!status.ok()) return status;
    status = LaunchRoundBf16(q_norm, tokens * layer.query_elements, stream_);
    if (!status.ok()) return status;
    status = LaunchRoundBf16(k_norm, tokens * layer.kv_elements, stream_);
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
      status = internal::LaunchAppendKvFp8Batch(
          k_fp8, v_fp8, layer.key_cache_fp8, layer.value_cache_fp8,
          start_position, tokens, layer.kv_elements, capacity, stream_);
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
    status = LaunchRoundBf16(attention, tokens * layer.query_elements, stream_);
    if (!status.ok()) return status;
    auto* o_activation = Pointer<std::uint8_t>(prefill_workspace_, prefill_offsets_.o_activation);
    float* o_scales = Pointer<float>(prefill_workspace_, prefill_offsets_.o_scales);
    status = internal::LaunchFp8ReferenceTokenQuantizationBatch(
        attention, o_activation, o_scales, tokens, layer.query_elements, stream_);
    if (!status.ok()) return status;
    status = LaunchFp8ProjectionBatch(o_activation, o_scales, layer.o, projection,
                                      tokens, stream_);
    if (!status.ok()) return status;
    status = LaunchRoundBf16(projection, hidden_elements, stream_);
    if (!status.ok()) return status;
    status = internal::LaunchRmsNormResidualBf16(
        projection, layer.post_attention_norm, hidden_a, nullptr, hidden_b,
        tokens, kHidden, kEpsilon, nullptr, stream_);
    if (!status.ok()) return status;

    auto* mlp_packed = Pointer<std::uint8_t>(prefill_workspace_, prefill_offsets_.mlp_packed);
    auto* mlp_scales = Pointer<std::uint8_t>(prefill_workspace_, prefill_offsets_.mlp_scales);
    float* gate = Pointer<float>(prefill_workspace_, prefill_offsets_.gate);
    float* up = Pointer<float>(prefill_workspace_, prefill_offsets_.up);
    status = internal::LaunchRmsNormNvfp4ActivationQuantizationBatch(
        hidden_b, layer.pre_mlp_norm, mlp_packed, mlp_scales, tokens, kHidden,
        kEpsilon, layer.gate.input_divisor, stream_);
    if (!status.ok()) return status;
    status = LaunchNvfp4ProjectionBatch(mlp_packed, mlp_scales, layer.gate,
                                        gate, tokens, stream_);
    if (!status.ok()) return status;
    status = LaunchNvfp4ProjectionBatch(mlp_packed, mlp_scales, layer.up,
                                        up, tokens, stream_);
    if (!status.ok()) return status;
    auto* down_packed = Pointer<std::uint8_t>(prefill_workspace_, prefill_offsets_.down_packed);
    auto* down_scales = Pointer<std::uint8_t>(prefill_workspace_, prefill_offsets_.down_scales);
    status = internal::LaunchGatedGeluNvfp4ActivationQuantization(
        gate, up, down_packed, down_scales, tokens * kIntermediate,
        layer.down.input_divisor, stream_);
    if (!status.ok()) return status;
    status = LaunchNvfp4ProjectionBatch(down_packed, down_scales, layer.down,
                                        projection, tokens, stream_);
    if (!status.ok()) return status;
    status = LaunchRoundBf16(projection, hidden_elements, stream_);
    if (!status.ok()) return status;
    return internal::LaunchRmsNormResidualBf16(
        projection, layer.post_mlp_norm, hidden_b, nullptr, hidden_a, tokens,
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
    Status status = LaunchDecodePrefix(layer);
    if (!status.ok()) return status;
    auto* control = Pointer<internal::DecodeControl>(
        workspace_, offsets_.decode_control);
    float* q_norm = Pointer<float>(workspace_, offsets_.q_norm);
    float* k_norm = Pointer<float>(workspace_, offsets_.k_norm);
    float* v_norm = Pointer<float>(workspace_, offsets_.v_norm);
    float* scores = Pointer<float>(workspace_, offsets_.scores);
    float* attention = Pointer<float>(workspace_, offsets_.attention);
    if (layer.global) {
      status = internal::LaunchProportionalRotaryEmbeddingControlled(
          q_norm, kQueryHeads, layer.head_dimension, 0.25, control,
          1000000.0, 1.0, stream_);
      if (!status.ok()) return status;
      status = internal::LaunchProportionalRotaryEmbeddingControlled(
          k_norm, layer.kv_heads, layer.head_dimension, 0.25, control,
          1000000.0, 1.0, stream_);
    } else {
      status = internal::LaunchRotaryEmbeddingControlled(
          q_norm, kQueryHeads, layer.head_dimension, layer.head_dimension,
          control, 10000.0, stream_);
      if (!status.ok()) return status;
      status = internal::LaunchRotaryEmbeddingControlled(
          k_norm, layer.kv_heads, layer.head_dimension, layer.head_dimension,
          control, 10000.0, stream_);
    }
    if (!status.ok()) return status;
    status = LaunchRoundBf16(q_norm, layer.query_elements, stream_);
    if (!status.ok()) return status;
    status = LaunchRoundBf16(k_norm, layer.kv_elements, stream_);
    if (!status.ok()) return status;
    const std::uint64_t capacity =
        layer.global ? max_context_ : std::min(max_context_, kSlidingWindow);
    if (kv_cache_mode_ == KvCacheMode::kCheckpointFp8) {
      status = internal::LaunchAppendKvFp8Controlled(
          k_norm, v_norm, layer.key_cache_fp8, layer.value_cache_fp8,
          layer.k_cache_scale, layer.v_cache_scale, control, layer.kv_heads,
          layer.head_dimension, capacity, !layer.global, stream_);
      if (!status.ok()) return status;
      status = internal::LaunchLocalAttentionDecodeFp8Controlled(
          q_norm, layer.key_cache_fp8, layer.value_cache_fp8,
          layer.k_cache_scale, layer.v_cache_scale, scores, attention, control,
          kQueryHeads, layer.kv_heads, layer.head_dimension, capacity,
          !layer.global, stream_);
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
    FusedOutputHeadCandidatesKernel<<<kFusedOutputHeadBlocks, kThreads, 0,
                                      stream_>>>(
        model_.embedding(), normalized,
        Pointer<std::uint32_t>(workspace_, offsets_.suppressed), 0U, control,
        Pointer<ArgmaxValue>(workspace_, offsets_.output_candidates), nullptr);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
      return CudaFailure("launch controlled fused output-head candidates",
                         error);
    }
    OutputHeadCandidateArgmaxKernel<<<1U, kThreads, 0, stream_>>>(
        Pointer<ArgmaxValue>(workspace_, offsets_.output_candidates), selected);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
      return CudaFailure("launch controlled fused output-head argmax", error);
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

  [[nodiscard]] Status LaunchDecodePrefix(const LayerBinding& layer) {
    float* hidden_a = Pointer<float>(workspace_, offsets_.hidden_a);
    float* normalized = Pointer<float>(workspace_, offsets_.normalized);
    auto* fp8_activation =
        Pointer<std::uint8_t>(workspace_, offsets_.fp8_activation);
    float* fp8_scale = Pointer<float>(workspace_, offsets_.fp8_scale);
    float* q = Pointer<float>(workspace_, offsets_.q);
    float* k = Pointer<float>(workspace_, offsets_.k);
    float* v = Pointer<float>(workspace_, offsets_.v);
    float* q_norm = Pointer<float>(workspace_, offsets_.q_norm);
    float* k_norm = Pointer<float>(workspace_, offsets_.k_norm);
    float* v_norm = Pointer<float>(workspace_, offsets_.v_norm);

    Status status = internal::LaunchRmsNormBf16(
        hidden_a, layer.input_norm, normalized, 1U, kHidden, kEpsilon, stream_);
    if (!status.ok()) return status;
    status = internal::LaunchFp8ReferenceTokenQuantization(
        normalized, fp8_activation, fp8_scale, kHidden, stream_);
    if (!status.ok()) return status;
    status = LaunchFp8Projection(fp8_activation, fp8_scale, layer.q, q, stream_);
    if (!status.ok()) return status;
    status = LaunchFp8Projection(fp8_activation, fp8_scale, layer.k, k, stream_);
    if (!status.ok()) return status;
    if (layer.global) {
      const cudaError_t error = cudaMemcpyAsync(
          v, k, layer.kv_elements * sizeof(float), cudaMemcpyDeviceToDevice,
          stream_);
      if (error != cudaSuccess) {
        return CudaFailure("reuse global K projection for V", error);
      }
    } else {
      status = LaunchFp8Projection(fp8_activation, fp8_scale, layer.v, v, stream_);
      if (!status.ok()) return status;
    }
    for (const Status next : {
             LaunchRoundBf16(q, layer.query_elements, stream_),
             LaunchRoundBf16(k, layer.kv_elements, stream_),
             LaunchRoundBf16(v, layer.kv_elements, stream_),
             internal::LaunchRmsNormBf16(q, layer.q_norm, q_norm, kQueryHeads,
                                     layer.head_dimension, kEpsilon, stream_),
             internal::LaunchRmsNormBf16(k, layer.k_norm, k_norm, layer.kv_heads,
                                     layer.head_dimension, kEpsilon, stream_),
             internal::LaunchRmsNormBf16(v, nullptr, v_norm, layer.kv_heads,
                                     layer.head_dimension, kEpsilon, stream_),
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
    }
    if (capture != nullptr) {
      status = capture_hidden(capture->post_attention_residual, hidden_b,
                              "copy post-attention residual state");
      if (!status.ok()) return status;
    }
    status = internal::LaunchRmsNormBf16(hidden_b, layer.pre_mlp_norm, normalized,
                                     1U, kHidden, kEpsilon, stream_);
    if (!status.ok()) return status;
    if (capture != nullptr) {
      status = capture_hidden(capture->pre_feedforward_norm, normalized,
                              "copy pre-feedforward norm state");
      if (!status.ok()) return status;
    }
    status = internal::LaunchNvfp4ReferenceActivationQuantization(
        normalized, mlp_packed, mlp_scales, kHidden, layer.gate.input_divisor,
        stream_);
    if (!status.ok()) return status;
    status = LaunchNvfp4Projection(mlp_packed, mlp_scales, layer.gate, gate,
                                   stream_);
    if (!status.ok()) return status;
    status = LaunchNvfp4Projection(mlp_packed, mlp_scales, layer.up, up,
                                   stream_);
    if (!status.ok()) return status;
    status = LaunchRoundBf16(gate, kIntermediate, stream_);
    if (!status.ok()) return status;
    status = LaunchRoundBf16(up, kIntermediate, stream_);
    if (!status.ok()) return status;
    status = internal::LaunchGeluTanhProduct(gate, up, product, kIntermediate,
                                             stream_);
    if (!status.ok()) return status;
    status = LaunchRoundBf16(product, kIntermediate, stream_);
    if (!status.ok()) return status;
    if (capture != nullptr) {
      status = capture_values(capture->gate, gate, kIntermediate,
                              "copy MLP gate state");
      if (!status.ok()) return status;
      status = capture_values(capture->up, up, kIntermediate,
                              "copy MLP up state");
      if (!status.ok()) return status;
      status = capture_values(capture->gelu_product, product, kIntermediate,
                              "copy MLP GELU product state");
      if (!status.ok()) return status;
    }
    status = internal::LaunchNvfp4ReferenceActivationQuantization(
        product, down_packed, down_scales, kIntermediate,
        layer.down.input_divisor, stream_);
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
    const NvtxRange range("gem16gb.decode.layer");
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

  LoadedModel model_;
  DeviceAllocation cache_;
  DeviceAllocation workspace_;
  DeviceAllocation prefill_workspace_;
  PinnedHostAllocation decode_host_state_;
  WorkspaceOffsets offsets_{};
  PrefillOffsets prefill_offsets_{};
  std::array<GraphExecutable, kLayers> decode_prefix_graphs_{};
  std::array<GraphExecutable, kLayers> decode_suffix_graphs_{};
  GraphExecutable full_decode_graph_;
  cudaStream_t stream_ = nullptr;
  std::uint64_t max_context_ = 0;
  std::uint64_t prefill_chunk_tokens_ = kMinimumPrefillChunkTokens;
  std::uint64_t decode_graph_device_bytes_ = 0;
  std::uint32_t suppressed_token_count_ = 0;
  KvCacheMode kv_cache_mode_ = KvCacheMode::kCheckpointFp8;
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
                                    options.kv_cache_mode);
  if (!status.ok()) return status;
  status = engine.SetSuppressedTokens(options.suppressed_token_ids);
  if (!status.ok()) return status;
  const auto load_end = std::chrono::steady_clock::now();

  GreedyInferenceResult result;
  result.output_token_ids.reserve(static_cast<std::size_t>(generation_steps));
  result.teacher_forced_token_ids = options.teacher_forced_token_ids;
  result.teacher_forcing = teacher_forcing;
  result.kv_cache_mode = options.kv_cache_mode;
  result.decode_graphs =
      options.state_dump_path.empty() && options.logits_dump_path.empty();
  result.model_load_milliseconds = Milliseconds(load_end - load_start);
  result.weight_arena_bytes = engine.weight_bytes();
  result.kv_cache_bytes = engine.cache_bytes();
  result.workspace_bytes = engine.workspace_bytes();
  result.decode_graph_device_bytes = engine.decode_graph_device_bytes();
  result.prefill_chunk_tokens = engine.prefill_chunk_tokens();
  result.packed_weight_source_layout_direct = true;
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
  for (std::uint64_t generated = 1U;
       generated < generation_steps && !result.stopped; ++generated) {
    const std::uint64_t position = options.input_token_ids.size() + generated - 1U;
    const std::size_t logit_offset =
        static_cast<std::size_t>(generated * kVocabulary);
    const std::span<float> logit_capture =
        captured_logits.span().empty()
            ? std::span<float>()
            : captured_logits.span().subspan(logit_offset,
                                             static_cast<std::size_t>(kVocabulary));
    const bool capture_state =
        options.state_dump_position.has_value() &&
        *options.state_dump_position == position;
    const std::uint32_t input_token =
        teacher_forcing
            ? options.teacher_forced_token_ids[static_cast<std::size_t>(generated - 1U)]
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
        std::find(options.stop_token_ids.begin(), options.stop_token_ids.end(), next_token) !=
        options.stop_token_ids.end()) {
      result.stopped = true;
      result.stop_token_id = next_token;
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
                                    options.kv_cache_mode);
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
         << (result.teacher_forcing ? "teacher_forced" : "greedy")
         << "\",\n"
         << "  \"fallbacks\": " << result.fallback_count << ",\n"
         << "  \"packed_weight_source_layout_direct\": "
         << (result.packed_weight_source_layout_direct ? "true" : "false") << ",\n"
         << "  \"weight_scale_layout\": \"sm120_row8_k64\",\n"
         << "  \"load_time_scale_swizzle\": true,\n"
         << "  \"persistent_repack_bytes\": 0,\n"
         << "  \"token_loop_allocations\": "
         << (result.token_loop_allocations ? "true" : "false") << ",\n"
         << "  \"fused_gate_up\": false,\n"
         << "  \"fused_prefill_attention\": true,\n"
         << "  \"fp8_prefill_tile\": \"m64n64k64\",\n"
         << "  \"fp8_prefill_pipeline_stages\": 2,\n"
         << "  \"grouped_qkv_prefill\": true,\n"
         << "  \"fused_rmsnorm_boundaries\": true,\n"
         << "  \"fused_prefill_rmsnorm_fp8_quantization\": true,\n"
         << "  \"fused_prefill_rmsnorm_nvfp4_quantization\": true,\n"
         << "  \"fused_prefill_gated_gelu_nvfp4_quantization\": true,\n"
         << "  \"fused_output_head\": true,\n"
         << "  \"decode_graphs\": "
         << (result.decode_graphs ? "true" : "false") << ",\n"
         << "  \"model_load_ms\": " << result.model_load_milliseconds << ",\n"
         << "  \"prompt_ms\": " << result.prompt_milliseconds << ",\n"
         << "  \"decode_ms\": " << result.decode_milliseconds << ",\n"
         << "  \"decode_tokens_per_second\": " << result.decode_tokens_per_second << ",\n"
         << "  \"weight_arena_bytes\": " << result.weight_arena_bytes << ",\n"
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
         << "  \"state_dump_format\": \"gem16gb_layer_state_v5\",\n"
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
         << "\"projection_path\":\"native_sm120\",\"kv_cache_mode\":\""
         << (result.options.kv_cache_mode == KvCacheMode::kCheckpointFp8
                 ? "checkpoint_fp8" : "bf16_correctness")
         << "\",\"kv_cache_layout\":\"hybrid_local_ring_global_contiguous"
         << "\",\"fused_gate_up\":false"
         << ",\"fused_prefill_attention\":true"
         << ",\"fp8_prefill_tile\":\"m64n64k64\""
         << ",\"fp8_prefill_pipeline_stages\":2"
         << ",\"grouped_qkv_prefill\":true"
         << ",\"fused_rmsnorm_boundaries\":true"
         << ",\"fused_prefill_rmsnorm_fp8_quantization\":true"
         << ",\"fused_prefill_rmsnorm_nvfp4_quantization\":true"
         << ",\"fused_prefill_gated_gelu_nvfp4_quantization\":true"
         << ",\"fused_output_head\":true"
         << ",\"decode_graphs\":true"
         << ",\"context_tokens\":" << result.options.context_tokens
         << ",\"generated_tokens\":" << result.options.generated_tokens
         << ",\"warmup_runs\":" << result.options.warmup_runs
         << ",\"measured_runs\":" << result.options.measured_runs
         << ",\"prompt_seed\":" << result.options.prompt_seed
         << ",\"prompt_token_formula\":\"1000+((seed+index*7919)%9000)\","
         << "\"timing_boundary\":\"host_end_to_end_forward_and_greedy_selection\","
         << "\"first_selected_token_excluded_from_decode\":true,"
         << "\"model_loaded_once\":true,\"cache_reset_outside_timing\":true,"
         << "\"prefill_path\":\"native_chunked_sm120\","
         << "\"packed_weight_source_layout_direct\":"
         << (result.packed_weight_source_layout_direct ? "true" : "false")
         << ",\"weight_scale_layout\":\"sm120_row8_k64\""
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
         << ",\"fp8_prefill_tile\":\"m64n64k64\""
         << ",\"fp8_prefill_pipeline_stages\":2"
         << ",\"grouped_qkv_prefill\":true"
         << ",\"fused_rmsnorm_boundaries\":true"
         << ",\"fused_prefill_rmsnorm_fp8_quantization\":true"
         << ",\"fused_prefill_rmsnorm_nvfp4_quantization\":true"
         << ",\"fused_prefill_gated_gelu_nvfp4_quantization\":true"
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

}  // namespace gem16gb
