#include "cuda/engine/target_model.h"

#include "cuda/nvfp4/sm120_layout.h"
#include "gem16/model.h"
#include "platform/mapped_file.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace gem16::internal {
namespace {

constexpr std::uint64_t kHidden = 3840U;
constexpr std::uint64_t kIntermediate = 15360U;
constexpr std::uint64_t kVocabulary = 262144U;
constexpr std::uint64_t kQueryHeads = 16U;
constexpr std::uint64_t kAlignment = 256U;
constexpr std::uint64_t kWeightLayoutHostStagingBytes = 4ULL * 1024ULL * 1024ULL;

Status Error(StatusCode code, std::string message) {
  return Status(code, std::move(message));
}

Status CudaFailure(const char* operation, cudaError_t error) {
  return Error(StatusCode::kInternal,
               std::string(operation) + ": " + cudaGetErrorName(error) + ": " +
                   cudaGetErrorString(error));
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

struct DeviceTensor {
  const TensorInfo* info = nullptr;
  std::byte* data = nullptr;
  float scalar_f32 = 0.0F;
  bool has_scalar_f32 = false;
  bool sm120_weight_tiled = false;
  bool sm120_scale_tiled = false;
};

Status UploadSm120TiledBlocks(
    std::byte* destination, std::span<const std::uint8_t> source,
    const internal::Sm120Nvfp4SourceLayout& layout,
    std::uint64_t bytes_per_k_block, std::uint64_t source_row_bytes,
    const char* operation) {
  if (destination == nullptr || bytes_per_k_block == 0U ||
      source_row_bytes != layout.k_blocks * bytes_per_k_block ||
      source.size() != layout.rows * source_row_bytes) {
    return Error(StatusCode::kDataLoss,
                 "invalid source buffer for SM120 tiled upload");
  }
  constexpr std::uint64_t kRowsPerTile = 8U;
  const std::uint64_t full_tile_bytes =
      kRowsPerTile * layout.k_blocks * bytes_per_k_block;
  const std::uint64_t tiles_per_batch = std::max<std::uint64_t>(
      1U, kWeightLayoutHostStagingBytes /
              std::max<std::uint64_t>(1U, full_tile_bytes));
  const std::uint64_t staging_bytes = std::min<std::uint64_t>(
      source.size(), tiles_per_batch * full_tile_bytes);
  std::vector<std::uint8_t> staging(
      static_cast<std::size_t>(staging_bytes));

  for (std::uint64_t first_tile = 0; first_tile < layout.row_tiles;
       first_tile += tiles_per_batch) {
    const std::uint64_t end_tile =
        std::min(layout.row_tiles, first_tile + tiles_per_batch);
    const std::uint64_t first_row = first_tile * kRowsPerTile;
    std::uint64_t cursor = 0U;
    for (std::uint64_t row_tile = first_tile; row_tile < end_tile;
         ++row_tile) {
      const std::uint64_t tile_first_row = row_tile * kRowsPerTile;
      const std::uint64_t tile_rows =
          std::min(kRowsPerTile, layout.rows - tile_first_row);
      for (std::uint64_t k_block = 0; k_block < layout.k_blocks;
           ++k_block) {
        for (std::uint64_t row = 0; row < tile_rows; ++row) {
          const std::uint64_t source_offset =
              (tile_first_row + row) * source_row_bytes +
              k_block * bytes_per_k_block;
          std::memcpy(staging.data() + cursor,
                      source.data() + source_offset,
                      static_cast<std::size_t>(bytes_per_k_block));
          cursor += bytes_per_k_block;
        }
      }
    }
    const std::uint64_t destination_offset =
        first_row * layout.k_blocks * bytes_per_k_block;
    const cudaError_t error =
        cudaMemcpy(destination + destination_offset, staging.data(),
                   static_cast<std::size_t>(cursor),
                   cudaMemcpyHostToDevice);
    if (error != cudaSuccess) return CudaFailure(operation, error);
  }
  return Status::Ok();
}

}  // namespace

class LoadedTargetModel::Impl {
 public:
  [[nodiscard]] Status Load(const std::filesystem::path& directory) {
    if (shared_ != nullptr) {
      return Error(StatusCode::kInvalidArgument,
                   "target model weights are already initialized");
    }
    shared_ = std::make_shared<SharedWeights>();
    auto inspected = InspectCheckpoint({directory, true});
    if (!inspected.ok()) return inspected.status();
    shared_->manifest = std::move(inspected).value();
    if (shared_->manifest.architecture != "Gemma4UnifiedForConditionalGeneration" ||
        shared_->manifest.model_type != "gemma4_unified") {
      return Error(StatusCode::kUnsupported,
                   "the inference runtime requires the primary Gemma 4 target; "
                   "assistant checkpoints are inspect-only until the MTP plan is enabled");
    }

    std::uint64_t arena_bytes = 0;
    for (const auto& tensor : shared_->manifest.tensors) {
      auto aligned = AlignUp(arena_bytes, kAlignment);
      if (!aligned.ok()) return aligned.status();
      if (tensor.byte_length > std::numeric_limits<std::uint64_t>::max() - aligned.value()) {
        return Error(StatusCode::kInternal, "weight arena size overflow");
      }
      arena_bytes = aligned.value() + tensor.byte_length;
    }
    auto final_size = AlignUp(arena_bytes, kAlignment);
    if (!final_size.ok()) return final_size.status();
    Status status = shared_->weights.Allocate(
        final_size.value(), "allocate unified model weight arena");
    if (!status.ok()) return status;

    std::uint64_t offset = 0;
    for (const auto& tensor : shared_->manifest.tensors) {
      auto aligned = AlignUp(offset, kAlignment);
      if (!aligned.ok()) return aligned.status();
      DeviceTensor view;
      view.info = &tensor;
      view.data = shared_->weights.data() + aligned.value();
      const auto inserted = shared_->tensors.emplace(tensor.name, view);
      if (!inserted.second) {
        return Error(StatusCode::kDataLoss, "duplicate device tensor: " + tensor.name);
      }
      offset = aligned.value() + tensor.byte_length;
    }

    std::unordered_set<std::string> shards;
    for (const auto& tensor : shared_->manifest.tensors) {
      shards.insert(tensor.source_shard);
    }
    for (const auto& shard : shards) {
      auto mapped = internal::MappedFile::Open(directory / shard);
      if (!mapped.ok()) return mapped.status();
      for (auto& [name, view] : shared_->tensors) {
        (void)name;
        const TensorInfo& tensor = *view.info;
        if (tensor.source_shard != shard) continue;
        if (tensor.byte_offset > mapped.value().size() ||
            tensor.byte_length > mapped.value().size() - tensor.byte_offset) {
          return Error(StatusCode::kDataLoss, "tensor upload range is invalid: " + tensor.name);
        }
        const std::byte* source = mapped.value().data() + tensor.byte_offset;
        bool uploaded = false;
        if (tensor.quantization_class == "NVFP4_PACKED") {
          if (tensor.logical_shape.size() != 2U) {
            return Error(StatusCode::kDataLoss,
                         "invalid NVFP4 packed-weight geometry: " + tensor.name);
          }
          const auto layout = internal::PlanSm120Nvfp4SourceLayout(
              tensor.logical_shape[0], tensor.logical_shape[1]);
          if (!layout.ok()) return layout.status();
          const Status tiled_upload = UploadSm120TiledBlocks(
              view.data,
              std::span<const std::uint8_t>(
                  reinterpret_cast<const std::uint8_t*>(source),
                  static_cast<std::size_t>(tensor.byte_length)),
              layout.value(), 32U, tensor.logical_shape[1] / 2U,
              "upload tiled NVFP4 weight");
          if (!tiled_upload.ok()) return tiled_upload;
          view.sm120_weight_tiled = true;
          uploaded = true;
        } else if (tensor.quantization_class == "NVFP4_LOCAL_SCALE_E4M3") {
          if (tensor.shape.size() != 2U ||
              tensor.shape[1] > std::numeric_limits<std::uint64_t>::max() / 16U) {
            return Error(StatusCode::kDataLoss,
                         "invalid NVFP4 local-scale geometry: " + tensor.name);
          }
          const auto layout = internal::PlanSm120Nvfp4SourceLayout(
              tensor.shape[0], tensor.shape[1] * 16U);
          if (!layout.ok()) return layout.status();
          const Status tiled_upload = UploadSm120TiledBlocks(
              view.data,
              std::span<const std::uint8_t>(
                  reinterpret_cast<const std::uint8_t*>(source),
                  static_cast<std::size_t>(tensor.byte_length)),
              layout.value(), 4U, tensor.shape[1],
              "upload tiled NVFP4 scales");
          if (!tiled_upload.ok()) return tiled_upload;
          view.sm120_scale_tiled = true;
          uploaded = true;
        }
        if (!uploaded) {
          const cudaError_t error = cudaMemcpy(
              view.data, source, static_cast<std::size_t>(tensor.byte_length),
              cudaMemcpyHostToDevice);
          if (error != cudaSuccess)
            return CudaFailure("upload checkpoint tensor", error);
        }
        if (tensor.storage_dtype == "F32" && tensor.byte_length == sizeof(float)) {
          std::uint32_t bits = 0;
          std::memcpy(&bits, source, sizeof(bits));
          view.scalar_f32 = std::bit_cast<float>(bits);
          view.has_scalar_f32 = true;
        }
      }
    }
    const Status bind_status = Bind();
    if (!bind_status.ok()) return bind_status;
    shared_->layers = layers_;
    shared_->embedding = embedding_;
    shared_->final_norm = final_norm_;
    shared_->audio_projection = audio_projection_;
    shared_->vision = vision_;
    return Status::Ok();
  }

  [[nodiscard]] Status ShareWeightsFrom(const Impl& source) {
    if (shared_ != nullptr) {
      return Error(StatusCode::kInvalidArgument,
                   "target model weights are already initialized");
    }
    if (source.shared_ == nullptr) {
      return Error(StatusCode::kInvalidArgument,
                   "cannot share unloaded target model weights");
    }
    shared_ = source.shared_;
    layers_ = shared_->layers;
    embedding_ = shared_->embedding;
    final_norm_ = shared_->final_norm;
    audio_projection_ = shared_->audio_projection;
    vision_ = shared_->vision;
    return Status::Ok();
  }

  [[nodiscard]] const std::array<LayerBinding, kTargetLayerCount>& layers() const { return layers_; }
  [[nodiscard]] const std::uint16_t* embedding() const { return embedding_; }
  [[nodiscard]] const std::uint16_t* final_norm() const { return final_norm_; }
  [[nodiscard]] const std::uint16_t* audio_projection() const {
    return audio_projection_;
  }
  [[nodiscard]] const VisionBinding& vision() const { return vision_; }
  [[nodiscard]] std::uint64_t weight_bytes() const {
    return shared_ == nullptr ? 0U : shared_->weights.bytes();
  }

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
  struct SharedWeights {
    ModelManifest manifest;
    DeviceAllocation weights;
    std::unordered_map<std::string, DeviceTensor> tensors;
    std::array<LayerBinding, kTargetLayerCount> layers{};
    const std::uint16_t* embedding = nullptr;
    const std::uint16_t* final_norm = nullptr;
    const std::uint16_t* audio_projection = nullptr;
    VisionBinding vision{};
  };

  [[nodiscard]] Result<const DeviceTensor*> Tensor(const std::string& name) const {
    const auto found = shared_->tensors.find(name);
    if (found == shared_->tensors.end()) {
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
        !packed.value()->sm120_weight_tiled ||
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
    auto audio_projection = Bf16(
        "model.embed_audio.embedding_projection.weight", kHidden * 640U);
    if (!embedding.ok()) return embedding.status();
    if (!final_norm.ok()) return final_norm.status();
    if (!audio_projection.ok()) return audio_projection.status();
    embedding_ = embedding.value();
    final_norm_ = final_norm.value();
    audio_projection_ = audio_projection.value();

    auto patch_ln1_weight = Bf16("model.vision_embedder.patch_ln1.weight", 6912U);
    auto patch_ln1_bias = Bf16("model.vision_embedder.patch_ln1.bias", 6912U);
    auto patch_dense_weight = Bf16(
        "model.vision_embedder.patch_dense.weight", kHidden * 6912U);
    auto patch_dense_bias = Bf16("model.vision_embedder.patch_dense.bias", kHidden);
    auto patch_ln2_weight = Bf16("model.vision_embedder.patch_ln2.weight", kHidden);
    auto patch_ln2_bias = Bf16("model.vision_embedder.patch_ln2.bias", kHidden);
    auto position_embedding = Bf16(
        "model.vision_embedder.pos_embedding", 1120U * 2U * kHidden);
    auto position_norm_weight = Bf16("model.vision_embedder.pos_norm.weight", kHidden);
    auto position_norm_bias = Bf16("model.vision_embedder.pos_norm.bias", kHidden);
    auto vision_projection = Bf16(
        "model.embed_vision.embedding_projection.weight", kHidden * kHidden);
    if (!patch_ln1_weight.ok()) return patch_ln1_weight.status();
    if (!patch_ln1_bias.ok()) return patch_ln1_bias.status();
    if (!patch_dense_weight.ok()) return patch_dense_weight.status();
    if (!patch_dense_bias.ok()) return patch_dense_bias.status();
    if (!patch_ln2_weight.ok()) return patch_ln2_weight.status();
    if (!patch_ln2_bias.ok()) return patch_ln2_bias.status();
    if (!position_embedding.ok()) return position_embedding.status();
    if (!position_norm_weight.ok()) return position_norm_weight.status();
    if (!position_norm_bias.ok()) return position_norm_bias.status();
    if (!vision_projection.ok()) return vision_projection.status();
    vision_ = VisionBinding{
        patch_ln1_weight.value(), patch_ln1_bias.value(),
        patch_dense_weight.value(), patch_dense_bias.value(),
        patch_ln2_weight.value(), patch_ln2_bias.value(),
        position_embedding.value(), position_norm_weight.value(),
        position_norm_bias.value(), vision_projection.value()};

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

  std::shared_ptr<SharedWeights> shared_;
  std::array<LayerBinding, kTargetLayerCount> layers_{};
  const std::uint16_t* embedding_ = nullptr;
  const std::uint16_t* final_norm_ = nullptr;
  const std::uint16_t* audio_projection_ = nullptr;
  VisionBinding vision_{};
};


LoadedTargetModel::LoadedTargetModel() : impl_(std::make_unique<Impl>()) {}
LoadedTargetModel::LoadedTargetModel(LoadedTargetModel&&) noexcept = default;
LoadedTargetModel& LoadedTargetModel::operator=(LoadedTargetModel&&) noexcept = default;
LoadedTargetModel::~LoadedTargetModel() = default;

Status LoadedTargetModel::Load(const std::filesystem::path& directory) {
  return impl_->Load(directory);
}
Status LoadedTargetModel::ShareWeightsFrom(const LoadedTargetModel& source) {
  return impl_->ShareWeightsFrom(*source.impl_);
}
const std::array<LayerBinding, kTargetLayerCount>& LoadedTargetModel::layers() const {
  return impl_->layers();
}
const std::uint16_t* LoadedTargetModel::embedding() const { return impl_->embedding(); }
const std::uint16_t* LoadedTargetModel::final_norm() const { return impl_->final_norm(); }
const std::uint16_t* LoadedTargetModel::audio_projection() const {
  return impl_->audio_projection();
}
const VisionBinding& LoadedTargetModel::vision() const { return impl_->vision(); }
std::uint64_t LoadedTargetModel::weight_bytes() const { return impl_->weight_bytes(); }
void LoadedTargetModel::SetLayerBf16Cache(std::size_t layer, float* key, float* value) {
  impl_->SetLayerBf16Cache(layer, key, value);
}
void LoadedTargetModel::SetLayerFp8Cache(std::size_t layer, std::uint8_t* key,
                                         std::uint8_t* value) {
  impl_->SetLayerFp8Cache(layer, key, value);
}

}  // namespace gem16::internal
