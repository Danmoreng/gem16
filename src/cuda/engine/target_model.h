#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>

#include "gem16/status.h"

namespace gem16::internal {

constexpr std::size_t kTargetLayerCount = 48U;

struct Fp8Binding {
  const std::uint8_t* weight = nullptr;
  const std::uint16_t* scales = nullptr;
  std::uint64_t rows = 0;
  std::uint64_t contracting = 0;
};

struct Nvfp4Binding {
  const std::uint8_t* packed_weight = nullptr;
  const std::uint8_t* scales = nullptr;
  const float* input_divisor_device = nullptr;
  float input_divisor = 0.0F;
  float weight_divisor = 0.0F;
  std::uint64_t rows = 0;
  std::uint64_t contracting = 0;
};

struct VisionBinding {
  const std::uint16_t* patch_ln1_weight = nullptr;
  const std::uint16_t* patch_ln1_bias = nullptr;
  const std::uint16_t* patch_dense_weight = nullptr;
  const std::uint16_t* patch_dense_bias = nullptr;
  const std::uint16_t* patch_ln2_weight = nullptr;
  const std::uint16_t* patch_ln2_bias = nullptr;
  const std::uint16_t* position_embedding = nullptr;
  const std::uint16_t* position_norm_weight = nullptr;
  const std::uint16_t* position_norm_bias = nullptr;
  const std::uint16_t* projection_weight = nullptr;
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

class LoadedTargetModel {
 public:
  LoadedTargetModel();
  LoadedTargetModel(const LoadedTargetModel&) = delete;
  LoadedTargetModel& operator=(const LoadedTargetModel&) = delete;
  LoadedTargetModel(LoadedTargetModel&&) noexcept;
  LoadedTargetModel& operator=(LoadedTargetModel&&) noexcept;
  ~LoadedTargetModel();

  [[nodiscard]] Status Load(const std::filesystem::path& directory);
  // Creates a lightweight model view which shares immutable device weights
  // while retaining independent, session-owned KV-cache bindings.
  [[nodiscard]] Status ShareWeightsFrom(const LoadedTargetModel& source);
  [[nodiscard]] const std::array<LayerBinding, kTargetLayerCount>& layers() const;
  [[nodiscard]] const std::uint16_t* embedding() const;
  [[nodiscard]] const std::uint16_t* final_norm() const;
  [[nodiscard]] const std::uint16_t* audio_projection() const;
  [[nodiscard]] const VisionBinding& vision() const;
  [[nodiscard]] std::uint64_t weight_bytes() const;
  void SetLayerBf16Cache(std::size_t layer, float* key, float* value);
  void SetLayerFp8Cache(std::size_t layer, std::uint8_t* key,
                        std::uint8_t* value);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gem16::internal
