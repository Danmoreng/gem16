#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>

#include "gem16/status.h"

namespace gem16::internal {

struct AssistantLayerBinding {
  bool global = false;
  std::uint64_t query_elements = 0;
  const std::uint16_t* input_norm = nullptr;
  const std::uint16_t* q_projection = nullptr;
  const std::uint16_t* q_norm = nullptr;
  const std::uint16_t* o_projection = nullptr;
  const std::uint16_t* post_attention_norm = nullptr;
  const std::uint16_t* pre_feedforward_norm = nullptr;
  const std::uint16_t* gate_projection = nullptr;
  const std::uint16_t* up_projection = nullptr;
  const std::uint16_t* down_projection = nullptr;
  const std::uint16_t* post_feedforward_norm = nullptr;
  const std::uint16_t* layer_scalar = nullptr;
};

struct AssistantBindings {
  const std::uint16_t* embedding = nullptr;
  const std::uint16_t* pre_projection = nullptr;
  const std::uint16_t* post_projection = nullptr;
  const std::uint16_t* final_norm = nullptr;
  std::array<AssistantLayerBinding, 4> layers{};
};

// Owns the official BF16 MTP assistant in one independent, fixed-address
// device arena. The target model and its KV cache remain separate owners.
class AssistantModel {
 public:
  AssistantModel();
  AssistantModel(const AssistantModel&) = delete;
  AssistantModel& operator=(const AssistantModel&) = delete;
  ~AssistantModel();

  [[nodiscard]] Status Load(const std::filesystem::path& directory);
  [[nodiscard]] bool loaded() const;
  [[nodiscard]] std::uint64_t arena_bytes() const;
  [[nodiscard]] std::uint64_t source_bytes() const;
  [[nodiscard]] std::uint64_t tensor_count() const;
  [[nodiscard]] const AssistantBindings& bindings() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gem16::internal
