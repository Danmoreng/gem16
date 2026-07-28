#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>

#include "gem16/engine.h"
#include "gem16/status.h"

namespace gem16::internal {

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
  std::array<LayerStateCapture, 48U> layers{};
  std::size_t elements = 0;
};

[[nodiscard]] StateCaptureLayout MakeStateCaptureLayout();
[[nodiscard]] Status WriteStateDump(const std::filesystem::path& path,
                                    std::uint64_t position,
                                    KvCacheMode kv_cache_mode,
                                    std::span<const float> captured_state);

}  // namespace gem16::internal
