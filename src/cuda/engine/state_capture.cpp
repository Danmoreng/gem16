#include "cuda/engine/state_capture.h"

#include <array>
#include <bit>
#include <cstdint>
#include <fstream>
#include <string>
#include <utility>

namespace gem16::internal {
namespace {

constexpr std::size_t kLayers = 48U;
constexpr std::size_t kHidden = 3840U;
constexpr std::size_t kIntermediate = 15360U;

Status Error(StatusCode code, std::string message) {
  return Status(code, std::move(message));
}

}  // namespace

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

}  // namespace gem16::internal
