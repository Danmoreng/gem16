#pragma once

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "gem16/status.h"
#include "gem16/types.h"
#include "model/config.h"

namespace gem16::internal {

enum class TensorRole {
  kTiedEmbeddingAndOutput,
  kFinalNorm,
  kAttentionQProjection,
  kAttentionKProjection,
  kAttentionVProjection,
  kAttentionOProjection,
  kAttentionQNorm,
  kAttentionKNorm,
  kAttentionKScale,
  kAttentionVScale,
  kSharedMlpGate,
  kSharedMlpUp,
  kSharedMlpDown,
  kRouterScale,
  kRouterProjection,
  kRouterPerExpertScale,
  kRoutedExpertGateUp,
  kRoutedExpertGate,
  kRoutedExpertUp,
  kRoutedExpertDown,
  kInputLayerNorm,
  kPostAttentionLayerNorm,
  kPreFeedForwardLayerNorm,
  kPreFeedForwardLayerNorm2,
  kPostFeedForwardLayerNorm,
  kPostFeedForwardLayerNorm1,
  kPostFeedForwardLayerNorm2,
  kLayerScalar,
  kVisionProjection,
  kVisionEmbedding,
  kVisionAttention,
  kVisionMlp,
  kVisionNorm,
};

enum class ResidencyClass {
  kCompilerSourceText,
  kExternalReferenceText,
  kImmutableDeviceText,
  kCompileExcludedVision,
};

enum class Gemma4Moe26BInventoryProfile {
  kSourceBf16,
  kExternalUnslothNvfp4,
};

enum class Gemma4Moe26BHeadFormat {
  kQ4_0,
  kNvfp4,
};

[[nodiscard]] std::string_view TensorRoleName(TensorRole role);
[[nodiscard]] std::string_view ResidencyClassName(ResidencyClass residency);
[[nodiscard]] std::string_view Gemma4Moe26BInventoryProfileName(
    Gemma4Moe26BInventoryProfile profile);
[[nodiscard]] std::string_view Gemma4Moe26BHeadFormatName(
    Gemma4Moe26BHeadFormat format);

[[nodiscard]] Result<std::vector<TensorInfo>>
BuildGemma4Moe26BSourceBf16Contract();
[[nodiscard]] Result<std::vector<TensorInfo>>
BuildGemma4Moe26BExternalUnslothNvfp4Contract();
[[nodiscard]] Result<std::vector<TensorInfo>>
BuildGemma4Moe26BCompiledHybridContract(Gemma4Moe26BHeadFormat head_format);

[[nodiscard]] Result<Gemma4Moe26BInventoryProfile>
ValidateAndAnnotateGemma4Moe26BInventory(
    const ModelConfig& config, std::vector<TensorInfo>* tensors);
[[nodiscard]] Status ValidateGemma4Moe26BCompiledHybridInventory(
    std::span<const TensorInfo> tensors, Gemma4Moe26BHeadFormat head_format);
[[nodiscard]] Status ValidateAndAnnotateGemma4Moe26BCompiledHybridInventory(
    std::vector<TensorInfo>* tensors, Gemma4Moe26BHeadFormat head_format);

[[nodiscard]] Result<std::uint64_t> Gemma4Moe26BAlignedArenaBytes(
    std::span<const TensorInfo> tensors, std::uint64_t alignment = 256);

}  // namespace gem16::internal
