#pragma once

#include "gem16/memory.h"
#include "gem16/status.h"
#include "gem16/types.h"
#include "model/config.h"

namespace gem16::internal {

[[nodiscard]] Result<std::uint64_t> ContextTokens(ContextProfile profile);
[[nodiscard]] Result<MemoryPlan> BuildMemoryPlan(
    const ModelConfig& config,
    const ModelManifest& manifest,
    const MemoryPlanOptions& options);

}  // namespace gem16::internal
