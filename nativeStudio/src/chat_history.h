#pragma once

#include "types.h"

#include <vector>

namespace gem16::studio {

// Removes the most recent user message and every response following it. This
// mirrors the legacy Studio undo-turn action; callers must also invalidate any
// resident server session because its KV state cannot be rolled back.
[[nodiscard]] bool RemoveLastExchange(std::vector<ChatMessage>& messages);

}  // namespace gem16::studio
