#pragma once

#include "types.h"

#include <vector>

namespace gem16::studio {

// Transport status is displayed separately and never becomes model content.
void ApplyChatEvent(ChatMessage& message, const ChatEvent& event);

// Removes the most recent user message and every response following it. This
// mirrors the legacy Studio undo-turn action; callers must also invalidate any
// resident server session because its KV state cannot be rolled back.
[[nodiscard]] bool RemoveLastExchange(std::vector<ChatMessage>& messages);

// Tool IDs may repeat in later assistant turns. Match only this call batch.
const ChatMessage* FindToolResult(const std::vector<ChatMessage>& messages,
                                  std::size_t assistant_index,
                                  const std::string& id);

}  // namespace gem16::studio
