#include "chat_history.h"

#include <algorithm>
#include <iterator>

namespace gem16::studio {

void ApplyChatEvent(ChatMessage& message, const ChatEvent& event) {
  switch (event.kind) {
    case ChatEvent::Kind::kText: message.content += event.value; break;
    case ChatEvent::Kind::kReasoning: message.reasoning += event.value; break;
    case ChatEvent::Kind::kError:
      message.error = true;
      message.interrupted = false;
      message.error_message = event.value;
      message.streaming = false;
      break;
    case ChatEvent::Kind::kFinished:
      message.streaming = false;
      if (event.value == "cancelled") {
        message.error = true;
        message.interrupted = true;
        message.error_message = "Generation stopped.";
      }
      break;
    default: break;
  }
}

bool RemoveLastExchange(std::vector<ChatMessage>& messages) {
  const auto user = std::find_if(messages.rbegin(), messages.rend(),
                                 [](const ChatMessage& message) {
                                   return message.role == "user";
                                 });
  if (user == messages.rend()) return false;
  const std::size_t user_index =
      static_cast<std::size_t>(std::distance(user, messages.rend()) - 1);
  messages.resize(user_index);
  return true;
}

const ChatMessage* FindToolResult(const std::vector<ChatMessage>& messages,
                                  std::size_t assistant_index,
                                  const std::string& id) {
  if (assistant_index >= messages.size() ||
      messages[assistant_index].role != "assistant")
    return nullptr;
  for (std::size_t i = assistant_index + 1;
       i < messages.size() && messages[i].role == "tool"; ++i)
    if (messages[i].tool_call_id == id) return &messages[i];
  return nullptr;
}

}  // namespace gem16::studio
