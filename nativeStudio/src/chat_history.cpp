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
      message.error_message = event.value;
      message.streaming = false;
      break;
    case ChatEvent::Kind::kFinished:
      message.streaming = false;
      if (event.value == "cancelled") {
        message.error = true;
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

}  // namespace gem16::studio
