#include "chat_history.h"

#include <algorithm>
#include <iterator>

namespace gem16::studio {

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
