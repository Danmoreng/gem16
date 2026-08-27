#pragma once

#include "api_client.h"
#include "model_manager.h"
#include "server_manager.h"
#include "types.h"

#include <array>
#include <string>
#include <unordered_set>
#include <vector>

namespace gem16::studio {

class StudioApp final {
 public:
  StudioApp();
  ~StudioApp();

  void Render();
  [[nodiscard]] bool DarkTheme() const { return settings_.dark_theme; }

 private:
  void ApplyTheme() const;
  void DrainChatEvents();
  void DrawSidebar();
  void DrawHeader();
  void DrawChat();
  void DrawModels();
  void DrawServer();
  void DrawSettings();
  void DrawMessage(const ChatMessage& message, std::size_t index);
  void SendMessage();
  void ClearChat();
  void RemoveLastExchange();
  void SelectProfile(ModelProfile profile);
  void SyncBuffersFromSettings();
  void SyncSettingsFromBuffers();

  StudioSettings settings_;
  ServerManager server_;
  ModelManager models_;
  ApiClient api_;
  Screen screen_ = Screen::kChat;
  std::vector<ChatMessage> messages_;
  std::string session_id_;
  std::array<char, 65536> composer_{};
  std::array<char, 4096> executable_{};
  std::array<char, 4096> model_directory_{};
  std::array<char, 4096> assistant_directory_{};
  std::array<char, 256> model_name_{};
  std::array<char, 256> host_{};
  std::array<char, 16384> system_prompt_{};
  bool scroll_to_bottom_ = false;
  bool show_reasoning_ = true;
  std::unordered_set<std::size_t> expanded_reasoning_;
  std::size_t copied_message_index_ = static_cast<std::size_t>(-1);
  double copied_message_at_ = -100.0;
  float sidebar_width_ = 214.0f;
};

}  // namespace gem16::studio
