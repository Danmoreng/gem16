#pragma once

#include "api_client.h"
#include "audio_recorder.h"
#include "image_texture.h"
#include "svg_preview.h"
#include "model_manager.h"
#include "server_manager.h"
#include "types.h"

#include <array>
#include <chrono>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

namespace gem16::studio {

class StudioApp final {
 public:
  StudioApp(StudioSettings settings, float automatic_ui_scale);
  ~StudioApp();

  void Render();
  void SetNavigationFlameTexture(ImTextureID texture) {
    navigation_flame_texture_ = texture;
  }
  [[nodiscard]] bool DarkTheme() const { return settings_.dark_theme; }

 private:
  void ApplyTheme() const;
  void DrainChatEvents();
  void DrawSidebar();
  void DrawChat();
  void DrawModels();
  void DrawServer();
  void DrawSettings();
  void DrawMessage(const ChatMessage& message, std::size_t index);
  void DrawAppLogo(ImVec2 position, float size);
  void DrawAttachmentGallery(const std::vector<MediaAttachment>& attachments,
                             std::vector<MediaAttachment>* removable = nullptr);
  [[nodiscard]] ImageTexture* AttachmentTexture(
      const MediaAttachment& attachment);
  void PruneAttachmentTextures();
  void SendMessage();
  void RetryLastRequest();
  void ClearChat();
  void RemoveLastExchange();
  void AddAttachments(const std::vector<std::filesystem::path>& paths);
  void FinishRecording();
  void SelectProfile(ModelProfile profile);
  void ApplyUiScale(float configured_scale);
  void SyncBuffersFromSettings();
  void SyncSettingsFromBuffers();

  StudioSettings settings_;
  ServerManager server_;
  ModelManager models_;
  ApiClient api_;
  AudioRecorder recorder_;
  Screen screen_ = Screen::kChat;
  std::vector<ChatMessage> messages_;
  std::vector<MediaAttachment> pending_attachments_;
  ImageTexture logo_texture_;
  SvgPreviewCache svg_previews_;
  std::unordered_map<std::uint64_t, std::unique_ptr<ImageTexture>>
      attachment_textures_;
  std::string attachment_error_;
  std::string session_id_;
  std::array<char, 65536> composer_{};
  std::array<char, 4096> executable_{};
  std::array<char, 4096> model_directory_{};
  std::array<char, 4096> assistant_directory_{};
  std::array<char, 4096> vision_directory_{};
  std::array<char, 256> model_name_{};
  std::array<char, 256> host_{};
  std::array<char, 16384> system_prompt_{};
  bool scroll_to_bottom_ = false;
  bool auto_follow_ = true;
  bool show_reasoning_ = true;
  std::unordered_set<std::size_t> expanded_reasoning_;
  std::size_t copied_message_index_ = static_cast<std::size_t>(-1);
  double copied_message_at_ = -100.0;
  float sidebar_width_ = 214.0f;
  ImTextureID navigation_flame_texture_ = ImTextureID_Invalid;
  float ui_scale_ = 1.0f;
  float automatic_ui_scale_ = 1.0f;
  std::chrono::steady_clock::time_point generation_started_{};
  std::chrono::steady_clock::time_point first_token_at_{};
  std::chrono::steady_clock::time_point generation_finished_{};
  std::int64_t prompt_tokens_ = 0;
  std::int64_t completion_tokens_ = 0;
  std::int64_t streamed_chunks_ = 0;
  std::optional<PerformanceStats> performance_;
  bool retry_requested_ = false;
};

}  // namespace gem16::studio
