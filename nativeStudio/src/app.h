#pragma once

#include <array>
#include <chrono>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "api_client.h"
#include "audio_recorder.h"
#include "canvas_browser.h"
#include "chat_store.h"
#include "image_texture.h"
#include "model_manager.h"
#include "server_manager.h"
#include "svg_preview.h"
#include "types.h"

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
  friend struct StudioAppTestAccess;
  void ApplyTheme() const;
  void DrainChatEvents();
  void DrawSidebar();
  enum class ServerAction { kStart, kStop, kRestart };
  void RequestServerAction(ServerAction action);
  void PollServerAction();
  bool ServerActionPending() const;
  void ResetUsage();
  std::int64_t ContextTokens() const;
  void DrawChatLibrary();
  void PollChatStore();
  void SaveChat();
  void NewConversation(bool temporary = false);
  void StartSavedRequest();
  bool CanNavigateChats() const;
  std::string ChatCompatibilityError() const;
  void DrawChat();
  void DrawCanvas();
  void PollCanvasTools();
  void CancelGeneration();
  bool CanvasBusy() const;
  void FinishCanvasTool(std::string result);
  void ImportCanvas(const std::string& source, const std::string& type);
  void ResetCanvasView();
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
  void ActivateModel(const ServerConfig& config, bool keep_chat);
  void RestoreChatModel();
  bool ModelSelectionReady(const ServerConfig& config) const;
  void ApplyUiScale(float configured_scale);
  void SyncBuffersFromSettings();
  void SyncSettingsFromBuffers();

  StudioSettings settings_;
  ChatStore chat_store_;
  StudioSettings pending_request_settings_;
  Conversation conversation_;
  std::vector<ConversationSummary> chat_list_;
  std::future<void> chat_save_;
  std::array<char, 513> chat_search_{};
  std::string listed_search_;
  bool listed_archived_ = false;
  std::chrono::steady_clock::time_point search_changed_{};
  std::int64_t jump_to_message_ = -1;
  std::future<Conversation> chat_load_;
  std::future<std::vector<ConversationSummary>> chat_listing_;
  std::uint64_t chat_revision_ = 0, saved_revision_ = 0, saving_revision_ = 0;
  std::chrono::steady_clock::time_point last_chat_save_{};
  std::string storage_error_;
  bool temporary_chat_ = false, pending_send_ = false, restore_latest_ = true;
  bool show_archived_ = false, delete_chat_requested_ = false;

  ServerManager server_;
  std::optional<ServerAction> pending_server_action_;
  std::future<void> server_action_;
  std::string server_action_error_;
  ModelManager models_;
  ApiClient api_;
  ApiClient canvas_vision_;
  CanvasBrowser canvas_browser_;
  ImageTexture canvas_texture_;
  std::string canvas_check_viewport_;
  std::string selected_canvas_, canvas_status_, canvas_visual_result_,
      canvas_prompt_context_;
  std::vector<ToolCall> canvas_calls_;
  std::size_t canvas_call_index_ = 0;
  int canvas_rounds_ = 0, canvas_checks_ = 0, canvas_format_retries_ = 0;
  std::string canvas_repair_instruction_;
  bool RecoverCanvasFormatError();
  bool canvas_visible_ = false, canvas_check_started_ = false,
       canvas_vision_started_ = false;
  bool canvas_cancelled_ = false;
  std::chrono::steady_clock::time_point canvas_check_at_{}, canvas_paint_at_{};
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
  bool usage_received_ = false;
  std::int64_t completion_tokens_ = 0;
  std::int64_t streamed_chunks_ = 0;
  std::optional<PerformanceStats> performance_;
  bool retry_requested_ = false;
  std::optional<ModelProfile> pending_profile_;
  std::optional<ServerConfig> pending_chat_model_;
  std::string model_selection_error_;
};

}  // namespace gem16::studio
