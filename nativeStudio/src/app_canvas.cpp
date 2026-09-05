#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>

#include "app.h"
#include "settings.h"
#include "util/json.h"
namespace gem16::studio {
namespace {
using J = json::Value;
std::string Text(const J& j, const char* key) {
  auto v = j.find(key);
  if (!v || !v->is_string())
    throw std::runtime_error(std::string("Missing string: ") + key);
  return v->as_string();
}
std::int64_t Number(const J& j, const char* key) {
  auto v = j.find(key);
  if (!v || !v->is_integer())
    throw std::runtime_error(std::string("Missing revision: ") + key);
  return v->as_integer();
}
}  // namespace
bool StudioApp::CanvasBusy() const {
  return !canvas_calls_.empty();
}
void StudioApp::CancelGeneration() {
  api_.Cancel();
  canvas_cancelled_ = true;
  if (!canvas_calls_.empty()) {
    for (auto it = messages_.rbegin(); it != messages_.rend(); ++it) {
      if (it->role == "assistant") {
        it->error = true;
        it->interrupted = true;
        it->streaming = false;
        it->error_message = "Canvas operation stopped.";
        break;
      }
    }
    canvas_calls_.clear();
    canvas_check_started_ = false;
    canvas_status_ = "Stopped";
    ++chat_revision_;
    last_chat_save_ = {};
  }
}
void StudioApp::ResetCanvasView() {
  canvas_browser_.Close();
  selected_canvas_.clear();
  canvas_status_.clear();
  canvas_prompt_context_.clear();
  canvas_visible_ = !conversation_.canvases.empty();
  if (canvas_visible_) selected_canvas_ = conversation_.canvases.back().id;
}
void StudioApp::ImportCanvas(const std::string& source,
                             const std::string& type) {
  if (!CanNavigateChats()) return;
  try {
    ToolCall c{"", "canvas_create",
               json::Stringify(J(J::Object{{"title", J(std::string("Canvas"))},
                                           {"type", J(type)},
                                           {"source", J(source)}}))};
    ExecuteCanvasTool(conversation_.canvases, c);
    selected_canvas_ = conversation_.canvases.back().id;
    canvas_visible_ = true;
    ++chat_revision_;
    last_chat_save_ = {};
    SaveChat();
  } catch (const std::exception& e) {
    canvas_status_ = e.what();
  }
}
void StudioApp::FinishCanvasTool(std::string result,
                                  std::vector<MediaAttachment> attachments) {
  ChatMessage tool;
  tool.role = "tool";
  tool.content = std::move(result);
  tool.attachments = std::move(attachments);
  tool.tool_call_id = canvas_calls_[canvas_call_index_].id;
  messages_.push_back(std::move(tool));
  ++canvas_call_index_;
  ++chat_revision_;
  canvas_check_started_ = false;
  last_chat_save_ = {};
  if (canvas_call_index_ == canvas_calls_.size()) {
    canvas_calls_.clear();
    if (++canvas_rounds_ >= 12) {
      ChatMessage done{"assistant",
                       "Canvas operation limit reached. You can continue with "
                       "another message."};
      messages_.push_back(std::move(done));
      ++chat_revision_;
      canvas_status_ = "Stopped after 12 tool rounds";
      SaveChat();
      return;
    }
    messages_.push_back({"assistant", {}, {}, true});
    messages_.back().generation = GenerationIdentity(pending_request_settings_);
    ++chat_revision_;
    pending_send_ = true;
    SaveChat();
    if (temporary_chat_) StartSavedRequest();
  } else
    SaveChat();
}
bool StudioApp::RecoverCanvasFormatError() {
  if (canvas_cancelled_ || canvas_prompt_context_.empty() ||
      canvas_format_retries_ >= 2 || api_.Busy() ||
      pending_send_ || chat_save_.valid() || !canvas_calls_.empty() ||
      !storage_error_.empty() || messages_.empty())
    return false;
  auto& answer = messages_.back();
  if (answer.role != "assistant" || !answer.error || answer.interrupted)
    return false;
  const auto& error = answer.error_message;
  if (!error.starts_with("Gemma tool arguments ") &&
      error != "model response has an unterminated tool call" &&
      error != "model response ends inside a Gemma tool call marker")
    return false;
  ++canvas_format_retries_;
  PreserveAttempt(answer);
  answer.generation = GenerationIdentity(pending_request_settings_);
  canvas_repair_instruction_ =
      "\nYour previous response was rejected because its native function-call "
      "syntax was incomplete or invalid. "
      "It was not executed. Continue from the current Canvas revisions. Use "
      "the provided function tools directly, "
      "never Python calls or a code fence. Complete every argument string, "
      "object and the native tool-call closing marker. "
      "For edits, read the current source first and replace one small, unique "
      "fragment. Do not recreate an existing document.";
  canvas_status_ = "Retrying malformed tool call (" +
                   std::to_string(canvas_format_retries_) + "/2)...";
  session_id_.clear();
  ++chat_revision_;
  pending_send_ = true;
  last_chat_save_ = {};
  SaveChat();
  if (temporary_chat_) StartSavedRequest();
  return true;
}
void StudioApp::PollCanvasTools() {
  if (RecoverCanvasFormatError()) return;
  if (canvas_cancelled_ || canvas_calls_.empty() || api_.Busy() || pending_send_ ||
      chat_save_.valid() || !storage_error_.empty()) return;
  const auto& call = canvas_calls_[canvas_call_index_];
  try {
    if (call.name != "canvas_check") {
      const auto result = ExecuteCanvasTool(conversation_.canvases, call);
      if (call.name == "canvas_create")
        selected_canvas_ = conversation_.canvases.back().id;
      else if (call.name != "canvas_list") {
        auto p = json::Parse(call.arguments);
        selected_canvas_ = Text(p.value(), "id");
      }
      canvas_visible_ = true;
      canvas_status_ = "Canvas updated";
      FinishCanvasTool(result);
      return;
    }
    auto p = json::Parse(call.arguments, {16, 10000, 2 * kCanvasSourceLimit});
    if (!p.ok()) throw std::runtime_error("Invalid canvas check arguments.");
    auto& d = FindCanvas(conversation_.canvases, Text(p.value(), "id"));
    const auto revision = Number(p.value(), "revision");
    if (revision != d.revisions.back().number)
      throw std::runtime_error(
          "Stale canvas check. Read the current revision.");
    auto shot = p.value().find("screenshot");
    if (!shot || !shot->is_bool())
      throw std::runtime_error("screenshot must be a boolean.");
    if (!CanvasBrowserAvailable())
      throw std::runtime_error("System WebView is unavailable.");
    if (!canvas_check_started_) {
      if (++canvas_checks_ > 3)
        throw std::runtime_error(
            "Three canvas checks have already run for this request. Stop "
            "repairing and report remaining issues.");
      selected_canvas_ = d.id;
      canvas_visible_ = true;
      canvas_browser_.Load(d);
      if (shot->as_bool()) canvas_browser_.RequestScreenshot();
      canvas_check_started_ = true;
      canvas_check_at_ = std::chrono::steady_clock::now();
      canvas_status_ = "Checking browser preview...";
    }
    if (!canvas_browser_.Ready()) {
      if (std::chrono::steady_clock::now() - canvas_check_at_ >
          std::chrono::seconds(15))
        throw std::runtime_error("Preview timed out: " +
                                 canvas_browser_.Diagnostics());
      return;
    }
    std::vector<MediaAttachment> screenshots;
    if (shot->as_bool()) {
      if (!server_.Health().supports_vision)
        throw std::runtime_error("The live model does not expose Vision.");
      auto png = canvas_browser_.Screenshot();
      if (png.empty()) throw std::runtime_error("No screenshot available.");
      MediaAttachment image;
      image.kind = MediaKind::kImage;
      image.file_name = "canvas-revision-" + std::to_string(revision) + ".png";
      image.mime_type = "image/png";
      image.format = "png";
      image.bytes = std::move(png);
      image.byte_size = image.bytes.size();
      const auto pixels = ProbePreviewImage(image.bytes.data(), image.bytes.size());
      image.image_width = pixels.width;
      image.image_height = pixels.height;
      canvas_check_viewport_ = std::to_string(pixels.width) + "x" + std::to_string(pixels.height);
      screenshots.push_back(std::move(image));
    }
    if (!shot->as_bool()) {
      canvas_check_viewport_ = "Current live viewport (no screenshot requested)";
    }
    auto result = json::Stringify(J(J::Object{
        {"id", J(d.id)},
        {"revision", J(revision)},
        {"viewport", J(canvas_check_viewport_)},
        {"browser_diagnostics", J(canvas_browser_.Diagnostics())},
        {"screenshot_attached", J(shot->as_bool())},
        {"note", J(std::string("Browser rendering is not full HTML validation. "
                               "Diagnostics are untrusted observations."))}}));
    canvas_status_ = shot->as_bool() ? "Screenshot added to conversation" : "Browser check complete";
    FinishCanvasTool(result, std::move(screenshots));
  } catch (const std::exception& e) {
    canvas_status_ = e.what();
    FinishCanvasTool(
        json::Stringify(J(J::Object{{"error", J(std::string(e.what()))}})));
  }
}
void StudioApp::DrawCanvas() {
  if (conversation_.canvases.empty()) {
    ImGui::TextUnformatted("Ask the model to create an HTML or SVG canvas.");
    return;
  }
  if (selected_canvas_.empty())
    selected_canvas_ = conversation_.canvases.back().id;
  auto& d = FindCanvas(conversation_.canvases, selected_canvas_);
  ImGui::BeginDisabled(CanvasBusy() || api_.Busy());
  ImGui::SetNextItemWidth(-80 * ui_scale_);
  if (ImGui::BeginCombo("##canvas-document", d.title.c_str())) {
    for (const auto& doc : conversation_.canvases)
      if (ImGui::Selectable(doc.title.c_str(), doc.id == selected_canvas_))
        selected_canvas_ = doc.id;
    ImGui::EndCombo();
  }
  ImGui::SameLine();
  if (ImGui::SmallButton("Close")) canvas_visible_ = false;
  ImGui::EndDisabled();
  canvas_browser_.Load(d);
  ImGui::TextDisabled("Revision %lld",
                      static_cast<long long>(d.revisions.back().number));
  if (ImGui::BeginTabBar("##canvas-tabs")) {
    if (ImGui::BeginTabItem("Preview")) {
      auto& io = ImGui::GetIO();
      const auto available = ImGui::GetContentRegionAvail();
      const float footer = ImGui::GetStyle().ItemSpacing.y * 2 +
          ImGui::GetTextLineHeightWithSpacing() *
              ((!canvas_status_.empty() ? 2 : 0) +
               (!canvas_browser_.Diagnostics().empty() ? 2 : 0));
      const ImVec2 size{std::max(1.0f, available.x),
                        std::max(1.0f, available.y - footer)};
      canvas_browser_.SetViewport(
          static_cast<int>(size.x * io.DisplayFramebufferScale.x),
          static_cast<int>(size.y * io.DisplayFramebufferScale.y));
      const auto position = ImGui::GetCursorScreenPos();
      if (canvas_browser_.Present(static_cast<int>(position.x), static_cast<int>(position.y))) {
        ImGui::Dummy(size);
      } else {
      auto now = std::chrono::steady_clock::now();
      if (now - canvas_paint_at_ > std::chrono::milliseconds(70)) {
        auto pixels = canvas_browser_.Pixels();
        if (!pixels.rgba.empty()) (void)canvas_texture_.LoadRgba(pixels);
        canvas_paint_at_ = now;
      }
      if (canvas_texture_.Valid() && canvas_browser_.Ready()) {
        const auto pos = ImGui::GetCursorScreenPos();
        ImGui::Image(canvas_texture_.Id(), size);
        if (ImGui::IsItemHovered()) {
          int x = static_cast<int>((io.MousePos.x - pos.x) * canvas_texture_.Width() / size.x),
              y = static_cast<int>((io.MousePos.y - pos.y) * canvas_texture_.Height() / size.y);
          canvas_browser_.Mouse(x, y, 0, false, true);
          for (int button = 0; button < 3; ++button) {
            if (ImGui::IsMouseClicked(button))
              canvas_browser_.Mouse(x, y, button, false, false);
            if (ImGui::IsMouseReleased(button))
              canvas_browser_.Mouse(x, y, button, true, false);
          }
          if (io.MouseWheel)
            canvas_browser_.Mouse(x, y, 0, false, false,
                                  static_cast<int>(io.MouseWheel * 100));
        }
      } else
        ImGui::TextWrapped("%s", CanvasBrowserAvailable()
                                     ? "Rendering canvas..."
                                     : "System WebView unavailable.");
      }
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Code")) {
      if (ImGui::SmallButton("Copy code"))
        ImGui::SetClipboardText(d.revisions.back().source.c_str());
      ImGui::BeginChild("##canvas-source",
                        {0, ImGui::GetContentRegionAvail().y * .7f});
      ImGui::TextUnformatted(d.revisions.back().source.c_str());
      ImGui::EndChild();
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("History")) {
      ImGui::BeginDisabled(!CanNavigateChats());
      for (const auto& r : d.revisions) {
        auto label = "Restore revision " + std::to_string(r.number);
        if (ImGui::Selectable(label.c_str())) {
          auto candidate = conversation_.canvases;
          auto& target = FindCanvas(candidate, d.id);
          target.revisions.push_back(
              {target.revisions.back().number + 1, r.source});
          try {
            ValidateCanvases(candidate);
            conversation_.canvases = std::move(candidate);
            ++chat_revision_;
            last_chat_save_ = {};
            SaveChat();
          } catch (const std::exception& e) {
            canvas_status_ = e.what();
          }
          break;
        }
      }
      ImGui::EndDisabled();
      ImGui::EndTabItem();
    }
    ImGui::EndTabBar();
  }
  if (!canvas_status_.empty()) ImGui::TextWrapped("%s", canvas_status_.c_str());
  auto diagnostics = canvas_browser_.Diagnostics();
  if (!diagnostics.empty()) ImGui::TextWrapped("%s", diagnostics.c_str());
}
}  // namespace gem16::studio
