#include "api_client.h"

#include "media_loader.h"
#include "util/json.h"

#include <algorithm>
#include <chrono>
#include <sstream>

namespace gem16::studio {
namespace {

std::string UserTextWithDocuments(const ChatMessage& message) {
  std::string result = message.content;
  for (const MediaAttachment& attachment : message.attachments) {
    if (attachment.kind != MediaKind::kDocument) continue;
    if (!result.empty()) result += "\n\n";
    std::string safe_name = attachment.file_name;
    std::replace(safe_name.begin(), safe_name.end(), '\n', ' ');
    std::replace(safe_name.begin(), safe_name.end(), '\r', ' ');
    result += "--- Begin attached document: " + safe_name + " ---\n";
    result += attachment.document_text;
    result += "\n--- End attached document: " + safe_name + " ---";
  }
  return result;
}

}  // namespace

std::string BuildChatPayload(const ServerConfig& server,
                             const GenerationConfig& generation,
                             const std::vector<ChatMessage>& messages) {
  std::ostringstream output;
  output << "{\"model\":" << json::Quote(server.model_name)
         << ",\"stream\":true,\"stream_options\":{\"include_usage\":true}"
         << ",\"max_completion_tokens\":" << generation.max_output_tokens
         << ",\"reasoning_effort\":" << json::Quote(generation.reasoning_effort)
         << ",\"messages\":[";
  bool first = true;
  if (!generation.system_prompt.empty()) {
    output << "{\"role\":\"system\",\"content\":"
           << json::Quote(generation.system_prompt) << '}';
    first = false;
  }
  for (const ChatMessage& message : messages) {
    if (message.role != "user" && message.role != "assistant") continue;
    if (!first) output << ',';
    first = false;
    std::string content = message.role == "user" ? UserTextWithDocuments(message)
                                                   : message.content;
    if (message.role == "assistant") {
      while (!content.empty() && (content.back() == ' ' || content.back() == '\n' || content.back() == '\r' || content.back() == '\t')) {
        content.pop_back();
      }
    }
    output << "{\"role\":" << json::Quote(message.role);
    const bool has_media = message.role == "user" &&
        std::any_of(message.attachments.begin(), message.attachments.end(),
                    [](const MediaAttachment& attachment) {
                      return attachment.kind != MediaKind::kDocument;
                    });
    if (!has_media) {
      output << ",\"content\":" << json::Quote(content);
    } else {
      output << ",\"content\":[";
      bool first_part = true;
      if (!content.empty()) {
        output << "{\"type\":\"text\",\"text\":" << json::Quote(content) << '}';
        first_part = false;
      }
      for (const MediaAttachment& attachment : message.attachments) {
        if (attachment.kind == MediaKind::kDocument) continue;
        if (!first_part) output << ',';
        first_part = false;
        const std::string encoded = EncodeBase64(attachment.bytes);
        if (attachment.kind == MediaKind::kImage) {
          output << "{\"type\":\"image_url\",\"image_url\":{\"url\":"
                 << json::Quote("data:" + attachment.mime_type + ";base64," + encoded)
                 << "}}";
        } else {
          output << "{\"type\":\"input_audio\",\"input_audio\":{\"format\":"
                 << json::Quote(attachment.format) << ",\"data\":"
                 << json::Quote(encoded) << "}}";
        }
      }
      output << ']';
    }
    output << '}';
  }
  output << "]}";
  return output.str();
}

namespace {

const json::Value* Member(const json::Value* value, std::string_view key) {
  return value && value->is_object() ? value->find(key) : nullptr;
}

std::string TextMember(const json::Value* value, std::string_view key) {
  const json::Value* member = Member(value, key);
  return member && member->is_string() ? member->as_string() : std::string{};
}

std::int64_t IntegerMember(const json::Value* value, std::string_view key) {
  const json::Value* member = Member(value, key);
  return member && member->is_integer() ? member->as_integer() : 0;
}

void ParseSseData(std::string_view data, const std::function<void(ChatEvent)>& emit,
                  bool& saw_done) {
  if (data == "[DONE]") {
    saw_done = true;
    return;
  }
  const auto parsed = json::Parse(data, {.max_depth = 32, .max_values = 10000,
                                         .max_string_bytes = 16U * 1024U * 1024U});
  if (!parsed.ok() || !parsed.value().is_object()) return;
  const json::Value& root = parsed.value();
  if (const json::Value* error = Member(&root, "error")) {
    std::string message = TextMember(error, "message");
    emit({ChatEvent::Kind::kError, message.empty() ? "The server returned a streaming error" : message});
    return;
  }
  if (const json::Value* usage = Member(&root, "usage");
      usage && usage->is_object()) {
    ChatEvent event{ChatEvent::Kind::kUsage, {}};
    event.prompt_tokens = IntegerMember(usage, "prompt_tokens");
    event.completion_tokens = IntegerMember(usage, "completion_tokens");
    emit(std::move(event));
  }
  const json::Value* choices = Member(&root, "choices");
  if (!choices || !choices->is_array() || choices->as_array().empty()) return;
  const json::Value* choice = &choices->as_array().front();
  const json::Value* delta = Member(choice, "delta");
  const std::string reasoning = TextMember(delta, "reasoning_content");
  const std::string content = TextMember(delta, "content");
  if (!reasoning.empty()) emit({ChatEvent::Kind::kReasoning, reasoning});
  if (!content.empty()) emit({ChatEvent::Kind::kText, content});
}

}  // namespace

ApiClient::~ApiClient() { Cancel(); }

void ApiClient::StreamChat(const ServerConfig& server,
                           const GenerationConfig& generation,
                           const std::vector<ChatMessage>& messages,
                           const std::string& session_id) {
  if (Busy()) return;
  if (worker_.joinable()) worker_.join();
  cancel_requested_.store(false);
  {
    std::lock_guard lock(mutex_);
    busy_ = true;
    events_.clear();
  }
  worker_ = std::jthread([this, server, generation, messages, session_id] {
    const std::string host = server.host == "0.0.0.0" ? "127.0.0.1" : server.host;
    auto client = std::make_shared<httplib::Client>(host, server.port);
    client->set_connection_timeout(std::chrono::seconds(5));
    client->set_read_timeout(std::chrono::minutes(30));
    client->set_write_timeout(std::chrono::seconds(30));
    client->set_tcp_nodelay(true);
    {
      std::lock_guard lock(mutex_);
      active_client_ = client;
    }

    httplib::Headers headers{{"Accept", "text/event-stream"},
                             {"Authorization", "Bearer gem16"}};
    if (!session_id.empty()) headers.emplace("X-Gem16-Session-Id", session_id);
    std::string buffer;
    std::string response_prefix;
    bool saw_done = false;
    const auto response = client->Post(
        "/v1/chat/completions", headers, BuildChatPayload(server, generation, messages),
        "application/json",
        [this, &buffer, &response_prefix, &saw_done](const char* data, std::size_t size) {
          if (response_prefix.size() < 16U * 1024U) {
            response_prefix.append(data, std::min(size, 16U * 1024U - response_prefix.size()));
          }
          buffer.append(data, size);
          std::size_t boundary = std::string::npos;
          while ((boundary = buffer.find('\n')) != std::string::npos) {
            std::string line = buffer.substr(0, boundary);
            buffer.erase(0, boundary + 1);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (!line.starts_with("data:")) continue;
            std::string_view value(line);
            value.remove_prefix(5);
            while (!value.empty() && value.front() == ' ') value.remove_prefix(1);
            ParseSseData(value, [this](ChatEvent event) { Emit(std::move(event)); }, saw_done);
          }
          return true;
        });

    if (!response && cancel_requested_.load()) {
      Emit({ChatEvent::Kind::kFinished, "cancelled"});
    } else if (!response) {
      Emit({ChatEvent::Kind::kError,
            "Could not reach gem16-server: " + httplib::to_string(response.error())});
    } else if (response->status < 200 || response->status >= 300) {
      std::string detail = response->body.empty() ? response_prefix : response->body;
      const auto parsed = json::Parse(detail);
      if (parsed.ok()) {
        const std::string message = TextMember(Member(&parsed.value(), "error"), "message");
        if (!message.empty()) detail = message;
      }
      Emit({ChatEvent::Kind::kError,
            "Server returned HTTP " + std::to_string(response->status) + ": " + detail.substr(0, 500)});
    } else if (!saw_done) {
      Emit({ChatEvent::Kind::kError, "The server stream ended without a [DONE] marker"});
    } else {
      const std::string returned_session = response->get_header_value("X-Gem16-Session-Id");
      if (!returned_session.empty()) Emit({ChatEvent::Kind::kSession, returned_session});
      Emit({ChatEvent::Kind::kFinished, {}});
    }
    std::lock_guard lock(mutex_);
    active_client_.reset();
    busy_ = false;
  });
}

void ApiClient::Cancel() {
  cancel_requested_.store(true);
  std::shared_ptr<httplib::Client> client;
  {
    std::lock_guard lock(mutex_);
    client = active_client_;
  }
  if (client) client->stop();
  if (worker_.joinable() && worker_.get_id() != std::this_thread::get_id()) worker_.join();
  std::lock_guard lock(mutex_);
  active_client_.reset();
  busy_ = false;
}

bool ApiClient::Busy() const {
  std::lock_guard lock(mutex_);
  return busy_;
}

std::vector<ChatEvent> ApiClient::DrainEvents() {
  std::lock_guard lock(mutex_);
  std::vector<ChatEvent> result{events_.begin(), events_.end()};
  events_.clear();
  return result;
}

void ApiClient::Emit(ChatEvent event) {
  std::lock_guard lock(mutex_);
  events_.push_back(std::move(event));
}

}  // namespace gem16::studio
