#include "api_client.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <optional>
#include <set>
#include <sstream>
#include <unordered_map>

#include "media_loader.h"
#include "util/json.h"

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

std::optional<ServerMetrics> ParseServerMetrics(std::string_view body) {
  std::unordered_map<std::string, double> values;
  while (!body.empty()) {
    const std::size_t newline = body.find('\n');
    std::string_view line = body.substr(0, newline);
    if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
    const std::size_t first = line.find_first_not_of(" \t");
    if (first != std::string_view::npos && line[first] != '#') {
      line.remove_prefix(first);
      const std::size_t separator = line.find_first_of(" \t");
      if (separator != std::string_view::npos && separator > 0) {
        std::string name(line.substr(0, separator));
        const std::size_t value_start = line.find_first_not_of(" \t", separator);
        if (value_start != std::string_view::npos) {
          std::string value_text(line.substr(value_start));
          char* end = nullptr;
          const double value = std::strtod(value_text.c_str(), &end);
          if (end != value_text.c_str() && *end == '\0' && std::isfinite(value))
            values.insert_or_assign(std::move(name), value);
        }
      }
    }
    if (newline == std::string_view::npos) break;
    body.remove_prefix(newline + 1);
  }
  const auto metric = [&values](const char* name) -> std::optional<double> {
    const auto found = values.find(name);
    return found == values.end() ? std::nullopt
                                 : std::optional<double>(found->second);
  };
  const auto input_tokens = metric("gem16_input_tokens_total");
  const auto cache_write_tokens = metric("gem16_cache_write_tokens_total");
  const auto prompt_microseconds = metric("gem16_prompt_microseconds_total");
  const auto decode_microseconds = metric("gem16_decode_microseconds_total");
  const auto decode_measured_tokens =
      metric("gem16_decode_measured_tokens_total");
  if (!input_tokens || !cache_write_tokens || !prompt_microseconds ||
      !decode_microseconds || !decode_measured_tokens)
    return std::nullopt;
  return ServerMetrics{*input_tokens, *cache_write_tokens,
                       *prompt_microseconds, *decode_microseconds,
                       *decode_measured_tokens};
}

std::optional<PerformanceStats> PerformanceDifference(
    const ServerMetrics& before, const ServerMetrics& after) {
  const double prompt_micros =
      after.prompt_microseconds - before.prompt_microseconds;
  const double decode_micros =
      after.decode_microseconds - before.decode_microseconds;
  const double input_tokens = after.input_tokens - before.input_tokens;
  const double cache_write_tokens =
      after.cache_write_tokens - before.cache_write_tokens;
  const double decode_tokens =
      after.decode_measured_tokens - before.decode_measured_tokens;
  if (prompt_micros < 0.0 || decode_micros <= 0.0 || input_tokens < 0.0 ||
      cache_write_tokens < 0.0 || decode_tokens < 0.0)
    return std::nullopt;
  return PerformanceStats{
      .decode_tokens_per_second = decode_tokens * 1'000'000.0 / decode_micros,
      .prefill_tokens_per_second = prompt_micros > 0.0
          ? cache_write_tokens * 1'000'000.0 / prompt_micros
          : 0.0,
      .prefill_milliseconds = prompt_micros / 1'000.0,
      .decode_milliseconds = decode_micros / 1'000.0,
  };
}

std::string BuildChatPayload(const ServerConfig& server,
                             const GenerationConfig& generation,
                             const std::vector<ChatMessage>& messages,
                             const std::string& tools) {
  std::ostringstream output;
  output << "{\"model\":" << json::Quote(server.model_name)
         << ",\"stream\":true,\"stream_options\":{\"include_usage\":true}"
         << ",\"max_completion_tokens\":" << generation.max_output_tokens
         << ",\"reasoning_effort\":" << json::Quote(generation.reasoning_effort);
  if (server.profile == ModelProfile::kGemma4Moe26BTrellis35VisionFp8) {
    output << ",\"vision_soft_token_budget\":"
           << server.vision_soft_token_budget;
  }
  if (!tools.empty()) output << ",\"tools\":" << tools;
  output << ",\"messages\":[";
  bool first = true;
  if (!generation.system_prompt.empty()) {
    output << "{\"role\":\"system\",\"content\":"
           << json::Quote(generation.system_prompt) << '}';
    first = false;
  }
  // A cancelled/incomplete tool exchange must never produce orphan tool
  // results.
  std::vector<bool> excluded(messages.size(), false);
  for (std::size_t begin = 0; begin < messages.size();) {
    std::size_t end = begin + 1;
    while (end < messages.size() && messages[end].role != "user") ++end;
    bool invalid = false;
    std::set<std::string> pending;
    for (std::size_t i = begin; i < end; ++i) {
      const auto& m = messages[i];
      invalid |= m.error || m.streaming;
      for (const auto& c : m.tool_calls)
        if (c.id.empty() || !pending.insert(c.id).second) invalid = true;
      if (m.role == "tool" && pending.erase(m.tool_call_id) != 1)
        invalid = true;
    }
    invalid |= !pending.empty();
    if (invalid)
      std::fill(excluded.begin() + begin, excluded.begin() + end, true);
    begin = end;
  }
  for (std::size_t index = 0; index < messages.size(); ++index) {
    const ChatMessage& message = messages[index];
    // A failed exchange remains visible in Studio, but is not canonical history.
    if (excluded[index] || message.error || message.streaming ||
        (message.role == "user" && index + 1 < messages.size() &&
         messages[index + 1].role == "assistant" && messages[index + 1].error))
      continue;
    if (message.role != "user" && message.role != "assistant" &&
        message.role != "tool")
      continue;
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
    if (!message.tool_calls.empty())
      output << ",\"tool_calls\":" << ToolCallsJson(message.tool_calls);
    if (message.role == "tool")
      output << ",\"tool_call_id\":" << json::Quote(message.tool_call_id);
    const bool has_media = message.role == "user" &&
        std::any_of(message.attachments.begin(), message.attachments.end(),
                    [](const MediaAttachment& attachment) {
                      return attachment.kind != MediaKind::kDocument;
                    });
    if (!has_media) {
      output << ",\"content\":"
             << (message.role == "assistant" && !message.tool_calls.empty() &&
                         content.empty()
                     ? "null"
                     : json::Quote(content));
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

std::optional<ServerMetrics> FetchServerMetrics(httplib::Client& client) {
  client.set_read_timeout(std::chrono::seconds(3));
  const auto response = client.Get("/metrics");
  if (!response || response->status < 200 || response->status >= 300)
    return std::nullopt;
  return ParseServerMetrics(response->body);
}

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
  if (const auto* calls = Member(delta, "tool_calls");
      calls && calls->is_array())
    emit({ChatEvent::Kind::kToolCall, json::Stringify(*calls)});
  const std::string reasoning = TextMember(delta, "reasoning_content");
  const std::string content = TextMember(delta, "content");
  if (!reasoning.empty()) emit({ChatEvent::Kind::kReasoning, reasoning});
  if (!content.empty()) emit({ChatEvent::Kind::kText, content});
}

}  // namespace

ApiClient::~ApiClient() {
  Cancel();
  if (worker_.joinable()) worker_.join();
}

void ApiClient::StreamChat(const ServerConfig& server,
                           const GenerationConfig& generation,
                           const std::vector<ChatMessage>& messages,
                           const std::string& session_id,
                           const std::string& tools) {
  if (Busy()) return;
  if (worker_.joinable()) worker_.join();
  cancel_requested_.store(false);
  {
    std::lock_guard lock(mutex_);
    busy_ = true;
    events_.clear();
  }
  worker_ = std::jthread([this, server, generation, messages, session_id,
                          tools] {
    const std::string host = server.host == "0.0.0.0" ? "127.0.0.1" : server.host;
    auto client = std::make_shared<httplib::Client>(host, server.port);
    client->set_connection_timeout(std::chrono::seconds(5));
    client->set_write_timeout(std::chrono::seconds(30));
    client->set_tcp_nodelay(true);
    {
      std::lock_guard lock(mutex_);
      active_client_ = client;
    }

    const auto finish_cancelled = [this] {
      Emit({ChatEvent::Kind::kFinished, "cancelled"});
      std::lock_guard lock(mutex_);
      active_client_.reset();
      busy_ = false;
    };
    if (cancel_requested_.load()) {
      finish_cancelled();
      return;
    }
    const auto metrics_before = FetchServerMetrics(*client);
    if (cancel_requested_.load()) {
      finish_cancelled();
      return;
    }
    client->set_read_timeout(std::chrono::minutes(30));
    httplib::Headers headers{{"Accept", "text/event-stream"},
                             {"Authorization", "Bearer gem16"}};
    if (!session_id.empty()) headers.emplace("X-Gem16-Session-Id", session_id);
    std::string buffer;
    std::string response_prefix;
    bool saw_done = false;
    bool saw_error = false;
    const auto payload = BuildChatPayload(server, generation, messages, tools);
    if (cancel_requested_.load()) {
      finish_cancelled();
      return;
    }
    const auto response = client->Post(
        "/v1/chat/completions", headers, payload.size(),
        [this, &payload](std::size_t offset, std::size_t length, httplib::DataSink& sink) {
          return !cancel_requested_.load() && sink.write(payload.data() + offset, length);
        },
        "application/json",
        [this, &buffer, &response_prefix, &saw_done, &saw_error](const char* data, std::size_t size) {
          if (cancel_requested_.load()) return false;
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
            ParseSseData(value, [this, &saw_error](ChatEvent event) {
              saw_error |= event.kind == ChatEvent::Kind::kError;
              Emit(std::move(event));
            }, saw_done);
          }
          return true;
        });

    if (cancel_requested_.load()) {
      Emit({ChatEvent::Kind::kFinished, "cancelled"});
    } else if (saw_error) {
      // Preserve the server error already emitted, including any partial answer.
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
      if (!returned_session.empty())
        Emit({ChatEvent::Kind::kSession, returned_session});
      if (metrics_before && !cancel_requested_.load()) {
        if (const auto metrics_after = FetchServerMetrics(*client)) {
          if (const auto performance =
                  PerformanceDifference(*metrics_before, *metrics_after)) {
            ChatEvent event{ChatEvent::Kind::kPerformance, {}};
            event.performance = *performance;
            Emit(std::move(event));
          }
        }
      }
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
  // The worker owns completion; the UI never waits for network I/O here.
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
