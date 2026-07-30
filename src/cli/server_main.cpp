#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#include "windows_utf8.h"
#endif

#include "httplib.h"

#include "gem16/chat.h"
#include "gem16/tokenizer.h"
#include "server/openai_chat.h"
#include "server/sse_chunk.h"
#include "util/json.h"

namespace {

struct Options {
  std::filesystem::path model_directory;
  std::filesystem::path assistant_model_directory;
  std::string model_name = "gem16";
  std::string host = "127.0.0.1";
  int port = 8080;
  std::uint64_t max_context = 8192U;
  gem16::KvCacheMode kv_cache_mode = gem16::KvCacheMode::kCheckpointFp8;
  std::uint32_t mtp_draft_tokens = 0U;
  std::uint32_t max_sessions = 2U;
  bool mtp_adaptive = false;
  bool greedy = false;
};

void PrintUsage() {
  std::cout
      << "Usage: gem16-server --model <checkpoint> [options]\n"
      << "  --model-name <id>       Served OpenAI model id (default: gem16)\n"
      << "  --host <address>        Listen address (default: 127.0.0.1)\n"
      << "  --port <port>           Listen port (default: 8080)\n"
      << "  --max-context <tokens>  Session context capacity (default: 8192)\n"
      << "  --max-sessions <count>   Resident execution slots (default: 2)\n"
      << "  --kv-cache fp8|bf16\n"
      << "  --greedy                Disable checkpoint-recommended sampling\n"
      << "  --assistant-model <checkpoint> --mtp-draft-tokens 1|2|4 [--mtp-adaptive]\n";
}

bool ParseUnsigned(std::string_view text, std::uint64_t& value) {
  const auto result =
      std::from_chars(text.data(), text.data() + text.size(), value);
  return result.ec == std::errc{} && result.ptr == text.data() + text.size();
}

gem16::Result<Options> ParseOptions(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--model" && index + 1 < argc) {
      options.model_directory = argv[++index];
    } else if (argument == "--assistant-model" && index + 1 < argc) {
      options.assistant_model_directory = argv[++index];
    } else if (argument == "--model-name" && index + 1 < argc) {
      options.model_name = argv[++index];
    } else if (argument == "--host" && index + 1 < argc) {
      options.host = argv[++index];
    } else if (argument == "--port" && index + 1 < argc) {
      std::uint64_t value = 0U;
      if (!ParseUnsigned(argv[++index], value) || value == 0U || value > 65535U) {
        return gem16::Status(gem16::StatusCode::kInvalidArgument,
                             "--port must be in [1, 65535]");
      }
      options.port = static_cast<int>(value);
    } else if (argument == "--max-context" && index + 1 < argc) {
      if (!ParseUnsigned(argv[++index], options.max_context) ||
          options.max_context == 0U) {
        return gem16::Status(gem16::StatusCode::kInvalidArgument,
                             "--max-context must be positive");
      }
    } else if (argument == "--max-sessions" && index + 1 < argc) {
      std::uint64_t value = 0U;
      if (!ParseUnsigned(argv[++index], value) || value == 0U ||
          value > 64U) {
        return gem16::Status(gem16::StatusCode::kInvalidArgument,
                             "--max-sessions must be in [1, 64]");
      }
      options.max_sessions = static_cast<std::uint32_t>(value);
    } else if (argument == "--kv-cache" && index + 1 < argc) {
      const std::string_view mode(argv[++index]);
      if (mode == "fp8") {
        options.kv_cache_mode = gem16::KvCacheMode::kCheckpointFp8;
      } else if (mode == "bf16") {
        options.kv_cache_mode = gem16::KvCacheMode::kBf16Correctness;
      } else {
        return gem16::Status(gem16::StatusCode::kInvalidArgument,
                             "--kv-cache must be fp8 or bf16");
      }
    } else if (argument == "--mtp-draft-tokens" && index + 1 < argc) {
      std::uint64_t value = 0U;
      if (!ParseUnsigned(argv[++index], value) ||
          (value != 1U && value != 2U && value != 4U)) {
        return gem16::Status(gem16::StatusCode::kInvalidArgument,
                             "--mtp-draft-tokens must be 1, 2, or 4");
      }
      options.mtp_draft_tokens = static_cast<std::uint32_t>(value);
    } else if (argument == "--mtp-adaptive") {
      options.mtp_adaptive = true;
    } else if (argument == "--greedy") {
      options.greedy = true;
    } else {
      return gem16::Status(gem16::StatusCode::kInvalidArgument,
                           "unknown or incomplete option: " +
                               std::string(argument));
    }
  }
  if (options.model_directory.empty()) {
    return gem16::Status(gem16::StatusCode::kInvalidArgument,
                         "--model is required");
  }
  if (options.model_name.empty() || options.host.empty()) {
    return gem16::Status(gem16::StatusCode::kInvalidArgument,
                         "--model-name and --host must not be empty");
  }
  if (options.mtp_draft_tokens != 0U &&
      options.assistant_model_directory.empty()) {
    return gem16::Status(gem16::StatusCode::kInvalidArgument,
                         "active MTP requires --assistant-model");
  }
  if (options.mtp_adaptive && options.mtp_draft_tokens == 0U) {
    return gem16::Status(gem16::StatusCode::kInvalidArgument,
                         "--mtp-adaptive requires active MTP");
  }
  return options;
}

int HttpStatus(const gem16::Status& status) {
  switch (status.code()) {
    case gem16::StatusCode::kInvalidArgument: return 400;
    case gem16::StatusCode::kNotFound: return 404;
    case gem16::StatusCode::kDataLoss: return 400;
    case gem16::StatusCode::kUnsupported: return 400;
    case gem16::StatusCode::kResourceExhausted: return 503;
    case gem16::StatusCode::kCancelled: return 409;
    case gem16::StatusCode::kOk: return 200;
    default: return 500;
  }
}

void SetError(const gem16::Status& status, httplib::Response& response) {
  response.status = HttpStatus(status);
  response.set_content(
      gem16::server::OpenAiErrorJson(
          status.message(), response.status >= 500 ? "server_error"
                                                   : "invalid_request_error"),
      "application/json; charset=utf-8");
}

std::int64_t UnixSeconds() {
  return std::chrono::duration_cast<std::chrono::seconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

bool WriteSse(httplib::DataSink& sink, std::string_view payload) {
  const std::size_t record_size = 6U + payload.size() + 2U;
  char header[2U * sizeof(std::size_t) + 2U]{};
  const auto converted = std::to_chars(
      header, header + sizeof(header) - 2U, record_size, 16);
  if (converted.ec != std::errc{}) return false;
  *converted.ptr = '\r';
  *(converted.ptr + 1U) = '\n';
  return sink.write(
             header, static_cast<std::size_t>(converted.ptr - header) + 2U) &&
         sink.write("data: ", 6U) && sink.write(payload.data(), payload.size()) &&
         sink.write("\n\n\r\n", 4U);
}

bool FinishSse(httplib::DataSink& sink) {
  const bool written = sink.write(gem16::server::kFinalHttpChunk.data(),
                                   gem16::server::kFinalHttpChunk.size());
  sink.done();
  return written;
}

struct Utf8Prefix {
  std::size_t complete = 0U;
  bool invalid = false;
};

Utf8Prefix CompleteUtf8Prefix(std::string_view text) {
  std::size_t position = 0U;
  while (position < text.size()) {
    const unsigned char first = static_cast<unsigned char>(text[position]);
    std::size_t width = 1U;
    if (first <= 0x7FU) {
      width = 1U;
    } else if (first >= 0xC2U && first <= 0xDFU) {
      width = 2U;
    } else if (first >= 0xE0U && first <= 0xEFU) {
      width = 3U;
    } else if (first >= 0xF0U && first <= 0xF4U) {
      width = 4U;
    } else {
      return {position, true};
    }
    if (position + width > text.size()) return {position, false};
    for (std::size_t offset = 1U; offset < width; ++offset) {
      const unsigned char continuation =
          static_cast<unsigned char>(text[position + offset]);
      if (continuation < 0x80U || continuation > 0xBFU) {
        return {position, true};
      }
    }
    const unsigned char second =
        width == 1U ? 0U : static_cast<unsigned char>(text[position + 1U]);
    if ((first == 0xE0U && second < 0xA0U) ||
        (first == 0xEDU && second >= 0xA0U) ||
        (first == 0xF0U && second < 0x90U) ||
        (first == 0xF4U && second >= 0x90U)) {
      return {position, true};
    }
    position += width;
  }
  return {position, false};
}

struct Utf8Pending {
  std::array<char, 3U> bytes{};
  std::size_t size = 0U;
};

struct StreamingContext {
  StreamingContext(const gem16::GemmaChatProcessor& chat_processor,
                   gem16::server::OpenAiResponseIdentity response_identity,
                   httplib::DataSink& data_sink)
      : processor(&chat_processor),
        identity(std::move(response_identity)),
        sink(&data_sink),
        channels(chat_processor.generation_controls()),
        decoded_token(
            std::max<std::size_t>(1U,
                                  chat_processor.maximum_decoded_token_bytes())),
        combined_token(decoded_token.size() + 3U),
        event_chunk(2048U +
                    6U * (decoded_token.size() + identity.id.size() +
                          identity.model.size())) {}

  const gem16::GemmaChatProcessor* processor = nullptr;
  gem16::server::OpenAiResponseIdentity identity;
  httplib::DataSink* sink = nullptr;
  gem16::ResponseChannelTracker channels;
  std::map<std::string, std::size_t, std::less<>> tool_indices;
  std::vector<char> decoded_token;
  std::vector<char> combined_token;
  gem16::server::SseChunkBuilder event_chunk;
  Utf8Pending utf8_pending;
  Utf8Pending reasoning_utf8_pending;
  bool inside_tool_call = false;
  std::atomic<bool>* cancel_requested = nullptr;
  std::atomic<std::uint64_t>* cancellations_observed = nullptr;
  std::atomic<std::uint64_t>* client_disconnects = nullptr;
};

gem16::Status WriteChatTokenDelta(StreamingContext& context,
                                  std::string_view field,
                                  std::string_view bytes) {
  gem16::server::SseChunkBuilder& chunk = context.event_chunk;
  chunk.Reset();
  const bool built =
      chunk.Append("{\"id\":") && chunk.AppendJsonString(context.identity.id) &&
      chunk.Append(",\"object\":\"chat.completion.chunk\",\"created\":") &&
      chunk.AppendSigned(context.identity.created) &&
      chunk.Append(",\"model\":") &&
      chunk.AppendJsonString(context.identity.model) &&
      chunk.Append(",\"choices\":[{\"index\":0,\"delta\":{\"") &&
      chunk.Append(field) && chunk.Append("\":") &&
      chunk.AppendJsonString(bytes) &&
      chunk.Append("},\"finish_reason\":null}],\"usage\":null}");
  const std::span<const char> record = built ? chunk.Finish()
                                              : std::span<const char>{};
  if (record.empty()) {
    return gem16::Status(gem16::StatusCode::kInternal,
                         "preallocated SSE token buffer is too small");
  }
  if (!context.sink->write(record.data(), record.size())) {
    return gem16::Status(gem16::StatusCode::kIoError,
                         "client disconnected during SSE generation");
  }
  return gem16::Status::Ok();
}

gem16::Status StreamDelta(StreamingContext& context,
                          const gem16::GenerationEvent& event) {
  std::string delta;
  if (event.kind == gem16::GenerationEventKind::kTextDelta) {
    delta = "{\"content\":" + gem16::json::Quote(event.text_delta) + "}";
  } else if (event.kind == gem16::GenerationEventKind::kToolCallStart) {
    const std::size_t index = context.tool_indices.size();
    context.tool_indices.emplace(event.tool_call_id, index);
    delta = "{\"tool_calls\":[{\"index\":" + std::to_string(index) +
            ",\"id\":" + gem16::json::Quote(event.tool_call_id) +
            ",\"type\":\"function\",\"function\":{\"name\":" +
            gem16::json::Quote(event.tool_name) +
            ",\"arguments\":\"\"}}]}";
  } else if (event.kind ==
             gem16::GenerationEventKind::kToolCallArgumentsDelta) {
    const auto index = context.tool_indices.find(event.tool_call_id);
    if (index == context.tool_indices.end()) {
      return gem16::Status(gem16::StatusCode::kInternal,
                           "tool arguments arrived before tool start");
    }
    delta = "{\"tool_calls\":[{\"index\":" +
            std::to_string(index->second) +
            ",\"function\":{\"arguments\":" +
            gem16::json::Quote(event.text_delta) + "}}]}";
  } else {
    return gem16::Status::Ok();
  }
  if (!WriteSse(*context.sink, gem16::server::ChatCompletionChunkJson(
                                   context.identity, delta))) {
    return gem16::Status(gem16::StatusCode::kIoError,
                         "client disconnected during SSE generation");
  }
  return gem16::Status::Ok();
}

gem16::Status FeedChatUtf8(StreamingContext& context, Utf8Pending& pending,
                           std::string_view bytes, bool reasoning,
                           bool final) {
  if (bytes.size() > context.combined_token.size() - pending.size) {
    return gem16::Status(gem16::StatusCode::kInternal,
                         "decoded token exceeds the preallocated UTF-8 buffer");
  }
  std::size_t total = 0U;
  for (std::size_t index = 0U; index < pending.size; ++index) {
    context.combined_token[total++] = pending.bytes[index];
  }
  for (const char byte : bytes) context.combined_token[total++] = byte;
  const std::string_view combined(context.combined_token.data(), total);
  const Utf8Prefix prefix = CompleteUtf8Prefix(combined);
  if (prefix.invalid) {
    return gem16::Status(gem16::StatusCode::kDataLoss,
                         "model response contains invalid UTF-8");
  }
  const std::size_t retained = total - prefix.complete;
  if (retained > pending.bytes.size()) {
    return gem16::Status(gem16::StatusCode::kInternal,
                         "UTF-8 carry exceeds its fixed buffer");
  }
  for (std::size_t index = 0U; index < retained; ++index) {
    pending.bytes[index] = context.combined_token[prefix.complete + index];
  }
  pending.size = retained;
  if (prefix.complete != 0U) {
    const gem16::Status status = WriteChatTokenDelta(
        context, reasoning ? "reasoning_content" : "content",
        combined.substr(0U, prefix.complete));
    if (!status.ok()) return status;
  }
  if (final && pending.size != 0U) {
    return gem16::Status(gem16::StatusCode::kDataLoss,
                         "model response ends with incomplete UTF-8");
  }
  return gem16::Status::Ok();
}

gem16::Status FeedVisibleText(StreamingContext& context,
                              std::string_view bytes, bool final) {
  return FeedChatUtf8(context, context.utf8_pending, bytes, false, final);
}

gem16::Status FeedReasoningText(StreamingContext& context,
                                std::string_view bytes, bool final) {
  return FeedChatUtf8(context, context.reasoning_utf8_pending, bytes, true,
                      final);
}

gem16::Status StreamToken(void* opaque_context,
                          const gem16::GenerationEvent& event) {
  auto* context = static_cast<StreamingContext*>(opaque_context);
  if (context == nullptr || context->processor == nullptr ||
      context->sink == nullptr ||
      event.kind != gem16::GenerationEventKind::kToken) {
    return gem16::Status(gem16::StatusCode::kInternal,
                         "invalid OpenAI stream callback");
  }
  if (context->cancel_requested != nullptr &&
      context->cancel_requested->load()) {
    if (context->cancellations_observed != nullptr) {
      context->cancellations_observed->fetch_add(1U);
    }
    return gem16::Status(gem16::StatusCode::kCancelled,
                         "generation was cancelled");
  }
  if (context->sink->is_writable && !context->sink->is_writable()) {
    if (context->client_disconnects != nullptr) {
      context->client_disconnects->fetch_add(1U);
    }
    return gem16::Status(gem16::StatusCode::kCancelled,
                         "client disconnected during generation");
  }
  const gem16::ResponseTokenChannel channel = context->channels.Observe(event.token_id);
  if (channel == gem16::ResponseTokenChannel::kControl) {
    return gem16::Status::Ok();
  }
  std::size_t decoded_size = 0U;
  if (channel == gem16::ResponseTokenChannel::kReasoning) {
    const gem16::Status status = context->processor->DecodeTokenInto(
        event.token_id, true, context->decoded_token, decoded_size);
    if (!status.ok()) return status;
    return FeedReasoningText(
        *context,
        std::string_view(context->decoded_token.data(), decoded_size), false);
  }
  gem16::Status status = context->processor->DecodeTokenInto(
      event.token_id, false, context->decoded_token, decoded_size);
  if (!status.ok()) return status;
  const std::string_view raw(context->decoded_token.data(), decoded_size);
  if (raw == "<|tool_call>") {
    context->inside_tool_call = true;
    return gem16::Status::Ok();
  }
  if (raw == "<tool_call|>") {
    context->inside_tool_call = false;
    return gem16::Status::Ok();
  }
  if (context->inside_tool_call) return gem16::Status::Ok();
  status = context->processor->DecodeTokenInto(
      event.token_id, true, context->decoded_token, decoded_size);
  if (!status.ok()) return status;
  return FeedVisibleText(
      *context, std::string_view(context->decoded_token.data(), decoded_size),
      false);
}

struct ResponsesStreamingContext {
  ResponsesStreamingContext(
      const gem16::GemmaChatProcessor& chat_processor,
      gem16::server::OpenAiResponseIdentity response_identity,
      httplib::DataSink& data_sink, std::uint64_t reasoning_token_capacity)
      : processor(&chat_processor),
        identity(std::move(response_identity)),
        sink(&data_sink),
        channels(chat_processor.generation_controls()),
        decoded_token(
            std::max<std::size_t>(1U,
                                  chat_processor.maximum_decoded_token_bytes())),
        combined_token(decoded_token.size() + 3U),
         event_chunk(3072U +
                     6U * (decoded_token.size() + identity.id.size())),
         escape_scratch(4096U) {
    const std::uint64_t bounded_tokens = std::min<std::uint64_t>(
        reasoning_token_capacity,
        std::numeric_limits<std::size_t>::max() / decoded_token.size());
    reasoning_text.resize(static_cast<std::size_t>(bounded_tokens) *
                          decoded_token.size());
  }

  const gem16::GemmaChatProcessor* processor = nullptr;
  gem16::server::OpenAiResponseIdentity identity;
  httplib::DataSink* sink = nullptr;
  gem16::ResponseChannelTracker channels;
  std::vector<char> decoded_token;
  std::vector<char> combined_token;
  gem16::server::SseChunkBuilder event_chunk;
  std::vector<char> escape_scratch;
  std::vector<char> reasoning_text;
  std::size_t reasoning_text_size = 0U;
  Utf8Pending text_pending;
  Utf8Pending reasoning_pending;
  std::uint64_t sequence = 1U;
  std::size_t output_index = 0U;
  std::size_t reasoning_index = 0U;
  std::size_t message_index = 0U;
  bool reasoning_started = false;
  bool reasoning_done = false;
  bool message_started = false;
  bool inside_tool_call = false;
  std::atomic<bool>* cancel_requested = nullptr;
  std::atomic<std::uint64_t>* cancellations_observed = nullptr;
  std::atomic<std::uint64_t>* client_disconnects = nullptr;
};

gem16::Status SendResponseChunk(ResponsesStreamingContext& context,
                                bool built) {
  const std::span<const char> record =
      built ? context.event_chunk.Finish() : std::span<const char>{};
  if (record.empty()) {
    return gem16::Status(gem16::StatusCode::kInternal,
                         "preallocated Responses SSE buffer is too small");
  }
  if (!context.sink->write(record.data(), record.size())) {
    return gem16::Status(gem16::StatusCode::kIoError,
                         "client disconnected during Responses streaming");
  }
  return gem16::Status::Ok();
}

std::optional<std::size_t> JsonEscapedSize(std::string_view value) {
  std::size_t size = 0U;
  for (const unsigned char byte : value) {
    const bool short_escape = byte == '"' || byte == '\\' || byte == '\b' ||
                              byte == '\f' || byte == '\n' || byte == '\r' ||
                              byte == '\t';
    const std::size_t increment = short_escape ? 2U : byte < 0x20U ? 6U : 1U;
    if (increment > std::numeric_limits<std::size_t>::max() - size) {
      return std::nullopt;
    }
    size += increment;
  }
  return size;
}

bool WriteJsonEscapedRaw(httplib::DataSink& sink, std::string_view value,
                         std::span<char> scratch) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::size_t size = 0U;
  const auto flush = [&]() {
    if (size == 0U) return true;
    const bool written = sink.write(scratch.data(), size);
    size = 0U;
    return written;
  };
  const auto append = [&](std::string_view bytes) {
    if (bytes.size() > scratch.size() - size && !flush()) return false;
    if (bytes.size() > scratch.size()) return false;
    for (const char byte : bytes) scratch[size++] = byte;
    return true;
  };
  for (const unsigned char byte : value) {
    char escaped[6U]{};
    std::string_view bytes;
    switch (byte) {
      case '"': bytes = "\\\""; break;
      case '\\': bytes = "\\\\"; break;
      case '\b': bytes = "\\b"; break;
      case '\f': bytes = "\\f"; break;
      case '\n': bytes = "\\n"; break;
      case '\r': bytes = "\\r"; break;
      case '\t': bytes = "\\t"; break;
      default:
        if (byte < 0x20U) {
          escaped[0] = '\\';
          escaped[1] = 'u';
          escaped[2] = '0';
          escaped[3] = '0';
          escaped[4] = kHex[byte >> 4U];
          escaped[5] = kHex[byte & 0x0FU];
          bytes = std::string_view(escaped, sizeof(escaped));
        } else {
          escaped[0] = static_cast<char>(byte);
          bytes = std::string_view(escaped, 1U);
        }
    }
    if (!append(bytes)) return false;
  }
  return flush();
}

gem16::Status SendResponseCompositeChunk(ResponsesStreamingContext& context,
                                         std::string_view value,
                                         std::string_view suffix) {
  const std::span<const char> prefix = context.event_chunk.Payload();
  const std::optional<std::size_t> escaped = JsonEscapedSize(value);
  if (prefix.empty() || !escaped.has_value() ||
      prefix.size() > std::numeric_limits<std::size_t>::max() - *escaped -
                          suffix.size() - 4U) {
    return gem16::Status(gem16::StatusCode::kInternal,
                         "Responses SSE composite size overflows");
  }
  const std::size_t payload_size =
      prefix.size() + *escaped + suffix.size() + 4U;
  char header[2U * sizeof(std::size_t) + 2U]{};
  const auto converted = std::to_chars(
      header, header + sizeof(header) - 2U, payload_size, 16);
  if (converted.ec != std::errc{}) {
    return gem16::Status(gem16::StatusCode::kInternal,
                         "Responses SSE chunk size cannot be formatted");
  }
  *converted.ptr = '\r';
  *(converted.ptr + 1U) = '\n';
  const bool written =
      context.sink->write(
          header, static_cast<std::size_t>(converted.ptr - header) + 2U) &&
      context.sink->write(prefix.data(), prefix.size()) &&
      context.sink->write("\"", 1U) &&
      WriteJsonEscapedRaw(*context.sink, value, context.escape_scratch) &&
      context.sink->write("\"", 1U) &&
      context.sink->write(suffix.data(), suffix.size()) &&
      context.sink->write("\n\n\r\n", 4U);
  if (!written) {
    return gem16::Status(gem16::StatusCode::kIoError,
                         "client disconnected during Responses streaming");
  }
  return gem16::Status::Ok();
}

gem16::Status StartResponseReasoning(ResponsesStreamingContext& context) {
  if (context.reasoning_started) return gem16::Status::Ok();
  context.reasoning_started = true;
  context.reasoning_index = context.output_index;
  auto& chunk = context.event_chunk;
  chunk.Reset();
  const bool built =
      chunk.Append("{\"type\":\"response.output_item.added\",\"output_index\":") &&
      chunk.AppendUnsigned(context.reasoning_index) &&
      chunk.Append(",\"item\":{\"id\":") &&
      chunk.AppendJsonString("rs_", context.identity.id) &&
      chunk.Append(",\"type\":\"reasoning\",\"summary\":[],\"content\":[],\"status\":\"in_progress\"},\"sequence_number\":") &&
      chunk.AppendUnsigned(context.sequence++) && chunk.Append("}");
  return SendResponseChunk(context, built);
}

gem16::Status WriteResponseReasoningDelta(ResponsesStreamingContext& context,
                                          std::string_view bytes) {
  auto& chunk = context.event_chunk;
  chunk.Reset();
  const bool built =
      chunk.Append("{\"type\":\"response.reasoning_text.delta\",\"item_id\":") &&
      chunk.AppendJsonString("rs_", context.identity.id) &&
      chunk.Append(",\"output_index\":") &&
      chunk.AppendUnsigned(context.reasoning_index) &&
      chunk.Append(",\"content_index\":0,\"delta\":") &&
      chunk.AppendJsonString(bytes) &&
      chunk.Append(",\"sequence_number\":") &&
      chunk.AppendUnsigned(context.sequence++) && chunk.Append("}");
  return SendResponseChunk(context, built);
}

gem16::Status FinalizeResponseReasoning(ResponsesStreamingContext& context) {
  if (!context.reasoning_started || context.reasoning_done) {
    return gem16::Status::Ok();
  }
  if (context.reasoning_pending.size != 0U) {
    return gem16::Status(gem16::StatusCode::kDataLoss,
                         "reasoning response ends with incomplete UTF-8");
  }
  const std::string_view text(context.reasoning_text.data(),
                              context.reasoning_text_size);
  auto& chunk = context.event_chunk;
  chunk.Reset();
  bool built =
      chunk.Append("{\"type\":\"response.reasoning_text.done\",\"sequence_number\":") &&
      chunk.AppendUnsigned(context.sequence) &&
      chunk.Append(",\"item_id\":") &&
      chunk.AppendJsonString("rs_", context.identity.id) &&
      chunk.Append(",\"output_index\":") &&
      chunk.AppendUnsigned(context.reasoning_index) &&
      chunk.Append(",\"content_index\":0,\"text\":");
  if (!built) return SendResponseChunk(context, false);
  gem16::Status status = SendResponseCompositeChunk(context, text, "}");
  if (!status.ok()) return status;
  ++context.sequence;

  chunk.Reset();
  built =
      chunk.Append("{\"type\":\"response.output_item.done\",\"sequence_number\":") &&
      chunk.AppendUnsigned(context.sequence) &&
      chunk.Append(",\"output_index\":") &&
      chunk.AppendUnsigned(context.reasoning_index) &&
      chunk.Append(",\"item\":{\"id\":") &&
      chunk.AppendJsonString("rs_", context.identity.id) &&
      chunk.Append(",\"type\":\"reasoning\",\"summary\":[],\"content\":[{\"type\":\"reasoning_text\",\"text\":");
  if (!built) return SendResponseChunk(context, false);
  status = SendResponseCompositeChunk(
      context, text, "}],\"status\":\"completed\"}}");
  if (!status.ok()) return status;
  ++context.sequence;
  context.reasoning_done = true;
  ++context.output_index;
  return gem16::Status::Ok();
}

gem16::Status StartResponseMessage(ResponsesStreamingContext& context) {
  if (context.message_started) return gem16::Status::Ok();
  gem16::Status status = FinalizeResponseReasoning(context);
  if (!status.ok()) return status;
  context.message_started = true;
  context.message_index = context.output_index;
  auto& chunk = context.event_chunk;
  chunk.Reset();
  bool built =
      chunk.Append("{\"type\":\"response.output_item.added\",\"output_index\":") &&
      chunk.AppendUnsigned(context.message_index) &&
      chunk.Append(",\"item\":{\"id\":") &&
      chunk.AppendJsonString("msg_", context.identity.id) &&
      chunk.Append(",\"type\":\"message\",\"status\":\"in_progress\",\"role\":\"assistant\",\"content\":[]},\"sequence_number\":") &&
      chunk.AppendUnsigned(context.sequence++) && chunk.Append("}");
  status = SendResponseChunk(context, built);
  if (!status.ok()) return status;

  chunk.Reset();
  built =
      chunk.Append("{\"type\":\"response.content_part.added\",\"item_id\":") &&
      chunk.AppendJsonString("msg_", context.identity.id) &&
      chunk.Append(",\"output_index\":") &&
      chunk.AppendUnsigned(context.message_index) &&
      chunk.Append(",\"content_index\":0,\"part\":{\"type\":\"output_text\",\"text\":\"\",\"annotations\":[]},\"sequence_number\":") &&
      chunk.AppendUnsigned(context.sequence++) && chunk.Append("}");
  return SendResponseChunk(context, built);
}

gem16::Status WriteResponseTextDelta(ResponsesStreamingContext& context,
                                     std::string_view bytes) {
  gem16::Status status = StartResponseMessage(context);
  if (!status.ok()) return status;
  auto& chunk = context.event_chunk;
  chunk.Reset();
  const bool built =
      chunk.Append("{\"type\":\"response.output_text.delta\",\"item_id\":") &&
      chunk.AppendJsonString("msg_", context.identity.id) &&
      chunk.Append(",\"output_index\":") &&
      chunk.AppendUnsigned(context.message_index) &&
      chunk.Append(",\"content_index\":0,\"delta\":") &&
      chunk.AppendJsonString(bytes) &&
      chunk.Append(",\"logprobs\":[],\"sequence_number\":") &&
      chunk.AppendUnsigned(context.sequence++) && chunk.Append("}");
  return SendResponseChunk(context, built);
}

gem16::Status FeedResponseUtf8(ResponsesStreamingContext& context,
                               Utf8Pending& pending, std::string_view bytes,
                               bool reasoning) {
  if (bytes.size() > context.combined_token.size() - pending.size) {
    return gem16::Status(gem16::StatusCode::kInternal,
                         "decoded token exceeds the Responses UTF-8 buffer");
  }
  std::size_t total = 0U;
  for (std::size_t index = 0U; index < pending.size; ++index) {
    context.combined_token[total++] = pending.bytes[index];
  }
  for (const char byte : bytes) context.combined_token[total++] = byte;
  const std::string_view combined(context.combined_token.data(), total);
  const Utf8Prefix prefix = CompleteUtf8Prefix(combined);
  if (prefix.invalid) {
    return gem16::Status(gem16::StatusCode::kDataLoss,
                         "model response contains invalid UTF-8");
  }
  const std::size_t retained = total - prefix.complete;
  if (retained > pending.bytes.size()) {
    return gem16::Status(gem16::StatusCode::kInternal,
                         "Responses UTF-8 carry exceeds its fixed buffer");
  }
  for (std::size_t index = 0U; index < retained; ++index) {
    pending.bytes[index] = context.combined_token[prefix.complete + index];
  }
  pending.size = retained;
  if (prefix.complete == 0U) return gem16::Status::Ok();
  const std::string_view complete = combined.substr(0U, prefix.complete);
  if (reasoning) {
    if (complete.size() >
        context.reasoning_text.size() - context.reasoning_text_size) {
      return gem16::Status(gem16::StatusCode::kInternal,
                           "reasoning output exceeds its preallocated buffer");
    }
    for (const char byte : complete) {
      context.reasoning_text[context.reasoning_text_size++] = byte;
    }
    gem16::Status status = StartResponseReasoning(context);
    if (!status.ok()) return status;
    return WriteResponseReasoningDelta(context, complete);
  }
  return WriteResponseTextDelta(context, complete);
}

gem16::Status StreamResponseToken(void* opaque_context,
                                  const gem16::GenerationEvent& event) {
  auto* context = static_cast<ResponsesStreamingContext*>(opaque_context);
  if (context == nullptr || context->processor == nullptr ||
      context->sink == nullptr ||
      event.kind != gem16::GenerationEventKind::kToken) {
    return gem16::Status(gem16::StatusCode::kInternal,
                         "invalid Responses stream callback");
  }
  if (context->cancel_requested != nullptr &&
      context->cancel_requested->load()) {
    if (context->cancellations_observed != nullptr) {
      context->cancellations_observed->fetch_add(1U);
    }
    return gem16::Status(gem16::StatusCode::kCancelled,
                         "generation was cancelled");
  }
  if (context->sink->is_writable && !context->sink->is_writable()) {
    if (context->client_disconnects != nullptr) {
      context->client_disconnects->fetch_add(1U);
    }
    return gem16::Status(gem16::StatusCode::kCancelled,
                         "client disconnected during generation");
  }
  const bool was_reasoning = context->channels.in_reasoning();
  const gem16::ResponseTokenChannel channel =
      context->channels.Observe(event.token_id);
  if (channel == gem16::ResponseTokenChannel::kControl) {
    if (was_reasoning && !context->channels.in_reasoning()) {
      return FinalizeResponseReasoning(*context);
    }
    return gem16::Status::Ok();
  }
  std::size_t decoded_size = 0U;
  if (channel == gem16::ResponseTokenChannel::kReasoning) {
    const gem16::Status status = context->processor->DecodeTokenInto(
        event.token_id, true, context->decoded_token, decoded_size);
    if (!status.ok()) return status;
    return FeedResponseUtf8(
        *context, context->reasoning_pending,
        std::string_view(context->decoded_token.data(), decoded_size), true);
  }
  gem16::Status status = context->processor->DecodeTokenInto(
      event.token_id, false, context->decoded_token, decoded_size);
  if (!status.ok()) return status;
  const std::string_view raw(context->decoded_token.data(), decoded_size);
  if (raw == "<|tool_call>") {
    context->inside_tool_call = true;
    return FinalizeResponseReasoning(*context);
  }
  if (raw == "<tool_call|>") {
    context->inside_tool_call = false;
    return gem16::Status::Ok();
  }
  if (context->inside_tool_call) return gem16::Status::Ok();
  status = context->processor->DecodeTokenInto(
      event.token_id, true, context->decoded_token, decoded_size);
  if (!status.ok()) return status;
  return FeedResponseUtf8(
      *context, context->text_pending,
      std::string_view(context->decoded_token.data(), decoded_size), false);
}

struct ResponsesChain {
    std::string latest_response_id;
    std::vector<gem16::GenerationMessage> messages;
    std::vector<gem16::GenerationToolDefinition> tools;
    gem16::GenerationToolChoice tool_choice;
    bool initialized = false;
};

struct SessionEntry {
  SessionEntry(std::string session_id, gem16::ChatSession chat_session)
      : id(std::move(session_id)), session(std::move(chat_session)) {}

  std::string id;
  gem16::ChatSession session;
  std::mutex inference_mutex;
  ResponsesChain responses_chain;
  std::atomic<std::uint32_t> active_requests{0U};
  std::atomic<bool> cancel_requested{false};
  std::atomic<std::uint64_t> last_used{0U};
  std::string active_response_id;
};

struct ServerMetrics {
  std::atomic<std::uint64_t> requests_total{0U};
  std::atomic<std::uint64_t> requests_failed{0U};
  std::atomic<std::uint64_t> active_requests{0U};
  std::atomic<std::uint64_t> sessions_created{0U};
  std::atomic<std::uint64_t> sessions_evicted{0U};
  std::atomic<std::uint64_t> cancellations_requested{0U};
  std::atomic<std::uint64_t> cancellations_observed{0U};
  std::atomic<std::uint64_t> client_disconnects{0U};
  std::atomic<std::uint64_t> input_tokens{0U};
  std::atomic<std::uint64_t> cached_input_tokens{0U};
  std::atomic<std::uint64_t> cache_write_tokens{0U};
  std::atomic<std::uint64_t> output_tokens{0U};
  std::atomic<std::uint64_t> generation_microseconds{0U};
  std::atomic<std::uint64_t> last_slot_bytes{0U};
};

struct ServerState {
  ServerState(std::string served_model_name, std::uint64_t context_limit,
              gem16::GemmaChatProcessor chat_processor,
              std::shared_ptr<gem16::ModelRuntime> model_runtime,
              gem16::ChatSessionOptions chat_session_options,
              std::uint32_t session_limit)
      : model_name(std::move(served_model_name)),
        max_context(context_limit),
        processor(std::move(chat_processor)),
        runtime(std::move(model_runtime)),
        session_options(std::move(chat_session_options)),
        max_sessions(session_limit) {}

  std::string model_name;
  std::uint64_t max_context = 0U;
  gem16::GemmaChatProcessor processor;
  std::shared_ptr<gem16::ModelRuntime> runtime;
  gem16::ChatSessionOptions session_options;
  std::uint32_t max_sessions = 2U;
  std::mutex pool_mutex;
  std::unordered_map<std::string, std::shared_ptr<SessionEntry>> sessions;
  std::unordered_map<std::string, std::weak_ptr<SessionEntry>> response_index;
  std::atomic<std::uint64_t> response_counter{1U};
  std::atomic<std::uint64_t> session_counter{1U};
  std::atomic<std::uint64_t> lru_clock{1U};
  ServerMetrics metrics;
};

gem16::server::OpenAiResponseIdentity MakeIdentity(ServerState& state) {
  return {"chatcmpl-gem16-" +
              std::to_string(state.response_counter.fetch_add(1U)),
          state.model_name, UnixSeconds()};
}

gem16::server::OpenAiResponseIdentity MakeResponsesIdentity(
    ServerState& state) {
  return {"resp_gem16_" +
              std::to_string(state.response_counter.fetch_add(1U)),
          state.model_name, UnixSeconds()};
}

void EraseSessionLocked(ServerState& state, const std::string& id) {
  const auto found = state.sessions.find(id);
  if (found == state.sessions.end()) return;
  const std::shared_ptr<SessionEntry> entry = found->second;
  for (auto iterator = state.response_index.begin();
       iterator != state.response_index.end();) {
    const std::shared_ptr<SessionEntry> indexed = iterator->second.lock();
    if (indexed == nullptr || indexed == entry) {
      iterator = state.response_index.erase(iterator);
    } else {
      ++iterator;
    }
  }
  state.sessions.erase(found);
}

gem16::Result<std::shared_ptr<SessionEntry>> CreateSession(
    ServerState& state, std::string id) {
  std::lock_guard pool_lock(state.pool_mutex);
  if (state.sessions.contains(id)) {
    return gem16::Status(gem16::StatusCode::kInvalidArgument,
                         "session ID is already resident");
  }
  if (state.sessions.size() >= state.max_sessions) {
    auto victim = state.sessions.end();
    for (auto iterator = state.sessions.begin();
         iterator != state.sessions.end(); ++iterator) {
      if (iterator->second->active_requests.load() != 0U) continue;
      if (victim == state.sessions.end() ||
          iterator->second->last_used.load() <
              victim->second->last_used.load()) {
        victim = iterator;
      }
    }
    if (victim == state.sessions.end()) {
      return gem16::Status(
          gem16::StatusCode::kResourceExhausted,
          "all resident execution slots are active");
    }
    const std::string victim_id = victim->first;
    EraseSessionLocked(state, victim_id);
    state.metrics.sessions_evicted.fetch_add(1U);
  }
  auto session = gem16::ChatSession::Create(
      state.runtime, state.session_options, state.processor);
  if (!session.ok()) return session.status();
  auto entry = std::make_shared<SessionEntry>(
      std::move(id), std::move(session).value());
  entry->active_requests.store(1U);
  entry->last_used.store(state.lru_clock.fetch_add(1U));
  state.sessions.emplace(entry->id, entry);
  state.metrics.sessions_created.fetch_add(1U);
  state.metrics.active_requests.fetch_add(1U);
  return entry;
}

gem16::Result<std::shared_ptr<SessionEntry>> AcquireNamedSession(
    ServerState& state, const std::string& id) {
  {
    std::lock_guard pool_lock(state.pool_mutex);
    const auto found = state.sessions.find(id);
    if (found != state.sessions.end()) {
      if (found->second->active_requests.load() != 0U) {
        return gem16::Status(
            gem16::StatusCode::kResourceExhausted,
            "session already has an active request");
      }
      found->second->active_requests.fetch_add(1U);
      found->second->last_used.store(state.lru_clock.fetch_add(1U));
      state.metrics.active_requests.fetch_add(1U);
      return found->second;
    }
  }
  return CreateSession(state, id);
}

gem16::Result<std::shared_ptr<SessionEntry>> AcquireResponseSession(
    ServerState& state, const std::string& response_id) {
  std::lock_guard pool_lock(state.pool_mutex);
  const auto found = state.response_index.find(response_id);
  if (found == state.response_index.end()) {
    return gem16::Status(gem16::StatusCode::kNotFound,
                         "previous_response_id is not resident");
  }
  std::shared_ptr<SessionEntry> entry = found->second.lock();
  if (entry == nullptr) {
    state.response_index.erase(found);
    return gem16::Status(gem16::StatusCode::kNotFound,
                         "previous_response_id was evicted");
  }
  if (entry->active_requests.load() != 0U) {
    return gem16::Status(gem16::StatusCode::kResourceExhausted,
                         "response session already has an active request");
  }
  entry->active_requests.fetch_add(1U);
  entry->last_used.store(state.lru_clock.fetch_add(1U));
  state.metrics.active_requests.fetch_add(1U);
  return entry;
}

void ReleaseSession(ServerState& state,
                    const std::shared_ptr<SessionEntry>& entry) {
  entry->last_used.store(state.lru_clock.fetch_add(1U));
  entry->active_requests.fetch_sub(1U);
  state.metrics.active_requests.fetch_sub(1U);
}

void DiscardSession(ServerState& state,
                    const std::shared_ptr<SessionEntry>& entry) {
  std::lock_guard pool_lock(state.pool_mutex);
  EraseSessionLocked(state, entry->id);
}

class SessionLease {
 public:
  SessionLease(ServerState& state, std::shared_ptr<SessionEntry> entry)
      : state_(&state), entry_(std::move(entry)) {}
  SessionLease(const SessionLease&) = delete;
  SessionLease& operator=(const SessionLease&) = delete;
  ~SessionLease() {
    if (state_ == nullptr) return;
    if (discard_) DiscardSession(*state_, entry_);
    ReleaseSession(*state_, entry_);
  }
  void Discard() { discard_ = true; }
  void Keep() { discard_ = false; }

 private:
  ServerState* state_ = nullptr;
  std::shared_ptr<SessionEntry> entry_;
  bool discard_ = false;
};

gem16::Status PrepareResponsesRequest(
    SessionEntry& entry, gem16::server::OpenAiResponsesRequest& request) {
  ResponsesChain& chain = entry.responses_chain;
  if (!request.previous_response_id.has_value()) {
    if (chain.initialized) {
      return gem16::Status(gem16::StatusCode::kInvalidArgument,
                           "response session already has a root");
    }
    return gem16::Status::Ok();
  }
  if (!chain.initialized ||
      *request.previous_response_id != chain.latest_response_id) {
    return gem16::Status(
        gem16::StatusCode::kNotFound,
        "previous_response_id is not the latest resident response");
  }
  if (request.tools_present && request.generation.tools != chain.tools) {
    return gem16::Status(
        gem16::StatusCode::kInvalidArgument,
        "tools must remain identical on a resident response chain");
  }
  if (!request.tools_present) request.generation.tools = chain.tools;
  if (request.tool_choice_present &&
      request.generation.tool_choice != chain.tool_choice) {
    return gem16::Status(
        gem16::StatusCode::kInvalidArgument,
        "tool_choice must remain identical on a resident response chain");
  }
  if (!request.tool_choice_present) {
    request.generation.tool_choice = chain.tool_choice;
  }
  std::vector<gem16::GenerationMessage> complete = chain.messages;
  complete.insert(complete.end(), request.generation.messages.begin(),
                  request.generation.messages.end());
  request.generation.messages = std::move(complete);
  return gem16::Status::Ok();
}

void CommitResponsesRequest(
    SessionEntry& entry, const gem16::server::OpenAiResponsesRequest& request,
    const gem16::ChatGenerationResponse& response,
    std::string response_id) {
  ResponsesChain& chain = entry.responses_chain;
  chain.messages = request.generation.messages;
  gem16::GenerationMessage assistant;
  assistant.role = "assistant";
  if (!response.assistant_text.empty() || response.tool_calls.empty()) {
    assistant.content.push_back(
        gem16::GenerationContentPart::Text(response.assistant_text));
  }
  for (const gem16::GenerationToolCall& call : response.tool_calls) {
    assistant.content.push_back(
        gem16::GenerationContentPart::ToolCall(call));
  }
  chain.messages.push_back(std::move(assistant));
  chain.tools = request.generation.tools;
  chain.tool_choice = request.generation.tool_choice;
  chain.latest_response_id = std::move(response_id);
  chain.initialized = true;
}

void IndexResponse(ServerState& state,
                   const std::shared_ptr<SessionEntry>& entry,
                   const std::string& response_id) {
  std::lock_guard pool_lock(state.pool_mutex);
  state.response_index[response_id] = entry;
}

void UnindexResponse(ServerState& state, std::string_view response_id) {
  std::lock_guard pool_lock(state.pool_mutex);
  state.response_index.erase(std::string(response_id));
}

void SetActiveResponse(ServerState& state,
                       const std::shared_ptr<SessionEntry>& entry,
                       std::string_view response_id) {
  std::lock_guard pool_lock(state.pool_mutex);
  entry->active_response_id = response_id;
}

void ClearActiveResponse(ServerState& state,
                         const std::shared_ptr<SessionEntry>& entry,
                         std::string_view response_id) {
  std::lock_guard pool_lock(state.pool_mutex);
  if (entry->active_response_id == response_id) {
    entry->active_response_id.clear();
  }
}

void RecordGeneration(ServerState& state,
                      const gem16::ChatGenerationResponse& response,
                      std::chrono::steady_clock::duration elapsed) {
  state.metrics.input_tokens.fetch_add(response.prompt_token_ids.size());
  state.metrics.cached_input_tokens.fetch_add(
      response.inference.prompt_cached_tokens);
  state.metrics.cache_write_tokens.fetch_add(
      response.inference.prompt_cache_write_tokens);
  state.metrics.output_tokens.fetch_add(
      response.inference.output_token_ids.size());
  state.metrics.generation_microseconds.fetch_add(
      static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::microseconds>(elapsed)
              .count()));
  state.metrics.last_slot_bytes.store(
      response.inference.kv_cache_bytes + response.inference.workspace_bytes +
      response.inference.assistant_workspace_bytes +
      response.inference.decode_graph_device_bytes);
}

struct CancellationContext {
  std::atomic<bool>* cancel_requested = nullptr;
  std::atomic<std::uint64_t>* cancellations_observed = nullptr;
  std::atomic<std::uint64_t>* client_disconnects = nullptr;
  httplib::DataSink* sink = nullptr;
};

gem16::Status CheckCancellation(void* opaque_context,
                                const gem16::GenerationEvent& event) {
  auto* context = static_cast<CancellationContext*>(opaque_context);
  if (context == nullptr ||
      event.kind != gem16::GenerationEventKind::kToken) {
    return gem16::Status(gem16::StatusCode::kInternal,
                         "invalid cancellation callback");
  }
  if (context->cancel_requested != nullptr &&
      context->cancel_requested->load()) {
    if (context->cancellations_observed != nullptr) {
      context->cancellations_observed->fetch_add(1U);
    }
    return gem16::Status(gem16::StatusCode::kCancelled,
                         "generation was cancelled");
  }
  if (context->sink != nullptr && context->sink->is_writable &&
      !context->sink->is_writable()) {
    if (context->client_disconnects != nullptr) {
      context->client_disconnects->fetch_add(1U);
    }
    return gem16::Status(gem16::StatusCode::kCancelled,
                         "client disconnected during generation");
  }
  return gem16::Status::Ok();
}

void HandleCompletion(ServerState& state, const httplib::Request& request,
                      httplib::Response& response) {
  state.metrics.requests_total.fetch_add(1U);
  auto parsed = gem16::server::ParseChatCompletionsRequest(
      request.body, {state.max_context});
  if (!parsed.ok()) {
    state.metrics.requests_failed.fetch_add(1U);
    SetError(parsed.status(), response);
    return;
  }
  if (parsed.value().model != state.model_name) {
    state.metrics.requests_failed.fetch_add(1U);
    SetError(gem16::Status(gem16::StatusCode::kNotFound,
                           "requested model is not served"),
             response);
    return;
  }
  const gem16::server::OpenAiResponseIdentity identity = MakeIdentity(state);
  std::string session_id = request.get_header_value("X-Gem16-Session-Id");
  if (session_id.empty()) {
    session_id = "session_" +
                 std::to_string(state.session_counter.fetch_add(1U));
  }
  if (session_id.size() > 128U ||
      !std::all_of(session_id.begin(), session_id.end(), [](char value) {
        return (value >= 'a' && value <= 'z') ||
               (value >= 'A' && value <= 'Z') ||
               (value >= '0' && value <= '9') || value == '-' ||
               value == '_' || value == '.';
      })) {
    state.metrics.requests_failed.fetch_add(1U);
    SetError(gem16::Status(gem16::StatusCode::kInvalidArgument,
                           "X-Gem16-Session-Id is invalid"),
             response);
    return;
  }
  auto acquired = AcquireNamedSession(state, "chat:" + session_id);
  if (!acquired.ok()) {
    state.metrics.requests_failed.fetch_add(1U);
    SetError(acquired.status(), response);
    return;
  }
  std::shared_ptr<SessionEntry> entry = std::move(acquired).value();
  response.set_header("X-Gem16-Session-Id", session_id);
  if (!parsed.value().stream) {
    std::lock_guard inference_lock(entry->inference_mutex);
    entry->cancel_requested.store(false);
    CancellationContext cancellation{
        &entry->cancel_requested, &state.metrics.cancellations_observed,
        &state.metrics.client_disconnects, nullptr};
    const auto generation_start = std::chrono::steady_clock::now();
    auto generated = entry->session.Generate(
        parsed.value().generation, CheckCancellation, &cancellation);
    if (!generated.ok()) {
      state.metrics.requests_failed.fetch_add(1U);
      SetError(generated.status(), response);
      DiscardSession(state, entry);
      ReleaseSession(state, entry);
      return;
    }
    RecordGeneration(state, generated.value(),
                     std::chrono::steady_clock::now() - generation_start);
    response.set_content(
        gem16::server::ChatCompletionJson(identity, generated.value()),
        "application/json; charset=utf-8");
    ReleaseSession(state, entry);
    return;
  }

  struct ProviderState {
    ServerState* server = nullptr;
    gem16::ChatGenerationRequest generation;
    gem16::server::OpenAiResponseIdentity identity;
    std::shared_ptr<SessionEntry> entry;
    std::unique_ptr<SessionLease> lease;
    bool include_usage = false;
    bool ran = false;
  };
  auto provider = std::make_shared<ProviderState>();
  provider->server = &state;
  provider->generation = std::move(parsed.value().generation);
  provider->identity = identity;
  provider->entry = entry;
  provider->lease = std::make_unique<SessionLease>(state, entry);
  provider->include_usage = parsed.value().include_usage;
  response.set_header("Cache-Control", "no-cache");
  response.set_header("X-Accel-Buffering", "no");
  response.set_header("Transfer-Encoding", "chunked");
  response.set_content_provider(
      "text/event-stream; charset=utf-8",
      [provider](std::size_t, httplib::DataSink& sink) {
        if (provider->ran) return false;
        provider->ran = true;
        std::lock_guard inference_lock(provider->entry->inference_mutex);
        provider->entry->cancel_requested.store(false);
        if (!WriteSse(sink, gem16::server::ChatCompletionChunkJson(
                                provider->identity,
                                "{\"role\":\"assistant\"}"))) {
          provider->lease->Discard();
          return false;
        }
        StreamingContext stream(provider->server->processor,
                                provider->identity, sink);
        stream.cancel_requested = &provider->entry->cancel_requested;
        stream.cancellations_observed =
            &provider->server->metrics.cancellations_observed;
        stream.client_disconnects =
            &provider->server->metrics.client_disconnects;
        const auto generation_start = std::chrono::steady_clock::now();
        provider->lease->Discard();
        auto generated = provider->entry->session.Generate(
            provider->generation, StreamToken, &stream);
        if (generated.ok()) {
          const gem16::Status final_status = FeedVisibleText(stream, {}, true);
          if (!final_status.ok()) generated = final_status;
        }
        if (generated.ok()) {
          const gem16::Status final_status =
              FeedReasoningText(stream, {}, true);
          if (!final_status.ok()) generated = final_status;
        }
        if (generated.ok() && stream.inside_tool_call) {
          generated = gem16::Status(
              gem16::StatusCode::kDataLoss,
              "model response ends inside a streamed tool call");
        }
        if (!generated.ok()) {
          provider->server->metrics.requests_failed.fetch_add(1U);
          WriteSse(sink, gem16::server::OpenAiErrorJson(
                             generated.status().message(), "server_error"));
          (void)FinishSse(sink);
          provider->lease->Discard();
          return true;
        }
        RecordGeneration(*provider->server, generated.value(),
                         std::chrono::steady_clock::now() - generation_start);
        for (const gem16::GenerationToolCall& call :
             generated.value().tool_calls) {
          gem16::GenerationEvent start;
          start.kind = gem16::GenerationEventKind::kToolCallStart;
          start.tool_call_id = call.id;
          start.tool_name = call.name;
          gem16::Status stream_status = StreamDelta(stream, start);
          if (!stream_status.ok()) return false;
          gem16::GenerationEvent arguments;
          arguments.kind =
              gem16::GenerationEventKind::kToolCallArgumentsDelta;
          arguments.tool_call_id = call.id;
          arguments.tool_name = call.name;
          arguments.text_delta = call.arguments_json;
          stream_status = StreamDelta(stream, arguments);
          if (!stream_status.ok()) return false;
        }
        if (!WriteSse(sink, gem16::server::ChatCompletionChunkJson(
                                provider->identity, {},
                                generated.value().finish_reason))) {
          return false;
        }
        if (provider->include_usage &&
            !WriteSse(sink, gem16::server::ChatCompletionChunkJson(
                                    provider->identity, {}, std::nullopt,
                                    &generated.value()))) {
          return false;
        }
        if (!WriteSse(sink, "[DONE]") || !FinishSse(sink)) return false;
        provider->lease->Keep();
        return true;
      });
}

bool WriteResponsesFinalEvents(
    httplib::DataSink& sink,
    const gem16::server::OpenAiResponseIdentity& identity,
    const gem16::server::OpenAiResponsesRequest& request,
    const gem16::ChatGenerationResponse& generated,
    ResponsesStreamingContext& live) {
  gem16::Status live_status = FinalizeResponseReasoning(live);
  if (!live_status.ok()) return false;
  if (!generated.assistant_text.empty() || generated.tool_calls.empty()) {
    const bool text_was_streamed = live.message_started;
    live_status = StartResponseMessage(live);
    if (!live_status.ok()) return false;
    const std::size_t output_index = live.message_index;
    const std::string item_id = "msg_" + identity.id;
    if (!text_was_streamed && !generated.assistant_text.empty() &&
        !WriteSse(sink, "{\"type\":\"response.output_text.delta\","
                        "\"item_id\":" +
                            gem16::json::Quote(item_id) +
                            ",\"output_index\":" +
                            std::to_string(output_index) +
                            ",\"content_index\":0,\"delta\":" +
                            gem16::json::Quote(generated.assistant_text) +
                            ",\"logprobs\":[],\"sequence_number\":" +
                            std::to_string(live.sequence++) + "}")) {
      return false;
    }
    const std::string completed_part =
        "{\"type\":\"output_text\",\"text\":" +
        gem16::json::Quote(generated.assistant_text) +
        ",\"annotations\":[]}";
    if (!WriteSse(sink, "{\"type\":\"response.output_text.done\","
                        "\"item_id\":" +
                            gem16::json::Quote(item_id) +
                            ",\"output_index\":" +
                            std::to_string(output_index) +
                            ",\"content_index\":0,\"text\":" +
                            gem16::json::Quote(generated.assistant_text) +
                            ",\"logprobs\":[],\"sequence_number\":" +
                            std::to_string(live.sequence++) + "}")) {
      return false;
    }
    if (!WriteSse(sink, "{\"type\":\"response.content_part.done\","
                        "\"item_id\":" +
                            gem16::json::Quote(item_id) +
                            ",\"output_index\":" +
                            std::to_string(output_index) +
                            ",\"content_index\":0,\"part\":" +
                            completed_part + ",\"sequence_number\":" +
                            std::to_string(live.sequence++) + "}")) {
      return false;
    }
    const std::string done_item =
        "{\"id\":" + gem16::json::Quote(item_id) +
        ",\"type\":\"message\",\"status\":\"completed\","
        "\"role\":\"assistant\",\"content\":[" + completed_part + "]}";
    if (!WriteSse(sink, "{\"type\":\"response.output_item.done\","
                        "\"output_index\":" +
                            std::to_string(output_index) + ",\"item\":" +
                            done_item + ",\"sequence_number\":" +
                            std::to_string(live.sequence++) + "}")) {
      return false;
    }
    ++live.output_index;
  }
  for (std::size_t call_index = 0U;
       call_index < generated.tool_calls.size(); ++call_index) {
    const std::size_t output_index = live.output_index;
    const gem16::GenerationToolCall& call = generated.tool_calls[call_index];
    const std::string item_id =
        "fc_" + identity.id + "_" + std::to_string(call_index);
    const auto item = [&](std::string_view status) {
      return "{\"id\":" + gem16::json::Quote(item_id) +
             ",\"type\":\"function_call\",\"status\":" +
             gem16::json::Quote(status) + ",\"arguments\":" +
             gem16::json::Quote(call.arguments_json) + ",\"call_id\":" +
             gem16::json::Quote(call.id) + ",\"name\":" +
             gem16::json::Quote(call.name) + "}";
    };
    if (!WriteSse(sink, "{\"type\":\"response.output_item.added\","
                        "\"output_index\":" +
                            std::to_string(output_index) + ",\"item\":" +
                            item("in_progress") +
                            ",\"sequence_number\":" +
                            std::to_string(live.sequence++) + "}")) {
      return false;
    }
    if (!WriteSse(sink, "{\"type\":\"response.function_call_arguments.done\","
                        "\"item_id\":" +
                            gem16::json::Quote(item_id) +
                            ",\"output_index\":" +
                            std::to_string(output_index) + ",\"arguments\":" +
                            gem16::json::Quote(call.arguments_json) +
                            ",\"name\":" + gem16::json::Quote(call.name) +
                            ",\"sequence_number\":" +
                            std::to_string(live.sequence++) + "}")) {
      return false;
    }
    if (!WriteSse(sink, "{\"type\":\"response.output_item.done\","
                        "\"output_index\":" +
                            std::to_string(output_index) + ",\"item\":" +
                            item("completed") + ",\"sequence_number\":" +
                            std::to_string(live.sequence++) + "}")) {
      return false;
    }
    ++live.output_index;
  }
  const std::string completed = gem16::server::ResponseJson(
      identity, request, generated);
  return WriteSse(sink, "{\"type\":\"response.completed\",\"response\":" +
                            completed + ",\"sequence_number\":" +
                            std::to_string(live.sequence++) + "}");
}

void HandleResponses(ServerState& state, const httplib::Request& request,
                     httplib::Response& response) {
  state.metrics.requests_total.fetch_add(1U);
  auto parsed = gem16::server::ParseResponsesRequest(
      request.body, {state.max_context});
  if (!parsed.ok()) {
    state.metrics.requests_failed.fetch_add(1U);
    SetError(parsed.status(), response);
    return;
  }
  if (parsed.value().model != state.model_name) {
    state.metrics.requests_failed.fetch_add(1U);
    SetError(gem16::Status(gem16::StatusCode::kNotFound,
                           "requested model is not served"),
             response);
    return;
  }
  const gem16::server::OpenAiResponseIdentity identity =
      MakeResponsesIdentity(state);
  gem16::Result<std::shared_ptr<SessionEntry>> acquired =
      parsed.value().previous_response_id.has_value()
          ? AcquireResponseSession(state,
                                   *parsed.value().previous_response_id)
          : CreateSession(state, "responses:" + identity.id);
  if (!acquired.ok()) {
    state.metrics.requests_failed.fetch_add(1U);
    SetError(acquired.status(), response);
    return;
  }
  std::shared_ptr<SessionEntry> entry = std::move(acquired).value();
  IndexResponse(state, entry, identity.id);
  if (!parsed.value().stream) {
    std::lock_guard inference_lock(entry->inference_mutex);
    gem16::Status status = PrepareResponsesRequest(*entry, parsed.value());
    if (!status.ok()) {
      state.metrics.requests_failed.fetch_add(1U);
      SetError(status, response);
      UnindexResponse(state, identity.id);
      ReleaseSession(state, entry);
      return;
    }
    entry->cancel_requested.store(false);
    CancellationContext cancellation{
        &entry->cancel_requested, &state.metrics.cancellations_observed,
        &state.metrics.client_disconnects, nullptr};
    const auto generation_start = std::chrono::steady_clock::now();
    auto generated = entry->session.Generate(
        parsed.value().generation, CheckCancellation, &cancellation);
    if (!generated.ok()) {
      state.metrics.requests_failed.fetch_add(1U);
      SetError(generated.status(), response);
      DiscardSession(state, entry);
      ReleaseSession(state, entry);
      return;
    }
    RecordGeneration(state, generated.value(),
                     std::chrono::steady_clock::now() - generation_start);
    CommitResponsesRequest(*entry, parsed.value(), generated.value(),
                           identity.id);
    response.set_content(
        gem16::server::ResponseJson(identity, parsed.value(),
                                    generated.value()),
        "application/json; charset=utf-8");
    ReleaseSession(state, entry);
    return;
  }

  struct ProviderState {
    ServerState* server = nullptr;
    gem16::server::OpenAiResponsesRequest request;
    gem16::server::OpenAiResponseIdentity identity;
    std::shared_ptr<SessionEntry> entry;
    std::unique_ptr<SessionLease> lease;
    std::atomic<bool> keep_response_index{false};
    bool ran = false;

    ~ProviderState() {
      if (server != nullptr &&
          !keep_response_index.load(std::memory_order_acquire)) {
        UnindexResponse(*server, identity.id);
      }
    }
  };
  auto provider = std::make_shared<ProviderState>();
  provider->server = &state;
  provider->request = std::move(parsed).value();
  provider->identity = identity;
  provider->entry = entry;
  provider->lease = std::make_unique<SessionLease>(state, entry);
  response.set_header("Cache-Control", "no-cache");
  response.set_header("X-Accel-Buffering", "no");
  response.set_header("Transfer-Encoding", "chunked");
  response.set_content_provider(
      "text/event-stream; charset=utf-8",
      [provider](std::size_t, httplib::DataSink& sink) {
        if (provider->ran) return false;
        provider->ran = true;
        std::lock_guard inference_lock(provider->entry->inference_mutex);
        gem16::Status status = PrepareResponsesRequest(
            *provider->entry, provider->request);
        if (!status.ok()) {
          provider->server->metrics.requests_failed.fetch_add(1U);
          WriteSse(sink, "{\"type\":\"error\",\"code\":null,\"message\":" +
                             gem16::json::Quote(status.message()) +
                             ",\"param\":null,\"sequence_number\":0}");
          (void)FinishSse(sink);
          return true;
        }
        provider->entry->cancel_requested.store(false);
        SetActiveResponse(*provider->server, provider->entry,
                          provider->identity.id);
        if (!WriteSse(
                sink,
                "{\"type\":\"response.created\",\"response\":" +
                    gem16::server::ResponseShellJson(
                        provider->identity, provider->request, "in_progress") +
                    ",\"sequence_number\":0}")) {
          ClearActiveResponse(*provider->server, provider->entry,
                              provider->identity.id);
          return false;
        }
        std::uint64_t reasoning_capacity = gem16::ThinkingBudgetTokens(
            provider->request.generation.thinking.effort);
        // The checkpoint template leaves a tool-result continuation at the
        // model boundary. Even with thinking disabled, Gemma emits an empty
        // thought channel containing one newline before the visible answer.
        if (reasoning_capacity == 0U &&
            !provider->request.generation.messages.empty() &&
            provider->request.generation.messages.back().role == "tool") {
          reasoning_capacity = 1U;
        }
        if (provider->request.generation.max_generated_tokens.has_value()) {
          reasoning_capacity = std::min(
              reasoning_capacity,
              *provider->request.generation.max_generated_tokens);
        }
        ResponsesStreamingContext stream(
            provider->server->processor, provider->identity, sink,
            reasoning_capacity);
        stream.cancel_requested = &provider->entry->cancel_requested;
        stream.cancellations_observed =
            &provider->server->metrics.cancellations_observed;
        stream.client_disconnects =
            &provider->server->metrics.client_disconnects;
        const auto generation_start = std::chrono::steady_clock::now();
        provider->lease->Discard();
        auto generated = provider->entry->session.Generate(
            provider->request.generation, StreamResponseToken, &stream);
        if (generated.ok() &&
            (stream.text_pending.size != 0U ||
             stream.reasoning_pending.size != 0U ||
             stream.inside_tool_call)) {
          generated = gem16::Status(
              gem16::StatusCode::kDataLoss,
              "model response ends with incomplete streamed content");
        }
        if (generated.ok()) {
          const std::string_view streamed_reasoning =
              stream.reasoning_text_size == 0U
                  ? std::string_view{}
                  : std::string_view(stream.reasoning_text.data(),
                                     stream.reasoning_text_size);
          if (streamed_reasoning != generated.value().reasoning_text) {
            generated = gem16::Status(
                gem16::StatusCode::kInternal,
                "live reasoning stream disagrees with final response");
          }
        }
        ClearActiveResponse(*provider->server, provider->entry,
                            provider->identity.id);
        if (!generated.ok()) {
          provider->server->metrics.requests_failed.fetch_add(1U);
          WriteSse(sink, "{\"type\":\"error\",\"code\":null,\"message\":" +
                             gem16::json::Quote(generated.status().message()) +
                             ",\"param\":null,\"sequence_number\":" +
                             std::to_string(stream.sequence++) + "}");
          (void)FinishSse(sink);
          return true;
        }
        RecordGeneration(*provider->server, generated.value(),
                         std::chrono::steady_clock::now() - generation_start);
        CommitResponsesRequest(*provider->entry, provider->request,
                               generated.value(), provider->identity.id);
        // Generation and KV commit are complete at this point. The OpenAI SDK
        // may stop reading immediately after response.completed, so a later
        // final-chunk write failure must not discard an otherwise safe chain.
        provider->keep_response_index.store(true, std::memory_order_release);
        provider->lease->Keep();
        const bool written = WriteResponsesFinalEvents(
            sink, provider->identity, provider->request, generated.value(),
            stream);
        return written && FinishSse(sink);
      });
}

void HandleCancelResponse(ServerState& state, std::string_view response_id,
                          httplib::Response& response) {
  bool cancellation_requested = false;
  {
    std::lock_guard pool_lock(state.pool_mutex);
    const auto found = state.response_index.find(std::string(response_id));
    if (found != state.response_index.end()) {
      const std::shared_ptr<SessionEntry> entry = found->second.lock();
      if (entry != nullptr && entry->active_response_id == response_id &&
          entry->active_requests.load() != 0U) {
        entry->cancel_requested.store(true);
        state.metrics.cancellations_requested.fetch_add(1U);
        cancellation_requested = true;
      }
    }
  }
  if (!cancellation_requested) {
    SetError(gem16::Status(gem16::StatusCode::kNotFound,
                           "response is not actively generating"),
             response);
    return;
  }
  response.set_content(
      "{\"id\":" + gem16::json::Quote(response_id) +
          ",\"object\":\"response\",\"status\":\"cancelling\"}",
      "application/json; charset=utf-8");
}

std::string MetricsText(ServerState& state) {
  std::size_t resident_sessions = 0U;
  {
    std::lock_guard pool_lock(state.pool_mutex);
    resident_sessions = state.sessions.size();
  }
  const auto metric = [](std::string_view name, std::uint64_t value) {
    return std::string(name) + " " + std::to_string(value) + "\n";
  };
  std::string output;
  output.append("# TYPE gem16_requests_total counter\n");
  output.append(metric("gem16_requests_total",
                       state.metrics.requests_total.load()));
  output.append("# TYPE gem16_requests_failed_total counter\n");
  output.append(metric("gem16_requests_failed_total",
                       state.metrics.requests_failed.load()));
  output.append("# TYPE gem16_active_requests gauge\n");
  output.append(metric("gem16_active_requests",
                       state.metrics.active_requests.load()));
  output.append("# TYPE gem16_resident_sessions gauge\n");
  output.append(metric("gem16_resident_sessions", resident_sessions));
  output.append(metric("gem16_session_limit", state.max_sessions));
  output.append(metric("gem16_sessions_created_total",
                       state.metrics.sessions_created.load()));
  output.append(metric("gem16_sessions_evicted_total",
                       state.metrics.sessions_evicted.load()));
  output.append(metric("gem16_cancellations_requested_total",
                       state.metrics.cancellations_requested.load()));
  output.append(metric("gem16_cancellations_observed_total",
                       state.metrics.cancellations_observed.load()));
  output.append(metric("gem16_client_disconnects_total",
                       state.metrics.client_disconnects.load()));
  output.append(metric("gem16_input_tokens_total",
                       state.metrics.input_tokens.load()));
  output.append(metric("gem16_cached_input_tokens_total",
                       state.metrics.cached_input_tokens.load()));
  output.append(metric("gem16_cache_write_tokens_total",
                       state.metrics.cache_write_tokens.load()));
  output.append(metric("gem16_output_tokens_total",
                       state.metrics.output_tokens.load()));
  output.append(metric("gem16_generation_microseconds_total",
                       state.metrics.generation_microseconds.load()));
  output.append(metric("gem16_model_weight_bytes",
                       state.runtime->weight_bytes()));
  output.append(metric("gem16_assistant_weight_bytes",
                       state.runtime->assistant_weight_bytes()));
  output.append(metric("gem16_last_execution_slot_bytes",
                       state.metrics.last_slot_bytes.load()));
  return output;
}

int ServerMain(int argc, char** argv) {
  if (argc == 2 && (std::string_view(argv[1]) == "--help" ||
                    std::string_view(argv[1]) == "-h")) {
    PrintUsage();
    return 0;
  }
  auto options = ParseOptions(argc, argv);
  if (!options.ok()) {
    std::cerr << "error: " << options.status().message() << '\n';
    PrintUsage();
    return 64;
  }
  auto processor =
      gem16::GemmaChatProcessor::Load(options.value().model_directory);
  if (!processor.ok()) {
    std::cerr << "error: " << processor.status().message() << '\n';
    return 2;
  }
  gem16::ChatSessionOptions session_options;
  session_options.model_directory = options.value().model_directory;
  session_options.assistant_model_directory =
      options.value().assistant_model_directory;
  session_options.max_context_tokens = options.value().max_context;
  session_options.kv_cache_mode = options.value().kv_cache_mode;
  session_options.sampling =
      processor.value().generation_controls().recommended_sampling;
  if (options.value().greedy) session_options.sampling.enabled = false;
  session_options.mtp_draft_tokens = options.value().mtp_draft_tokens;
  session_options.mtp_adaptive = options.value().mtp_adaptive;
  auto runtime = gem16::ModelRuntime::Load(
      {options.value().model_directory,
       options.value().assistant_model_directory});
  if (!runtime.ok()) {
    std::cerr << "error: " << runtime.status().message() << '\n';
    return 2;
  }
  std::cout << "model_runtime weights=" << runtime.value()->weight_bytes()
            << " assistant_weights="
            << runtime.value()->assistant_weight_bytes()
            << " load_ms=" << runtime.value()->load_milliseconds() << '\n';
  ServerState state(options.value().model_name, options.value().max_context,
                    std::move(processor).value(), runtime.value(),
                    session_options, options.value().max_sessions);
  httplib::Server server;
  server.Get("/health",
             [&state](const httplib::Request&, httplib::Response& response) {
               std::size_t resident = 0U;
               {
                 std::lock_guard pool_lock(state.pool_mutex);
                 resident = state.sessions.size();
               }
               response.set_content(
                   "{\"status\":\"ok\",\"resident_sessions\":" +
                       std::to_string(resident) +
                       ",\"session_limit\":" +
                       std::to_string(state.max_sessions) + "}",
                   "application/json; charset=utf-8");
             });
  server.Get("/metrics",
             [&state](const httplib::Request&, httplib::Response& response) {
               response.set_content(MetricsText(state),
                                    "text/plain; version=0.0.4; charset=utf-8");
             });
  server.Get("/v1/models",
             [&state](const httplib::Request&, httplib::Response& response) {
               response.set_content(
                   "{\"object\":\"list\",\"data\":[{\"id\":" +
                       gem16::json::Quote(state.model_name) +
                       ",\"object\":\"model\",\"owned_by\":\"gem16\"}]}",
                   "application/json; charset=utf-8");
             });
  server.Post("/v1/chat/completions",
              [&state](const httplib::Request& request,
                       httplib::Response& response) {
                HandleCompletion(state, request, response);
              });
  server.Post("/v1/responses",
              [&state](const httplib::Request& request,
                       httplib::Response& response) {
                HandleResponses(state, request, response);
              });
  server.Post(R"(/v1/responses/([^/]+)/cancel)",
              [&state](const httplib::Request& request,
                       httplib::Response& response) {
                HandleCancelResponse(state, request.matches[1].str(), response);
              });
  server.set_payload_max_length(16U * 1024U * 1024U);
  std::cout << "gem16 OpenAI-compatible server listening on http://"
            << options.value().host << ':' << options.value().port
            << " (model " << state.model_name
            << ", one resident conversation / one execution slot)\n";
  if (!server.listen(options.value().host, options.value().port)) {
    std::cerr << "error: failed to listen on requested address\n";
    return 2;
  }
  return 0;
}

}  // namespace

#if defined(_WIN32)

int wmain(int argc, wchar_t** wide_argv) {
  std::vector<std::string> utf8_arguments;
  utf8_arguments.reserve(static_cast<std::size_t>(argc));
  for (int index = 0; index < argc; ++index) {
    auto converted = gem16::cli::WideToUtf8(wide_argv[index]);
    if (!converted.has_value()) {
      std::cerr << "error: a command-line argument is not valid Unicode\n";
      return 64;
    }
    utf8_arguments.push_back(std::move(*converted));
  }
  std::vector<char*> arguments;
  arguments.reserve(utf8_arguments.size());
  for (std::string& argument : utf8_arguments) {
    arguments.push_back(argument.data());
  }
  return ServerMain(argc, arguments.data());
}

#else

int main(int argc, char** argv) { return ServerMain(argc, argv); }

#endif
