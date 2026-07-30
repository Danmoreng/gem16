#include "server/http_streaming.h"

#include <array>
#include <charconv>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "server/sse_chunk.h"
#include "util/json.h"

namespace gem16::server {

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

namespace {

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


}  // namespace

struct ChatCompletionStream::Impl {
  Impl(const GemmaChatProcessor& processor, OpenAiResponseIdentity identity,
       httplib::DataSink& sink, StreamCancellation cancellation)
      : context(processor, std::move(identity), sink) {
    context.cancel_requested = cancellation.requested;
    context.cancellations_observed = cancellation.observed;
    context.client_disconnects = cancellation.disconnects;
  }

  StreamingContext context;
};

ChatCompletionStream::ChatCompletionStream(
    const GemmaChatProcessor& processor, OpenAiResponseIdentity identity,
    httplib::DataSink& sink, StreamCancellation cancellation)
    : impl_(std::make_unique<Impl>(processor, std::move(identity), sink,
                                   cancellation)) {}

ChatCompletionStream::~ChatCompletionStream() = default;

Result<ChatGenerationResponse> ChatCompletionStream::Generate(
    ChatSession& session, const ChatGenerationRequest& request) {
  auto generated = session.Generate(request, StreamToken, &impl_->context);
  if (generated.ok()) {
    const Status status = FeedVisibleText(impl_->context, {}, true);
    if (!status.ok()) generated = status;
  }
  if (generated.ok()) {
    const Status status = FeedReasoningText(impl_->context, {}, true);
    if (!status.ok()) generated = status;
  }
  if (generated.ok() && impl_->context.inside_tool_call) {
    generated = Status(StatusCode::kDataLoss,
                       "model response ends inside a streamed tool call");
  }
  return generated;
}

Status ChatCompletionStream::WriteToolCalls(
    const ChatGenerationResponse& response) {
  for (const GenerationToolCall& call : response.tool_calls) {
    GenerationEvent start;
    start.kind = GenerationEventKind::kToolCallStart;
    start.tool_call_id = call.id;
    start.tool_name = call.name;
    Status status = StreamDelta(impl_->context, start);
    if (!status.ok()) return status;
    GenerationEvent arguments;
    arguments.kind = GenerationEventKind::kToolCallArgumentsDelta;
    arguments.tool_call_id = call.id;
    arguments.tool_name = call.name;
    arguments.text_delta = call.arguments_json;
    status = StreamDelta(impl_->context, arguments);
    if (!status.ok()) return status;
  }
  return Status::Ok();
}

struct ResponsesStream::Impl {
  Impl(const GemmaChatProcessor& processor, OpenAiResponseIdentity identity,
       httplib::DataSink& sink, std::uint64_t reasoning_token_capacity,
       StreamCancellation cancellation)
      : context(processor, std::move(identity), sink,
                reasoning_token_capacity) {
    context.cancel_requested = cancellation.requested;
    context.cancellations_observed = cancellation.observed;
    context.client_disconnects = cancellation.disconnects;
  }

  ResponsesStreamingContext context;
};

ResponsesStream::ResponsesStream(
    const GemmaChatProcessor& processor, OpenAiResponseIdentity identity,
    httplib::DataSink& sink, std::uint64_t reasoning_token_capacity,
    StreamCancellation cancellation)
    : impl_(std::make_unique<Impl>(processor, std::move(identity), sink,
                                   reasoning_token_capacity, cancellation)) {}

ResponsesStream::~ResponsesStream() = default;

Result<ChatGenerationResponse> ResponsesStream::Generate(
    ChatSession& session, const ChatGenerationRequest& request) {
  auto generated = session.Generate(request, StreamResponseToken,
                                    &impl_->context);
  if (generated.ok() &&
      (impl_->context.text_pending.size != 0U ||
       impl_->context.reasoning_pending.size != 0U ||
       impl_->context.inside_tool_call)) {
    generated = Status(StatusCode::kDataLoss,
                       "model response ends with incomplete streamed content");
  }
  if (generated.ok()) {
    const std::string_view streamed_reasoning =
        impl_->context.reasoning_text_size == 0U
            ? std::string_view{}
            : std::string_view(impl_->context.reasoning_text.data(),
                               impl_->context.reasoning_text_size);
    if (streamed_reasoning != generated.value().reasoning_text) {
      generated = Status(StatusCode::kInternal,
                         "live reasoning stream disagrees with final response");
    }
  }
  return generated;
}

bool ResponsesStream::WriteFinalEvents(
    const OpenAiResponsesRequest& request,
    const ChatGenerationResponse& response) {
  return WriteResponsesFinalEvents(*impl_->context.sink,
                                   impl_->context.identity, request, response,
                                   impl_->context);
}

std::uint64_t ResponsesStream::sequence() const {
  return impl_->context.sequence;
}

}  // namespace gem16::server
