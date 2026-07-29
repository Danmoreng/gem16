#include <atomic>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#if defined(_WIN32)
#include "windows_utf8.h"
#endif

#include "httplib.h"

#include "gem16/chat.h"
#include "gem16/tokenizer.h"
#include "server/openai_chat.h"
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
  const std::string record = "data: " + std::string(payload) + "\n\n";
  return sink.write(record.data(), record.size());
}

std::size_t CompleteUtf8Prefix(std::string_view text) {
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
      return position;
    }
    if (position + width > text.size()) return position;
    bool valid = true;
    for (std::size_t offset = 1U; offset < width; ++offset) {
      const unsigned char continuation =
          static_cast<unsigned char>(text[position + offset]);
      valid = valid && continuation >= 0x80U && continuation <= 0xBFU;
    }
    if (!valid) return position;
    position += width;
  }
  return position;
}

struct StreamingContext {
  StreamingContext(const gem16::GemmaChatProcessor& chat_processor,
                   gem16::server::OpenAiResponseIdentity response_identity,
                   httplib::DataSink& data_sink)
      : processor(&chat_processor),
        identity(std::move(response_identity)),
        sink(&data_sink),
        channels(chat_processor.generation_controls()) {}

  const gem16::GemmaChatProcessor* processor = nullptr;
  gem16::server::OpenAiResponseIdentity identity;
  httplib::DataSink* sink = nullptr;
  gem16::ResponseChannelTracker channels;
  std::map<std::string, std::size_t, std::less<>> tool_indices;
  std::string utf8_pending;
  std::string reasoning_utf8_pending;
  bool inside_tool_call = false;
};

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

gem16::Status FeedVisibleText(StreamingContext& context,
                              std::string_view bytes, bool final) {
  context.utf8_pending.append(bytes);
  const std::size_t complete = CompleteUtf8Prefix(context.utf8_pending);
  if (complete != 0U) {
    gem16::GenerationEvent event;
    event.kind = gem16::GenerationEventKind::kTextDelta;
    event.text_delta = context.utf8_pending.substr(0U, complete);
    context.utf8_pending.erase(0U, complete);
    const gem16::Status status = StreamDelta(context, event);
    if (!status.ok()) return status;
  }
  if (!final) return gem16::Status::Ok();
  if (!context.utf8_pending.empty()) {
    return gem16::Status(gem16::StatusCode::kDataLoss,
                         "model response ends with incomplete UTF-8");
  }
  return gem16::Status::Ok();
}

gem16::Status FeedReasoningText(StreamingContext& context,
                                std::string_view bytes, bool final) {
  context.reasoning_utf8_pending.append(bytes);
  const std::size_t complete =
      CompleteUtf8Prefix(context.reasoning_utf8_pending);
  if (complete != 0U) {
    const std::string delta =
        "{\"reasoning_content\":" +
        gem16::json::Quote(
            std::string_view(context.reasoning_utf8_pending)
                .substr(0U, complete)) +
        "}";
    context.reasoning_utf8_pending.erase(0U, complete);
    if (!WriteSse(*context.sink,
                  gem16::server::ChatCompletionChunkJson(context.identity,
                                                          delta))) {
      return gem16::Status(gem16::StatusCode::kIoError,
                           "client disconnected during SSE generation");
    }
  }
  if (final && !context.reasoning_utf8_pending.empty()) {
    return gem16::Status(gem16::StatusCode::kDataLoss,
                         "reasoning response ends with incomplete UTF-8");
  }
  return gem16::Status::Ok();
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
  const gem16::ResponseTokenChannel channel =
      context->channels.Observe(event.token_id);
  if (channel == gem16::ResponseTokenChannel::kControl) {
    return gem16::Status::Ok();
  }
  const std::uint32_t token_id = event.token_id;
  auto visible = context->processor->Decode(
      std::span<const std::uint32_t>(&token_id, 1U), true);
  if (!visible.ok()) return visible.status();
  if (channel == gem16::ResponseTokenChannel::kReasoning) {
    return FeedReasoningText(*context, visible.value(), false);
  }
  auto raw = context->processor->Decode(
      std::span<const std::uint32_t>(&token_id, 1U), false);
  if (!raw.ok()) return raw.status();
  if (raw.value() == "<|tool_call>") {
    context->inside_tool_call = true;
    return gem16::Status::Ok();
  }
  if (raw.value() == "<tool_call|>") {
    context->inside_tool_call = false;
    return gem16::Status::Ok();
  }
  if (context->inside_tool_call) return gem16::Status::Ok();
  return FeedVisibleText(*context, visible.value(), false);
}

struct ServerState {
  std::string model_name;
  std::uint64_t max_context = 0U;
  gem16::GemmaChatProcessor processor;
  gem16::ChatSession session;
  std::mutex session_mutex;
  std::atomic<std::uint64_t> response_counter{1U};
};

gem16::server::OpenAiResponseIdentity MakeIdentity(ServerState& state) {
  return {"chatcmpl-gem16-" +
              std::to_string(state.response_counter.fetch_add(1U)),
          state.model_name, UnixSeconds()};
}

void HandleCompletion(ServerState& state, const httplib::Request& request,
                      httplib::Response& response) {
  auto parsed = gem16::server::ParseChatCompletionsRequest(
      request.body, {state.max_context});
  if (!parsed.ok()) {
    SetError(parsed.status(), response);
    return;
  }
  if (parsed.value().model != state.model_name) {
    SetError(gem16::Status(gem16::StatusCode::kNotFound,
                           "requested model is not served"),
             response);
    return;
  }
  const gem16::server::OpenAiResponseIdentity identity = MakeIdentity(state);
  if (!parsed.value().stream) {
    std::lock_guard lock(state.session_mutex);
    auto generated = state.session.Generate(parsed.value().generation);
    if (!generated.ok()) {
      SetError(generated.status(), response);
      return;
    }
    response.set_content(
        gem16::server::ChatCompletionJson(identity, generated.value()),
        "application/json; charset=utf-8");
    return;
  }

  struct ProviderState {
    ServerState* server = nullptr;
    gem16::ChatGenerationRequest generation;
    gem16::server::OpenAiResponseIdentity identity;
    bool include_usage = false;
    bool ran = false;
  };
  auto provider = std::make_shared<ProviderState>();
  provider->server = &state;
  provider->generation = std::move(parsed.value().generation);
  provider->identity = identity;
  provider->include_usage = parsed.value().include_usage;
  response.set_header("Cache-Control", "no-cache");
  response.set_header("X-Accel-Buffering", "no");
  response.set_chunked_content_provider(
      "text/event-stream; charset=utf-8",
      [provider](std::size_t, httplib::DataSink& sink) {
        if (provider->ran) return false;
        provider->ran = true;
        if (!WriteSse(sink, gem16::server::ChatCompletionChunkJson(
                                provider->identity,
                                "{\"role\":\"assistant\"}"))) {
          return false;
        }
        std::lock_guard lock(provider->server->session_mutex);
        StreamingContext stream(provider->server->processor,
                                provider->identity, sink);
        auto generated = provider->server->session.Generate(
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
          WriteSse(sink, gem16::server::OpenAiErrorJson(
                             generated.status().message(), "server_error"));
          sink.done();
          return true;
        }
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
        WriteSse(sink, "[DONE]");
        sink.done();
        return true;
      });
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
  auto session = gem16::ChatSession::Create(session_options, processor.value());
  if (!session.ok()) {
    std::cerr << "error: " << session.status().message() << '\n';
    return 2;
  }
  ServerState state{options.value().model_name, options.value().max_context,
                    std::move(processor).value(),
                    std::move(session).value()};
  httplib::Server server;
  server.Get("/health", [](const httplib::Request&, httplib::Response& response) {
    response.set_content("{\"status\":\"ok\"}",
                         "application/json; charset=utf-8");
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
