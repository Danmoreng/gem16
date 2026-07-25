#include <algorithm>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "gem16gb/engine.h"
#include "gem16gb/tokenizer.h"

namespace {

bool ParseUnsigned(std::string_view text, std::uint64_t& value) {
  const auto result =
      std::from_chars(text.data(), text.data() + text.size(), value);
  return result.ec == std::errc{} && result.ptr == text.data() + text.size();
}

std::string JsonEscape(std::string_view value) {
  std::string result;
  result.reserve(value.size() + 2U);
  result.push_back('"');
  constexpr char kHex[] = "0123456789abcdef";
  for (const unsigned char byte : value) {
    switch (byte) {
      case '"': result.append("\\\""); break;
      case '\\': result.append("\\\\"); break;
      case '\b': result.append("\\b"); break;
      case '\f': result.append("\\f"); break;
      case '\n': result.append("\\n"); break;
      case '\r': result.append("\\r"); break;
      case '\t': result.append("\\t"); break;
      default:
        if (byte < 0x20U) {
          result.append("\\u00");
          result.push_back(kHex[byte >> 4U]);
          result.push_back(kHex[byte & 0x0FU]);
        } else {
          result.push_back(static_cast<char>(byte));
        }
    }
  }
  result.push_back('"');
  return result;
}

void WriteTokenIds(std::span<const std::uint32_t> token_ids) {
  std::cout << '[';
  for (std::size_t index = 0; index < token_ids.size(); ++index) {
    if (index != 0U) std::cout << ',';
    std::cout << token_ids[index];
  }
  std::cout << ']';
}

void PrintUsage() {
  std::cout
      << "Usage:\n"
      << "  gem16gb-chat --model <checkpoint> [--max-tokens N] [--max-context N]\n"
      << "                [--thinking] [--system <text>]\n"
      << "                [--kv-cache fp8|bf16]\n"
      << "                [--dump-state <path> --dump-state-position N]\n"
      << "  gem16gb-chat --model <checkpoint> --message <text> [--json]\n"
      << "  gem16gb-chat --model <checkpoint> --message <text> --render-only --json\n";
}

struct Options {
  std::filesystem::path model_directory;
  std::string system_message;
  std::string one_shot_message;
  std::uint64_t max_tokens = 128;
  std::uint64_t max_context = 1024;
  bool has_system_message = false;
  bool has_one_shot_message = false;
  bool thinking = false;
  bool render_only = false;
  bool json = false;
  std::filesystem::path state_dump_path;
  std::optional<std::uint64_t> state_dump_position;
  gem16gb::KvCacheMode kv_cache_mode =
      gem16gb::KvCacheMode::kCheckpointFp8;
};

gem16gb::Result<Options> ParseOptions(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--model" && index + 1 < argc) {
      options.model_directory = argv[++index];
    } else if (argument == "--system" && index + 1 < argc) {
      options.system_message = argv[++index];
      options.has_system_message = true;
    } else if (argument == "--message" && index + 1 < argc) {
      options.one_shot_message = argv[++index];
      options.has_one_shot_message = true;
    } else if (argument == "--max-tokens" && index + 1 < argc) {
      if (!ParseUnsigned(argv[++index], options.max_tokens)) {
        return gem16gb::Status(gem16gb::StatusCode::kInvalidArgument,
                              "--max-tokens must be an unsigned integer");
      }
    } else if (argument == "--max-context" && index + 1 < argc) {
      if (!ParseUnsigned(argv[++index], options.max_context)) {
        return gem16gb::Status(gem16gb::StatusCode::kInvalidArgument,
                              "--max-context must be an unsigned integer");
      }
    } else if (argument == "--thinking") {
      options.thinking = true;
    } else if (argument == "--render-only") {
      options.render_only = true;
    } else if (argument == "--json") {
      options.json = true;
    } else if (argument == "--dump-state" && index + 1 < argc) {
      options.state_dump_path = argv[++index];
    } else if (argument == "--dump-state-position" && index + 1 < argc) {
      std::uint64_t position = 0;
      if (!ParseUnsigned(argv[++index], position)) {
        return gem16gb::Status(
            gem16gb::StatusCode::kInvalidArgument,
            "--dump-state-position must be an unsigned integer");
      }
      options.state_dump_position = position;
    } else if (argument == "--kv-cache" && index + 1 < argc) {
      const std::string_view mode = argv[++index];
      if (mode == "fp8") {
        options.kv_cache_mode = gem16gb::KvCacheMode::kCheckpointFp8;
      } else if (mode == "bf16") {
        options.kv_cache_mode = gem16gb::KvCacheMode::kBf16Correctness;
      } else {
        return gem16gb::Status(gem16gb::StatusCode::kInvalidArgument,
                              "--kv-cache must be fp8 or bf16");
      }
    } else {
      return gem16gb::Status(gem16gb::StatusCode::kInvalidArgument,
                            "unknown or incomplete option: " +
                                std::string(argument));
    }
  }
  if (options.model_directory.empty()) {
    return gem16gb::Status(gem16gb::StatusCode::kInvalidArgument,
                          "--model is required");
  }
  if ((options.render_only || options.json) &&
      !options.has_one_shot_message) {
    return gem16gb::Status(
        gem16gb::StatusCode::kInvalidArgument,
        "--render-only and --json require a one-shot --message");
  }
  if ((!options.state_dump_path.empty() ||
       options.state_dump_position.has_value()) &&
      !options.has_one_shot_message) {
    return gem16gb::Status(
        gem16gb::StatusCode::kInvalidArgument,
        "state capture requires a one-shot --message");
  }
  if (options.max_tokens == 0U || options.max_context == 0U) {
    return gem16gb::Status(gem16gb::StatusCode::kInvalidArgument,
                          "token and context limits must be positive");
  }
  return options;
}

struct TurnOutput {
  std::string content;
  std::string display_text;
};

struct ConversationPromptState {
  std::vector<std::uint32_t> cached_prefix_token_ids;
  std::optional<std::uint32_t> pending_assistant_token_id;
};

struct TokenStreamContext {
  const gem16gb::GemmaChatProcessor* processor = nullptr;
  std::ostream* output = nullptr;
};

gem16gb::Status StreamGeneratedToken(void* opaque_context,
                                     std::uint32_t token_id) {
  auto* context = static_cast<TokenStreamContext*>(opaque_context);
  if (context == nullptr || context->processor == nullptr ||
      context->output == nullptr) {
    return gem16gb::Status(gem16gb::StatusCode::kInternal,
                           "chat token stream is not initialized");
  }
  auto status =
      context->processor->WriteDecodedToken(token_id, true, *context->output);
  if (!status.ok()) return status;
  context->output->flush();
  if (!*context->output) {
    return gem16gb::Status(gem16gb::StatusCode::kIoError,
                           "failed to flush chat token stream");
  }
  return gem16gb::Status::Ok();
}

gem16gb::Result<TurnOutput> RunTurn(
    const Options& cli, const gem16gb::GemmaChatProcessor& processor,
    std::vector<gem16gb::ChatMessage>& messages, bool write_json,
    bool stream_tokens, gem16gb::ConversationSession* session,
    ConversationPromptState* prompt_state) {
  std::optional<std::string> rendered;
  if (cli.render_only) {
    auto render_result = processor.Render(messages, cli.thinking);
    if (!render_result.ok()) return render_result.status();
    rendered = std::move(render_result).value();
  }
  gem16gb::Result<std::vector<std::uint32_t>> prompt_ids = [&]() {
    if (prompt_state == nullptr ||
        prompt_state->cached_prefix_token_ids.empty()) {
      return processor.Encode(messages, cli.thinking);
    }
    auto continuation =
        processor.EncodeContinuation(messages.back().content, cli.thinking);
    if (!continuation.ok()) {
      return gem16gb::Result<std::vector<std::uint32_t>>(
          continuation.status());
    }
    std::vector<std::uint32_t> token_ids =
        prompt_state->cached_prefix_token_ids;
    if (prompt_state->pending_assistant_token_id.has_value()) {
      token_ids.push_back(*prompt_state->pending_assistant_token_id);
    }
    token_ids.insert(token_ids.end(), continuation.value().begin(),
                     continuation.value().end());
    return gem16gb::Result<std::vector<std::uint32_t>>(
        std::move(token_ids));
  }();
  if (!prompt_ids.ok()) return prompt_ids.status();

  if (cli.render_only) {
    if (write_json) {
      std::cout << "{\"rendered_prompt\":" << JsonEscape(*rendered)
                << ",\"prompt_token_ids\":";
      WriteTokenIds(prompt_ids.value());
      std::cout << "}\n";
    } else {
      std::cout << *rendered << '\n';
    }
    return TurnOutput{};
  }

  gem16gb::GreedyInferenceOptions inference_options;
  inference_options.model_directory = cli.model_directory;
  inference_options.input_token_ids = std::move(prompt_ids).value();
  inference_options.stop_token_ids =
      processor.generation_controls().stop_token_ids;
  inference_options.suppressed_token_ids =
      processor.generation_controls().suppressed_token_ids;
  inference_options.max_generated_tokens = cli.max_tokens;
  inference_options.max_context_tokens = cli.max_context;
  inference_options.kv_cache_mode = cli.kv_cache_mode;
  inference_options.state_dump_path = cli.state_dump_path;
  inference_options.state_dump_position = cli.state_dump_position;
  TokenStreamContext stream_context{&processor, &std::cout};
  if (stream_tokens) {
    inference_options.generated_token_callback = StreamGeneratedToken;
    inference_options.generated_token_callback_context = &stream_context;
  }
  auto inference =
      session == nullptr
          ? gem16gb::RunGreedyInference(inference_options)
          : session->Generate(
                inference_options.input_token_ids,
                inference_options.max_generated_tokens,
                inference_options.generated_token_callback,
                inference_options.generated_token_callback_context);
  if (!inference.ok()) return inference.status();

  if (prompt_state != nullptr) {
    prompt_state->cached_prefix_token_ids =
        inference_options.input_token_ids;
    if (inference.value().output_token_ids.size() > 1U) {
      prompt_state->cached_prefix_token_ids.insert(
          prompt_state->cached_prefix_token_ids.end(),
          inference.value().output_token_ids.begin(),
          inference.value().output_token_ids.end() - 1);
    }
    prompt_state->pending_assistant_token_id.reset();
    if (!inference.value().stopped &&
        !inference.value().output_token_ids.empty()) {
      prompt_state->pending_assistant_token_id =
          inference.value().output_token_ids.back();
    }
  }

  std::vector<std::uint32_t> content_ids =
      inference.value().output_token_ids;
  if (!content_ids.empty() &&
      std::find(processor.generation_controls().stop_token_ids.begin(),
                processor.generation_controls().stop_token_ids.end(),
                content_ids.back()) !=
          processor.generation_controls().stop_token_ids.end()) {
    content_ids.pop_back();
  }
  auto assistant_content = processor.Decode(content_ids, false);
  if (!assistant_content.ok()) return assistant_content.status();
  auto assistant_text = processor.Decode(content_ids, true);
  if (!assistant_text.ok()) return assistant_text.status();

  if (write_json) {
    std::cout << "{\"assistant_content\":"
              << JsonEscape(assistant_content.value())
              << ",\"assistant_text\":" << JsonEscape(assistant_text.value())
              << ",\"prompt_token_ids\":";
    WriteTokenIds(inference_options.input_token_ids);
    std::cout << ",\"output_token_ids\":";
    WriteTokenIds(inference.value().output_token_ids);
    std::cout << ",\"finish_reason\":"
              << JsonEscape(inference.value().stopped ? "stop" : "length")
              << ",\"projection_path\":\"native_sm120\""
              << ",\"state_dumped\":"
              << (inference.value().state_dumped ? "true" : "false")
              << ",\"fused_gate_up\":false"
              << ",\"fused_prefill_attention\":true"
              << ",\"fused_output_head\":true"
              << ",\"decode_graphs\":"
              << (inference.value().decode_graphs ? "true" : "false")
              << ",\"kv_cache_mode\":"
              << JsonEscape(
                     inference.value().kv_cache_mode ==
                             gem16gb::KvCacheMode::kCheckpointFp8
                         ? "checkpoint_fp8"
                         : "bf16_correctness")
              << ",\"benchmark_qualified\":false}\n";
  }
  return TurnOutput{std::move(assistant_content).value(),
                    std::move(assistant_text).value()};
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 &&
      (std::string_view(argv[1]) == "--help" ||
       std::string_view(argv[1]) == "-h")) {
    PrintUsage();
    return 0;
  }
  auto parsed = ParseOptions(argc, argv);
  if (!parsed.ok()) {
    std::cerr << "error: " << parsed.status().message() << '\n';
    PrintUsage();
    return 64;
  }
  const Options options = std::move(parsed).value();
  auto processor =
      gem16gb::GemmaChatProcessor::Load(options.model_directory);
  if (!processor.ok()) {
    std::cerr << "error: " << processor.status().message() << '\n';
    return 2;
  }

  std::vector<gem16gb::ChatMessage> messages;
  if (options.has_system_message) {
    messages.push_back({"system", options.system_message});
  }
  if (options.has_one_shot_message) {
    messages.push_back({"user", options.one_shot_message});
    const bool stream_tokens = !options.json && !options.render_only;
    auto response = RunTurn(options, processor.value(), messages,
                            options.json, stream_tokens, nullptr, nullptr);
    if (!response.ok()) {
      std::cerr << "error: " << response.status().message() << '\n';
      return 2;
    }
    if (stream_tokens) std::cout << '\n';
    return 0;
  }

  gem16gb::ConversationSessionOptions session_options;
  session_options.model_directory = options.model_directory;
  session_options.stop_token_ids =
      processor.value().generation_controls().stop_token_ids;
  session_options.suppressed_token_ids =
      processor.value().generation_controls().suppressed_token_ids;
  session_options.max_context_tokens = options.max_context;
  session_options.kv_cache_mode = options.kv_cache_mode;
  auto session = gem16gb::ConversationSession::Create(session_options);
  if (!session.ok()) {
    std::cerr << "error: " << session.status().message() << '\n';
    return 2;
  }

  std::cout << "gem16gb resident chat session (/quit to exit)\n"
            << "Model weights and the exact conversation KV prefix stay resident.\n";
  ConversationPromptState prompt_state;
  prompt_state.cached_prefix_token_ids.reserve(
      static_cast<std::size_t>(options.max_context));
  while (true) {
    std::cout << "you> " << std::flush;
    std::string input;
    if (!std::getline(std::cin, input)) {
      std::cout << '\n';
      break;
    }
    if (input == "/quit" || input == "/exit") break;
    if (input.empty()) continue;
    messages.push_back({"user", input});
    std::cout << "model> " << std::flush;
    auto response = RunTurn(options, processor.value(), messages, false, true,
                            &session.value(), &prompt_state);
    if (!response.ok()) {
      messages.pop_back();
      std::cerr << "\nerror: " << response.status().message() << '\n';
      break;
    }
    std::cout << '\n';
    messages.push_back(
        {"assistant", std::move(response).value().content});
  }
  return 0;
}
