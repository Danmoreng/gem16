#include <algorithm>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#if defined(_WIN32)
#include "windows_utf8.h"
#endif

#include "gem16/chat.h"
#include "gem16/engine.h"
#include "gem16/tokenizer.h"

namespace {

bool ParseUnsigned(std::string_view text, std::uint64_t& value) {
  const auto result =
      std::from_chars(text.data(), text.data() + text.size(), value);
  return result.ec == std::errc{} && result.ptr == text.data() + text.size();
}

bool ParseFloat(std::string_view text, float& value) {
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
      << "  gem16-chat --model <checkpoint> [--max-tokens N] [--max-context N]\n"
      << "                [--assistant-model <official-mtp-checkpoint>]\n"
      << "                [--mtp-draft-tokens 1|2|4] [--mtp-adaptive]\n"
      << "                [--stats]\n"
      << "                [--thinking] [--system <text>]\n"
      << "                [--kv-cache fp8|bf16]\n"
      << "                [--greedy|--sample] [--temperature F] [--top-k N] [--top-p F]\n"
      << "                [--min-p F] [--repetition-penalty F] [--seed N]\n"
      << "                [--dump-state <path> --dump-state-position N]\n"
      << "  gem16-chat --model <checkpoint> --message <text> [--json]\n"
      << "  gem16-chat --model <checkpoint> --message <text> --render-only --json\n";
}

struct Options {
  std::filesystem::path model_directory;
  std::filesystem::path assistant_model_directory;
  std::string system_message;
  std::string one_shot_message;
  std::uint64_t max_tokens = 128;
  std::uint64_t max_context = 1024;
  bool has_system_message = false;
  bool has_one_shot_message = false;
  bool thinking = false;
  bool render_only = false;
  bool json = false;
  bool stats = false;
  std::filesystem::path state_dump_path;
  std::optional<std::uint64_t> state_dump_position;
  gem16::KvCacheMode kv_cache_mode =
      gem16::KvCacheMode::kCheckpointFp8;
  gem16::SamplingOptions sampling;
  bool greedy_explicit = false;
  bool temperature_set = false;
  bool top_p_set = false;
  bool min_p_set = false;
  bool repetition_penalty_set = false;
  bool top_k_set = false;
  bool seed_set = false;
  std::uint32_t mtp_draft_tokens = 0U;
  bool mtp_adaptive = false;
};

gem16::Result<Options> ParseOptions(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--model" && index + 1 < argc) {
      options.model_directory = argv[++index];
    } else if (argument == "--assistant-model" && index + 1 < argc) {
      options.assistant_model_directory = argv[++index];
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
    } else if (argument == "--system" && index + 1 < argc) {
      options.system_message = argv[++index];
      options.has_system_message = true;
    } else if (argument == "--message" && index + 1 < argc) {
      options.one_shot_message = argv[++index];
      options.has_one_shot_message = true;
    } else if (argument == "--max-tokens" && index + 1 < argc) {
      if (!ParseUnsigned(argv[++index], options.max_tokens)) {
        return gem16::Status(gem16::StatusCode::kInvalidArgument,
                              "--max-tokens must be an unsigned integer");
      }
    } else if (argument == "--max-context" && index + 1 < argc) {
      if (!ParseUnsigned(argv[++index], options.max_context)) {
        return gem16::Status(gem16::StatusCode::kInvalidArgument,
                              "--max-context must be an unsigned integer");
      }
    } else if (argument == "--thinking") {
      options.thinking = true;
    } else if (argument == "--render-only") {
      options.render_only = true;
    } else if (argument == "--json") {
      options.json = true;
    } else if (argument == "--stats") {
      options.stats = true;
    } else if (argument == "--dump-state" && index + 1 < argc) {
      options.state_dump_path = argv[++index];
    } else if (argument == "--dump-state-position" && index + 1 < argc) {
      std::uint64_t position = 0;
      if (!ParseUnsigned(argv[++index], position)) {
        return gem16::Status(
            gem16::StatusCode::kInvalidArgument,
            "--dump-state-position must be an unsigned integer");
      }
      options.state_dump_position = position;
    } else if (argument == "--greedy") {
      options.greedy_explicit = true;
      options.sampling = {};
    } else if (argument == "--sample") {
      options.sampling.enabled = true;
    } else if (argument == "--temperature" && index + 1 < argc) {
      options.temperature_set = true;
      options.sampling.enabled = true;
      if (!ParseFloat(argv[++index], options.sampling.temperature)) {
        return gem16::Status(gem16::StatusCode::kInvalidArgument,
                             "--temperature must be a number");
      }
    } else if (argument == "--top-p" && index + 1 < argc) {
      options.top_p_set = true;
      options.sampling.enabled = true;
      if (!ParseFloat(argv[++index], options.sampling.top_p)) {
        return gem16::Status(gem16::StatusCode::kInvalidArgument,
                             "--top-p must be a number");
      }
    } else if (argument == "--min-p" && index + 1 < argc) {
      options.min_p_set = true;
      options.sampling.enabled = true;
      if (!ParseFloat(argv[++index], options.sampling.min_p)) {
        return gem16::Status(gem16::StatusCode::kInvalidArgument,
                             "--min-p must be a number");
      }
    } else if (argument == "--repetition-penalty" && index + 1 < argc) {
      options.repetition_penalty_set = true;
      options.sampling.enabled = true;
      if (!ParseFloat(argv[++index], options.sampling.repetition_penalty)) {
        return gem16::Status(gem16::StatusCode::kInvalidArgument,
                             "--repetition-penalty must be a number");
      }
    } else if (argument == "--top-k" && index + 1 < argc) {
      std::uint64_t value = 0U;
      options.top_k_set = true;
      options.sampling.enabled = true;
      if (!ParseUnsigned(argv[++index], value) ||
          value > std::numeric_limits<std::uint32_t>::max()) {
        return gem16::Status(gem16::StatusCode::kInvalidArgument,
                             "--top-k must be an unsigned 32-bit integer");
      }
      options.sampling.top_k = static_cast<std::uint32_t>(value);
    } else if (argument == "--seed" && index + 1 < argc) {
      options.seed_set = true;
      options.sampling.enabled = true;
      if (!ParseUnsigned(argv[++index], options.sampling.seed)) {
        return gem16::Status(gem16::StatusCode::kInvalidArgument,
                             "--seed must be an unsigned integer");
      }
    } else if (argument == "--kv-cache" && index + 1 < argc) {
      const std::string_view mode = argv[++index];
      if (mode == "fp8") {
        options.kv_cache_mode = gem16::KvCacheMode::kCheckpointFp8;
      } else if (mode == "bf16") {
        options.kv_cache_mode = gem16::KvCacheMode::kBf16Correctness;
      } else {
        return gem16::Status(gem16::StatusCode::kInvalidArgument,
                              "--kv-cache must be fp8 or bf16");
      }
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
  if ((options.render_only || options.json) &&
      !options.has_one_shot_message) {
    return gem16::Status(
        gem16::StatusCode::kInvalidArgument,
        "--render-only and --json require a one-shot --message");
  }
  if ((!options.state_dump_path.empty() ||
       options.state_dump_position.has_value()) &&
      !options.has_one_shot_message) {
    return gem16::Status(
        gem16::StatusCode::kInvalidArgument,
        "state capture requires a one-shot --message");
  }
  if (options.greedy_explicit && options.sampling.enabled) {
    return gem16::Status(
        gem16::StatusCode::kInvalidArgument,
        "--greedy cannot be combined with sampling options");
  }
  if (options.max_tokens == 0U || options.max_context == 0U) {
    return gem16::Status(gem16::StatusCode::kInvalidArgument,
                          "token and context limits must be positive");
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

gem16::ChatSessionOptions MakeChatSessionOptions(const Options& options) {
  gem16::ChatSessionOptions session_options;
  session_options.model_directory = options.model_directory;
  session_options.assistant_model_directory = options.assistant_model_directory;
  session_options.max_context_tokens = options.max_context;
  session_options.kv_cache_mode = options.kv_cache_mode;
  session_options.sampling = options.sampling;
  session_options.mtp_draft_tokens = options.mtp_draft_tokens;
  session_options.mtp_adaptive = options.mtp_adaptive;
  return session_options;
}

struct TurnOutput {
  std::string content;
  std::string display_text;
};

struct TokenStreamContext {
  const gem16::GemmaChatProcessor* processor = nullptr;
  std::ostream* output = nullptr;
};

gem16::Status StreamGeneratedToken(void* opaque_context,
                                    std::uint32_t token_id) {
  auto* context = static_cast<TokenStreamContext*>(opaque_context);
  if (context == nullptr || context->processor == nullptr ||
      context->output == nullptr) {
    return gem16::Status(gem16::StatusCode::kInternal,
                           "chat token stream is not initialized");
  }
  auto status =
      context->processor->WriteDecodedToken(token_id, true, *context->output);
  if (!status.ok()) return status;
  context->output->flush();
  if (!*context->output) {
    return gem16::Status(gem16::StatusCode::kIoError,
                           "failed to flush chat token stream");
  }
  return gem16::Status::Ok();
}

gem16::Status StreamGenerationEvent(
    void* opaque_context, const gem16::GenerationEvent& event) {
  if (event.kind != gem16::GenerationEventKind::kToken) {
    return gem16::Status(gem16::StatusCode::kUnsupported,
                         "chat CLI received an unsupported generation event");
  }
  return StreamGeneratedToken(opaque_context, event.token_id);
}

void PrintTurnStats(const gem16::GreedyInferenceResult& inference) {
  std::cerr << "\n[stats] decode " << std::fixed << std::setprecision(3)
            << inference.decode_tokens_per_second << " tok/s";
  if (inference.mtp_enabled) {
    std::cerr << ", MTP D" << inference.mtp_draft_tokens
              << ", proposed " << inference.mtp_proposed_tokens
              << ", accepted " << inference.mtp_accepted_tokens
              << ", rejected " << inference.mtp_rejected_tokens
              << ", groups " << inference.mtp_verification_groups
              << ", GPU chained "
              << (inference.mtp_gpu_chained ? "yes" : "no");
  } else {
    std::cerr << ", MTP disabled";
  }
  std::cerr << '\n';
}

gem16::Result<TurnOutput> RunTurn(
    const Options& cli, const gem16::GemmaChatProcessor& processor,
    std::vector<gem16::ChatMessage>& messages, bool write_json,
    bool stream_tokens, gem16::ChatSession* session) {
  std::optional<std::string> rendered;
  if (cli.render_only) {
    auto render_result = processor.Render(messages, cli.thinking);
    if (!render_result.ok()) return render_result.status();
    rendered = std::move(render_result).value();
  }
  if (session != nullptr) {
    gem16::ChatGenerationRequest request;
    request.max_generated_tokens = cli.max_tokens;
    request.enable_thinking = cli.thinking;
    request.messages.reserve(messages.size());
    for (const gem16::ChatMessage& message : messages) {
      request.messages.push_back(
          gem16::GenerationMessage::Text(message.role, message.content));
    }
    TokenStreamContext stream_context{&processor, &std::cout};
    auto generated = session->Generate(
        request, stream_tokens ? StreamGenerationEvent : nullptr,
        stream_tokens ? &stream_context : nullptr);
    if (!generated.ok()) return generated.status();
    if (cli.stats && !write_json) PrintTurnStats(generated.value().inference);
    return TurnOutput{std::move(generated.value().assistant_content),
                      std::move(generated.value().assistant_text)};
  }

  auto prompt_ids = processor.Encode(messages, cli.thinking);
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

  gem16::GreedyInferenceOptions inference_options;
  inference_options.model_directory = cli.model_directory;
  inference_options.assistant_model_directory = cli.assistant_model_directory;
  inference_options.input_token_ids = std::move(prompt_ids).value();
  inference_options.stop_token_ids =
      processor.generation_controls().stop_token_ids;
  inference_options.suppressed_token_ids =
      processor.generation_controls().suppressed_token_ids;
  inference_options.max_generated_tokens = cli.max_tokens;
  inference_options.max_context_tokens = cli.max_context;
  inference_options.kv_cache_mode = cli.kv_cache_mode;
  inference_options.sampling = cli.sampling;
  inference_options.mtp_draft_tokens = cli.mtp_draft_tokens;
  inference_options.mtp_adaptive = cli.mtp_adaptive;
  inference_options.state_dump_path = cli.state_dump_path;
  inference_options.state_dump_position = cli.state_dump_position;
  TokenStreamContext stream_context{&processor, &std::cout};
  if (stream_tokens) {
    inference_options.generated_token_callback = StreamGeneratedToken;
    inference_options.generated_token_callback_context = &stream_context;
  }
  auto inference = gem16::RunGreedyInference(inference_options);
  if (!inference.ok()) return inference.status();

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
  auto assistant_text = processor.DecodeResponseText(content_ids);
  if (!assistant_text.ok()) return assistant_text.status();

  if (cli.stats && !write_json) PrintTurnStats(inference.value());

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
              << ",\"mtp\":{\"enabled\":"
              << (inference.value().mtp_enabled ? "true" : "false")
              << ",\"draft_tokens\":"
              << inference.value().mtp_draft_tokens
              << ",\"adaptive\":"
              << (inference.value().mtp_adaptive ? "true" : "false")
              << ",\"gpu_chained\":"
              << (inference.value().mtp_gpu_chained ? "true" : "false")
              << ",\"proposed_tokens\":"
              << inference.value().mtp_proposed_tokens
              << ",\"accepted_tokens\":"
              << inference.value().mtp_accepted_tokens
              << ",\"rejected_tokens\":"
              << inference.value().mtp_rejected_tokens
              << ",\"verification_groups\":"
              << inference.value().mtp_verification_groups
              << ",\"ordinary_fallback_tokens\":"
              << inference.value().mtp_ordinary_fallback_tokens << '}'
              << ",\"decoding_mode\":"
              << JsonEscape(inference.value().sampling.enabled ? "sampled"
                                                                : "greedy")
              << ",\"sampling\":{\"enabled\":"
              << (inference.value().sampling.enabled ? "true" : "false")
              << ",\"temperature\":"
              << inference.value().sampling.temperature
              << ",\"top_k\":" << inference.value().sampling.top_k
              << ",\"top_p\":" << inference.value().sampling.top_p
              << ",\"min_p\":" << inference.value().sampling.min_p
              << ",\"repetition_penalty\":"
              << inference.value().sampling.repetition_penalty
              << ",\"seed\":" << inference.value().sampling.seed << '}'
              << ",\"kv_cache_mode\":"
              << JsonEscape(
                     inference.value().kv_cache_mode ==
                             gem16::KvCacheMode::kCheckpointFp8
                         ? "checkpoint_fp8"
                         : "bf16_correctness")
              << ",\"benchmark_qualified\":false}\n";
  }
  return TurnOutput{std::move(assistant_content).value(),
                    std::move(assistant_text).value()};
}

}  // namespace

int ChatMain(int argc, char** argv) {
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
  Options options = std::move(parsed).value();
  auto processor =
      gem16::GemmaChatProcessor::Load(options.model_directory);
  if (!processor.ok()) {
    std::cerr << "error: " << processor.status().message() << '\n';
    return 2;
  }
  if (!options.greedy_explicit) {
    const gem16::SamplingOptions overrides = options.sampling;
    options.sampling =
        processor.value().generation_controls().recommended_sampling;
    if (options.temperature_set) {
      options.sampling.temperature = overrides.temperature;
    }
    if (options.top_p_set) options.sampling.top_p = overrides.top_p;
    if (options.min_p_set) options.sampling.min_p = overrides.min_p;
    if (options.repetition_penalty_set) {
      options.sampling.repetition_penalty = overrides.repetition_penalty;
    }
    if (options.top_k_set) options.sampling.top_k = overrides.top_k;
    if (options.seed_set) options.sampling.seed = overrides.seed;
  }

  std::vector<gem16::ChatMessage> messages;
  if (options.has_system_message) {
    messages.push_back({"system", options.system_message});
  }
  if (options.has_one_shot_message) {
    messages.push_back({"user", options.one_shot_message});
    const bool stream_tokens = !options.json && !options.render_only;
    const bool diagnostic_path = options.json ||
                                 !options.state_dump_path.empty() ||
                                 options.state_dump_position.has_value();
    if (options.render_only || diagnostic_path) {
      auto response = RunTurn(options, processor.value(), messages,
                              options.json, stream_tokens, nullptr);
      if (!response.ok()) {
        std::cerr << "error: " << response.status().message() << '\n';
        return 2;
      }
    } else {
      auto session = gem16::ChatSession::Create(
          MakeChatSessionOptions(options), processor.value());
      if (!session.ok()) {
        std::cerr << "error: " << session.status().message() << '\n';
        return 2;
      }
      auto response = RunTurn(options, processor.value(), messages,
                              options.json, stream_tokens, &session.value());
      if (!response.ok()) {
        std::cerr << "error: " << response.status().message() << '\n';
        return 2;
      }
    }
    if (stream_tokens) std::cout << '\n';
    return 0;
  }

  auto session = gem16::ChatSession::Create(
      MakeChatSessionOptions(options), processor.value());
  if (!session.ok()) {
    std::cerr << "error: " << session.status().message() << '\n';
    return 2;
  }

  std::cout << "gem16 resident chat session (/quit to exit)\n"
            << "Model weights and the exact conversation KV prefix stay resident.\n";
  if (options.mtp_draft_tokens != 0U) {
    std::cout << "MTP enabled: D" << options.mtp_draft_tokens
              << (options.mtp_adaptive ? " adaptive" : " fixed")
              << ", exact target verification, "
              << (options.sampling.enabled ? "sampled" : "greedy")
              << " decoding.\n";
  }
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
                            &session.value());
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

#if defined(_WIN32)

namespace {

class ConsoleUtf8Scope {
 public:
  ConsoleUtf8Scope()
      : original_input_code_page_(GetConsoleCP()),
        original_output_code_page_(GetConsoleOutputCP()) {
    if (original_input_code_page_ != 0U) {
      input_changed_ = SetConsoleCP(CP_UTF8) != 0;
    }
    if (original_output_code_page_ != 0U) {
      output_changed_ = SetConsoleOutputCP(CP_UTF8) != 0;
    }
  }

  ~ConsoleUtf8Scope() {
    if (input_changed_) (void)SetConsoleCP(original_input_code_page_);
    if (output_changed_) (void)SetConsoleOutputCP(original_output_code_page_);
  }

  ConsoleUtf8Scope(const ConsoleUtf8Scope&) = delete;
  ConsoleUtf8Scope& operator=(const ConsoleUtf8Scope&) = delete;

 private:
  UINT original_input_code_page_ = 0U;
  UINT original_output_code_page_ = 0U;
  bool input_changed_ = false;
  bool output_changed_ = false;
};

}  // namespace

int wmain(int argc, wchar_t** wide_argv) {
  ConsoleUtf8Scope console_utf8;
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
  return ChatMain(argc, arguments.data());
}

#else

int main(int argc, char** argv) { return ChatMain(argc, argv); }

#endif
