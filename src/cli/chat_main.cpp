#include <algorithm>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
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
#include "model/config.h"
#include "model/model_variant.h"

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
      << "                [--thinking-budget off|small|medium|high] [--thinking|--no-thinking]\n"
      << "                [--show-thinking|--hide-thinking] [--system <text>]\n"
      << "                [--tool <name> <description> <parameters-schema.json>]...\n"
      << "                [--kv-cache fp8|bf16]\n"
      << "                [--greedy|--sample] [--temperature F] [--top-k N] [--top-p F]\n"
      << "                [--min-p F] [--repetition-penalty F] [--seed N]\n"
      << "                [--dump-state <path> --dump-state-position N]\n"
      << "                [--print-model-report]\n"
      << "  gem16-chat --model <checkpoint> --message <text> [--json]\n"
      << "  gem16-chat --model <checkpoint> --message <text> [--audio <file>|--image <file>]...\n"
      << "  gem16-chat --model <checkpoint> --message <text> --render-only --json\n";
}

enum class MediaFileKind { kAudio, kImage };

struct MediaFile {
  MediaFileKind kind = MediaFileKind::kAudio;
  std::filesystem::path path;
};

struct Options {
  std::filesystem::path model_directory;
  std::filesystem::path assistant_model_directory;
  std::string system_message;
  std::string one_shot_message;
  std::vector<MediaFile> media_files;
  std::vector<gem16::GenerationContentPart> media_parts;
  std::optional<std::uint64_t> max_tokens;
  std::uint64_t max_context = 1024;
  bool max_context_explicit = false;
  bool has_system_message = false;
  bool has_one_shot_message = false;
  gem16::ThinkingEffort thinking_effort = gem16::ThinkingEffort::kMedium;
  bool show_thinking = true;
  bool render_only = false;
  bool json = false;
  bool stats = false;
  bool print_model_report = false;
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
  std::vector<gem16::GenerationToolDefinition> tools;
};

gem16::Result<std::string> ReadToolSchema(const std::filesystem::path& path) {
  std::error_code error;
  const std::uintmax_t size = std::filesystem::file_size(path, error);
  if (error || size > 1024U * 1024U) {
    return gem16::Status(gem16::StatusCode::kIoError,
                         "cannot read tool schema or it exceeds 1 MiB: " + path.string());
  }
  std::string text(static_cast<std::size_t>(size), '\0');
  std::ifstream input(path, std::ios::binary);
  input.read(text.data(), static_cast<std::streamsize>(text.size()));
  if (!input || input.gcount() != static_cast<std::streamsize>(text.size())) {
    return gem16::Status(gem16::StatusCode::kIoError,
                         "cannot read tool schema: " + path.string());
  }
  return text;
}

gem16::Result<std::vector<gem16::GenerationContentPart>> LoadMediaParts(
    std::span<const MediaFile> files, std::uint64_t context_tokens,
    std::optional<std::uint64_t> max_tokens, bool stats) {
  std::vector<gem16::GenerationContentPart> parts(files.size());
  std::uint64_t audio_tokens = 0U;
  std::size_t image_count = 0U;
  for (std::size_t index = 0U; index < files.size(); ++index) {
    if (files[index].kind == MediaFileKind::kAudio) {
      auto audio = gem16::LoadAudioFile(files[index].path);
      if (!audio.ok()) return audio.status();
      audio_tokens += (audio.value().samples.size() + 639U) / 640U;
      parts[index] = gem16::GenerationContentPart::Audio(
          std::move(audio).value());
    } else {
      ++image_count;
    }
  }
  const std::uint64_t output_reserve = max_tokens.value_or(
      std::min<std::uint64_t>(128U, context_tokens / 4U));
  const std::uint64_t fixed_reserve =
      output_reserve + audio_tokens + 64U + files.size() * 2U;
  const std::uint32_t image_budget =
      gem16::AutomaticVisionSoftTokenBudget(
          context_tokens, fixed_reserve, image_count);
  for (std::size_t index = 0U; index < files.size(); ++index) {
    if (files[index].kind != MediaFileKind::kImage) continue;
    auto image = gem16::LoadVisionImage(
        files[index].path, gem16::VisionImageOptions{image_budget, false});
    if (!image.ok()) return image.status();
    if (stats) {
      std::cerr << "[media] image " << image.value().source_width << 'x'
                << image.value().source_height << " -> "
                << image.value().processed_width << 'x'
                << image.value().processed_height << ", "
                << image.value().patch_count << '/' << image_budget
                << " soft tokens\n";
    }
    parts[index] = gem16::GenerationContentPart::Image(
        std::move(image).value());
  }
  return parts;
}

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
    } else if (argument == "--tool" && index + 3 < argc) {
      gem16::GenerationToolDefinition tool;
      tool.name = argv[++index];
      tool.description = argv[++index];
      auto schema = ReadToolSchema(argv[++index]);
      if (!schema.ok()) return schema.status();
      tool.parameters_json = std::move(schema).value();
      options.tools.push_back(std::move(tool));
    } else if (argument == "--message" && index + 1 < argc) {
      options.one_shot_message = argv[++index];
      options.has_one_shot_message = true;
    } else if (argument == "--audio" && index + 1 < argc) {
      options.media_files.push_back(
          {MediaFileKind::kAudio, std::filesystem::path(argv[++index])});
    } else if (argument == "--image" && index + 1 < argc) {
      options.media_files.push_back(
          {MediaFileKind::kImage, std::filesystem::path(argv[++index])});
    } else if (argument == "--max-tokens" && index + 1 < argc) {
      std::uint64_t value = 0U;
      if (!ParseUnsigned(argv[++index], value)) {
        return gem16::Status(gem16::StatusCode::kInvalidArgument,
                             "--max-tokens must be an unsigned integer");
      }
      options.max_tokens = value;
    } else if (argument == "--max-context" && index + 1 < argc) {
      if (!ParseUnsigned(argv[++index], options.max_context)) {
        return gem16::Status(gem16::StatusCode::kInvalidArgument,
                              "--max-context must be an unsigned integer");
      }
      options.max_context_explicit = true;
    } else if (argument == "--thinking") {
      options.thinking_effort = gem16::ThinkingEffort::kMedium;
    } else if (argument == "--no-thinking") {
      options.thinking_effort = gem16::ThinkingEffort::kOff;
    } else if (argument == "--show-thinking") {
      options.show_thinking = true;
    } else if (argument == "--hide-thinking") {
      options.show_thinking = false;
    } else if (argument == "--thinking-budget" && index + 1 < argc) {
      const std::string_view effort = argv[++index];
      if (effort == "off") {
        options.thinking_effort = gem16::ThinkingEffort::kOff;
      } else if (effort == "small") {
        options.thinking_effort = gem16::ThinkingEffort::kSmall;
      } else if (effort == "medium") {
        options.thinking_effort = gem16::ThinkingEffort::kMedium;
      } else if (effort == "high") {
        options.thinking_effort = gem16::ThinkingEffort::kHigh;
      } else {
        return gem16::Status(
            gem16::StatusCode::kInvalidArgument,
            "--thinking-budget must be off, small, medium, or high");
      }
    } else if (argument == "--render-only") {
      options.render_only = true;
    } else if (argument == "--json") {
      options.json = true;
    } else if (argument == "--stats") {
      options.stats = true;
    } else if (argument == "--print-model-report") {
      options.print_model_report = true;
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
      !options.print_model_report &&
      !options.has_one_shot_message) {
    return gem16::Status(
        gem16::StatusCode::kInvalidArgument,
        "--render-only and --json require a one-shot --message");
  }
  if (!options.media_files.empty() && !options.has_one_shot_message) {
    return gem16::Status(gem16::StatusCode::kInvalidArgument,
                         "--audio and --image require a one-shot --message");
  }
  if (!options.media_files.empty() &&
      (options.render_only || options.json ||
       !options.state_dump_path.empty() ||
       options.state_dump_position.has_value())) {
    return gem16::Status(
        gem16::StatusCode::kUnsupported,
        "media currently use the live session path and cannot be combined with diagnostic output");
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
  if ((options.max_tokens.has_value() && *options.max_tokens == 0U) ||
      options.max_context == 0U) {
    return gem16::Status(gem16::StatusCode::kInvalidArgument,
                         "specified token and context limits must be positive");
  }
  if (options.mtp_adaptive && options.mtp_draft_tokens == 0U) {
    return gem16::Status(gem16::StatusCode::kInvalidArgument,
                         "--mtp-adaptive requires active MTP");
  }
  if (options.print_model_report &&
      (options.has_one_shot_message || options.render_only ||
       !options.media_files.empty() || !options.state_dump_path.empty() ||
       options.state_dump_position.has_value())) {
    return gem16::Status(
        gem16::StatusCode::kInvalidArgument,
        "--print-model-report cannot be combined with generation or media options");
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
  std::vector<gem16::GenerationToolCall> tool_calls;
};

gem16::ChatMessage TextMessage(std::string role, std::string content) {
  gem16::ChatMessage message;
  message.role = std::move(role);
  message.content = std::move(content);
  return message;
}

std::vector<gem16::ChatToolDefinition> MakeChatTools(const Options& options) {
  std::vector<gem16::ChatToolDefinition> tools;
  tools.reserve(options.tools.size());
  for (const auto& tool : options.tools) {
    tools.push_back({tool.name, tool.description, tool.parameters_json});
  }
  return tools;
}

struct TokenStreamContext {
  TokenStreamContext(const gem16::GemmaChatProcessor& chat_processor,
                     std::ostream& token_output, bool show_reasoning)
      : processor(&chat_processor),
        output(&token_output),
        channel_tracker(chat_processor.generation_controls()),
        show_thinking(show_reasoning) {}

  const gem16::GemmaChatProcessor* processor;
  std::ostream* output;
  gem16::ResponseChannelTracker channel_tracker;
  bool show_thinking = true;
  bool section_started = false;
  bool thinking_header_written = false;
  bool answer_header_written = false;
};

void BeginStreamSection(TokenStreamContext& context, std::string_view name) {
  *context.output << (context.section_started ? "\n\n" : "\n")
                  << "--- " << name << " ---\n";
  context.section_started = true;
}

gem16::Status StreamGeneratedToken(void* opaque_context,
                                    std::uint32_t token_id) {
  auto* context = static_cast<TokenStreamContext*>(opaque_context);
  if (context == nullptr || context->processor == nullptr ||
      context->output == nullptr) {
    return gem16::Status(gem16::StatusCode::kInternal,
                           "chat token stream is not initialized");
  }
  const bool was_reasoning = context->channel_tracker.in_reasoning();
  const gem16::ResponseTokenChannel channel =
      context->channel_tracker.Observe(token_id);
  const bool is_reasoning = context->channel_tracker.in_reasoning();

  if (!was_reasoning && is_reasoning && context->show_thinking &&
      !context->thinking_header_written) {
    BeginStreamSection(*context, "thinking");
    context->thinking_header_written = true;
  }
  if (was_reasoning && !is_reasoning && !context->answer_header_written) {
    BeginStreamSection(*context, "answer");
    context->answer_header_written = true;
  }

  if (channel == gem16::ResponseTokenChannel::kText &&
      !context->answer_header_written) {
    BeginStreamSection(*context, "answer");
    context->answer_header_written = true;
  }

  gem16::Status status = gem16::Status::Ok();
  if (channel == gem16::ResponseTokenChannel::kText ||
      (channel == gem16::ResponseTokenChannel::kReasoning &&
       context->show_thinking)) {
    status = context->processor->WriteDecodedToken(token_id, true,
                                                   *context->output);
  }
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
  if (inference.reasoning_enabled) {
    std::cerr << ", thinking " << inference.reasoning_tokens << '/'
              << inference.reasoning_budget_tokens
              << (inference.reasoning_budget_forced ? " forced" : " natural")
              << ", thinking ordinary target tokens "
              << inference.reasoning_ordinary_target_tokens;
  }
  std::cerr << '\n';
}

gem16::Result<TurnOutput> RunTurn(
    const Options& cli, const gem16::GemmaChatProcessor& processor,
    std::vector<gem16::ChatMessage>& messages, bool write_json,
    bool stream_tokens, gem16::ChatSession* session,
    std::span<const std::vector<gem16::GenerationContentPart>> message_media = {}) {
  std::optional<std::string> rendered;
  const std::vector<gem16::ChatToolDefinition> chat_tools = MakeChatTools(cli);
  if (cli.render_only) {
    auto render_result = processor.Render(
        messages, cli.thinking_effort != gem16::ThinkingEffort::kOff, true,
        chat_tools);
    if (!render_result.ok()) return render_result.status();
    rendered = std::move(render_result).value();
  }
  if (session != nullptr) {
    if (!message_media.empty() && message_media.size() != messages.size()) {
      return gem16::Status(
          gem16::StatusCode::kInvalidArgument,
          "resident message/media history size mismatch");
    }
    gem16::ChatGenerationRequest request;
    request.max_generated_tokens = cli.max_tokens;
    request.thinking.effort = cli.thinking_effort;
    request.tools = cli.tools;
    request.messages.reserve(messages.size());
    for (std::size_t message_index = 0U;
         message_index < messages.size(); ++message_index) {
      const gem16::ChatMessage& message = messages[message_index];
      gem16::GenerationMessage converted;
      converted.role = message.role;
      if (message.role == "tool") {
        converted.content.push_back(gem16::GenerationContentPart::ToolResult(
            {message.tool_call_id, message.content}));
      } else {
        if (!message.content.empty()) {
          converted.content.push_back(
              gem16::GenerationContentPart::Text(message.content));
        }
        for (const auto& call : message.tool_calls) {
          converted.content.push_back(gem16::GenerationContentPart::ToolCall(
              {call.id, call.name, call.arguments_json}));
        }
        if (!message_media.empty()) {
          for (const auto& media : message_media[message_index]) {
            converted.content.push_back(media);
          }
        }
      }
      request.messages.push_back(std::move(converted));
    }
    TokenStreamContext stream_context(processor, std::cout,
                                      cli.show_thinking);
    auto generated = session->Generate(
        request, stream_tokens ? StreamGenerationEvent : nullptr,
        stream_tokens ? &stream_context : nullptr);
    if (!generated.ok()) return generated.status();
    if (cli.stats && !write_json) PrintTurnStats(generated.value().inference);
    return TurnOutput{std::move(generated.value().assistant_content),
                      std::move(generated.value().assistant_text),
                      std::move(generated.value().tool_calls)};
  }

  auto prompt_ids = processor.Encode(
      messages, cli.thinking_effort != gem16::ThinkingEffort::kOff, true,
      chat_tools);
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
  if (inference_options.input_token_ids.size() > cli.max_context) {
    return gem16::Status(gem16::StatusCode::kInvalidArgument,
                         "conversation prompt exceeds --max-context");
  }
  inference_options.max_generated_tokens = cli.max_tokens.value_or(
      cli.max_context - inference_options.input_token_ids.size() + 1U);
  inference_options.max_context_tokens = cli.max_context;
  inference_options.kv_cache_mode = cli.kv_cache_mode;
  inference_options.sampling = cli.sampling;
  inference_options.mtp_draft_tokens = cli.mtp_draft_tokens;
  inference_options.mtp_adaptive = cli.mtp_adaptive;
  inference_options.state_dump_path = cli.state_dump_path;
  inference_options.state_dump_position = cli.state_dump_position;
  TokenStreamContext stream_context(processor, std::cout, cli.show_thinking);
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
  TurnOutput output;
  output.content = std::move(assistant_content).value();
  output.display_text = std::move(assistant_text).value();
  return output;
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
  auto config = gem16::internal::LoadModelConfig(
      options.model_directory / "config.json");
  if (!config.ok()) {
    std::cerr << "error: " << config.status().message() << '\n';
    return 2;
  }
  const bool moe26b = gem16::internal::ClassifyModelVariant(config.value()) ==
                      gem16::internal::ModelVariant::kGemma4Moe26BA4B;
  if (!options.max_context_explicit && moe26b) {
    options.max_context = 32768U;
  }
  if (options.mtp_draft_tokens != 0U &&
      options.assistant_model_directory.empty()) {
    std::cerr << "error: active MTP requires --assistant-model\n";
    return 2;
  }
  if (moe26b && !options.media_files.empty()) {
    std::cerr << "error: Gemma 4 26B is a text-only profile; audio and vision input are unsupported\n";
    return 2;
  }
  if (options.print_model_report) {
    auto runtime = gem16::ModelRuntime::Load(
        {options.model_directory, options.assistant_model_directory,
         options.max_context, 0});
    if (!runtime.ok()) {
      std::cerr << "error: " << runtime.status().message() << '\n';
      return 2;
    }
    auto memory = gem16::QueryDeviceMemoryInfo();
    if (!memory.ok()) {
      std::cerr << "error: " << memory.status().message() << '\n';
      return 2;
    }
    const std::uint64_t admission_margin =
        moe26b && options.max_context >= 65536U
            ? 400U * 1024U * 1024U
            : 700U * 1024U * 1024U;
    std::cout
        << "{\"schema_version\":1,\"model_variant\":"
        << JsonEscape(runtime.value()->model_variant_name())
        << ",\"artifact_profile\":"
        << JsonEscape(runtime.value()->artifact_profile())
        << ",\"head_format\":"
        << JsonEscape(runtime.value()->head_format())
        << ",\"artifact_content_sha256\":"
        << JsonEscape(runtime.value()->artifact_content_sha256())
        << ",\"source_lock_sha256\":"
        << JsonEscape(runtime.value()->source_lock_sha256())
        << ",\"compiler_commit\":"
        << JsonEscape(runtime.value()->compiler_commit())
        << ",\"native_path\":"
        << JsonEscape(runtime.value()->selected_native_path())
        << ",\"text_only\":"
        << ((!runtime.value()->supports_audio() &&
             !runtime.value()->supports_vision())
                ? "true"
                : "false")
        << ",\"supports_mtp\":"
        << (runtime.value()->supports_mtp() ? "true" : "false")
        << ",\"resident_weight_bytes\":"
        << runtime.value()->weight_bytes()
        << ",\"kv_cache_bytes\":";
    if (moe26b) {
      std::cout << runtime.value()->kv_cache_bytes();
    } else {
      std::cout << "null";
    }
    std::cout << ",\"workspace_bytes\":";
    if (moe26b) {
      std::cout << runtime.value()->workspace_bytes();
    } else {
      std::cout << "null";
    }
    std::cout
        << ",\"configured_context_tokens\":"
        << runtime.value()->max_context_tokens()
        << ",\"default_context\":"
        << runtime.value()->default_context_tokens()
        << ",\"default_context_tokens\":"
        << runtime.value()->default_context_tokens()
        << ",\"qualified_64k\":";
    if (moe26b) {
      std::cout << (runtime.value()->qualified_64k() ? "true" : "false");
    } else {
      std::cout << "null";
    }
    std::cout << ",\"base_max_context\":";
    if (moe26b) {
      std::cout << runtime.value()->base_max_context_tokens();
    } else {
      std::cout << "null";
    }
    std::cout << ",\"mtp_max_context\":";
    if (runtime.value()->supports_mtp()) {
      std::cout << runtime.value()->max_context_tokens();
    } else {
      std::cout << "null";
    }
    std::cout
        << ",\"admission_free_bytes\":" << memory.value().free_bytes
        << ",\"required_admission_margin_bytes\":" << admission_margin
        << ",\"admission_headroom_bytes\":"
        << (memory.value().free_bytes > admission_margin
                ? memory.value().free_bytes - admission_margin
                : 0U)
        << "}\n";
    return 0;
  }
  auto initial_media = LoadMediaParts(
      options.media_files, options.max_context, options.max_tokens,
      options.stats);
  if (!initial_media.ok()) {
    std::cerr << "error: " << initial_media.status().message() << '\n';
    return 2;
  }
  options.media_parts = std::move(initial_media).value();
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
  std::vector<std::vector<gem16::GenerationContentPart>> message_media;
  if (options.has_system_message) {
    messages.push_back(TextMessage("system", options.system_message));
    message_media.emplace_back();
  }
  if (options.has_one_shot_message) {
    messages.push_back(TextMessage("user", options.one_shot_message));
    message_media.push_back(options.media_parts);
    const bool stream_tokens = !options.json && !options.render_only;
    const bool diagnostic_path = options.json ||
                                 !options.state_dump_path.empty() ||
                                 options.state_dump_position.has_value();
    if (options.render_only || diagnostic_path) {
      auto response = RunTurn(options, processor.value(), messages,
                              options.json, stream_tokens, nullptr,
                              message_media);
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
                              options.json, stream_tokens, &session.value(),
                              message_media);
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

  std::cout << "gem16 resident chat session (/quit to exit)\n";
  if (!moe26b) {
    std::cout << "Media commands: /image <path>, /audio <path>, /media, "
                 "/clear-media.\n";
  } else {
    std::cout << "Gemma 4 26B profile: text-only; vision and audio are unsupported.\n";
  }
  std::cout << "Model weights and the exact conversation KV prefix stay resident.\n";
  if (options.mtp_draft_tokens != 0U) {
    std::cout << "MTP enabled: D" << options.mtp_draft_tokens
              << (options.mtp_adaptive ? " adaptive" : " fixed")
              << ", exact target verification, "
              << (options.sampling.enabled ? "sampled" : "greedy")
              << " decoding.\n";
  }
  std::vector<MediaFile> pending_media;
  while (true) {
    std::cout << "you> " << std::flush;
    std::string input;
    if (!std::getline(std::cin, input)) {
      std::cout << '\n';
      break;
    }
    if (input == "/quit" || input == "/exit") break;
    if (input.starts_with("/image ") || input.starts_with("/audio ")) {
      if (moe26b) {
        std::cerr << "error: Gemma 4 26B is text-only; media commands are unsupported\n";
        continue;
      }
      const bool image = input.starts_with("/image ");
      const std::string_view path =
          std::string_view(input).substr(7U);
      if (path.empty()) {
        std::cerr << "error: media path cannot be empty\n";
        continue;
      }
      pending_media.push_back(
          {image ? MediaFileKind::kImage : MediaFileKind::kAudio,
           std::filesystem::path(path)});
      std::cout << "queued " << (image ? "image" : "audio") << ": "
                << path << '\n';
      continue;
    }
    if (input == "/media") {
      if (pending_media.empty()) {
        std::cout << "no pending media\n";
      } else {
        for (std::size_t index = 0U; index < pending_media.size(); ++index) {
          std::cout << index + 1U << ". "
                    << (pending_media[index].kind == MediaFileKind::kImage
                            ? "image "
                            : "audio ")
                    << pending_media[index].path.string() << '\n';
        }
      }
      continue;
    }
    if (input == "/clear-media") {
      pending_media.clear();
      std::cout << "pending media cleared\n";
      continue;
    }
    if (input.empty()) continue;
    const std::uint64_t cached_tokens = session.value().cached_token_count();
    const std::uint64_t remaining_context =
        cached_tokens < options.max_context
            ? options.max_context - cached_tokens
            : 0U;
    auto turn_media = LoadMediaParts(
        pending_media, remaining_context, options.max_tokens, options.stats);
    if (!turn_media.ok()) {
      std::cerr << "error: " << turn_media.status().message() << '\n';
      continue;
    }
    messages.push_back(TextMessage("user", input));
    message_media.push_back(std::move(turn_media).value());
    pending_media.clear();
    bool turn_failed = false;
    while (true) {
      std::cout << "model> " << std::flush;
      const bool stream_tokens = options.tools.empty();
      auto response = RunTurn(options, processor.value(), messages, false,
                              stream_tokens, &session.value(), message_media);
      if (!response.ok()) {
        std::cerr << "\nerror: " << response.status().message() << '\n';
        turn_failed = true;
        break;
      }
      TurnOutput output = std::move(response).value();
      if (!stream_tokens && !output.display_text.empty()) {
        std::cout << output.display_text;
      }
      std::cout << '\n';
      gem16::ChatMessage assistant;
      assistant.role = "assistant";
      assistant.content = std::move(output.display_text);
      for (const auto& call : output.tool_calls) {
        assistant.tool_calls.push_back(
            {call.id, call.name, call.arguments_json});
      }
      messages.push_back(std::move(assistant));
      message_media.emplace_back();
      if (output.tool_calls.empty()) break;
      for (const auto& call : output.tool_calls) {
        std::cout << "tool call " << call.id << ": " << call.name << ' '
                  << call.arguments_json << "\n";
        std::cout << "tool result> " << std::flush;
        std::string tool_output;
        if (!std::getline(std::cin, tool_output)) {
          turn_failed = true;
          break;
        }
        gem16::ChatMessage result;
        result.role = "tool";
        result.content = std::move(tool_output);
        result.tool_call_id = call.id;
        result.tool_name = call.name;
        messages.push_back(std::move(result));
        message_media.emplace_back();
      }
      if (turn_failed) break;
    }
    if (turn_failed) break;
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
