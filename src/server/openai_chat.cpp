#include "server/openai_chat.h"

#include <algorithm>
#include <limits>
#include <string>
#include <vector>

#include "util/json.h"
#include "runtime/tool_call_parser.h"

namespace gem16::server {
namespace {

Status Invalid(std::string message) {
  return Status(StatusCode::kInvalidArgument, std::move(message));
}

Result<std::string> RequiredString(const json::Value::Object& object,
                                   std::string_view name) {
  const auto iterator = object.find(name);
  if (iterator == object.end() || !iterator->second.is_string()) {
    return Invalid("'" + std::string(name) + "' must be a string");
  }
  return iterator->second.as_string();
}

Result<std::vector<std::uint8_t>> DecodeBase64(std::string_view encoded) {
  if (encoded.empty() || encoded.size() > 16U * 1024U * 1024U ||
      encoded.size() % 4U != 0U) {
    return Invalid("base64 media payload has an invalid bounded length");
  }
  auto value = [](char character) -> int {
    if (character >= 'A' && character <= 'Z') return character - 'A';
    if (character >= 'a' && character <= 'z') return character - 'a' + 26;
    if (character >= '0' && character <= '9') return character - '0' + 52;
    if (character == '+') return 62;
    if (character == '/') return 63;
    return -1;
  };
  std::vector<std::uint8_t> decoded;
  decoded.reserve(encoded.size() / 4U * 3U);
  for (std::size_t index = 0U; index < encoded.size(); index += 4U) {
    const bool last = index + 4U == encoded.size();
    const bool pad2 = encoded[index + 2U] == '=';
    const bool pad3 = encoded[index + 3U] == '=';
    if ((!last && (pad2 || pad3)) || (pad2 && !pad3)) {
      return Invalid("base64 media payload has invalid padding");
    }
    const int a = value(encoded[index]);
    const int b = value(encoded[index + 1U]);
    const int c = pad2 ? 0 : value(encoded[index + 2U]);
    const int d = pad3 ? 0 : value(encoded[index + 3U]);
    if (a < 0 || b < 0 || c < 0 || d < 0) {
      return Invalid("base64 media payload contains an invalid character");
    }
    const std::uint32_t bits =
        (static_cast<std::uint32_t>(a) << 18U) |
        (static_cast<std::uint32_t>(b) << 12U) |
        (static_cast<std::uint32_t>(c) << 6U) |
        static_cast<std::uint32_t>(d);
    decoded.push_back(static_cast<std::uint8_t>(bits >> 16U));
    if (!pad2) decoded.push_back(static_cast<std::uint8_t>(bits >> 8U));
    if (!pad3) decoded.push_back(static_cast<std::uint8_t>(bits));
  }
  return decoded;
}

struct PendingImage {
  std::size_t part_index = 0U;
  std::vector<std::uint8_t> encoded;
  std::string source_name;
};

struct ParsedMessage {
  GenerationMessage message;
  std::vector<PendingImage> images;
};

Result<ParsedMessage> ParseMessage(const json::Value& value) {
  if (!value.is_object()) return Invalid("each message must be an object");
  const auto& object = value.as_object();
  auto role = RequiredString(object, "role");
  if (!role.ok()) return role.status();
  ParsedMessage parsed;
  GenerationMessage& message = parsed.message;
  message.role = std::move(role).value();
  if (message.role == "developer") message.role = "system";
  if (message.role != "system" && message.role != "user" &&
      message.role != "assistant" && message.role != "tool") {
    return Invalid("message role is unsupported");
  }

  const json::Value* content = nullptr;
  if (const auto iterator = object.find("content"); iterator != object.end()) {
    content = &iterator->second;
  }
  if (message.role == "tool") {
    auto call_id = RequiredString(object, "tool_call_id");
    if (!call_id.ok()) return call_id.status();
    if (content == nullptr || !content->is_string()) {
      return Invalid("tool message content must be a string");
    }
    message.content.push_back(GenerationContentPart::ToolResult(
        {std::move(call_id).value(), content->as_string()}));
    return parsed;
  }

  if (content != nullptr && content->is_string()) {
    message.content.push_back(GenerationContentPart::Text(content->as_string()));
  } else if (content != nullptr && content->is_array()) {
    for (const json::Value& part : content->as_array()) {
      if (!part.is_object()) return Invalid("message content part must be an object");
      auto type = RequiredString(part.as_object(), "type");
      if (!type.ok()) return type.status();
      if (type.value() == "text" || type.value() == "input_text") {
        auto text = RequiredString(part.as_object(), "text");
        if (!text.ok()) return text.status();
        message.content.push_back(
            GenerationContentPart::Text(std::move(text).value()));
        continue;
      }
      if (type.value() == "image_url") {
        const json::Value* image_url = part.find("image_url");
        if (image_url == nullptr || !image_url->is_object()) {
          return Invalid("image_url content requires an image_url object");
        }
        auto url = RequiredString(image_url->as_object(), "url");
        if (!url.ok()) return url.status();
        constexpr std::string_view kDataPrefix = "data:image/";
        if (!url.value().starts_with(kDataPrefix)) {
          return Status(StatusCode::kUnsupported,
                        "image_url currently requires an inline data URL");
        }
        const std::size_t separator = url.value().find(";base64,");
        if (separator == std::string::npos || separator <= kDataPrefix.size()) {
          return Invalid("image data URL must use a supported base64 MIME type");
        }
        const std::string_view subtype(url.value().data() + kDataPrefix.size(),
                                       separator - kDataPrefix.size());
        if (subtype != "png" && subtype != "jpeg" && subtype != "jpg" &&
            subtype != "bmp") {
          return Status(StatusCode::kUnsupported,
                        "image data URL must be PNG, JPEG, or BMP");
        }
        auto encoded = DecodeBase64(
            std::string_view(url.value()).substr(separator + 8U));
        if (!encoded.ok()) return encoded.status();
        const std::size_t part_index = message.content.size();
        message.content.push_back(
            GenerationContentPart::Image(VisionImage{}));
        parsed.images.push_back(
            {part_index, std::move(encoded).value(), "OpenAI image data URL"});
        continue;
      }
      if (type.value() == "input_audio") {
        const json::Value* input_audio = part.find("input_audio");
        if (input_audio == nullptr || !input_audio->is_object()) {
          return Invalid("input_audio content requires an input_audio object");
        }
        auto data = RequiredString(input_audio->as_object(), "data");
        if (!data.ok()) return data.status();
        auto format = RequiredString(input_audio->as_object(), "format");
        if (!format.ok()) return format.status();
        if (format.value() != "wav" && format.value() != "mp3" &&
            format.value() != "flac") {
          return Status(StatusCode::kUnsupported,
                        "input_audio format must be wav, mp3, or flac");
        }
        auto encoded = DecodeBase64(data.value());
        if (!encoded.ok()) return encoded.status();
        auto audio = LoadAudioBytes(encoded.value(), "OpenAI input_audio");
        if (!audio.ok()) return audio.status();
        message.content.push_back(
            GenerationContentPart::Audio(std::move(audio).value()));
        continue;
      }
      return Status(StatusCode::kUnsupported,
                    "message content part type is unsupported");
    }
  } else if (content != nullptr && !content->is_null()) {
    return Invalid("message content must be a string, array, or null");
  }

  if (const auto calls = object.find("tool_calls"); calls != object.end()) {
    if (message.role != "assistant" || !calls->second.is_array()) {
      return Invalid("tool_calls must be an array on an assistant message");
    }
    for (const json::Value& call : calls->second.as_array()) {
      if (!call.is_object()) return Invalid("tool call must be an object");
      auto id = RequiredString(call.as_object(), "id");
      if (!id.ok()) return id.status();
      const json::Value* function = call.find("function");
      if (function == nullptr || !function->is_object()) {
        return Invalid("tool call function must be an object");
      }
      auto name = RequiredString(function->as_object(), "name");
      if (!name.ok()) return name.status();
      auto arguments = RequiredString(function->as_object(), "arguments");
      if (!arguments.ok()) return arguments.status();
      message.content.push_back(GenerationContentPart::ToolCall(
          {std::move(id).value(), std::move(name).value(),
           std::move(arguments).value()}));
    }
  }
  if (message.content.empty()) {
    return Invalid("message must contain text or tool calls");
  }
  return parsed;
}

Result<GenerationToolDefinition> ParseTool(const json::Value& value) {
  if (!value.is_object()) return Invalid("each tool must be an object");
  auto type = RequiredString(value.as_object(), "type");
  if (!type.ok()) return type.status();
  if (type.value() != "function") return Invalid("only function tools are supported");
  const json::Value* function = value.find("function");
  if (function == nullptr || !function->is_object()) {
    return Invalid("tool function must be an object");
  }
  GenerationToolDefinition tool;
  auto name = RequiredString(function->as_object(), "name");
  if (!name.ok()) return name.status();
  tool.name = std::move(name).value();
  if (const json::Value* description = function->find("description");
      description != nullptr) {
    if (!description->is_string()) return Invalid("tool description must be a string");
    tool.description = description->as_string();
  }
  if (const json::Value* parameters = function->find("parameters");
      parameters != nullptr) {
    if (!parameters->is_object()) return Invalid("tool parameters must be an object");
    tool.parameters_json = json::Stringify(*parameters);
  } else {
    tool.parameters_json = R"({"type":"object","properties":{}})";
  }
  if (const json::Value* strict = function->find("strict"); strict != nullptr) {
    if (!strict->is_bool()) return Invalid("tool strict must be a boolean");
    tool.strict = strict->as_bool();
  }
  return tool;
}

Result<GenerationToolChoice> ParseToolChoice(const json::Value& value) {
  GenerationToolChoice choice;
  if (value.is_string()) {
    if (value.as_string() == "auto") return choice;
    if (value.as_string() == "none") {
      choice.mode = GenerationToolChoiceMode::kNone;
      return choice;
    }
    if (value.as_string() == "required") {
      choice.mode = GenerationToolChoiceMode::kRequired;
      return choice;
    }
    return Invalid("tool_choice string must be auto, none, or required");
  }
  if (!value.is_object()) return Invalid("tool_choice must be a string or object");
  const json::Value* function = value.find("function");
  if (function == nullptr || !function->is_object()) {
    return Invalid("named tool_choice requires a function object");
  }
  auto name = RequiredString(function->as_object(), "name");
  if (!name.ok()) return name.status();
  choice.mode = GenerationToolChoiceMode::kFunction;
  choice.function_name = std::move(name).value();
  return choice;
}

Result<GenerationToolDefinition> ParseResponsesTool(
    const json::Value& value) {
  if (!value.is_object()) return Invalid("each tool must be an object");
  auto type = RequiredString(value.as_object(), "type");
  if (!type.ok()) return type.status();
  if (type.value() != "function") {
    return Status(StatusCode::kUnsupported,
                  "only function tools are supported");
  }
  GenerationToolDefinition tool;
  auto name = RequiredString(value.as_object(), "name");
  if (!name.ok()) return name.status();
  tool.name = std::move(name).value();
  if (const json::Value* description = value.find("description");
      description != nullptr) {
    if (!description->is_string()) {
      return Invalid("tool description must be a string");
    }
    tool.description = description->as_string();
  }
  if (const json::Value* parameters = value.find("parameters");
      parameters != nullptr) {
    if (!parameters->is_object()) {
      return Invalid("tool parameters must be an object");
    }
    tool.parameters_json = json::Stringify(*parameters);
  } else {
    tool.parameters_json = R"({"type":"object","properties":{}})";
  }
  if (const json::Value* strict = value.find("strict"); strict != nullptr) {
    if (!strict->is_bool()) return Invalid("tool strict must be a boolean");
    tool.strict = strict->as_bool();
  }
  return tool;
}

Result<GenerationToolChoice> ParseResponsesToolChoice(
    const json::Value& value) {
  if (value.is_string()) return ParseToolChoice(value);
  if (!value.is_object()) {
    return Invalid("tool_choice must be a string or object");
  }
  auto type = RequiredString(value.as_object(), "type");
  if (!type.ok()) return type.status();
  if (type.value() != "function") {
    return Status(StatusCode::kUnsupported,
                  "only function tool_choice is supported");
  }
  auto name = RequiredString(value.as_object(), "name");
  if (!name.ok()) return name.status();
  GenerationToolChoice choice;
  choice.mode = GenerationToolChoiceMode::kFunction;
  choice.function_name = std::move(name).value();
  return choice;
}

Result<PendingImage> ParseResponsesImage(const json::Value& part,
                                         std::size_t part_index) {
  auto url = RequiredString(part.as_object(), "image_url");
  if (!url.ok()) return url.status();
  constexpr std::string_view kDataPrefix = "data:image/";
  if (!url.value().starts_with(kDataPrefix)) {
    return Status(StatusCode::kUnsupported,
                  "input_image currently requires an inline data URL");
  }
  const std::size_t separator = url.value().find(";base64,");
  if (separator == std::string::npos || separator <= kDataPrefix.size()) {
    return Invalid("image data URL must use a supported base64 MIME type");
  }
  const std::string_view subtype(url.value().data() + kDataPrefix.size(),
                                 separator - kDataPrefix.size());
  if (subtype != "png" && subtype != "jpeg" && subtype != "jpg" &&
      subtype != "bmp") {
    return Status(StatusCode::kUnsupported,
                  "image data URL must be PNG, JPEG, or BMP");
  }
  auto encoded =
      DecodeBase64(std::string_view(url.value()).substr(separator + 8U));
  if (!encoded.ok()) return encoded.status();
  return PendingImage{part_index, std::move(encoded).value(),
                      "Responses input_image data URL"};
}

Result<ParsedMessage> ParseResponsesMessage(const json::Value& value) {
  if (!value.is_object()) return Invalid("each input item must be an object");
  const auto& object = value.as_object();
  auto role = RequiredString(object, "role");
  if (!role.ok()) return role.status();
  ParsedMessage parsed;
  parsed.message.role = std::move(role).value();
  if (parsed.message.role == "developer") parsed.message.role = "system";
  if (parsed.message.role != "system" && parsed.message.role != "user" &&
      parsed.message.role != "assistant") {
    return Invalid("Responses message role is unsupported");
  }
  const json::Value* content = value.find("content");
  if (content == nullptr) return Invalid("message content is required");
  if (content->is_string()) {
    parsed.message.content.push_back(
        GenerationContentPart::Text(content->as_string()));
    return parsed;
  }
  if (!content->is_array() || content->as_array().empty()) {
    return Invalid("message content must be a string or non-empty array");
  }
  for (const json::Value& part : content->as_array()) {
    if (!part.is_object()) return Invalid("message content part must be an object");
    auto type = RequiredString(part.as_object(), "type");
    if (!type.ok()) return type.status();
    if (type.value() == "input_text" || type.value() == "output_text") {
      auto text = RequiredString(part.as_object(), "text");
      if (!text.ok()) return text.status();
      parsed.message.content.push_back(
          GenerationContentPart::Text(std::move(text).value()));
    } else if (type.value() == "input_image") {
      const std::size_t part_index = parsed.message.content.size();
      auto image = ParseResponsesImage(part, part_index);
      if (!image.ok()) return image.status();
      parsed.message.content.push_back(
          GenerationContentPart::Image(VisionImage{}));
      parsed.images.push_back(std::move(image).value());
    } else if (type.value() == "input_audio") {
      const json::Value* input_audio = part.find("input_audio");
      if (input_audio == nullptr || !input_audio->is_object()) {
        return Invalid("input_audio content requires an input_audio object");
      }
      auto data = RequiredString(input_audio->as_object(), "data");
      if (!data.ok()) return data.status();
      auto format = RequiredString(input_audio->as_object(), "format");
      if (!format.ok()) return format.status();
      if (format.value() != "wav" && format.value() != "mp3" &&
          format.value() != "flac") {
        return Status(StatusCode::kUnsupported,
                      "input_audio format must be wav, mp3, or flac");
      }
      auto encoded = DecodeBase64(data.value());
      if (!encoded.ok()) return encoded.status();
      auto audio = LoadAudioBytes(encoded.value(), "Responses input_audio");
      if (!audio.ok()) return audio.status();
      parsed.message.content.push_back(
          GenerationContentPart::Audio(std::move(audio).value()));
    } else {
      return Status(StatusCode::kUnsupported,
                    "Responses message content type is unsupported");
    }
  }
  return parsed;
}

std::string FinishReasonJson(GenerationFinishReason reason) {
  return json::Quote(GenerationFinishReasonName(reason));
}

void AppendUsage(const ChatGenerationResponse& response, std::string& output) {
  const std::size_t prompt = response.prompt_token_ids.size();
  const std::size_t completion = response.inference.output_token_ids.size();
  output.append("{\"prompt_tokens\":");
  output.append(std::to_string(prompt));
  output.append(",\"completion_tokens\":");
  output.append(std::to_string(completion));
  output.append(",\"total_tokens\":");
  output.append(std::to_string(prompt + completion));
  output.push_back('}');
}

void AppendResponsesUsage(const ChatGenerationResponse& response,
                          std::string& output) {
  const std::size_t input = response.prompt_token_ids.size();
  const std::size_t generated = response.inference.output_token_ids.size();
  output.append("{\"input_tokens\":");
  output.append(std::to_string(input));
  output.append(",\"input_tokens_details\":{\"cached_tokens\":0,"
                "\"cache_write_tokens\":0},\"output_tokens\":");
  output.append(std::to_string(generated));
  output.append(",\"output_tokens_details\":{\"reasoning_tokens\":");
  output.append(std::to_string(response.inference.reasoning_tokens));
  output.append("},\"total_tokens\":");
  output.append(std::to_string(input + generated));
  output.push_back('}');
}

std::string ResponsesToolChoiceJson(const GenerationToolChoice& choice) {
  if (choice.mode == GenerationToolChoiceMode::kNone) return "\"none\"";
  if (choice.mode == GenerationToolChoiceMode::kAuto) return "\"auto\"";
  if (choice.mode == GenerationToolChoiceMode::kRequired) {
    return "\"required\"";
  }
  return "{\"type\":\"function\",\"name\":" +
         json::Quote(choice.function_name) + "}";
}

std::string ResponsesToolsJson(
    const std::vector<GenerationToolDefinition>& tools) {
  std::string output = "[";
  for (std::size_t index = 0U; index < tools.size(); ++index) {
    if (index != 0U) output.push_back(',');
    const GenerationToolDefinition& tool = tools[index];
    output.append("{\"type\":\"function\",\"name\":");
    output.append(json::Quote(tool.name));
    output.append(",\"description\":");
    output.append(json::Quote(tool.description));
    output.append(",\"parameters\":");
    output.append(tool.parameters_json);
    output.append(",\"strict\":");
    output.append(tool.strict ? "true" : "false");
    output.push_back('}');
  }
  output.push_back(']');
  return output;
}

}  // namespace

Result<OpenAiChatRequest> ParseChatCompletionsRequest(
    std::string_view body, const OpenAiChatAdapterOptions& options) {
  if (body.size() > 16U * 1024U * 1024U) {
    return Invalid("request body exceeds 16 MiB");
  }
  auto root = json::Parse(body, json::ParseLimits{64U, 1'000'000U,
                                                  16U * 1024U * 1024U});
  if (!root.ok()) return root.status();
  if (!root.value().is_object()) return Invalid("request body must be an object");
  const auto& object = root.value().as_object();
  OpenAiChatRequest request;
  auto model = RequiredString(object, "model");
  if (!model.ok()) return model.status();
  request.model = std::move(model).value();

  const json::Value* messages = root.value().find("messages");
  if (messages == nullptr || !messages->is_array() || messages->as_array().empty()) {
    return Invalid("'messages' must be a non-empty array");
  }
  request.generation.messages.reserve(messages->as_array().size());
  struct LocatedImage {
    std::size_t message_index = 0U;
    PendingImage image;
  };
  std::vector<LocatedImage> images;
  for (const json::Value& message_value : messages->as_array()) {
    auto message = ParseMessage(message_value);
    if (!message.ok()) return message.status();
    const std::size_t message_index = request.generation.messages.size();
    for (PendingImage& image : message.value().images) {
      images.push_back({message_index, std::move(image)});
    }
    request.generation.messages.push_back(
        std::move(message).value().message);
  }

  const json::Value* max_tokens = root.value().find("max_completion_tokens");
  if (max_tokens == nullptr) max_tokens = root.value().find("max_tokens");
  if (max_tokens != nullptr) {
    if (!max_tokens->is_integer() || max_tokens->as_integer() <= 0) {
      return Invalid("max completion tokens must be a positive integer");
    }
    request.generation.max_generated_tokens =
        static_cast<std::uint64_t>(max_tokens->as_integer());
  }
  if (const json::Value* stream = root.value().find("stream"); stream != nullptr) {
    if (!stream->is_bool()) return Invalid("stream must be a boolean");
    request.stream = stream->as_bool();
  }
  if (const json::Value* stream_options = root.value().find("stream_options");
      stream_options != nullptr) {
    if (!stream_options->is_object()) return Invalid("stream_options must be an object");
    if (const json::Value* include = stream_options->find("include_usage");
        include != nullptr) {
      if (!include->is_bool()) return Invalid("include_usage must be a boolean");
      request.include_usage = include->as_bool();
    }
  }
  if (const json::Value* effort = root.value().find("reasoning_effort");
      effort != nullptr) {
    if (!effort->is_string()) return Invalid("reasoning_effort must be a string");
    if (effort->as_string() == "none") {
      request.generation.thinking.effort = ThinkingEffort::kOff;
    } else if (effort->as_string() == "low") {
      request.generation.thinking.effort = ThinkingEffort::kSmall;
    } else if (effort->as_string() == "medium") {
      request.generation.thinking.effort = ThinkingEffort::kMedium;
    } else if (effort->as_string() == "high") {
      request.generation.thinking.effort = ThinkingEffort::kHigh;
    } else {
      return Invalid("reasoning_effort must be none, low, medium, or high");
    }
  }
  if (const json::Value* tools = root.value().find("tools"); tools != nullptr) {
    if (!tools->is_array()) return Invalid("tools must be an array");
    for (const json::Value& tool_value : tools->as_array()) {
      auto tool = ParseTool(tool_value);
      if (!tool.ok()) return tool.status();
      request.generation.tools.push_back(std::move(tool).value());
    }
  }
  if (const json::Value* choice = root.value().find("tool_choice");
      choice != nullptr) {
    auto parsed_choice = ParseToolChoice(*choice);
    if (!parsed_choice.ok()) return parsed_choice.status();
    request.generation.tool_choice = std::move(parsed_choice).value();
  }
  if (const json::Value* parallel = root.value().find("parallel_tool_calls");
      parallel != nullptr) {
    if (!parallel->is_bool()) return Invalid("parallel_tool_calls must be a boolean");
    request.generation.parallel_tool_calls = parallel->as_bool();
  }
  if (const json::Value* count = root.value().find("n"); count != nullptr &&
      (!count->is_integer() || count->as_integer() != 1)) {
    return Status(StatusCode::kUnsupported, "only n=1 is supported");
  }
  for (const std::string_view unsupported : {"temperature", "top_p",
                                              "frequency_penalty",
                                              "presence_penalty", "seed"}) {
    if (root.value().find(unsupported) != nullptr) {
      return Status(StatusCode::kUnsupported,
                    "per-request sampling field '" + std::string(unsupported) +
                        "' is not supported by the single-slot server");
    }
  }
  const Status tool_status =
      internal::ValidateToolDefinitions(request.generation.tools);
  if (!tool_status.ok()) return tool_status;
  std::uint64_t audio_tokens = 0U;
  for (const GenerationMessage& message : request.generation.messages) {
    for (const GenerationContentPart& part : message.content) {
      if (part.kind == GenerationContentKind::kAudio) {
        audio_tokens += (part.audio.samples.size() + 639U) / 640U;
      }
    }
  }
  // Keep historical media identity stable when a later OpenAI turn changes
  // max_completion_tokens. The session performs the exact output-capacity
  // check separately for every turn.
  const std::uint64_t output_reserve =
      std::min<std::uint64_t>(128U, options.context_tokens / 4U);
  const std::uint64_t fixed_reserve =
      output_reserve + audio_tokens + 64U + images.size() * 2U;
  const std::uint32_t image_budget = AutomaticVisionSoftTokenBudget(
      options.context_tokens, fixed_reserve, images.size());
  for (LocatedImage& located : images) {
    auto image = LoadVisionImageBytes(
        located.image.encoded, located.image.source_name,
        VisionImageOptions{image_budget, false});
    if (!image.ok()) return image.status();
    request.generation.messages[located.message_index]
        .content[located.image.part_index] =
        GenerationContentPart::Image(std::move(image).value());
  }
  return request;
}

Result<OpenAiResponsesRequest> ParseResponsesRequest(
    std::string_view body, const OpenAiChatAdapterOptions& options) {
  if (body.size() > 16U * 1024U * 1024U) {
    return Invalid("request body exceeds 16 MiB");
  }
  auto root = json::Parse(body, json::ParseLimits{64U, 1'000'000U,
                                                  16U * 1024U * 1024U});
  if (!root.ok()) return root.status();
  if (!root.value().is_object()) return Invalid("request body must be an object");
  const auto& object = root.value().as_object();
  OpenAiResponsesRequest request;
  auto model = RequiredString(object, "model");
  if (!model.ok()) return model.status();
  request.model = std::move(model).value();

  if (const json::Value* previous = root.value().find("previous_response_id");
      previous != nullptr && !previous->is_null()) {
    if (!previous->is_string() || previous->as_string().empty()) {
      return Invalid("previous_response_id must be a non-empty string or null");
    }
    request.previous_response_id = previous->as_string();
  }
  if (const json::Value* instructions = root.value().find("instructions");
      instructions != nullptr && !instructions->is_null()) {
    if (!instructions->is_string()) {
      return Invalid("instructions must be a string or null");
    }
    if (request.previous_response_id.has_value()) {
      return Status(
          StatusCode::kUnsupported,
          "changing instructions on a resident previous_response_id chain is not yet supported");
    }
    request.instructions = instructions->as_string();
    GenerationMessage instruction;
    instruction.role = "system";
    instruction.content.push_back(
        GenerationContentPart::Text(*request.instructions));
    request.generation.messages.push_back(std::move(instruction));
  }

  const json::Value* input = root.value().find("input");
  if (input == nullptr) return Invalid("'input' is required");
  struct LocatedImage {
    std::size_t message_index = 0U;
    PendingImage image;
  };
  std::vector<LocatedImage> images;
  if (input->is_string()) {
    GenerationMessage message;
    message.role = "user";
    message.content.push_back(
        GenerationContentPart::Text(input->as_string()));
    request.generation.messages.push_back(std::move(message));
  } else if (input->is_array() && !input->as_array().empty()) {
    for (const json::Value& item : input->as_array()) {
      if (!item.is_object()) return Invalid("each input item must be an object");
      const json::Value* type_value = item.find("type");
      const std::string type =
          type_value != nullptr && type_value->is_string()
              ? type_value->as_string()
              : "message";
      if (type == "message") {
        auto parsed = ParseResponsesMessage(item);
        if (!parsed.ok()) return parsed.status();
        const std::size_t message_index = request.generation.messages.size();
        for (PendingImage& image : parsed.value().images) {
          images.push_back({message_index, std::move(image)});
        }
        request.generation.messages.push_back(
            std::move(parsed).value().message);
      } else if (type == "function_call") {
        auto call_id = RequiredString(item.as_object(), "call_id");
        if (!call_id.ok()) return call_id.status();
        auto name = RequiredString(item.as_object(), "name");
        if (!name.ok()) return name.status();
        auto arguments = RequiredString(item.as_object(), "arguments");
        if (!arguments.ok()) return arguments.status();
        GenerationMessage message;
        message.role = "assistant";
        message.content.push_back(GenerationContentPart::ToolCall(
            {std::move(call_id).value(), std::move(name).value(),
             std::move(arguments).value()}));
        request.generation.messages.push_back(std::move(message));
      } else if (type == "function_call_output") {
        auto call_id = RequiredString(item.as_object(), "call_id");
        if (!call_id.ok()) return call_id.status();
        const json::Value* output = item.find("output");
        if (output == nullptr) return Invalid("function_call_output requires output");
        std::string output_text;
        if (output->is_string()) {
          output_text = output->as_string();
        } else if (output->is_array()) {
          for (const json::Value& part : output->as_array()) {
            if (!part.is_object()) {
              return Invalid("function output content must be an object");
            }
            auto part_type = RequiredString(part.as_object(), "type");
            if (!part_type.ok()) return part_type.status();
            if (part_type.value() != "input_text") {
              return Status(StatusCode::kUnsupported,
                            "only text function outputs are supported");
            }
            auto text = RequiredString(part.as_object(), "text");
            if (!text.ok()) return text.status();
            output_text.append(text.value());
          }
        } else {
          return Invalid("function output must be a string or content array");
        }
        GenerationMessage message;
        message.role = "tool";
        message.content.push_back(GenerationContentPart::ToolResult(
            {std::move(call_id).value(), std::move(output_text)}));
        request.generation.messages.push_back(std::move(message));
      } else {
        return Status(StatusCode::kUnsupported,
                      "Responses input item type is unsupported");
      }
    }
  } else {
    return Invalid("'input' must be a string or non-empty array");
  }

  if (const json::Value* max_tokens = root.value().find("max_output_tokens");
      max_tokens != nullptr) {
    if (!max_tokens->is_integer() || max_tokens->as_integer() <= 0) {
      return Invalid("max_output_tokens must be a positive integer");
    }
    request.generation.max_generated_tokens =
        static_cast<std::uint64_t>(max_tokens->as_integer());
  }
  if (const json::Value* stream = root.value().find("stream");
      stream != nullptr) {
    if (!stream->is_bool()) return Invalid("stream must be a boolean");
    request.stream = stream->as_bool();
  }
  if (const json::Value* store = root.value().find("store"); store != nullptr) {
    if (!store->is_bool()) return Invalid("store must be a boolean");
    request.store = store->as_bool();
    if (!request.store) {
      return Status(StatusCode::kUnsupported,
                    "the resident Responses chain currently requires store=true");
    }
  }
  if (const json::Value* reasoning = root.value().find("reasoning");
      reasoning != nullptr) {
    if (!reasoning->is_object()) return Invalid("reasoning must be an object");
    if (const json::Value* effort = reasoning->find("effort");
        effort != nullptr && !effort->is_null()) {
      if (!effort->is_string()) return Invalid("reasoning.effort must be a string");
      if (effort->as_string() == "none") {
        request.generation.thinking.effort = ThinkingEffort::kOff;
      } else if (effort->as_string() == "low") {
        request.generation.thinking.effort = ThinkingEffort::kSmall;
      } else if (effort->as_string() == "medium") {
        request.generation.thinking.effort = ThinkingEffort::kMedium;
      } else if (effort->as_string() == "high") {
        request.generation.thinking.effort = ThinkingEffort::kHigh;
      } else {
        return Invalid("reasoning.effort must be none, low, medium, or high");
      }
    }
  }
  if (const json::Value* tools = root.value().find("tools"); tools != nullptr) {
    if (!tools->is_array()) return Invalid("tools must be an array");
    request.tools_present = true;
    for (const json::Value& tool_value : tools->as_array()) {
      auto tool = ParseResponsesTool(tool_value);
      if (!tool.ok()) return tool.status();
      request.generation.tools.push_back(std::move(tool).value());
    }
  }
  if (const json::Value* choice = root.value().find("tool_choice");
      choice != nullptr) {
    request.tool_choice_present = true;
    auto parsed = ParseResponsesToolChoice(*choice);
    if (!parsed.ok()) return parsed.status();
    request.generation.tool_choice = std::move(parsed).value();
  }
  if (const json::Value* parallel = root.value().find("parallel_tool_calls");
      parallel != nullptr) {
    if (!parallel->is_bool()) {
      return Invalid("parallel_tool_calls must be a boolean");
    }
    request.generation.parallel_tool_calls = parallel->as_bool();
  }
  for (const std::string_view unsupported : {
           "background", "conversation", "prompt", "temperature", "top_p",
           "truncation", "text", "include", "max_tool_calls"}) {
    if (root.value().find(unsupported) != nullptr) {
      return Status(StatusCode::kUnsupported,
                    "Responses field '" + std::string(unsupported) +
                        "' is not supported by the resident server");
    }
  }
  const Status tool_status =
      internal::ValidateToolDefinitions(request.generation.tools);
  if (!tool_status.ok()) return tool_status;

  std::uint64_t audio_tokens = 0U;
  for (const GenerationMessage& message : request.generation.messages) {
    for (const GenerationContentPart& part : message.content) {
      if (part.kind == GenerationContentKind::kAudio) {
        audio_tokens += (part.audio.samples.size() + 639U) / 640U;
      }
    }
  }
  const std::uint64_t output_reserve =
      std::min<std::uint64_t>(128U, options.context_tokens / 4U);
  const std::uint64_t fixed_reserve =
      output_reserve + audio_tokens + 64U + images.size() * 2U;
  const std::uint32_t image_budget = AutomaticVisionSoftTokenBudget(
      options.context_tokens, fixed_reserve, images.size());
  for (LocatedImage& located : images) {
    auto image = LoadVisionImageBytes(
        located.image.encoded, located.image.source_name,
        VisionImageOptions{image_budget, false});
    if (!image.ok()) return image.status();
    request.generation.messages[located.message_index]
        .content[located.image.part_index] =
        GenerationContentPart::Image(std::move(image).value());
  }
  return request;
}

std::string ChatCompletionJson(const OpenAiResponseIdentity& identity,
                               const ChatGenerationResponse& response) {
  std::string output = "{\"id\":" + json::Quote(identity.id) +
                       ",\"object\":\"chat.completion\",\"created\":" +
                       std::to_string(identity.created) + ",\"model\":" +
                       json::Quote(identity.model) +
                       ",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":";
  if (response.assistant_text.empty() && !response.tool_calls.empty()) {
    output.append("null");
  } else {
    output.append(json::Quote(response.assistant_text));
  }
  if (!response.tool_calls.empty()) {
    output.append(",\"tool_calls\":[");
    for (std::size_t index = 0U; index < response.tool_calls.size(); ++index) {
      if (index != 0U) output.push_back(',');
      const GenerationToolCall& call = response.tool_calls[index];
      output.append("{\"id\":");
      output.append(json::Quote(call.id));
      output.append(",\"type\":\"function\",\"function\":{\"name\":");
      output.append(json::Quote(call.name));
      output.append(",\"arguments\":");
      output.append(json::Quote(call.arguments_json));
      output.append("}}");
    }
    output.push_back(']');
  }
  output.append("},\"finish_reason\":");
  output.append(FinishReasonJson(response.finish_reason));
  output.append("}],\"usage\":");
  AppendUsage(response, output);
  output.push_back('}');
  return output;
}

std::string ChatCompletionChunkJson(
    const OpenAiResponseIdentity& identity, std::string_view delta_json,
    std::optional<GenerationFinishReason> finish_reason,
    const ChatGenerationResponse* usage) {
  std::string output = "{\"id\":" + json::Quote(identity.id) +
                       ",\"object\":\"chat.completion.chunk\",\"created\":" +
                       std::to_string(identity.created) + ",\"model\":" +
                       json::Quote(identity.model) + ",\"choices\":[";
  if (usage != nullptr && delta_json.empty() && !finish_reason.has_value()) {
    output.append("],\"usage\":");
    AppendUsage(*usage, output);
    output.push_back('}');
    return output;
  }
  output.append("{\"index\":0,\"delta\":");
  output.append(delta_json.empty() ? "{}" : delta_json);
  output.append(",\"finish_reason\":");
  output.append(finish_reason.has_value() ? FinishReasonJson(*finish_reason)
                                          : "null");
  output.append("}],\"usage\":null}");
  return output;
}

std::string ResponseShellJson(const OpenAiResponseIdentity& identity,
                              const OpenAiResponsesRequest& request,
                              std::string_view status,
                              const ChatGenerationResponse* response) {
  const bool incomplete = response != nullptr &&
                          response->finish_reason ==
                              GenerationFinishReason::kLength;
  std::string output = "{\"id\":" + json::Quote(identity.id) +
                       ",\"object\":\"response\",\"created_at\":" +
                       std::to_string(identity.created) + ",\"status\":" +
                       json::Quote(status) + ",\"completed_at\":";
  output.append(status == "in_progress" ? "null"
                                          : std::to_string(identity.created));
  output.append(",\"error\":null,\"incomplete_details\":");
  output.append(incomplete ? "{\"reason\":\"max_output_tokens\"}" : "null");
  output.append(",\"instructions\":");
  output.append(request.instructions.has_value()
                    ? json::Quote(*request.instructions)
                    : "null");
  output.append(",\"max_output_tokens\":");
  output.append(request.generation.max_generated_tokens.has_value()
                    ? std::to_string(
                          *request.generation.max_generated_tokens)
                    : "null");
  output.append(",\"model\":");
  output.append(json::Quote(identity.model));
  output.append(",\"output\":[");
  if (response != nullptr) {
    bool needs_comma = false;
    if (!response->reasoning_text.empty()) {
      output.append("{\"id\":");
      output.append(json::Quote("rs_" + identity.id));
      output.append(",\"type\":\"reasoning\",\"summary\":[],"
                    "\"content\":[{\"type\":\"reasoning_text\",\"text\":");
      output.append(json::Quote(response->reasoning_text));
      output.append("}],\"status\":\"completed\"}");
      needs_comma = true;
    }
    if (!response->assistant_text.empty() || response->tool_calls.empty()) {
      if (needs_comma) output.push_back(',');
      output.append("{\"id\":");
      output.append(json::Quote("msg_" + identity.id));
      output.append(",\"type\":\"message\",\"status\":\"completed\","
                    "\"role\":\"assistant\",\"content\":[{"
                    "\"type\":\"output_text\",\"text\":");
      output.append(json::Quote(response->assistant_text));
      output.append(",\"annotations\":[]}]}");
      needs_comma = true;
    }
    for (std::size_t index = 0U; index < response->tool_calls.size();
         ++index) {
      if (needs_comma) output.push_back(',');
      const GenerationToolCall& call = response->tool_calls[index];
      output.append("{\"id\":");
      output.append(json::Quote("fc_" + identity.id + "_" +
                                std::to_string(index)));
      output.append(",\"type\":\"function_call\",\"status\":\"completed\","
                    "\"arguments\":");
      output.append(json::Quote(call.arguments_json));
      output.append(",\"call_id\":");
      output.append(json::Quote(call.id));
      output.append(",\"name\":");
      output.append(json::Quote(call.name));
      output.push_back('}');
      needs_comma = true;
    }
  }
  output.append("],\"parallel_tool_calls\":");
  output.append(request.generation.parallel_tool_calls ? "true" : "false");
  output.append(",\"previous_response_id\":");
  output.append(request.previous_response_id.has_value()
                    ? json::Quote(*request.previous_response_id)
                    : "null");
  output.append(",\"reasoning\":null,\"store\":true,\"temperature\":null,"
                "\"text\":{\"format\":{\"type\":\"text\"}},"
                "\"tool_choice\":");
  output.append(ResponsesToolChoiceJson(request.generation.tool_choice));
  output.append(",\"tools\":");
  output.append(ResponsesToolsJson(request.generation.tools));
  output.append(",\"top_p\":null,\"truncation\":\"disabled\","
                "\"usage\":");
  if (response == nullptr) {
    output.append("null");
  } else {
    AppendResponsesUsage(*response, output);
  }
  output.append(",\"metadata\":{}}");
  return output;
}

std::string ResponseJson(const OpenAiResponseIdentity& identity,
                         const OpenAiResponsesRequest& request,
                         const ChatGenerationResponse& response) {
  return ResponseShellJson(
      identity, request,
      response.finish_reason == GenerationFinishReason::kLength
          ? "incomplete"
          : "completed",
      &response);
}

std::string OpenAiErrorJson(std::string_view message, std::string_view type) {
  return "{\"error\":{\"message\":" + json::Quote(message) +
         ",\"type\":" + json::Quote(type) +
         ",\"param\":null,\"code\":null}}";
}

}  // namespace gem16::server
