#include "model/tokenizer_config.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <string_view>
#include <utility>

#include "util/json.h"

namespace gem16gb::internal {
namespace {

constexpr std::uint64_t kMaximumTokenizerConfigBytes = 1024U * 1024U;

Status Error(StatusCode code, std::string message) {
  return Status(code, std::move(message));
}

Result<std::string> ReadConfig(const std::filesystem::path& path) {
  std::error_code error;
  const auto size = std::filesystem::file_size(path, error);
  if (error) {
    return Error(StatusCode::kIoError,
                 "cannot stat tokenizer_config.json: " + path.string() +
                     ": " + error.message());
  }
  if (size > kMaximumTokenizerConfigBytes) {
    return Error(StatusCode::kDataLoss,
                 "tokenizer_config.json exceeds 1 MiB safety limit: " +
                     path.string());
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return Error(StatusCode::kIoError,
                 "cannot open tokenizer_config.json: " + path.string());
  }
  std::ostringstream contents;
  contents << input.rdbuf();
  if (!input.good() && !input.eof()) {
    return Error(StatusCode::kIoError,
                 "failed while reading tokenizer_config.json: " + path.string());
  }
  return contents.str();
}

const json::Value* Member(const json::Value* value, std::string_view name) {
  return value != nullptr && value->is_object() ? value->find(name) : nullptr;
}

Result<std::string> RequiredString(const json::Value* object,
                                   std::string_view field) {
  const json::Value* value = Member(object, field);
  if (value == nullptr || !value->is_string()) {
    return Error(StatusCode::kDataLoss,
                 "tokenizer_config.json field " + std::string(field) +
                     " must be a string");
  }
  return value->as_string();
}

Result<std::vector<std::string>> RequiredStringArray(
    const json::Value* object, std::string_view field) {
  const json::Value* value = Member(object, field);
  if (value == nullptr || !value->is_array() || value->as_array().empty()) {
    return Error(StatusCode::kDataLoss,
                 "tokenizer_config.json field " + std::string(field) +
                     " must be a non-empty string array");
  }
  std::vector<std::string> result;
  result.reserve(value->as_array().size());
  for (const auto& item : value->as_array()) {
    if (!item.is_string()) {
      return Error(StatusCode::kDataLoss,
                   "tokenizer_config.json field " + std::string(field) +
                       " must contain only strings");
    }
    result.push_back(item.as_string());
  }
  return result;
}

Status ContractError(std::string field, std::string expected) {
  return Error(StatusCode::kUnsupported,
               "unsupported primary tokenizer_config.json: " +
                   std::move(field) + " must be " + std::move(expected));
}

std::string Trim(std::string_view value) {
  std::size_t begin = 0;
  std::size_t end = value.size();
  while (begin < end &&
         std::isspace(static_cast<unsigned char>(value[begin])) != 0) {
    ++begin;
  }
  while (end > begin &&
         std::isspace(static_cast<unsigned char>(value[end - 1U])) != 0) {
    --end;
  }
  return std::string(value.substr(begin, end - begin));
}

}  // namespace

Result<TokenizerConfig> LoadTokenizerConfig(
    const std::filesystem::path& path) {
  auto text = ReadConfig(path);
  if (!text.ok()) return text.status();
  auto parsed = json::Parse(text.value(),
                            {.max_depth = 64,
                             .max_values = 10'000,
                             .max_string_bytes = kMaximumTokenizerConfigBytes});
  if (!parsed.ok()) {
    return Error(parsed.status().code(),
                 path.string() + ": " + parsed.status().message());
  }
  if (!parsed.value().is_object()) {
    return Error(StatusCode::kDataLoss,
                 "tokenizer_config.json root must be an object");
  }
  const json::Value* root = &parsed.value();

  TokenizerConfig config;
  auto tokenizer_class = RequiredString(root, "tokenizer_class");
  auto bos_token = RequiredString(root, "bos_token");
  auto eos_token = RequiredString(root, "eos_token");
  auto eot_token = RequiredString(root, "eot_token");
  auto tool_response_end = RequiredString(root, "etr_token");
  auto tool_call_start = RequiredString(root, "stc_token");
  if (!tokenizer_class.ok()) return tokenizer_class.status();
  if (!bos_token.ok()) return bos_token.status();
  if (!eos_token.ok()) return eos_token.status();
  if (!eot_token.ok()) return eot_token.status();
  if (!tool_response_end.ok()) return tool_response_end.status();
  if (!tool_call_start.ok()) return tool_call_start.status();
  config.tokenizer_class = std::move(tokenizer_class).value();
  config.bos_token = std::move(bos_token).value();
  config.eos_token = std::move(eos_token).value();
  config.eot_token = std::move(eot_token).value();
  config.tool_response_end_token = std::move(tool_response_end).value();
  config.tool_call_start_token = std::move(tool_call_start).value();

  const json::Value* maximum = Member(root, "model_max_length");
  if (maximum == nullptr || !maximum->is_number() ||
      !std::isfinite(maximum->as_number()) || maximum->as_number() <= 0.0) {
    return Error(StatusCode::kDataLoss,
                 "tokenizer_config.json model_max_length must be a positive finite number");
  }
  config.model_max_length = maximum->as_number();

  const json::Value* response = Member(root, "response_template");
  const json::Value* defaults = Member(response, "defaults");
  const json::Value* fields = Member(response, "fields");
  const json::Value* content = Member(fields, "content");
  const json::Value* thinking = Member(fields, "thinking");
  const json::Value* tool_calls = Member(fields, "tool_calls");
  if (response == nullptr || !response->is_object() || defaults == nullptr ||
      fields == nullptr || content == nullptr || thinking == nullptr ||
      tool_calls == nullptr) {
    return Error(StatusCode::kDataLoss,
                 "tokenizer_config.json has no complete response_template");
  }

  auto role = RequiredString(defaults, "role");
  auto anchors = RequiredStringArray(response, "start_anchor");
  auto content_type = RequiredString(content, "content");
  auto content_close = RequiredStringArray(content, "close");
  auto thinking_type = RequiredString(thinking, "content");
  auto thinking_open = RequiredString(thinking, "open");
  auto thinking_close = RequiredString(thinking, "close");
  auto tool_type = RequiredString(tool_calls, "content");
  auto tool_open = RequiredString(tool_calls, "open_pattern");
  auto tool_close = RequiredString(tool_calls, "close");
  if (!role.ok()) return role.status();
  if (!anchors.ok()) return anchors.status();
  if (!content_type.ok()) return content_type.status();
  if (!content_close.ok()) return content_close.status();
  if (!thinking_type.ok()) return thinking_type.status();
  if (!thinking_open.ok()) return thinking_open.status();
  if (!thinking_close.ok()) return thinking_close.status();
  if (!tool_type.ok()) return tool_type.status();
  if (!tool_open.ok()) return tool_open.status();
  if (!tool_close.ok()) return tool_close.status();
  if (content_type.value() != "text" || thinking_type.value() != "text" ||
      tool_type.value() != "json") {
    return Error(StatusCode::kUnsupported,
                 "tokenizer_config.json response_template content types are unsupported");
  }
  const json::Value* repeats = Member(tool_calls, "repeats");
  if (repeats == nullptr || !repeats->is_bool()) {
    return Error(StatusCode::kDataLoss,
                 "tokenizer_config.json tool_calls.repeats must be a boolean");
  }
  config.response_role = std::move(role).value();
  config.response_start_anchors = std::move(anchors).value();
  config.content_close_tokens = std::move(content_close).value();
  config.thinking_open = std::move(thinking_open).value();
  config.thinking_close = std::move(thinking_close).value();
  config.tool_call_open_pattern = std::move(tool_open).value();
  config.tool_call_close = std::move(tool_close).value();
  config.tool_calls_repeat = repeats->as_bool();
  return config;
}

Status ValidatePrimaryTokenizerConfig(const TokenizerConfig& config) {
  if (config.tokenizer_class != "GemmaTokenizer") {
    return ContractError("tokenizer_class", "GemmaTokenizer");
  }
  if (config.bos_token != "<bos>") {
    return ContractError("bos_token", "<bos>");
  }
  if (config.eos_token != "<eos>") {
    return ContractError("eos_token", "<eos>");
  }
  if (config.eot_token != "<turn|>") {
    return ContractError("eot_token", "<turn|>");
  }
  if (config.tool_response_end_token != "<tool_response|>" ||
      config.tool_call_start_token != "<|tool_call>") {
    return ContractError("tool boundary tokens", "the Google Gemma 4 contract");
  }
  if (config.model_max_length <=
      static_cast<double>(std::numeric_limits<std::uint64_t>::max())) {
    return ContractError(
        "model_max_length",
        "Google's unbounded tokenizer sentinel (model context comes from config.json)");
  }
  if (config.response_role != "assistant") {
    return ContractError("response_template.defaults.role", "assistant");
  }
  if (config.response_start_anchors !=
      std::vector<std::string>{"<|turn>model\n", "<tool_response|>"}) {
    return ContractError("response_template.start_anchor",
                         "the Google Gemma 4 anchors");
  }
  if (config.content_close_tokens !=
      std::vector<std::string>{"<turn|>", "<|tool_response>", "<eos>"}) {
    return ContractError("response_template.fields.content.close",
                         "the Google Gemma 4 close-token list");
  }
  if (config.thinking_open != "<|channel>thought\n" ||
      config.thinking_close != "<channel|>") {
    return ContractError("response_template.fields.thinking",
                         "the Google Gemma 4 thinking delimiters");
  }
  if (config.tool_call_open_pattern !=
          "<\\|tool_call>call:(?P<name>\\w+)" ||
      config.tool_call_close != "<tool_call|>" || !config.tool_calls_repeat) {
    return ContractError("response_template.fields.tool_calls",
                         "the repeatable Google Gemma 4 tool-call contract");
  }
  return Status::Ok();
}

Result<std::string> ExtractResponseContent(
    std::string_view text, std::string_view thinking_open,
    std::string_view thinking_close,
    std::span<const std::string> content_close_tokens,
    std::string_view tool_call_start_token) {
  std::string result;
  std::size_t begin = 0;
  while (begin <= text.size()) {
    const std::size_t open = text.find(thinking_open, begin);
    if (open == std::string_view::npos) {
      result.append(text.substr(begin));
      break;
    }
    result.append(text.substr(begin, open - begin));
    const std::size_t close =
        text.find(thinking_close, open + thinking_open.size());
    if (close == std::string_view::npos) {
      return Error(StatusCode::kDataLoss,
                   "model response has an unterminated thinking field");
    }
    begin = close + thinking_close.size();
  }
  std::size_t content_end = result.size();
  for (const std::string& close : content_close_tokens) {
    const std::size_t position = result.find(close);
    if (position != std::string::npos) {
      content_end = std::min(content_end, position);
    }
  }
  result.resize(content_end);
  if (result.find(tool_call_start_token) != std::string::npos) {
    return Error(StatusCode::kUnsupported,
                 "native chat does not yet support generated tool calls");
  }
  return Trim(result);
}

}  // namespace gem16gb::internal
