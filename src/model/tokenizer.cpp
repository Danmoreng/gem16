#include "gem16/tokenizer.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <ostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "model/tokenizer_config.h"
#include "model/config.h"
#include "util/json.h"

namespace gem16 {
namespace {

constexpr std::uint64_t kMaximumTokenizerBytes = 64U * 1024U * 1024U;
constexpr std::uint64_t kMaximumTemplateBytes = 1024U * 1024U;
constexpr std::uint64_t kPinnedTemplateFnv1a = 0xe9f262823e5bda06ULL;
constexpr std::uint64_t kPinnedMoe26BTemplateFnv1a = 0x645a5f8cd6ec0ad0ULL;
constexpr std::string_view kSpaceMarker = "\xE2\x96\x81";

Status Error(StatusCode code, std::string message) { return Status(code, std::move(message)); }

Result<std::string> ReadFile(const std::filesystem::path& path, std::uint64_t limit) {
  std::error_code error;
  const std::uint64_t size = std::filesystem::file_size(path, error);
  if (error) {
    return Error(StatusCode::kIoError, "cannot stat " + path.string() + ": " + error.message());
  }
  if (size > limit || size > std::numeric_limits<std::size_t>::max()) {
    return Error(StatusCode::kDataLoss, "file exceeds safety limit: " + path.string());
  }
  std::string contents(static_cast<std::size_t>(size), '\0');
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return Error(StatusCode::kIoError, "cannot open " + path.string());
  }
  input.read(contents.data(), static_cast<std::streamsize>(contents.size()));
  if (!input || input.gcount() != static_cast<std::streamsize>(contents.size())) {
    return Error(StatusCode::kIoError, "cannot read " + path.string());
  }
  return contents;
}

const json::Value* Member(const json::Value& value, std::string_view name) {
  return value.is_object() ? value.find(name) : nullptr;
}

const json::Value* Nested(const json::Value& value, std::string_view parent, std::string_view child) {
  const json::Value* object = Member(value, parent);
  return object == nullptr ? nullptr : Member(*object, child);
}

std::uint64_t PairKey(std::uint32_t left, std::uint32_t right) {
  return (static_cast<std::uint64_t>(left) << 32U) | right;
}

std::uint64_t Fnv1a(std::string_view value) {
  std::uint64_t hash = 14695981039346656037ULL;
  for (const unsigned char byte : value) {
    hash ^= byte;
    hash *= 1099511628211ULL;
  }
  return hash;
}

std::string Trim(std::string_view value) {
  std::size_t begin = 0;
  std::size_t end = value.size();
  while (begin < end && std::isspace(static_cast<unsigned char>(value[begin])) != 0) ++begin;
  while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1U])) != 0) --end;
  return std::string(value.substr(begin, end - begin));
}

std::string Upper(std::string_view value) {
  std::string result(value);
  std::transform(result.begin(), result.end(), result.begin(),
                 [](unsigned char character) { return static_cast<char>(std::toupper(character)); });
  return result;
}

Result<std::string> FormatToolArgument(const json::Value& value, bool escape_keys = true);

Result<std::string> FormatToolObject(const json::Value::Object& object, bool escape_keys) {
  std::string result = "{";
  bool first = true;
  for (const auto& [key, value] : object) {
    auto formatted = FormatToolArgument(value, escape_keys);
    if (!formatted.ok()) return formatted.status();
    if (!first) result.push_back(',');
    first = false;
    if (escape_keys) {
      result.append("<|\"|>");
      result.append(key);
      result.append("<|\"|>");
    } else {
      result.append(key);
    }
    result.push_back(':');
    result.append(formatted.value());
  }
  result.push_back('}');
  return result;
}

Result<std::string> FormatToolArgument(const json::Value& value, bool escape_keys) {
  if (value.is_null()) return std::string("null");
  if (value.is_bool()) return std::string(value.as_bool() ? "true" : "false");
  if (value.is_string()) {
    return std::string("<|\"|>") + value.as_string() + "<|\"|>";
  }
  if (value.is_integer()) return std::to_string(value.as_integer());
  if (value.is_number()) {
    std::array<char, 64> buffer{};
    const auto converted = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value.as_number());
    if (converted.ec != std::errc{}) {
      return Error(StatusCode::kDataLoss, "tool schema contains an unformattable number");
    }
    return std::string(buffer.data(), converted.ptr);
  }
  if (value.is_object()) return FormatToolObject(value.as_object(), escape_keys);
  std::string result = "[";
  bool first = true;
  for (const json::Value& item : value.as_array()) {
    auto formatted = FormatToolArgument(item, escape_keys);
    if (!formatted.ok()) return formatted.status();
    if (!first) result.push_back(',');
    first = false;
    result.append(formatted.value());
  }
  result.push_back(']');
  return result;
}

Result<std::string> FormatRequired(const json::Value& required) {
  if (!required.is_array()) {
    return Error(StatusCode::kInvalidArgument, "tool parameter required must be an array");
  }
  std::string result = "required:[";
  bool first = true;
  for (const json::Value& item : required.as_array()) {
    if (!item.is_string()) {
      return Error(StatusCode::kInvalidArgument, "tool parameter required entries must be strings");
    }
    if (!first) result.push_back(',');
    first = false;
    result.append("<|\"|>");
    result.append(item.as_string());
    result.append("<|\"|>");
  }
  result.push_back(']');
  return result;
}

Result<std::string> FormatSchemaProperties(const json::Value::Object& properties) {
  std::string result;
  bool first_property = true;
  for (const auto& [name, property] : properties) {
    if (!property.is_object()) {
      return Error(StatusCode::kInvalidArgument, "each tool property must be an object");
    }
    const json::Value* type = property.find("type");
    if (type == nullptr || !type->is_string()) {
      return Error(StatusCode::kInvalidArgument, "each tool property requires a string type");
    }
    if (!first_property) result.push_back(',');
    first_property = false;
    result.append(name);
    result.append(":{");
    bool has_field = false;
    auto append_field = [&](std::string_view key, std::string_view value) {
      if (has_field) result.push_back(',');
      has_field = true;
      result.append(key);
      result.append(value);
    };

    const json::Value* description = property.find("description");
    if (description != nullptr && description->is_string() && !description->as_string().empty()) {
      append_field("description:", std::string("<|\"|>") + description->as_string() + "<|\"|>");
    }
    const std::string upper_type = Upper(type->as_string());
    const json::Value* enumeration = property.find("enum");
    if (enumeration != nullptr) {
      auto formatted = FormatToolArgument(*enumeration);
      if (!formatted.ok()) return formatted.status();
      append_field("enum:", formatted.value());
    }
    if (upper_type == "ARRAY") {
      const json::Value* items = property.find("items");
      if (items != nullptr && items->is_object() && !items->as_object().empty()) {
        std::string item_result = "{";
        bool first_item = true;
        for (const auto& [item_key, item_value] : items->as_object()) {
          if (!first_item) item_result.push_back(',');
          first_item = false;
          item_result.append(item_key);
          item_result.push_back(':');
          if (item_key == "properties") {
            if (!item_value.is_object()) {
              return Error(StatusCode::kInvalidArgument, "array item properties must be an object");
            }
            auto nested = FormatSchemaProperties(item_value.as_object());
            if (!nested.ok()) return nested.status();
            item_result.push_back('{');
            item_result.append(nested.value());
            item_result.push_back('}');
          } else if (item_key == "required") {
            auto required = FormatRequired(item_value);
            if (!required.ok()) return required.status();
            item_result.append(required.value().substr(std::string_view("required:").size()));
          } else if (item_key == "type" && item_value.is_string()) {
            item_result.append("<|\"|>");
            item_result.append(Upper(item_value.as_string()));
            item_result.append("<|\"|>");
          } else {
            auto formatted = FormatToolArgument(item_value);
            if (!formatted.ok()) return formatted.status();
            item_result.append(formatted.value());
          }
        }
        item_result.push_back('}');
        append_field("items:", item_result);
      }
    }
    const json::Value* nullable = property.find("nullable");
    if (nullable != nullptr && nullable->is_bool() && nullable->as_bool()) {
      append_field("nullable:", "true");
    }
    if (upper_type == "OBJECT") {
      const json::Value* nested_properties = property.find("properties");
      if (nested_properties != nullptr && nested_properties->is_object()) {
        auto nested = FormatSchemaProperties(nested_properties->as_object());
        if (!nested.ok()) return nested.status();
        append_field("properties:", "{" + nested.value() + "}");
      }
      const json::Value* nested_required = property.find("required");
      if (nested_required != nullptr && nested_required->is_array() && !nested_required->as_array().empty()) {
        auto required = FormatRequired(*nested_required);
        if (!required.ok()) return required.status();
        append_field("", required.value());
      }
    }
    for (const auto& [key, value] : property.as_object()) {
      if (key == "description" || key == "enum" || key == "items" ||
          key == "nullable" || key == "properties" || key == "required" ||
          key == "type") {
        continue;
      }
      auto formatted = FormatToolArgument(value);
      if (!formatted.ok()) return formatted.status();
      append_field(key + ":", formatted.value());
    }
    append_field("type:", "<|\"|>" + upper_type + "<|\"|>");
    result.push_back('}');
  }
  return result;
}

Result<std::string> FormatToolParameters(const json::Value& parameters) {
  if (!parameters.is_object()) {
    return Error(StatusCode::kInvalidArgument, "tool parameters must be a JSON object");
  }
  const json::Value* type = parameters.find("type");
  if (type == nullptr || !type->is_string()) {
    return Error(StatusCode::kInvalidArgument, "tool parameters require a string type");
  }
  std::string result = "parameters:{";
  const json::Value* properties = parameters.find("properties");
  if (properties != nullptr && properties->is_object() && !properties->as_object().empty()) {
    auto formatted = FormatSchemaProperties(properties->as_object());
    if (!formatted.ok()) return formatted.status();
    result.append("properties:{");
    result.append(formatted.value());
    result.append("},");
  }
  const json::Value* required = parameters.find("required");
  if (required != nullptr && required->is_array() && !required->as_array().empty()) {
    auto formatted = FormatRequired(*required);
    if (!formatted.ok()) return formatted.status();
    result.append(formatted.value());
    result.push_back(',');
  }
  for (const auto& [key, value] : parameters.as_object()) {
    if (key == "properties" || key == "required" || key == "type") {
      continue;
    }
    auto formatted = FormatToolArgument(value);
    if (!formatted.ok()) return formatted.status();
    result.append(key);
    result.push_back(':');
    result.append(formatted.value());
    result.push_back(',');
  }
  result.append("type:<|\"|>");
  result.append(Upper(type->as_string()));
  result.append("<|\"|>}");
  return result;
}

Result<std::string> FormatToolDefinition(const ChatToolDefinition& tool) {
  if (tool.name.empty()) {
    return Error(StatusCode::kInvalidArgument, "tool name must not be empty");
  }
  auto parameters =
      json::Parse(tool.parameters_json, {.max_depth = 32, .max_values = 10000, .max_string_bytes = 1024U * 1024U});
  if (!parameters.ok()) {
    return Error(StatusCode::kInvalidArgument, "tool parameters are not valid JSON: " + parameters.status().message());
  }
  auto formatted_parameters = FormatToolParameters(parameters.value());
  if (!formatted_parameters.ok()) return formatted_parameters.status();
  std::string result = "<|tool>declaration:";
  result.append(tool.name);
  result.append("{description:<|\"|>");
  result.append(tool.description);
  result.append("<|\"|>,");
  result.append(formatted_parameters.value());
  result.append("}<tool|>");
  return result;
}

Result<std::string> FormatToolCall(const ChatMessage::ToolCall& call) {
  auto arguments =
      json::Parse(call.arguments_json, {.max_depth = 32, .max_values = 10000, .max_string_bytes = 1024U * 1024U});
  if (!arguments.ok() || !arguments.value().is_object()) {
    return Error(StatusCode::kInvalidArgument, "tool call arguments must be a JSON object");
  }
  auto formatted = FormatToolObject(arguments.value().as_object(), false);
  if (!formatted.ok()) return formatted.status();
  return std::string("<|tool_call>call:") + call.name + formatted.value() + "<tool_call|>";
}

std::string FormatToolResponse(std::string_view name, std::string_view output) {
  return std::string("<|tool_response>response:") + std::string(name) + "{value:<|\"|>" + std::string(output) +
         "<|\"|>}<tool_response|>";
}

Result<std::vector<std::string_view>> Utf8Characters(std::string_view text) {
  std::vector<std::string_view> result;
  result.reserve(text.size());
  std::size_t offset = 0;
  while (offset < text.size()) {
    const unsigned char first = static_cast<unsigned char>(text[offset]);
    std::size_t length = 0;
    if (first < 0x80U) {
      length = 1;
    } else if ((first & 0xE0U) == 0xC0U) {
      length = 2;
    } else if ((first & 0xF0U) == 0xE0U) {
      length = 3;
    } else if ((first & 0xF8U) == 0xF0U) {
      length = 4;
    } else {
      return Error(StatusCode::kInvalidArgument, "text contains invalid UTF-8");
    }
    if (length > text.size() - offset) {
      return Error(StatusCode::kInvalidArgument, "text contains truncated UTF-8");
    }
    for (std::size_t index = 1; index < length; ++index) {
      if ((static_cast<unsigned char>(text[offset + index]) & 0xC0U) != 0x80U) {
        return Error(StatusCode::kInvalidArgument, "text contains invalid UTF-8 continuation");
      }
    }
    result.push_back(text.substr(offset, length));
    offset += length;
  }
  return result;
}

bool ParseByteFallback(std::string_view token, unsigned char& value) {
  if (token.size() != 6U || !token.starts_with("<0x") || token.back() != '>') return false;
  unsigned parsed = 0;
  const auto conversion = std::from_chars(token.data() + 3, token.data() + 5, parsed, 16);
  if (conversion.ec != std::errc{} || conversion.ptr != token.data() + 5 || parsed > 0xFFU) {
    return false;
  }
  value = static_cast<unsigned char>(parsed);
  return true;
}

Result<std::vector<std::uint32_t>> IntegerList(const json::Value* value, std::string_view field, bool allow_scalar) {
  std::vector<std::uint32_t> result;
  if (allow_scalar && value != nullptr && value->is_integer()) {
    if (value->as_integer() < 0 ||
        static_cast<std::uint64_t>(value->as_integer()) > std::numeric_limits<std::uint32_t>::max()) {
      return Error(StatusCode::kDataLoss, "generation_config.json has invalid " + std::string(field));
    }
    result.push_back(static_cast<std::uint32_t>(value->as_integer()));
    return result;
  }
  if (value == nullptr || !value->is_array()) {
    return Error(StatusCode::kDataLoss, "generation_config.json has invalid " + std::string(field));
  }
  for (const auto& item : value->as_array()) {
    if (!item.is_integer() || item.as_integer() < 0 ||
        static_cast<std::uint64_t>(item.as_integer()) > std::numeric_limits<std::uint32_t>::max()) {
      return Error(StatusCode::kDataLoss, "generation_config.json has invalid " + std::string(field));
    }
    result.push_back(static_cast<std::uint32_t>(item.as_integer()));
  }
  if (result.empty() && field == "eos_token_id") {
    return Error(StatusCode::kDataLoss, "generation_config.json has no EOS token");
  }
  return result;
}

}  // namespace

namespace internal {

Result<std::string> RenderGemmaToolDefinition(std::string_view name, std::string_view description,
                                              std::string_view parameters_json) {
  return FormatToolDefinition(
      ChatToolDefinition{std::string(name), std::string(description), std::string(parameters_json)});
}

Result<std::string> RenderGemmaToolCall(std::string_view name, std::string_view arguments_json) {
  return FormatToolCall(ChatMessage::ToolCall{{}, std::string(name), std::string(arguments_json)});
}

}  // namespace internal

struct Tokenizer::Impl {
  struct Merge {
    std::uint32_t rank = 0;
    std::uint32_t result = 0;
  };
  struct AddedToken {
    std::string content;
    std::uint32_t id = 0;
  };

  std::unordered_map<std::string, std::uint32_t> vocabulary;
  std::vector<std::string> tokens;
  std::unordered_map<std::uint64_t, Merge> merges;
  std::vector<AddedToken> added_tokens;
  std::unordered_set<std::uint32_t> special_ids;
  std::size_t maximum_decoded_token_bytes = 0U;

  [[nodiscard]] Result<std::vector<std::uint32_t>> EncodeOrdinary(std::string_view ordinary) const {
    std::string normalized;
    normalized.reserve(ordinary.size());
    for (const char byte : ordinary) {
      if (byte == ' ') {
        normalized.append(kSpaceMarker);
      } else {
        normalized.push_back(byte);
      }
    }
    auto characters = Utf8Characters(normalized);
    if (!characters.ok()) return characters.status();

    std::vector<std::uint32_t> symbols;
    symbols.reserve(characters.value().size());
    for (const std::string_view character : characters.value()) {
      const auto found = vocabulary.find(std::string(character));
      if (found != vocabulary.end()) {
        symbols.push_back(found->second);
        continue;
      }
      for (const unsigned char byte : character) {
        std::array<char, 7> fallback{};
        (void)std::snprintf(fallback.data(), fallback.size(), "<0x%02X>", byte);
        const auto byte_token = vocabulary.find(fallback.data());
        if (byte_token == vocabulary.end()) {
          return Error(StatusCode::kDataLoss, "tokenizer byte fallback is incomplete");
        }
        symbols.push_back(byte_token->second);
      }
    }

    while (symbols.size() > 1U) {
      std::uint32_t best_rank = std::numeric_limits<std::uint32_t>::max();
      std::uint64_t best_pair = 0;
      bool found_pair = false;
      for (std::size_t index = 0; index + 1U < symbols.size(); ++index) {
        const std::uint64_t key = PairKey(symbols[index], symbols[index + 1U]);
        const auto merge = merges.find(key);
        if (merge != merges.end() && merge->second.rank < best_rank) {
          best_rank = merge->second.rank;
          best_pair = key;
          found_pair = true;
        }
      }
      if (!found_pair) break;
      const auto merge = merges.find(best_pair);
      std::vector<std::uint32_t> next;
      next.reserve(symbols.size());
      for (std::size_t index = 0; index < symbols.size();) {
        if (index + 1U < symbols.size() && PairKey(symbols[index], symbols[index + 1U]) == best_pair) {
          next.push_back(merge->second.result);
          index += 2U;
        } else {
          next.push_back(symbols[index]);
          ++index;
        }
      }
      symbols = std::move(next);
    }
    return symbols;
  }
};

Result<Tokenizer> Tokenizer::Load(const std::filesystem::path& tokenizer_json) {
  auto text = ReadFile(tokenizer_json, kMaximumTokenizerBytes);
  if (!text.ok()) return text.status();
  auto parsed =
      json::Parse(text.value(), {.max_depth = 128, .max_values = 2'000'000, .max_string_bytes = 256U * 1024U * 1024U});
  if (!parsed.ok()) {
    return Error(parsed.status().code(), tokenizer_json.string() + ": " + parsed.status().message());
  }
  const json::Value* model = Member(parsed.value(), "model");
  const json::Value* vocabulary = model == nullptr ? nullptr : Member(*model, "vocab");
  const json::Value* merges = model == nullptr ? nullptr : Member(*model, "merges");
  const json::Value* model_type = model == nullptr ? nullptr : Member(*model, "type");
  const json::Value* byte_fallback = model == nullptr ? nullptr : Member(*model, "byte_fallback");
  if (model == nullptr || !model->is_object() || model_type == nullptr || !model_type->is_string() ||
      model_type->as_string() != "BPE" || byte_fallback == nullptr || !byte_fallback->is_bool() ||
      !byte_fallback->as_bool() || vocabulary == nullptr || !vocabulary->is_object() || merges == nullptr ||
      !merges->is_array()) {
    return Error(StatusCode::kUnsupported, "tokenizer must be a byte-fallback BPE tokenizer");
  }

  auto implementation = std::make_shared<Impl>();
  std::uint32_t maximum_id = 0;
  for (const auto& [token, id_value] : vocabulary->as_object()) {
    if (!id_value.is_integer() || id_value.as_integer() < 0 ||
        static_cast<std::uint64_t>(id_value.as_integer()) > std::numeric_limits<std::uint32_t>::max()) {
      return Error(StatusCode::kDataLoss, "tokenizer vocabulary has an invalid ID");
    }
    maximum_id = std::max(maximum_id, static_cast<std::uint32_t>(id_value.as_integer()));
  }
  implementation->tokens.resize(static_cast<std::size_t>(maximum_id) + 1U);
  for (const auto& [token, id_value] : vocabulary->as_object()) {
    const auto id = static_cast<std::uint32_t>(id_value.as_integer());
    if (!implementation->tokens[id].empty()) {
      return Error(StatusCode::kDataLoss, "tokenizer vocabulary has duplicate IDs");
    }
    implementation->tokens[id] = token;
    implementation->vocabulary.emplace(token, id);
    implementation->maximum_decoded_token_bytes =
        std::max(implementation->maximum_decoded_token_bytes, token.size());
  }

  std::uint32_t rank = 0;
  implementation->merges.reserve(merges->as_array().size());
  for (const auto& entry : merges->as_array()) {
    if (!entry.is_array() || entry.as_array().size() != 2U || !entry.as_array()[0].is_string() ||
        !entry.as_array()[1].is_string()) {
      return Error(StatusCode::kDataLoss, "tokenizer merge is malformed");
    }
    const std::string& left_text = entry.as_array()[0].as_string();
    const std::string& right_text = entry.as_array()[1].as_string();
    const auto left = implementation->vocabulary.find(left_text);
    const auto right = implementation->vocabulary.find(right_text);
    const auto result = implementation->vocabulary.find(left_text + right_text);
    if (left == implementation->vocabulary.end() || right == implementation->vocabulary.end() ||
        result == implementation->vocabulary.end()) {
      return Error(StatusCode::kDataLoss, "tokenizer merge references an absent vocabulary token");
    }
    implementation->merges.emplace(PairKey(left->second, right->second), Impl::Merge{rank++, result->second});
  }

  const json::Value* added = Member(parsed.value(), "added_tokens");
  if (added == nullptr || !added->is_array()) {
    return Error(StatusCode::kDataLoss, "tokenizer has no added_tokens array");
  }
  for (const auto& entry : added->as_array()) {
    const json::Value* content = Member(entry, "content");
    const json::Value* id_value = Member(entry, "id");
    const json::Value* special = Member(entry, "special");
    if (content == nullptr || !content->is_string() || id_value == nullptr || !id_value->is_integer() ||
        id_value->as_integer() < 0 ||
        static_cast<std::uint64_t>(id_value->as_integer()) > std::numeric_limits<std::uint32_t>::max() ||
        special == nullptr || !special->is_bool()) {
      return Error(StatusCode::kDataLoss, "tokenizer added token is malformed");
    }
    const auto id = static_cast<std::uint32_t>(id_value->as_integer());
    implementation->added_tokens.push_back({content->as_string(), id});
    if (special->as_bool()) implementation->special_ids.insert(id);
  }
  std::sort(implementation->added_tokens.begin(), implementation->added_tokens.end(),
            [](const Impl::AddedToken& left, const Impl::AddedToken& right) {
              return left.content.size() > right.content.size();
            });
  return Tokenizer(std::move(implementation));
}

Result<std::vector<std::uint32_t>> Tokenizer::Encode(std::string_view text) const {
  if (implementation_ == nullptr) {
    return Error(StatusCode::kInternal, "tokenizer is not initialized");
  }
  std::vector<std::uint32_t> result;
  std::size_t ordinary_begin = 0;
  std::size_t offset = 0;
  while (offset < text.size()) {
    const Impl::AddedToken* matched = nullptr;
    for (const auto& added : implementation_->added_tokens) {
      if (text.substr(offset).starts_with(added.content)) {
        matched = &added;
        break;
      }
    }
    if (matched == nullptr) {
      ++offset;
      continue;
    }
    if (offset > ordinary_begin) {
      auto ordinary = implementation_->EncodeOrdinary(text.substr(ordinary_begin, offset - ordinary_begin));
      if (!ordinary.ok()) return ordinary.status();
      result.insert(result.end(), ordinary.value().begin(), ordinary.value().end());
    }
    result.push_back(matched->id);
    offset += matched->content.size();
    ordinary_begin = offset;
  }
  if (ordinary_begin < text.size()) {
    auto ordinary = implementation_->EncodeOrdinary(text.substr(ordinary_begin));
    if (!ordinary.ok()) return ordinary.status();
    result.insert(result.end(), ordinary.value().begin(), ordinary.value().end());
  }
  return result;
}

Result<std::string> Tokenizer::Decode(std::span<const std::uint32_t> token_ids, bool skip_special_tokens) const {
  if (implementation_ == nullptr) {
    return Error(StatusCode::kInternal, "tokenizer is not initialized");
  }
  std::string result;
  for (const std::uint32_t id : token_ids) {
    if (id >= implementation_->tokens.size() || implementation_->tokens[id].empty()) {
      return Error(StatusCode::kInvalidArgument, "token ID is absent from tokenizer vocabulary");
    }
    if (skip_special_tokens && implementation_->special_ids.contains(id)) continue;
    const std::string& token = implementation_->tokens[id];
    if (implementation_->special_ids.contains(id)) {
      result.append(token);
      continue;
    }
    unsigned char fallback = 0;
    if (ParseByteFallback(token, fallback)) {
      result.push_back(static_cast<char>(fallback));
      continue;
    }
    std::size_t begin = 0;
    while (begin < token.size()) {
      const std::size_t marker = token.find(kSpaceMarker, begin);
      if (marker == std::string::npos) {
        result.append(token.substr(begin));
        break;
      }
      result.append(token.substr(begin, marker - begin));
      result.push_back(' ');
      begin = marker + kSpaceMarker.size();
    }
  }
  return result;
}

Status Tokenizer::DecodeTokenInto(std::uint32_t token_id,
                                  bool skip_special_tokens,
                                  std::span<char> output,
                                  std::size_t& written) const {
  written = 0U;
  if (implementation_ == nullptr) {
    return Error(StatusCode::kInternal, "tokenizer is not initialized");
  }
  if (token_id >= implementation_->tokens.size() ||
      implementation_->tokens[token_id].empty()) {
    return Error(StatusCode::kInvalidArgument,
                 "token ID is absent from tokenizer vocabulary");
  }
  const bool special = implementation_->special_ids.contains(token_id);
  if (skip_special_tokens && special) return Status::Ok();

  const auto append = [&](std::string_view bytes) {
    if (bytes.size() > output.size() - written) return false;
    for (const char byte : bytes) output[written++] = byte;
    return true;
  };
  const std::string& token = implementation_->tokens[token_id];
  if (special) {
    if (!append(token)) {
      return Error(StatusCode::kResourceExhausted,
                   "decoded-token buffer is too small");
    }
    return Status::Ok();
  }
  unsigned char fallback = 0U;
  if (ParseByteFallback(token, fallback)) {
    if (output.empty()) {
      return Error(StatusCode::kResourceExhausted,
                   "decoded-token buffer is too small");
    }
    output[written++] = static_cast<char>(fallback);
    return Status::Ok();
  }
  std::size_t begin = 0U;
  while (begin < token.size()) {
    const std::size_t marker = token.find(kSpaceMarker, begin);
    if (marker == std::string::npos) {
      if (!append(std::string_view(token).substr(begin))) {
        return Error(StatusCode::kResourceExhausted,
                     "decoded-token buffer is too small");
      }
      break;
    }
    if (!append(std::string_view(token).substr(begin, marker - begin)) ||
        !append(" ")) {
      return Error(StatusCode::kResourceExhausted,
                   "decoded-token buffer is too small");
    }
    begin = marker + kSpaceMarker.size();
  }
  return Status::Ok();
}

std::size_t Tokenizer::maximum_decoded_token_bytes() const {
  return implementation_ == nullptr
             ? 0U
             : implementation_->maximum_decoded_token_bytes;
}

Status Tokenizer::WriteDecodedToken(std::uint32_t token_id, bool skip_special_tokens, std::ostream& output) const {
  if (implementation_ == nullptr) {
    return Error(StatusCode::kInternal, "tokenizer is not initialized");
  }
  if (token_id >= implementation_->tokens.size() || implementation_->tokens[token_id].empty()) {
    return Error(StatusCode::kInvalidArgument, "token ID is absent from tokenizer vocabulary");
  }
  const bool special = implementation_->special_ids.contains(token_id);
  if (skip_special_tokens && special) return Status::Ok();

  const std::string& token = implementation_->tokens[token_id];
  if (special) {
    output.write(token.data(), static_cast<std::streamsize>(token.size()));
  } else {
    unsigned char fallback = 0;
    if (ParseByteFallback(token, fallback)) {
      output.put(static_cast<char>(fallback));
    } else {
      std::size_t begin = 0;
      while (begin < token.size()) {
        const std::size_t marker = token.find(kSpaceMarker, begin);
        if (marker == std::string::npos) {
          output.write(token.data() + begin, static_cast<std::streamsize>(token.size() - begin));
          break;
        }
        output.write(token.data() + begin, static_cast<std::streamsize>(marker - begin));
        output.put(' ');
        begin = marker + kSpaceMarker.size();
      }
    }
  }
  if (!output) {
    return Error(StatusCode::kIoError, "failed to write decoded token");
  }
  return Status::Ok();
}

Result<GemmaChatProcessor> GemmaChatProcessor::Load(const std::filesystem::path& model_directory) {
  auto model_config = internal::LoadModelConfig(model_directory / "config.json");
  if (!model_config.ok()) return model_config.status();
  const bool text_only_moe26b = internal::IsGemma4Moe26BModel(model_config.value());
  if (text_only_moe26b) {
    Status contract = internal::ValidateGemma4Moe26BContract(model_config.value());
    if (!contract.ok()) return contract;
  } else {
    auto processor_text = ReadFile(model_directory / "processor_config.json", 1024U * 1024U);
    if (!processor_text.ok()) return processor_text.status();
    auto processor = json::Parse(processor_text.value());
    if (!processor.ok() || !processor.value().is_object()) {
      return Error(StatusCode::kDataLoss, "processor_config.json is malformed");
    }
    const json::Value* sampling_rate = Nested(processor.value(), "feature_extractor", "sampling_rate");
    const json::Value* samples_per_token = Nested(processor.value(), "feature_extractor", "audio_samples_per_token");
    const json::Value* feature_size = Nested(processor.value(), "feature_extractor", "feature_size");
    const json::Value* sequence_length = Member(processor.value(), "audio_seq_length");
    const json::Value* milliseconds_per_token = Member(processor.value(), "audio_ms_per_token");
    const json::Value* image_patch_size = Nested(processor.value(), "image_processor", "patch_size");
    const json::Value* image_pooling = Nested(processor.value(), "image_processor", "pooling_kernel_size");
    const json::Value* image_tokens = Member(processor.value(), "image_seq_length");
    if (sampling_rate == nullptr || !sampling_rate->is_integer() || sampling_rate->as_integer() != 16000 ||
        samples_per_token == nullptr || !samples_per_token->is_integer() || samples_per_token->as_integer() != 640 ||
        feature_size == nullptr || !feature_size->is_integer() || feature_size->as_integer() != 640 ||
        sequence_length == nullptr || !sequence_length->is_integer() || sequence_length->as_integer() != 750 ||
        milliseconds_per_token == nullptr || !milliseconds_per_token->is_integer() ||
        milliseconds_per_token->as_integer() != 40 || image_patch_size == nullptr || !image_patch_size->is_integer() ||
        image_patch_size->as_integer() != 16 || image_pooling == nullptr || !image_pooling->is_integer() ||
        image_pooling->as_integer() != 3 || image_tokens == nullptr || !image_tokens->is_integer() ||
        image_tokens->as_integer() != 280) {
      return Error(StatusCode::kUnsupported, "processor_config.json differs from the qualified audio/vision schema");
    }
  }
  auto tokenizer = Tokenizer::Load(model_directory / "tokenizer.json");
  if (!tokenizer.ok()) return tokenizer.status();
  auto tokenizer_config = internal::LoadTokenizerConfig(model_directory / "tokenizer_config.json");
  if (!tokenizer_config.ok()) return tokenizer_config.status();
  auto tokenizer_contract = internal::ValidatePrimaryTokenizerConfig(tokenizer_config.value());
  if (!tokenizer_contract.ok()) return tokenizer_contract;
  auto chat_template = ReadFile(model_directory / "chat_template.jinja", kMaximumTemplateBytes);
  if (!chat_template.ok()) return chat_template.status();
  const std::uint64_t expected_template =
      text_only_moe26b ? kPinnedMoe26BTemplateFnv1a : kPinnedTemplateFnv1a;
  if (Fnv1a(chat_template.value()) != expected_template) {
    return Error(StatusCode::kUnsupported,
                 "chat_template.jinja differs from the natively supported pinned Gemma template");
  }

  auto generation_text = ReadFile(model_directory / "generation_config.json", 1024U * 1024U);
  if (!generation_text.ok()) return generation_text.status();
  auto generation = json::Parse(generation_text.value());
  if (!generation.ok() || !generation.value().is_object()) {
    return Error(StatusCode::kDataLoss, "generation_config.json is malformed");
  }
  auto stop = IntegerList(Member(generation.value(), "eos_token_id"), "eos_token_id", true);
  if (!stop.ok()) return stop.status();
  const json::Value* do_sample = Member(generation.value(), "do_sample");
  const json::Value* temperature = Member(generation.value(), "temperature");
  const json::Value* top_k = Member(generation.value(), "top_k");
  const json::Value* top_p = Member(generation.value(), "top_p");
  if (do_sample == nullptr || !do_sample->is_bool() || !do_sample->as_bool() || temperature == nullptr ||
      !temperature->is_number() || !std::isfinite(temperature->as_number()) || temperature->as_number() <= 0.0 ||
      top_k == nullptr || !top_k->is_integer() || top_k->as_integer() < 0 ||
      static_cast<std::uint64_t>(top_k->as_integer()) > std::numeric_limits<std::uint32_t>::max() || top_p == nullptr ||
      !top_p->is_number() || !std::isfinite(top_p->as_number()) || top_p->as_number() <= 0.0 ||
      top_p->as_number() > 1.0) {
    return Error(StatusCode::kDataLoss, "generation_config.json has invalid sampling defaults");
  }
  SamplingOptions recommended_sampling;
  recommended_sampling.enabled = true;
  recommended_sampling.temperature = static_cast<float>(temperature->as_number());
  recommended_sampling.top_k = static_cast<std::uint32_t>(top_k->as_integer());
  recommended_sampling.top_p = static_cast<float>(top_p->as_number());
  auto sampling_status = ValidateSamplingOptions(recommended_sampling, 262144U);
  if (!sampling_status.ok()) return sampling_status;

  const json::Value* suppressed_value = Member(generation.value(), "suppress_tokens");
  std::vector<std::uint32_t> suppressed;
  if (suppressed_value != nullptr) {
    auto parsed_suppressed = IntegerList(suppressed_value, "suppress_tokens", false);
    if (!parsed_suppressed.ok()) return parsed_suppressed.status();
    suppressed = std::move(parsed_suppressed).value();
  }

  std::string thinking_open_marker = tokenizer_config.value().thinking_open;
  while (!thinking_open_marker.empty() && std::isspace(static_cast<unsigned char>(thinking_open_marker.back()))) {
    thinking_open_marker.pop_back();
  }
  auto thinking_open = tokenizer.value().Encode(thinking_open_marker);
  if (!thinking_open.ok()) return thinking_open.status();
  if (thinking_open.value().empty()) {
    return Error(StatusCode::kDataLoss, "tokenizer thinking channel opener is empty");
  }
  auto thinking_close = tokenizer.value().Encode(tokenizer_config.value().thinking_close);
  if (!thinking_close.ok()) return thinking_close.status();
  if (thinking_close.value().size() != 1U) {
    return Error(StatusCode::kDataLoss, "tokenizer thinking channel close marker is not one token");
  }

  for (const std::string& token : tokenizer_config.value().content_close_tokens) {
    auto encoded = tokenizer.value().Encode(token);
    if (!encoded.ok()) return encoded.status();
    if (encoded.value().size() != 1U) {
      return Error(StatusCode::kDataLoss, "tokenizer_config.json response close marker is not one token: " + token);
    }
    if (std::find(stop.value().begin(), stop.value().end(), encoded.value().front()) == stop.value().end()) {
      return Error(StatusCode::kDataLoss,
                   "generation_config.json does not stop on tokenizer response close marker: " + token);
    }
  }
  return GemmaChatProcessor(
      std::move(tokenizer).value(),
      GenerationTokenControls{std::move(stop).value(), std::move(suppressed), std::move(thinking_open).value(),
                              thinking_close.value().front(), recommended_sampling},
      tokenizer_config.value().thinking_open, tokenizer_config.value().thinking_close,
      tokenizer_config.value().content_close_tokens, tokenizer_config.value().tool_call_start_token);
}

ResponseTokenChannel ResponseChannelTracker::Observe(std::uint32_t token_id) {
  if (in_reasoning_) {
    if (token_id == thinking_close_token_id_) {
      in_reasoning_ = false;
      reasoning_complete_ = true;
      return ResponseTokenChannel::kControl;
    }
    ++reasoning_token_count_;
    return ResponseTokenChannel::kReasoning;
  }
  if (suppressing_additional_reasoning_) {
    if (token_id == thinking_close_token_id_) {
      suppressing_additional_reasoning_ = false;
    }
    return ResponseTokenChannel::kControl;
  }
  if (token_id == thinking_close_token_id_) {
    open_match_length_ = 0U;
    return ResponseTokenChannel::kControl;
  }
  if (thinking_open_token_ids_.empty()) return ResponseTokenChannel::kText;
  if (token_id == thinking_open_token_ids_[open_match_length_]) {
    ++open_match_length_;
    if (open_match_length_ == thinking_open_token_ids_.size()) {
      open_match_length_ = 0U;
      if (reasoning_complete_ && suppress_additional_reasoning_fields_) {
        suppressing_additional_reasoning_ = true;
      } else {
        in_reasoning_ = true;
      }
    }
    return ResponseTokenChannel::kControl;
  }
  open_match_length_ = token_id == thinking_open_token_ids_.front() ? 1U : 0U;
  return open_match_length_ == 0U ? ResponseTokenChannel::kText
                                  : ResponseTokenChannel::kControl;
}

Result<std::string> GemmaChatProcessor::Render(std::span<const ChatMessage> messages, bool enable_thinking,
                                               bool add_generation_prompt,
                                               std::span<const ChatToolDefinition> tools) const {
  return internal::RenderGemmaChat(messages, enable_thinking, add_generation_prompt, tools,
                                   thinking_open_, thinking_close_, content_close_tokens_, tool_call_start_token_);
}

Result<std::string> internal::RenderGemmaChat(
    std::span<const ChatMessage> messages, bool enable_thinking, bool add_generation_prompt,
    std::span<const ChatToolDefinition> tools, std::string_view thinking_open,
    std::string_view thinking_close, std::span<const std::string> content_close_tokens,
    std::string_view tool_call_start_token) {
  if (messages.empty()) {
    return Error(StatusCode::kInvalidArgument, "chat requires at least one message");
  }
  std::string result = "<bos>";
  std::size_t message_index = 0;
  if (enable_thinking || !tools.empty() || messages.front().role == "system" || messages.front().role == "developer") {
    result.append("<|turn>system\n");
    if (enable_thinking) result.append("<|think|>\n");
    if (messages.front().role == "system" || messages.front().role == "developer") {
      result.append(Trim(messages.front().content));
      message_index = 1;
    }
    for (const ChatToolDefinition& tool : tools) {
      auto formatted = FormatToolDefinition(tool);
      if (!formatted.ok()) return formatted.status();
      result.append(formatted.value());
    }
    result.append("<turn|>\n");
  }

  std::string previous_role;
  for (; message_index < messages.size(); ++message_index) {
    const ChatMessage& message = messages[message_index];
    if (message.role == "tool") {
      return Error(StatusCode::kInvalidArgument, "tool results must immediately follow an assistant tool call");
    }
    if (message.role != "user" && message.role != "assistant") {
      return Error(StatusCode::kUnsupported, "native chat message role is unsupported");
    }
    if (message.role == previous_role) {
      return Error(StatusCode::kInvalidArgument, "native chat requires alternating user and assistant messages");
    }
    const bool continue_model_turn = message.role == "assistant" && previous_role == "tool";
    const std::string_view rendered_role = message.role == "assistant" ? "model" : "user";
    if (!continue_model_turn) {
      result.append("<|turn>");
      result.append(rendered_role);
      result.push_back('\n');
    }
    if (message.role == "assistant") {
      bool assistant_has_content = false;
      if (!message.tool_calls.empty()) {
        for (const ChatMessage::ToolCall& call : message.tool_calls) {
          auto formatted = FormatToolCall(call);
          if (!formatted.ok()) return formatted.status();
          result.append(formatted.value());
        }
      }
      if (!message.content.empty()) {
        auto content = internal::ExtractResponseContent(message.content, thinking_open, thinking_close,
                                                        content_close_tokens, tool_call_start_token);
        if (!content.ok()) return content.status();
        result.append(content.value());
        assistant_has_content = !content.value().empty();
      }
      bool rendered_tool_result = false;
      while (message_index + 1U < messages.size() && messages[message_index + 1U].role == "tool") {
        const ChatMessage& tool_result = messages[++message_index];
        std::string tool_name = tool_result.tool_name;
        if (tool_name.empty()) {
          for (const ChatMessage::ToolCall& call : message.tool_calls) {
            if (call.id == tool_result.tool_call_id) {
              tool_name = call.name;
              break;
            }
          }
        }
        if (tool_name.empty()) {
          return Error(StatusCode::kInvalidArgument, "tool result does not match an assistant tool call");
        }
        result.append(FormatToolResponse(tool_name, tool_result.content));
        rendered_tool_result = true;
      }
      if (!message.tool_calls.empty() && !rendered_tool_result) {
        result.append("<|tool_response>");
        previous_role = "assistant";
        continue;
      }
      // A tool result followed by another assistant message stays inside the
      // same model turn in both pinned Gemma templates. Only user turns close it.
      const bool next_is_assistant = message_index + 1U < messages.size() &&
                                     messages[message_index + 1U].role == "assistant";
      if (rendered_tool_result &&
          (next_is_assistant || (!assistant_has_content && message_index + 1U == messages.size()))) {
        previous_role = "tool";
        continue;
      }
    } else {
      result.append(Trim(message.content));
    }
    result.append("<turn|>\n");
    previous_role = message.role;
  }
  if (add_generation_prompt) {
    if (previous_role == "tool") {
      if (enable_thinking) result.append("<|channel>thought\n");
      return result;
    }
    if (previous_role != "user") {
      return Error(StatusCode::kInvalidArgument, "generation prompt requires a final user message");
    }
    result.append("<|turn>model\n");
    if (!enable_thinking) result.append("<|channel>thought\n<channel|>");
  }
  return result;
}

Result<std::vector<std::uint32_t>> GemmaChatProcessor::Encode(std::span<const ChatMessage> messages,
                                                              bool enable_thinking, bool add_generation_prompt,
                                                              std::span<const ChatToolDefinition> tools) const {
  auto rendered = Render(messages, enable_thinking, add_generation_prompt, tools);
  if (!rendered.ok()) return rendered.status();
  return tokenizer_.Encode(rendered.value());
}

Result<std::vector<std::uint32_t>> GemmaChatProcessor::EncodeContinuation(std::string_view user_content,
                                                                          bool enable_thinking) const {
  std::string rendered = "<turn|>\n<|turn>user\n";
  rendered.append(Trim(user_content));
  rendered.append("<turn|>\n<|turn>model\n");
  if (!enable_thinking) {
    rendered.append("<|channel>thought\n<channel|>");
  }
  return tokenizer_.Encode(rendered);
}

Result<std::vector<std::uint32_t>> GemmaChatProcessor::EncodeToolResultsContinuation(
    std::span<const ChatToolResult> results, bool enable_thinking) const {
  if (results.empty()) {
    return Error(StatusCode::kInvalidArgument,
                 "tool continuation requires at least one result");
  }
  std::string rendered;
  for (const ChatToolResult& result : results) {
    if (result.name.empty()) {
      return Error(StatusCode::kInvalidArgument,
                   "tool continuation requires resolved tool names");
    }
    rendered.append(FormatToolResponse(result.name, result.output));
  }
  if (enable_thinking) rendered.append("<|channel>thought\n");
  return tokenizer_.Encode(rendered);
}

Result<std::string> GemmaChatProcessor::Decode(std::span<const std::uint32_t> token_ids,
                                               bool skip_special_tokens) const {
  return tokenizer_.Decode(token_ids, skip_special_tokens);
}

Result<std::string> GemmaChatProcessor::DecodeResponseText(std::span<const std::uint32_t> token_ids) const {
  auto decoded = tokenizer_.Decode(token_ids, false);
  if (!decoded.ok()) return decoded.status();
  return internal::ExtractResponseContent(decoded.value(), thinking_open_, thinking_close_, content_close_tokens_,
                                          tool_call_start_token_);
}

Status GemmaChatProcessor::DecodeTokenInto(
    std::uint32_t token_id, bool skip_special_tokens, std::span<char> output,
    std::size_t& written) const {
  return tokenizer_.DecodeTokenInto(token_id, skip_special_tokens, output,
                                    written);
}

Status GemmaChatProcessor::WriteDecodedToken(std::uint32_t token_id, bool skip_special_tokens,
                                             std::ostream& output) const {
  return tokenizer_.WriteDecodedToken(token_id, skip_special_tokens, output);
}

}  // namespace gem16
