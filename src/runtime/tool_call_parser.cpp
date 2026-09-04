#include "runtime/tool_call_parser.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <optional>
#include <set>
#include <string>
#include <utility>

#include "util/json.h"

namespace gem16::internal {
namespace {

constexpr std::string_view kToolOpen = "<|tool_call>";
constexpr std::string_view kCallPrefix = "call:";
constexpr std::string_view kToolClose = "<tool_call|>";
constexpr std::string_view kQuote = "<|\"|>";

Status ParseError(std::string message) { return Status(StatusCode::kDataLoss, std::move(message)); }

Status InvalidTool(std::string message) {
  return Status(StatusCode::kInvalidArgument, std::move(message));
}

bool IsProtocolName(std::string_view name) {
  return !name.empty() && name.size() <= 64U &&
         std::all_of(name.begin(), name.end(), [](const char character) {
           return (character >= 'a' && character <= 'z') ||
                  (character >= 'A' && character <= 'Z') ||
                  (character >= '0' && character <= '9') || character == '_';
         });
}

std::optional<std::size_t> Utf8CodePointCount(std::string_view value) {
  std::size_t count = 0U;
  for (std::size_t index = 0U; index < value.size();) {
    const auto lead = static_cast<unsigned char>(value[index]);
    std::size_t width = 0U;
    if (lead <= 0x7FU) {
      width = 1U;
    } else if (lead >= 0xC2U && lead <= 0xDFU) {
      width = 2U;
    } else if (lead >= 0xE0U && lead <= 0xEFU) {
      width = 3U;
    } else if (lead >= 0xF0U && lead <= 0xF4U) {
      width = 4U;
    } else {
      return std::nullopt;
    }
    if (width > value.size() - index) return std::nullopt;
    for (std::size_t continuation = 1U; continuation < width;
         ++continuation) {
      const auto byte =
          static_cast<unsigned char>(value[index + continuation]);
      if ((byte & 0xC0U) != 0x80U) return std::nullopt;
    }
    if ((lead == 0xE0U &&
         static_cast<unsigned char>(value[index + 1U]) < 0xA0U) ||
        (lead == 0xEDU &&
         static_cast<unsigned char>(value[index + 1U]) >= 0xA0U) ||
        (lead == 0xF0U &&
         static_cast<unsigned char>(value[index + 1U]) < 0x90U) ||
        (lead == 0xF4U &&
         static_cast<unsigned char>(value[index + 1U]) >= 0x90U)) {
      return std::nullopt;
    }
    index += width;
    ++count;
  }
  return count;
}

bool IsSchemaType(std::string_view type) {
  return type == "null" || type == "boolean" || type == "object" ||
         type == "array" || type == "number" || type == "integer" ||
         type == "string";
}

Status ValidateSchemaDefinition(const json::Value& schema, bool strict,
                                std::string_view path,
                                std::uint32_t depth = 0U) {
  if (!schema.is_object()) {
    return InvalidTool(std::string(path) + " must be a JSON Schema object");
  }
  if (depth > 32U) {
    return InvalidTool(std::string(path) + " exceeds the schema depth limit");
  }
  static const std::set<std::string, std::less<>> kSupportedKeywords = {
      "$defs",          "$ref",       "additionalProperties",
      "allOf",          "anyOf",      "const",
      "default",        "description", "enum",
      "exclusiveMaximum", "exclusiveMinimum", "items",
      "maximum",        "maxItems",   "maxLength",
      "minimum",        "minItems",   "minLength",
      "oneOf",          "properties", "required",
      "title",          "type"};
  if (strict) {
    for (const auto& [keyword, value] : schema.as_object()) {
      (void)value;
      if (!kSupportedKeywords.contains(keyword)) {
        return Status(StatusCode::kUnsupported,
                      std::string(path) + " uses unsupported strict-schema keyword '" +
                          keyword + "'");
      }
    }
  }
  if (const json::Value* type = schema.find("type"); type != nullptr) {
    if (type->is_string()) {
      if (!IsSchemaType(type->as_string())) {
        return InvalidTool(std::string(path) + " has an unknown schema type");
      }
    } else if (type->is_array() && !type->as_array().empty()) {
      for (const json::Value& item : type->as_array()) {
        if (!item.is_string() || !IsSchemaType(item.as_string())) {
          return InvalidTool(std::string(path) +
                             " has an invalid schema type union");
        }
      }
    } else {
      return InvalidTool(std::string(path) +
                         " type must be a string or non-empty array");
    }
  }
  for (const std::string_view keyword : {"description", "title"}) {
    const json::Value* value = schema.find(keyword);
    if (value != nullptr && !value->is_string()) {
      return InvalidTool(std::string(path) + "." + std::string(keyword) +
                         " must be a string");
    }
  }
  if (const json::Value* properties = schema.find("properties");
      properties != nullptr) {
    if (!properties->is_object()) {
      return InvalidTool(std::string(path) + " properties must be an object");
    }
    for (const auto& [name, child] : properties->as_object()) {
      if (!IsProtocolName(name)) {
        return Status(StatusCode::kUnsupported,
                      std::string(path) + " property '" + name +
                          "' cannot be represented by the Gemma tool protocol");
      }
      const Status status = ValidateSchemaDefinition(
          child, strict, std::string(path) + ".properties." + name,
          depth + 1U);
      if (!status.ok()) return status;
    }
  }
  if (const json::Value* required = schema.find("required");
      required != nullptr) {
    if (!required->is_array()) {
      return InvalidTool(std::string(path) + " required must be an array");
    }
    std::set<std::string, std::less<>> required_names;
    for (const json::Value& name : required->as_array()) {
      if (!name.is_string() || !IsProtocolName(name.as_string())) {
        return InvalidTool(std::string(path) +
                           " required entries must be protocol-safe names");
      }
      if (!required_names.insert(name.as_string()).second) {
        return InvalidTool(std::string(path) +
                           " required entries must be unique");
      }
    }
  }
  if (const json::Value* additional = schema.find("additionalProperties");
      additional != nullptr && !additional->is_bool()) {
    const Status status = ValidateSchemaDefinition(
        *additional, strict, std::string(path) + ".additionalProperties",
        depth + 1U);
    if (!status.ok()) return status;
  }
  if (const json::Value* items = schema.find("items"); items != nullptr) {
    const Status status = ValidateSchemaDefinition(
        *items, strict, std::string(path) + ".items", depth + 1U);
    if (!status.ok()) return status;
  }
  for (const std::string_view keyword : {"anyOf", "oneOf", "allOf"}) {
    const json::Value* alternatives = schema.find(keyword);
    if (alternatives == nullptr) continue;
    if (!alternatives->is_array() || alternatives->as_array().empty()) {
      return InvalidTool(std::string(path) + "." + std::string(keyword) +
                         " must be a non-empty array");
    }
    for (std::size_t index = 0U; index < alternatives->as_array().size();
         ++index) {
      const Status status = ValidateSchemaDefinition(
          alternatives->as_array()[index], strict,
          std::string(path) + "." + std::string(keyword) + "[" +
              std::to_string(index) + "]",
          depth + 1U);
      if (!status.ok()) return status;
    }
  }
  if (const json::Value* definitions = schema.find("$defs");
      definitions != nullptr) {
    if (!definitions->is_object()) {
      return InvalidTool(std::string(path) + " $defs must be an object");
    }
    for (const auto& [name, child] : definitions->as_object()) {
      const Status status = ValidateSchemaDefinition(
          child, strict, std::string(path) + ".$defs." + name, depth + 1U);
      if (!status.ok()) return status;
    }
  }
  if (const json::Value* reference = schema.find("$ref");
      reference != nullptr &&
      (!reference->is_string() ||
       !reference->as_string().starts_with("#/$defs/"))) {
    return Status(StatusCode::kUnsupported,
                  std::string(path) + " supports only local #/$defs references");
  }
  if (const json::Value* enumeration = schema.find("enum");
      enumeration != nullptr &&
      (!enumeration->is_array() || enumeration->as_array().empty())) {
    return InvalidTool(std::string(path) + " enum must be a non-empty array");
  }
  for (const std::string_view keyword :
       {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"}) {
    const json::Value* value = schema.find(keyword);
    if (value != nullptr && (!value->is_number() ||
                             !std::isfinite(value->as_number()))) {
      return InvalidTool(std::string(path) + "." + std::string(keyword) +
                         " must be finite numeric data");
    }
  }
  for (const std::string_view keyword :
       {"minLength", "maxLength", "minItems", "maxItems"}) {
    const json::Value* value = schema.find(keyword);
    if (value != nullptr &&
        (!value->is_integer() || value->as_integer() < 0)) {
      return InvalidTool(std::string(path) + "." + std::string(keyword) +
                         " must be a non-negative integer");
    }
  }
  return Status::Ok();
}

const json::Value* ResolveReference(const json::Value& root,
                                    std::string_view reference) {
  constexpr std::string_view kPrefix = "#/$defs/";
  if (!reference.starts_with(kPrefix)) return nullptr;
  const json::Value* definitions = root.find("$defs");
  return definitions == nullptr || !definitions->is_object()
             ? nullptr
             : definitions->find(reference.substr(kPrefix.size()));
}

bool JsonEqual(const json::Value& left, const json::Value& right) {
  if (left.is_number() && right.is_number()) {
    return left.as_number() == right.as_number();
  }
  return json::Stringify(left) == json::Stringify(right);
}

bool TypeMatches(const json::Value& value, std::string_view type) {
  if (type == "null") return value.is_null();
  if (type == "boolean") return value.is_bool();
  if (type == "object") return value.is_object();
  if (type == "array") return value.is_array();
  if (type == "number") return value.is_number();
  if (type == "integer") return value.is_integer();
  if (type == "string") return value.is_string();
  return false;
}

bool MatchesStrictSchema(const json::Value& value, const json::Value& schema,
                         const json::Value& root, std::string& reason,
                         std::uint32_t depth = 0U) {
  if (depth > 32U) {
    reason = "schema recursion exceeds 32 levels";
    return false;
  }
  if (const json::Value* reference = schema.find("$ref");
      reference != nullptr) {
    const json::Value* target =
        ResolveReference(root, reference->as_string());
    if (target == nullptr) {
      reason = "schema reference cannot be resolved";
      return false;
    }
    if (!MatchesStrictSchema(value, *target, root, reason, depth + 1U)) {
      return false;
    }
  }
  for (const std::string_view keyword : {"allOf", "anyOf", "oneOf"}) {
    const json::Value* alternatives = schema.find(keyword);
    if (alternatives == nullptr) continue;
    std::size_t matches = 0U;
    for (const json::Value& alternative : alternatives->as_array()) {
      std::string ignored;
      if (MatchesStrictSchema(value, alternative, root, ignored, depth + 1U)) {
        ++matches;
      }
    }
    const bool valid = keyword == "allOf"
                           ? matches == alternatives->as_array().size()
                       : keyword == "anyOf" ? matches != 0U
                                             : matches == 1U;
    if (!valid) {
      reason = std::string(keyword) + " constraint did not match";
      return false;
    }
  }
  if (const json::Value* type = schema.find("type"); type != nullptr) {
    bool matched = false;
    if (type->is_string()) {
      matched = TypeMatches(value, type->as_string());
    } else {
      for (const json::Value& member : type->as_array()) {
        matched = matched || TypeMatches(value, member.as_string());
      }
    }
    if (!matched) {
      reason = "value has the wrong JSON type";
      return false;
    }
  }
  if (const json::Value* enumeration = schema.find("enum");
      enumeration != nullptr) {
    const bool found = std::any_of(
        enumeration->as_array().begin(), enumeration->as_array().end(),
        [&](const json::Value& candidate) { return JsonEqual(value, candidate); });
    if (!found) {
      reason = "value is not in the schema enum";
      return false;
    }
  }
  if (const json::Value* constant = schema.find("const");
      constant != nullptr && !JsonEqual(value, *constant)) {
    reason = "value differs from the schema const";
    return false;
  }
  if (value.is_object()) {
    const json::Value* properties = schema.find("properties");
    if (const json::Value* required = schema.find("required");
        required != nullptr) {
      for (const json::Value& name : required->as_array()) {
        if (value.find(name.as_string()) == nullptr) {
          reason = "required property '" + name.as_string() + "' is missing";
          return false;
        }
      }
    }
    for (const auto& [name, member] : value.as_object()) {
      const json::Value* member_schema =
          properties == nullptr ? nullptr : properties->find(name);
      if (member_schema != nullptr) {
        if (!MatchesStrictSchema(member, *member_schema, root, reason,
                                 depth + 1U)) {
          reason = "property '" + name + "': " + reason;
          return false;
        }
        continue;
      }
      const json::Value* additional = schema.find("additionalProperties");
      if (additional != nullptr && additional->is_bool() &&
          !additional->as_bool()) {
        reason = "additional property '" + name + "' is forbidden";
        return false;
      }
      if (additional != nullptr && additional->is_object() &&
          !MatchesStrictSchema(member, *additional, root, reason,
                               depth + 1U)) {
        reason = "additional property '" + name + "': " + reason;
        return false;
      }
    }
  }
  if (value.is_array()) {
    const std::size_t size = value.as_array().size();
    if (const json::Value* minimum = schema.find("minItems");
        minimum != nullptr &&
        size < static_cast<std::size_t>(minimum->as_integer())) {
      reason = "array is shorter than minItems";
      return false;
    }
    if (const json::Value* maximum = schema.find("maxItems");
        maximum != nullptr &&
        size > static_cast<std::size_t>(maximum->as_integer())) {
      reason = "array is longer than maxItems";
      return false;
    }
    if (const json::Value* items = schema.find("items"); items != nullptr) {
      for (const json::Value& member : value.as_array()) {
        if (!MatchesStrictSchema(member, *items, root, reason, depth + 1U)) {
          reason = "array item: " + reason;
          return false;
        }
      }
    }
  }
  if (value.is_string()) {
    const std::optional<std::size_t> size =
        Utf8CodePointCount(value.as_string());
    if (!size.has_value()) {
      reason = "string contains invalid UTF-8";
      return false;
    }
    if (const json::Value* minimum = schema.find("minLength");
        minimum != nullptr &&
        *size < static_cast<std::size_t>(minimum->as_integer())) {
      reason = "string is shorter than minLength";
      return false;
    }
    if (const json::Value* maximum = schema.find("maxLength");
        maximum != nullptr &&
        *size > static_cast<std::size_t>(maximum->as_integer())) {
      reason = "string is longer than maxLength";
      return false;
    }
  }
  if (value.is_number()) {
    const double number = value.as_number();
    if (const json::Value* minimum = schema.find("minimum");
        minimum != nullptr && number < minimum->as_number()) {
      reason = "number is below minimum";
      return false;
    }
    if (const json::Value* maximum = schema.find("maximum");
        maximum != nullptr && number > maximum->as_number()) {
      reason = "number is above maximum";
      return false;
    }
    if (const json::Value* minimum = schema.find("exclusiveMinimum");
        minimum != nullptr && number <= minimum->as_number()) {
      reason = "number is not above exclusiveMinimum";
      return false;
    }
    if (const json::Value* maximum = schema.find("exclusiveMaximum");
        maximum != nullptr && number >= maximum->as_number()) {
      reason = "number is not below exclusiveMaximum";
      return false;
    }
  }
  return true;
}

std::size_t RetainedPrefixSuffix(std::string_view value, std::string_view marker) {
  const std::size_t maximum = std::min(value.size(), marker.size() - 1U);
  for (std::size_t length = maximum; length != 0U; --length) {
    if (value.substr(value.size() - length) == marker.substr(0U, length)) {
      return length;
    }
  }
  return 0U;
}

class ArgumentDslParser {
 public:
  explicit ArgumentDslParser(std::string_view input) : input_(input) {}

  Result<std::string> Parse() {
    auto value = ParseValue();
    if (!value.ok()) return value.status();
    SkipWhitespace();
    if (position_ != input_.size()) {
      return ParseError("Gemma tool arguments contain trailing data");
    }
    return value;
  }

 private:
  void SkipWhitespace() {
    while (position_ < input_.size() &&
           (input_[position_] == ' ' || input_[position_] == '\t' ||
            input_[position_] == '\r' || input_[position_] == '\n'))
      ++position_;
  }

  Result<std::string> ParseValue(std::size_t depth = 0U) {
    if (depth > 32U)
      return ParseError("Gemma tool arguments exceed the nesting limit");
    SkipWhitespace();
    if (position_ >= input_.size()) {
      return ParseError("Gemma tool arguments contain an incomplete value");
    }
    if (input_.substr(position_).starts_with(kQuote)) return ParseString();
    if (input_[position_] == '{') return ParseObject(depth);
    if (input_[position_] == '[') return ParseArray(depth);
    const std::size_t begin = position_;
    while (position_ < input_.size() && input_[position_] != ',' &&
           input_[position_] != '}' && input_[position_] != ']') {
      ++position_;
    }
    if (begin == position_) {
      return ParseError("Gemma tool arguments contain an empty value");
    }
    auto end = position_;
    while (end > begin &&
           (input_[end - 1U] == ' ' || input_[end - 1U] == '\t' ||
            input_[end - 1U] == '\r' || input_[end - 1U] == '\n'))
      --end;
    const std::string_view literal = input_.substr(begin, end - begin);
    if (literal == "true" || literal == "false" || literal == "null") {
      return std::string(literal);
    }
    double number = 0.0;
    const auto parsed = std::from_chars(
        literal.data(), literal.data() + literal.size(), number);
    if (parsed.ec != std::errc{} ||
        parsed.ptr != literal.data() + literal.size()) {
      return ParseError("Gemma tool arguments contain an unsupported literal");
    }
    return std::string(literal);
  }

  Result<std::string> ParseString() {
    position_ += kQuote.size();
    const std::size_t close = input_.find(kQuote, position_);
    if (close == std::string_view::npos) {
      return ParseError("Gemma tool arguments contain an unterminated string");
    }
    std::string result = "\"";
    result.append(json::Escape(input_.substr(position_, close - position_)));
    result.push_back('\"');
    position_ = close + kQuote.size();
    return result;
  }

  Result<std::string> ParseObject(std::size_t depth) {
    ++position_;
    SkipWhitespace();
    std::string result = "{";
    bool first = true;
    while (position_ < input_.size() && input_[position_] != '}') {
      const std::size_t key_begin = position_;
      while (position_ < input_.size() &&
             (std::isalnum(static_cast<unsigned char>(input_[position_])) ||
              input_[position_] == '_')) {
        ++position_;
      }
      const auto key_end = position_;
      SkipWhitespace();
      if (key_begin == key_end || position_ >= input_.size() ||
          input_[position_] != ':') {
        return ParseError("Gemma tool arguments contain an invalid object key");
      }
      const std::string_view key =
          input_.substr(key_begin, key_end - key_begin);
      ++position_;
      auto value = ParseValue(depth + 1U);
      SkipWhitespace();
      if (!value.ok()) return value.status();
      if (!first) result.push_back(',');
      first = false;
      result.push_back('\"');
      result.append(json::Escape(key));
      result.append("\":");
      result.append(value.value());
      if (position_ < input_.size() && input_[position_] == ',') {
        ++position_;
        SkipWhitespace();
      } else if (position_ >= input_.size() || input_[position_] != '}') {
        return ParseError(
            "Gemma tool arguments contain an unterminated object");
      }
    }
    if (position_ >= input_.size()) {
      return ParseError("Gemma tool arguments contain an unterminated object");
    }
    ++position_;
    result.push_back('}');
    return result;
  }

  Result<std::string> ParseArray(std::size_t depth) {
    ++position_;
    SkipWhitespace();
    std::string result = "[";
    bool first = true;
    while (position_ < input_.size() && input_[position_] != ']') {
      auto value = ParseValue(depth + 1U);
      SkipWhitespace();
      if (!value.ok()) return value.status();
      if (!first) result.push_back(',');
      first = false;
      result.append(value.value());
      if (position_ < input_.size() && input_[position_] == ',') {
        ++position_;
        SkipWhitespace();
      } else if (position_ >= input_.size() || input_[position_] != ']') {
        return ParseError("Gemma tool arguments contain an unterminated array");
      }
    }
    if (position_ >= input_.size()) {
      return ParseError("Gemma tool arguments contain an unterminated array");
    }
    ++position_;
    result.push_back(']');
    return result;
  }

  std::string_view input_;
  std::size_t position_ = 0U;
};

}  // namespace

Status ValidateToolDefinitions(
    std::span<const GenerationToolDefinition> tools) {
  std::set<std::string, std::less<>> names;
  for (const GenerationToolDefinition& tool : tools) {
    if (!IsProtocolName(tool.name)) {
      return Status(
          StatusCode::kUnsupported,
          "tool name '" + tool.name +
              "' must contain 1..64 ASCII letters, digits, or underscores");
    }
    if (!names.insert(tool.name).second) {
      return InvalidTool("tool names must be unique: " + tool.name);
    }
    auto schema = json::Parse(
        tool.parameters_json,
        {.max_depth = 32U,
         .max_values = 10'000U,
         .max_string_bytes = 1024U * 1024U});
    if (!schema.ok()) {
      return InvalidTool("tool '" + tool.name +
                         "' parameters are invalid JSON: " +
                         schema.status().message());
    }
    const Status status = ValidateSchemaDefinition(
        schema.value(), tool.strict, "tool '" + tool.name + "' parameters");
    if (!status.ok()) return status;
    const json::Value* root_type = schema.value().find("type");
    if (root_type == nullptr || !root_type->is_string() ||
        root_type->as_string() != "object") {
      return InvalidTool("tool '" + tool.name +
                         "' parameters must have root type object");
    }
  }
  return Status::Ok();
}

Status ValidateGeneratedToolCalls(
    std::span<const GenerationToolDefinition> tools,
    const GenerationToolChoice& tool_choice,
    std::span<const GenerationToolCall> calls) {
  if (tool_choice.mode == GenerationToolChoiceMode::kNone && !calls.empty()) {
    return ParseError("model emitted a tool call while tool_choice is none");
  }
  std::set<std::string, std::less<>> call_ids;
  for (const GenerationToolCall& call : calls) {
    if (call.id.empty() || !call_ids.insert(call.id).second) {
      return ParseError("model emitted an empty or duplicate tool-call ID");
    }
    const auto definition = std::find_if(
        tools.begin(), tools.end(), [&](const GenerationToolDefinition& tool) {
          return tool.name == call.name;
        });
    if (definition == tools.end()) {
      return ParseError("model emitted undeclared tool '" + call.name + "'");
    }
    if (tool_choice.mode == GenerationToolChoiceMode::kFunction &&
        call.name != tool_choice.function_name) {
      return ParseError("model emitted a tool other than the selected function");
    }
    auto arguments = json::Parse(
        call.arguments_json,
        {.max_depth = 32U,
         .max_values = 10'000U,
         .max_string_bytes = 1024U * 1024U});
    if (!arguments.ok() || !arguments.value().is_object()) {
      return ParseError("model emitted non-object arguments for tool '" +
                        call.name + "'");
    }
    if (!definition->strict) continue;
    auto schema = json::Parse(
        definition->parameters_json,
        {.max_depth = 32U,
         .max_values = 10'000U,
         .max_string_bytes = 1024U * 1024U});
    if (!schema.ok()) {
      return Status(StatusCode::kInternal,
                    "validated strict tool schema cannot be reparsed");
    }
    std::string reason;
    if (!MatchesStrictSchema(arguments.value(), schema.value(), schema.value(),
                             reason)) {
      return ParseError("strict tool '" + call.name +
                        "' arguments violate its schema: " + reason);
    }
  }
  return Status::Ok();
}

Result<std::vector<GenerationEvent>> GemmaToolCallParser::Push(std::string_view text, bool final) {
  pending_.append(text);
  return Consume(final);
}

Result<std::vector<GenerationEvent>> GemmaToolCallParser::Consume(bool final) {
  std::vector<GenerationEvent> events;
  while (true) {
    if (state_ == State::kText) {
      const std::size_t open = pending_.find(kToolOpen);
      if (open != std::string::npos) {
        if (open != 0U) {
          GenerationEvent event;
          event.kind = GenerationEventKind::kTextDelta;
          event.text_delta = pending_.substr(0U, open);
          visible_text_.append(event.text_delta);
          events.push_back(std::move(event));
        }
        pending_.erase(0U, open + kToolOpen.size());
        state_ = State::kName;
        continue;
      }
      const std::size_t retained = final ? 0U : RetainedPrefixSuffix(pending_, kToolOpen);
      const std::size_t emitted = pending_.size() - retained;
      if (emitted != 0U) {
        GenerationEvent event;
        event.kind = GenerationEventKind::kTextDelta;
        event.text_delta = pending_.substr(0U, emitted);
        visible_text_.append(event.text_delta);
        events.push_back(std::move(event));
        pending_.erase(0U, emitted);
      }
      break;
    }

    if (state_ == State::kName) {
      if (pending_.size() < kCallPrefix.size()) {
        if (!kCallPrefix.starts_with(pending_)) {
          return ParseError("Gemma tool call is missing the call: prefix");
        }
        break;
      }
      if (!pending_.starts_with(kCallPrefix)) {
        return ParseError("Gemma tool call is missing the call: prefix");
      }
      const std::size_t brace = pending_.find('{', kCallPrefix.size());
      if (brace == std::string::npos) break;
      current_name_ = pending_.substr(kCallPrefix.size(), brace - kCallPrefix.size());
      if (current_name_.empty()) {
        return ParseError("Gemma tool call has an empty function name");
      }
      for (const char character : current_name_) {
        if (!std::isalnum(static_cast<unsigned char>(character)) && character != '_') {
          return ParseError("Gemma tool call has an invalid function name");
        }
      }
      current_id_ = "call_" + std::to_string(tool_calls_.size() + 1U);
      current_arguments_ = "{";
      pending_.erase(0U, brace + 1U);
      state_ = State::kArguments;
      GenerationEvent event;
      event.kind = GenerationEventKind::kToolCallStart;
      event.tool_call_id = current_id_;
      event.tool_name = current_name_;
      events.push_back(std::move(event));
      continue;
    }

    const std::size_t close = pending_.find(kToolClose);
    if (close == std::string::npos) {
      const std::size_t retained = RetainedPrefixSuffix(pending_, kToolClose);
      current_arguments_.append(pending_.substr(0U, pending_.size() - retained));
      pending_.erase(0U, pending_.size() - retained);
      break;
    }
    current_arguments_.append(pending_.substr(0U, close));
    pending_.erase(0U, close + kToolClose.size());
    auto arguments = ArgumentDslParser(current_arguments_).Parse();
    if (!arguments.ok()) return arguments.status();
    tool_calls_.push_back({current_id_, current_name_, arguments.value()});
    GenerationEvent arguments_event;
    arguments_event.kind = GenerationEventKind::kToolCallArgumentsDelta;
    arguments_event.text_delta = arguments.value();
    arguments_event.tool_call_id = current_id_;
    arguments_event.tool_name = current_name_;
    events.push_back(std::move(arguments_event));
    GenerationEvent end_event;
    end_event.kind = GenerationEventKind::kToolCallEnd;
    end_event.tool_call_id = current_id_;
    end_event.tool_name = current_name_;
    events.push_back(std::move(end_event));
    current_name_.clear();
    current_arguments_.clear();
    current_id_.clear();
    state_ = State::kText;
  }
  if (final && (state_ != State::kText || !pending_.empty())) {
    return ParseError("model response ends inside a Gemma tool call marker");
  }
  return events;
}

}  // namespace gem16::internal
