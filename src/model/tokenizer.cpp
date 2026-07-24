#include "gem16gb/tokenizer.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cctype>
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

#include "util/json.h"

namespace gem16gb {
namespace {

constexpr std::uint64_t kMaximumTokenizerBytes = 64U * 1024U * 1024U;
constexpr std::uint64_t kMaximumTemplateBytes = 1024U * 1024U;
constexpr std::uint64_t kPinnedTemplateFnv1a = 0xe9f262823e5bda06ULL;
constexpr std::string_view kSpaceMarker = "\xE2\x96\x81";

Status Error(StatusCode code, std::string message) {
  return Status(code, std::move(message));
}

Result<std::string> ReadFile(const std::filesystem::path& path, std::uint64_t limit) {
  std::error_code error;
  const std::uint64_t size = std::filesystem::file_size(path, error);
  if (error) {
    return Error(StatusCode::kIoError,
                 "cannot stat " + path.string() + ": " + error.message());
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
  const auto conversion =
      std::from_chars(token.data() + 3, token.data() + 5, parsed, 16);
  if (conversion.ec != std::errc{} || conversion.ptr != token.data() + 5 ||
      parsed > 0xFFU) {
    return false;
  }
  value = static_cast<unsigned char>(parsed);
  return true;
}

std::string StripThinking(std::string_view text) {
  std::string result;
  std::size_t begin = 0;
  while (begin <= text.size()) {
    const std::size_t delimiter = text.find("<channel|>", begin);
    const std::size_t end = delimiter == std::string_view::npos ? text.size() : delimiter;
    const std::string_view part = text.substr(begin, end - begin);
    const std::size_t channel = part.find("<|channel>");
    result.append(part.substr(0, channel == std::string_view::npos ? part.size() : channel));
    if (delimiter == std::string_view::npos) break;
    begin = delimiter + std::string_view("<channel|>").size();
  }
  return Trim(result);
}

Result<std::vector<std::uint32_t>> IntegerList(const json::Value* value,
                                               std::string_view field,
                                               bool allow_scalar) {
  std::vector<std::uint32_t> result;
  if (allow_scalar && value != nullptr && value->is_integer()) {
    if (value->as_integer() < 0 ||
        static_cast<std::uint64_t>(value->as_integer()) >
            std::numeric_limits<std::uint32_t>::max()) {
      return Error(StatusCode::kDataLoss,
                   "generation_config.json has invalid " + std::string(field));
    }
    result.push_back(static_cast<std::uint32_t>(value->as_integer()));
    return result;
  }
  if (value == nullptr || !value->is_array()) {
    return Error(StatusCode::kDataLoss,
                 "generation_config.json has invalid " + std::string(field));
  }
  for (const auto& item : value->as_array()) {
    if (!item.is_integer() || item.as_integer() < 0 ||
        static_cast<std::uint64_t>(item.as_integer()) >
            std::numeric_limits<std::uint32_t>::max()) {
      return Error(StatusCode::kDataLoss,
                   "generation_config.json has invalid " + std::string(field));
    }
    result.push_back(static_cast<std::uint32_t>(item.as_integer()));
  }
  if (result.empty() && field == "eos_token_id") {
    return Error(StatusCode::kDataLoss, "generation_config.json has no EOS token");
  }
  return result;
}

}  // namespace

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

  [[nodiscard]] Result<std::vector<std::uint32_t>> EncodeOrdinary(
      std::string_view ordinary) const {
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
          return Error(StatusCode::kDataLoss,
                       "tokenizer byte fallback is incomplete");
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
        if (index + 1U < symbols.size() &&
            PairKey(symbols[index], symbols[index + 1U]) == best_pair) {
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
  auto parsed = json::Parse(
      text.value(),
      {.max_depth = 128, .max_values = 2'000'000,
       .max_string_bytes = 256U * 1024U * 1024U});
  if (!parsed.ok()) {
    return Error(parsed.status().code(),
                 tokenizer_json.string() + ": " + parsed.status().message());
  }
  const json::Value* model = Member(parsed.value(), "model");
  const json::Value* vocabulary = model == nullptr ? nullptr : Member(*model, "vocab");
  const json::Value* merges = model == nullptr ? nullptr : Member(*model, "merges");
  const json::Value* model_type = model == nullptr ? nullptr : Member(*model, "type");
  const json::Value* byte_fallback =
      model == nullptr ? nullptr : Member(*model, "byte_fallback");
  if (model == nullptr || !model->is_object() || model_type == nullptr ||
      !model_type->is_string() || model_type->as_string() != "BPE" ||
      byte_fallback == nullptr || !byte_fallback->is_bool() ||
      !byte_fallback->as_bool() || vocabulary == nullptr ||
      !vocabulary->is_object() || merges == nullptr || !merges->is_array()) {
    return Error(StatusCode::kUnsupported,
                 "tokenizer must be a byte-fallback BPE tokenizer");
  }

  auto implementation = std::make_shared<Impl>();
  std::uint32_t maximum_id = 0;
  for (const auto& [token, id_value] : vocabulary->as_object()) {
    if (!id_value.is_integer() || id_value.as_integer() < 0 ||
        static_cast<std::uint64_t>(id_value.as_integer()) >
            std::numeric_limits<std::uint32_t>::max()) {
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
  }

  std::uint32_t rank = 0;
  implementation->merges.reserve(merges->as_array().size());
  for (const auto& entry : merges->as_array()) {
    if (!entry.is_array() || entry.as_array().size() != 2U ||
        !entry.as_array()[0].is_string() || !entry.as_array()[1].is_string()) {
      return Error(StatusCode::kDataLoss, "tokenizer merge is malformed");
    }
    const std::string& left_text = entry.as_array()[0].as_string();
    const std::string& right_text = entry.as_array()[1].as_string();
    const auto left = implementation->vocabulary.find(left_text);
    const auto right = implementation->vocabulary.find(right_text);
    const auto result = implementation->vocabulary.find(left_text + right_text);
    if (left == implementation->vocabulary.end() ||
        right == implementation->vocabulary.end() ||
        result == implementation->vocabulary.end()) {
      return Error(StatusCode::kDataLoss,
                   "tokenizer merge references an absent vocabulary token");
    }
    implementation->merges.emplace(
        PairKey(left->second, right->second),
        Impl::Merge{rank++, result->second});
  }

  const json::Value* added = Member(parsed.value(), "added_tokens");
  if (added == nullptr || !added->is_array()) {
    return Error(StatusCode::kDataLoss, "tokenizer has no added_tokens array");
  }
  for (const auto& entry : added->as_array()) {
    const json::Value* content = Member(entry, "content");
    const json::Value* id_value = Member(entry, "id");
    const json::Value* special = Member(entry, "special");
    if (content == nullptr || !content->is_string() || id_value == nullptr ||
        !id_value->is_integer() || id_value->as_integer() < 0 ||
        static_cast<std::uint64_t>(id_value->as_integer()) >
            std::numeric_limits<std::uint32_t>::max() ||
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
      auto ordinary =
          implementation_->EncodeOrdinary(text.substr(ordinary_begin, offset - ordinary_begin));
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

Result<std::string> Tokenizer::Decode(std::span<const std::uint32_t> token_ids,
                                      bool skip_special_tokens) const {
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

Status Tokenizer::WriteDecodedToken(std::uint32_t token_id,
                                    bool skip_special_tokens,
                                    std::ostream& output) const {
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
          output.write(token.data() + begin,
                       static_cast<std::streamsize>(token.size() - begin));
          break;
        }
        output.write(token.data() + begin,
                     static_cast<std::streamsize>(marker - begin));
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

Result<GemmaChatProcessor> GemmaChatProcessor::Load(
    const std::filesystem::path& model_directory) {
  auto tokenizer = Tokenizer::Load(model_directory / "tokenizer.json");
  if (!tokenizer.ok()) return tokenizer.status();
  auto chat_template =
      ReadFile(model_directory / "chat_template.jinja", kMaximumTemplateBytes);
  if (!chat_template.ok()) return chat_template.status();
  if (Fnv1a(chat_template.value()) != kPinnedTemplateFnv1a) {
    return Error(
        StatusCode::kUnsupported,
        "chat_template.jinja differs from the natively supported pinned Gemma template");
  }

  auto generation_text =
      ReadFile(model_directory / "generation_config.json", 1024U * 1024U);
  if (!generation_text.ok()) return generation_text.status();
  auto generation = json::Parse(generation_text.value());
  if (!generation.ok() || !generation.value().is_object()) {
    return Error(StatusCode::kDataLoss, "generation_config.json is malformed");
  }
  auto stop = IntegerList(Member(generation.value(), "eos_token_id"),
                          "eos_token_id", true);
  if (!stop.ok()) return stop.status();
  const json::Value* suppressed_value =
      Member(generation.value(), "suppress_tokens");
  std::vector<std::uint32_t> suppressed;
  if (suppressed_value != nullptr) {
    auto parsed_suppressed =
        IntegerList(suppressed_value, "suppress_tokens", false);
    if (!parsed_suppressed.ok()) return parsed_suppressed.status();
    suppressed = std::move(parsed_suppressed).value();
  }
  return GemmaChatProcessor(
      std::move(tokenizer).value(),
      GenerationTokenControls{std::move(stop).value(), std::move(suppressed)});
}

Result<std::string> GemmaChatProcessor::Render(
    std::span<const ChatMessage> messages, bool enable_thinking,
    bool add_generation_prompt) const {
  if (messages.empty()) {
    return Error(StatusCode::kInvalidArgument, "chat requires at least one message");
  }
  std::string result = "<bos>";
  std::size_t message_index = 0;
  if (enable_thinking || messages.front().role == "system" ||
      messages.front().role == "developer") {
    result.append("<|turn>system\n");
    if (enable_thinking) result.append("<|think|>\n");
    if (messages.front().role == "system" ||
        messages.front().role == "developer") {
      result.append(Trim(messages.front().content));
      message_index = 1;
    }
    result.append("<turn|>\n");
  }

  std::string previous_role;
  for (; message_index < messages.size(); ++message_index) {
    const ChatMessage& message = messages[message_index];
    if (message.role != "user" && message.role != "assistant") {
      return Error(StatusCode::kUnsupported,
                   "native chat currently supports system/developer, user, and assistant roles");
    }
    if (message.role == previous_role) {
      return Error(StatusCode::kInvalidArgument,
                   "native chat requires alternating user and assistant messages");
    }
    const std::string_view rendered_role =
        message.role == "assistant" ? "model" : "user";
    result.append("<|turn>");
    result.append(rendered_role);
    result.push_back('\n');
    result.append(message.role == "assistant"
                      ? StripThinking(message.content)
                      : Trim(message.content));
    result.append("<turn|>\n");
    previous_role = message.role;
  }
  if (add_generation_prompt) {
    if (previous_role != "user") {
      return Error(StatusCode::kInvalidArgument,
                   "generation prompt requires a final user message");
    }
    result.append("<|turn>model\n");
    if (!enable_thinking) result.append("<|channel>thought\n<channel|>");
  }
  return result;
}

Result<std::vector<std::uint32_t>> GemmaChatProcessor::Encode(
    std::span<const ChatMessage> messages, bool enable_thinking,
    bool add_generation_prompt) const {
  auto rendered = Render(messages, enable_thinking, add_generation_prompt);
  if (!rendered.ok()) return rendered.status();
  return tokenizer_.Encode(rendered.value());
}

Result<std::string> GemmaChatProcessor::Decode(
    std::span<const std::uint32_t> token_ids, bool skip_special_tokens) const {
  return tokenizer_.Decode(token_ids, skip_special_tokens);
}

Status GemmaChatProcessor::WriteDecodedToken(std::uint32_t token_id,
                                             bool skip_special_tokens,
                                             std::ostream& output) const {
  return tokenizer_.WriteDecodedToken(token_id, skip_special_tokens, output);
}

}  // namespace gem16gb
