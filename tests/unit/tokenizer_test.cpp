#include "gem16/tokenizer.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "model/tokenizer_config.h"
#include "test.h"

namespace {

void TestBpeEncodeDecodeAndSpecialTokens() {
  const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
  const std::filesystem::path directory =
      std::filesystem::temp_directory_path() / ("gem16-tokenizer-test-" + std::to_string(suffix));
  std::error_code error;
  std::filesystem::create_directories(directory, error);
  GEM16_CHECK(!error);
  const std::filesystem::path path = directory / "tokenizer.json";
  {
    std::ofstream output(path, std::ios::binary);
    output << R"({
      "added_tokens": [
        {"id": 0, "content": "<s>", "special": true}
      ],
      "model": {
        "type": "BPE",
        "byte_fallback": true,
        "vocab": {
          "<s>": 0,
          "a": 1,
          "b": 2,
          "ab": 3,
          "▁": 4,
          "▁ab": 5,
          "<0xC3>": 6,
          "<0xA9>": 7
        },
        "merges": [
          ["a", "b"],
          ["▁", "ab"]
        ]
      }
    })";
    GEM16_CHECK(output.good());
  }

  auto tokenizer = gem16::Tokenizer::Load(path);
  GEM16_CHECK(tokenizer.ok());
  if (tokenizer.ok()) {
    auto encoded = tokenizer.value().Encode("<s>ab ab");
    GEM16_CHECK(encoded.ok());
    if (encoded.ok()) {
      GEM16_CHECK(encoded.value() == std::vector<std::uint32_t>({0U, 3U, 5U}));
      auto decoded = tokenizer.value().Decode(encoded.value(), false);
      GEM16_CHECK(decoded.ok());
      if (decoded.ok()) GEM16_CHECK(decoded.value() == "<s>ab ab");
      auto without_special = tokenizer.value().Decode(encoded.value(), true);
      GEM16_CHECK(without_special.ok());
      if (without_special.ok()) GEM16_CHECK(without_special.value() == "ab ab");

      std::ostringstream streamed;
      auto status = tokenizer.value().WriteDecodedToken(0U, true, streamed);
      GEM16_CHECK(status.ok());
      status = tokenizer.value().WriteDecodedToken(3U, true, streamed);
      GEM16_CHECK(status.ok());
      status = tokenizer.value().WriteDecodedToken(5U, true, streamed);
      GEM16_CHECK(status.ok());
      status = tokenizer.value().WriteDecodedToken(6U, true, streamed);
      GEM16_CHECK(status.ok());
      status = tokenizer.value().WriteDecodedToken(7U, true, streamed);
      GEM16_CHECK(status.ok());
      GEM16_CHECK(streamed.str() == "ab ab\xC3\xA9");
    }
  }
  std::filesystem::remove(path, error);
  GEM16_CHECK(!error);
  std::filesystem::remove(directory, error);
  GEM16_CHECK(!error);
}

void TestGoogleTokenizerConfigAndResponseTemplate() {
  const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
  const std::filesystem::path directory =
      std::filesystem::temp_directory_path() / ("gem16-tokenizer-config-test-" + std::to_string(suffix));
  std::error_code error;
  std::filesystem::create_directories(directory, error);
  GEM16_CHECK(!error);
  const std::filesystem::path path = directory / "tokenizer_config.json";
  {
    std::ofstream output(path, std::ios::binary);
    output << R"json({
      "tokenizer_class": "GemmaTokenizer",
      "bos_token": "<bos>",
      "eos_token": "<eos>",
      "eot_token": "<turn|>",
      "etr_token": "<tool_response|>",
      "stc_token": "<|tool_call>",
      "model_max_length": 1000000000000000019884624838656,
      "response_template": {
        "defaults": {"role": "assistant"},
        "fields": {
          "content": {
            "close": ["<turn|>", "<|tool_response>", "<eos>"],
            "content": "text"
          },
          "thinking": {
            "close": "<channel|>",
            "content": "text",
            "open": "<|channel>thought\n"
          },
          "tool_calls": {
            "close": "<tool_call|>",
            "content": "json",
            "open_pattern": "<\\|tool_call>call:(?P<name>\\w+)",
            "repeats": true
          }
        },
        "start_anchor": ["<|turn>model\n", "<tool_response|>"]
      }
    })json";
    GEM16_CHECK(output.good());
  }

  auto config = gem16::internal::LoadTokenizerConfig(path);
  GEM16_CHECK(config.ok());
  if (config.ok()) {
    GEM16_CHECK(gem16::internal::ValidatePrimaryTokenizerConfig(config.value()).ok());
    GEM16_CHECK(config.value().model_max_length > 1.0e29);
    auto content = gem16::internal::ExtractResponseContent(
        "<|channel>thought\nsecret<channel|>  visible answer <turn|>", config.value().thinking_open,
        config.value().thinking_close, config.value().content_close_tokens, config.value().tool_call_start_token);
    GEM16_CHECK(content.ok());
    if (content.ok()) GEM16_CHECK(content.value() == "visible answer");

    auto tool_call = gem16::internal::ExtractResponseContent(
        "<|tool_call>call:test{}<tool_call|>", config.value().thinking_open, config.value().thinking_close,
        config.value().content_close_tokens, config.value().tool_call_start_token);
    GEM16_CHECK(tool_call.ok());
    if (tool_call.ok()) GEM16_CHECK(tool_call.value().empty());

    auto definition = gem16::internal::RenderGemmaToolDefinition(
        "weather", "Get weather",
        R"({"type":"object","properties":{"location":{"type":"string","description":"City"}},"required":["location"]})");
    GEM16_CHECK(definition.ok());
    if (definition.ok()) {
      GEM16_CHECK(definition.value() ==
                  "<|tool>declaration:weather{description:<|\"|>Get "
                  "weather<|\"|>,parameters:{properties:{location:{description:<|\"|>City<|\"|>,type:<|\"|>STRING<|\"|>"
                  "}},required:[<|\"|>location<|\"|>],type:<|\"|>OBJECT<|\"|>}}<tool|>");
    }
    auto rendered_call = gem16::internal::RenderGemmaToolCall("weather", R"({"location":"Berlin","days":[1,2]})");
    GEM16_CHECK(rendered_call.ok());
    if (rendered_call.ok()) {
      GEM16_CHECK(rendered_call.value() ==
                  "<|tool_call>call:weather{days:[1,2],location:<|\"|>Berlin<|\"|>}<tool_call|>");
    }

    auto old_config = config.value();
    old_config.eos_token = "<turn|>";
    GEM16_CHECK(!gem16::internal::ValidatePrimaryTokenizerConfig(old_config).ok());
  }

  std::filesystem::remove(path, error);
  GEM16_CHECK(!error);
  std::filesystem::remove(directory, error);
  GEM16_CHECK(!error);
}

struct TokenizerTests {
  TokenizerTests() {
    TestBpeEncodeDecodeAndSpecialTokens();
    TestGoogleTokenizerConfigAndResponseTemplate();
  }
} tokenizer_tests;

}  // namespace
