#include "gem16gb/tokenizer.h"

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
  const auto suffix =
      std::chrono::steady_clock::now().time_since_epoch().count();
  const std::filesystem::path directory =
      std::filesystem::temp_directory_path() /
      ("gem16gb-tokenizer-test-" + std::to_string(suffix));
  std::error_code error;
  std::filesystem::create_directories(directory, error);
  GEM16GB_CHECK(!error);
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
    GEM16GB_CHECK(output.good());
  }

  auto tokenizer = gem16gb::Tokenizer::Load(path);
  GEM16GB_CHECK(tokenizer.ok());
  if (tokenizer.ok()) {
    auto encoded = tokenizer.value().Encode("<s>ab ab");
    GEM16GB_CHECK(encoded.ok());
    if (encoded.ok()) {
      GEM16GB_CHECK(encoded.value() ==
                    std::vector<std::uint32_t>({0U, 3U, 5U}));
      auto decoded = tokenizer.value().Decode(encoded.value(), false);
      GEM16GB_CHECK(decoded.ok());
      if (decoded.ok()) GEM16GB_CHECK(decoded.value() == "<s>ab ab");
      auto without_special = tokenizer.value().Decode(encoded.value(), true);
      GEM16GB_CHECK(without_special.ok());
      if (without_special.ok()) GEM16GB_CHECK(without_special.value() == "ab ab");

      std::ostringstream streamed;
      auto status = tokenizer.value().WriteDecodedToken(0U, true, streamed);
      GEM16GB_CHECK(status.ok());
      status = tokenizer.value().WriteDecodedToken(3U, true, streamed);
      GEM16GB_CHECK(status.ok());
      status = tokenizer.value().WriteDecodedToken(5U, true, streamed);
      GEM16GB_CHECK(status.ok());
      status = tokenizer.value().WriteDecodedToken(6U, true, streamed);
      GEM16GB_CHECK(status.ok());
      status = tokenizer.value().WriteDecodedToken(7U, true, streamed);
      GEM16GB_CHECK(status.ok());
      GEM16GB_CHECK(streamed.str() == "ab ab\xC3\xA9");
    }
  }
  std::filesystem::remove(path, error);
  GEM16GB_CHECK(!error);
  std::filesystem::remove(directory, error);
  GEM16GB_CHECK(!error);
}

void TestGoogleTokenizerConfigAndResponseTemplate() {
  const auto suffix =
      std::chrono::steady_clock::now().time_since_epoch().count();
  const std::filesystem::path directory =
      std::filesystem::temp_directory_path() /
      ("gem16gb-tokenizer-config-test-" + std::to_string(suffix));
  std::error_code error;
  std::filesystem::create_directories(directory, error);
  GEM16GB_CHECK(!error);
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
    GEM16GB_CHECK(output.good());
  }

  auto config = gem16gb::internal::LoadTokenizerConfig(path);
  GEM16GB_CHECK(config.ok());
  if (config.ok()) {
    GEM16GB_CHECK(
        gem16gb::internal::ValidatePrimaryTokenizerConfig(config.value()).ok());
    GEM16GB_CHECK(config.value().model_max_length > 1.0e29);
    auto content = gem16gb::internal::ExtractResponseContent(
        "<|channel>thought\nsecret<channel|>  visible answer <turn|>",
        config.value().thinking_open, config.value().thinking_close,
        config.value().content_close_tokens,
        config.value().tool_call_start_token);
    GEM16GB_CHECK(content.ok());
    if (content.ok()) GEM16GB_CHECK(content.value() == "visible answer");

    auto tool_call = gem16gb::internal::ExtractResponseContent(
        "<|tool_call>call:test{}<tool_call|>",
        config.value().thinking_open, config.value().thinking_close,
        config.value().content_close_tokens,
        config.value().tool_call_start_token);
    GEM16GB_CHECK(!tool_call.ok());

    auto old_config = config.value();
    old_config.eos_token = "<turn|>";
    GEM16GB_CHECK(
        !gem16gb::internal::ValidatePrimaryTokenizerConfig(old_config).ok());
  }

  std::filesystem::remove(path, error);
  GEM16GB_CHECK(!error);
  std::filesystem::remove(directory, error);
  GEM16GB_CHECK(!error);
}

struct TokenizerTests {
  TokenizerTests() {
    TestBpeEncodeDecodeAndSpecialTokens();
    TestGoogleTokenizerConfigAndResponseTemplate();
  }
} tokenizer_tests;

}  // namespace
