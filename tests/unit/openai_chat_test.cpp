#include "test.h"

#include <string>
#include <vector>

#include "server/openai_chat.h"
#include "util/json.h"

namespace {

void Put16(std::vector<std::uint8_t>& bytes, std::uint16_t value) {
  bytes.push_back(static_cast<std::uint8_t>(value));
  bytes.push_back(static_cast<std::uint8_t>(value >> 8U));
}

void Put32(std::vector<std::uint8_t>& bytes, std::uint32_t value) {
  for (unsigned shift = 0U; shift < 32U; shift += 8U) {
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
  }
}

std::string Base64(const std::vector<std::uint8_t>& bytes) {
  constexpr char kAlphabet[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string output;
  for (std::size_t index = 0U; index < bytes.size(); index += 3U) {
    const std::size_t remaining = bytes.size() - index;
    const std::uint32_t value =
        (static_cast<std::uint32_t>(bytes[index]) << 16U) |
        (remaining > 1U ? static_cast<std::uint32_t>(bytes[index + 1U]) << 8U
                        : 0U) |
        (remaining > 2U ? bytes[index + 2U] : 0U);
    output.push_back(kAlphabet[(value >> 18U) & 63U]);
    output.push_back(kAlphabet[(value >> 12U) & 63U]);
    output.push_back(remaining > 1U ? kAlphabet[(value >> 6U) & 63U] : '=');
    output.push_back(remaining > 2U ? kAlphabet[value & 63U] : '=');
  }
  return output;
}

std::vector<std::uint8_t> TinyBmp() {
  std::vector<std::uint8_t> bytes;
  bytes.push_back('B'); bytes.push_back('M'); Put32(bytes, 58U);
  Put16(bytes, 0U); Put16(bytes, 0U); Put32(bytes, 54U);
  Put32(bytes, 40U); Put32(bytes, 1U); Put32(bytes, 1U);
  Put16(bytes, 1U); Put16(bytes, 24U); Put32(bytes, 0U); Put32(bytes, 4U);
  Put32(bytes, 2835U); Put32(bytes, 2835U); Put32(bytes, 0U); Put32(bytes, 0U);
  bytes.push_back(255U); bytes.push_back(0U); bytes.push_back(0U); bytes.push_back(0U);
  return bytes;
}

std::vector<std::uint8_t> TinyWav() {
  std::vector<std::uint8_t> bytes;
  const auto text = [&bytes](const char* value) {
    for (unsigned index = 0U; index < 4U; ++index) {
      bytes.push_back(static_cast<std::uint8_t>(value[index]));
    }
  };
  text("RIFF"); Put32(bytes, 40U); text("WAVE");
  text("fmt "); Put32(bytes, 16U); Put16(bytes, 1U); Put16(bytes, 1U);
  Put32(bytes, 16000U); Put32(bytes, 32000U); Put16(bytes, 2U); Put16(bytes, 16U);
  text("data"); Put32(bytes, 4U); Put16(bytes, 0U); Put16(bytes, 16384U);
  return bytes;
}

}  // namespace

void RunOpenAiChatTests() {
  const std::string request_json = R"({
    "model":"gem16",
    "messages":[{"role":"user","content":"Weather in Berlin?"}],
    "tools":[{"type":"function","function":{
      "name":"get_weather","description":"Get weather","strict":true,
      "parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}
    }}],
    "tool_choice":"auto","parallel_tool_calls":true,
    "max_completion_tokens":64,"reasoning_effort":"none",
    "stream":true,"stream_options":{"include_usage":true}
  })";
  auto request = gem16::server::ParseChatCompletionsRequest(request_json);
  GEM16_CHECK(request.ok());
  if (request.ok()) {
    GEM16_CHECK(request.value().model == "gem16");
    GEM16_CHECK(request.value().stream);
    GEM16_CHECK(request.value().include_usage);
    GEM16_CHECK(request.value().generation.messages.size() == 1U);
    GEM16_CHECK(request.value().generation.tools.size() == 1U);
    GEM16_CHECK(request.value().generation.tools.front().strict);
    GEM16_CHECK(request.value().generation.tools.front().parameters_json ==
                R"({"properties":{"location":{"type":"string"}},"required":["location"],"type":"object"})");
    GEM16_CHECK(request.value().generation.max_generated_tokens == 64U);
    GEM16_CHECK(request.value().generation.thinking.effort ==
                gem16::ThinkingEffort::kOff);
  }

  auto continuation = gem16::server::ParseChatCompletionsRequest(R"({
    "model":"gem16","messages":[
      {"role":"user","content":"Weather?"},
      {"role":"assistant","content":null,"tool_calls":[{
        "id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Berlin\"}"}
      }]},
      {"role":"tool","tool_call_id":"call_1","content":"Sunny"}
    ]
  })");
  GEM16_CHECK(continuation.ok());
  if (continuation.ok()) {
    GEM16_CHECK(continuation.value().generation.messages.size() == 3U);
    GEM16_CHECK(continuation.value().generation.messages[1].content[0].kind ==
                gem16::GenerationContentKind::kToolCall);
    GEM16_CHECK(continuation.value().generation.messages[2].content[0]
                    .tool_result.output == "Sunny");
  }

  GEM16_CHECK(!gem16::server::ParseChatCompletionsRequest(
                   R"({"model":"gem16","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"x"}}]}]})")
                   .ok());

  const std::string media_request =
      "{\"model\":\"gem16\",\"messages\":[{\"role\":\"user\",\"content\":["
      "{\"type\":\"text\",\"text\":\"Describe and transcribe\"},"
      "{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/bmp;base64," +
      Base64(TinyBmp()) +
      "\"}},{\"type\":\"input_audio\",\"input_audio\":{\"format\":\"wav\",\"data\":\"" +
      Base64(TinyWav()) + "\"}}]}],\"max_completion_tokens\":16}";
  auto media = gem16::server::ParseChatCompletionsRequest(
      media_request, {256U});
  GEM16_CHECK(media.ok());
  if (media.ok()) {
    const auto& content = media.value().generation.messages[0].content;
    GEM16_CHECK(content.size() == 3U);
    GEM16_CHECK(content[1].kind == gem16::GenerationContentKind::kImage);
    GEM16_CHECK(content[1].image.patch_count == 1U);
    GEM16_CHECK(content[2].kind == gem16::GenerationContentKind::kAudio);
    GEM16_CHECK(content[2].audio.samples.size() == 2U);
  }
  std::string media_request_larger_output = media_request;
  const std::size_t output_limit =
      media_request_larger_output.find("\"max_completion_tokens\":16");
  GEM16_CHECK(output_limit != std::string::npos);
  if (output_limit != std::string::npos) {
    media_request_larger_output.replace(
        output_limit, std::string("\"max_completion_tokens\":16").size(),
        "\"max_completion_tokens\":96");
    auto larger_output = gem16::server::ParseChatCompletionsRequest(
        media_request_larger_output, {256U});
    GEM16_CHECK(larger_output.ok());
    if (media.ok() && larger_output.ok()) {
      GEM16_CHECK(media.value().generation.messages[0].content[1].image ==
                  larger_output.value().generation.messages[0].content[1]
                      .image);
    }
  }
  GEM16_CHECK(!gem16::server::ParseChatCompletionsRequest(
                   R"({"model":"gem16","messages":[{"role":"user","content":"x"}],"temperature":0.5})")
                   .ok());

  gem16::ChatGenerationResponse generated;
  generated.assistant_text = "Calling weather";
  generated.tool_calls.push_back(
      {"call_1", "get_weather", R"({"location":"Berlin"})"});
  generated.finish_reason = gem16::GenerationFinishReason::kToolCalls;
  generated.prompt_token_ids = {1U, 2U, 3U};
  generated.inference.output_token_ids = {4U, 5U};
  const gem16::server::OpenAiResponseIdentity identity{
      "chatcmpl-test", "gem16", 123};
  const std::string response =
      gem16::server::ChatCompletionJson(identity, generated);
  auto parsed_response = gem16::json::Parse(response);
  GEM16_CHECK(parsed_response.ok());
  if (parsed_response.ok()) {
    GEM16_CHECK(parsed_response.value().find("object")->as_string() ==
                "chat.completion");
    GEM16_CHECK(parsed_response.value().find("usage")
                    ->find("total_tokens")
                    ->as_integer() == 5);
  }

  const std::string chunk = gem16::server::ChatCompletionChunkJson(
      identity, R"({"content":"hello"})");
  GEM16_CHECK(gem16::json::Parse(chunk).ok());
  const std::string usage = gem16::server::ChatCompletionChunkJson(
      identity, {}, std::nullopt, &generated);
  auto parsed_usage = gem16::json::Parse(usage);
  GEM16_CHECK(parsed_usage.ok());
  if (parsed_usage.ok()) {
    GEM16_CHECK(parsed_usage.value().find("choices")->as_array().empty());
  }

  auto responses = gem16::server::ParseResponsesRequest(R"({
    "model":"gem16","instructions":"Be concise","input":"Weather in Berlin?",
    "tools":[{"type":"function","name":"get_weather",
      "description":"Get weather","strict":true,
      "parameters":{"type":"object","properties":{"location":{"type":"string"}}}}
    ],"tool_choice":"auto","max_output_tokens":64,
    "reasoning":{"effort":"low"},"store":true
  })");
  GEM16_CHECK(responses.ok());
  if (responses.ok()) {
    GEM16_CHECK(responses.value().generation.messages.size() == 2U);
    GEM16_CHECK(responses.value().generation.messages[0].role == "system");
    GEM16_CHECK(responses.value().generation.messages[1].role == "user");
    GEM16_CHECK(responses.value().generation.tools.size() == 1U);
    GEM16_CHECK(responses.value().generation.tools[0].name == "get_weather");
    GEM16_CHECK(responses.value().generation.thinking.effort ==
                gem16::ThinkingEffort::kSmall);
  }

  auto response_continuation = gem16::server::ParseResponsesRequest(R"({
    "model":"gem16","previous_response_id":"resp_1",
    "input":[{"type":"function_call_output","call_id":"call_1","output":"Sunny"}]
  })");
  GEM16_CHECK(response_continuation.ok());
  if (response_continuation.ok()) {
    GEM16_CHECK(response_continuation.value().previous_response_id ==
                "resp_1");
    GEM16_CHECK(response_continuation.value().generation.messages.size() == 1U);
    GEM16_CHECK(response_continuation.value().generation.messages[0].role ==
                "tool");
  }
  GEM16_CHECK(!gem16::server::ParseResponsesRequest(
                   R"({"model":"gem16","input":"x","store":false})")
                   .ok());

  gem16::server::OpenAiResponsesRequest response_request;
  response_request.model = "gem16";
  response_request.generation.tools.push_back(
      {"get_weather", "Get weather", R"({"type":"object"})", true});
  response_request.generation.max_generated_tokens = 64U;
  const gem16::server::OpenAiResponseIdentity response_identity{
      "resp_test", "gem16", 123};
  const std::string response_json = gem16::server::ResponseJson(
      response_identity, response_request, generated);
  auto parsed_responses = gem16::json::Parse(response_json);
  GEM16_CHECK(parsed_responses.ok());
  if (parsed_responses.ok()) {
    GEM16_CHECK(parsed_responses.value().find("object")->as_string() ==
                "response");
    GEM16_CHECK(parsed_responses.value().find("output")->as_array().size() ==
                2U);
    GEM16_CHECK(parsed_responses.value()
                    .find("usage")
                    ->find("total_tokens")
                    ->as_integer() == 5);
  }
}
