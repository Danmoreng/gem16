#include "test.h"

#include <algorithm>
#include <charconv>
#include <span>
#include <string>
#include <vector>

#include "server/openai_chat.h"
#include "server/secure_id.h"
#include "runtime/chat_internal.h"
#include "server/sse_chunk.h"
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
  auto secure_id_a = gem16::server::MakeSecureId("resp_test_");
  auto secure_id_b = gem16::server::MakeSecureId("resp_test_");
  GEM16_CHECK(secure_id_a.ok());
  GEM16_CHECK(secure_id_b.ok());
  if (secure_id_a.ok() && secure_id_b.ok()) {
    GEM16_CHECK(secure_id_a.value().size() == 42U);
    GEM16_CHECK(secure_id_a.value().starts_with("resp_test_"));
    GEM16_CHECK(secure_id_a.value() != secure_id_b.value());
    GEM16_CHECK(std::all_of(
        secure_id_a.value().begin() + 10, secure_id_a.value().end(),
        [](char value) {
          return (value >= '0' && value <= '9') ||
                 (value >= 'a' && value <= 'f');
        }));
  }

  gem16::server::SseChunkBuilder fixed_chunk(128U);
  fixed_chunk.Reset();
  GEM16_CHECK(fixed_chunk.Append("{\"text\":"));
  GEM16_CHECK(fixed_chunk.AppendJsonString("line\n\"quoted\""));
  GEM16_CHECK(fixed_chunk.Append(",\"n\":"));
  GEM16_CHECK(fixed_chunk.AppendUnsigned(42U));
  GEM16_CHECK(fixed_chunk.Append("}"));
  const std::span<const char> framed = fixed_chunk.Finish();
  GEM16_CHECK(!framed.empty());
  if (!framed.empty()) {
    const std::string frame(framed.data(), framed.size());
    const std::size_t header_end = frame.find("\r\n");
    GEM16_CHECK(header_end != std::string::npos);
    if (header_end != std::string::npos) {
      std::size_t payload_size = 0U;
      const auto parsed = std::from_chars(
          frame.data(), frame.data() + header_end, payload_size, 16);
      GEM16_CHECK(parsed.ec == std::errc{});
      GEM16_CHECK(payload_size ==
                  frame.size() - header_end - 2U - 2U);
      GEM16_CHECK(frame.substr(header_end + 2U, payload_size) ==
                  "data: {\"text\":\"line\\n\\\"quoted\\\"\",\"n\":42}\n\n");
      GEM16_CHECK(frame.ends_with("\r\n"));
    }
  }
  gem16::server::SseChunkBuilder undersized(4U);
  undersized.Reset();
  GEM16_CHECK(!undersized.Append("payload"));
  GEM16_CHECK(undersized.Finish().empty());

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

  const std::string encoded_image = Base64(TinyBmp());
  const std::string one_image_request =
      "{\"model\":\"gem16\",\"messages\":[{\"role\":\"user\",\"content\":["
      "{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/bmp;base64," +
      encoded_image + "\"}}]}]}";
  const std::string two_turn_image_request =
      "{\"model\":\"gem16\",\"messages\":[{\"role\":\"user\",\"content\":["
      "{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/bmp;base64," +
      encoded_image +
      "\"}}]},{\"role\":\"assistant\",\"content\":\"first\"},"
      "{\"role\":\"user\",\"content\":[{\"type\":\"image_url\","
      "\"image_url\":{\"url\":\"data:image/bmp;base64," +
      encoded_image + "\"}}]}]}";
  auto one_image = gem16::server::ParseChatCompletionsRequest(
      one_image_request, {512U});
  auto two_images = gem16::server::ParseChatCompletionsRequest(
      two_turn_image_request, {512U});
  auto moe26b_image = gem16::server::ParseChatCompletionsRequest(
      one_image_request, {512U, true});
  auto moe26b_multiple_images = gem16::server::ParseChatCompletionsRequest(
      two_turn_image_request, {512U, true});
  const std::string requested_budget_image =
      one_image_request.substr(0U, one_image_request.size() - 1U) +
      ",\"vision_soft_token_budget\":70}";
  auto moe26b_requested_budget =
      gem16::server::ParseChatCompletionsRequest(requested_budget_image,
                                                 {512U, true});
  auto moe26b_configured_70 = gem16::server::ParseChatCompletionsRequest(
      one_image_request, {512U, true, 70U});
  GEM16_CHECK(one_image.ok());
  GEM16_CHECK(two_images.ok());
  GEM16_CHECK(moe26b_image.ok());
  GEM16_CHECK(!moe26b_multiple_images.ok());
  GEM16_CHECK(moe26b_requested_budget.ok());
  GEM16_CHECK(moe26b_configured_70.ok());
  GEM16_CHECK(!gem16::server::ParseChatCompletionsRequest(
                   requested_budget_image, {512U, false})
                   .ok());
  const std::string invalid_budget_image =
      one_image_request.substr(0U, one_image_request.size() - 1U) +
      ",\"vision_soft_token_budget\":71}";
  GEM16_CHECK(!gem16::server::ParseChatCompletionsRequest(
                   invalid_budget_image, {512U, true})
                   .ok());
  const std::string excessive_budget_image =
      one_image_request.substr(0U, one_image_request.size() - 1U) +
      ",\"vision_soft_token_budget\":280}";
  GEM16_CHECK(!gem16::server::ParseChatCompletionsRequest(
                   excessive_budget_image, {512U, true, 140U})
                   .ok());
  auto excessive_budget = gem16::server::ParseChatCompletionsRequest(
      excessive_budget_image, {400U, true});
  GEM16_CHECK(!excessive_budget.ok());
  if (!excessive_budget.ok()) {
    GEM16_CHECK(excessive_budget.status().message() ==
                "remaining context cannot fit the requested 26B image budget");
  }
  if (!moe26b_multiple_images.ok()) {
    GEM16_CHECK(moe26b_multiple_images.status().message() ==
                "the Gemma 4 26B Vision profile supports exactly one image");
  }
  if (moe26b_image.ok()) {
    const auto& content =
        moe26b_image.value().generation.messages.front().content.front();
    GEM16_CHECK(content.kind ==
                gem16::GenerationContentKind::kGemma4Moe26BImage);
    GEM16_CHECK(content.moe26b_image.raw_patch_count ==
                content.moe26b_image.soft_token_count * 9U);
    GEM16_CHECK(content.moe26b_image.patches.size() ==
                static_cast<std::size_t>(
                    content.moe26b_image.raw_patch_count) * 768U);
  }
  if (moe26b_configured_70.ok()) {
    GEM16_CHECK(moe26b_configured_70.value()
                    .generation.messages.front()
                    .content.front()
                    .moe26b_image.soft_token_budget == 70U);
  }
  if (moe26b_requested_budget.ok()) {
    const auto& content = moe26b_requested_budget.value()
                              .generation.messages.front()
                              .content.front();
    GEM16_CHECK(content.moe26b_image.soft_token_budget == 70U);
  }
  if (one_image.ok() && two_images.ok()) {
    const auto& original = one_image.value().generation.messages.front();
    const auto& reprocessed = two_images.value().generation.messages.front();
    GEM16_CHECK(original.content.front().image.soft_token_budget !=
                reprocessed.content.front().image.soft_token_budget);
    GEM16_CHECK(original.content.front().image.source_identity ==
                reprocessed.content.front().image.source_identity);
    GEM16_CHECK(gem16::internal::ResidentMessageEquivalent(original,
                                                           reprocessed));
  }

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
  GEM16_CHECK(!gem16::server::ParseChatCompletionsRequest(
                   R"({"model":"gem16","messages":[{"role":"user","content":"x"}],"stop":"END"})")
                   .ok());
  GEM16_CHECK(!gem16::server::ParseChatCompletionsRequest(
                   R"({"model":"gem16","messages":[{"role":"user","content":"x","name":"ignored"}]})")
                   .ok());
  GEM16_CHECK(!gem16::server::ParseChatCompletionsRequest(
                   R"({"model":"gem16","messages":[{"role":"user","content":"x"}],"stream_options":{"include_usage":true,"ignored":1}})")
                   .ok());

  gem16::ChatGenerationResponse generated;
  generated.assistant_text = "Calling weather";
  generated.tool_calls.push_back(
      {"call_1", "get_weather", R"({"location":"Berlin"})"});
  generated.finish_reason = gem16::GenerationFinishReason::kToolCalls;
  generated.prompt_token_ids = {1U, 2U, 3U};
  generated.inference.output_token_ids = {4U, 5U};
  generated.inference.prompt_cached_tokens = 2U;
  generated.inference.prompt_cache_write_tokens = 1U;
  const gem16::server::OpenAiResponseIdentity identity{
      "chatcmpl-test", "gem16", 123, std::nullopt};
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
    GEM16_CHECK(parsed_response.value()
                    .find("usage")
                    ->find("prompt_tokens_details")
                    ->find("cached_tokens")
                    ->as_integer() == 2);
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

  const std::string responses_two_images =
      "{\"model\":\"gem16\",\"input\":[{\"type\":\"message\","
      "\"role\":\"user\",\"content\":["
      "{\"type\":\"input_image\",\"image_url\":\"data:image/bmp;base64," +
      encoded_image +
      "\"},{\"type\":\"input_image\",\"image_url\":\"data:image/bmp;base64," +
      encoded_image + "\"}]}]}";
  auto responses_multiple_images = gem16::server::ParseResponsesRequest(
      responses_two_images, {512U, true});
  GEM16_CHECK(!responses_multiple_images.ok());
  if (!responses_multiple_images.ok()) {
    GEM16_CHECK(responses_multiple_images.status().message() ==
                "the Gemma 4 26B Vision profile supports exactly one image");
  }
  const std::string responses_requested_budget =
      "{\"model\":\"gem16\",\"vision_soft_token_budget\":140,"
      "\"input\":[{\"type\":\"message\",\"role\":\"user\",\"content\":["
      "{\"type\":\"input_image\",\"image_url\":\"data:image/bmp;base64," +
      encoded_image + "\"}]}]}";
  auto responses_budget = gem16::server::ParseResponsesRequest(
      responses_requested_budget, {512U, true});
  GEM16_CHECK(responses_budget.ok());
  if (responses_budget.ok()) {
    GEM16_CHECK(responses_budget.value()
                    .generation.messages.front()
                    .content.front()
                    .moe26b_image.soft_token_budget == 140U);
  }

  auto parallel_history = gem16::server::ParseResponsesRequest(R"({
    "model":"gem16","input":[
      {"role":"user","content":"Look up both keys"},
      {"type":"function_call","call_id":"a","name":"lookup","arguments":"{}"},
      {"type":"function_call","call_id":"b","name":"lookup","arguments":"{}"},
      {"type":"function_call_output","call_id":"a","output":"A"},
      {"type":"function_call_output","call_id":"b","output":"B"},
      {"type":"function_call","call_id":"c","name":"read","arguments":"{}"}
    ]
  })");
  GEM16_CHECK(parallel_history.ok());
  if (parallel_history.ok()) {
    const auto& messages = parallel_history.value().generation.messages;
    GEM16_CHECK(messages.size() == 5U);
    GEM16_CHECK(messages[1].role == "assistant");
    GEM16_CHECK(messages[1].content.size() == 2U);
    GEM16_CHECK(messages[1].content[0].tool_call.id == "a");
    GEM16_CHECK(messages[1].content[1].tool_call.id == "b");
    GEM16_CHECK(messages[2].role == "tool" && messages[3].role == "tool");
    GEM16_CHECK(messages[4].role == "assistant");
    GEM16_CHECK(messages[4].content.size() == 1U);
    GEM16_CHECK(messages[4].content[0].tool_call.id == "c");
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
  GEM16_CHECK(!gem16::server::ParseResponsesRequest(
                   R"({"model":"gem16","input":"x","metadata":{"ignored":true}})")
                   .ok());
  GEM16_CHECK(!gem16::server::ParseResponsesRequest(
                   R"({"model":"gem16","input":"x","reasoning":{"effort":"none","summary":"auto"}})")
                   .ok());
  GEM16_CHECK(!gem16::server::ParseResponsesRequest(
                   R"({"model":"gem16","input":[{"type":"message","role":"user","content":"x","status":"completed"}]})")
                   .ok());

  gem16::server::OpenAiResponsesRequest response_request;
  response_request.model = "gem16";
  response_request.generation.tools.push_back(
      {"get_weather", "Get weather", R"({"type":"object"})", true});
  response_request.generation.max_generated_tokens = 64U;
  const gem16::server::OpenAiResponseIdentity response_identity{
      "resp_test", "gem16", 123, 456};
  generated.reasoning_text = "Inspect the requested location.";
  generated.inference.reasoning_tokens = 4U;
  const std::string response_json = gem16::server::ResponseJson(
      response_identity, response_request, generated);
  auto parsed_responses = gem16::json::Parse(response_json);
  GEM16_CHECK(parsed_responses.ok());
  if (parsed_responses.ok()) {
    GEM16_CHECK(parsed_responses.value().find("object")->as_string() ==
                "response");
    GEM16_CHECK(parsed_responses.value().find("created_at")->as_integer() ==
                123);
    GEM16_CHECK(parsed_responses.value().find("completed_at")->as_integer() ==
                456);
    GEM16_CHECK(parsed_responses.value().find("output")->as_array().size() ==
                3U);
    GEM16_CHECK(parsed_responses.value()
                    .find("output")
                    ->as_array()[0]
                    .find("type")
                    ->as_string() == "reasoning");
    GEM16_CHECK(parsed_responses.value()
                    .find("output")
                    ->as_array()[0]
                    .find("content")
                    ->as_array()[0]
                    .find("text")
                    ->as_string() == "Inspect the requested location.");
    GEM16_CHECK(parsed_responses.value()
                    .find("usage")
                    ->find("total_tokens")
                    ->as_integer() == 5);
    GEM16_CHECK(parsed_responses.value()
                    .find("usage")
                    ->find("input_tokens_details")
                    ->find("cached_tokens")
                    ->as_integer() == 2);
    GEM16_CHECK(parsed_responses.value()
                    .find("usage")
                    ->find("input_tokens_details")
                    ->find("cache_write_tokens")
                    ->as_integer() == 1);
  }
  for (const auto reason : {gem16::GenerationFinishReason::kStop,
                            gem16::GenerationFinishReason::kToolCalls,
                            gem16::GenerationFinishReason::kLength}) {
    generated.finish_reason = reason;
    const auto terminal = gem16::json::Parse(gem16::server::ResponseTerminalEventJson(
        response_identity, response_request, generated, 17U));
    GEM16_CHECK(terminal.ok());
    if (!terminal.ok()) continue;
    const bool incomplete = reason == gem16::GenerationFinishReason::kLength;
    const auto& event = terminal.value();
    GEM16_CHECK(event.find("type")->as_string() ==
                (incomplete ? "response.incomplete" : "response.completed"));
    GEM16_CHECK(event.find("sequence_number")->as_integer() == 17);
    const auto* terminal_response = event.find("response");
    GEM16_CHECK(terminal_response->find("status")->as_string() ==
                (incomplete ? "incomplete" : "completed"));
    GEM16_CHECK(terminal_response->find("completed_at")->is_null() == incomplete);
    if (incomplete) GEM16_CHECK(terminal_response->find("incomplete_details")
                                   ->find("reason")->as_string() == "max_output_tokens");
    GEM16_CHECK(terminal_response->find("usage")->find("input_tokens")->as_integer() ==
                parsed_responses.value().find("usage")->find("input_tokens")->as_integer());
  }

}
