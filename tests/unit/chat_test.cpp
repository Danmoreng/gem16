#include "gem16/chat.h"

#include <string>
#include <vector>

#include "runtime/tool_call_parser.h"
#include "runtime/chat_internal.h"
#include "cuda/engine/media_chunk_plan.h"
#include "test.h"

namespace {

void TestResponseChannelRecognition() {
  gem16::GenerationTokenControls controls;
  controls.thinking_open_token_ids = {100U, 45518U, 108U};
  controls.thinking_close_token_id = 101U;
  gem16::ResponseChannelTracker tracker(controls);
  GEM16_CHECK(tracker.Observe(100U) == gem16::ResponseTokenChannel::kControl);
  GEM16_CHECK(tracker.Observe(45518U) == gem16::ResponseTokenChannel::kControl);
  GEM16_CHECK(tracker.Observe(108U) == gem16::ResponseTokenChannel::kControl);
  GEM16_CHECK(tracker.in_reasoning());
  GEM16_CHECK(tracker.Observe(7U) == gem16::ResponseTokenChannel::kReasoning);
  GEM16_CHECK(tracker.Observe(8U) == gem16::ResponseTokenChannel::kReasoning);
  GEM16_CHECK(tracker.reasoning_token_count() == 2U);
  GEM16_CHECK(tracker.Observe(101U) == gem16::ResponseTokenChannel::kControl);
  GEM16_CHECK(!tracker.in_reasoning());
  GEM16_CHECK(tracker.Observe(9U) == gem16::ResponseTokenChannel::kText);
  GEM16_CHECK(tracker.reasoning_token_count() == 2U);
}

void TestProtocolNeutralGenerationTypes() {
  const auto first = gem16::GenerationMessage::Text("user", "hello");
  const auto same = gem16::GenerationMessage::Text("user", "hello");
  const auto different = gem16::GenerationMessage::Text("assistant", "hello");
  GEM16_CHECK(first == same);
  GEM16_CHECK(!(first == different));
  GEM16_CHECK(first.content.size() == 1U);
  GEM16_CHECK(first.content.front().kind == gem16::GenerationContentKind::kText);
  GEM16_CHECK(first.content.front().text == "hello");
  gem16::AudioWaveform waveform;
  waveform.samples = {0.0F, 0.5F};
  auto audio = gem16::GenerationContentPart::Audio(waveform);
  GEM16_CHECK(audio.kind == gem16::GenerationContentKind::kAudio);
  GEM16_CHECK(audio.audio == waveform);
  gem16::VisionImage vision_image;
  vision_image.patch_count = 1U;
  vision_image.patches.resize(48U * 48U * 3U, 0.25F);
  vision_image.positions = {0, 0};
  auto image = gem16::GenerationContentPart::Image(vision_image);
  GEM16_CHECK(image.kind == gem16::GenerationContentKind::kImage);
  GEM16_CHECK(image.image == vision_image);
  const gem16::GenerationToolCall tool_call{"call_1", "weather", R"({"location":"Berlin"})"};
  const auto tool_call_part = gem16::GenerationContentPart::ToolCall(tool_call);
  GEM16_CHECK(tool_call_part.kind == gem16::GenerationContentKind::kToolCall);
  GEM16_CHECK(tool_call_part.tool_call == tool_call);
  const gem16::GenerationToolResult tool_result{"call_1", R"({"temperature":25})"};
  const auto tool_result_part = gem16::GenerationContentPart::ToolResult(tool_result);
  GEM16_CHECK(tool_result_part.kind == gem16::GenerationContentKind::kToolResult);
  GEM16_CHECK(tool_result_part.tool_result == tool_result);

  gem16::ChatGenerationRequest request;
  request.messages.push_back(first);
  request.tools.push_back(gem16::GenerationToolDefinition{
      "weather", "Get current weather", R"({"type":"object","properties":{"location":{"type":"string"}}})", true});
  request.tool_choice = {gem16::GenerationToolChoiceMode::kFunction, "weather"};
  GEM16_CHECK(!request.max_generated_tokens.has_value());
  GEM16_CHECK(request.thinking.effort == gem16::ThinkingEffort::kMedium);
  GEM16_CHECK(request.tools.size() == 1U);
  GEM16_CHECK(request.tools.front().strict);
  GEM16_CHECK(request.tool_choice.mode == gem16::GenerationToolChoiceMode::kFunction);
  GEM16_CHECK(request.tool_choice.function_name == "weather");
  GEM16_CHECK(request.parallel_tool_calls);
  gem16::GenerationEvent event;
  event.kind = gem16::GenerationEventKind::kToolCallArgumentsDelta;
  event.text_delta = R"({"location":)";
  event.tool_call_id = "call_1";
  event.tool_name = "weather";
  GEM16_CHECK(event.text_delta == R"({"location":)");
  GEM16_CHECK(event.tool_call_id == "call_1");
  GEM16_CHECK(event.tool_name == "weather");
  GEM16_CHECK(gem16::ThinkingBudgetTokens(gem16::ThinkingEffort::kOff) == 0U);
  GEM16_CHECK(gem16::ThinkingBudgetTokens(gem16::ThinkingEffort::kSmall) == 1024U);
  GEM16_CHECK(gem16::ThinkingBudgetTokens(gem16::ThinkingEffort::kMedium) == 4096U);
  GEM16_CHECK(gem16::ThinkingBudgetTokens(gem16::ThinkingEffort::kHigh) == 8192U);
  GEM16_CHECK(std::string(gem16::GenerationFinishReasonName(gem16::GenerationFinishReason::kStop)) == "stop");
  GEM16_CHECK(std::string(gem16::GenerationFinishReasonName(gem16::GenerationFinishReason::kLength)) == "length");
  GEM16_CHECK(std::string(gem16::GenerationFinishReasonName(gem16::GenerationFinishReason::kToolCalls)) ==
              "tool_calls");
}

void TestIncrementalGemmaToolCallParser() {
  gem16::internal::GemmaToolCallParser parser;
  auto first = parser.Push("Answer <|tool_");
  GEM16_CHECK(first.ok());
  GEM16_CHECK(parser.visible_text() == "Answer ");
  auto second = parser.Push("call>call:weather{location:<|\"|>Berlin<|\"|>,days:[1,2]}");
  GEM16_CHECK(second.ok());
  if (second.ok()) {
    GEM16_CHECK(second.value().size() == 1U);
    GEM16_CHECK(second.value().front().kind == gem16::GenerationEventKind::kToolCallStart);
    GEM16_CHECK(second.value().front().tool_name == "weather");
  }
  auto third = parser.Push("<tool_call|> tail", true);
  GEM16_CHECK(third.ok());
  if (third.ok()) {
    GEM16_CHECK(third.value().size() == 3U);
    GEM16_CHECK(third.value()[0].kind == gem16::GenerationEventKind::kToolCallArgumentsDelta);
    GEM16_CHECK(third.value()[0].text_delta == R"({"location":"Berlin","days":[1,2]})");
    GEM16_CHECK(third.value()[1].kind == gem16::GenerationEventKind::kToolCallEnd);
    GEM16_CHECK(third.value()[2].kind == gem16::GenerationEventKind::kTextDelta);
  }
  GEM16_CHECK(parser.visible_text() == "Answer  tail");
  GEM16_CHECK(parser.tool_calls().size() == 1U);
  GEM16_CHECK(parser.tool_calls().front().arguments_json == R"({"location":"Berlin","days":[1,2]})");

  gem16::internal::GemmaToolCallParser malformed;
  GEM16_CHECK(!malformed.Push("<|tool_call>call:x{a:1}", true).ok());
}

void TestMultipleImageChunkPlanning() {
  std::vector<float> first_patches(2U * 6912U);
  std::vector<float> second_patches(2U * 6912U);
  std::vector<std::int32_t> positions(2U * 2U);
  const std::vector<gem16::VisionEmbeddingSegment> segments = {
      {10U, first_patches, positions}, {20U, second_patches, positions}};
  GEM16_CHECK(gem16::internal::PlanVisionAwarePrefillChunk(
                  0U, 32U, segments) == 10U);
  GEM16_CHECK(gem16::internal::PlanVisionAwarePrefillChunk(
                  10U, 22U, segments) == 2U);
  GEM16_CHECK(gem16::internal::PlanVisionAwarePrefillChunk(
                  12U, 20U, segments) == 8U);
  GEM16_CHECK(gem16::internal::PlanVisionAwarePrefillChunk(
                  20U, 12U, segments) == 2U);
}

void TestReasoningMaterialization() {
  gem16::GenerationTokenControls controls;
  controls.thinking_open_token_ids = {100U, 45518U, 108U};
  controls.thinking_close_token_id = 101U;
  const std::vector<std::uint32_t> generated = {
      100U, 45518U, 108U, 7U, 8U, 101U, 9U};
  const auto reasoning = gem16::internal::ExtractReasoningTokenIds(
      generated, controls);
  GEM16_CHECK(reasoning == std::vector<std::uint32_t>({7U, 8U}));
  const std::vector<std::uint32_t> leading_text = {
      42U, 100U, 45518U, 108U, 7U, 8U, 101U, 9U};
  const auto delayed_reasoning = gem16::internal::ExtractReasoningTokenIds(
      leading_text, controls);
  GEM16_CHECK(delayed_reasoning ==
              std::vector<std::uint32_t>({7U, 8U}));
  const std::vector<std::uint32_t> repeated_reasoning = {
      100U, 45518U, 108U, 7U, 101U, 9U,
      100U, 45518U, 108U, 8U, 101U, 10U};
  const auto first_reasoning_only =
      gem16::internal::ExtractReasoningTokenIds(repeated_reasoning, controls);
  GEM16_CHECK(first_reasoning_only == std::vector<std::uint32_t>({7U}));
}

void TestResidentImageIdentity() {
  gem16::VisionImage cached_image;
  cached_image.patch_count = 4U;
  cached_image.soft_token_budget = 280U;
  cached_image.source_fingerprint = 1234U;
  cached_image.patches.resize(4U * 6912U, 0.25F);
  cached_image.positions.resize(8U);
  gem16::VisionImage reprocessed_image = cached_image;
  reprocessed_image.patch_count = 2U;
  reprocessed_image.soft_token_budget = 128U;
  reprocessed_image.patches.resize(2U * 6912U);
  reprocessed_image.positions.resize(4U);

  gem16::GenerationMessage cached;
  cached.role = "user";
  cached.content.push_back(
      gem16::GenerationContentPart::Image(std::move(cached_image)));
  gem16::GenerationMessage supplied;
  supplied.role = "user";
  supplied.content.push_back(
      gem16::GenerationContentPart::Image(std::move(reprocessed_image)));
  GEM16_CHECK(gem16::internal::ResidentMessageEquivalent(cached, supplied));

  supplied.content.front().image.source_fingerprint = 5678U;
  GEM16_CHECK(!gem16::internal::ResidentMessageEquivalent(cached, supplied));
}

void TestStrictToolContract() {
  const std::vector<gem16::GenerationToolDefinition> tools = {{
      "get_weather", "Get weather",
      R"({"type":"object","properties":{"location":{"type":"string"},"days":{"type":"integer","minimum":1,"maximum":7}},"required":["location"],"additionalProperties":false})",
      true}};
  GEM16_CHECK(gem16::internal::ValidateToolDefinitions(tools).ok());

  const gem16::GenerationToolChoice automatic;
  const std::vector<gem16::GenerationToolCall> valid = {
      {"call_1", "get_weather", R"({"location":"Berlin","days":3})"}};
  GEM16_CHECK(gem16::internal::ValidateGeneratedToolCalls(
                  tools, automatic, valid)
                  .ok());
  const std::vector<gem16::GenerationToolCall> extra = {
      {"call_1", "get_weather",
       R"({"location":"Berlin","unexpected":true})"}};
  GEM16_CHECK(!gem16::internal::ValidateGeneratedToolCalls(
                   tools, automatic, extra)
                   .ok());
  const std::vector<gem16::GenerationToolCall> out_of_range = {
      {"call_1", "get_weather", R"({"location":"Berlin","days":8})"}};
  GEM16_CHECK(!gem16::internal::ValidateGeneratedToolCalls(
                   tools, automatic, out_of_range)
                   .ok());
  const std::vector<gem16::GenerationToolDefinition> unicode_tools = {{
      "get_weather", "Get weather",
      R"({"type":"object","properties":{"location":{"type":"string","minLength":2}},"required":["location"]})",
      true}};
  const std::vector<gem16::GenerationToolCall> unicode_too_short = {
      {"call_1", "get_weather", R"({"location":"ä"})"}};
  GEM16_CHECK(!gem16::internal::ValidateGeneratedToolCalls(
                   unicode_tools, automatic, unicode_too_short)
                   .ok());
  const std::vector<gem16::GenerationToolCall> unknown = {
      {"call_1", "delete_files", R"({})"}};
  GEM16_CHECK(!gem16::internal::ValidateGeneratedToolCalls(
                   tools, automatic, unknown)
                   .ok());

  const std::vector<gem16::GenerationToolDefinition> incompatible = {{
      "get-weather", "Get weather", R"({"type":"object"})", false}};
  GEM16_CHECK(!gem16::internal::ValidateToolDefinitions(incompatible).ok());
  const std::vector<gem16::GenerationToolDefinition> unsupported_strict = {{
      "search", "Search", R"({"type":"object","patternProperties":{}})",
      true}};
  GEM16_CHECK(
      !gem16::internal::ValidateToolDefinitions(unsupported_strict).ok());
}

}  // namespace

void RunChatTests() {
  TestResponseChannelRecognition();
  TestProtocolNeutralGenerationTypes();
  TestIncrementalGemmaToolCallParser();
  TestMultipleImageChunkPlanning();
  TestReasoningMaterialization();
  TestResidentImageIdentity();
  TestStrictToolContract();
}
