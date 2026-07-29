#include "gem16/chat.h"

#include <string>

#include "test.h"

namespace {

void TestResponseChannelRecognition() {
  gem16::GenerationTokenControls controls;
  controls.thinking_open_token_ids = {100U, 45518U, 108U};
  controls.thinking_close_token_id = 101U;
  gem16::ResponseChannelTracker tracker(controls);
  GEM16_CHECK(tracker.Observe(100U) ==
              gem16::ResponseTokenChannel::kControl);
  GEM16_CHECK(tracker.Observe(45518U) ==
              gem16::ResponseTokenChannel::kControl);
  GEM16_CHECK(tracker.Observe(108U) ==
              gem16::ResponseTokenChannel::kControl);
  GEM16_CHECK(tracker.in_reasoning());
  GEM16_CHECK(tracker.Observe(7U) ==
              gem16::ResponseTokenChannel::kReasoning);
  GEM16_CHECK(tracker.Observe(8U) ==
              gem16::ResponseTokenChannel::kReasoning);
  GEM16_CHECK(tracker.reasoning_token_count() == 2U);
  GEM16_CHECK(tracker.Observe(101U) ==
              gem16::ResponseTokenChannel::kControl);
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
  GEM16_CHECK(first.content.front().kind ==
              gem16::GenerationContentKind::kText);
  GEM16_CHECK(first.content.front().text == "hello");

  gem16::ChatGenerationRequest request;
  request.messages.push_back(first);
  GEM16_CHECK(!request.max_generated_tokens.has_value());
  GEM16_CHECK(request.thinking.effort == gem16::ThinkingEffort::kMedium);
  GEM16_CHECK(gem16::ThinkingBudgetTokens(gem16::ThinkingEffort::kOff) == 0U);
  GEM16_CHECK(gem16::ThinkingBudgetTokens(
                  gem16::ThinkingEffort::kSmall) == 1024U);
  GEM16_CHECK(gem16::ThinkingBudgetTokens(
                  gem16::ThinkingEffort::kMedium) == 4096U);
  GEM16_CHECK(gem16::ThinkingBudgetTokens(
                  gem16::ThinkingEffort::kHigh) == 8192U);
  GEM16_CHECK(std::string(gem16::GenerationFinishReasonName(
                  gem16::GenerationFinishReason::kStop)) == "stop");
  GEM16_CHECK(std::string(gem16::GenerationFinishReasonName(
                  gem16::GenerationFinishReason::kLength)) == "length");
}

}  // namespace

void RunChatTests() {
  TestResponseChannelRecognition();
  TestProtocolNeutralGenerationTypes();
}
