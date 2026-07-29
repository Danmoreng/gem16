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
