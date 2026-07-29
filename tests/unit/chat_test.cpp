#include "gem16/chat.h"

#include <string>

#include "test.h"

namespace {

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
  GEM16_CHECK(!request.enable_thinking);
  GEM16_CHECK(std::string(gem16::GenerationFinishReasonName(
                  gem16::GenerationFinishReason::kStop)) == "stop");
  GEM16_CHECK(std::string(gem16::GenerationFinishReasonName(
                  gem16::GenerationFinishReason::kLength)) == "length");
}

}  // namespace

void RunChatTests() { TestProtocolNeutralGenerationTypes(); }
