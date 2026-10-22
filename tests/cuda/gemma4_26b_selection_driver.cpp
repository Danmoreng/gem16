#include "cuda/engine/gemma4_26b_reference.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using Engine = gem16::internal::Gemma4Moe26BReferenceEngine;
constexpr std::uint32_t kVocabulary = 262144U;
constexpr std::array<std::uint32_t, 20> kPrompt{
    2U, 105U, 2364U, 107U, 40654U, 607U, 7121U, 506U, 3658U, 3730U,
    236761U, 106U, 107U, 105U, 4368U, 107U, 100U, 45518U, 107U, 101U};

void Require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void Check(const gem16::Status& status) {
  if (!status.ok()) throw std::runtime_error(status.message());
}
std::uint32_t Oracle(std::span<const float> logits,
                     std::span<const std::uint32_t> mask) {
  std::uint32_t winner = kVocabulary;
  for (std::uint32_t token = 0U; token < kVocabulary; ++token) {
    Require(std::isfinite(logits[token]), "non-finite fixture logit");
    if (std::find(mask.begin(), mask.end(), token) != mask.end()) continue;
    if (winner == kVocabulary || logits[token] > logits[winner]) winner = token;
  }
  Require(winner != kVocabulary, "empty candidate set");
  return winner;
}
void Start(Engine& engine, std::span<const std::uint32_t> mask,
           const gem16::SamplingOptions& options = {}) {
  Check(engine.Reset());
  Check(engine.ConfigureTokenSelection(options, mask));
  Check(engine.PrefillTokens(kPrompt));
}
}  // namespace

int main(int argc, char** argv) {
  const char* model = std::getenv("GEM16_26B_SELECTION_MODEL");
  if (argc == 3 && std::string_view(argv[1]) == "--model") model = argv[2];
  else if (argc != 1) return 2;
  if (model == nullptr || *model == '\0') {
    std::cout << "GEM16_26B_SELECTION_MODEL is required; not qualified\n";
    return 77;
  }
  try {
    auto created = Engine::Create(
        model, 4096U, 0,
        gem16::internal::Gemma4Moe26BBackend::kSm120Integrated);
    Require(created.ok(), created.status().message().c_str());
    Engine engine = std::move(created).value();
    std::vector<float> logits(kVocabulary);
    Start(engine, {});
    Check(engine.CopyLogits(logits));
    const auto initial_logits = logits;
    const auto winner = Oracle(logits, {});
    std::vector<std::uint32_t> ranked(kVocabulary);
    std::iota(ranked.begin(), ranked.end(), 0U);
    std::partial_sort(ranked.begin(), ranked.begin() + 16, ranked.end(),
        [&](auto a, auto b) {
          return logits[a] > logits[b] || (logits[a] == logits[b] && a < b);
        });
    const std::vector<std::vector<std::uint32_t>> masks{
        {}, {1U, 50U, 106U, 258882U, 258883U}, {winner},
        {winner, winner}, {0U, winner},
        {ranked.begin(), ranked.begin() + 16}};
    unsigned cases = 0U;
    for (const auto& mask : masks) {
      Start(engine, mask);
      Check(engine.CopyLogits(logits));
      Require(logits == initial_logits, "mask changed raw greedy logits");
      const auto expected = Oracle(logits, mask);
      auto before = engine.Prediction();
      Require(before.ok() && before.value().token == winner,
              "Prediction must retain unmasked winner");
      auto first = engine.SelectToken();
      auto after = engine.Prediction();
      auto repeat = engine.SelectToken();
      Require(first.ok() && first.value() == expected, "masked oracle mismatch");
      Require(repeat.ok() && repeat.value() == expected, "repeated selection changed");
      Require(after.ok() && after.value().token == winner &&
                  after.value().logit == before.value().logit,
              "selection changed Prediction API");
      Check(engine.ForwardToken(first.value()));
      Check(engine.CopyLogits(logits));
      const auto next_expected = Oracle(logits, mask);
      auto next = engine.SelectToken();
      Require(next.ok() && next.value() == next_expected,
              "self-feed/fallback next-token mismatch");
      ++cases;
    }
    // Exercise the untouched sampled/penalty branch across a resident reset.
    gem16::SamplingOptions options;
    options.enabled = true;
    options.seed = 42U;
    options.top_k = 40U;
    options.top_p = 0.9F;
    options.temperature = 0.8F;
    options.penalties = {0.1F, -0.1F};
    std::array<std::uint32_t, 4> sampled{};
    for (unsigned run = 0U; run < 2U; ++run) {
      Start(engine, masks[1], options);
      for (std::size_t i = 0U; i < sampled.size(); ++i) {
        auto token = engine.SelectToken();
        Require(token.ok(), "sampled selection failed");
        if (run == 0U) sampled[i] = token.value();
        else Require(sampled[i] == token.value(), "sampled reset changed sequence");
        Check(engine.ForwardToken(token.value()));
      }
    }
    const auto evidence = engine.execution_evidence();
    Require(evidence.fallback_count == 0U && evidence.recurring_allocation_count == 0U,
            "fallback or recurring allocation");
    std::cout << "{\"greedy_mask_cases\":" << cases
              << ",\"outputs_per_case\":2,\"sampled_reset_parity\":true,"
                 "\"fallbacks\":0,\"recurring_allocations\":0}\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "26B selection qualification: " << e.what() << '\n';
    return 1;
  }
}
