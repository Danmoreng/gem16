#include "decode_replay_support.h"
#include "cuda/engine/gemma4_26b_reference.h"

namespace {
void Check(const gem16::Status& status) {
  if (!status.ok()) throw std::runtime_error(status.message());
}
}  // namespace

int main(int argc, char** argv) {
  using namespace decode_replay;
  if (argc != 5 && argc != 6) return 2;
  try {
    const auto prompt = ReadTokens(argv[2], 16384U);
    const auto history = ReadTokens(argv[3], 1135U);
    const bool windows = std::string_view(argv[4]) == "d2-windows";
    const bool d2 = windows || std::string_view(argv[4]) == "d2";
    Require(d2 || std::string_view(argv[4]) == "ordinary", "invalid replay mode");
    Require(!d2 || argc == 6, "D2 requires the pinned assistant");
    auto created = gem16::internal::Gemma4Moe26BReferenceEngine::Create(
        argv[1], kContext, 0,
        gem16::internal::Gemma4Moe26BBackend::kSm120Integrated, true, true);
    if (!created.ok()) throw std::runtime_error(created.status().message());
    auto engine = std::move(created).value();
    Check(engine.ConfigureTokenSelection({}, kMask));
    std::vector<Row> rows(1134U);
    if (!d2) {
      Check(engine.PrefillTokens(prompt));
      auto initial = engine.SelectToken();
      Require(initial.ok(), "native initial replay selection failed");
      for (std::size_t i = 0U; i < rows.size(); ++i) {
        const auto start = Clock::now();
        Check(engine.ForwardToken(history[i]));
        auto prediction = engine.SelectToken();
        Require(prediction.ok(), "native replay selection failed");
        rows[i] = {history[i], prediction.value(), Ns(start)};
        if (i == 0U) LabSave(0U);
        if (i == 567U) LabSave(1U);
        if (i == 1133U) LabSave(2U);
      }
      const auto evidence = engine.execution_evidence();
      Require(evidence.fallback_count == 0U && evidence.recurring_allocation_count == 0U,
              "native replay fallback or allocation");
      WriteRows("native-trellis35", rows, initial.value());
    } else {
      Check(engine.LoadMtpAssistant(argv[5]));
      Check(engine.ConfigureMtpVerifierBackend(
          gem16::internal::Gemma4Moe26BMtpVerifierBackend::kExactSharedBatchedMoe));
      if (windows) {
        constexpr std::array<std::size_t, 3> starts{0U, 567U, 1128U};
        std::cout << "{\"diagnostic_only\":true,\"engine\":\"native-trellis35\","
                     "\"mode\":\"forced_d2_windows\",\"cases\":[";
        for (unsigned phase = 0U; phase < starts.size(); ++phase) {
          Check(engine.Reset());
          Check(engine.PrefillTokens(prompt));
          for (std::size_t i = 0U; i < starts[phase]; ++i) Check(engine.ForwardToken(history[i]));
          std::array<std::uint32_t, 2> drafts{history[starts[phase] + 1U], history[starts[phase] + 2U]};
          gem16::internal::MtpGroupResult result{};
          Check(engine.RunForcedMtpProposalDiagnostic(history[starts[phase]], drafts, &result));
          LabSave(phase);
          std::cout << (phase ? "," : "") << "{\"position\":" << 16384U + starts[phase]
                    << ",\"accepted\":" << result.accepted_count << ",\"outputs\":[";
          for (unsigned i = 0U; i < result.output_count; ++i) std::cout << (i ? "," : "") << result.verified[i];
          std::cout << "]}";
        }
        std::cout << "]}\n";
        LabFinish();
        return 0;
      }
      std::array<gem16::internal::MtpGroupResult, 3> results{};
      // The two pinned histories share the early prefix. The same three draft
      // pairs induce 0/1/2 acceptance in each engine; routing remains its own.
      for (std::uint32_t accepted = 0U; accepted < 3U; ++accepted) {
        Check(engine.Reset());
        Check(engine.PrefillTokens(prompt));
        std::array<std::uint32_t, 2> drafts{history[1], history[2]};
        if (accepted < 2U) drafts[accepted] = (drafts[accepted] + 1U) % kVocabulary;
        Check(engine.RunForcedMtpProposalDiagnostic(history[0], drafts, &results[accepted]));
        const auto& r = results[accepted];
        Require(r.accepted_count == accepted, "unexpected native forced acceptance");
        Require(r.output_count == accepted + 1U, "incorrect native forced output count");
        for (std::uint32_t i = 0U; i < r.output_count; ++i) {
          Require(r.verified[i] == history[i + 1U], "native forced output mismatch");
        }
        Require(engine.position() == prompt.size() + accepted + 1U,
                "native forced commit position mismatch");
        Check(engine.ForwardToken(r.verified[r.output_count - 1U]));
        auto next = engine.SelectToken();
        Require(next.ok() && next.value() == history[accepted + 2U],
                "native post-rollback continuation mismatch");
        LabSave(accepted);
      }
      std::cout << "{\"diagnostic_only\":true,\"engine\":\"native-trellis35\","
                   "\"mode\":\"forced_d2\",\"accepted\":[0,1,2],"
                   "\"commit_and_continuation_exact\":true}\n";
    }
    LabFinish();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "native diagnostic replay: " << error.what() << '\n';
    return 1;
  }
}
