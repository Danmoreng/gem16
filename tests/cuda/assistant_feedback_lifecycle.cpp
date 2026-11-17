// AP4 offline parent/candidate lifecycle and consumed-state comparison.
#include "decode_replay_support.h"
#include "cuda/engine/gemma4_26b_reference.h"
#include "gem16/engine.h"

using namespace decode_replay;
using Engine = gem16::internal::Gemma4Moe26BReferenceEngine;
namespace {
void Check(const gem16::Status& s) {
  if (!s.ok()) throw std::runtime_error(s.message());
}
template <class T>
void Save(const std::filesystem::path& path, const std::vector<T>& values) {
  Require(!std::filesystem::exists(path), "refusing to replace lifecycle output");
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char*>(values.data()), values.size()*sizeof(T));
  Require(static_cast<bool>(f), "lifecycle output write failed");
}
gem16::Status Cancel(void* opaque, std::uint32_t) {
  ++*static_cast<unsigned*>(opaque);
  return {gem16::StatusCode::kCancelled, "AP4 intentional cancellation"};
}
}
int main(int argc, char** argv) {
  try {
    if (argc != 5) return 2;
    const auto history = ReadTokens(argv[3], 32);
    const std::filesystem::path out(argv[4]);
    Require(!std::filesystem::exists(out), "lifecycle directory exists");
    std::filesystem::create_directory(out);
    const std::array<std::uint32_t,20> prompt{
      2,105,2364,107,40654,607,7121,506,3658,3730,236761,106,107,
      105,4368,107,100,45518,107,101};
    {
      auto created = Engine::Create(argv[1], 17519, 0,
          gem16::internal::Gemma4Moe26BBackend::kSm120Integrated, false);
      Require(created.ok(), "lifecycle engine create failed");
      auto engine = std::move(created).value();
      Check(engine.ConfigureTokenSelection({}, kMask));
      Check(engine.LoadMtpAssistant(argv[2]));
      Check(engine.ConfigureMtpVerifierBackend(
          gem16::internal::Gemma4Moe26BMtpVerifierBackend::kExactSharedBatchedMoe));
      std::vector<std::uint32_t> observed;
      for (unsigned repeat=0; repeat<2; ++repeat) {
        Check(engine.Reset());
        Check(engine.PrefillTokens(prompt));
        auto pending = engine.SelectToken();
        Require(pending.ok() && pending.value()==history[0], "pending mismatch");
        unsigned slot=0;
        std::vector<std::uint32_t> previous_d2;
        for (unsigned depth : {1U,2U,4U,2U}) {
          std::vector<std::uint32_t> drafts(depth);
          Check(engine.GenerateMtpAssistantDraftsForPending(pending.value(), drafts));
          Require(engine.position()==prompt.size(), "proposal changed committed position");
          if (depth==2) {
            if (!previous_d2.empty()) Require(previous_d2==drafts, "repeated D2 changed");
            previous_d2=drafts;
          }
          observed.insert(observed.end(), drafts.begin(), drafts.end());
          std::vector<float> concatenated(5632), logits(262144);
          std::vector<std::uint8_t> sk(2048), sv(2048), fk(1024), fv(1024);
          std::vector<std::uint16_t> scales(4);
          Check(engine.CopyMtpAssistantOracleInputs(concatenated,logits,sk,sv,fk,fv,scales));
          const auto stem=std::to_string(repeat)+"-"+std::to_string(slot++);
          Save(out/(stem+"-concatenated.f32"),concatenated);
          Save(out/(stem+"-logits.f32"),logits);
          Save(out/(stem+"-sk.bin"),sk); Save(out/(stem+"-sv.bin"),sv);
          Save(out/(stem+"-fk.bin"),fk); Save(out/(stem+"-fv.bin"),fv);
          Save(out/(stem+"-scales.bin"),scales);
        }
      }
      // Exercise all commit lengths after executing the changed proposal path.
      for (unsigned accepted=0; accepted<3; ++accepted) {
        Check(engine.Reset()); Check(engine.PrefillTokens(prompt));
        std::array<std::uint32_t,2> actual{};
        Check(engine.GenerateMtpAssistantDraftsForPending(history[0],actual));
        std::array<std::uint32_t,2> forced{history[1],history[2]};
        if (accepted<2) forced[accepted]=(forced[accepted]+1)%kVocabulary;
        gem16::internal::MtpGroupResult result{};
        Check(engine.RunForcedMtpProposalDiagnostic(history[0],forced,&result));
        Require(result.accepted_count==accepted && result.output_count==accepted+1,
                "forced acceptance mismatch");
        for (unsigned i=0;i<result.output_count;++i)
          Require(result.verified[i]==history[i+1], "commit token mismatch");
        Require(engine.position()==prompt.size()+accepted+1,"commit position mismatch");
        Check(engine.ForwardToken(result.verified[result.output_count-1]));
        const auto next=engine.SelectToken();
        Require(next.ok() && next.value()==history[accepted+2],"rollback continuation mismatch");
      }
      const auto evidence=engine.execution_evidence();
      Require(!evidence.fallback_count && !evidence.recurring_allocation_count,
              "lifecycle fallback/allocation");
      Save(out/"drafts.u32",observed);
    }
    gem16::ConversationSessionOptions options;
    options.model_directory=argv[1]; options.assistant_model_directory=argv[2];
    options.max_context_tokens=4096; options.mtp_draft_tokens=2;
    options.suppressed_token_ids.assign(kMask.begin(),kMask.end());
    auto created=gem16::ConversationSession::Create(options);
    Require(created.ok(),"D2 product session create failed");
    auto session=std::move(created).value();
    auto first=session.Generate(prompt,16);
    Require(first.ok(),"first D2 request failed");
    Save(out/"session-first.u32",first.value().output_token_ids);
    std::vector<std::uint32_t> continuation(prompt.begin(),prompt.end());
    continuation.insert(continuation.end(),first.value().output_token_ids.begin(),
                        first.value().output_token_ids.end());
    continuation.push_back(107);
    auto second=session.Generate(continuation,8);
    Require(second.ok() && second.value().prompt_cached_tokens>0,
            "cached D2 continuation failed");
    Save(out/"session-continuation.u32",second.value().output_token_ids);
    Check(session.Reset());
    unsigned callbacks=0;
    auto cancelled=session.Generate(prompt,16,{},Cancel,&callbacks);
    Require(!cancelled.ok() && cancelled.status().code()==gem16::StatusCode::kCancelled
            && callbacks==1 && session.is_poisoned(),"D2 cancellation did not poison");
    auto poisoned=session.Generate(prompt,1);
    Require(!poisoned.ok(),"poisoned D2 session accepted generation");
    Check(session.Reset());
    auto relaunched=session.Generate(prompt,16);
    Require(relaunched.ok() && relaunched.value().output_token_ids==first.value().output_token_ids,
            "post-cancel D2 reset changed output");
    std::cout << "{\"draft_depths\":[1,2,4,2],\"repeats\":2,\"oracle_export\":true,"
                 "\"forced_commit_lengths\":[0,1,2],\"continuation\":true,"
                 "\"cancel_poison_reset\":true}\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "AP4 lifecycle: " << e.what() << '\n'; return 1;
  }
}
