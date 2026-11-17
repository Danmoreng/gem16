// Diagnostic resident-chat audit. Compare output/state with Ordinary/parent;
// never use these timings as throughput evidence.
#include "gem16/engine.h"
#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void Require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
void Check(const gem16::Status& s) {
  if (!s.ok()) throw std::runtime_error(s.message());
}
struct Stream {
  std::array<std::uint32_t, 256> ids{};
  unsigned count = 0;
  unsigned cancel_after = 0;
};
gem16::Status Collect(void* opaque, std::uint32_t token) {
  auto& s = *static_cast<Stream*>(opaque);
  Require(s.count < s.ids.size(), "stream overflow");
  s.ids[s.count++] = token;
  if (s.cancel_after && s.count == s.cancel_after)
    return {gem16::StatusCode::kCancelled, "audit cancellation after seven tokens"};
  return gem16::Status::Ok();
}
bool first = true;
void Record(const std::string& name, const gem16::GreedyInferenceResult& r) {
  std::cout << (first ? "" : ",") << '"' << name << "\":{\"ids\":[";
  first = false;
  for (std::size_t i = 0; i < r.output_token_ids.size(); ++i)
    std::cout << (i ? "," : "") << r.output_token_ids[i];
  std::cout << "],\"cached\":" << r.prompt_cached_tokens
            << ",\"reasoning_tokens\":" << r.reasoning_tokens
            << ",\"forced\":" << (r.reasoning_budget_forced ? "true" : "false")
            << ",\"stopped\":" << (r.stopped ? "true" : "false") << '}';
}
}

int main(int argc, char** argv) {
  try {
    Require(argc == 6, "usage: audit target assistant depth sampled expect-chain");
    const unsigned depth = static_cast<unsigned>(std::stoul(argv[3]));
    const bool sampled = std::stoul(argv[4]) != 0;
    const bool expect_chain = std::stoul(argv[5]) != 0;
    Require(depth == 0 || depth == 2 || depth == 3 || depth == 4, "invalid depth");
    const std::array<std::uint32_t, 20> prompt{
      2,105,2364,107,40654,607,7121,506,3658,3730,236761,106,107,
      105,4368,107,100,45518,107,101};
    gem16::ConversationSessionOptions options;
    options.model_directory = argv[1];
    if (depth) options.assistant_model_directory = argv[2];
    options.mtp_draft_tokens = depth;
    options.max_context_tokens = 4096;
    options.suppressed_token_ids = {1,50,106,258882,258883};
    options.sampling.enabled = sampled;
    options.sampling.temperature = 1.0F;
    options.sampling.top_k = 64;
    options.sampling.top_p = 0.95F;
    options.sampling.repetition_penalty = 1.15F;
    options.sampling.seed = 123;
    gem16::ModelRuntimeOptions ro;
    ro.model_directory = options.model_directory;
    ro.assistant_model_directory = options.assistant_model_directory;
    ro.max_context_tokens = options.max_context_tokens;
    auto loaded = gem16::ModelRuntime::Load(ro);
    Require(loaded.ok(), "runtime load failed");
    auto runtime = std::move(loaded).value();
    auto made = gem16::ConversationSession::Create(runtime, options);
    Require(made.ok(), "session create failed");
    auto session = std::move(made).value();
    const auto generate = [&](std::span<const std::uint32_t> input, unsigned n,
                              const gem16::ReasoningTokenOptions& reasoning = {},
                              const gem16::SamplingPenalties& penalties = {}) {
      Stream stream;
      auto generated = session.Generate(input, n, reasoning, Collect, &stream,
                                       {}, {}, {}, {}, penalties);
      if (!generated.ok()) throw std::runtime_error(generated.status().message());
      auto r = std::move(generated).value();
      Require(r.output_token_ids.size() == n, "wrong output length");
      Require(stream.count == n && std::equal(r.output_token_ids.begin(),
                  r.output_token_ids.end(), stream.ids.begin()), "stream IDs differ");
      Require(!r.token_loop_allocations && !r.fallback_count, "runtime invariant");
      Require(r.mtp_gpu_chained == (expect_chain && n >= depth + 2U), "wrong chain route");
      return r;
    };
    std::cout << "{\"cases\":{";
    auto baseline = generate(prompt, 32);
    Record("first", baseline);
    std::vector<std::uint32_t> continuation(prompt.begin(), prompt.end());
    continuation.insert(continuation.end(), baseline.output_token_ids.begin(),
                        baseline.output_token_ids.end());
    continuation.push_back(107);
    gem16::SamplingPenalties penalties;
    if (sampled) { penalties.frequency = 0.2F; penalties.presence = 0.3F; }
    auto cached = generate(continuation, 16, {}, penalties);
    Require(cached.prompt_cached_tokens > 0, "continuation did not reuse cache");
    Record("cached_penalties", cached);
    Check(session.Reset());
    auto repeated = generate(prompt, 32);
    Require(repeated.output_token_ids == baseline.output_token_ids, "reset changed output");
    Record("reset", repeated);
    for (unsigned n = 1; n <= 6; ++n) {
      Check(session.Reset());
      auto r = generate(prompt, n);
      Require(std::equal(r.output_token_ids.begin(), r.output_token_ids.end(),
                         baseline.output_token_ids.begin()), "tail changed prefix");
      Record("tail_" + std::to_string(n), r);
    }
    gem16::ReasoningTokenOptions reasoning;
    reasoning.enabled = true;
    reasoning.starts_in_reasoning = true;
    reasoning.channel_open_token_ids = {105,4368};
    reasoning.channel_close_token_id = 99;
    for (unsigned budget : {1U,4U,5U,6U,9U}) {
      Check(session.Reset());
      reasoning.max_reasoning_tokens = budget;
      auto r = generate(prompt, 32, reasoning);
      Require(r.reasoning_tokens <= budget, "reasoning budget exceeded");
      Record("budget_" + std::to_string(budget), r);
    }
    Check(session.Reset());
    reasoning.starts_in_reasoning = false;
    reasoning.max_reasoning_tokens = 6;
    reasoning.channel_open_token_ids = {baseline.output_token_ids[0], baseline.output_token_ids[1]};
    Record("partial_opener", generate(prompt, 32, reasoning));
    Check(session.Reset());
    Stream cancellation;
    cancellation.cancel_after = 7;
    auto cancelled = session.Generate(prompt, 64, {}, Collect, &cancellation);
    Require(!cancelled.ok() && cancelled.status().code() == gem16::StatusCode::kCancelled,
            "callback cancellation failed");
    Require(cancellation.count == 7 && session.is_poisoned(), "cancellation boundary");
    Check(session.Reset());
    auto relaunched = generate(prompt, 32);
    Require(relaunched.output_token_ids == baseline.output_token_ids, "cancel/reset changed output");
    Record("cancel_reset", relaunched);
    auto stop_options = options;
    auto stop = std::find_if(baseline.output_token_ids.begin() + 3,
                            baseline.output_token_ids.end(),
                            [&](auto t) { return t != baseline.output_token_ids[0]; });
    Require(stop != baseline.output_token_ids.end(), "no interior stop fixture");
    stop_options.stop_token_ids = {*stop};
    auto stop_made = gem16::ConversationSession::Create(runtime, stop_options);
    Require(stop_made.ok(), "stop session create failed");
    auto stop_session = std::move(stop_made).value();
    Stream stopped_stream;
    auto stopped = stop_session.Generate(prompt, 32, {}, Collect, &stopped_stream);
    Require(stopped.ok() && stopped.value().stopped, "stop token ignored");
    Require(stopped.value().output_token_ids.back() == *stop, "wrong stop token");
    Require(stopped_stream.count == stopped.value().output_token_ids.size() &&
            std::equal(stopped.value().output_token_ids.begin(), stopped.value().output_token_ids.end(),
                       stopped_stream.ids.begin()), "stopped stream mismatch");
    Record("stop", stopped.value());
    std::cout << "},\"stream_exact\":true,\"cancel_after\":7}\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
