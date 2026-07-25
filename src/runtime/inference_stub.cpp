#include "gem16gb/engine.h"

#include <ostream>

namespace gem16gb {

struct ConversationSession::Impl {};

ConversationSession::ConversationSession(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
ConversationSession::ConversationSession(ConversationSession&&) noexcept =
    default;
ConversationSession& ConversationSession::operator=(
    ConversationSession&&) noexcept = default;
ConversationSession::~ConversationSession() = default;

Result<ConversationSession> ConversationSession::Create(
    const ConversationSessionOptions&) {
  return Status(StatusCode::kUnsupported,
                "conversation sessions require a CUDA build compiled for SM120a");
}

Result<GreedyInferenceResult> ConversationSession::Generate(
    std::span<const std::uint32_t>, std::uint64_t,
    GeneratedTokenCallback, void*) {
  return Status(StatusCode::kUnsupported,
                "conversation sessions require a CUDA build compiled for SM120a");
}

std::uint64_t ConversationSession::cached_token_count() const { return 0U; }

Result<GreedyInferenceResult> RunGreedyInference(const GreedyInferenceOptions&) {
  return Status(StatusCode::kUnsupported,
                "greedy inference requires a CUDA build compiled for SM120a");
}

Status WriteGreedyInferenceJson(const GreedyInferenceResult&, std::ostream&) {
  return Status(StatusCode::kUnsupported,
                "greedy inference JSON requires a CUDA inference result");
}

Result<DecodeBenchmarkResult> RunDecodeBenchmark(const DecodeBenchmarkOptions&) {
  return Status(StatusCode::kUnsupported,
                "decode benchmarking requires a CUDA build compiled for SM120a");
}

Status WriteDecodeBenchmarkJson(const DecodeBenchmarkResult&, std::ostream&) {
  return Status(StatusCode::kUnsupported,
                "decode benchmark JSON requires a CUDA benchmark result");
}

Status WritePrefillBenchmarkJson(const DecodeBenchmarkResult&, std::ostream&) {
  return Status(StatusCode::kUnsupported,
                "prefill benchmark JSON requires a CUDA benchmark result");
}

}  // namespace gem16gb
