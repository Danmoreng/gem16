#include "gem16/engine.h"

namespace gem16 {

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
    const ReasoningTokenOptions&, GeneratedTokenCallback, void*,
    std::span<const AudioEmbeddingSegment>,
    std::span<const VisionEmbeddingSegment>) {
  return Status(StatusCode::kUnsupported,
                "conversation sessions require a CUDA build compiled for SM120a");
}

std::uint64_t ConversationSession::cached_token_count() const { return 0U; }

Result<GreedyInferenceResult> RunGreedyInference(const GreedyInferenceOptions&) {
  return Status(StatusCode::kUnsupported,
                "greedy inference requires a CUDA build compiled for SM120a");
}

Result<DecodeBenchmarkResult> RunDecodeBenchmark(const DecodeBenchmarkOptions&) {
  return Status(StatusCode::kUnsupported,
                "decode benchmarking requires a CUDA build compiled for SM120a");
}

}  // namespace gem16
