#include "gem16/engine.h"

namespace gem16 {

Result<DeviceMemoryInfo> QueryDeviceMemoryInfo() {
  return Status(StatusCode::kUnsupported,
                "device memory queries require a CUDA build compiled for SM120a");
}

struct ModelRuntime::Impl {};
ModelRuntime::ModelRuntime(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
ModelRuntime::~ModelRuntime() = default;
Result<std::shared_ptr<ModelRuntime>> ModelRuntime::Load(
    const ModelRuntimeOptions&) {
  return Status(StatusCode::kUnsupported,
                "model runtimes require a CUDA build compiled for SM120a");
}
std::uint64_t ModelRuntime::weight_bytes() const { return 0U; }
std::uint64_t ModelRuntime::assistant_weight_bytes() const { return 0U; }
bool ModelRuntime::assistant_loaded() const { return false; }
double ModelRuntime::load_milliseconds() const { return 0.0; }

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

Result<ConversationSession> ConversationSession::Create(
    std::shared_ptr<ModelRuntime>, const ConversationSessionOptions&) {
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

std::uint64_t ConversationSession::reserved_device_bytes() const { return 0U; }

bool ConversationSession::is_poisoned() const { return true; }

Result<GreedyInferenceResult> RunGreedyInference(const GreedyInferenceOptions&) {
  return Status(StatusCode::kUnsupported,
                "greedy inference requires a CUDA build compiled for SM120a");
}

Result<DecodeBenchmarkResult> RunDecodeBenchmark(const DecodeBenchmarkOptions&) {
  return Status(StatusCode::kUnsupported,
                "decode benchmarking requires a CUDA build compiled for SM120a");
}

}  // namespace gem16
