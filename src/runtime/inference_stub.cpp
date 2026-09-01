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
std::uint64_t ModelRuntime::assistant_workspace_bytes() const { return 0U; }
bool ModelRuntime::assistant_loaded() const { return false; }
std::uint64_t ModelRuntime::vision_weight_bytes() const { return 0U; }
bool ModelRuntime::vision_module_loaded() const { return false; }
double ModelRuntime::load_milliseconds() const { return 0.0; }
const char* ModelRuntime::weight_load_path() const { return "none"; }
const char* ModelRuntime::model_variant_name() const { return "unsupported"; }
const char* ModelRuntime::selected_native_path() const { return "none"; }
const char* ModelRuntime::artifact_profile() const { return "unsupported"; }
const char* ModelRuntime::head_format() const { return "unsupported"; }
const char* ModelRuntime::artifact_content_sha256() const { return ""; }
const char* ModelRuntime::source_lock_sha256() const { return ""; }
const char* ModelRuntime::compiler_commit() const { return ""; }
std::uint64_t ModelRuntime::max_context_tokens() const { return 0U; }
std::uint64_t ModelRuntime::default_context_tokens() const { return 0U; }
std::uint64_t ModelRuntime::base_max_context_tokens() const { return 0U; }
bool ModelRuntime::qualified_64k() const { return false; }
std::uint64_t ModelRuntime::kv_cache_bytes() const { return 0U; }
std::uint64_t ModelRuntime::workspace_bytes() const { return 0U; }
bool ModelRuntime::supports_audio() const { return false; }
bool ModelRuntime::supports_vision() const { return false; }
bool ModelRuntime::supports_mtp() const { return false; }
bool ModelRuntime::vision_mtp_supported() const { return false; }
std::uint32_t ModelRuntime::maximum_execution_slots() const { return 0U; }

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
    std::span<const VisionEmbeddingSegment>,
    std::span<const Gemma4Moe26BVisionInputSegment>) {
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
