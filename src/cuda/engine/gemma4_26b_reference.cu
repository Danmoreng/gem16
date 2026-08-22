#include "cuda/engine/gemma4_26b_reference.h"

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cuda/attention/gemma4_26b_reference.h"
#include "cuda/engine/gemma4_26b_artifact.h"
#include "cuda/layer/reference.h"
#include "cuda/moe/prefill.h"
#include "cuda/moe/reference.h"
#include "cuda/nvfp4/reference.h"
#include "cuda/nvfp4/sm120.h"
#include "gem16/model.h"
#include "model/config.h"
#include "model/gemma4_26b_attention.h"
#include "model/gemma4_26b_residency.h"

namespace gem16::internal {
namespace {

constexpr std::uint64_t kWidth = 2816U;
constexpr std::uint64_t kVocabulary = 262144U;
constexpr std::uint64_t kShared = 2112U;
constexpr std::uint64_t kExpert = 704U;
constexpr std::uint32_t kExperts = 128U;
constexpr std::uint32_t kTopK = 8U;
constexpr std::uint64_t kLayers = 30U;
constexpr std::uint64_t kPrefillMaxTokens = 128U;
constexpr std::uint64_t kPrefillScoreElements =
    64U * 1024U * 1024U / sizeof(float);
constexpr std::uint64_t kNvfp4Block = 16U;
constexpr std::uint64_t kSm120KBlock = 64U;
constexpr std::uint64_t kRowsPerTile = 8U;
constexpr std::array<std::uint32_t, 4> kCaptureLayers{0U, 5U, 6U, 29U};

Status Invalid(std::string message) {
  return Status(StatusCode::kInvalidArgument, std::move(message));
}

Status CudaFailure(const char* operation, cudaError_t error) {
  return Status(StatusCode::kInternal,
                std::string(operation) + ": " + cudaGetErrorName(error) +
                    ": " + cudaGetErrorString(error));
}

class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  ~DeviceBuffer() {
    if (pointer_ != nullptr) (void)cudaFree(pointer_);
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : pointer_(std::exchange(other.pointer_, nullptr)),
        bytes_(std::exchange(other.bytes_, 0U)) {}
  Status Allocate(std::uint64_t bytes, const char* label) {
    if (bytes == 0U || bytes > std::numeric_limits<std::size_t>::max()) {
      return Invalid(std::string(label) + " has an invalid size");
    }
    const cudaError_t error =
        cudaMalloc(&pointer_, static_cast<std::size_t>(bytes));
    if (error != cudaSuccess) return CudaFailure(label, error);
    bytes_ = bytes;
    return Status::Ok();
  }
  template <typename T>
  T* As(std::uint64_t offset = 0U) const {
    return reinterpret_cast<T*>(static_cast<std::byte*>(pointer_) + offset);
  }
  std::uint64_t bytes() const { return bytes_; }

 private:
  void* pointer_ = nullptr;
  std::uint64_t bytes_ = 0U;
};

struct LayoutBuilder {
  std::uint64_t bytes = 0U;
  template <typename T>
  std::uint64_t Add(std::uint64_t elements) {
    constexpr std::uint64_t alignment = 256U;
    bytes = (bytes + alignment - 1U) & ~(alignment - 1U);
    const std::uint64_t offset = bytes;
    if (elements > (std::numeric_limits<std::uint64_t>::max() - bytes) /
                       sizeof(T)) {
      bytes = std::numeric_limits<std::uint64_t>::max();
    } else {
      bytes += elements * sizeof(T);
    }
    return offset;
  }
};

__device__ __forceinline__ std::uint64_t DeviceTiledPackedOffset(
    std::uint64_t row, std::uint64_t column, std::uint64_t k_blocks) {
  return (((row / kRowsPerTile) * k_blocks + column / kSm120KBlock) *
               kRowsPerTile +
           row % kRowsPerTile) *
              (kSm120KBlock / 2U) +
          (column % kSm120KBlock) / 2U;
}

__device__ __forceinline__ std::uint64_t DeviceTiledScaleOffset(
    std::uint64_t row, std::uint64_t column, std::uint64_t k_blocks) {
  return (((row / kRowsPerTile) * k_blocks + column / kSm120KBlock) *
               kRowsPerTile +
           row % kRowsPerTile) *
              (kSm120KBlock / kNvfp4Block) +
          (column % kSm120KBlock) / kNvfp4Block;
}

__global__ void TiledEmbeddingLookupKernel(
    const std::uint8_t* packed, const std::uint8_t* scales, float divisor,
    std::uint32_t token, float* output) {
  const std::uint64_t column =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (column >= kWidth) return;
  const std::uint64_t k_blocks = kWidth / kSm120KBlock;
  const std::uint64_t offset =
      DeviceTiledPackedOffset(token, column, k_blocks);
  const std::uint8_t byte = packed[offset];
  __nv_fp4_e2m1 value;
  value.__x = static_cast<std::uint8_t>(
      (byte >> ((column & 1U) == 0U ? 0U : 4U)) & 0x0FU);
  __nv_fp8_e4m3 scale;
  scale.__x = scales[DeviceTiledScaleOffset(token, column, k_blocks)];
  const float embedding_scale = static_cast<float>(
      __float2bfloat16_rn(sqrtf(static_cast<float>(kWidth))));
  output[column] = static_cast<float>(__float2bfloat16_rn(
      static_cast<float>(value) * static_cast<float>(scale) / divisor *
      embedding_scale));
}

__global__ void TiledEmbeddingLookupBatchKernel(
    const std::uint8_t* packed, const std::uint8_t* scales, float divisor,
    const std::uint32_t* tokens, float* output, std::uint64_t elements) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  const std::uint64_t token_index = index / kWidth;
  const std::uint64_t column = index % kWidth;
  const std::uint64_t row = tokens[token_index];
  const std::uint64_t k_blocks = kWidth / kSm120KBlock;
  const std::uint64_t offset = DeviceTiledPackedOffset(row, column, k_blocks);
  const std::uint8_t byte = packed[offset];
  __nv_fp4_e2m1 value;
  value.__x = static_cast<std::uint8_t>(
      (byte >> ((column & 1U) == 0U ? 0U : 4U)) & 0x0FU);
  __nv_fp8_e4m3 scale;
  scale.__x = scales[DeviceTiledScaleOffset(row, column, k_blocks)];
  const float embedding_scale = static_cast<float>(
      __float2bfloat16_rn(sqrtf(static_cast<float>(kWidth))));
  output[index] = static_cast<float>(__float2bfloat16_rn(
      static_cast<float>(value) * static_cast<float>(scale) / divisor *
      embedding_scale));
}

__global__ void TiledEmbeddingLookupControlledKernel(
    const std::uint8_t* packed, const std::uint8_t* scales, float divisor,
    const DecodeControl* control, float* output) {
  const std::uint64_t column =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (column >= kWidth) return;
  const std::uint64_t row = control->token;
  const std::uint64_t k_blocks = kWidth / kSm120KBlock;
  const std::uint64_t offset = DeviceTiledPackedOffset(row, column, k_blocks);
  const std::uint8_t byte = packed[offset];
  __nv_fp4_e2m1 value;
  value.__x = static_cast<std::uint8_t>(
      (byte >> ((column & 1U) == 0U ? 0U : 4U)) & 0x0FU);
  __nv_fp8_e4m3 scale;
  scale.__x = scales[DeviceTiledScaleOffset(row, column, k_blocks)];
  const float embedding_scale = static_cast<float>(
      __float2bfloat16_rn(sqrtf(static_cast<float>(kWidth))));
  output[column] = static_cast<float>(__float2bfloat16_rn(
      static_cast<float>(value) * static_cast<float>(scale) / divisor *
      embedding_scale));
}

__global__ void CapturePrefillRouterIdsKernel(
    const Gemma4MoePrefillAssignment* assignments, std::uint32_t* output,
    std::uint64_t token) {
  const std::uint32_t slot = threadIdx.x;
  if (blockIdx.x == 0U && slot < kTopK) {
    output[slot] = assignments[token * kTopK + slot].expert_id;
  }
}

__global__ void SetDecodeControlKernel(DecodeControl* control,
                                       std::uint32_t token,
                                       std::uint64_t position) {
  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    *control = DecodeControl{token, 0U, position, position};
  }
}

__global__ void SoftcapFiniteKernel(float* logits, float softcap,
                                    int* all_finite) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= kVocabulary) return;
  const float value = tanhf(logits[index] / softcap) * softcap;
  logits[index] = value;
  if (!isfinite(value)) atomicExch(all_finite, 0);
}

__global__ void DeterministicArgmaxKernel(const float* logits,
                                          std::uint32_t* token,
                                          float* value) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) return;
  float best = -3.402823466e+38F;
  std::uint32_t best_id = 0U;
  for (std::uint32_t id = 0U; id < kVocabulary; ++id) {
    const float candidate = logits[id];
    if (candidate > best || (candidate == best && id < best_id)) {
      best = candidate;
      best_id = id;
    }
  }
  *token = best_id;
  *value = best;
}

int CaptureIndex(std::uint32_t layer) {
  for (std::size_t index = 0; index < kCaptureLayers.size(); ++index) {
    if (kCaptureLayers[index] == layer) return static_cast<int>(index);
  }
  return -1;
}

template <typename T>
Result<const T*> ArtifactPointer(const Gemma4Moe26BDeviceArtifact& artifact,
                                 const char* name) {
  auto pointer = artifact.Pointer(name);
  if (!pointer.ok()) return pointer.status();
  return reinterpret_cast<const T*>(pointer.value());
}

}  // namespace

struct Gemma4Moe26BReferenceEngine::Impl {
  int device = 0;
  std::uint64_t context = 0U;
  std::uint64_t position = 0U;
  Gemma4Moe26BBackend backend = Gemma4Moe26BBackend::kReference;
  cudaStream_t stream = nullptr;
  Gemma4Moe26BDeviceArtifact artifact;
  Gemma4Moe26BAttentionTraits traits{};
  std::array<Gemma4Moe26BAttentionReferenceWeights, kLayers> attention_weights{};
  std::array<Gemma4MoeReferenceWeights, kLayers> moe_weights{};
  std::array<Gemma4Moe26BKvCacheView, kLayers> caches{};
  Gemma4Moe26BAttentionReferenceWorkspace attention_workspace{};
  Gemma4MoeReferenceWorkspace moe_workspace{};
  Gemma4MoeReferenceConfig moe_config{kWidth, kShared, kExpert, kExperts,
                                      kTopK, 1.0e-6F};
  Gemma4MoeNvfp4Matrix head{};
  const std::uint16_t* final_norm = nullptr;
  float softcap = 30.0F;
  DeviceBuffer kv;
  DeviceBuffer workspace;
  DeviceBuffer prefill_workspace;
  DecodeControl* decode_control = nullptr;
  cudaGraphExec_t decode_graph = nullptr;
  std::uint32_t* prefill_tokens = nullptr;
  std::uint32_t* prefill_host_tokens = nullptr;
  float* prefill_hidden_a = nullptr;
  float* prefill_hidden_b = nullptr;
  Gemma4Moe26BAttentionReferenceWorkspace prefill_attention_workspace{};
  Gemma4MoePrefillWorkspace prefill_moe_workspace{};
  float* hidden_a = nullptr;
  float* hidden_b = nullptr;
  float* final_hidden = nullptr;
  std::uint8_t* head_activation = nullptr;
  std::uint8_t* head_activation_scales = nullptr;
  float* logits = nullptr;
  std::uint32_t* prediction_token = nullptr;
  float* prediction_logit = nullptr;
  int* finite = nullptr;
  std::array<float*, kCaptureLayers.size()> layer_captures{};
  std::array<float*, kCaptureLayers.size()> router_probability_captures{};
  std::array<std::uint32_t*, kCaptureLayers.size()> router_id_captures{};

  Status LaunchControlledDecodeBody();
  Status PrepareDecodeGraph();

  ~Impl() {
    if (stream != nullptr) {
      (void)cudaStreamSynchronize(stream);
    }
    if (decode_graph != nullptr) (void)cudaGraphExecDestroy(decode_graph);
    if (prefill_host_tokens != nullptr) {
      (void)cudaFreeHost(prefill_host_tokens);
    }
    if (stream != nullptr) {
      (void)cudaStreamDestroy(stream);
    }
  }
};

Status Gemma4Moe26BReferenceEngine::Impl::LaunchControlledDecodeBody() {
  if (decode_control == nullptr) {
    return Invalid("M17 decode graph control is not initialized");
  }
  cudaError_t error = cudaSuccess;
  constexpr unsigned threads = 256U;
  TiledEmbeddingLookupControlledKernel<<<
      static_cast<unsigned>((kWidth + threads - 1U) / threads), threads, 0,
      stream>>>(head.packed_e2m1, head.scales_e4m3fn,
                head.weight_global_divisor, decode_control, hidden_a);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaFailure("launch M17 controlled embedding", error);
  }
  for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
    Status status = LaunchGemma4Moe26BAttentionReferenceControlledLayer(
        hidden_a, hidden_b, traits[layer], attention_weights[layer],
        caches[layer], attention_workspace, decode_control, 1.0e-6F, stream);
    if (!status.ok()) return status;
    status = LaunchGemma4MoeSm120Layer(
        hidden_b, hidden_a, moe_config, moe_weights[layer], moe_workspace,
        stream);
    if (!status.ok()) return status;
    const int capture = CaptureIndex(layer);
    if (capture >= 0) {
      const std::size_t index = static_cast<std::size_t>(capture);
      error = cudaMemcpyAsync(layer_captures[index], hidden_a,
                              kWidth * sizeof(float),
                              cudaMemcpyDeviceToDevice, stream);
      if (error == cudaSuccess) {
        error = cudaMemcpyAsync(
            router_probability_captures[index],
            moe_workspace.router_probabilities, kExperts * sizeof(float),
            cudaMemcpyDeviceToDevice, stream);
      }
      if (error == cudaSuccess) {
        error = cudaMemcpyAsync(router_id_captures[index],
                                moe_workspace.top_ids,
                                kTopK * sizeof(std::uint32_t),
                                cudaMemcpyDeviceToDevice, stream);
      }
      if (error != cudaSuccess) {
        return CudaFailure("capture M17 decode layer", error);
      }
    }
  }
  Status status = LaunchRmsNormBf16(hidden_a, final_norm, final_hidden, 1U,
                                    kWidth, 1.0e-6F, stream);
  if (!status.ok()) return status;
  status = LaunchNvfp4ReferenceActivationQuantization(
      final_hidden, head_activation, head_activation_scales, kWidth,
      head.activation_global_divisor, stream);
  if (!status.ok()) return status;
  status = LaunchNvfp4Sm120DirectProjectionBf16Float(
      head_activation, head_activation_scales, head.packed_e2m1,
      head.scales_e4m3fn, logits, head.rows, head.columns,
      head.activation_global_divisor, head.weight_global_divisor, stream);
  if (!status.ok()) return status;
  error = cudaMemsetAsync(finite, 1, sizeof(int), stream);
  if (error != cudaSuccess) {
    return CudaFailure("initialize M17 decode finite flag", error);
  }
  SoftcapFiniteKernel<<<
      static_cast<unsigned>((kVocabulary + threads - 1U) / threads), threads,
      0, stream>>>(logits, softcap, finite);
  error = cudaGetLastError();
  if (error != cudaSuccess) return CudaFailure("launch M17 softcap", error);
  DeterministicArgmaxKernel<<<1, 1, 0, stream>>>(
      logits, prediction_token, prediction_logit);
  error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch M17 argmax", error);
}

Status Gemma4Moe26BReferenceEngine::Impl::PrepareDecodeGraph() {
  cudaError_t error = cudaStreamSynchronize(stream);
  if (error != cudaSuccess) {
    return CudaFailure("synchronize before M17 graph capture", error);
  }
  error = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
  if (error != cudaSuccess) return CudaFailure("begin M17 graph capture", error);
  const Status body = LaunchControlledDecodeBody();
  cudaGraph_t graph = nullptr;
  error = cudaStreamEndCapture(stream, &graph);
  if (!body.ok()) {
    if (graph != nullptr) (void)cudaGraphDestroy(graph);
    return body;
  }
  if (error != cudaSuccess) {
    if (graph != nullptr) (void)cudaGraphDestroy(graph);
    return CudaFailure("end M17 graph capture", error);
  }
  error = cudaGraphInstantiate(&decode_graph, graph, nullptr, nullptr, 0U);
  const cudaError_t destroy_error = cudaGraphDestroy(graph);
  if (error != cudaSuccess) return CudaFailure("instantiate M17 graph", error);
  if (destroy_error != cudaSuccess) {
    (void)cudaGraphExecDestroy(decode_graph);
    decode_graph = nullptr;
    return CudaFailure("destroy captured M17 graph", destroy_error);
  }
  return Status::Ok();
}

Gemma4Moe26BReferenceEngine::Gemma4Moe26BReferenceEngine() = default;
Gemma4Moe26BReferenceEngine::~Gemma4Moe26BReferenceEngine() = default;
Gemma4Moe26BReferenceEngine::Gemma4Moe26BReferenceEngine(
    Gemma4Moe26BReferenceEngine&&) noexcept = default;
Gemma4Moe26BReferenceEngine& Gemma4Moe26BReferenceEngine::operator=(
    Gemma4Moe26BReferenceEngine&&) noexcept = default;
Gemma4Moe26BReferenceEngine::Gemma4Moe26BReferenceEngine(
    std::unique_ptr<Impl> implementation)
    : implementation_(std::move(implementation)) {}

Result<Gemma4Moe26BReferenceEngine> Gemma4Moe26BReferenceEngine::Create(
    const std::filesystem::path& model_directory,
    std::uint64_t context_tokens, int device, Gemma4Moe26BBackend backend) {
  if (context_tokens == 0U || context_tokens > 32768U || device < 0) {
    return Invalid("M13 reference context must be in [1, 32768]");
  }
  cudaError_t error = cudaSetDevice(device);
  if (error != cudaSuccess) return CudaFailure("select M13 CUDA device", error);

  auto config = LoadModelConfig(model_directory / "config.json");
  if (!config.ok()) return config.status();
  Status valid = ValidateGemma4Moe26BContract(config.value());
  if (!valid.ok()) return valid;
  auto manifest = InspectCheckpoint({model_directory, true});
  if (!manifest.ok()) return manifest.status();
  auto traits = BuildGemma4Moe26BAttentionTraits(config.value());
  if (!traits.ok()) return traits.status();
  valid = ValidateGemma4Moe26BAttentionBindings(manifest.value().tensors,
                                                traits.value());
  if (!valid.ok()) return valid;
  auto plan = BuildGemma4Moe26BResidencyPlan(manifest.value(), config.value());
  if (!plan.ok()) return plan.status();
  auto artifact = Gemma4Moe26BDeviceArtifact::Load(
      model_directory, manifest.value(), plan.value());
  if (!artifact.ok()) return artifact.status();

  auto impl = std::make_unique<Impl>();
  impl->device = device;
  impl->context = context_tokens;
  impl->backend = backend;
  impl->traits = traits.value();
  impl->artifact = std::move(artifact).value();
  error = cudaStreamCreateWithFlags(&impl->stream, cudaStreamNonBlocking);
  if (error != cudaSuccess) return CudaFailure("create M13 stream", error);

  for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
    auto attention = BindGemma4Moe26BAttentionReferenceWeights(
        impl->artifact, impl->traits[layer]);
    if (!attention.ok()) return attention.status();
    impl->attention_weights[layer] = attention.value();
    auto moe = BindGemma4Moe26BReferenceWeights(impl->artifact, layer);
    if (!moe.ok()) return moe.status();
    impl->moe_weights[layer] = moe.value();
  }

  auto head_packed = ArtifactPointer<std::uint8_t>(
      impl->artifact, "model.language_model.embed_tokens.weight_packed");
  auto head_scales = ArtifactPointer<std::uint8_t>(
      impl->artifact, "model.language_model.embed_tokens.weight_scale");
  auto final_norm = ArtifactPointer<std::uint16_t>(
      impl->artifact, "model.language_model.norm.weight");
  auto head_activation_divisor = impl->artifact.HostFloat32(
      "model.language_model.embed_tokens.input_global_scale");
  auto head_weight_divisor = impl->artifact.HostFloat32(
      "model.language_model.embed_tokens.weight_global_scale");
  if (!head_packed.ok()) return head_packed.status();
  if (!head_scales.ok()) return head_scales.status();
  if (!final_norm.ok()) return final_norm.status();
  if (!head_activation_divisor.ok()) return head_activation_divisor.status();
  if (!head_weight_divisor.ok()) return head_weight_divisor.status();
  impl->head = {head_packed.value(), head_scales.value(), kVocabulary, kWidth,
                head_activation_divisor.value(), head_weight_divisor.value()};
  impl->final_norm = final_norm.value();
  impl->softcap = static_cast<float>(config.value().final_logit_softcap);

  auto kv_bytes = Gemma4Moe26BFp8KvBytes(impl->traits, context_tokens);
  if (!kv_bytes.ok()) return kv_bytes.status();
  valid = impl->kv.Allocate(kv_bytes.value(), "allocate M13 FP8 K/V arena");
  if (!valid.ok()) return valid;
  std::uint64_t kv_offset = 0U;
  for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
    const auto& trait = impl->traits[layer];
    const std::uint64_t capacity =
        trait.attention == Gemma4Moe26BAttentionType::kSliding
            ? trait.cache_capacity
            : context_tokens;
    const std::uint64_t one = capacity * trait.kv_heads * trait.head_dimension;
    impl->caches[layer] = {impl->kv.As<std::uint8_t>(kv_offset),
                           impl->kv.As<std::uint8_t>(kv_offset + one),
                           capacity};
    kv_offset += 2U * one;
  }
  if (kv_offset != kv_bytes.value()) {
    return Status(StatusCode::kInternal,
                  "M13 K/V partition does not match the accepted byte formula");
  }

  LayoutBuilder layout;
  const auto hidden_a = layout.Add<float>(kWidth);
  const auto hidden_b = layout.Add<float>(kWidth);
  const auto final_hidden = layout.Add<float>(kWidth);
  const auto a_input_fp8 = layout.Add<std::uint8_t>(kWidth);
  const auto a_input_scale = layout.Add<float>(1U);
  const auto q_raw = layout.Add<float>(16U * 512U);
  const auto k_raw = layout.Add<float>(8U * 512U);
  const auto v_raw = layout.Add<float>(8U * 512U);
  const auto q_norm = layout.Add<float>(16U * 512U);
  const auto k_norm = layout.Add<float>(8U * 512U);
  const auto v_norm = layout.Add<float>(8U * 512U);
  const auto cosine = layout.Add<float>(256U);
  const auto sine = layout.Add<float>(256U);
  const auto staged_k = layout.Add<std::uint8_t>(8U * 512U);
  const auto staged_v = layout.Add<std::uint8_t>(8U * 512U);
  const auto scores = layout.Add<float>(16U * context_tokens);
  const auto attention = layout.Add<float>(16U * 512U);
  const auto output_fp8 = layout.Add<std::uint8_t>(16U * 512U);
  const auto output_scale = layout.Add<float>(1U);
  const auto output_projection = layout.Add<float>(kWidth);
  const auto post_attention = layout.Add<float>(kWidth);

  const auto shared_input = layout.Add<float>(kWidth);
  const auto shared_input_packed = layout.Add<std::uint8_t>(kWidth / 2U);
  const auto shared_input_scales = layout.Add<std::uint8_t>(kWidth / 16U);
  const auto shared_gate = layout.Add<float>(kShared);
  const auto shared_up = layout.Add<float>(kShared);
  const auto shared_product = layout.Add<float>(kShared);
  const auto shared_product_packed = layout.Add<std::uint8_t>(kShared / 2U);
  const auto shared_product_scales = layout.Add<std::uint8_t>(kShared / 16U);
  const auto shared_output = layout.Add<float>(kWidth);
  const auto shared_post = layout.Add<float>(kWidth);
  const auto router_normalized = layout.Add<float>(kWidth);
  const auto router_transformed = layout.Add<float>(kWidth);
  const auto router_logits = layout.Add<float>(kExperts);
  const auto router_probabilities = layout.Add<float>(kExperts);
  const auto top_ids = layout.Add<std::uint32_t>(kTopK);
  const auto top_weights = layout.Add<float>(kTopK);
  const auto expert_input = layout.Add<float>(kWidth);
  const auto expert_input_packed = layout.Add<std::uint8_t>(kWidth / 2U);
  const auto expert_input_scales = layout.Add<std::uint8_t>(kWidth / 16U);
  const auto expert_gate_up = layout.Add<float>(kTopK * 2U * kExpert);
  const auto expert_product = layout.Add<float>(kTopK * kExpert);
  const auto expert_product_packed =
      layout.Add<std::uint8_t>(kTopK * kExpert / 2U);
  const auto expert_product_scales =
      layout.Add<std::uint8_t>(kTopK * kExpert / 16U);
  const auto expert_down = layout.Add<float>(kTopK * kWidth);
  const auto expert_contributions = layout.Add<float>(kTopK * kWidth);
  const auto routed_sum = layout.Add<float>(kWidth);
  const auto routed_post = layout.Add<float>(kWidth);
  const auto combined = layout.Add<float>(kWidth);
  const auto feed_forward = layout.Add<float>(kWidth);
  const auto head_activation = layout.Add<std::uint8_t>(kWidth / 2U);
  const auto head_activation_scale_offset =
      layout.Add<std::uint8_t>(kWidth / 16U);
  const auto logits = layout.Add<float>(kVocabulary);
  const auto prediction_token = layout.Add<std::uint32_t>(1U);
  const auto prediction_logit = layout.Add<float>(1U);
  const auto finite = layout.Add<int>(1U);
  const auto decode_control = layout.Add<DecodeControl>(1U);
  std::array<std::uint64_t, kCaptureLayers.size()> capture_outputs{};
  std::array<std::uint64_t, kCaptureLayers.size()> capture_probs{};
  std::array<std::uint64_t, kCaptureLayers.size()> capture_ids{};
  for (std::size_t i = 0; i < kCaptureLayers.size(); ++i) {
    capture_outputs[i] = layout.Add<float>(kWidth);
    capture_probs[i] = layout.Add<float>(kExperts);
    capture_ids[i] = layout.Add<std::uint32_t>(kTopK);
  }
  if (layout.bytes == std::numeric_limits<std::uint64_t>::max()) {
    return Invalid("M13 workspace layout overflow");
  }
  valid = impl->workspace.Allocate(layout.bytes, "allocate M13 fixed workspace");
  if (!valid.ok()) return valid;
  auto ptr = [&](std::uint64_t offset) {
    return impl->workspace.As<std::byte>(offset);
  };
  impl->hidden_a = reinterpret_cast<float*>(ptr(hidden_a));
  impl->hidden_b = reinterpret_cast<float*>(ptr(hidden_b));
  impl->final_hidden = reinterpret_cast<float*>(ptr(final_hidden));
  impl->attention_workspace = {
      reinterpret_cast<std::uint8_t*>(ptr(a_input_fp8)),
      reinterpret_cast<float*>(ptr(a_input_scale)),
      reinterpret_cast<float*>(ptr(q_raw)), reinterpret_cast<float*>(ptr(k_raw)),
      reinterpret_cast<float*>(ptr(v_raw)), reinterpret_cast<float*>(ptr(q_norm)),
      reinterpret_cast<float*>(ptr(k_norm)), reinterpret_cast<float*>(ptr(v_norm)),
      reinterpret_cast<float*>(ptr(cosine)), reinterpret_cast<float*>(ptr(sine)),
      reinterpret_cast<std::uint8_t*>(ptr(staged_k)),
      reinterpret_cast<std::uint8_t*>(ptr(staged_v)),
      reinterpret_cast<float*>(ptr(scores)), 16U * context_tokens,
      reinterpret_cast<float*>(ptr(attention)),
      reinterpret_cast<std::uint8_t*>(ptr(output_fp8)),
      reinterpret_cast<float*>(ptr(output_scale)),
      reinterpret_cast<float*>(ptr(output_projection)),
      reinterpret_cast<float*>(ptr(post_attention))};
  impl->moe_workspace = {
      reinterpret_cast<float*>(ptr(shared_input)),
      reinterpret_cast<std::uint8_t*>(ptr(shared_input_packed)),
      reinterpret_cast<std::uint8_t*>(ptr(shared_input_scales)),
      reinterpret_cast<float*>(ptr(shared_gate)),
      reinterpret_cast<float*>(ptr(shared_up)),
      reinterpret_cast<float*>(ptr(shared_product)),
      reinterpret_cast<std::uint8_t*>(ptr(shared_product_packed)),
      reinterpret_cast<std::uint8_t*>(ptr(shared_product_scales)),
      reinterpret_cast<float*>(ptr(shared_output)),
      reinterpret_cast<float*>(ptr(shared_post)),
      reinterpret_cast<float*>(ptr(router_normalized)),
      reinterpret_cast<float*>(ptr(router_transformed)),
      reinterpret_cast<float*>(ptr(router_logits)),
      reinterpret_cast<float*>(ptr(router_probabilities)),
      reinterpret_cast<std::uint32_t*>(ptr(top_ids)),
      reinterpret_cast<float*>(ptr(top_weights)),
      reinterpret_cast<float*>(ptr(expert_input)),
      reinterpret_cast<std::uint8_t*>(ptr(expert_input_packed)),
      reinterpret_cast<std::uint8_t*>(ptr(expert_input_scales)),
      reinterpret_cast<float*>(ptr(expert_gate_up)),
      reinterpret_cast<float*>(ptr(expert_product)),
      reinterpret_cast<std::uint8_t*>(ptr(expert_product_packed)),
      reinterpret_cast<std::uint8_t*>(ptr(expert_product_scales)),
      reinterpret_cast<float*>(ptr(expert_down)),
      reinterpret_cast<float*>(ptr(expert_contributions)),
      reinterpret_cast<float*>(ptr(routed_sum)),
      reinterpret_cast<float*>(ptr(routed_post)),
      reinterpret_cast<float*>(ptr(combined)),
      reinterpret_cast<float*>(ptr(feed_forward))};
  impl->head_activation = reinterpret_cast<std::uint8_t*>(ptr(head_activation));
  impl->head_activation_scales =
      reinterpret_cast<std::uint8_t*>(ptr(head_activation_scale_offset));
  impl->logits = reinterpret_cast<float*>(ptr(logits));
  impl->prediction_token = reinterpret_cast<std::uint32_t*>(ptr(prediction_token));
  impl->prediction_logit = reinterpret_cast<float*>(ptr(prediction_logit));
  impl->finite = reinterpret_cast<int*>(ptr(finite));
  impl->decode_control = reinterpret_cast<DecodeControl*>(ptr(decode_control));
  for (std::size_t i = 0; i < kCaptureLayers.size(); ++i) {
    impl->layer_captures[i] = reinterpret_cast<float*>(ptr(capture_outputs[i]));
    impl->router_probability_captures[i] =
        reinterpret_cast<float*>(ptr(capture_probs[i]));
    impl->router_id_captures[i] =
        reinterpret_cast<std::uint32_t*>(ptr(capture_ids[i]));
  }

  if (backend == Gemma4Moe26BBackend::kSm120Integrated) {
    LayoutBuilder prefill;
    const auto p_tokens = prefill.Add<std::uint32_t>(kPrefillMaxTokens);
    const auto p_hidden_a = prefill.Add<float>(kPrefillMaxTokens * kWidth);
    const auto p_hidden_b = prefill.Add<float>(kPrefillMaxTokens * kWidth);
    const auto p_input_fp8 =
        prefill.Add<std::uint8_t>(kPrefillMaxTokens * kWidth);
    const auto p_input_scale = prefill.Add<float>(kPrefillMaxTokens);
    const auto p_q_raw =
        prefill.Add<float>(kPrefillMaxTokens * 16U * 512U);
    const auto p_k_raw =
        prefill.Add<float>(kPrefillMaxTokens * 8U * 512U);
    const auto p_v_raw =
        prefill.Add<float>(kPrefillMaxTokens * 8U * 512U);
    const auto p_q_norm =
        prefill.Add<float>(kPrefillMaxTokens * 16U * 512U);
    const auto p_k_norm =
        prefill.Add<float>(kPrefillMaxTokens * 8U * 512U);
    const auto p_v_norm =
        prefill.Add<float>(kPrefillMaxTokens * 8U * 512U);
    const auto p_cosine = prefill.Add<float>(kPrefillMaxTokens * 256U);
    const auto p_sine = prefill.Add<float>(kPrefillMaxTokens * 256U);
    const auto p_staged_k =
        prefill.Add<std::uint8_t>(kPrefillMaxTokens * 8U * 512U);
    const auto p_staged_v =
        prefill.Add<std::uint8_t>(kPrefillMaxTokens * 8U * 512U);
    const auto p_scores = prefill.Add<float>(kPrefillScoreElements);
    const auto p_attention =
        prefill.Add<float>(kPrefillMaxTokens * 16U * 512U);
    const auto p_output_fp8 =
        prefill.Add<std::uint8_t>(kPrefillMaxTokens * 16U * 512U);
    const auto p_output_scale = prefill.Add<float>(kPrefillMaxTokens);
    const auto p_output_projection =
        prefill.Add<float>(kPrefillMaxTokens * kWidth);
    const auto p_post_attention =
        prefill.Add<float>(kPrefillMaxTokens * kWidth);

    const auto p_router_logits =
        prefill.Add<float>(kPrefillMaxTokens * kExperts);
    const auto p_router_probabilities =
        prefill.Add<float>(kPrefillMaxTokens * kExperts);
    const auto p_token_hidden =
        prefill.Add<float>(kPrefillMaxTokens * kWidth);
    const auto p_token_packed =
        prefill.Add<std::uint8_t>(kPrefillMaxTokens * kWidth / 2U);
    const auto p_token_scales =
        prefill.Add<std::uint8_t>(kPrefillMaxTokens * kWidth / 16U);
    const auto p_expert_product = prefill.Add<float>(
        kPrefillMaxTokens * kTopK * kExpert);
    const auto p_expert_product_packed = prefill.Add<std::uint8_t>(
        kPrefillMaxTokens * kTopK * kExpert / 2U);
    const auto p_expert_product_scales = prefill.Add<std::uint8_t>(
        kPrefillMaxTokens * kTopK * kExpert / 16U);
    const auto p_expert_down = prefill.Add<float>(
        kPrefillMaxTokens * kTopK * kWidth);
    const auto p_shared_product =
        prefill.Add<float>(kPrefillMaxTokens * kShared);
    const auto p_shared_product_packed =
        prefill.Add<std::uint8_t>(kPrefillMaxTokens * kShared / 2U);
    const auto p_shared_product_scales =
        prefill.Add<std::uint8_t>(kPrefillMaxTokens * kShared / 16U);
    const auto p_shared_output =
        prefill.Add<float>(kPrefillMaxTokens * kWidth);
    const auto p_reduced_output =
        prefill.Add<float>(kPrefillMaxTokens * kWidth);
    const auto p_assignments = prefill.Add<Gemma4MoePrefillAssignment>(
        kPrefillMaxTokens * kTopK);
    const auto p_histogram = prefill.Add<std::uint32_t>(kExperts);
    const auto p_prefix = prefill.Add<std::uint32_t>(kExperts + 1U);
    const auto p_permutation =
        prefill.Add<std::uint32_t>(kPrefillMaxTokens * kTopK);
    const auto p_inverse =
        prefill.Add<std::uint32_t>(kPrefillMaxTokens * kTopK);
    constexpr std::uint64_t kM09MoePrefillCap = 192U * 1024U * 1024U;
    if (prefill.bytes == std::numeric_limits<std::uint64_t>::max() ||
        prefill.bytes > kM09MoePrefillCap) {
      return Invalid("M17 fixed prefill workspace exceeds the M09 cap");
    }
    valid = impl->prefill_workspace.Allocate(
        prefill.bytes, "allocate M17 fixed prefill workspace");
    if (!valid.ok()) return valid;
    auto pptr = [&](std::uint64_t offset) {
      return impl->prefill_workspace.As<std::byte>(offset);
    };
    impl->prefill_tokens = reinterpret_cast<std::uint32_t*>(pptr(p_tokens));
    error = cudaMallocHost(&impl->prefill_host_tokens,
                           kPrefillMaxTokens * sizeof(std::uint32_t));
    if (error != cudaSuccess) {
      return CudaFailure("allocate M17 pinned prefill tokens", error);
    }
    impl->prefill_hidden_a = reinterpret_cast<float*>(pptr(p_hidden_a));
    impl->prefill_hidden_b = reinterpret_cast<float*>(pptr(p_hidden_b));
    impl->prefill_attention_workspace = {
        reinterpret_cast<std::uint8_t*>(pptr(p_input_fp8)),
        reinterpret_cast<float*>(pptr(p_input_scale)),
        reinterpret_cast<float*>(pptr(p_q_raw)),
        reinterpret_cast<float*>(pptr(p_k_raw)),
        reinterpret_cast<float*>(pptr(p_v_raw)),
        reinterpret_cast<float*>(pptr(p_q_norm)),
        reinterpret_cast<float*>(pptr(p_k_norm)),
        reinterpret_cast<float*>(pptr(p_v_norm)),
        reinterpret_cast<float*>(pptr(p_cosine)),
        reinterpret_cast<float*>(pptr(p_sine)),
        reinterpret_cast<std::uint8_t*>(pptr(p_staged_k)),
        reinterpret_cast<std::uint8_t*>(pptr(p_staged_v)),
        reinterpret_cast<float*>(pptr(p_scores)), kPrefillScoreElements,
        reinterpret_cast<float*>(pptr(p_attention)),
        reinterpret_cast<std::uint8_t*>(pptr(p_output_fp8)),
        reinterpret_cast<float*>(pptr(p_output_scale)),
        reinterpret_cast<float*>(pptr(p_output_projection)),
        reinterpret_cast<float*>(pptr(p_post_attention))};
    impl->prefill_moe_workspace = {
        reinterpret_cast<float*>(pptr(p_router_logits)),
        reinterpret_cast<float*>(pptr(p_router_probabilities)),
        reinterpret_cast<float*>(pptr(p_token_hidden)),
        reinterpret_cast<std::uint8_t*>(pptr(p_token_packed)),
        reinterpret_cast<std::uint8_t*>(pptr(p_token_scales)),
        reinterpret_cast<float*>(pptr(p_expert_product)),
        reinterpret_cast<std::uint8_t*>(pptr(p_expert_product_packed)),
        reinterpret_cast<std::uint8_t*>(pptr(p_expert_product_scales)),
        reinterpret_cast<float*>(pptr(p_expert_down)),
        reinterpret_cast<float*>(pptr(p_shared_product)),
        reinterpret_cast<std::uint8_t*>(pptr(p_shared_product_packed)),
        reinterpret_cast<std::uint8_t*>(pptr(p_shared_product_scales)),
        reinterpret_cast<float*>(pptr(p_shared_output)),
        reinterpret_cast<float*>(pptr(p_reduced_output)),
        reinterpret_cast<Gemma4MoePrefillAssignment*>(pptr(p_assignments)),
        reinterpret_cast<std::uint32_t*>(pptr(p_histogram)),
        reinterpret_cast<std::uint32_t*>(pptr(p_prefix)),
        reinterpret_cast<std::uint32_t*>(pptr(p_permutation)),
        reinterpret_cast<std::uint32_t*>(pptr(p_inverse))};
  }

  Gemma4Moe26BReferenceEngine engine(std::move(impl));
  valid = engine.Reset();
  if (!valid.ok()) return valid;
  if (backend == Gemma4Moe26BBackend::kSm120Integrated) {
    valid = engine.implementation_->PrepareDecodeGraph();
    if (!valid.ok()) return valid;
  }
  return engine;
}

Status Gemma4Moe26BReferenceEngine::Reset() {
  if (!implementation_) return Invalid("M13 engine is not initialized");
  cudaError_t error = cudaMemsetAsync(implementation_->kv.As<std::byte>(), 0,
                                      implementation_->kv.bytes(),
                                      implementation_->stream);
  if (error != cudaSuccess) return CudaFailure("clear M13 K/V arena", error);
  error = cudaMemsetAsync(implementation_->workspace.As<std::byte>(), 0,
                          implementation_->workspace.bytes(),
                          implementation_->stream);
  if (error != cudaSuccess) return CudaFailure("clear M13 workspace", error);
  if (implementation_->prefill_workspace.bytes() != 0U) {
    error = cudaMemsetAsync(
        implementation_->prefill_workspace.As<std::byte>(), 0,
        implementation_->prefill_workspace.bytes(), implementation_->stream);
    if (error != cudaSuccess) {
      return CudaFailure("clear M17 prefill workspace", error);
    }
  }
  error = cudaStreamSynchronize(implementation_->stream);
  if (error != cudaSuccess) return CudaFailure("synchronize M13 reset", error);
  implementation_->position = 0U;
  return Status::Ok();
}

Status Gemma4Moe26BReferenceEngine::ForwardToken(std::uint32_t token) {
  if (!implementation_) return Invalid("M13 engine is not initialized");
  auto& x = *implementation_;
  if (token >= kVocabulary || x.position >= x.context) {
    return Invalid("M13 token or position exceeds the fixed contract");
  }
  if (x.backend == Gemma4Moe26BBackend::kSm120Integrated) {
    if (x.decode_graph == nullptr) {
      return Invalid("M17 decode graph is not initialized");
    }
    SetDecodeControlKernel<<<1, 1, 0, x.stream>>>(x.decode_control, token,
                                                  x.position);
    cudaError_t graph_error = cudaGetLastError();
    if (graph_error == cudaSuccess) {
      graph_error = cudaGraphLaunch(x.decode_graph, x.stream);
    }
    if (graph_error != cudaSuccess) {
      return CudaFailure("launch M17 decode graph", graph_error);
    }
    ++x.position;
    return Status::Ok();
  }
  constexpr unsigned threads = 256U;
  TiledEmbeddingLookupKernel<<<static_cast<unsigned>((kWidth + threads - 1U) /
                                                       threads),
                               threads, 0, x.stream>>>(
      x.head.packed_e2m1, x.head.scales_e4m3fn,
      x.head.weight_global_divisor, token, x.hidden_a);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) return CudaFailure("launch M13 embedding", error);

  for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
    Status status = LaunchGemma4Moe26BAttentionReferenceLayer(
        x.hidden_a, x.hidden_b, x.position, x.traits[layer],
        x.attention_weights[layer], x.caches[layer], x.attention_workspace,
        1.0e-6F, x.stream);
    if (!status.ok()) return status;
    status = x.backend != Gemma4Moe26BBackend::kReference
                 ? LaunchGemma4MoeSm120Layer(
                       x.hidden_b, x.hidden_a, x.moe_config,
                       x.moe_weights[layer], x.moe_workspace, x.stream)
                 : LaunchGemma4MoeReferenceLayer(
                       x.hidden_b, x.hidden_a, x.moe_config,
                       x.moe_weights[layer], x.moe_workspace, x.stream);
    if (!status.ok()) return status;
    const int capture = CaptureIndex(layer);
    if (capture >= 0) {
      error = cudaMemcpyAsync(x.layer_captures[static_cast<std::size_t>(capture)],
                              x.hidden_a, kWidth * sizeof(float),
                              cudaMemcpyDeviceToDevice, x.stream);
      if (error == cudaSuccess) {
        error = cudaMemcpyAsync(
            x.router_probability_captures[static_cast<std::size_t>(capture)],
            x.moe_workspace.router_probabilities, kExperts * sizeof(float),
            cudaMemcpyDeviceToDevice, x.stream);
      }
      if (error == cudaSuccess) {
        error = cudaMemcpyAsync(
            x.router_id_captures[static_cast<std::size_t>(capture)],
            x.moe_workspace.top_ids, kTopK * sizeof(std::uint32_t),
            cudaMemcpyDeviceToDevice, x.stream);
      }
      if (error != cudaSuccess) return CudaFailure("capture M13 layer", error);
    }
  }
  Status status = LaunchRmsNormBf16(x.hidden_a, x.final_norm, x.final_hidden,
                                    1U, kWidth, 1.0e-6F, x.stream);
  if (!status.ok()) return status;
  status = LaunchNvfp4ReferenceActivationQuantization(
      x.final_hidden, x.head_activation, x.head_activation_scales, kWidth,
      x.head.activation_global_divisor, x.stream);
  if (!status.ok()) return status;
  status = x.backend != Gemma4Moe26BBackend::kReference
               ? LaunchNvfp4Sm120DirectProjectionBf16Float(
                     x.head_activation, x.head_activation_scales,
                     x.head.packed_e2m1, x.head.scales_e4m3fn, x.logits,
                     x.head.rows, x.head.columns,
                     x.head.activation_global_divisor,
                     x.head.weight_global_divisor, x.stream)
               : LaunchGemma4MoeTiledNvfp4ReferenceProjection(
                     x.head, x.head_activation, x.head_activation_scales,
                     x.logits, x.stream);
  if (!status.ok()) return status;
  error = cudaMemsetAsync(x.finite, 1, sizeof(int), x.stream);
  if (error != cudaSuccess) return CudaFailure("initialize M13 finite flag", error);
  SoftcapFiniteKernel<<<static_cast<unsigned>((kVocabulary + threads - 1U) /
                                               threads),
                        threads, 0, x.stream>>>(x.logits, x.softcap, x.finite);
  error = cudaGetLastError();
  if (error != cudaSuccess) return CudaFailure("launch M13 softcap", error);
  DeterministicArgmaxKernel<<<1, 1, 0, x.stream>>>(
      x.logits, x.prediction_token, x.prediction_logit);
  error = cudaGetLastError();
  if (error != cudaSuccess) return CudaFailure("launch M13 argmax", error);
  ++x.position;
  return Status::Ok();
}

Status Gemma4Moe26BReferenceEngine::PrefillTokens(
    std::span<const std::uint32_t> tokens) {
  if (!implementation_ || tokens.empty() ||
      implementation_->backend != Gemma4Moe26BBackend::kSm120Integrated ||
      implementation_->prefill_workspace.bytes() == 0U ||
      tokens.size() > implementation_->context - implementation_->position) {
    return Invalid("M17 prefill request exceeds the initialized contract");
  }
  for (const std::uint32_t token : tokens) {
    if (token >= kVocabulary) return Invalid("M17 prefill token is invalid");
  }
  auto& x = *implementation_;
  constexpr unsigned threads = 256U;
  std::size_t consumed = 0U;
  while (consumed < tokens.size()) {
    std::uint64_t chunk = std::min<std::uint64_t>(
        kPrefillMaxTokens, tokens.size() - consumed);
    while (chunk > 0U &&
           chunk * 16U * (x.position + chunk) > kPrefillScoreElements) {
      --chunk;
    }
    if (chunk == 0U) {
      return Invalid("M17 prefill score workspace cannot fit one token");
    }
    cudaError_t error = cudaStreamSynchronize(x.stream);
    if (error != cudaSuccess) {
      return CudaFailure("synchronize M17 prefill token staging", error);
    }
    std::copy_n(tokens.data() + consumed, static_cast<std::size_t>(chunk),
                x.prefill_host_tokens);
    error = cudaMemcpyAsync(
        x.prefill_tokens, x.prefill_host_tokens,
        chunk * sizeof(std::uint32_t), cudaMemcpyHostToDevice, x.stream);
    if (error != cudaSuccess) return CudaFailure("copy M17 prefill tokens", error);
    const std::uint64_t hidden_elements = chunk * kWidth;
    TiledEmbeddingLookupBatchKernel<<<
        static_cast<unsigned>((hidden_elements + threads - 1U) / threads),
        threads, 0, x.stream>>>(
        x.head.packed_e2m1, x.head.scales_e4m3fn,
        x.head.weight_global_divisor, x.prefill_tokens, x.prefill_hidden_a,
        hidden_elements);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
      return CudaFailure("launch M17 prefill embedding", error);
    }
    for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
      Status status = LaunchGemma4Moe26BAttentionReferencePrefillLayer(
          x.prefill_hidden_a, x.prefill_hidden_b, x.position, chunk,
          x.traits[layer], x.attention_weights[layer], x.caches[layer],
          x.prefill_attention_workspace, 1.0e-6F, x.stream);
      if (!status.ok()) return status;
      status = LaunchGemma4MoeSm120PrefillLayer(
          x.prefill_hidden_b, x.prefill_hidden_a, chunk, x.moe_config,
          x.moe_weights[layer], x.prefill_moe_workspace, x.stream);
      if (!status.ok()) return status;
      const int capture = CaptureIndex(layer);
      if (capture >= 0) {
        const std::size_t capture_index = static_cast<std::size_t>(capture);
        error = cudaMemcpyAsync(
            x.layer_captures[capture_index],
            x.prefill_hidden_a + (chunk - 1U) * kWidth,
            kWidth * sizeof(float), cudaMemcpyDeviceToDevice, x.stream);
        if (error == cudaSuccess) {
          error = cudaMemcpyAsync(
              x.router_probability_captures[capture_index],
              x.prefill_moe_workspace.router_probabilities +
                  (chunk - 1U) * kExperts,
              kExperts * sizeof(float), cudaMemcpyDeviceToDevice, x.stream);
        }
        if (error == cudaSuccess) {
          CapturePrefillRouterIdsKernel<<<1, kTopK, 0, x.stream>>>(
              x.prefill_moe_workspace.assignments,
              x.router_id_captures[capture_index], chunk - 1U);
          error = cudaGetLastError();
        }
        if (error != cudaSuccess) {
          return CudaFailure("capture M17 prefill layer", error);
        }
      }
    }
    float* last_hidden = x.prefill_hidden_a + (chunk - 1U) * kWidth;
    Status status = LaunchRmsNormBf16(last_hidden, x.final_norm, x.final_hidden,
                                      1U, kWidth, 1.0e-6F, x.stream);
    if (!status.ok()) return status;
    status = LaunchNvfp4ReferenceActivationQuantization(
        x.final_hidden, x.head_activation, x.head_activation_scales, kWidth,
        x.head.activation_global_divisor, x.stream);
    if (!status.ok()) return status;
    status = LaunchNvfp4Sm120DirectProjectionBf16Float(
        x.head_activation, x.head_activation_scales, x.head.packed_e2m1,
        x.head.scales_e4m3fn, x.logits, x.head.rows, x.head.columns,
        x.head.activation_global_divisor, x.head.weight_global_divisor,
        x.stream);
    if (!status.ok()) return status;
    error = cudaMemsetAsync(x.finite, 1, sizeof(int), x.stream);
    if (error != cudaSuccess) {
      return CudaFailure("initialize M17 prefill finite flag", error);
    }
    SoftcapFiniteKernel<<<
        static_cast<unsigned>((kVocabulary + threads - 1U) / threads),
        threads, 0, x.stream>>>(x.logits, x.softcap, x.finite);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
      return CudaFailure("launch M17 prefill softcap", error);
    }
    DeterministicArgmaxKernel<<<1, 1, 0, x.stream>>>(
        x.logits, x.prediction_token, x.prediction_logit);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
      return CudaFailure("launch M17 prefill argmax", error);
    }
    x.position += chunk;
    consumed += static_cast<std::size_t>(chunk);
  }
  return Status::Ok();
}

Result<Gemma4Moe26BReferencePrediction>
Gemma4Moe26BReferenceEngine::Prediction() {
  if (!implementation_) return Invalid("M13 engine is not initialized");
  auto& x = *implementation_;
  cudaError_t error = cudaStreamSynchronize(x.stream);
  if (error != cudaSuccess) return CudaFailure("synchronize M13 prediction", error);
  Gemma4Moe26BReferencePrediction result;
  int finite = 0;
  error = cudaMemcpy(&result.token, x.prediction_token, sizeof(result.token),
                     cudaMemcpyDeviceToHost);
  if (error == cudaSuccess) {
    error = cudaMemcpy(&result.logit, x.prediction_logit, sizeof(result.logit),
                       cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(&finite, x.finite, sizeof(finite), cudaMemcpyDeviceToHost);
  }
  if (error != cudaSuccess) return CudaFailure("copy M13 prediction", error);
  result.all_logits_finite = finite != 0;
  return result;
}

Status Gemma4Moe26BReferenceEngine::CopyLogits(std::span<float> output) {
  if (!implementation_ || output.size() != kVocabulary) {
    return Invalid("M13 logit destination has the wrong extent");
  }
  cudaError_t error = cudaStreamSynchronize(implementation_->stream);
  if (error == cudaSuccess) {
    error = cudaMemcpy(output.data(), implementation_->logits,
                       output.size_bytes(), cudaMemcpyDeviceToHost);
  }
  return error == cudaSuccess ? Status::Ok()
                              : CudaFailure("copy M13 logits", error);
}

Status Gemma4Moe26BReferenceEngine::CopyLayerOutput(
    std::uint32_t layer, std::span<float> output) {
  const int index = CaptureIndex(layer);
  if (!implementation_ || index < 0 || output.size() != kWidth) {
    return Invalid("M13 layer capture request is invalid");
  }
  cudaError_t error = cudaStreamSynchronize(implementation_->stream);
  if (error == cudaSuccess) {
    error = cudaMemcpy(output.data(),
                       implementation_->layer_captures[static_cast<std::size_t>(index)],
                       output.size_bytes(), cudaMemcpyDeviceToHost);
  }
  return error == cudaSuccess ? Status::Ok()
                              : CudaFailure("copy M13 layer output", error);
}

Status Gemma4Moe26BReferenceEngine::CopyRouterProbabilities(
    std::uint32_t layer, std::span<float> output) {
  const int index = CaptureIndex(layer);
  if (!implementation_ || index < 0 || output.size() != kExperts) {
    return Invalid("M13 router probability capture request is invalid");
  }
  cudaError_t error = cudaStreamSynchronize(implementation_->stream);
  if (error == cudaSuccess) {
    error = cudaMemcpy(
        output.data(),
        implementation_->router_probability_captures[static_cast<std::size_t>(index)],
        output.size_bytes(), cudaMemcpyDeviceToHost);
  }
  return error == cudaSuccess ? Status::Ok()
                              : CudaFailure("copy M13 router probabilities", error);
}

Status Gemma4Moe26BReferenceEngine::CopyRouterTopIds(
    std::uint32_t layer, std::span<std::uint32_t> output) {
  const int index = CaptureIndex(layer);
  if (!implementation_ || index < 0 || output.size() != kTopK) {
    return Invalid("M13 router ID capture request is invalid");
  }
  cudaError_t error = cudaStreamSynchronize(implementation_->stream);
  if (error == cudaSuccess) {
    error = cudaMemcpy(
        output.data(),
        implementation_->router_id_captures[static_cast<std::size_t>(index)],
        output.size_bytes(), cudaMemcpyDeviceToHost);
  }
  return error == cudaSuccess ? Status::Ok()
                              : CudaFailure("copy M13 router IDs", error);
}

std::uint64_t Gemma4Moe26BReferenceEngine::position() const {
  return implementation_ ? implementation_->position : 0U;
}
std::uint64_t Gemma4Moe26BReferenceEngine::context_capacity() const {
  return implementation_ ? implementation_->context : 0U;
}
std::uint64_t Gemma4Moe26BReferenceEngine::weight_arena_bytes() const {
  return implementation_ ? implementation_->artifact.arena_bytes() : 0U;
}
std::uint64_t Gemma4Moe26BReferenceEngine::kv_cache_bytes() const {
  return implementation_ ? implementation_->kv.bytes() : 0U;
}
std::uint64_t Gemma4Moe26BReferenceEngine::workspace_bytes() const {
  return implementation_ ? implementation_->workspace.bytes() +
                               implementation_->prefill_workspace.bytes()
                         : 0U;
}

}  // namespace gem16::internal
