constexpr std::uint64_t kWidth = 2816U;
constexpr std::uint64_t kVocabulary = 262144U;
constexpr std::uint64_t kShared = 2112U;
constexpr std::uint64_t kExpert = 704U;
constexpr std::uint32_t kExperts = 128U;
constexpr std::uint32_t kTopK = 8U;
constexpr std::uint64_t kLayers = 30U;
constexpr std::uint64_t kMaximumContextTokens = 262144U;
constexpr std::uint64_t kDefaultPrefillChunkTokens = 1024U;
constexpr std::uint64_t kMaximumTrellis35PrefillChunkTokens = 2048U;
constexpr std::uint64_t kPreparedGlobalPrefillTokens = 16384U;
// The native online SM120 attention path never materializes a score slab.
// Keep one validated sentinel element because the shared workspace contract
// still requires a non-null pointer, and spend the reclaimed 64 MiB on larger
// chunks that reuse resident expert weights across more tokens.
constexpr std::uint64_t kPrefillScoreElements = 1U;
constexpr std::uint64_t kNvfp4Block = 16U;
constexpr std::uint64_t kSm120KBlock = 64U;
constexpr std::uint64_t kRowsPerTile = 8U;
constexpr unsigned kArgmaxThreads = 256U;
constexpr unsigned kArgmaxBlocks =
    static_cast<unsigned>(kVocabulary / kArgmaxThreads);
constexpr std::array<std::uint32_t, 4> kCaptureLayers{0U, 5U, 6U, 29U};
constexpr std::uint64_t kMiB = 1024U * 1024U;
constexpr std::uint64_t kPrimaryContextMargin = 700U * kMiB;
constexpr std::uint64_t kLongContextMargin = 400U * kMiB;
constexpr std::uint32_t kMaximumSuppressedTokens = 16U;
constexpr std::uint32_t kRepetitionMaskWords =
    static_cast<std::uint32_t>((kVocabulary + 31U) / 32U);
constexpr std::uint64_t kM25MaximumVerifyTokens = 5U;
constexpr std::uint32_t kM25FixedD2Rows = 3U;

class NvtxRange {
 public:
  explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
  NvtxRange(const NvtxRange&) = delete;
  NvtxRange& operator=(const NvtxRange&) = delete;
  ~NvtxRange() { nvtxRangePop(); }
};

Status Invalid(std::string message) {
  return Status(StatusCode::kInvalidArgument, std::move(message));
}

Status CudaFailure(const char* operation, cudaError_t error) {
  return Status(StatusCode::kInternal,
                std::string(operation) + ": " + cudaGetErrorName(error) +
                    ": " + cudaGetErrorString(error));
}

Status CudaAllocationFailure(const char* operation, cudaError_t error) {
  if (error == cudaErrorMemoryAllocation) {
    return Status(StatusCode::kResourceExhausted,
                  std::string(operation) + ": " + cudaGetErrorName(error) +
                      ": " + cudaGetErrorString(error));
  }
  return CudaFailure(operation, error);
}

Result<std::uint64_t> ResolvePrefillChunkTokens(bool trellis35) {
  if (!trellis35) return kDefaultPrefillChunkTokens;
  const char* value =
      GetEnvironmentVariable("GEM16_TRELLIS35_PREFILL_CHUNK_TOKENS");
  if (value != nullptr && std::string_view(value) == "1024") {
    return kDefaultPrefillChunkTokens;
  }
  if (value == nullptr || std::string_view(value) == "2048") {
    return kMaximumTrellis35PrefillChunkTokens;
  }
  return Invalid(
      "GEM16_TRELLIS35_PREFILL_CHUNK_TOKENS must be 1024 or 2048");
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
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this == &other) return *this;
    if (pointer_ != nullptr) (void)cudaFree(pointer_);
    pointer_ = std::exchange(other.pointer_, nullptr);
    bytes_ = std::exchange(other.bytes_, 0U);
    return *this;
  }
  Status Allocate(std::uint64_t bytes, const char* label) {
    if (bytes == 0U || bytes > std::numeric_limits<std::size_t>::max()) {
      return Invalid(std::string(label) + " has an invalid size");
    }
    const cudaError_t error =
        cudaMalloc(&pointer_, static_cast<std::size_t>(bytes));
    if (error != cudaSuccess) return CudaAllocationFailure(label, error);
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
    if (bytes > std::numeric_limits<std::uint64_t>::max() - (alignment - 1U)) {
      bytes = std::numeric_limits<std::uint64_t>::max();
      return bytes;
    }
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

__global__ void BuildMtpDecodeControlsKernel(
    DecodeControl* controls, std::uint64_t start_position,
    std::uint64_t sampling_step, std::uint64_t tokens) {
  const std::uint64_t row = threadIdx.x;
  if (blockIdx.x == 0U && row < tokens) {
    controls[row] = DecodeControl{0U, 0U, start_position + row,
                                  sampling_step + row};
  }
}

struct LogitCandidate {
  float value;
  std::uint32_t id;
};

struct PredictionStatus {
  std::uint32_t token;
  float logit;
  int finite;
  int routing_finite;
};
static_assert(sizeof(PredictionStatus) == 16U);

__global__ void CommitSampledSelectionKernel(
    const std::uint32_t* selected, PredictionStatus* status,
    DecodeControl* next_control, std::uint64_t position) {
  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    const std::uint32_t token = *selected;
    status->token = token;
    *next_control = DecodeControl{token, 0U, position, position};
  }
}

struct MtpVerifierHostResult {
  MtpGroupTransaction transaction{};
  std::array<int, kM25MaximumVerifyTokens> logits_finite{};
  std::array<float, kM25MaximumVerifyTokens> top_values{};
  std::array<float, kM25MaximumVerifyTokens> second_values{};
  int routing_finite = 0;
};

struct MtpVerifierLayerOffsets {
  std::uint64_t backup_key = 0U;
  std::uint64_t backup_value = 0U;
  std::uint64_t compact_key = 0U;
  std::uint64_t compact_value = 0U;
};

__device__ __forceinline__ LogitCandidate BetterCandidate(
    LogitCandidate left, LogitCandidate right) {
  return right.value > left.value ||
                 (right.value == left.value && right.id < left.id)
             ? right
             : left;
}

__device__ bool MtpTokenSuppressed(std::uint32_t token,
                                   const std::uint32_t* suppressed,
                                   std::uint32_t suppressed_count) {
  for (std::uint32_t index = 0U; index < suppressed_count; ++index) {
    if (suppressed[index] == token) return true;
  }
  return false;
}

__global__ void SoftcapArgmaxBlocksKernel(float* logits, float softcap,
                                          int* all_finite,
                                          LogitCandidate* candidates,
                                          const std::uint32_t* suppressed,
                                          std::uint32_t suppressed_count,
                                          const std::uint32_t* excluded,
                                          bool apply_softcap) {
  __shared__ LogitCandidate partial[kArgmaxThreads];
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  LogitCandidate candidate{-3.402823466e+38F,
                           static_cast<std::uint32_t>(index)};
  if (index < kVocabulary) {
    const float value = apply_softcap
                            ? tanhf(logits[index] / softcap) * softcap
                            : logits[index];
    if (apply_softcap) logits[index] = value;
    if (!isfinite(value)) {
      atomicExch(all_finite, 0);
    } else if ((excluded == nullptr || *excluded != index) &&
               !MtpTokenSuppressed(static_cast<std::uint32_t>(index),
                                   suppressed, suppressed_count)) {
      candidate.value = value;
    }
  }
  partial[threadIdx.x] = candidate;
  __syncthreads();
  for (unsigned stride = kArgmaxThreads / 2U; stride != 0U; stride /= 2U) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] = BetterCandidate(
          partial[threadIdx.x], partial[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0U) candidates[blockIdx.x] = partial[0];
}

__global__ void FinalArgmaxKernel(const LogitCandidate* candidates,
                                  std::uint32_t* token, float* value,
                                  DecodeControl* next_control) {
  __shared__ LogitCandidate partial[kArgmaxThreads];
  LogitCandidate best{-3.402823466e+38F, 0U};
  for (unsigned index = threadIdx.x; index < kArgmaxBlocks;
       index += blockDim.x) {
    best = BetterCandidate(best, candidates[index]);
  }
  partial[threadIdx.x] = best;
  __syncthreads();
  for (unsigned stride = kArgmaxThreads / 2U; stride != 0U; stride /= 2U) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] = BetterCandidate(
          partial[threadIdx.x], partial[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0U) {
    *token = partial[0].id;
    *value = partial[0].value;
    if (next_control != nullptr) {
      // Prepare the ordinary greedy successor inside the captured graph. The
      // host validates token and position before it omits an explicit control
      // update, so teacher forcing and sampled overrides remain exact.
      next_control->token = partial[0].id;
      ++next_control->position;
    }
  }
}

Status LaunchSoftcapArgmax(float* logits, float softcap, int* all_finite,
                           LogitCandidate* candidates,
                           std::uint32_t* token, float* value,
                           cudaStream_t stream,
                           DecodeControl* next_control = nullptr,
                           const std::uint32_t* suppressed = nullptr,
                           std::uint32_t suppressed_count = 0U,
                           const std::uint32_t* excluded = nullptr,
                           bool apply_softcap = true) {
  cudaError_t error = cudaMemsetAsync(all_finite, 1, sizeof(int), stream);
  if (error != cudaSuccess) {
    return CudaFailure("initialize M17 finite flag", error);
  }
  SoftcapArgmaxBlocksKernel<<<kArgmaxBlocks, kArgmaxThreads, 0, stream>>>(
      logits, softcap, all_finite, candidates, suppressed, suppressed_count,
      excluded, apply_softcap);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaFailure("launch M17 fused softcap/argmax blocks", error);
  }
  FinalArgmaxKernel<<<1, kArgmaxThreads, 0, stream>>>(
      candidates, token, value, next_control);
  error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch M17 final argmax", error);
}

template <bool kCommitLogits>
__global__ void CommitMtpPredictionKernel(
    const float* batch_logits, float* committed_logits,
    const MtpGroupResult* result, const int* logits_finite,
    const int* routing_finite, PredictionStatus* status) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint32_t row = result->output_count - 1U;
  if constexpr (kCommitLogits) {
    if (index < kVocabulary) {
      committed_logits[index] = batch_logits[row * kVocabulary + index];
    }
  }
  if (index == 0U) {
    const std::uint32_t token = result->verified[row];
    status->token = token;
    status->logit = batch_logits[row * kVocabulary + token];
    status->finite = logits_finite[row];
    status->routing_finite = *routing_finite;
  }
}

__global__ void LatchMtpGroupNonFiniteKernel(
    const int* logits_finite, std::uint32_t rows,
    const int* routing_finite, MtpChainResult* chain_result) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) return;
  bool non_finite = *routing_finite == 0;
  for (std::uint32_t row = 0U; row < rows; ++row) {
    non_finite = non_finite || logits_finite[row] == 0;
  }
  if (non_finite) ++chain_result->non_finite_step_count;
}

__global__ void LatchMtpOrdinaryNonFiniteKernel(
    const PredictionStatus* status, MtpChainResult* chain_result) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) return;
  if (status->finite == 0 || status->routing_finite == 0) {
    ++chain_result->non_finite_step_count;
  }
}

int CaptureIndex(std::uint32_t layer, bool capture_all_layers) {
  if (layer >= kLayers) return -1;
  if (capture_all_layers) return static_cast<int>(layer);
  for (const std::uint32_t selected : kCaptureLayers) {
    if (selected == layer) return static_cast<int>(layer);
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
