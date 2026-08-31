#include "cuda/engine/gemma4_26b_vision_artifact.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <utility>

#include "compiler/sha256.h"

namespace gem16::internal {
namespace {

constexpr std::uint64_t kStagingBytes = 64U * 1024U * 1024U;

Status CudaFailure(std::string_view operation, cudaError_t error) {
  return Status(error == cudaErrorMemoryAllocation
                    ? StatusCode::kResourceExhausted
                    : StatusCode::kInternal,
                std::string(operation) + ": " + cudaGetErrorName(error) +
                    ": " + cudaGetErrorString(error));
}

Status Upload(const Gemma4Moe26BVisionModulePlan& plan,
              std::byte* destination) {
  if (destination == nullptr ||
      plan.artifact_bytes >
          static_cast<std::uint64_t>(
              std::numeric_limits<std::streamoff>::max()) ||
      plan.payload_file_offset > plan.artifact_bytes ||
      plan.artifact_bytes - plan.payload_file_offset !=
          kGemma4Moe26BVisionPayloadBytes) {
    return Status(StatusCode::kInvalidArgument,
                  "invalid Vision module upload plan");
  }
  cudaStream_t stream = nullptr;
  std::byte* staging = nullptr;
  cudaError_t error = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (error == cudaSuccess) {
    error = cudaHostAlloc(reinterpret_cast<void**>(&staging),
                          static_cast<std::size_t>(kStagingBytes),
                          cudaHostAllocDefault);
  }
  if (error != cudaSuccess) {
    if (stream != nullptr) (void)cudaStreamDestroy(stream);
    return CudaFailure("create Vision upload staging", error);
  }
  auto cleanup = [&] {
    (void)cudaFreeHost(staging);
    (void)cudaStreamDestroy(stream);
  };
  std::ifstream input(plan.artifact, std::ios::binary);
  if (!input) {
    cleanup();
    return Status(StatusCode::kIoError,
                  "cannot open Vision module payload");
  }
  compiler::Sha256 sha256;
  std::uint64_t file_offset = 0U;
  while (file_offset != plan.artifact_bytes) {
    const std::uint64_t count = std::min<std::uint64_t>(
        kStagingBytes, plan.artifact_bytes - file_offset);
    input.read(reinterpret_cast<char*>(staging),
               static_cast<std::streamsize>(count));
    if (input.gcount() != static_cast<std::streamsize>(count)) {
      cleanup();
      return Status(StatusCode::kDataLoss,
                    "short read from Vision module payload");
    }
    sha256.Update(staging, static_cast<std::size_t>(count));
    const std::uint64_t begin =
        std::max(file_offset, plan.payload_file_offset);
    const std::uint64_t end = std::min(
        file_offset + count,
        plan.payload_file_offset + kGemma4Moe26BVisionPayloadBytes);
    if (begin < end) {
      error = cudaMemcpyAsync(
          destination + (begin - plan.payload_file_offset),
          staging + (begin - file_offset), static_cast<std::size_t>(end - begin),
          cudaMemcpyHostToDevice, stream);
      if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
      if (error != cudaSuccess) {
        cleanup();
        return CudaFailure("upload Vision module payload", error);
      }
    }
    file_offset += count;
  }
  cleanup();
  if (sha256.HexDigest() != plan.artifact_sha256) {
    return Status(StatusCode::kDataLoss,
                  "Vision module changed between validation and upload");
  }
  return Status::Ok();
}

}  // namespace

Gemma4Moe26BVisionDeviceArtifact::~Gemma4Moe26BVisionDeviceArtifact() {
  if (arena_ != nullptr) (void)cudaFree(arena_);
}

Gemma4Moe26BVisionDeviceArtifact::Gemma4Moe26BVisionDeviceArtifact(
    Gemma4Moe26BVisionDeviceArtifact&& other) noexcept {
  *this = std::move(other);
}

Gemma4Moe26BVisionDeviceArtifact&
Gemma4Moe26BVisionDeviceArtifact::operator=(
    Gemma4Moe26BVisionDeviceArtifact&& other) noexcept {
  if (this == &other) return *this;
  if (arena_ != nullptr) (void)cudaFree(arena_);
  arena_ = std::exchange(other.arena_, nullptr);
  arena_bytes_ = std::exchange(other.arena_bytes_, 0U);
  tensors_ = std::move(other.tensors_);
  stats_ = std::move(other.stats_);
  return *this;
}

Result<Gemma4Moe26BVisionDeviceArtifact>
Gemma4Moe26BVisionDeviceArtifact::Load(
    const std::filesystem::path& module_root) {
  auto plan = LoadGemma4Moe26BVisionModulePlan(module_root);
  if (!plan.ok()) return plan.status();
  Gemma4Moe26BVisionDeviceArtifact artifact;
  artifact.arena_bytes_ = kGemma4Moe26BVisionPayloadBytes;
  const cudaError_t allocated = cudaMalloc(
      reinterpret_cast<void**>(&artifact.arena_),
      static_cast<std::size_t>(artifact.arena_bytes_));
  if (allocated != cudaSuccess) {
    return CudaFailure("allocate Vision immutable weight arena", allocated);
  }
  Status status = Upload(plan.value(), artifact.arena_);
  if (!status.ok()) return status;
  for (const auto& [name, tensor] : plan.value().tensors) {
    if (!artifact.tensors_
             .emplace(name, artifact.arena_ + tensor.offset)
             .second) {
      return Status(StatusCode::kDataLoss,
                    "duplicate Vision device tensor binding");
    }
  }
  artifact.stats_.arena_bytes = artifact.arena_bytes_;
  artifact.stats_.uploaded_bytes = artifact.arena_bytes_;
  artifact.stats_.tensor_count = artifact.tensors_.size();
  artifact.stats_.device_allocations = 1U;
  artifact.stats_.host_staging_peak_bytes = kStagingBytes;
  artifact.stats_.artifact_sha256 = plan.value().artifact_sha256;
  artifact.stats_.load_path =
      "vision_fp8_single_arena_pinned_sha256_no_repack";
  return artifact;
}

Result<const std::byte*> Gemma4Moe26BVisionDeviceArtifact::Pointer(
    std::string_view name) const {
  const auto found = tensors_.find(name);
  if (arena_ == nullptr || found == tensors_.end()) {
    return Status(StatusCode::kNotFound,
                  "Vision device tensor is not bound: " +
                      std::string(name));
  }
  return found->second;
}

}  // namespace gem16::internal
