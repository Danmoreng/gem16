#include "cuda/engine/gemma4_26b_trellis35_artifact.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
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

class UploadContext {
 public:
  UploadContext() = default;
  UploadContext(const UploadContext&) = delete;
  UploadContext& operator=(const UploadContext&) = delete;
  ~UploadContext() {
    if (staging_ != nullptr) (void)cudaFreeHost(staging_);
    if (stream_ != nullptr) (void)cudaStreamDestroy(stream_);
  }

  Status Initialize() {
    cudaError_t error =
        cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
    if (error != cudaSuccess) {
      return CudaFailure("create Trellis35 upload stream", error);
    }
    error = cudaHostAlloc(reinterpret_cast<void**>(&staging_),
                          static_cast<std::size_t>(kStagingBytes),
                          cudaHostAllocDefault);
    if (error != cudaSuccess) {
      return CudaFailure("allocate Trellis35 pinned upload staging", error);
    }
    return Status::Ok();
  }

  Status Upload(const Trellis35FileIdentity& file, std::byte* destination) {
    if (destination == nullptr || file.bytes == 0U || file.sha256.size() != 64U ||
        file.bytes >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::streamoff>::max())) {
      return Status(StatusCode::kInvalidArgument,
                    "invalid Trellis35 upload file identity");
    }
    std::ifstream input(file.path, std::ios::binary);
    if (!input) {
      return Status(StatusCode::kIoError,
                    "cannot open Trellis35 payload: " + file.path.string());
    }
    compiler::Sha256 sha256;
    std::uint64_t offset = 0U;
    while (offset != file.bytes) {
      const std::uint64_t count =
          std::min<std::uint64_t>(kStagingBytes, file.bytes - offset);
      input.read(reinterpret_cast<char*>(staging_),
                 static_cast<std::streamsize>(count));
      if (input.gcount() != static_cast<std::streamsize>(count)) {
        return Status(StatusCode::kDataLoss,
                      "short read from Trellis35 payload: " +
                          file.path.string());
      }
      sha256.Update(staging_, static_cast<std::size_t>(count));
      const cudaError_t copied = cudaMemcpyAsync(
          destination + offset, staging_, static_cast<std::size_t>(count),
          cudaMemcpyHostToDevice, stream_);
      if (copied != cudaSuccess) {
        return CudaFailure("upload Trellis35 payload", copied);
      }
      const cudaError_t synchronized = cudaStreamSynchronize(stream_);
      if (synchronized != cudaSuccess) {
        return CudaFailure("synchronize Trellis35 payload upload",
                           synchronized);
      }
      offset += count;
    }
    if (sha256.HexDigest() != file.sha256) {
      return Status(StatusCode::kDataLoss,
                    "Trellis35 payload SHA-256 mismatch: " +
                        file.path.string());
    }
    return Status::Ok();
  }

 private:
  cudaStream_t stream_ = nullptr;
  std::byte* staging_ = nullptr;
};

Trellis35DeviceFamilyBinding BindFamily(
    const std::byte* layer_base, const Trellis35FamilyPlan& plan) {
  Trellis35DeviceFamilyBinding binding;
  binding.k3_payload_pool = layer_base + plan.k3_payload_pool.offset;
  binding.k4_payload_pool = layer_base + plan.k4_payload_pool.offset;
  binding.descriptors = reinterpret_cast<const Trellis35ExpertDescriptor*>(
      layer_base + plan.descriptor.offset);
  binding.suh_f16 = reinterpret_cast<const std::uint16_t*>(
      layer_base + plan.suh.offset);
  binding.svh_f16 = reinterpret_cast<const std::uint16_t*>(
      layer_base + plan.svh.offset);
  binding.rate_map = plan.rate_map;
  return binding;
}

}  // namespace

Gemma4Moe26BTrellis35DeviceArtifact::
    ~Gemma4Moe26BTrellis35DeviceArtifact() {
  if (arena_ != nullptr) (void)cudaFree(arena_);
}

Gemma4Moe26BTrellis35DeviceArtifact::
    Gemma4Moe26BTrellis35DeviceArtifact(
        Gemma4Moe26BTrellis35DeviceArtifact&& other) noexcept {
  *this = std::move(other);
}

Gemma4Moe26BTrellis35DeviceArtifact&
Gemma4Moe26BTrellis35DeviceArtifact::operator=(
    Gemma4Moe26BTrellis35DeviceArtifact&& other) noexcept {
  if (this == &other) return *this;
  if (arena_ != nullptr) (void)cudaFree(arena_);
  arena_ = std::exchange(other.arena_, nullptr);
  arena_bytes_ = std::exchange(other.arena_bytes_, 0U);
  non_routed_ = std::move(other.non_routed_);
  host_f32_ = std::move(other.host_f32_);
  layers_ = other.layers_;
  stats_ = std::move(other.stats_);
  return *this;
}

Result<Gemma4Moe26BTrellis35DeviceArtifact>
Gemma4Moe26BTrellis35DeviceArtifact::Load(
    const std::filesystem::path& checkpoint_root) {
  auto plan = LoadGemma4Moe26BTrellis35CheckpointPlan(checkpoint_root);
  if (!plan.ok()) return plan.status();
  if (plan.value().arena_bytes != kTrellis35CheckpointBytes ||
      plan.value().nvfp4_routed_expert_bytes != 0U) {
    return Status(StatusCode::kDataLoss,
                  "Trellis35 plan violates the one-arena/no-NVFP4 contract");
  }
  Gemma4Moe26BTrellis35DeviceArtifact artifact;
  artifact.arena_bytes_ = plan.value().arena_bytes;
  const cudaError_t allocated = cudaMalloc(
      reinterpret_cast<void**>(&artifact.arena_),
      static_cast<std::size_t>(artifact.arena_bytes_));
  if (allocated != cudaSuccess) {
    return CudaFailure("allocate Trellis35 immutable weight arena", allocated);
  }
  UploadContext upload;
  Status status = upload.Initialize();
  if (!status.ok()) return status;
  status = upload.Upload(plan.value().non_routed, artifact.arena_);
  if (!status.ok()) return status;
  artifact.stats_.uploaded_bytes += plan.value().non_routed.bytes;
  ++artifact.stats_.files;
  for (const auto& layer : plan.value().layers) {
    status = upload.Upload(layer.artifact, artifact.arena_ + layer.arena_offset);
    if (!status.ok()) return status;
    artifact.stats_.uploaded_bytes += layer.artifact.bytes;
    ++artifact.stats_.files;
  }
  if (artifact.stats_.uploaded_bytes != artifact.arena_bytes_ ||
      artifact.stats_.files != kTrellis35LayerCount + 1U) {
    return Status(StatusCode::kDataLoss,
                  "Trellis35 upload did not consume the exact checkpoint");
  }
  for (const auto& [name, tensor] : plan.value().non_routed_tensors) {
    if (!artifact.non_routed_
             .emplace(name, artifact.arena_ + tensor.offset)
             .second) {
      return Status(StatusCode::kDataLoss,
                    "duplicate Trellis35 non-routed device binding");
    }
    if (tensor.bytes == sizeof(float)) {
      float value = 0.0F;
      const cudaError_t copied = cudaMemcpy(
          &value, artifact.arena_ + tensor.offset, sizeof(value),
          cudaMemcpyDeviceToHost);
      if (copied != cudaSuccess) {
        return CudaFailure("copy Trellis35 host scalar", copied);
      }
      artifact.host_f32_.emplace(name, value);
    }
  }
  for (const auto& layer : plan.value().layers) {
    const std::byte* base = artifact.arena_ + layer.arena_offset;
    artifact.layers_[layer.layer].gate_up =
        BindFamily(base, layer.gate_up);
    artifact.layers_[layer.layer].down = BindFamily(base, layer.down);
  }
  artifact.stats_.arena_bytes = artifact.arena_bytes_;
  artifact.stats_.non_routed_tensors = artifact.non_routed_.size();
  artifact.stats_.device_allocations = 1U;
  artifact.stats_.host_staging_peak_bytes = kStagingBytes;
  artifact.stats_.checkpoint_content_sha256 =
      plan.value().checkpoint_content_sha256;
  artifact.stats_.load_path =
      "trellis35_single_arena_pinned_sha256_no_nvfp4_experts";
  return artifact;
}

Result<float> Gemma4Moe26BTrellis35DeviceArtifact::HostFloat32(
    std::string_view name) const {
  const auto found = host_f32_.find(name);
  if (found == host_f32_.end()) {
    return Status(StatusCode::kNotFound,
                  "Trellis35 host scalar metadata is not bound: " +
                      std::string(name));
  }
  return found->second;
}

Result<const std::byte*>
Gemma4Moe26BTrellis35DeviceArtifact::NonRoutedPointer(
    std::string_view name) const {
  const auto found = non_routed_.find(name);
  if (arena_ == nullptr || found == non_routed_.end()) {
    return Status(StatusCode::kNotFound,
                  "Trellis35 non-routed tensor is not bound: " +
                      std::string(name));
  }
  return found->second;
}

}  // namespace gem16::internal
