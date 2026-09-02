#include "cuda/engine/gemma4_26b_trellis35_artifact.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string>
#include <tuple>
#include <utility>

#include "cuda/engine/gemma4_26b_trellis35_device_image.h"

namespace gem16::internal {
namespace {

constexpr std::uint64_t kStagingBytes = 64U * 1024U * 1024U;

enum class Trellis35StorageFormat { kDeviceImageV2, kLegacyV1 };

Status CudaFailure(std::string_view operation, cudaError_t error) {
  return Status(error == cudaErrorMemoryAllocation
                    ? StatusCode::kResourceExhausted
                    : StatusCode::kInternal,
                std::string(operation) + ": " + cudaGetErrorName(error) +
                    ": " + cudaGetErrorString(error));
}

Result<Trellis35StorageFormat> RequestedStorageFormat() {
  const char* value = std::getenv("GEM16_TRELLIS35_FORMAT");
  if (value == nullptr || std::string_view(value) == "v2") {
    return Trellis35StorageFormat::kDeviceImageV2;
  }
  if (std::string_view(value) == "legacy-v1") {
    return Trellis35StorageFormat::kLegacyV1;
  }
  return Status(StatusCode::kInvalidArgument,
                "GEM16_TRELLIS35_FORMAT must be v2 or legacy-v1");
}

bool PathExists(const std::filesystem::path& path) {
  std::error_code error;
  const auto status = std::filesystem::symlink_status(path, error);
  return !error && status.type() != std::filesystem::file_type::not_found;
}

bool HasAnyV2Marker(const std::filesystem::path& root) {
  return PathExists(root / "model.gem16") ||
         PathExists(root / "gem16_model.json") ||
         PathExists(root / "gem16_compilation.json") ||
         PathExists(root / "gem16.lock.json");
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

Status ValidateUploadedDescriptors(
    const std::array<Trellis35LayerPlan, kTrellis35LayerCount>& layers,
    const std::byte* arena) {
  if (arena == nullptr) {
    return Status(StatusCode::kInvalidArgument,
                  "missing uploaded Trellis35 descriptor arena");
  }
  for (const auto& layer : layers) {
    for (const auto& [family, rows, columns, description] :
         std::array<std::tuple<const Trellis35FamilyPlan*, std::uint64_t,
                               std::uint64_t, std::string_view>,
                    2>{std::tuple{&layer.gate_up, 2816U, 1408U, "gate_up"},
                       std::tuple{&layer.down, 768U, 2816U, "down"}}) {
      std::array<Trellis35ExpertDescriptor, kTrellis35ExpertCount> descriptors{};
      const cudaError_t copied = cudaMemcpy(
          descriptors.data(),
          arena + layer.arena_offset + family->descriptor.offset,
          sizeof(descriptors), cudaMemcpyDeviceToHost);
      if (copied != cudaSuccess) {
        return CudaFailure("read uploaded Trellis35 v2 descriptors", copied);
      }
      std::array<std::uint64_t, 5> next{};
      for (std::size_t expert = 0U; expert < descriptors.size(); ++expert) {
        const std::uint16_t rate = family->rate_map[expert];
        const auto& descriptor = descriptors[expert];
        if (descriptor.rate_bits != rate ||
            descriptor.codebook_id != kTrellis35CodebookId ||
            descriptor.pool_offset != next[rate]) {
          return Status(StatusCode::kDataLoss,
                        "uploaded Trellis35 v2 descriptor is invalid: " +
                            std::string(description));
        }
        next[rate] += rows * columns * rate / 8U;
      }
      if (next[3] != family->k3_payload_pool.bytes ||
          next[4] != family->k4_payload_pool.bytes) {
        return Status(StatusCode::kDataLoss,
                      "uploaded Trellis35 v2 descriptor coverage is invalid: " +
                          std::string(description));
      }
    }
  }
  return Status::Ok();
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
  const auto load_started = std::chrono::steady_clock::now();
  auto requested_format = RequestedStorageFormat();
  if (!requested_format.ok()) return requested_format.status();
  if (requested_format.value() == Trellis35StorageFormat::kLegacyV1 &&
      HasAnyV2Marker(checkpoint_root)) {
    return Status(StatusCode::kDataLoss,
                  "legacy Trellis35 v1 was requested but v2 files are present");
  }
  if (requested_format.value() == Trellis35StorageFormat::kDeviceImageV2 &&
      !HasAnyV2Marker(checkpoint_root) &&
      PathExists(checkpoint_root / "trellis35-checkpoint.json")) {
    return Status(
        StatusCode::kUnsupported,
        "Trellis35 v1 is legacy-only; set GEM16_TRELLIS35_FORMAT=legacy-v1 "
        "explicitly or install the v2 device image");
  }
  auto plan =
      requested_format.value() == Trellis35StorageFormat::kDeviceImageV2
          ? LoadGemma4Moe26BTrellis35DeviceImagePlan(checkpoint_root)
          : LoadGemma4Moe26BTrellis35CheckpointPlan(checkpoint_root);
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
  Status status;
  if (requested_format.value() == Trellis35StorageFormat::kDeviceImageV2) {
    auto uploaded = UploadGemma4Moe26BTrellis35DeviceImage(
        plan.value().non_routed.path, artifact.arena_, artifact.arena_bytes_);
    if (!uploaded.ok()) return uploaded.status();
    artifact.stats_.uploaded_bytes = uploaded.value().uploaded_bytes;
    artifact.stats_.files = 1U;
    artifact.stats_.host_staging_peak_bytes =
        uploaded.value().host_staging_peak_bytes;
    artifact.stats_.upload_milliseconds =
        uploaded.value().upload_milliseconds;
    artifact.stats_.load_path = uploaded.value().load_path;
    artifact.stats_.storage_format_version = 2U;
    status = ValidateUploadedDescriptors(plan.value().layers, artifact.arena_);
    if (!status.ok()) return status;
  } else {
    const auto upload_started = std::chrono::steady_clock::now();
    status = upload.Initialize();
    if (!status.ok()) return status;
    status = upload.Upload(plan.value().non_routed, artifact.arena_);
    if (!status.ok()) return status;
    artifact.stats_.uploaded_bytes += plan.value().non_routed.bytes;
    ++artifact.stats_.files;
    for (const auto& layer : plan.value().layers) {
      status =
          upload.Upload(layer.artifact, artifact.arena_ + layer.arena_offset);
      if (!status.ok()) return status;
      artifact.stats_.uploaded_bytes += layer.artifact.bytes;
      ++artifact.stats_.files;
    }
    artifact.stats_.upload_milliseconds =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - upload_started)
            .count();
    artifact.stats_.host_staging_peak_bytes = kStagingBytes;
    artifact.stats_.storage_format_version = 1U;
    artifact.stats_.load_path =
        "trellis35_legacy_v1_explicit_single_staging_structural";
  }
  if (artifact.stats_.uploaded_bytes != artifact.arena_bytes_ ||
      artifact.stats_.files !=
          (requested_format.value() == Trellis35StorageFormat::kDeviceImageV2
               ? 1U
               : kTrellis35LayerCount + 1U)) {
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
  artifact.stats_.checkpoint_content_sha256 =
      plan.value().checkpoint_content_sha256;
  artifact.stats_.runtime_payload_sha256 = false;
  artifact.stats_.load_milliseconds =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - load_started)
          .count();
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
