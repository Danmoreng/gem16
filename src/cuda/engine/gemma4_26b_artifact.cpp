#include "cuda/engine/gemma4_26b_artifact.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "cuda/nvfp4/sm120_layout.h"
#include "platform/mapped_file.h"

namespace gem16::internal {
namespace {

constexpr std::uint64_t kUploadStagingBytes = 4U * 1024U * 1024U;

Status CudaFailure(const char* operation, cudaError_t error) {
  return Status(StatusCode::kInternal,
                std::string(operation) + ": " + cudaGetErrorName(error) +
                    ": " + cudaGetErrorString(error));
}

Result<std::uint64_t> CheckedProduct(std::span<const std::uint64_t> factors,
                                     std::string_view name) {
  std::uint64_t result = 1U;
  for (const std::uint64_t factor : factors) {
    if (factor != 0U &&
        result > std::numeric_limits<std::uint64_t>::max() / factor) {
      return Status(StatusCode::kDataLoss,
                    std::string(name) + " geometry overflows uint64");
    }
    result *= factor;
  }
  return result;
}

Result<std::pair<std::uint64_t, std::uint64_t>> FlattenGeometry(
    const std::vector<std::uint64_t>& shape, std::uint64_t factor,
    std::string_view name) {
  if (shape.size() < 2U || factor == 0U) {
    return Status(StatusCode::kDataLoss,
                  "invalid NVFP4 geometry: " + std::string(name));
  }
  auto rows = CheckedProduct(
      std::span<const std::uint64_t>(shape.data(), shape.size() - 1U), name);
  if (!rows.ok()) return rows.status();
  const std::array<std::uint64_t, 2> columns = {shape.back(), factor};
  auto contracting = CheckedProduct(columns, name);
  if (!contracting.ok()) return contracting.status();
  return std::make_pair(rows.value(), contracting.value());
}

Status UploadTiled(std::byte* destination,
                   std::span<const std::uint8_t> source,
                   const Sm120Nvfp4SourceLayout& layout,
                   std::uint64_t bytes_per_k_block,
                   std::uint64_t source_row_bytes,
                   Gemma4Moe26BDeviceArtifactStats* stats) {
  if (destination == nullptr || bytes_per_k_block == 0U ||
      source_row_bytes != layout.k_blocks * bytes_per_k_block ||
      source.size() != layout.rows * source_row_bytes) {
    return Status(StatusCode::kDataLoss,
                  "invalid source buffer for M11 device artifact tiling");
  }
  const std::uint64_t full_tile_bytes =
      8U * layout.k_blocks * bytes_per_k_block;
  const std::uint64_t tiles_per_batch = std::max<std::uint64_t>(
      1U, kUploadStagingBytes / std::max<std::uint64_t>(1U, full_tile_bytes));
  const std::uint64_t staging_bytes = std::min<std::uint64_t>(
      source.size(), tiles_per_batch * full_tile_bytes);
  std::vector<std::uint8_t> staging(static_cast<std::size_t>(staging_bytes));
  stats->host_staging_peak_bytes =
      std::max(stats->host_staging_peak_bytes, staging_bytes);
  for (std::uint64_t first_tile = 0; first_tile < layout.row_tiles;
       first_tile += tiles_per_batch) {
    const std::uint64_t end_tile =
        std::min(layout.row_tiles, first_tile + tiles_per_batch);
    std::uint64_t cursor = 0U;
    for (std::uint64_t row_tile = first_tile; row_tile < end_tile;
         ++row_tile) {
      const std::uint64_t tile_first_row = row_tile * 8U;
      const std::uint64_t tile_rows =
          std::min<std::uint64_t>(8U, layout.rows - tile_first_row);
      for (std::uint64_t k_block = 0; k_block < layout.k_blocks; ++k_block) {
        for (std::uint64_t row = 0; row < tile_rows; ++row) {
          const std::uint64_t source_offset =
              (tile_first_row + row) * source_row_bytes +
              k_block * bytes_per_k_block;
          std::copy_n(source.data() + source_offset, bytes_per_k_block,
                      staging.data() + cursor);
          cursor += bytes_per_k_block;
        }
      }
    }
    const std::uint64_t destination_offset =
        first_tile * 8U * layout.k_blocks * bytes_per_k_block;
    const cudaError_t error = cudaMemcpy(
        destination + destination_offset, staging.data(),
        static_cast<std::size_t>(cursor), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) return CudaFailure("upload tiled M11 weight", error);
  }
  return Status::Ok();
}

Status UploadOne(std::byte* arena, const TensorInfo& tensor,
                 const Gemma4Moe26BUploadRange& range,
                 const MappedFile& mapped,
                 Gemma4Moe26BDeviceArtifactStats* stats,
                 std::map<std::string, float, std::less<>>* host_f32) {
  if (range.source_offset > mapped.size() ||
      range.bytes > mapped.size() - range.source_offset ||
      range.bytes > std::numeric_limits<std::size_t>::max()) {
    return Status(StatusCode::kDataLoss,
                  "invalid M11 artifact source range: " + tensor.name);
  }
  const auto* source = reinterpret_cast<const std::uint8_t*>(
      mapped.data() + range.source_offset);
  std::byte* destination = arena + range.destination_offset;
  Status status;
  if (range.runtime_layout.ends_with("sm120_row8_k64")) {
    auto geometry = FlattenGeometry(tensor.logical_shape, 1U, tensor.name);
    if (!geometry.ok()) return geometry.status();
    auto layout = PlanSm120Nvfp4SourceLayout(geometry.value().first,
                                             geometry.value().second);
    if (!layout.ok()) return layout.status();
    const std::uint64_t row_bytes = geometry.value().second / 2U;
    if (range.bytes != geometry.value().first * row_bytes) {
      return Status(StatusCode::kDataLoss,
                    "M11 packed NVFP4 byte count mismatch: " + tensor.name);
    }
    status = UploadTiled(destination,
                         {source, static_cast<std::size_t>(range.bytes)},
                         layout.value(), 32U, row_bytes, stats);
    if (!status.ok()) return status;
    ++stats->tiled_weight_tensors;
  } else if (range.runtime_layout.ends_with("sm120_row8_group16_e4m3")) {
    auto geometry = FlattenGeometry(tensor.shape, 16U, tensor.name);
    if (!geometry.ok()) return geometry.status();
    auto layout = PlanSm120Nvfp4SourceLayout(geometry.value().first,
                                             geometry.value().second);
    if (!layout.ok()) return layout.status();
    const std::uint64_t row_bytes = tensor.shape.back();
    if (range.bytes != geometry.value().first * row_bytes) {
      return Status(StatusCode::kDataLoss,
                    "M11 NVFP4 scale byte count mismatch: " + tensor.name);
    }
    status = UploadTiled(destination,
                         {source, static_cast<std::size_t>(range.bytes)},
                         layout.value(), 4U, row_bytes, stats);
    if (!status.ok()) return status;
    ++stats->tiled_scale_tensors;
  } else {
    const cudaError_t error = cudaMemcpy(
        destination, source, static_cast<std::size_t>(range.bytes),
        cudaMemcpyHostToDevice);
    if (error != cudaSuccess) return CudaFailure("upload M11 artifact tensor", error);
    ++stats->direct_tensors;
  }
  if (tensor.storage_dtype == "F32" && tensor.shape.size() == 1U &&
      tensor.shape[0] == 1U && range.bytes == sizeof(float)) {
    const float value = std::bit_cast<float>(
        std::array<std::uint8_t, 4>{source[0], source[1], source[2], source[3]});
    if (!std::isfinite(value) || value <= 0.0F) {
      return Status(StatusCode::kDataLoss,
                    "invalid positive F32 artifact scalar: " + tensor.name);
    }
    host_f32->emplace(tensor.name, value);
  }
  ++stats->tensors;
  stats->payload_bytes += range.bytes;
  return Status::Ok();
}

}  // namespace

Gemma4Moe26BDeviceArtifact::~Gemma4Moe26BDeviceArtifact() {
  if (arena_ != nullptr) (void)cudaFree(arena_);
}

Gemma4Moe26BDeviceArtifact::Gemma4Moe26BDeviceArtifact(
    Gemma4Moe26BDeviceArtifact&& other) noexcept {
  *this = std::move(other);
}

Gemma4Moe26BDeviceArtifact& Gemma4Moe26BDeviceArtifact::operator=(
    Gemma4Moe26BDeviceArtifact&& other) noexcept {
  if (this == &other) return *this;
  if (arena_ != nullptr) (void)cudaFree(arena_);
  arena_ = std::exchange(other.arena_, nullptr);
  arena_bytes_ = std::exchange(other.arena_bytes_, 0U);
  offsets_ = std::move(other.offsets_);
  host_f32_ = std::move(other.host_f32_);
  stats_ = other.stats_;
  return *this;
}

Result<Gemma4Moe26BDeviceArtifact> Gemma4Moe26BDeviceArtifact::Load(
    const std::filesystem::path& model_directory,
    const ModelManifest& manifest, const Gemma4Moe26BResidencyPlan& plan) {
  if (plan.upload_ranges.size() != manifest.tensors.size() ||
      plan.immutable_weight_arena_bytes == 0U ||
      plan.immutable_weight_arena_bytes >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return Status(StatusCode::kInvalidArgument,
                  "invalid M11 immutable device artifact plan");
  }
  Gemma4Moe26BDeviceArtifact artifact;
  artifact.arena_bytes_ = plan.immutable_weight_arena_bytes;
  const cudaError_t allocated = cudaMalloc(
      reinterpret_cast<void**>(&artifact.arena_),
      static_cast<std::size_t>(artifact.arena_bytes_));
  if (allocated != cudaSuccess) {
    return CudaFailure("allocate M11 immutable weight arena", allocated);
  }
  std::map<std::string, const TensorInfo*, std::less<>> tensors;
  for (const auto& tensor : manifest.tensors) tensors.emplace(tensor.name, &tensor);
  std::set<std::string> shards;
  for (const auto& range : plan.upload_ranges) {
    shards.insert(range.source_shard);
    if (!artifact.offsets_.emplace(range.tensor_name,
                                   range.destination_offset).second) {
      return Status(StatusCode::kDataLoss,
                    "duplicate M11 device artifact tensor");
    }
  }
  artifact.stats_.shards = shards.size();
  for (const auto& shard : shards) {
    auto mapped = MappedFile::Open(model_directory / shard);
    if (!mapped.ok()) return mapped.status();
    for (const auto& range : plan.upload_ranges) {
      if (range.source_shard != shard) continue;
      const auto found = tensors.find(range.tensor_name);
      if (found == tensors.end()) {
        return Status(StatusCode::kDataLoss,
                      "M11 upload tensor is missing from manifest");
      }
      Status status = UploadOne(artifact.arena_, *found->second, range,
                                mapped.value(), &artifact.stats_,
                                &artifact.host_f32_);
      if (!status.ok()) return status;
    }
  }
  if (artifact.stats_.tensors != plan.upload_ranges.size() ||
      artifact.stats_.payload_bytes != plan.artifact_payload_bytes) {
    return Status(StatusCode::kDataLoss,
                  "M11 upload did not consume the exact artifact");
  }
  const cudaError_t synchronized = cudaDeviceSynchronize();
  if (synchronized != cudaSuccess) {
    return CudaFailure("synchronize M11 immutable artifact upload",
                       synchronized);
  }
  return artifact;
}

Result<const std::byte*> Gemma4Moe26BDeviceArtifact::Pointer(
    std::string_view name) const {
  const auto found = offsets_.find(name);
  if (arena_ == nullptr || found == offsets_.end()) {
    return Status(StatusCode::kNotFound,
                  "M11 device tensor is not bound: " + std::string(name));
  }
  return arena_ + found->second;
}

Result<float> Gemma4Moe26BDeviceArtifact::HostFloat32(
    std::string_view name) const {
  const auto found = host_f32_.find(name);
  if (found == host_f32_.end()) {
    return Status(StatusCode::kNotFound,
                  "M11 host scalar metadata is not bound: " +
                      std::string(name));
  }
  return found->second;
}

}  // namespace gem16::internal
