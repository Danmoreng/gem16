#include "cuda/engine/gemma4_26b_artifact.h"

#include <cuda_runtime_api.h>

#if defined(GEM16_HAS_OPENSSL)
#include <openssl/evp.h>
#endif

#if defined(GEM16_HAS_CUFILE)
#include <cufile.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "compiler/sha256.h"
#include "cuda/nvfp4/sm120_layout.h"
#include "model/gemma4_26b_device_image.h"
#include "platform/mapped_file.h"

namespace gem16::internal {
namespace {

constexpr std::uint64_t kUploadStagingBytes = 4U * 1024U * 1024U;
constexpr std::uint64_t kImageUploadBufferBytes = 64U * 1024U * 1024U;
constexpr std::size_t kImageUploadBuffers = 4U;
constexpr std::uint64_t kCuFileRequestBytes = 1024U * 1024U * 1024U;

enum class DeviceImageIo { kAuto, kPinned, kCuFile };

Status CudaFailure(const char* operation, cudaError_t error) {
  return Status(StatusCode::kInternal,
                std::string(operation) + ": " + cudaGetErrorName(error) +
                    ": " + cudaGetErrorString(error));
}

#if defined(GEM16_HAS_OPENSSL)
std::string HexDigest(const std::array<std::uint8_t, 32>& digest) {
  constexpr std::array<char, 16> kHex = {
      '0', '1', '2', '3', '4', '5', '6', '7',
      '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};
  std::string result(digest.size() * 2U, '0');
  for (std::size_t index = 0; index < digest.size(); ++index) {
    result[index * 2U] = kHex[digest[index] >> 4U];
    result[index * 2U + 1U] = kHex[digest[index] & 0x0fU];
  }
  return result;
}
#endif

class DeviceImageSha256 {
 public:
  DeviceImageSha256() = default;
  DeviceImageSha256(const DeviceImageSha256&) = delete;
  DeviceImageSha256& operator=(const DeviceImageSha256&) = delete;
#if defined(GEM16_HAS_OPENSSL)
  ~DeviceImageSha256() {
    if (context_ != nullptr) EVP_MD_CTX_free(context_);
  }
#endif

  Status Initialize() {
#if defined(GEM16_HAS_OPENSSL)
    context_ = EVP_MD_CTX_new();
    if (context_ == nullptr ||
        EVP_DigestInit_ex(context_, EVP_sha256(), nullptr) != 1) {
      return Status(StatusCode::kInternal,
                    "initialize accelerated SM120 device-image SHA-256");
    }
#endif
    return Status::Ok();
  }

  Status Update(const void* data, std::size_t size) {
#if defined(GEM16_HAS_OPENSSL)
    if (context_ == nullptr || EVP_DigestUpdate(context_, data, size) != 1) {
      return Status(StatusCode::kInternal,
                    "update accelerated SM120 device-image SHA-256");
    }
#else
    fallback_.Update(data, size);
#endif
    return Status::Ok();
  }

  Result<std::string> FinalHex() {
#if defined(GEM16_HAS_OPENSSL)
    std::array<std::uint8_t, 32> digest{};
    unsigned int digest_bytes = 0U;
    if (context_ == nullptr ||
        EVP_DigestFinal_ex(context_, digest.data(), &digest_bytes) != 1 ||
        digest_bytes != digest.size()) {
      return Status(StatusCode::kInternal,
                    "finish accelerated SM120 device-image SHA-256");
    }
    return HexDigest(digest);
#else
    return fallback_.HexDigest();
#endif
  }

 private:
#if defined(GEM16_HAS_OPENSSL)
  EVP_MD_CTX* context_ = nullptr;
#else
  compiler::Sha256 fallback_;
#endif
};

Result<DeviceImageIo> RequestedDeviceImageIo() {
  const char* value = std::getenv("GEM16_DEVICE_IMAGE_IO");
  if (value == nullptr || std::string_view(value) == "auto") {
    return DeviceImageIo::kAuto;
  }
  if (std::string_view(value) == "pinned") return DeviceImageIo::kPinned;
  if (std::string_view(value) == "cufile") return DeviceImageIo::kCuFile;
  return Status(StatusCode::kInvalidArgument,
                "GEM16_DEVICE_IMAGE_IO must be auto, pinned, or cufile");
}

Result<bool> GpuSupportsPeerDirectMemory() {
  int device = 0;
  cudaError_t error = cudaGetDevice(&device);
  if (error != cudaSuccess) {
    return CudaFailure("query current CUDA device", error);
  }
  int supported = 0;
  error = cudaDeviceGetAttribute(&supported, cudaDevAttrGPUDirectRDMASupported,
                                 device);
  if (error != cudaSuccess) {
    return CudaFailure("query CUDA GPUDirect capability", error);
  }
  return supported != 0;
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

class PinnedImageUpload {
 public:
  PinnedImageUpload() = default;
  PinnedImageUpload(const PinnedImageUpload&) = delete;
  PinnedImageUpload& operator=(const PinnedImageUpload&) = delete;
  ~PinnedImageUpload() {
    for (std::size_t index = 0; index < kImageUploadBuffers; ++index) {
      if (events_[index] != nullptr) (void)cudaEventDestroy(events_[index]);
      if (buffers_[index] != nullptr) (void)cudaFreeHost(buffers_[index]);
    }
    if (stream_ != nullptr) (void)cudaStreamDestroy(stream_);
  }

  Status Initialize() {
    cudaError_t error = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
    if (error != cudaSuccess) {
      return CudaFailure("create SM120 device-image upload stream", error);
    }
    for (std::size_t index = 0; index < kImageUploadBuffers; ++index) {
      error = cudaHostAlloc(reinterpret_cast<void**>(&buffers_[index]),
                            static_cast<std::size_t>(kImageUploadBufferBytes),
                            cudaHostAllocDefault);
      if (error != cudaSuccess) {
        return CudaFailure("allocate SM120 device-image pinned staging", error);
      }
      error = cudaEventCreateWithFlags(&events_[index], cudaEventDisableTiming);
      if (error != cudaSuccess) {
        return CudaFailure("create SM120 device-image upload event", error);
      }
    }
    return Status::Ok();
  }

  Status Upload(const std::filesystem::path& path, std::byte* destination,
                std::uint64_t bytes, bool verify_sha256,
                std::string* digest) {
    if (destination == nullptr || digest == nullptr) {
      return Status(StatusCode::kInvalidArgument,
                    "invalid SM120 device-image upload destination");
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) {
      return Status(StatusCode::kIoError,
                    "cannot open SM120 device image: " + path.string());
    }
    DeviceImageSha256 sha256;
    if (verify_sha256) {
      Status hash_initialized = sha256.Initialize();
      if (!hash_initialized.ok()) return hash_initialized;
    }
    std::array<bool, kImageUploadBuffers> in_flight{};
    std::uint64_t offset = 0U;
    std::size_t slot = 0U;
    while (offset != bytes) {
      if (in_flight[slot]) {
        const cudaError_t waited = cudaEventSynchronize(events_[slot]);
        if (waited != cudaSuccess) {
          return CudaFailure("wait for SM120 device-image upload", waited);
        }
      }
      const std::uint64_t count =
          std::min<std::uint64_t>(kImageUploadBufferBytes, bytes - offset);
      input.read(reinterpret_cast<char*>(buffers_[slot]),
                 static_cast<std::streamsize>(count));
      if (input.gcount() != static_cast<std::streamsize>(count)) {
        return Status(StatusCode::kDataLoss,
                      "short read from SM120 device image: " + path.string());
      }
      if (verify_sha256) {
        Status hashed =
            sha256.Update(buffers_[slot], static_cast<std::size_t>(count));
        if (!hashed.ok()) return hashed;
      }
      cudaError_t error = cudaMemcpyAsync(
          destination + offset, buffers_[slot], static_cast<std::size_t>(count),
          cudaMemcpyHostToDevice, stream_);
      if (error != cudaSuccess) {
        return CudaFailure("upload final-layout SM120 device image", error);
      }
      error = cudaEventRecord(events_[slot], stream_);
      if (error != cudaSuccess) {
        return CudaFailure("record SM120 device-image upload event", error);
      }
      in_flight[slot] = true;
      offset += count;
      slot = (slot + 1U) % kImageUploadBuffers;
    }
    const cudaError_t synchronized = cudaStreamSynchronize(stream_);
    if (synchronized != cudaSuccess) {
      return CudaFailure("synchronize final-layout SM120 device image",
                         synchronized);
    }
    if (verify_sha256) {
      auto final_digest = sha256.FinalHex();
      if (!final_digest.ok()) return final_digest.status();
      *digest = std::move(final_digest.value());
    } else {
      digest->clear();
    }
    return Status::Ok();
  }

 private:
  cudaStream_t stream_ = nullptr;
  std::array<std::byte*, kImageUploadBuffers> buffers_{};
  std::array<cudaEvent_t, kImageUploadBuffers> events_{};
};

#if defined(GEM16_HAS_CUFILE)
Status CuFileFailure(std::string_view operation, CUfileError_t error) {
  std::string message(operation);
  message.append(": ");
  message.append(CUFILE_ERRSTR(error.err));
  return Status(StatusCode::kIoError, std::move(message));
}

class CuFileImageUpload {
 public:
  CuFileImageUpload() = default;
  CuFileImageUpload(const CuFileImageUpload&) = delete;
  CuFileImageUpload& operator=(const CuFileImageUpload&) = delete;
  ~CuFileImageUpload() {
    if (handle_ != nullptr) cuFileHandleDeregister(handle_);
    if (file_descriptor_ >= 0) (void)close(file_descriptor_);
    if (driver_open_) (void)cuFileDriverClose();
  }

  Status Initialize(const std::filesystem::path& path) {
    CUfileError_t error = cuFileDriverOpen();
    if (error.err != CU_FILE_SUCCESS) {
      return CuFileFailure("open cuFile driver", error);
    }
    driver_open_ = true;
    file_descriptor_ = open(path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
    if (file_descriptor_ < 0) {
      return Status(StatusCode::kIoError,
                    "cannot open SM120 device image for cuFile: " +
                        std::string(std::strerror(errno)));
    }
    struct stat opened {};
    struct stat named {};
    if (fstat(file_descriptor_, &opened) != 0 || lstat(path.c_str(), &named) != 0 ||
        !S_ISREG(opened.st_mode) || !S_ISREG(named.st_mode) ||
        S_ISLNK(named.st_mode) || opened.st_dev != named.st_dev ||
        opened.st_ino != named.st_ino || opened.st_size < 0 ||
        static_cast<std::uint64_t>(opened.st_size) !=
            kAcceptedM08DeviceImageBytes) {
      return Status(StatusCode::kDataLoss,
                    "cuFile SM120 device image changed during secure open");
    }
    CUfileDescr_t descriptor{};
    descriptor.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    descriptor.handle.fd = file_descriptor_;
    error = cuFileHandleRegister(&handle_, &descriptor);
    if (error.err != CU_FILE_SUCCESS) {
      return CuFileFailure("register SM120 device image with cuFile", error);
    }
    return Status::Ok();
  }

  Status VerifyDigest(std::uint64_t bytes) const {
    if (file_descriptor_ < 0 || bytes > std::numeric_limits<std::size_t>::max()) {
      return Status(StatusCode::kInvalidArgument,
                    "invalid cuFile SM120 device-image hash range");
    }
    void* mapping = mmap(nullptr, static_cast<std::size_t>(bytes), PROT_READ,
                         MAP_PRIVATE, file_descriptor_, 0);
    if (mapping == MAP_FAILED) {
      return Status(StatusCode::kIoError,
                    "cannot map cuFile SM120 device image for validation: " +
                        std::string(std::strerror(errno)));
    }
    DeviceImageSha256 sha256;
    Status initialized = sha256.Initialize();
    if (!initialized.ok()) {
      (void)munmap(mapping, static_cast<std::size_t>(bytes));
      return initialized;
    }
    Status hashed = sha256.Update(mapping, static_cast<std::size_t>(bytes));
    if (!hashed.ok()) {
      (void)munmap(mapping, static_cast<std::size_t>(bytes));
      return hashed;
    }
    const int unmapped = munmap(mapping, static_cast<std::size_t>(bytes));
    if (unmapped != 0) {
      return Status(StatusCode::kIoError,
                    "cannot unmap validated cuFile SM120 device image");
    }
    auto digest = sha256.FinalHex();
    if (!digest.ok()) return digest.status();
    if (digest.value() != kAcceptedM08DeviceImageSha256) {
      return Status(StatusCode::kDataLoss,
                    "SM120 device image hash does not match the accepted image");
    }
    return Status::Ok();
  }

  Status Upload(std::byte* destination, std::uint64_t bytes) {
    if (destination == nullptr || handle_ == nullptr) {
      return Status(StatusCode::kInvalidArgument,
                    "invalid cuFile SM120 device-image destination");
    }
    std::uint64_t offset = 0U;
    while (offset != bytes) {
      const std::uint64_t request =
          std::min<std::uint64_t>(kCuFileRequestBytes, bytes - offset);
      const ssize_t transferred = cuFileRead(
          handle_, destination, static_cast<std::size_t>(request),
          static_cast<off_t>(offset), static_cast<off_t>(offset));
      if (transferred <= 0) {
        if (transferred == -1) {
          return Status(StatusCode::kIoError,
                        "cuFile read of SM120 device image failed: " +
                            std::string(std::strerror(errno)));
        }
        return Status(StatusCode::kIoError,
                      "cuFile read of SM120 device image failed: " +
                          std::string(CUFILE_ERRSTR(transferred)));
      }
      if (static_cast<std::uint64_t>(transferred) > request) {
        return Status(StatusCode::kDataLoss,
                      "cuFile returned an invalid SM120 device-image byte count");
      }
      offset += static_cast<std::uint64_t>(transferred);
    }
    const cudaError_t synchronized = cudaDeviceSynchronize();
    return synchronized == cudaSuccess
               ? Status::Ok()
               : CudaFailure("synchronize cuFile SM120 device image",
                             synchronized);
  }

 private:
  bool driver_open_ = false;
  int file_descriptor_ = -1;
  CUfileHandle_t handle_ = nullptr;
};
#endif

Status BindDeviceImageMetadata(
    const ModelManifest& manifest, const Gemma4Moe26BResidencyPlan& plan,
    std::uint64_t image_bytes,
    std::map<std::string, std::uint64_t, std::less<>>* offsets,
    std::vector<const TensorInfo*>* scalar_tensors) {
  if (offsets == nullptr || scalar_tensors == nullptr) {
    return Status(StatusCode::kInvalidArgument,
                  "missing SM120 device-image host metadata");
  }
  std::map<std::string, const TensorInfo*, std::less<>> tensors;
  for (const auto& tensor : manifest.tensors) tensors.emplace(tensor.name, &tensor);
  for (const auto& range : plan.upload_ranges) {
    const auto found = tensors.find(range.tensor_name);
    if (found == tensors.end() ||
        !offsets->emplace(range.tensor_name, range.destination_offset).second ||
        range.destination_offset > image_bytes ||
        range.bytes > image_bytes - range.destination_offset) {
      return Status(StatusCode::kDataLoss,
                    "invalid SM120 device-image tensor binding: " +
                        range.tensor_name);
    }
    const auto& tensor = *found->second;
    if (tensor.storage_dtype == "F32" && tensor.shape.size() == 1U &&
        tensor.shape[0] == 1U && range.bytes == sizeof(float)) {
      scalar_tensors->push_back(&tensor);
    }
  }
  return Status::Ok();
}

Status BindDeviceImageScalars(
    const std::vector<const TensorInfo*>& scalar_tensors,
    const std::map<std::string, std::uint64_t, std::less<>>& offsets,
    const std::byte* arena,
    std::map<std::string, float, std::less<>>* host_f32) {
  if (arena == nullptr || host_f32 == nullptr) {
    return Status(StatusCode::kInvalidArgument,
                  "missing uploaded SM120 device-image scalar storage");
  }
  for (const TensorInfo* tensor : scalar_tensors) {
    const auto offset = offsets.find(tensor->name);
    if (offset == offsets.end()) {
      return Status(StatusCode::kDataLoss,
                    "missing SM120 device-image scalar offset: " + tensor->name);
    }
    float value = 0.0F;
    const cudaError_t copied = cudaMemcpy(&value, arena + offset->second,
                                          sizeof(value),
                                          cudaMemcpyDeviceToHost);
    if (copied != cudaSuccess) {
      return CudaFailure("read uploaded SM120 device-image scalar", copied);
    }
    if (!std::isfinite(value) || value <= 0.0F) {
      return Status(StatusCode::kDataLoss,
                    "invalid positive F32 device-image scalar: " +
                        tensor->name);
    }
    host_f32->emplace(tensor->name, value);
  }
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
    const ModelManifest& manifest, const Gemma4Moe26BResidencyPlan& plan,
    bool verify_image_sha256) {
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
  auto image_candidate = ProbeAcceptedGemma4Moe26BDeviceImage(model_directory);
  if (!image_candidate.ok()) return image_candidate.status();
  if (image_candidate.value()) {
    if (artifact.arena_bytes_ != kAcceptedM08DeviceImageBytes) {
      return Status(StatusCode::kDataLoss,
                    "SM120 device image disagrees with the resident arena plan");
    }
    const auto image_path = Gemma4Moe26BDeviceImagePath(model_directory);
    std::vector<const TensorInfo*> scalar_tensors;
    Status metadata = BindDeviceImageMetadata(
        manifest, plan, artifact.arena_bytes_, &artifact.offsets_,
        &scalar_tensors);
    if (!metadata.ok()) return metadata;
    auto requested_io = RequestedDeviceImageIo();
    if (!requested_io.ok()) return requested_io.status();
    auto gdr = GpuSupportsPeerDirectMemory();
    if (!gdr.ok()) return gdr.status();
    bool used_cufile = false;
    bool cufile_attempted = false;
#if defined(GEM16_HAS_CUFILE)
    if (requested_io.value() == DeviceImageIo::kCuFile ||
        (requested_io.value() == DeviceImageIo::kAuto && gdr.value())) {
      cufile_attempted = true;
      CuFileImageUpload upload;
      Status status = upload.Initialize(image_path);
      if (status.ok() && verify_image_sha256) {
        status = upload.VerifyDigest(artifact.arena_bytes_);
      }
      if (status.ok()) status = upload.Upload(artifact.arena_, artifact.arena_bytes_);
      if (status.ok()) {
        used_cufile = true;
      } else if (requested_io.value() == DeviceImageIo::kCuFile) {
        return status;
      }
    }
#else
    if (requested_io.value() == DeviceImageIo::kCuFile) {
      return Status(StatusCode::kUnsupported,
                    "this build does not include NVIDIA cuFile support");
    }
#endif
    if (!used_cufile) {
      PinnedImageUpload upload;
      Status initialized = upload.Initialize();
      if (!initialized.ok()) return initialized;
      std::string digest;
      Status uploaded = upload.Upload(image_path, artifact.arena_,
                                      artifact.arena_bytes_,
                                      verify_image_sha256, &digest);
      if (!uploaded.ok()) return uploaded;
      if (verify_image_sha256 &&
          digest != kAcceptedM08DeviceImageSha256) {
        return Status(StatusCode::kDataLoss,
                      "SM120 device image hash does not match the accepted image");
      }
    }
    Status scalars = BindDeviceImageScalars(
        scalar_tensors, artifact.offsets_, artifact.arena_,
        &artifact.host_f32_);
    if (!scalars.ok()) return scalars;
    artifact.stats_.tensors = plan.upload_ranges.size();
    artifact.stats_.payload_bytes = plan.artifact_payload_bytes;
    artifact.stats_.shards = 1U;
    artifact.stats_.direct_tensors = plan.upload_ranges.size();
    artifact.stats_.host_staging_peak_bytes =
        used_cufile ? 0U : kImageUploadBuffers * kImageUploadBufferBytes;
    artifact.stats_.image_bytes = artifact.arena_bytes_;
    const std::string verification =
        verify_image_sha256 ? "_sha256" : "_structural";
    artifact.stats_.load_path =
        used_cufile
            ? ((gdr.value() ? "sm120_device_image_cufile_gpu_gdr_capable"
                            : "sm120_device_image_cufile_compat") +
               verification)
            : (cufile_attempted
                   ? "sm120_device_image_pinned_after_cufile_unavailable" +
                         verification
                   : "sm120_device_image_pinned_async" + verification);
    return artifact;
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
