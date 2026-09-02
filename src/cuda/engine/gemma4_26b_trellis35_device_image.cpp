#include "cuda/engine/gemma4_26b_trellis35_device_image.h"

#include <cuda_runtime_api.h>

#if defined(GEM16_HAS_CUFILE)
#include <cufile.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <string_view>

namespace gem16::internal {
namespace {

constexpr std::uint64_t kBufferBytes = 64U * 1024U * 1024U;
constexpr std::size_t kBufferCount = 4U;
constexpr std::uint64_t kCuFileRequestBytes = 1024U * 1024U * 1024U;

enum class DeviceImageIo { kAuto, kPinned, kCuFile };

Status CudaFailure(std::string_view operation, cudaError_t error) {
  return Status(error == cudaErrorMemoryAllocation
                    ? StatusCode::kResourceExhausted
                    : StatusCode::kInternal,
                std::string(operation) + ": " + cudaGetErrorName(error) +
                    ": " + cudaGetErrorString(error));
}

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

class PinnedUpload {
 public:
  PinnedUpload() = default;
  PinnedUpload(const PinnedUpload&) = delete;
  PinnedUpload& operator=(const PinnedUpload&) = delete;
  ~PinnedUpload() {
    for (std::size_t index = 0U; index < kBufferCount; ++index) {
      if (events_[index] != nullptr) (void)cudaEventDestroy(events_[index]);
      if (buffers_[index] != nullptr) (void)cudaFreeHost(buffers_[index]);
    }
    if (stream_ != nullptr) (void)cudaStreamDestroy(stream_);
  }

  Status Initialize() {
    cudaError_t error =
        cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
    if (error != cudaSuccess) {
      return CudaFailure("create Trellis35 v2 upload stream", error);
    }
    for (std::size_t index = 0U; index < kBufferCount; ++index) {
      error = cudaHostAlloc(reinterpret_cast<void**>(&buffers_[index]),
                            static_cast<std::size_t>(kBufferBytes),
                            cudaHostAllocDefault);
      if (error != cudaSuccess) {
        return CudaFailure("allocate Trellis35 v2 pinned staging", error);
      }
      error = cudaEventCreateWithFlags(&events_[index], cudaEventDisableTiming);
      if (error != cudaSuccess) {
        return CudaFailure("create Trellis35 v2 upload event", error);
      }
    }
    return Status::Ok();
  }

  Status Upload(const std::filesystem::path& path, std::byte* destination,
                std::uint64_t bytes) {
    if (destination == nullptr || bytes == 0U ||
        bytes > static_cast<std::uint64_t>(
                    std::numeric_limits<std::streamoff>::max())) {
      return Status(StatusCode::kInvalidArgument,
                    "invalid Trellis35 v2 upload destination");
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) {
      return Status(StatusCode::kIoError,
                    "cannot open Trellis35 v2 model image: " +
                        path.string());
    }
    std::array<bool, kBufferCount> in_flight{};
    std::uint64_t offset = 0U;
    std::size_t slot = 0U;
    while (offset != bytes) {
      if (in_flight[slot]) {
        const cudaError_t waited = cudaEventSynchronize(events_[slot]);
        if (waited != cudaSuccess) {
          return CudaFailure("wait for Trellis35 v2 upload slot", waited);
        }
      }
      const std::uint64_t count =
          std::min<std::uint64_t>(kBufferBytes, bytes - offset);
      input.read(reinterpret_cast<char*>(buffers_[slot]),
                 static_cast<std::streamsize>(count));
      if (input.gcount() != static_cast<std::streamsize>(count)) {
        return Status(StatusCode::kDataLoss,
                      "short read from Trellis35 v2 model image");
      }
      cudaError_t error = cudaMemcpyAsync(
          destination + offset, buffers_[slot], static_cast<std::size_t>(count),
          cudaMemcpyHostToDevice, stream_);
      if (error != cudaSuccess) {
        return CudaFailure("upload Trellis35 v2 model image", error);
      }
      error = cudaEventRecord(events_[slot], stream_);
      if (error != cudaSuccess) {
        return CudaFailure("record Trellis35 v2 upload slot", error);
      }
      in_flight[slot] = true;
      offset += count;
      slot = (slot + 1U) % kBufferCount;
    }
    const cudaError_t synchronized = cudaStreamSynchronize(stream_);
    return synchronized == cudaSuccess
               ? Status::Ok()
               : CudaFailure("synchronize Trellis35 v2 upload", synchronized);
  }

 private:
  cudaStream_t stream_ = nullptr;
  std::array<std::byte*, kBufferCount> buffers_{};
  std::array<cudaEvent_t, kBufferCount> events_{};
};

#if defined(GEM16_HAS_CUFILE)
Status CuFileFailure(std::string_view operation, CUfileError_t error) {
  return Status(StatusCode::kIoError,
                std::string(operation) + ": " + CUFILE_ERRSTR(error.err));
}

class CuFileUpload {
 public:
  CuFileUpload() = default;
  CuFileUpload(const CuFileUpload&) = delete;
  CuFileUpload& operator=(const CuFileUpload&) = delete;
  ~CuFileUpload() {
    if (handle_ != nullptr) cuFileHandleDeregister(handle_);
    if (file_descriptor_ >= 0) (void)close(file_descriptor_);
    if (driver_open_) (void)cuFileDriverClose();
  }

  Status Initialize(const std::filesystem::path& path, std::uint64_t bytes) {
    CUfileError_t error = cuFileDriverOpen();
    if (error.err != CU_FILE_SUCCESS) {
      return CuFileFailure("open cuFile driver for Trellis35 v2", error);
    }
    driver_open_ = true;
    file_descriptor_ = open(path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
    if (file_descriptor_ < 0) {
      return Status(StatusCode::kIoError,
                    "cannot open Trellis35 v2 image for cuFile: " +
                        std::string(std::strerror(errno)));
    }
    struct stat opened {};
    struct stat named {};
    if (fstat(file_descriptor_, &opened) != 0 ||
        lstat(path.c_str(), &named) != 0 || !S_ISREG(opened.st_mode) ||
        !S_ISREG(named.st_mode) || S_ISLNK(named.st_mode) ||
        opened.st_dev != named.st_dev || opened.st_ino != named.st_ino ||
        opened.st_size < 0 ||
        static_cast<std::uint64_t>(opened.st_size) != bytes) {
      return Status(StatusCode::kDataLoss,
                    "Trellis35 v2 image changed during secure cuFile open");
    }
    CUfileDescr_t descriptor{};
    descriptor.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    descriptor.handle.fd = file_descriptor_;
    error = cuFileHandleRegister(&handle_, &descriptor);
    if (error.err != CU_FILE_SUCCESS) {
      return CuFileFailure("register Trellis35 v2 cuFile handle", error);
    }
    return Status::Ok();
  }

  Status Upload(std::byte* destination, std::uint64_t bytes) {
    if (destination == nullptr || handle_ == nullptr) {
      return Status(StatusCode::kInvalidArgument,
                    "invalid Trellis35 v2 cuFile destination");
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
                        "Trellis35 v2 cuFile read failed: " +
                            std::string(std::strerror(errno)));
        }
        return Status(StatusCode::kIoError,
                      "Trellis35 v2 cuFile read failed: " +
                          std::string(CUFILE_ERRSTR(transferred)));
      }
      if (static_cast<std::uint64_t>(transferred) > request) {
        return Status(StatusCode::kDataLoss,
                      "Trellis35 v2 cuFile returned an invalid byte count");
      }
      offset += static_cast<std::uint64_t>(transferred);
    }
    const cudaError_t synchronized = cudaDeviceSynchronize();
    return synchronized == cudaSuccess
               ? Status::Ok()
               : CudaFailure("synchronize Trellis35 v2 cuFile upload",
                             synchronized);
  }

 private:
  bool driver_open_ = false;
  int file_descriptor_ = -1;
  CUfileHandle_t handle_ = nullptr;
};
#endif

}  // namespace

Result<Trellis35DeviceImageUploadStats>
UploadGemma4Moe26BTrellis35DeviceImage(
    const std::filesystem::path& path, std::byte* destination,
    std::uint64_t bytes) {
  if (destination == nullptr || bytes == 0U) {
    return Status(StatusCode::kInvalidArgument,
                  "invalid Trellis35 v2 device-image upload");
  }
  auto requested = RequestedDeviceImageIo();
  if (!requested.ok()) return requested.status();
  auto gdr = GpuSupportsPeerDirectMemory();
  if (!gdr.ok()) return gdr.status();
  Trellis35DeviceImageUploadStats stats;
  const auto started = std::chrono::steady_clock::now();
#if defined(GEM16_HAS_CUFILE)
  if (requested.value() == DeviceImageIo::kCuFile ||
      (requested.value() == DeviceImageIo::kAuto && gdr.value())) {
    stats.cufile_attempted = true;
    CuFileUpload upload;
    Status status = upload.Initialize(path, bytes);
    if (status.ok()) status = upload.Upload(destination, bytes);
    if (status.ok()) {
      stats.used_cufile = true;
    } else if (requested.value() == DeviceImageIo::kCuFile) {
      return status;
    }
  }
#else
  if (requested.value() == DeviceImageIo::kCuFile) {
    return Status(StatusCode::kUnsupported,
                  "this build does not include NVIDIA cuFile support");
  }
#endif
  if (!stats.used_cufile) {
    PinnedUpload upload;
    Status status = upload.Initialize();
    if (!status.ok()) return status;
    status = upload.Upload(path, destination, bytes);
    if (!status.ok()) return status;
    stats.host_staging_peak_bytes = kBufferCount * kBufferBytes;
  }
  const auto finished = std::chrono::steady_clock::now();
  stats.uploaded_bytes = bytes;
  stats.upload_milliseconds =
      std::chrono::duration<double, std::milli>(finished - started).count();
  stats.load_path = stats.used_cufile
                        ? "trellis35_device_image_v2_cufile_structural"
                        : (stats.cufile_attempted
                               ? "trellis35_device_image_v2_pinned_after_"
                                 "cufile_unavailable_structural"
                               : "trellis35_device_image_v2_pinned_async_"
                                 "structural");
  return stats;
}

}  // namespace gem16::internal
