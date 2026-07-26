#include "platform/mapped_file.h"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace gem16::internal {

Result<MappedFile> MappedFile::Open(const std::filesystem::path& path) {
  MappedFile result;
  result.file_descriptor_ = open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (result.file_descriptor_ < 0) {
    return Status(StatusCode::kIoError, "cannot open " + path.string() + ": " + std::strerror(errno));
  }

  struct stat metadata {};
  if (fstat(result.file_descriptor_, &metadata) != 0) {
    return Status(StatusCode::kIoError, "cannot stat " + path.string() + ": " + std::strerror(errno));
  }
  if (metadata.st_size < 0) {
    return Status(StatusCode::kDataLoss, "negative file size: " + path.string());
  }
  result.size_ = static_cast<std::uint64_t>(metadata.st_size);
  if (result.size_ == 0) {
    return Status(StatusCode::kDataLoss, "empty file: " + path.string());
  }
  if (result.size_ > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return Status(StatusCode::kUnsupported, "file cannot be mapped in this address space: " + path.string());
  }

  result.data_ = mmap(nullptr, static_cast<std::size_t>(result.size_), PROT_READ, MAP_PRIVATE, result.file_descriptor_, 0);
  if (result.data_ == MAP_FAILED) {
    result.data_ = nullptr;
    return Status(StatusCode::kIoError, "cannot mmap " + path.string() + ": " + std::strerror(errno));
  }
  return result;
}

void MappedFile::Reset() noexcept {
  if (data_ != nullptr) {
    (void)munmap(data_, static_cast<std::size_t>(size_));
    data_ = nullptr;
  }
  if (file_descriptor_ >= 0) {
    (void)close(file_descriptor_);
    file_descriptor_ = -1;
  }
  size_ = 0;
}

void MappedFile::Swap(MappedFile& other) noexcept {
  std::swap(file_descriptor_, other.file_descriptor_);
  std::swap(data_, other.data_);
  std::swap(size_, other.size_);
}

}  // namespace gem16::internal
