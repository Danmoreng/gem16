#include "platform/mapped_file.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>

#include <limits>
#include <string>
#include <system_error>

namespace gem16::internal {
namespace {

std::string WindowsError(DWORD error) {
  return std::system_category().message(static_cast<int>(error));
}

}  // namespace

Result<MappedFile> MappedFile::Open(const std::filesystem::path& path) {
  MappedFile result;
  const HANDLE file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                  nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, nullptr);
  if (file == INVALID_HANDLE_VALUE) {
    return Status(StatusCode::kIoError, "cannot open " + path.string() + ": " + WindowsError(GetLastError()));
  }
  result.file_handle_ = file;

  LARGE_INTEGER size {};
  if (GetFileSizeEx(file, &size) == 0) {
    return Status(StatusCode::kIoError, "cannot stat " + path.string() + ": " + WindowsError(GetLastError()));
  }
  if (size.QuadPart < 0) {
    return Status(StatusCode::kDataLoss, "negative file size: " + path.string());
  }
  result.size_ = static_cast<std::uint64_t>(size.QuadPart);
  if (result.size_ == 0) {
    return Status(StatusCode::kDataLoss, "empty file: " + path.string());
  }
  if (result.size_ > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return Status(StatusCode::kUnsupported, "file cannot be mapped in this address space: " + path.string());
  }

  const HANDLE mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (mapping == nullptr) {
    return Status(StatusCode::kIoError, "cannot create mapping for " + path.string() + ": " + WindowsError(GetLastError()));
  }
  result.mapping_handle_ = mapping;
  result.data_ = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
  if (result.data_ == nullptr) {
    return Status(StatusCode::kIoError, "cannot map " + path.string() + ": " + WindowsError(GetLastError()));
  }
  return result;
}

void MappedFile::Reset() noexcept {
  if (data_ != nullptr) {
    (void)UnmapViewOfFile(data_);
    data_ = nullptr;
  }
  if (mapping_handle_ != nullptr) {
    (void)CloseHandle(static_cast<HANDLE>(mapping_handle_));
    mapping_handle_ = nullptr;
  }
  if (file_handle_ != nullptr) {
    (void)CloseHandle(static_cast<HANDLE>(file_handle_));
    file_handle_ = nullptr;
  }
  size_ = 0;
}

void MappedFile::Swap(MappedFile& other) noexcept {
  std::swap(file_handle_, other.file_handle_);
  std::swap(mapping_handle_, other.mapping_handle_);
  std::swap(data_, other.data_);
  std::swap(size_, other.size_);
}

}  // namespace gem16::internal
