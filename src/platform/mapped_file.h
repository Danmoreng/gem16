#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <utility>

#include "gem16/status.h"

namespace gem16::internal {

class MappedFile {
 public:
  MappedFile() = default;
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;
  MappedFile(MappedFile&& other) noexcept { Swap(other); }
  MappedFile& operator=(MappedFile&& other) noexcept {
    if (this != &other) {
      Reset();
      Swap(other);
    }
    return *this;
  }
  ~MappedFile() { Reset(); }

  [[nodiscard]] static Result<MappedFile> Open(const std::filesystem::path& path);

  [[nodiscard]] const std::byte* data() const { return static_cast<const std::byte*>(data_); }
  [[nodiscard]] std::uint64_t size() const { return size_; }

 private:
  void Reset() noexcept;
  void Swap(MappedFile& other) noexcept;

#if defined(_WIN32)
  void* file_handle_ = nullptr;
  void* mapping_handle_ = nullptr;
#else
  int file_descriptor_ = -1;
#endif
  void* data_ = nullptr;
  std::uint64_t size_ = 0;
};

}  // namespace gem16::internal
