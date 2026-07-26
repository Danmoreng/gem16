#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "gem16/status.h"

namespace gem16::internal {

struct StoredTensor {
  std::string name;
  std::vector<std::uint64_t> shape;
  std::string dtype;
  std::uint64_t absolute_offset = 0;
  std::uint64_t length = 0;
  std::uint64_t alignment = 1;
  std::string shard;
};

[[nodiscard]] Result<std::vector<StoredTensor>> LoadSafetensorsDirectory(const std::filesystem::path& model_directory);

}  // namespace gem16::internal

