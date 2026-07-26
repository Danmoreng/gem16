#include "model/safetensors.h"

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string_view>

#include "platform/mapped_file.h"
#include "util/json.h"

namespace gem16::internal {
namespace {

constexpr std::uint64_t kMaxHeaderBytes = 256U * 1024U * 1024U;
constexpr std::uint64_t kMaxIndexBytes = 256U * 1024U * 1024U;

std::uint64_t ReadLittleEndian64(const std::byte* bytes) {
  std::uint64_t result = 0;
  for (unsigned shift = 0; shift < 64U; shift += 8U) {
    result |= static_cast<std::uint64_t>(std::to_integer<unsigned char>(bytes[shift / 8U])) << shift;
  }
  return result;
}

Result<std::uint64_t> DtypeBytes(std::string_view dtype) {
  if (dtype == "BOOL" || dtype == "I8" || dtype == "U8" || dtype == "F8_E4M3" || dtype == "F8_E5M2") return 1;
  if (dtype == "I16" || dtype == "U16" || dtype == "F16" || dtype == "BF16") return 2;
  if (dtype == "I32" || dtype == "U32" || dtype == "F32") return 4;
  if (dtype == "I64" || dtype == "U64" || dtype == "F64") return 8;
  return Status(StatusCode::kUnsupported, "unsupported Safetensors dtype: " + std::string(dtype));
}

Result<std::uint64_t> TensorBytes(const std::vector<std::uint64_t>& shape, std::uint64_t element_bytes) {
  std::uint64_t elements = 1;
  for (const std::uint64_t dimension : shape) {
    if (dimension != 0 && elements > std::numeric_limits<std::uint64_t>::max() / dimension) {
      return Status(StatusCode::kDataLoss, "tensor shape product overflows uint64");
    }
    elements *= dimension;
  }
  if (element_bytes != 0 && elements > std::numeric_limits<std::uint64_t>::max() / element_bytes) {
    return Status(StatusCode::kDataLoss, "tensor byte count overflows uint64");
  }
  return elements * element_bytes;
}

std::uint64_t Alignment(std::uint64_t offset) {
  for (std::uint64_t alignment = 4096; alignment >= 2; alignment /= 2) {
    if ((offset % alignment) == 0) return alignment;
  }
  return 1;
}

Result<std::vector<StoredTensor>> LoadFile(const std::filesystem::path& path) {
  auto mapped = MappedFile::Open(path);
  if (!mapped.ok()) return mapped.status();
  if (mapped.value().size() < 8) {
    return Status(StatusCode::kDataLoss, "Safetensors file is shorter than its length prefix: " + path.string());
  }
  const std::uint64_t header_length = ReadLittleEndian64(mapped.value().data());
  if (header_length < 2 || header_length > kMaxHeaderBytes || header_length > mapped.value().size() - 8U) {
    return Status(StatusCode::kDataLoss, "invalid Safetensors header length in " + path.string());
  }
  const auto* header_data = reinterpret_cast<const char*>(mapped.value().data() + 8);
  const std::string_view header(header_data, static_cast<std::size_t>(header_length));
  auto parsed = json::Parse(header, {.max_depth = 64, .max_values = 2'000'000, .max_string_bytes = kMaxHeaderBytes});
  if (!parsed.ok()) {
    return Status(parsed.status().code(), path.string() + ": " + parsed.status().message());
  }
  if (!parsed.value().is_object()) {
    return Status(StatusCode::kDataLoss, "Safetensors header root must be an object: " + path.string());
  }

  const std::uint64_t data_base = 8U + header_length;
  const std::uint64_t payload_size = mapped.value().size() - data_base;
  std::vector<StoredTensor> tensors;
  std::vector<std::pair<std::uint64_t, std::uint64_t>> intervals;
  for (const auto& [name, metadata] : parsed.value().as_object()) {
    if (name == "__metadata__") {
      if (!metadata.is_object()) return Status(StatusCode::kDataLoss, "Safetensors __metadata__ must be an object");
      continue;
    }
    if (!metadata.is_object()) return Status(StatusCode::kDataLoss, "tensor metadata must be an object: " + name);
    const auto* dtype_value = metadata.find("dtype");
    const auto* shape_value = metadata.find("shape");
    const auto* offsets_value = metadata.find("data_offsets");
    if (dtype_value == nullptr || !dtype_value->is_string() || shape_value == nullptr || !shape_value->is_array() || offsets_value == nullptr || !offsets_value->is_array()) {
      return Status(StatusCode::kDataLoss, "incomplete tensor metadata: " + name);
    }
    std::vector<std::uint64_t> shape;
    for (const auto& dimension : shape_value->as_array()) {
      if (!dimension.is_integer() || dimension.as_integer() < 0) return Status(StatusCode::kDataLoss, "tensor shape contains a non-negative-integer violation: " + name);
      shape.push_back(static_cast<std::uint64_t>(dimension.as_integer()));
    }
    if (offsets_value->as_array().size() != 2) return Status(StatusCode::kDataLoss, "data_offsets must contain two integers: " + name);
    const auto& begin_value = offsets_value->as_array()[0];
    const auto& end_value = offsets_value->as_array()[1];
    if (!begin_value.is_integer() || !end_value.is_integer() || begin_value.as_integer() < 0 || end_value.as_integer() < begin_value.as_integer()) {
      return Status(StatusCode::kDataLoss, "invalid data_offsets: " + name);
    }
    const std::uint64_t begin = static_cast<std::uint64_t>(begin_value.as_integer());
    const std::uint64_t end = static_cast<std::uint64_t>(end_value.as_integer());
    if (end > payload_size) return Status(StatusCode::kDataLoss, "tensor extends beyond Safetensors payload: " + name);
    auto element_bytes = DtypeBytes(dtype_value->as_string());
    if (!element_bytes.ok()) return Status(element_bytes.status().code(), element_bytes.status().message() + " (tensor " + name + ")");
    auto expected_bytes = TensorBytes(shape, element_bytes.value());
    if (!expected_bytes.ok()) return Status(expected_bytes.status().code(), expected_bytes.status().message() + " (tensor " + name + ")");
    if (end - begin != expected_bytes.value()) {
      return Status(StatusCode::kDataLoss, "tensor byte length does not match dtype and shape: " + name);
    }
    const std::uint64_t absolute = data_base + begin;
    tensors.push_back({name, std::move(shape), dtype_value->as_string(), absolute, end - begin, Alignment(absolute), path.filename().string()});
    if (end > begin) intervals.emplace_back(begin, end);
  }
  std::sort(intervals.begin(), intervals.end());
  for (std::size_t index = 1; index < intervals.size(); ++index) {
    if (intervals[index].first < intervals[index - 1].second) {
      return Status(StatusCode::kDataLoss, "overlapping tensor payload ranges in " + path.string());
    }
  }
  return tensors;
}

Result<std::string> ReadSmallText(const std::filesystem::path& path, std::uint64_t limit) {
  auto mapped = MappedFile::Open(path);
  if (!mapped.ok()) return mapped.status();
  if (mapped.value().size() > limit) return Status(StatusCode::kDataLoss, "file exceeds safety limit: " + path.string());
  return std::string(reinterpret_cast<const char*>(mapped.value().data()), static_cast<std::size_t>(mapped.value().size()));
}

Result<std::map<std::string, std::string, std::less<>>> LoadIndex(const std::filesystem::path& path) {
  auto text = ReadSmallText(path, kMaxIndexBytes);
  if (!text.ok()) return text.status();
  auto parsed = json::Parse(text.value(), {.max_depth = 32, .max_values = 2'000'000, .max_string_bytes = kMaxIndexBytes});
  if (!parsed.ok()) return parsed.status();
  const auto* weight_map = parsed.value().find("weight_map");
  if (!parsed.value().is_object() || weight_map == nullptr || !weight_map->is_object()) {
    return Status(StatusCode::kDataLoss, "Safetensors index must contain an object weight_map: " + path.string());
  }
  std::map<std::string, std::string, std::less<>> result;
  for (const auto& [tensor, shard_value] : weight_map->as_object()) {
    if (!shard_value.is_string()) return Status(StatusCode::kDataLoss, "index shard name must be a string: " + tensor);
    const std::filesystem::path shard(shard_value.as_string());
    if (shard.empty() || shard.is_absolute() || shard.has_parent_path() || shard.extension() != ".safetensors") {
      return Status(StatusCode::kDataLoss, "unsafe shard path in index: " + shard.string());
    }
    result.emplace(tensor, shard.string());
  }
  return result;
}

}  // namespace

Result<std::vector<StoredTensor>> LoadSafetensorsDirectory(const std::filesystem::path& model_directory) {
  std::error_code canonical_error;
  const auto canonical_root = std::filesystem::canonical(model_directory, canonical_error);
  if (canonical_error) return Status(StatusCode::kIoError, "cannot resolve model directory: " + canonical_error.message());
  const auto index_path = model_directory / "model.safetensors.index.json";
  std::vector<std::filesystem::path> files;
  std::map<std::string, std::string, std::less<>> index;
  if (std::filesystem::is_regular_file(index_path)) {
    auto loaded_index = LoadIndex(index_path);
    if (!loaded_index.ok()) return loaded_index.status();
    index = std::move(loaded_index).value();
    std::set<std::string> unique_files;
    for (const auto& [unused, file] : index) {
      (void)unused;
      unique_files.insert(file);
    }
    for (const auto& file : unique_files) files.push_back(model_directory / file);
  } else {
    const auto single_file = model_directory / "model.safetensors";
    if (!std::filesystem::is_regular_file(single_file)) {
      return Status(StatusCode::kNotFound, "model directory has neither model.safetensors nor model.safetensors.index.json: " + model_directory.string());
    }
    files.push_back(single_file);
  }

  std::vector<StoredTensor> result;
  std::set<std::string> tensor_names;
  for (const auto& file : files) {
    if (!std::filesystem::is_regular_file(file)) return Status(StatusCode::kNotFound, "Safetensors shard is missing: " + file.string());
    const auto canonical_file = std::filesystem::canonical(file, canonical_error);
    if (canonical_error || canonical_file.parent_path() != canonical_root) {
      return Status(StatusCode::kDataLoss, "Safetensors shard resolves outside the model directory: " + file.string());
    }
    auto tensors = LoadFile(canonical_file);
    if (!tensors.ok()) return tensors.status();
    for (auto& tensor : tensors.value()) {
      if (!tensor_names.insert(tensor.name).second) return Status(StatusCode::kDataLoss, "duplicate tensor across shards: " + tensor.name);
      if (!index.empty()) {
        const auto assignment = index.find(tensor.name);
        if (assignment == index.end() || assignment->second != tensor.shard) {
          return Status(StatusCode::kDataLoss, "tensor/index shard disagreement: " + tensor.name);
        }
      }
      result.push_back(std::move(tensor));
    }
  }
  if (!index.empty() && tensor_names.size() != index.size()) {
    return Status(StatusCode::kDataLoss, "Safetensors index and shard tensor counts disagree");
  }
  std::sort(result.begin(), result.end(), [](const auto& left, const auto& right) { return left.name < right.name; });
  return result;
}

}  // namespace gem16::internal
