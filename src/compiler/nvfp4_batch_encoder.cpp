#include "compiler/nvfp4_batch_encoder.h"

#include "compiler/sha256.h"
#include "util/json.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cerrno>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <set>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace gem16::compiler {
namespace {
using json::Value;

constexpr std::uint64_t kMaxJobBytes = 64U * 1024U * 1024U;
constexpr std::uint64_t kMaxOperations = 1000U;
constexpr std::uint64_t kMaxDimension = 1U << 20U;
constexpr std::uint64_t kMaxThreads = 64U;
constexpr std::uint64_t kMaxSourceBytes = 1024U * 1024U * 1024U;
constexpr std::uint64_t kM07TiedHeadSourceBytes = 1'476'395'008U;
constexpr std::array<float, 8> kE2M1 = {0.F, .5F, 1.F, 1.5F, 2.F, 3.F, 4.F, 6.F};

#ifndef GEM16_NATIVE_COMPILER_ID
#define GEM16_NATIVE_COMPILER_ID "unknown"
#define GEM16_NATIVE_COMPILER_VERSION "unknown"
#define GEM16_NATIVE_BUILD_TYPE "unknown"
#define GEM16_NATIVE_CXX_STANDARD "20"
#define GEM16_NATIVE_SYSTEM "unknown"
#define GEM16_NATIVE_PROCESSOR "unknown"
#endif

Status Invalid(std::string message) { return Status(StatusCode::kInvalidArgument, std::move(message)); }
Status Io(std::string message) { return Status(StatusCode::kIoError, std::move(message)); }
Status Data(std::string message) { return Status(StatusCode::kDataLoss, std::move(message)); }
Status Resource(std::string message) { return Status(StatusCode::kResourceExhausted, std::move(message)); }
[[maybe_unused]] Status Unsupported(std::string message) {
  return Status(StatusCode::kUnsupported, std::move(message));
}
Status Internal(std::string message) { return Status(StatusCode::kInternal, std::move(message)); }

Value Integer(std::uint64_t value) { return Value(static_cast<std::int64_t>(value)); }
Value Decimal(double value) { return Value(value); }
Value String(std::string value) { return Value(std::move(value)); }

const Value* Required(const Value::Object& object, std::string_view key) {
  const auto it = object.find(std::string(key));
  return it == object.end() ? nullptr : &it->second;
}

bool ExactKeys(const Value::Object& object,
               std::initializer_list<std::string_view> keys) {
  if (object.size() != keys.size()) return false;
  for (const auto key : keys) {
    if (!object.contains(std::string(key))) return false;
  }
  return true;
}

Result<std::uint64_t> Unsigned(const Value* value, std::string_view field) {
  if (value == nullptr || !value->is_integer() || value->as_integer() < 0) {
    return Invalid(std::string(field) + " must be a nonnegative integer");
  }
  return static_cast<std::uint64_t>(value->as_integer());
}

Result<std::string> Text(const Value* value, std::string_view field) {
  if (value == nullptr || !value->is_string() || value->as_string().empty() ||
      value->as_string().find('\0') != std::string::npos) {
    return Invalid(std::string(field) + " must be a nonempty string");
  }
  return value->as_string();
}

Result<std::uint64_t> Product(std::uint64_t lhs, std::uint64_t rhs,
                              std::string_view field) {
  if (rhs != 0 && lhs > std::numeric_limits<std::uint64_t>::max() / rhs) {
    return Invalid(std::string(field) + " overflows");
  }
  return lhs * rhs;
}

bool HexSha256(std::string_view value) {
  return value.size() == 64 &&
         std::all_of(value.begin(), value.end(), [](char c) {
           return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
         });
}

#ifndef _WIN32
struct FileIdentity {
  dev_t device = 0;
  ino_t inode = 0;
};

bool SameFile(FileIdentity lhs, FileIdentity rhs) {
  return lhs.device == rhs.device && lhs.inode == rhs.inode;
}

struct FileHandle {
  int fd = -1;
  FileIdentity identity{};
  std::string path;
  ~FileHandle() {
    if (fd >= 0) close(fd);
  }
  FileHandle() = default;
  FileHandle(const FileHandle&) = delete;
  FileHandle& operator=(const FileHandle&) = delete;
  FileHandle(FileHandle&& other) noexcept
      : fd(other.fd), identity(other.identity), path(std::move(other.path)) {
    other.fd = -1;
  }
  FileHandle& operator=(FileHandle&& other) noexcept {
    if (this != &other) {
      if (fd >= 0) close(fd);
      fd = other.fd;
      identity = other.identity;
      path = std::move(other.path);
      other.fd = -1;
    }
    return *this;
  }
};

Result<FileHandle> OpenRegular(const std::string& path, int flags,
                               std::uint64_t required_bytes) {
  FileHandle handle;
  handle.path = path;
  handle.fd = open(path.c_str(), flags | O_CLOEXEC | O_NOFOLLOW);
  if (handle.fd < 0) return Io("cannot open " + path + ": " + std::strerror(errno));
  struct stat statbuf{};
  if (fstat(handle.fd, &statbuf) != 0) {
    return Io("cannot inspect " + path + ": " + std::strerror(errno));
  }
  if (!S_ISREG(statbuf.st_mode)) return Invalid("not a regular file: " + path);
  if (required_bytes > static_cast<std::uint64_t>(statbuf.st_size)) {
    return Io("file is shorter than requested range: " + path);
  }
  handle.identity = {statbuf.st_dev, statbuf.st_ino};
  return handle;
}

Status CheckIdentity(const FileHandle& handle) {
  struct stat statbuf{};
  if (fstat(handle.fd, &statbuf) != 0) {
    return Io("cannot recheck " + handle.path + ": " + std::strerror(errno));
  }
  if (!S_ISREG(statbuf.st_mode) ||
      !SameFile(handle.identity, {statbuf.st_dev, statbuf.st_ino})) {
    return Data("file identity changed: " + handle.path);
  }
  return Status::Ok();
}

Status ReadExact(int fd, std::uint64_t offset, void* destination, std::size_t bytes) {
  std::size_t completed = 0;
  while (completed < bytes) {
    const auto position = offset + completed;
    if (position > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
      return Data("file offset exceeds host limit");
    }
    const ssize_t count = pread(fd, static_cast<char*>(destination) + completed,
                                bytes - completed, static_cast<off_t>(position));
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) return Io("short read");
    completed += static_cast<std::size_t>(count);
  }
  return Status::Ok();
}

Status WriteExact(int fd, std::uint64_t offset, const void* source, std::size_t bytes) {
  std::size_t completed = 0;
  while (completed < bytes) {
    const auto position = offset + completed;
    if (position > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
      return Data("file offset exceeds host limit");
    }
    const ssize_t count = pwrite(fd, static_cast<const char*>(source) + completed,
                                 bytes - completed, static_cast<off_t>(position));
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) return Io("short write");
    completed += static_cast<std::size_t>(count);
  }
  return Status::Ok();
}

Status Fsync(const FileHandle& handle) {
  if (fsync(handle.fd) != 0) return Io("cannot fsync " + handle.path);
  return Status::Ok();
}
#endif

std::uint16_t LoadLe16(const std::uint8_t* bytes) {
  return static_cast<std::uint16_t>(bytes[0]) |
         static_cast<std::uint16_t>(bytes[1] << 8U);
}

void StoreLe32(float value, std::uint8_t* bytes) {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  bytes[0] = static_cast<std::uint8_t>(bits);
  bytes[1] = static_cast<std::uint8_t>(bits >> 8U);
  bytes[2] = static_cast<std::uint8_t>(bits >> 16U);
  bytes[3] = static_cast<std::uint8_t>(bits >> 24U);
}

float DecodeBf16(std::uint16_t value) {
  return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U);
}

std::uint16_t EncodeBf16Rne(float value) {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  std::uint32_t upper = bits >> 16U;
  const std::uint32_t lower = bits & 0xffffU;
  if (lower > 0x8000U || (lower == 0x8000U && (upper & 1U) != 0)) ++upper;
  return static_cast<std::uint16_t>(upper);
}

float DecodeE2M1(std::uint8_t code) {
  const float value = kE2M1[code & 7U];
  return (code & 8U) != 0 ? -value : value;
}

std::uint8_t EncodeE2M1(float value) {
  const float magnitude = std::fabs(value);
  // NaN has no representable NVFP4 value; use canonical positive zero for
  // this noexcept diagnostic helper. Infinities and finite outliers follow
  // the finite-saturation contract and map to the largest signed code.
  if (std::isnan(magnitude)) return 0;
  if (magnitude >= kE2M1.back() || std::isinf(magnitude)) {
    return static_cast<std::uint8_t>(7U | (std::signbit(value) ? 8U : 0U));
  }
  std::uint8_t low = 0;
  std::uint8_t high = 7;
  while (low < high) {
    const std::uint8_t middle = static_cast<std::uint8_t>((low + high) / 2U);
    if (kE2M1[middle] < magnitude) low = static_cast<std::uint8_t>(middle + 1U);
    else high = middle;
  }
  std::uint8_t best = low;
  if (low != 0) {
    const std::uint8_t prior = static_cast<std::uint8_t>(low - 1U);
    const float upper_error = std::fabs(magnitude - kE2M1[low]);
    const float lower_error = std::fabs(magnitude - kE2M1[prior]);
    if (lower_error < upper_error ||
        (lower_error == upper_error && (prior & 1U) == 0U)) best = prior;
  }
  return static_cast<std::uint8_t>(best | (std::signbit(value) ? 8U : 0U));
}

float DecodeE4M3(std::uint8_t code) {
  const std::uint8_t magnitude_code = code & 0x7fU;
  if (magnitude_code == 0x7fU) return std::numeric_limits<float>::quiet_NaN();
  const int exponent = (magnitude_code >> 3U) & 0xf;
  const int mantissa = magnitude_code & 7U;
  const float value = exponent == 0
                          ? std::ldexp(static_cast<float>(mantissa), -9)
                          : std::ldexp(1.F + static_cast<float>(mantissa) / 8.F,
                                       exponent - 7);
  return (code & 0x80U) != 0 ? -value : value;
}

std::uint8_t EncodeE4M3(float value) {
  const float magnitude = std::fabs(value);
  if (magnitude >= 448.F) return static_cast<std::uint8_t>(0x7eU |
                                                            (std::signbit(value) ? 0x80U : 0));
  if (std::isnan(magnitude)) return 0;
  std::uint8_t low = 0;
  std::uint8_t high = 0x7eU;
  while (low < high) {
    const std::uint8_t middle = static_cast<std::uint8_t>(
        low + (static_cast<unsigned>(high - low) / 2U));
    if (DecodeE4M3(middle) < magnitude) low = static_cast<std::uint8_t>(middle + 1U);
    else high = middle;
  }
  std::uint8_t best = low;
  if (low != 0) {
    const std::uint8_t prior = static_cast<std::uint8_t>(low - 1U);
    const float upper_error = std::fabs(magnitude - DecodeE4M3(low));
    const float lower_error = std::fabs(magnitude - DecodeE4M3(prior));
    if (lower_error < upper_error ||
        (lower_error == upper_error && (prior & 1U) == 0U)) best = prior;
  }
  return static_cast<std::uint8_t>(best | (std::signbit(value) ? 0x80U : 0));
}

void PutNibble(std::uint8_t* destination, std::size_t index, std::uint8_t code) {
  const std::uint8_t shift = static_cast<std::uint8_t>((index & 1U) * 4U);
  *destination = static_cast<std::uint8_t>(*destination | ((code & 0xfU) << shift));
}

struct Output {
  std::string component;
  std::string name;
  std::string path;
  std::uint64_t offset = 0;
  std::uint64_t bytes = 0;
};
struct Operation {
  std::string operation_id;
  std::string source_name;
  std::string source_path;
  std::string source_hash;
  std::uint64_t source_offset = 0;
  std::uint64_t source_bytes = 0;
  std::vector<std::uint64_t> shape;
  std::uint64_t rows = 0;
  std::uint64_t columns = 0;
  std::string source_dtype;
  std::string role;
  std::string axis_transformation;
  std::string disk_layout;
  std::string runtime_layout;
  Output packed;
  Output local_scale;
  Output weight_global;
  Output input_global;
};
struct Job {
  std::uint64_t schema_version = 0;
  std::string protocol;
  std::string profile;
  std::string scope;
  std::string contract_id;
  std::uint64_t contract_version = 0;
  std::uint64_t threads = 1;
  std::vector<Operation> operations;
};

Result<Output> ParseOutput(const Value* value, std::string_view expected_component,
                           std::string_view field) {
  if (value == nullptr || !value->is_object() ||
      !ExactKeys(value->as_object(), {"component", "name", "path", "offset", "bytes"})) {
    return Invalid(std::string(field) + " output schema is invalid");
  }
  auto component = Text(Required(value->as_object(), "component"), "output.component");
  auto name = Text(Required(value->as_object(), "name"), "output.name");
  auto path = Text(Required(value->as_object(), "path"), "output.path");
  auto offset = Unsigned(Required(value->as_object(), "offset"), "output.offset");
  auto bytes = Unsigned(Required(value->as_object(), "bytes"), "output.bytes");
  if (!component.ok() || !name.ok() || !path.ok() || !offset.ok() || !bytes.ok()) {
    return Invalid(std::string(field) + " output value is invalid");
  }
  if (component.value() != expected_component) {
    return Invalid(std::string(field) + " component mismatch");
  }
  const std::string suffix = expected_component == "packed"
                                 ? ".weight_packed"
                                 : expected_component == "local_scale"
                                       ? ".weight_scale"
                                       : expected_component == "weight_global"
                                             ? ".weight_global_scale"
                                             : ".input_global_scale";
  if (name.value().size() < suffix.size() ||
      name.value().compare(name.value().size() - suffix.size(), suffix.size(), suffix) != 0) {
    return Invalid(std::string(field) + " name has invalid component suffix");
  }
  return Output{component.value(), name.value(), path.value(), offset.value(), bytes.value()};
}

Result<Operation> ParseOperation(const Value& value, std::set<std::string>& names,
                                 std::string_view scope, std::string_view profile) {
  if (!value.is_object() ||
      !ExactKeys(value.as_object(), {"operation_id", "source_name", "source_path", "source_sha256",
                                     "source_offset", "source_bytes", "source_dtype", "logical_shape",
                                     "rows", "columns", "role", "axis_transformation", "disk_layout",
                                     "runtime_layout", "packed", "local_scale", "weight_global",
                                     "input_global"})) {
    return Invalid("operation key set is invalid");
  }
  auto operation_id = Text(Required(value.as_object(), "operation_id"), "operation_id");
  auto source_name = Text(Required(value.as_object(), "source_name"), "source_name");
  auto source_path = Text(Required(value.as_object(), "source_path"), "source_path");
  auto source_hash = Text(Required(value.as_object(), "source_sha256"), "source_sha256");
  auto source_offset = Unsigned(Required(value.as_object(), "source_offset"), "source_offset");
  auto source_bytes = Unsigned(Required(value.as_object(), "source_bytes"), "source_bytes");
  auto source_dtype = Text(Required(value.as_object(), "source_dtype"), "source_dtype");
  auto role = Text(Required(value.as_object(), "role"), "role");
  auto axis = Text(Required(value.as_object(), "axis_transformation"), "axis_transformation");
  auto disk = Text(Required(value.as_object(), "disk_layout"), "disk_layout");
  auto runtime = Text(Required(value.as_object(), "runtime_layout"), "runtime_layout");
  auto rows = Unsigned(Required(value.as_object(), "rows"), "rows");
  auto columns = Unsigned(Required(value.as_object(), "columns"), "columns");
  if (!operation_id.ok() || !source_name.ok() || !source_path.ok() || !source_hash.ok() ||
      !source_offset.ok() || !source_bytes.ok() || !source_dtype.ok() || !role.ok() ||
      !axis.ok() || !disk.ok() || !runtime.ok() || !rows.ok() || !columns.ok() ||
      !HexSha256(source_hash.value()) || !std::filesystem::path(source_path.value()).is_absolute()) {
    return Invalid("operation scalar is invalid");
  }
  if (!names.insert(source_name.value()).second) return Invalid("duplicate source_name");
  const std::string source_stem =
      source_name.value().ends_with(".weight")
          ? source_name.value().substr(0, source_name.value().size() - 7U)
          : source_name.value();
  const bool tied_head =
      profile == "nvfp4-tied-head-partial-v1" ||
      (profile == "sm120-text-hybrid-v1" &&
       source_name.value() == "model.language_model.embed_tokens.weight");
  const bool complete = scope == "complete";
  const std::string expected_operation_prefix =
      (scope == "full" || complete) ? "nvfp4-experts:" : "fixture:";
  if (tied_head) {
    if ((scope != "tied_head" && !complete) ||
        operation_id.value() != "nvfp4-head:model.language_model.embed_tokens" ||
        source_name.value() != "model.language_model.embed_tokens.weight") {
      return Invalid("M07 tied-head operation identity is not canonical");
    }
  } else if (operation_id.value() != expected_operation_prefix + source_stem &&
             !(scope == "fixture" && operation_id.value() == "op:" + source_stem)) {
    return Invalid("operation_id does not match canonical source stem");
  }
  const bool routed_down = role.value() == "routed_expert_down";
  const bool routed_gate_up = role.value() == "routed_expert_gate_up";
  const bool shared = role.value() == "shared_mlp_down" ||
                      role.value() == "shared_mlp_gate" || role.value() == "shared_mlp_up";
  const bool legal_axis = (routed_down && axis.value() == "expert,output,input") ||
                          (routed_gate_up && axis.value() == "expert,gate_then_up,input") ||
                          (shared && axis.value() == "output,input") ||
                          (tied_head && role.value() == "tied_embedding_and_output" && axis.value() == "vocabulary,hidden") ||
                          (scope == "fixture" && axis.value() == "identity");
  const bool legal_runtime = (routed_down || routed_gate_up) &&
                                 runtime.value() == "expert_major_sm120_row8_k64";
  const bool legal_shared_runtime = shared &&
                                    runtime.value() == "sm120_row8_k64";
  const bool legal_tied_runtime = tied_head && runtime.value() == "sm120_row8_k64";
  if (source_dtype.value() != "BF16" || !legal_axis ||
      disk.value() != "canonical_row_major_low_nibble_first" ||
      (!legal_runtime && !legal_shared_runtime && !legal_tied_runtime && scope != "fixture")) {
    return Invalid("unsupported NVFP4 operation role/layout");
  }
  const auto* shape_value = Required(value.as_object(), "logical_shape");
  if (shape_value == nullptr || !shape_value->is_array() ||
      (shape_value->as_array().size() != 2 && shape_value->as_array().size() != 3)) {
    return Invalid("logical_shape must have rank 2 or 3");
  }
  std::vector<std::uint64_t> shape;
  std::uint64_t elements = 1;
  for (const auto& dimension : shape_value->as_array()) {
    auto parsed = Unsigned(&dimension, "logical_shape dimension");
    if (!parsed.ok() || parsed.value() == 0 || parsed.value() > kMaxDimension) {
      return Invalid("logical_shape dimension is invalid");
    }
    auto product = Product(elements, parsed.value(), "logical_shape");
    if (!product.ok()) return product.status();
    elements = product.value();
    shape.push_back(parsed.value());
  }
  if (columns.value() != shape.back() || rows.value() != elements / columns.value() ||
      columns.value() == 0 || columns.value() % 16 != 0) {
    return Invalid("rows/columns do not match logical_shape");
  }
  auto element_count = Product(rows.value(), columns.value(), "source shape");
  if (!element_count.ok()) return element_count.status();
  auto source_size = Product(element_count.value(), 2, "source bytes");
  const bool canonical_m07_source =
      tied_head && (scope == "tied_head" || complete) &&
      source_name.value() == "model.language_model.embed_tokens.weight" &&
      role.value() == "tied_embedding_and_output" &&
      shape == std::vector<std::uint64_t>{262144, 2816} &&
      rows.value() == 262144 && columns.value() == 2816 &&
      source_size.ok() && source_size.value() == kM07TiedHeadSourceBytes;
  const std::uint64_t source_limit = canonical_m07_source
                                         ? kM07TiedHeadSourceBytes
                                         : kMaxSourceBytes;
  if (!source_size.ok() || source_bytes.value() != source_size.value() ||
      source_bytes.value() > source_limit) {
    return Invalid("source byte count does not match shape or profile limit");
  }
  auto packed = ParseOutput(Required(value.as_object(), "packed"), "packed", "packed");
  auto local_scale = ParseOutput(Required(value.as_object(), "local_scale"), "local_scale",
                                "local_scale");
  auto weight_global = ParseOutput(Required(value.as_object(), "weight_global"), "weight_global",
                                   "weight_global");
  auto input_global = ParseOutput(Required(value.as_object(), "input_global"), "input_global",
                                  "input_global");
  if (!packed.ok() || !local_scale.ok() || !weight_global.ok() || !input_global.ok() ||
      !std::filesystem::path(packed.value().path).is_absolute() ||
      !std::filesystem::path(local_scale.value().path).is_absolute() ||
      !std::filesystem::path(weight_global.value().path).is_absolute() ||
      !std::filesystem::path(input_global.value().path).is_absolute()) {
    return Invalid("operation output is invalid");
  }
  if (packed.value().bytes != elements / 2 ||
      local_scale.value().bytes != elements / 16 || weight_global.value().bytes != 4 ||
      input_global.value().bytes != 4) {
    return Invalid("output byte count does not match shape");
  }
  const std::array<const Output*, 4> outputs = {
      &packed.value(), &local_scale.value(), &weight_global.value(), &input_global.value()};
  const std::array<std::string_view, 4> suffixes = {
      ".weight_packed", ".weight_scale", ".weight_global_scale", ".input_global_scale"};
  for (std::size_t index = 0; index < outputs.size(); ++index) {
    if (outputs[index]->name != source_stem + std::string(suffixes[index])) {
      return Invalid("output name does not match canonical source stem");
    }
  }
  if (tied_head) {
    if (shape != std::vector<std::uint64_t>{262144, 2816} || rows.value() != 262144 ||
        columns.value() != 2816 || role.value() != "tied_embedding_and_output" ||
        axis.value() != "vocabulary,hidden" || runtime.value() != "sm120_row8_k64") {
      return Invalid("M07 tied-head name/role/shape/layout mismatch");
    }
  }
  if ((scope == "full" || complete) && !tied_head) {
    const std::string prefix = "model.language_model.layers.";
    if (source_name.value().find(prefix) != 0) {
      return Invalid("full operation is not canonical");
    }
    const auto layer_start = prefix.size();
    const auto layer_end = source_name.value().find('.', layer_start);
    if (layer_end == std::string::npos || layer_end == layer_start) {
      return Invalid("full operation layer name is invalid");
    }
    try {
      const auto layer = std::stoul(source_name.value().substr(layer_start, layer_end - layer_start));
      if (layer >= 30) return Invalid("full operation layer is outside Gemma 4 26B");
    } catch (const std::exception&) {
      return Invalid("full operation layer is invalid");
    }
    const std::string suffix = source_name.value().substr(layer_end);
    const bool expected =
        (suffix == ".experts.down_proj" && routed_down && shape == std::vector<std::uint64_t>{128, 2816, 704}) ||
        (suffix == ".experts.gate_up_proj" && routed_gate_up && shape == std::vector<std::uint64_t>{128, 1408, 2816}) ||
        (suffix == ".mlp.down_proj.weight" && role.value() == "shared_mlp_down" && shape == std::vector<std::uint64_t>{2816, 2112}) ||
        (suffix == ".mlp.gate_proj.weight" && role.value() == "shared_mlp_gate" && shape == std::vector<std::uint64_t>{2112, 2816}) ||
        (suffix == ".mlp.up_proj.weight" && role.value() == "shared_mlp_up" && shape == std::vector<std::uint64_t>{2112, 2816});
    if (!expected) return Invalid("full operation name/role/shape mismatch");
  }
  return Operation{operation_id.value(), source_name.value(), source_path.value(), source_hash.value(),
                   source_offset.value(), source_bytes.value(), std::move(shape), rows.value(),
                   columns.value(), source_dtype.value(), role.value(), axis.value(), disk.value(),
                   runtime.value(), packed.value(), local_scale.value(), weight_global.value(),
                   input_global.value()};
}

Result<Job> ParseJob(const Value& root) {
  if (!root.is_object() ||
      !ExactKeys(root.as_object(), {"schema_version", "protocol", "artifact_profile", "scope",
                                     "contract_id", "contract_version", "threads", "operations"})) {
    return Invalid("job key set is invalid");
  }
  auto schema = Unsigned(Required(root.as_object(), "schema_version"), "schema_version");
  auto protocol = Text(Required(root.as_object(), "protocol"), "protocol");
  auto profile = Text(Required(root.as_object(), "artifact_profile"), "artifact_profile");
  auto scope = Text(Required(root.as_object(), "scope"), "scope");
  auto contract = Text(Required(root.as_object(), "contract_id"), "contract_id");
  auto version = Unsigned(Required(root.as_object(), "contract_version"), "contract_version");
  auto threads = Unsigned(Required(root.as_object(), "threads"), "threads");
  if (!schema.ok() || !protocol.ok() || !profile.ok() || !scope.ok() || !contract.ok() ||
      !version.ok() || !threads.ok() || schema.value() != 1 ||
      protocol.value() != "gem16-nvfp4-direct-v1" ||
      (profile.value() != "nvfp4-experts-partial-v1" &&
       profile.value() != "nvfp4-tied-head-partial-v1" &&
       profile.value() != "sm120-text-hybrid-v1") ||
      ((profile.value() == "nvfp4-experts-partial-v1" && scope.value() != "fixture" && scope.value() != "full") ||
       (profile.value() == "nvfp4-tied-head-partial-v1" && scope.value() != "tied_head") ||
       (profile.value() == "sm120-text-hybrid-v1" && scope.value() != "complete")) ||
      contract.value() != "gem16.nvfp4_bf16_group16" || version.value() != 1 ||
      threads.value() < 1 || threads.value() > kMaxThreads) {
    return Invalid("unsupported NVFP4 job identity");
  }
  const auto* operations = Required(root.as_object(), "operations");
  if (operations == nullptr || !operations->is_array() || operations->as_array().empty() ||
      operations->as_array().size() > kMaxOperations) {
    return Invalid("operations must be a nonempty bounded array");
  }
  if (scope.value() == "full" && operations->as_array().size() != 150) {
    return Invalid("full scope requires exactly 150 expert operations");
  }
  if (scope.value() == "tied_head" && operations->as_array().size() != 1) {
    return Invalid("tied_head scope requires exactly one operation");
  }
  if (scope.value() == "complete" && operations->as_array().size() != 151) {
    return Invalid("complete scope requires 150 expert operations and one tied head");
  }
  Job job{schema.value(), protocol.value(), profile.value(), scope.value(), contract.value(),
          version.value(), threads.value(), {}};
  std::set<std::string> names;
  for (const auto& operation : operations->as_array()) {
    auto parsed = ParseOperation(operation, names, scope.value(), profile.value());
    if (!parsed.ok()) return parsed.status();
    job.operations.push_back(std::move(parsed.value()));
  }
  if (scope.value() == "tied_head" &&
      names != std::set<std::string>{"model.language_model.embed_tokens.weight"}) {
    return Invalid("tied_head operation inventory is not canonical");
  }
  if (scope.value() == "full") {
    std::set<std::string> expected;
    for (std::size_t layer = 0; layer < 30; ++layer) {
      const std::string prefix = "model.language_model.layers." + std::to_string(layer);
      expected.insert(prefix + ".experts.down_proj");
      expected.insert(prefix + ".experts.gate_up_proj");
      expected.insert(prefix + ".mlp.down_proj.weight");
      expected.insert(prefix + ".mlp.gate_proj.weight");
      expected.insert(prefix + ".mlp.up_proj.weight");
    }
    if (names != expected) return Invalid("full scope operation inventory is not canonical");
  }
  return job;
}

#ifndef _WIN32
struct Range {
  FileIdentity identity{};
  std::uint64_t begin = 0;
  std::uint64_t end = 0;
  std::string description;
};

bool Overlap(const Range& lhs, const Range& rhs) {
  return SameFile(lhs.identity, rhs.identity) && lhs.begin < rhs.end && rhs.begin < lhs.end;
}

struct Preflight {
  std::shared_ptr<FileHandle> source;
  std::array<std::shared_ptr<FileHandle>, 4> outputs;
};

Status PreflightFiles(const Job& job, const std::filesystem::path& job_path,
                      const std::filesystem::path& telemetry_path,
                      std::vector<Preflight>* files) {
  auto job_handle = OpenRegular(job_path.string(), O_RDONLY, 1);
  if (!job_handle.ok()) return job_handle.status();
  auto telemetry = telemetry_path.string();
  struct stat telemetry_stat{};
  if (lstat(telemetry.c_str(), &telemetry_stat) == 0) {
    return Invalid("telemetry output already exists");
  }
  if (errno != ENOENT) return Io("cannot inspect telemetry output");

  std::vector<Range> ranges;
  ranges.reserve(job.operations.size() * 5);
  files->reserve(job.operations.size());
  for (const auto& operation : job.operations) {
    if (operation.source_offset > std::numeric_limits<std::uint64_t>::max() - operation.source_bytes) {
      return Invalid("source range overflows");
    }
    auto source = OpenRegular(operation.source_path, O_RDONLY,
                              operation.source_offset + operation.source_bytes);
    if (!source.ok()) return source.status();
    if (SameFile(source.value().identity, job_handle.value().identity)) {
      return Invalid("source aliases job descriptor");
    }
    auto source_handle = std::make_shared<FileHandle>(std::move(source.value()));
    for (const auto& prior : *files) {
      if (SameFile(source_handle->identity, prior.source->identity)) {
        source_handle = prior.source;
        break;
      }
      for (const auto& prior_output : prior.outputs) {
        if (SameFile(source_handle->identity, prior_output->identity)) {
          return Invalid("source aliases output by inode");
        }
      }
    }
    Preflight entry;
    entry.source = std::move(source_handle);
    ranges.push_back({entry.source->identity, operation.source_offset,
                      operation.source_offset + operation.source_bytes, operation.source_name});
    const std::array<const Output*, 4> outputs = {&operation.packed, &operation.local_scale,
                                                   &operation.weight_global, &operation.input_global};
    for (std::size_t i = 0; i < outputs.size(); ++i) {
      const Output& output = *outputs[i];
      if (output.offset > std::numeric_limits<std::uint64_t>::max() - output.bytes) {
        return Invalid("output range overflows");
      }
      auto handle = OpenRegular(output.path, O_RDWR, output.offset + output.bytes);
      if (!handle.ok()) return handle.status();
      if (SameFile(handle.value().identity, job_handle.value().identity)) {
        return Invalid("output aliases job descriptor");
      }
      if (SameFile(handle.value().identity, entry.source->identity)) {
        return Invalid("output aliases source by inode");
      }
      for (const auto& prior : *files) {
        if (SameFile(handle.value().identity, prior.source->identity)) {
          return Invalid("output aliases source by inode");
        }
      }
      for (const auto& range : ranges) {
        if (Overlap({handle.value().identity, output.offset, output.offset + output.bytes, output.name},
                    range)) {
          return Invalid("source/output ranges overlap");
        }
      }
      auto output_handle = std::make_shared<FileHandle>(std::move(handle.value()));
      for (const auto& prior : *files) {
        for (const auto& prior_output : prior.outputs) {
          if (SameFile(output_handle->identity, prior_output->identity)) {
            output_handle = prior_output;
            break;
          }
        }
      }
      entry.outputs[i] = std::move(output_handle);
      ranges.push_back({entry.outputs[i]->identity, output.offset, output.offset + output.bytes,
                        output.name});
    }
    files->push_back(std::move(entry));
  }
  for (std::size_t i = 0; i < ranges.size(); ++i) {
    for (std::size_t j = i + 1; j < ranges.size(); ++j) {
      if (Overlap(ranges[i], ranges[j])) return Invalid("overlapping source/output ranges");
    }
  }
  return Status::Ok();
}
#endif

#ifndef _WIN32
struct SourceAnalysis {
  std::string source_hash;
  float tensor_amax = 0.F;
  float source_min = 0.F;
  float source_max = 0.F;
  double source_sum_squares = 0.0;
};

Status AnalyzeSources(const Job& job, const std::vector<Preflight>& files,
                      std::vector<SourceAnalysis>* analyses) {
  analyses->resize(job.operations.size());
  const std::size_t workers = std::min<std::size_t>(job.threads, job.operations.size());
  std::atomic<std::size_t> next{0};
  std::atomic<bool> cancelled{false};
  std::mutex error_mutex;
  Status first_error = Status::Ok();
  std::vector<std::thread> threads;
  try {
    threads.reserve(workers);
    for (std::size_t worker = 0; worker < workers; ++worker) {
      threads.emplace_back([&, worker] {
      (void)worker;
      try {
        while (!cancelled.load(std::memory_order_acquire)) {
          const std::size_t index = next.fetch_add(1, std::memory_order_relaxed);
          if (index >= job.operations.size()) break;
          const auto& operation = job.operations[index];
          const std::size_t row_bytes = static_cast<std::size_t>(operation.columns * 2U);
          std::vector<std::uint8_t> source_row(row_bytes);
          Sha256 hash;
          SourceAnalysis analysis;
          analysis.source_min = std::numeric_limits<float>::infinity();
          analysis.source_max = -std::numeric_limits<float>::infinity();
          for (std::uint64_t row = 0; row < operation.rows; ++row) {
            auto status = ReadExact(files[index].source->fd,
                                    operation.source_offset + row * row_bytes,
                                    source_row.data(), source_row.size());
            if (!status.ok()) {
              std::lock_guard lock(error_mutex);
              if (first_error.ok()) first_error = status;
              cancelled.store(true, std::memory_order_release);
              break;
            }
            hash.Update(source_row.data(), source_row.size());
            for (std::size_t offset = 0; offset < source_row.size(); offset += 2U) {
              const float value = DecodeBf16(LoadLe16(source_row.data() + offset));
              if (!std::isfinite(value)) {
                std::lock_guard lock(error_mutex);
                if (first_error.ok()) first_error = Data("source contains nonfinite BF16");
                cancelled.store(true, std::memory_order_release);
                break;
              }
              analysis.tensor_amax = std::max(analysis.tensor_amax, std::fabs(value));
              analysis.source_min = std::min(analysis.source_min, value);
              analysis.source_max = std::max(analysis.source_max, value);
              analysis.source_sum_squares += static_cast<double>(value) * value;
            }
            if (cancelled.load(std::memory_order_acquire)) break;
          }
          if (cancelled.load(std::memory_order_acquire)) continue;
          analysis.source_hash = hash.HexDigest();
          if (analysis.source_hash != operation.source_hash) {
            std::lock_guard lock(error_mutex);
            if (first_error.ok()) first_error = Data("source hash mismatch before output: " + operation.source_name);
            cancelled.store(true, std::memory_order_release);
            continue;
          }
          (*analyses)[index] = std::move(analysis);
        }
      } catch (const std::bad_alloc&) {
        std::lock_guard lock(error_mutex);
        if (first_error.ok()) first_error = Resource("native NVFP4 analysis allocation failed");
        cancelled.store(true, std::memory_order_release);
      } catch (...) {
        std::lock_guard lock(error_mutex);
        if (first_error.ok()) first_error = Internal("native NVFP4 source analysis failed");
        cancelled.store(true, std::memory_order_release);
      }
      });
    }
  } catch (const std::bad_alloc&) {
    cancelled.store(true, std::memory_order_release);
    for (auto& thread : threads) if (thread.joinable()) thread.join();
    return Resource("native NVFP4 analysis worker allocation failed");
  } catch (...) {
    cancelled.store(true, std::memory_order_release);
    for (auto& thread : threads) if (thread.joinable()) thread.join();
    return Internal("native NVFP4 analysis worker creation failed");
  }
  for (auto& thread : threads) thread.join();
  return first_error;
}
#endif

struct OperationResult {
  Operation operation;
  std::string source_hash;
  std::string packed_hash;
  std::string scale_hash;
  std::string weight_hash;
  std::string input_hash;
  float tensor_amax = 0.F;
  float weight_divisor = 1.F;
  float input_divisor = 1.F;
  float source_min = 0.F;
  float source_max = 0.F;
  double source_sum_squares = 0.0;
  double reconstruction_sum_squares = 0.0;
  double source_reconstruction_dot = 0.0;
  double error_sum_squares = 0.0;
  float max_absolute_error = 0.F;
  float scale_min = 0.F;
  float scale_max = 0.F;
  std::uint64_t zero_blocks = 0;
  std::uint64_t underflow_blocks = 0;
  std::uint64_t saturation_count = 0;
  std::array<std::uint64_t, 16> code_histogram{};
  std::array<std::uint64_t, 256> scale_histogram{};
};

#ifndef _WIN32
Status ConvertOperation(const Operation& operation, const Preflight& files,
                        const SourceAnalysis& analysis, OperationResult* result) {
  const std::size_t row_bytes = static_cast<std::size_t>(operation.columns * 2U);
  std::vector<std::uint8_t> source_row(row_bytes);
  const float tensor_amax = analysis.tensor_amax;
  const std::string& first_digest = analysis.source_hash;
  const float source_min = analysis.source_min;
  const float source_max = analysis.source_max;
  const double source_sum_squares = analysis.source_sum_squares;
  float weight_divisor = 1.F;
  if (tensor_amax != 0.F) {
    const float raw = 2688.F / tensor_amax;
    weight_divisor = DecodeBf16(EncodeBf16Rne(raw));
  }
  if (!std::isfinite(weight_divisor) || weight_divisor <= 0.F) {
    return Data("weight divisor is not finite and positive");
  }
  const float input_divisor = 1.F;
  Sha256 packed_hash;
  Sha256 scale_hash;
  Sha256 second_hash;
  std::vector<std::uint8_t> packed_row(static_cast<std::size_t>(operation.columns / 2));
  std::vector<std::uint8_t> scale_row(static_cast<std::size_t>(operation.columns / 16));
  std::array<std::uint8_t, 4> scalar_bytes{};
  StoreLe32(weight_divisor, scalar_bytes.data());
  auto status = WriteExact(files.outputs[2]->fd, operation.weight_global.offset,
                           scalar_bytes.data(), scalar_bytes.size());
  if (!status.ok()) return status;
  StoreLe32(input_divisor, scalar_bytes.data());
  status = WriteExact(files.outputs[3]->fd, operation.input_global.offset,
                      scalar_bytes.data(), scalar_bytes.size());
  if (!status.ok()) return status;

  float scale_min = std::numeric_limits<float>::infinity();
  float scale_max = 0.F;
  std::uint64_t zero_blocks = 0;
  std::uint64_t underflow_blocks = 0;
  std::uint64_t saturation_count = 0;
  std::array<std::uint64_t, 16> code_histogram{};
  std::array<std::uint64_t, 256> scale_histogram{};
  double reconstruction_sum_squares = 0.0;
  double source_reconstruction_dot = 0.0;
  double error_sum_squares = 0.0;
  float max_absolute_error = 0.F;

  for (std::uint64_t row = 0; row < operation.rows; ++row) {
    status = ReadExact(files.source->fd, operation.source_offset + row * row_bytes,
                       source_row.data(), source_row.size());
    if (!status.ok()) return status;
    second_hash.Update(source_row.data(), source_row.size());
    std::fill(packed_row.begin(), packed_row.end(), 0);
    std::fill(scale_row.begin(), scale_row.end(), 0);
    for (std::uint64_t block = 0; block < operation.columns / 16; ++block) {
      float block_amax = 0.F;
      for (std::uint64_t k = 0; k < 16; ++k) {
        const auto index = static_cast<std::size_t>((block * 16 + k) * 2);
        block_amax = std::max(block_amax, std::fabs(DecodeBf16(LoadLe16(source_row.data() + index))));
      }
      const float raw_scale = (block_amax * (1.F / 6.F)) * weight_divisor;
      const std::uint8_t scale_code = EncodeE4M3(raw_scale);
      scale_row[static_cast<std::size_t>(block)] = scale_code;
      ++scale_histogram[scale_code];
      const float decoded_scale = DecodeE4M3(scale_code);
      if (block_amax == 0.F) {
        ++zero_blocks;
      } else if (scale_code == 0) {
        ++underflow_blocks;
      } else {
        scale_min = std::min(scale_min, decoded_scale);
        scale_max = std::max(scale_max, decoded_scale);
      }
      for (std::uint64_t k = 0; k < 16; ++k) {
        const auto index = static_cast<std::size_t>((block * 16 + k) * 2);
        const float value = DecodeBf16(LoadLe16(source_row.data() + index));
        std::uint8_t code = 0;
        float reconstructed = 0.F;
        if (scale_code != 0) {
          const float normalized = (value * weight_divisor) / decoded_scale;
          if (std::fabs(normalized) > 6.F) ++saturation_count;
          code = EncodeE2M1(normalized);
          reconstructed = DecodeE2M1(code) * decoded_scale / weight_divisor;
        }
        ++code_histogram[code];
        PutNibble(&packed_row[static_cast<std::size_t>(block * 8 + k / 2)], k, code);
        reconstruction_sum_squares += static_cast<double>(reconstructed) * reconstructed;
        source_reconstruction_dot += static_cast<double>(value) * reconstructed;
        const float error = reconstructed - value;
        error_sum_squares += static_cast<double>(error) * error;
        max_absolute_error = std::max(max_absolute_error, std::fabs(error));
      }
    }
    status = WriteExact(files.outputs[0]->fd, operation.packed.offset + row * packed_row.size(),
                        packed_row.data(), packed_row.size());
    if (!status.ok()) return status;
    status = WriteExact(files.outputs[1]->fd, operation.local_scale.offset + row * scale_row.size(),
                        scale_row.data(), scale_row.size());
    if (!status.ok()) return status;
    packed_hash.Update(packed_row.data(), packed_row.size());
    scale_hash.Update(scale_row.data(), scale_row.size());
  }
  if (second_hash.HexDigest() != first_digest) return Data("source changed during conversion");
  std::array<std::uint8_t, 4> weight_bytes{};
  std::array<std::uint8_t, 4> input_bytes{};
  StoreLe32(weight_divisor, weight_bytes.data());
  StoreLe32(input_divisor, input_bytes.data());
  const std::string weight_hash = Sha256Hex(weight_bytes.data(), weight_bytes.size());
  const std::string input_hash = Sha256Hex(input_bytes.data(), input_bytes.size());
  *result = OperationResult{operation, first_digest, packed_hash.HexDigest(), scale_hash.HexDigest(),
                            weight_hash, input_hash, tensor_amax, weight_divisor, input_divisor,
                            source_min, source_max, source_sum_squares, reconstruction_sum_squares,
                            source_reconstruction_dot, error_sum_squares, max_absolute_error,
                            std::isfinite(scale_min) ? scale_min : 0.F, scale_max, zero_blocks,
                            underflow_blocks, saturation_count, code_histogram, scale_histogram};
  return Status::Ok();
}
#endif

Value BuildTelemetry(const Job& job, const std::vector<OperationResult>& results,
                     std::uint64_t maximum_source_row_bytes,
                     double analysis_seconds, double conversion_seconds) {
  Value::Object root;
  root.emplace("schema_version", Integer(1));
  root.emplace("protocol", String(job.protocol));
  root.emplace("artifact_profile", String(job.profile));
  root.emplace("scope", String(job.scope));
  root.emplace("contract_id", String(job.contract_id));
  root.emplace("contract_version", Integer(job.contract_version));
  root.emplace("threads", Integer(job.threads));
  root.emplace("source_passes", Integer(2));
  root.emplace("maximum_source_row_bytes", Integer(maximum_source_row_bytes));
  root.emplace("analysis_seconds", Decimal(analysis_seconds));
  root.emplace("conversion_seconds", Decimal(conversion_seconds));
  Value::Object build;
  build.emplace("compiler_id", String(GEM16_NATIVE_COMPILER_ID));
  build.emplace("compiler_version", String(GEM16_NATIVE_COMPILER_VERSION));
  build.emplace("build_type", String(GEM16_NATIVE_BUILD_TYPE));
  build.emplace("cxx_standard", String(GEM16_NATIVE_CXX_STANDARD));
  build.emplace("system", String(GEM16_NATIVE_SYSTEM));
  build.emplace("processor", String(GEM16_NATIVE_PROCESSOR));
  root.emplace("native_build", Value(std::move(build)));
  Value::Array operations;
  for (const auto& result : results) {
    const auto& op = result.operation;
    Value::Object item;
    item.emplace("operation_id", String(op.operation_id));
    item.emplace("source_name", String(op.source_name));
    item.emplace("source_sha256", String(result.source_hash));
    item.emplace("source_bytes", Integer(op.source_bytes));
    item.emplace("source_dtype", String(op.source_dtype));
    item.emplace("role", String(op.role));
    item.emplace("axis_transformation", String(op.axis_transformation));
    item.emplace("disk_layout", String(op.disk_layout));
    item.emplace("runtime_layout", String(op.runtime_layout));
    Value::Array shape;
    for (const auto dimension : op.shape) shape.emplace_back(Integer(dimension));
    item.emplace("logical_shape", Value(std::move(shape)));
    item.emplace("rows", Integer(op.rows));
    item.emplace("columns", Integer(op.columns));
    item.emplace("packed_name", String(op.packed.name));
    item.emplace("packed_sha256", String(result.packed_hash));
    item.emplace("local_scale_name", String(op.local_scale.name));
    item.emplace("local_scale_sha256", String(result.scale_hash));
    item.emplace("weight_global_name", String(op.weight_global.name));
    item.emplace("weight_global_sha256", String(result.weight_hash));
    item.emplace("input_global_name", String(op.input_global.name));
    item.emplace("input_global_sha256", String(result.input_hash));
    item.emplace("tensor_amax", Decimal(result.tensor_amax));
    item.emplace("weight_divisor", Decimal(result.weight_divisor));
    item.emplace("input_divisor", Decimal(result.input_divisor));
    item.emplace("source_min", Decimal(result.source_min));
    item.emplace("source_max", Decimal(result.source_max));
    item.emplace("source_sum_squares", Decimal(result.source_sum_squares));
    item.emplace("reconstruction_sum_squares", Decimal(result.reconstruction_sum_squares));
    item.emplace("source_reconstruction_dot", Decimal(result.source_reconstruction_dot));
    item.emplace("error_sum_squares", Decimal(result.error_sum_squares));
    item.emplace("max_absolute_error", Decimal(result.max_absolute_error));
    item.emplace("scale_min", Decimal(result.scale_min));
    item.emplace("scale_max", Decimal(result.scale_max));
    item.emplace("zero_blocks", Integer(result.zero_blocks));
    item.emplace("underflow_blocks", Integer(result.underflow_blocks));
    item.emplace("saturation_count", Integer(result.saturation_count));
    Value::Array codes;
    for (const auto count : result.code_histogram) codes.emplace_back(Integer(count));
    item.emplace("code_histogram", Value(std::move(codes)));
    Value::Array scales;
    for (const auto count : result.scale_histogram) scales.emplace_back(Integer(count));
    item.emplace("scale_histogram", Value(std::move(scales)));
    operations.emplace_back(Value(std::move(item)));
  }
  root.emplace("operations", Value(std::move(operations)));
  return Value(std::move(root));
}

#ifndef _WIN32
Status PublishTelemetry(const std::filesystem::path& telemetry_path, std::string_view text) {
  const std::string target = telemetry_path.string();
  struct stat existing{};
  if (lstat(target.c_str(), &existing) == 0) return Invalid("telemetry output already exists");
  if (errno != ENOENT) return Io("cannot inspect telemetry output");
  const std::string temporary = target + ".incomplete";
  const int fd = open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
  if (fd < 0) return Io("cannot create telemetry staging file");
  auto cleanup = [&] { unlink(temporary.c_str()); };
  std::size_t completed = 0;
  while (completed < text.size()) {
    const ssize_t count = write(fd, text.data() + completed, text.size() - completed);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) {
      close(fd);
      cleanup();
      return Io("cannot write telemetry staging file");
    }
    completed += static_cast<std::size_t>(count);
  }
  if (fsync(fd) != 0 || close(fd) != 0) {
    cleanup();
    return Io("cannot fsync telemetry staging file");
  }
  if (link(temporary.c_str(), target.c_str()) != 0) {
    cleanup();
    return errno == EEXIST ? Invalid("telemetry output already exists")
                           : Io("cannot publish telemetry");
  }
  cleanup();
  return Status::Ok();
}
#endif

}  // namespace

float DecodeNvfp4E2M1(std::uint8_t code) noexcept { return DecodeE2M1(code); }

std::uint8_t EncodeNvfp4E2M1(float value) noexcept { return EncodeE2M1(value); }

float DecodeNvfp4E4M3(std::uint8_t code) noexcept { return DecodeE4M3(code); }

std::uint8_t EncodeNvfp4E4M3(float value) noexcept { return EncodeE4M3(value); }

bool Nvfp4BuildSupportsFullJob() noexcept {
  return std::string_view(GEM16_NATIVE_BUILD_TYPE) == "Release";
}

int ExitCodeForNvfp4Status(StatusCode code) noexcept {
  switch (code) {
    case StatusCode::kInvalidArgument: return 2;
    case StatusCode::kDataLoss: return 4;
    case StatusCode::kIoError: return 5;
    case StatusCode::kResourceExhausted:
    case StatusCode::kInternal:
    case StatusCode::kUnsupported: return 1;
    default: return 1;
  }
}

const char* Nvfp4BuildInfoJson() noexcept {
  static const char kInfo[] =
      "{\"native_build\":{\"build_type\":\"" GEM16_NATIVE_BUILD_TYPE
      "\",\"compiler_id\":\"" GEM16_NATIVE_COMPILER_ID
      "\",\"compiler_version\":\"" GEM16_NATIVE_COMPILER_VERSION
      "\",\"cxx_standard\":\"" GEM16_NATIVE_CXX_STANDARD
      "\",\"processor\":\"" GEM16_NATIVE_PROCESSOR
      "\",\"system\":\"" GEM16_NATIVE_SYSTEM
      "\"},\"protocol\":\"gem16-nvfp4-direct-v1\"}";
  return kInfo;
}

Status EncodeNvfp4JobFile(const std::filesystem::path& job_path,
                          const std::filesystem::path& telemetry_path) {
#ifdef _WIN32
  (void)job_path;
  (void)telemetry_path;
  return Unsupported("native NVFP4 compiler requires POSIX direct-range I/O");
#else
  if (std::fegetround() != FE_TONEAREST && std::fesetround(FE_TONEAREST) != 0) {
    return Internal("cannot establish round-to-nearest floating mode");
  }
  std::error_code error;
  const auto size = std::filesystem::file_size(job_path, error);
  if (error || size > kMaxJobBytes) return Io("cannot read bounded NVFP4 job");
  auto descriptor = OpenRegular(job_path.string(), O_RDONLY, size);
  if (!descriptor.ok()) return descriptor.status();
  std::string text(size, '\0');
  auto status = ReadExact(descriptor.value().fd, 0, text.data(), text.size());
  if (!status.ok()) return status;
  auto parsed = json::Parse(text, {128, 2'000'000, kMaxJobBytes});
  if (!parsed.ok()) return Invalid("invalid NVFP4 job JSON");
  auto parsed_job = ParseJob(parsed.value());
  if (!parsed_job.ok()) return parsed_job.status();
  const Job& job = parsed_job.value();
  if ((job.scope == "full" || job.scope == "tied_head") && !Nvfp4BuildSupportsFullJob()) {
    return Invalid(job.scope == "tied_head"
                       ? "M07 tied-head conversion requires a native Release build"
                       : "full NVFP4 conversion requires a native Release build");
  }
  std::vector<Preflight> files;
  status = PreflightFiles(job, job_path, telemetry_path, &files);
  if (!status.ok()) return status;
  const auto analysis_start = std::chrono::steady_clock::now();
  std::vector<SourceAnalysis> analyses;
  status = AnalyzeSources(job, files, &analyses);
  if (!status.ok()) return status;
  const double analysis_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - analysis_start).count();
  const auto conversion_start = std::chrono::steady_clock::now();
  std::vector<OperationResult> results(job.operations.size());
  std::atomic<bool> cancelled = false;
  std::mutex error_mutex;
  Status first_error = Status::Ok();
  const std::size_t worker_count = std::min<std::size_t>(job.threads, job.operations.size());
  std::vector<std::thread> workers;
  std::atomic<std::size_t> next_operation{0};
  try {
    workers.reserve(worker_count);
    for (std::size_t worker = 0; worker < worker_count; ++worker) {
      workers.emplace_back([&, worker] {
        try {
          (void)worker;
          while (!cancelled.load(std::memory_order_acquire)) {
            const std::size_t index = next_operation.fetch_add(1, std::memory_order_relaxed);
            if (index >= job.operations.size()) break;
            OperationResult result;
            const Status operation_status =
                ConvertOperation(job.operations[index], files[index], analyses[index], &result);
            if (!operation_status.ok()) {
              std::lock_guard lock(error_mutex);
              if (first_error.ok()) first_error = operation_status;
              cancelled.store(true, std::memory_order_release);
              break;
            }
            results[index] = std::move(result);
          }
        } catch (const std::bad_alloc&) {
          std::lock_guard lock(error_mutex);
          if (first_error.ok()) first_error = Resource("native NVFP4 worker allocation failed");
          cancelled.store(true, std::memory_order_release);
        } catch (const std::exception& exception) {
          std::lock_guard lock(error_mutex);
          if (first_error.ok()) first_error = Internal(std::string("native NVFP4 worker failed: ") + exception.what());
          cancelled.store(true, std::memory_order_release);
        } catch (...) {
          std::lock_guard lock(error_mutex);
          if (first_error.ok()) first_error = Internal("native NVFP4 worker failed");
          cancelled.store(true, std::memory_order_release);
        }
      });
    }
  } catch (const std::bad_alloc&) {
    cancelled.store(true, std::memory_order_release);
    for (auto& worker : workers) if (worker.joinable()) worker.join();
    return Resource("native NVFP4 worker allocation failed");
  } catch (...) {
    cancelled.store(true, std::memory_order_release);
    for (auto& worker : workers) if (worker.joinable()) worker.join();
    return Internal("native NVFP4 worker creation failed");
  }
  for (auto& worker : workers) worker.join();
  if (!first_error.ok()) return first_error;
  // Direct jobs commonly place many tensor ranges in the same small set of
  // Safetensors shards.  Durability is therefore established once per unique
  // output inode, after all workers have finished writing.
  std::vector<std::shared_ptr<FileHandle>> durable_outputs;
  for (const auto& file : files) {
    for (const auto& output : file.outputs) {
      bool seen = false;
      for (const auto& prior : durable_outputs) {
        if (SameFile(output->identity, prior->identity)) {
          seen = true;
          break;
        }
      }
      if (!seen) durable_outputs.push_back(output);
    }
  }
  for (const auto& output : durable_outputs) {
    auto identity_status = CheckIdentity(*output);
    if (!identity_status.ok()) return identity_status;
    auto fsync_status = Fsync(*output);
    if (!fsync_status.ok()) return fsync_status;
  }
  const double conversion_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - conversion_start).count();
  std::vector<std::size_t> order(results.size());
  for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::sort(order.begin(), order.end(), [&](std::size_t lhs, std::size_t rhs) {
    return results[lhs].operation.operation_id < results[rhs].operation.operation_id;
  });
  std::vector<OperationResult> sorted;
  sorted.reserve(results.size());
  for (const auto index : order) sorted.push_back(std::move(results[index]));
  std::uint64_t maximum_source_row_bytes = 0;
  for (const auto& operation : job.operations) maximum_source_row_bytes =
      std::max(maximum_source_row_bytes, operation.columns * 2);
  return PublishTelemetry(telemetry_path,
                          json::Stringify(BuildTelemetry(job, sorted, maximum_source_row_bytes,
                                                         analysis_seconds, conversion_seconds)));
#endif
}

}  // namespace gem16::compiler
