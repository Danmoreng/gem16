#include "compiler/fp8_batch_encoder.h"

#include "compiler/sha256.h"
#include "util/json.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cerrno>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <optional>
#include <set>
#include <string>
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

constexpr std::uint32_t kE4M3NaN = 0x7FU;
constexpr float kE4M3Max = 448.0F;
constexpr std::uint16_t kBf16One = 0x3F80U;
constexpr std::uint16_t kBf16Min = 0x0001U;
constexpr std::size_t kMaxJobBytes = 16U * 1024U * 1024U;
// The text-attention contract contains 115 matrices.  The independent Gemma 4
// The 26B Vision module contains 191 linear matrices. Keep comparison jobs at
// the narrower text boundary while allowing the same bounded row-wise encoder
// to service the explicitly larger offline Vision job.
constexpr std::uint64_t kMaxEncodeMatrices = 191U;
constexpr std::uint64_t kMaxCompareMatrices = 115U;
constexpr std::uint64_t kMaxDimension = 8192U;
constexpr std::uint64_t kMaxSourceRowBytes = 16384U;
constexpr std::uint64_t kMaxScaleBytes = 16384U;
constexpr std::uint64_t kMaxPayloadBytes = 2ULL * 1024ULL * 1024ULL * 1024ULL;

Status Invalid(std::string message) { return Status(StatusCode::kInvalidArgument, std::move(message)); }
Status IoFailure(std::string message) { return Status(StatusCode::kIoError, std::move(message)); }
Status SourceFailure(std::string message) { return Status(StatusCode::kDataLoss, std::move(message)); }
Status NumericFailure(std::string message) { return Status(StatusCode::kInternal, std::move(message)); }

Value I(std::uint64_t value) {
  return Value(value > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())
                   ? std::int64_t(std::numeric_limits<std::int64_t>::max())
                   : static_cast<std::int64_t>(value));
}
Value D(double value) { return Value(value); }
Value S(std::string value) { return Value(std::move(value)); }

#ifndef GEM16_FP8_NATIVE_COMPILER_ID
#define GEM16_FP8_NATIVE_COMPILER_ID "unknown"
#define GEM16_FP8_NATIVE_COMPILER_VERSION "unknown"
#define GEM16_FP8_NATIVE_BUILD_TYPE "unknown"
#define GEM16_FP8_NATIVE_CXX_STANDARD "20"
#define GEM16_FP8_NATIVE_SYSTEM "unknown"
#define GEM16_FP8_NATIVE_PROCESSOR "unknown"
#endif

Value NativeBuildValue() {
  Value::Object object;
  object.emplace("compiler_id", S(GEM16_FP8_NATIVE_COMPILER_ID));
  object.emplace("compiler_version", S(GEM16_FP8_NATIVE_COMPILER_VERSION));
  object.emplace("build_type", S(GEM16_FP8_NATIVE_BUILD_TYPE));
  object.emplace("cxx_standard", S(GEM16_FP8_NATIVE_CXX_STANDARD));
  object.emplace("system", S(GEM16_FP8_NATIVE_SYSTEM));
  object.emplace("processor", S(GEM16_FP8_NATIVE_PROCESSOR));
  return Value(std::move(object));
}

const Value* Required(const Value::Object& object, std::string_view key) {
  const auto iterator = object.find(std::string(key));
  return iterator == object.end() ? nullptr : &iterator->second;
}

bool ExactKeys(const Value::Object& object, std::initializer_list<std::string_view> expected) {
  if (object.size() != expected.size()) return false;
  for (const auto key : expected) {
    if (object.find(std::string(key)) == object.end()) return false;
  }
  return true;
}

Result<std::uint64_t> UInt(const Value& value, std::string_view description) {
  if (!value.is_integer() || value.as_integer() < 0) {
    return Invalid(std::string(description) + " must be a non-negative integer");
  }
  return static_cast<std::uint64_t>(value.as_integer());
}

Result<std::string> String(const Value& value, std::string_view description) {
  if (!value.is_string() || value.as_string().empty() || value.as_string().find('\0') != std::string::npos) {
    return Invalid(std::string(description) + " must be a non-empty string");
  }
  return value.as_string();
}

Result<std::uint64_t> Product(std::uint64_t left, std::uint64_t right, std::string_view description) {
  if (right != 0U && left > std::numeric_limits<std::uint64_t>::max() / right) {
    return Invalid(std::string(description) + " overflows uint64");
  }
  return left * right;
}

bool IsHexDigest(std::string_view value) {
  if (value.size() != 64U) return false;
  return std::all_of(value.begin(), value.end(), [](char character) {
    return (character >= '0' && character <= '9') || (character >= 'a' && character <= 'f');
  });
}

float DecodeBf16(std::uint16_t bits) { return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U); }

std::uint16_t Bf16RneUnchecked(float value) {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  std::uint32_t upper = bits >> 16U;
  const std::uint32_t lower = bits & 0xFFFFU;
  if (lower > 0x8000U || (lower == 0x8000U && (upper & 1U) != 0U)) ++upper;
  return static_cast<std::uint16_t>(upper);
}

std::array<float, 127> E4M3Values() {
  std::array<float, 127> values{};
  for (std::uint32_t code = 0; code < values.size(); ++code) {
    const std::uint32_t exponent = (code >> 3U) & 0xFU;
    const std::uint32_t mantissa = code & 7U;
    values[code] = exponent == 0U
                       ? std::ldexp(static_cast<float>(mantissa), -9)
                       : std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F,
                                    static_cast<int>(exponent) - 7);
  }
  return values;
}
const std::array<float, 127>& PositiveValues() {
  static const auto values = E4M3Values();
  return values;
}
const std::array<float, 126>& PositiveMidpoints() {
  static const auto midpoints = [] {
    std::array<float, 126> result{};
    const auto& values = PositiveValues();
    for (std::size_t index = 0; index < result.size(); ++index) result[index] = (values[index] + values[index + 1U]) / 2.0F;
    return result;
  }();
  return midpoints;
}

const std::array<float, 256>& CompareDecodeValues() {
  static const auto values = [] {
    std::array<float, 256> result{};
    for (std::size_t index = 0; index < result.size(); ++index) {
      result[index] = DecodeE4M3Fn(static_cast<std::uint8_t>(index));
    }
    return result;
  }();
  return values;
}

struct MatrixJob {
  std::string source_name;
  std::filesystem::path source_path;
  std::string source_sha256;
  std::uint64_t source_offset = 0;
  std::uint64_t source_bytes = 0;
  std::uint64_t rows = 0;
  std::uint64_t columns = 0;
  std::string weight_output_name;
  std::uint64_t weight_offset = 0;
  std::uint64_t weight_bytes = 0;
  std::string scale_output_name;
  std::uint64_t scale_offset = 0;
  std::uint64_t scale_bytes = 0;
};
struct Job {
  std::uint64_t threads = 1;
  std::uint64_t payload_bytes = 0;
  std::vector<MatrixJob> matrices;
};
struct MatrixTelemetry {
  std::string source_name;
  std::string weight_output_name;
  std::string scale_output_name;
  std::string source_sha256;
  std::string weight_sha256;
  std::string scale_sha256;
  std::uint64_t rows = 0;
  std::uint64_t columns = 0;
  double source_min = std::numeric_limits<double>::infinity();
  double source_max = -std::numeric_limits<double>::infinity();
  double source_sum_squares = 0.0;
  double reconstruction_sum_squares = 0.0;
  double source_reconstruction_dot = 0.0;
  double error_sum_squares = 0.0;
  double max_absolute_error = 0.0;
  double scale_min = std::numeric_limits<double>::infinity();
  double scale_max = -std::numeric_limits<double>::infinity();
  std::uint64_t saturation_count = 0;
  std::uint64_t zero_rows = 0;
  std::uint64_t underflow_clamped_rows = 0;
  std::array<std::uint64_t, 256> histogram{};
};

Result<MatrixJob> ParseMatrix(const Value& value, std::size_t index, std::uint64_t& cursor,
                              std::set<std::string>& output_names) {
  if (!value.is_object() || !ExactKeys(value.as_object(), {"source_name", "source_path", "source_sha256", "source_offset", "source_bytes", "rows", "columns", "weight_output_name", "weight_offset", "weight_bytes", "scale_output_name", "scale_offset", "scale_bytes"})) return Invalid("matrix " + std::to_string(index) + " has an invalid key set");
  const auto& object = value.as_object();
  auto get_string = [&](std::string_view key) -> Result<std::string> {
    const Value* field = Required(object, key);
    return field == nullptr ? Result<std::string>(Invalid("matrix field missing: " + std::string(key))) : String(*field, "matrix." + std::string(key));
  };
  auto get_uint = [&](std::string_view key) -> Result<std::uint64_t> {
    const Value* field = Required(object, key);
    return field == nullptr ? Result<std::uint64_t>(Invalid("matrix field missing: " + std::string(key))) : UInt(*field, "matrix." + std::string(key));
  };
  MatrixJob matrix;
  auto source_name = get_string("source_name");
  auto source_path = get_string("source_path");
  auto source_hash = get_string("source_sha256");
  if (!source_name.ok()) return source_name.status();
  if (!source_path.ok()) return source_path.status();
  if (!source_hash.ok()) return source_hash.status();
  matrix.source_name = std::move(source_name).value();
  matrix.source_path = std::filesystem::path(std::move(source_path).value());
  matrix.source_sha256 = std::move(source_hash).value();
  if (!IsHexDigest(matrix.source_sha256)) return Invalid("matrix source_sha256 must be lowercase SHA-256: " + matrix.source_name);
  if (!matrix.source_path.is_absolute() || std::any_of(matrix.source_path.begin(), matrix.source_path.end(), [](const auto& part) { return part == ".."; })) {
    return Invalid("matrix source_path must be absolute and normalized: " + matrix.source_name);
  }
  auto source_offset = get_uint("source_offset");
  auto source_bytes = get_uint("source_bytes");
  auto rows = get_uint("rows");
  auto columns = get_uint("columns");
  if (!source_offset.ok()) return source_offset.status();
  if (!source_bytes.ok()) return source_bytes.status();
  if (!rows.ok()) return rows.status();
  if (!columns.ok()) return columns.status();
  matrix.source_offset = source_offset.value();
  matrix.source_bytes = source_bytes.value();
  matrix.rows = rows.value();
  matrix.columns = columns.value();
  if (matrix.rows == 0U || matrix.rows > kMaxDimension || matrix.columns == 0U || matrix.columns > kMaxDimension || matrix.columns * 2U > kMaxSourceRowBytes || matrix.rows * 2U > kMaxScaleBytes) {
    return Invalid("matrix shape exceeds bounded M05 limits: " + matrix.source_name);
  }
  auto elements = Product(matrix.rows, matrix.columns, "matrix element count");
  if (!elements.ok()) return elements.status();
  auto source_expected = Product(elements.value(), 2U, "matrix BF16 bytes");
  if (!source_expected.ok() || matrix.source_bytes != source_expected.value()) return Invalid("matrix source_bytes does not match BF16 shape: " + matrix.source_name);
  auto weight_name = get_string("weight_output_name");
  auto weight_offset = get_uint("weight_offset");
  auto weight_bytes = get_uint("weight_bytes");
  auto scale_name = get_string("scale_output_name");
  auto scale_offset = get_uint("scale_offset");
  auto scale_bytes = get_uint("scale_bytes");
  if (!weight_name.ok()) return weight_name.status();
  if (!weight_offset.ok()) return weight_offset.status();
  if (!weight_bytes.ok()) return weight_bytes.status();
  if (!scale_name.ok()) return scale_name.status();
  if (!scale_offset.ok()) return scale_offset.status();
  if (!scale_bytes.ok()) return scale_bytes.status();
  matrix.weight_output_name = std::move(weight_name).value();
  matrix.weight_offset = weight_offset.value();
  matrix.weight_bytes = weight_bytes.value();
  matrix.scale_output_name = std::move(scale_name).value();
  matrix.scale_offset = scale_offset.value();
  matrix.scale_bytes = scale_bytes.value();
  auto scale_expected = Product(matrix.rows, 2U, "matrix scale bytes");
  if (!scale_expected.ok() || matrix.weight_bytes != elements.value() || matrix.scale_bytes != scale_expected.value() || matrix.weight_offset != cursor || matrix.weight_bytes > std::numeric_limits<std::uint64_t>::max() - cursor || matrix.scale_offset != cursor + matrix.weight_bytes || matrix.scale_bytes > std::numeric_limits<std::uint64_t>::max() - matrix.scale_offset) {
    return Invalid("matrix output ranges are not canonical: " + matrix.source_name);
  }
  if (!output_names.insert(matrix.weight_output_name).second || !output_names.insert(matrix.scale_output_name).second) return Invalid("duplicate matrix output name: " + matrix.source_name);
  cursor = matrix.scale_offset + matrix.scale_bytes;
  return matrix;
}

Result<Job> ParseJob(const std::filesystem::path& path) {
  std::error_code error;
  const auto size = std::filesystem::file_size(path, error);
  if (error) return Invalid("cannot inspect job file: " + path.string());
  if (size > kMaxJobBytes) return Invalid("job JSON exceeds 16 MiB limit");
  std::ifstream input(path, std::ios::binary);
  if (!input) return IoFailure("cannot open job file: " + path.string());
  std::string text(static_cast<std::size_t>(size), '\0');
  if (size != 0U) input.read(text.data(), static_cast<std::streamsize>(size));
  if (!input && !input.eof()) return IoFailure("cannot read job file: " + path.string());
  auto parsed = json::Parse(text);
  if (!parsed.ok()) return Invalid("invalid job JSON: " + parsed.status().message());
  if (!parsed.value().is_object() || !ExactKeys(parsed.value().as_object(), {"schema_version", "contract_id", "contract_version", "threads", "payload_bytes", "matrices"})) return Invalid("job JSON has an invalid root key set");
  const auto& object = parsed.value().as_object();
  const Value* schema = Required(object, "schema_version");
  const Value* contract = Required(object, "contract_id");
  const Value* version = Required(object, "contract_version");
  const Value* threads = Required(object, "threads");
  const Value* payload = Required(object, "payload_bytes");
  const Value* matrices = Required(object, "matrices");
  if (schema == nullptr || !schema->is_integer() || schema->as_integer() != 1 || contract == nullptr || !contract->is_string() || contract->as_string() != "gem16.fp8_attention_rowwise" || version == nullptr || !version->is_integer() || version->as_integer() != 1 || threads == nullptr || payload == nullptr || matrices == nullptr) return Invalid("job contract/version is unsupported");
  auto thread_count = UInt(*threads, "job.threads");
  auto payload_size = UInt(*payload, "job.payload_bytes");
  if (!thread_count.ok()) return thread_count.status();
  if (!payload_size.ok()) return payload_size.status();
  if (thread_count.value() == 0U || thread_count.value() > 64U) return Invalid("job.threads must be between 1 and 64");
  if (payload_size.value() > kMaxPayloadBytes) return Invalid("job.payload_bytes exceeds 2 GiB limit");
  if (!matrices->is_array() || matrices->as_array().empty() || matrices->as_array().size() > kMaxEncodeMatrices) return Invalid("job.matrices must contain 1..191 matrices");
  Job job;
  job.threads = thread_count.value();
  job.payload_bytes = payload_size.value();
  std::uint64_t cursor = 0;
  std::set<std::string> output_names;
  job.matrices.reserve(matrices->as_array().size());
  for (std::size_t index = 0; index < matrices->as_array().size(); ++index) {
    auto matrix = ParseMatrix(matrices->as_array()[index], index, cursor, output_names);
    if (!matrix.ok()) return matrix.status();
    job.matrices.push_back(std::move(matrix).value());
  }
  if (cursor != job.payload_bytes) return Invalid("job payload_bytes does not cover matrix outputs");
  return job;
}

struct OwnedIdentity {
  std::uint64_t device = 0;
  std::uint64_t inode = 0;
  bool valid = false;
};

#ifndef _WIN32
bool RemoveOwnedPath(const std::filesystem::path& path, const OwnedIdentity& identity) {
  if (!identity.valid) return false;
  struct stat current {};
  if (::lstat(path.c_str(), &current) != 0) return errno == ENOENT;
  if (current.st_dev != identity.device || current.st_ino != identity.inode) return false;
  return ::unlink(path.c_str()) == 0 || errno == ENOENT;
}

class SourceFile {
 public:
  ~SourceFile() { if (fd_ >= 0) ::close(fd_); }
  SourceFile(const SourceFile&) = delete;
  SourceFile& operator=(const SourceFile&) = delete;
  SourceFile(SourceFile&& other) noexcept : fd_(std::exchange(other.fd_, -1)) {}
  SourceFile& operator=(SourceFile&& other) noexcept {
    if (this != &other) {
      if (fd_ >= 0) ::close(fd_);
      fd_ = std::exchange(other.fd_, -1);
    }
    return *this;
  }
  static Result<SourceFile> Open(const MatrixJob& matrix) {
    const int fd = ::open(matrix.source_path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) return SourceFailure("cannot open canonical source: " + matrix.source_path.string());
    SourceFile file(fd);
    struct stat info {};
    if (::fstat(fd, &info) != 0 || !S_ISREG(info.st_mode)) return SourceFailure("source is not a regular file: " + matrix.source_name);
    if (matrix.source_offset > static_cast<std::uint64_t>(info.st_size) || matrix.source_bytes > static_cast<std::uint64_t>(info.st_size) - matrix.source_offset) return SourceFailure("source tensor range is outside file: " + matrix.source_name);
    return file;
  }
  Result<std::size_t> Read(std::uint64_t offset, std::uint8_t* data, std::size_t size) const {
    std::size_t total = 0;
    while (total < size) {
      const ssize_t count = ::pread(fd_, data + total, size - total, static_cast<off_t>(offset + total));
      if (count < 0) {
        if (errno == EINTR) continue;
        return SourceFailure("cannot read source tensor");
      }
      if (count == 0) break;
      total += static_cast<std::size_t>(count);
    }
    return total;
  }
 private:
  explicit SourceFile(int fd) : fd_(fd) {}
  int fd_ = -1;
};

class OutputFile {
 public:
  ~OutputFile() { if (fd_ >= 0) ::close(fd_); }
  OutputFile(const OutputFile&) = delete;
  OutputFile& operator=(const OutputFile&) = delete;
  OutputFile(OutputFile&& other) noexcept
      : fd_(std::exchange(other.fd_, -1)), path_(std::move(other.path_)), identity_(other.identity_) {
    other.identity_.valid = false;
  }
  OutputFile& operator=(OutputFile&& other) noexcept {
    if (this != &other) {
      if (fd_ >= 0) ::close(fd_);
      fd_ = std::exchange(other.fd_, -1);
      path_ = std::move(other.path_);
      identity_ = other.identity_;
      other.identity_.valid = false;
    }
    return *this;
  }
  static Result<OutputFile> Create(const std::filesystem::path& path, std::uint64_t size) {
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (fd < 0) return IoFailure("cannot exclusively create output: " + path.string());
    struct stat identity {};
    if (::fstat(fd, &identity) != 0) {
      ::close(fd);
      return IoFailure("cannot inspect created output: " + path.string());
    }
    OutputFile file(fd, path, OwnedIdentity{static_cast<std::uint64_t>(identity.st_dev), static_cast<std::uint64_t>(identity.st_ino), true});
    if (size > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()) || ::ftruncate(fd, static_cast<off_t>(size)) != 0) {
      file.RemoveIfOwned();
      return IoFailure("cannot size output: " + path.string());
    }
    return file;
  }
  Status Write(std::uint64_t offset, const std::uint8_t* data, std::size_t size) const {
    std::size_t total = 0;
    while (total < size) {
      const ssize_t count = ::pwrite(fd_, data + total, size - total, static_cast<off_t>(offset + total));
      if (count < 0) {
        if (errno == EINTR) continue;
        return IoFailure("cannot write output");
      }
      if (count == 0) return IoFailure("short output write");
      total += static_cast<std::size_t>(count);
    }
    return Status::Ok();
  }
  Status Sync() const { return ::fsync(fd_) == 0 ? Status::Ok() : IoFailure("cannot fsync output"); }
  void RemoveIfOwned() { RemoveOwnedPath(path_, identity_); identity_.valid = false; }
 private:
  explicit OutputFile(int fd, std::filesystem::path path, OwnedIdentity identity)
      : fd_(fd), path_(std::move(path)), identity_(identity) {}
  int fd_ = -1;
  std::filesystem::path path_;
  OwnedIdentity identity_;
};
#else
class SourceFile {};
class OutputFile {};
#endif

Status SyncParent(const std::filesystem::path& path) {
#ifndef _WIN32
  const int fd = ::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (fd < 0) return IoFailure("cannot open output directory for fsync");
  const int result = ::fsync(fd);
  ::close(fd);
  return result == 0 ? Status::Ok() : IoFailure("cannot fsync output directory");
#else
  (void)path;
  return Status::Ok();
#endif
}

Status EncodeMatrix(const MatrixJob& matrix, const OutputFile& output, MatrixTelemetry& telemetry) {
#ifdef _WIN32
  (void)matrix;
  (void)output;
  (void)telemetry;
  return Status(StatusCode::kUnsupported, "native M05 compiler requires descriptor I/O on Linux");
#else
  auto source_result = SourceFile::Open(matrix);
  if (!source_result.ok()) return source_result.status();
  SourceFile source = std::move(source_result).value();
  const std::size_t row_bytes = static_cast<std::size_t>(matrix.columns * 2U);
  std::vector<std::uint8_t> row(row_bytes);
  std::vector<float> values(static_cast<std::size_t>(matrix.columns));
  std::vector<std::uint8_t> encoded(static_cast<std::size_t>(matrix.columns));
  std::vector<std::uint8_t> scales(static_cast<std::size_t>(matrix.rows * 2U));
  Sha256 source_hash, weight_hash, scale_hash;
  telemetry.source_name = matrix.source_name;
  telemetry.weight_output_name = matrix.weight_output_name;
  telemetry.scale_output_name = matrix.scale_output_name;
  telemetry.rows = matrix.rows;
  telemetry.columns = matrix.columns;
  for (std::uint64_t row_index = 0; row_index < matrix.rows; ++row_index) {
    auto read = source.Read(matrix.source_offset + row_index * row_bytes, row.data(), row.size());
    if (!read.ok() || read.value() != row.size()) return SourceFailure("short BF16 source row: " + matrix.source_name);
    source_hash.Update(row.data(), row.size());
    float maximum = 0.0F;
    for (std::uint64_t column = 0; column < matrix.columns; ++column) {
      const std::uint16_t bits = static_cast<std::uint16_t>(row[column * 2U]) | static_cast<std::uint16_t>(row[column * 2U + 1U] << 8U);
      const float value = DecodeBf16(bits);
      if (!std::isfinite(value)) return NumericFailure("BF16 source contains NaN/Inf: " + matrix.source_name);
      values[static_cast<std::size_t>(column)] = value;
      maximum = std::max(maximum, std::fabs(value));
    }
    std::uint16_t scale_bits = kBf16One;
    bool clamped = false;
    if (maximum != 0.0F) {
      scale_bits = Bf16RneUnchecked(maximum / kE4M3Max);
      if (scale_bits == 0U) { scale_bits = kBf16Min; clamped = true; }
      const float scale = DecodeBf16(scale_bits);
      if (!std::isfinite(scale) || scale <= 0.0F) return NumericFailure("invalid row scale");
      telemetry.scale_min = std::min(telemetry.scale_min, static_cast<double>(scale));
      telemetry.scale_max = std::max(telemetry.scale_max, static_cast<double>(scale));
      if (clamped) ++telemetry.underflow_clamped_rows;
    } else {
      ++telemetry.zero_rows;
      telemetry.scale_min = std::min(telemetry.scale_min, 1.0);
      telemetry.scale_max = std::max(telemetry.scale_max, 1.0);
    }
    scales[static_cast<std::size_t>(row_index * 2U)] = static_cast<std::uint8_t>(scale_bits & 0xFFU);
    scales[static_cast<std::size_t>(row_index * 2U + 1U)] = static_cast<std::uint8_t>(scale_bits >> 8U);
    const float scale = DecodeBf16(scale_bits);
    // The versioned telemetry contract closes each row's binary64 sums before
    // adding that row to the matrix aggregate.  Keep these accumulators local:
    // a single matrix-wide element loop has a different association and can
    // change the retained statistics even when payload bytes are identical.
    double row_source_sum_squares = 0.0;
    double row_reconstruction_sum_squares = 0.0;
    double row_source_reconstruction_dot = 0.0;
    double row_error_sum_squares = 0.0;
    for (std::uint64_t column = 0; column < matrix.columns; ++column) {
      const float value = values[static_cast<std::size_t>(column)];
      const float normalized = value / scale;
      auto code = EncodeE4M3Fn(normalized);
      if (!code.ok()) return NumericFailure(code.status().message());
      encoded[static_cast<std::size_t>(column)] = code.value();
      const float reconstructed = DecodeE4M3Fn(code.value()) * scale;
      const double source_value = static_cast<double>(value);
      const double reconstruction = static_cast<double>(reconstructed);
      const double difference = source_value - reconstruction;
      telemetry.source_min = std::min(telemetry.source_min, source_value);
      telemetry.source_max = std::max(telemetry.source_max, source_value);
      row_source_sum_squares += source_value * source_value;
      row_reconstruction_sum_squares += reconstruction * reconstruction;
      row_source_reconstruction_dot += source_value * reconstruction;
      row_error_sum_squares += difference * difference;
      telemetry.max_absolute_error = std::max(telemetry.max_absolute_error, std::fabs(difference));
      ++telemetry.histogram[code.value()];
      if (std::fabs(normalized) > kE4M3Max) ++telemetry.saturation_count;
    }
    telemetry.source_sum_squares += row_source_sum_squares;
    telemetry.reconstruction_sum_squares += row_reconstruction_sum_squares;
    telemetry.source_reconstruction_dot += row_source_reconstruction_dot;
    telemetry.error_sum_squares += row_error_sum_squares;
    auto write_status = output.Write(matrix.weight_offset + row_index * matrix.columns, encoded.data(), encoded.size());
    if (!write_status.ok()) return write_status;
    weight_hash.Update(encoded.data(), encoded.size());
  }
  auto scale_status = output.Write(matrix.scale_offset, scales.data(), scales.size());
  if (!scale_status.ok()) return scale_status;
  scale_hash.Update(scales.data(), scales.size());
  telemetry.source_sha256 = source_hash.HexDigest();
  telemetry.weight_sha256 = weight_hash.HexDigest();
  telemetry.scale_sha256 = scale_hash.HexDigest();
  if (telemetry.source_sha256 != matrix.source_sha256) return SourceFailure("source SHA-256 mismatch: " + matrix.source_name);
  return Status::Ok();
#endif
}

Value MatrixValue(const MatrixTelemetry& telemetry) {
  Value::Object object;
  object.emplace("columns", I(telemetry.columns));
  object.emplace("elements", I(telemetry.rows * telemetry.columns));
  object.emplace("error_sum_squares", D(telemetry.error_sum_squares));
  object.emplace("histogram", [&] { Value::Array values; values.reserve(256); for (const auto count : telemetry.histogram) values.emplace_back(I(count)); return Value(std::move(values)); }());
  object.emplace("max_absolute_error", D(telemetry.max_absolute_error));
  object.emplace("reconstruction_sum_squares", D(telemetry.reconstruction_sum_squares));
  object.emplace("rows", I(telemetry.rows));
  object.emplace("saturation_count", I(telemetry.saturation_count));
  object.emplace("scale_max", D(telemetry.scale_max));
  object.emplace("scale_min", D(telemetry.scale_min));
  object.emplace("scale_output_name", S(telemetry.scale_output_name));
  object.emplace("scale_sha256", S(telemetry.scale_sha256));
  object.emplace("source_max", D(telemetry.source_max));
  object.emplace("source_min", D(telemetry.source_min));
  object.emplace("source_name", S(telemetry.source_name));
  object.emplace("source_reconstruction_dot", D(telemetry.source_reconstruction_dot));
  object.emplace("source_sha256", S(telemetry.source_sha256));
  object.emplace("source_sum_squares", D(telemetry.source_sum_squares));
  object.emplace("underflow_clamped_rows", I(telemetry.underflow_clamped_rows));
  object.emplace("weight_output_name", S(telemetry.weight_output_name));
  object.emplace("weight_sha256", S(telemetry.weight_sha256));
  object.emplace("zero_rows", I(telemetry.zero_rows));
  return Value(std::move(object));
}

Status WriteTelemetry(const std::filesystem::path& path, const Job& job, std::uint64_t maximum_row_bytes,
                      const std::vector<MatrixTelemetry>& telemetry, OwnedIdentity* ownership) {
  if (ownership != nullptr) ownership->valid = false;
  Value::Object root;
  root.emplace("contract_id", S("gem16.fp8_attention_rowwise"));
  root.emplace("contract_version", I(1));
  root.emplace("native_build", NativeBuildValue());
  root.emplace("matrices", [&] { Value::Array values; values.reserve(telemetry.size()); for (const auto& matrix : telemetry) values.emplace_back(MatrixValue(matrix)); return Value(std::move(values)); }());
  root.emplace("maximum_source_row_bytes", I(maximum_row_bytes));
  root.emplace("payload_bytes", I(job.payload_bytes));
  root.emplace("schema_version", I(1));
  root.emplace("threads", I(job.threads));
  const std::string serialized = json::Stringify(Value(std::move(root))) + "\n";
#ifndef _WIN32
  const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
  if (fd < 0) return IoFailure("cannot exclusively create telemetry: " + path.string());
  struct stat identity {};
  if (::fstat(fd, &identity) != 0) {
    ::close(fd);
    return IoFailure("cannot inspect created telemetry: " + path.string());
  }
  if (ownership != nullptr) *ownership = OwnedIdentity{static_cast<std::uint64_t>(identity.st_dev), static_cast<std::uint64_t>(identity.st_ino), true};
  std::size_t total = 0;
  while (total < serialized.size()) {
    const ssize_t count = ::write(fd, serialized.data() + total, serialized.size() - total);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) {
      ::close(fd);
      if (ownership != nullptr) { RemoveOwnedPath(path, *ownership); ownership->valid = false; }
      return IoFailure("cannot write telemetry: " + path.string());
    }
    total += static_cast<std::size_t>(count);
  }
  const int sync = ::fsync(fd);
  ::close(fd);
  if (sync != 0) {
    if (ownership != nullptr) { RemoveOwnedPath(path, *ownership); ownership->valid = false; }
    return IoFailure("cannot fsync telemetry: " + path.string());
  }
#else
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) return IoFailure("cannot create telemetry: " + path.string());
  output.write(serialized.data(), static_cast<std::streamsize>(serialized.size()));
  output.flush();
  if (!output) return IoFailure("cannot write telemetry: " + path.string());
#endif
  return Status::Ok();
}

bool Existing(const std::filesystem::path& path) {
  std::error_code error;
  const auto status = std::filesystem::symlink_status(path, error);
  if (!error) return std::filesystem::exists(status);
  return error != std::make_error_code(std::errc::no_such_file_or_directory);
}

Status PrepareOutputs(const std::filesystem::path& payload, const std::filesystem::path& telemetry) {
  if (payload.empty() || telemetry.empty()) return Invalid("output paths must be non-empty");
  std::error_code error;
  if (!payload.parent_path().empty()) std::filesystem::create_directories(payload.parent_path(), error);
  if (error) return IoFailure("cannot create payload parent: " + error.message());
  if (!telemetry.parent_path().empty()) std::filesystem::create_directories(telemetry.parent_path(), error);
  if (error) return IoFailure("cannot create telemetry parent: " + error.message());
  const auto payload_canonical = std::filesystem::weakly_canonical(payload, error);
  if (error) return Invalid("cannot normalize payload path: " + error.message());
  const auto telemetry_canonical = std::filesystem::weakly_canonical(telemetry, error);
  if (error) return Invalid("cannot normalize telemetry path: " + error.message());
  if (payload_canonical == telemetry_canonical) return Invalid("payload and telemetry paths alias");
  if (Existing(payload) || Existing(telemetry)) return Invalid("payload and telemetry outputs must not already exist");
  return Status::Ok();
}

struct CompareRange {
  std::filesystem::path path;
  std::uint64_t offset = 0;
  std::uint64_t bytes = 0;
  std::string sha256;
};

struct CompareMatrix {
  std::string name;
  std::uint64_t layer = 0;
  char role = 0;
  std::uint64_t rows = 0;
  std::uint64_t columns = 0;
  CompareRange left_weight;
  CompareRange left_scale;
  CompareRange right_weight;
  CompareRange right_scale;
};

struct CompareJob {
  std::uint64_t threads = 1;
  std::vector<CompareMatrix> matrices;
};

Result<CompareRange> ParseCompareRange(const Value& value, std::string_view description) {
  if (!value.is_object() || !ExactKeys(value.as_object(), {"path", "offset", "bytes", "sha256"})) {
    return Invalid(std::string(description) + " has an invalid key set");
  }
  const auto& object = value.as_object();
  const Value* path_value = Required(object, "path");
  const Value* offset_value = Required(object, "offset");
  const Value* bytes_value = Required(object, "bytes");
  const Value* hash_value = Required(object, "sha256");
  if (path_value == nullptr || offset_value == nullptr || bytes_value == nullptr || hash_value == nullptr) {
    return Invalid(std::string(description) + " is missing a field");
  }
  auto path = String(*path_value, std::string(description) + ".path");
  auto offset = UInt(*offset_value, std::string(description) + ".offset");
  auto bytes = UInt(*bytes_value, std::string(description) + ".bytes");
  auto hash = String(*hash_value, std::string(description) + ".sha256");
  if (!path.ok()) return path.status();
  if (!offset.ok()) return offset.status();
  if (!bytes.ok()) return bytes.status();
  if (!hash.ok()) return hash.status();
  CompareRange range;
  range.path = std::filesystem::path(std::move(path).value());
  range.offset = offset.value();
  range.bytes = bytes.value();
  range.sha256 = std::move(hash).value();
  if (!range.path.is_absolute() || std::any_of(range.path.begin(), range.path.end(),
                                               [](const auto& part) { return part == ".."; })) {
    return Invalid(std::string(description) + ".path must be absolute and normalized");
  }
  if (!IsHexDigest(range.sha256)) return Invalid(std::string(description) + ".sha256 is not lowercase SHA-256");
  return range;
}

Result<CompareMatrix> ParseCompareMatrix(const Value& value, std::size_t index,
                                         std::set<std::string>& names,
                                         std::set<std::string>& roles) {
  if (!value.is_object() || !ExactKeys(value.as_object(), {
          "name", "layer", "role", "rows", "columns", "left_weight", "left_scale",
          "right_weight", "right_scale"})) {
    return Invalid("compare matrix " + std::to_string(index) + " has an invalid key set");
  }
  const auto& object = value.as_object();
  auto required = [&](std::string_view key) -> const Value* { return Required(object, key); };
  const Value* name_value = required("name");
  const Value* layer_value = required("layer");
  const Value* role_value = required("role");
  const Value* rows_value = required("rows");
  const Value* columns_value = required("columns");
  if (name_value == nullptr || layer_value == nullptr || role_value == nullptr ||
      rows_value == nullptr || columns_value == nullptr) {
    return Invalid("compare matrix fields are incomplete");
  }
  auto name = String(*name_value, "compare matrix.name");
  auto layer = UInt(*layer_value, "compare matrix.layer");
  auto rows = UInt(*rows_value, "compare matrix.rows");
  auto columns = UInt(*columns_value, "compare matrix.columns");
  if (!name.ok()) return name.status();
  if (!layer.ok()) return layer.status();
  if (!rows.ok()) return rows.status();
  if (!columns.ok()) return columns.status();
  if (!role_value->is_string() || role_value->as_string().size() != 1U ||
      std::string("qkvo").find(role_value->as_string()[0]) == std::string::npos) {
    return Invalid("compare matrix.role must be one of q,k,v,o");
  }
  if (layer.value() >= 30U || rows.value() == 0U || rows.value() > kMaxDimension ||
      columns.value() == 0U || columns.value() > kMaxDimension) {
    return Invalid("compare matrix layer or shape is outside M05 bounds");
  }
  const std::string expected_stem = "model.language_model.layers." + std::to_string(layer.value()) +
                                    ".self_attn." + std::string(1, role_value->as_string()[0]) + "_proj";
  if (name.value() != expected_stem && name.value() != expected_stem + ".weight") {
    return Invalid("compare matrix name does not match layer/role: " + name.value());
  }
  auto elements = Product(rows.value(), columns.value(), "compare matrix elements");
  if (!elements.ok()) return elements.status();
  if (elements.value() > kMaxPayloadBytes) return Invalid("compare matrix is too large");
  CompareMatrix matrix;
  matrix.name = std::move(name).value();
  matrix.layer = layer.value();
  matrix.role = role_value->as_string()[0];
  matrix.rows = rows.value();
  matrix.columns = columns.value();
  if (!names.insert(matrix.name).second) return Invalid("duplicate compare matrix name: " + matrix.name);
  const std::string role_key = std::to_string(matrix.layer) + ":" + matrix.role;
  if (!roles.insert(role_key).second) return Invalid("duplicate compare matrix layer/role");
  auto left_weight = ParseCompareRange(*required("left_weight"), "compare matrix.left_weight");
  auto left_scale = ParseCompareRange(*required("left_scale"), "compare matrix.left_scale");
  auto right_weight = ParseCompareRange(*required("right_weight"), "compare matrix.right_weight");
  auto right_scale = ParseCompareRange(*required("right_scale"), "compare matrix.right_scale");
  if (!left_weight.ok()) return left_weight.status();
  if (!left_scale.ok()) return left_scale.status();
  if (!right_weight.ok()) return right_weight.status();
  if (!right_scale.ok()) return right_scale.status();
  matrix.left_weight = std::move(left_weight).value();
  matrix.left_scale = std::move(left_scale).value();
  matrix.right_weight = std::move(right_weight).value();
  matrix.right_scale = std::move(right_scale).value();
  const auto weight_bytes = elements.value();
  const auto scale_bytes = rows.value() * 2U;
  if (matrix.left_weight.bytes != weight_bytes || matrix.right_weight.bytes != weight_bytes ||
      matrix.left_scale.bytes != scale_bytes || matrix.right_scale.bytes != scale_bytes) {
    return Invalid("compare matrix range byte lengths do not match shape");
  }
  return matrix;
}

Result<CompareJob> ParseCompareJob(const std::filesystem::path& path) {
  std::string text;
#ifndef _WIN32
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (fd < 0) return Invalid("cannot open compare job: " + path.string());
  struct stat info {};
  if (::fstat(fd, &info) != 0 || !S_ISREG(info.st_mode)) {
    ::close(fd);
    return Invalid("compare job is not a regular file: " + path.string());
  }
  if (info.st_size < 0 || static_cast<std::uint64_t>(info.st_size) > kMaxJobBytes) {
    ::close(fd);
    return Invalid("compare job JSON exceeds 16 MiB limit");
  }
  text.assign(static_cast<std::size_t>(info.st_size), '\0');
  std::size_t total = 0;
  while (total < text.size()) {
    const ssize_t count = ::pread(fd, text.data() + total, text.size() - total,
                                  static_cast<off_t>(total));
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) {
      ::close(fd);
      return IoFailure("compare job changed while reading: " + path.string());
    }
    total += static_cast<std::size_t>(count);
  }
  ::close(fd);
#else
  std::error_code error;
  const auto size = std::filesystem::file_size(path, error);
  if (error) return Invalid("cannot inspect compare job: " + path.string());
  if (size > kMaxJobBytes) return Invalid("compare job JSON exceeds 16 MiB limit");
  std::ifstream input(path, std::ios::binary);
  if (!input) return IoFailure("cannot open compare job: " + path.string());
  text.assign(static_cast<std::size_t>(size), '\0');
  if (size != 0U) input.read(text.data(), static_cast<std::streamsize>(size));
  if (static_cast<std::uint64_t>(input.gcount()) != size) return IoFailure("compare job changed while reading: " + path.string());
  input.peek();
  if (!input.eof()) return IoFailure("compare job changed while reading: " + path.string());
#endif
  auto parsed = json::Parse(text);
  if (!parsed.ok()) return Invalid("invalid compare job JSON: " + parsed.status().message());
  if (!parsed.value().is_object() || !ExactKeys(parsed.value().as_object(), {
          "schema_version", "contract_id", "contract_version", "threads", "matrices"})) {
    return Invalid("compare job has an invalid root key set");
  }
  const auto& object = parsed.value().as_object();
  const Value* schema = Required(object, "schema_version");
  const Value* contract = Required(object, "contract_id");
  const Value* version = Required(object, "contract_version");
  const Value* threads = Required(object, "threads");
  const Value* matrices = Required(object, "matrices");
  if (schema == nullptr || !schema->is_integer() || schema->as_integer() != 1 ||
      contract == nullptr || !contract->is_string() || contract->as_string() != "gem16.fp8_attention_compare" ||
      version == nullptr || !version->is_integer() || version->as_integer() != 1 ||
      threads == nullptr || matrices == nullptr) {
    return Invalid("compare job contract/version is unsupported");
  }
  auto thread_count = UInt(*threads, "compare job.threads");
  if (!thread_count.ok()) return thread_count.status();
  if (thread_count.value() == 0U || thread_count.value() > 64U) return Invalid("compare job.threads must be between 1 and 64");
  if (!matrices->is_array() || matrices->as_array().empty() || matrices->as_array().size() > kMaxCompareMatrices) {
    return Invalid("compare job.matrices must contain 1..115 matrices");
  }
  CompareJob job;
  job.threads = thread_count.value();
  std::set<std::string> names;
  std::set<std::string> roles;
  job.matrices.reserve(matrices->as_array().size());
  for (std::size_t index = 0; index < matrices->as_array().size(); ++index) {
    auto matrix = ParseCompareMatrix(matrices->as_array()[index], index, names, roles);
    if (!matrix.ok()) return matrix.status();
    job.matrices.push_back(std::move(matrix).value());
  }
  if (job.matrices.size() == 115U) {
    const std::set<char> required_roles = {'q', 'k', 'o'};
    for (std::uint64_t layer = 0; layer < 30U; ++layer) {
      for (const char role : required_roles) {
        if (!roles.contains(std::to_string(layer) + ":" + role)) return Invalid("production compare job misses Q/K/O projection");
      }
      const bool global = layer == 5U || layer == 11U || layer == 17U || layer == 23U || layer == 29U;
      if (global) {
        if (roles.contains(std::to_string(layer) + ":v")) return Invalid("production compare job contains forbidden global V projection");
      } else if (!roles.contains(std::to_string(layer) + ":v")) {
        return Invalid("production compare job misses local V projection");
      }
    }
  }
  return job;
}

#ifndef _WIN32
class CompareRangeFile {
 public:
  ~CompareRangeFile() { if (fd_ >= 0) ::close(fd_); }
  CompareRangeFile(const CompareRangeFile&) = delete;
  CompareRangeFile& operator=(const CompareRangeFile&) = delete;
  CompareRangeFile(CompareRangeFile&& other) noexcept : fd_(std::exchange(other.fd_, -1)) {}
  CompareRangeFile& operator=(CompareRangeFile&& other) noexcept {
    if (this != &other) {
      if (fd_ >= 0) ::close(fd_);
      fd_ = std::exchange(other.fd_, -1);
    }
    return *this;
  }
  static Result<CompareRangeFile> Open(const CompareRange& range) {
    const int fd = ::open(range.path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) return SourceFailure("cannot open comparison range: " + range.path.string());
    CompareRangeFile file(fd);
    struct stat info {};
    if (::fstat(fd, &info) != 0 || !S_ISREG(info.st_mode)) return SourceFailure("comparison range is not a regular file: " + range.path.string());
    const auto max_offset = static_cast<std::uint64_t>(std::numeric_limits<off_t>::max());
    if (range.offset > static_cast<std::uint64_t>(info.st_size) || range.bytes > static_cast<std::uint64_t>(info.st_size) - range.offset ||
        range.offset > max_offset || range.bytes > max_offset - range.offset) {
      return SourceFailure("comparison range is outside its descriptor: " + range.path.string());
    }
    return file;
  }
  Status Read(std::uint64_t offset, std::uint8_t* data, std::size_t bytes) const {
    if (offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) return SourceFailure("comparison offset exceeds off_t");
    std::size_t total = 0;
    while (total < bytes) {
      const ssize_t count = ::pread(fd_, data + total, bytes - total, static_cast<off_t>(offset + total));
      if (count < 0) {
        if (errno == EINTR) continue;
        return SourceFailure("cannot read comparison range");
      }
      if (count == 0) return SourceFailure("short comparison range read");
      total += static_cast<std::size_t>(count);
    }
    return Status::Ok();
  }
 private:
  explicit CompareRangeFile(int fd) : fd_(fd) {}
  int fd_ = -1;
};
#endif

struct Neumaier {
  double total = 0.0;
  double correction = 0.0;
  void Add(double value) {
    const double trial = total + value;
    if (std::fabs(total) >= std::fabs(value)) correction += (total - trial) + value;
    else correction += (value - trial) + total;
    total = trial;
  }
  [[nodiscard]] double Value() const { return total + correction; }
};

Result<Value> CompareMatrixValues(const CompareMatrix& matrix) {
#ifdef _WIN32
  (void)matrix;
  return Status(StatusCode::kUnsupported, "native comparison requires descriptor I/O on Linux");
#else
  auto left_weight_result = CompareRangeFile::Open(matrix.left_weight);
  auto left_scale_result = CompareRangeFile::Open(matrix.left_scale);
  auto right_weight_result = CompareRangeFile::Open(matrix.right_weight);
  auto right_scale_result = CompareRangeFile::Open(matrix.right_scale);
  if (!left_weight_result.ok()) return left_weight_result.status();
  if (!left_scale_result.ok()) return left_scale_result.status();
  if (!right_weight_result.ok()) return right_weight_result.status();
  if (!right_scale_result.ok()) return right_scale_result.status();
  auto left_weight = std::move(left_weight_result).value();
  auto left_scale = std::move(left_scale_result).value();
  auto right_weight = std::move(right_weight_result).value();
  auto right_scale = std::move(right_scale_result).value();
  std::vector<float> left_scales(static_cast<std::size_t>(matrix.rows));
  std::vector<float> right_scales(static_cast<std::size_t>(matrix.rows));
  std::vector<std::uint16_t> left_scale_bits(static_cast<std::size_t>(matrix.rows));
  std::vector<std::uint16_t> right_scale_bits(static_cast<std::size_t>(matrix.rows));
  const std::size_t chunk = static_cast<std::size_t>(matrix.columns);
  std::vector<std::uint8_t> left_buffer(chunk), right_buffer(chunk), left_scale_bytes(2U), right_scale_bytes(2U);
  Sha256 left_weight_hash, right_weight_hash, left_scale_hash, right_scale_hash;
  Neumaier left_scale_sum, right_scale_sum, left_scale_energy, right_scale_energy, scale_difference, scale_dot;
  double left_scale_min = std::numeric_limits<double>::infinity();
  double left_scale_max = -std::numeric_limits<double>::infinity();
  double right_scale_min = std::numeric_limits<double>::infinity();
  double right_scale_max = -std::numeric_limits<double>::infinity();
  std::uint64_t scale_mismatches = 0;
  for (std::uint64_t row = 0; row < matrix.rows; ++row) {
    const std::uint64_t offset = matrix.left_scale.offset + row * 2U;
    auto status = left_scale.Read(offset, left_scale_bytes.data(), 2U);
    if (!status.ok()) return status;
    left_scale_hash.Update(left_scale_bytes.data(), 2U);
    status = right_scale.Read(matrix.right_scale.offset + row * 2U, right_scale_bytes.data(), 2U);
    if (!status.ok()) return status;
    right_scale_hash.Update(right_scale_bytes.data(), 2U);
    const std::uint16_t left_bits = static_cast<std::uint16_t>(left_scale_bytes[0]) | static_cast<std::uint16_t>(left_scale_bytes[1] << 8U);
    const std::uint16_t right_bits = static_cast<std::uint16_t>(right_scale_bytes[0]) | static_cast<std::uint16_t>(right_scale_bytes[1] << 8U);
    const float left_value = DecodeBf16(left_bits);
    const float right_value = DecodeBf16(right_bits);
    if (!std::isfinite(left_value) || left_value <= 0.0F || !std::isfinite(right_value) || right_value <= 0.0F) {
      return NumericFailure("comparison scale must be finite and positive: " + matrix.name);
    }
    left_scales[static_cast<std::size_t>(row)] = left_value;
    right_scales[static_cast<std::size_t>(row)] = right_value;
    left_scale_bits[static_cast<std::size_t>(row)] = left_bits;
    right_scale_bits[static_cast<std::size_t>(row)] = right_bits;
    scale_mismatches += left_bits != right_bits;
    left_scale_min = std::min(left_scale_min, static_cast<double>(left_value));
    left_scale_max = std::max(left_scale_max, static_cast<double>(left_value));
    right_scale_min = std::min(right_scale_min, static_cast<double>(right_value));
    right_scale_max = std::max(right_scale_max, static_cast<double>(right_value));
    left_scale_sum.Add(left_value); right_scale_sum.Add(right_value);
    left_scale_energy.Add(static_cast<double>(left_value) * left_value);
    right_scale_energy.Add(static_cast<double>(right_value) * right_value);
    scale_difference.Add((static_cast<double>(left_value) - right_value) * (static_cast<double>(left_value) - right_value));
    scale_dot.Add(static_cast<double>(left_value) * right_value);
  }
  Neumaier left_energy, right_energy, difference_energy, reconstruction_dot;
  double left_min = std::numeric_limits<double>::infinity();
  double left_max = -std::numeric_limits<double>::infinity();
  double right_min = std::numeric_limits<double>::infinity();
  double right_max = -std::numeric_limits<double>::infinity();
  double max_error = 0.0;
  std::uint64_t raw_mismatches = 0, left_7e = 0, left_fe = 0, right_7e = 0, right_fe = 0, left_nan = 0, right_nan = 0;
  for (std::uint64_t row = 0; row < matrix.rows; ++row) {
    const float left_scale_value = left_scales[static_cast<std::size_t>(row)];
    const float right_scale_value = right_scales[static_cast<std::size_t>(row)];
    for (std::uint64_t column = 0; column < matrix.columns; column += chunk) {
      const std::size_t count = static_cast<std::size_t>(std::min<std::uint64_t>(chunk, matrix.columns - column));
      auto status = left_weight.Read(matrix.left_weight.offset + row * matrix.columns + column, left_buffer.data(), count);
      if (!status.ok()) return status;
      left_weight_hash.Update(left_buffer.data(), count);
      status = right_weight.Read(matrix.right_weight.offset + row * matrix.columns + column, right_buffer.data(), count);
      if (!status.ok()) return status;
      right_weight_hash.Update(right_buffer.data(), count);
      for (std::size_t position = 0; position < count; ++position) {
        const std::uint8_t left_code = left_buffer[position];
        const std::uint8_t right_code = right_buffer[position];
        raw_mismatches += left_code != right_code;
        left_7e += left_code == 0x7EU; left_fe += left_code == 0xFEU;
        right_7e += right_code == 0x7EU; right_fe += right_code == 0xFEU;
        const bool left_is_nan = (left_code & 0x7FU) == kE4M3NaN;
        const bool right_is_nan = (right_code & 0x7FU) == kE4M3NaN;
        left_nan += left_is_nan; right_nan += right_is_nan;
        if (left_is_nan || right_is_nan) return NumericFailure("comparison FP8 range contains NaN code: " + matrix.name);
        const auto& decode_values = CompareDecodeValues();
        const float left_value_f = decode_values[left_code] * left_scale_value;
        const float right_value_f = decode_values[right_code] * right_scale_value;
        if (!std::isfinite(left_value_f) || !std::isfinite(right_value_f)) return NumericFailure("comparison reconstruction is non-finite: " + matrix.name);
        const double left_value = left_value_f, right_value = right_value_f;
        const double error = left_value - right_value;
        if (!std::isfinite(error)) return NumericFailure("comparison error is non-finite: " + matrix.name);
        left_min = std::min(left_min, left_value); left_max = std::max(left_max, left_value);
        right_min = std::min(right_min, right_value); right_max = std::max(right_max, right_value);
        left_energy.Add(left_value * left_value); right_energy.Add(right_value * right_value);
        difference_energy.Add(error * error); reconstruction_dot.Add(left_value * right_value);
        max_error = std::max(max_error, std::fabs(error));
      }
    }
  }
  const auto left_weight_digest = left_weight_hash.HexDigest();
  const auto right_weight_digest = right_weight_hash.HexDigest();
  const auto left_scale_digest = left_scale_hash.HexDigest();
  const auto right_scale_digest = right_scale_hash.HexDigest();
  if (left_weight_digest != matrix.left_weight.sha256 || right_weight_digest != matrix.right_weight.sha256 ||
      left_scale_digest != matrix.left_scale.sha256 || right_scale_digest != matrix.right_scale.sha256) {
    return SourceFailure("comparison range SHA-256 mismatch: " + matrix.name);
  }
  const double left_scale_sum_value = left_scale_sum.Value();
  const double right_scale_sum_value = right_scale_sum.Value();
  const double left_scale_energy_value = left_scale_energy.Value();
  const double right_scale_energy_value = right_scale_energy.Value();
  const double scale_difference_value = scale_difference.Value();
  const double scale_dot_value = scale_dot.Value();
  const double left_energy_value = left_energy.Value();
  const double right_energy_value = right_energy.Value();
  const double difference_value = difference_energy.Value();
  const double dot_value = reconstruction_dot.Value();
  if (!std::isfinite(left_scale_sum_value) || !std::isfinite(right_scale_sum_value) || !std::isfinite(left_scale_energy_value) ||
      !std::isfinite(right_scale_energy_value) || !std::isfinite(scale_difference_value) || !std::isfinite(scale_dot_value) ||
      !std::isfinite(left_energy_value) || !std::isfinite(right_energy_value) || !std::isfinite(difference_value) || !std::isfinite(dot_value)) {
    return NumericFailure("comparison reduction is non-finite: " + matrix.name);
  }
  const double scale_left_variance = static_cast<double>(matrix.rows) * left_scale_energy_value - left_scale_sum_value * left_scale_sum_value;
  const double scale_right_variance = static_cast<double>(matrix.rows) * right_scale_energy_value - right_scale_sum_value * right_scale_sum_value;
  const double scale_covariance = static_cast<double>(matrix.rows) * scale_dot_value - left_scale_sum_value * right_scale_sum_value;
  const bool left_constant = std::all_of(left_scale_bits.begin(), left_scale_bits.end(),
                                         [&](std::uint16_t value) { return value == left_scale_bits.front(); });
  const bool right_constant = std::all_of(right_scale_bits.begin(), right_scale_bits.end(),
                                          [&](std::uint16_t value) { return value == right_scale_bits.front(); });
  double pearson = 0.0;
  if (left_constant || right_constant) {
    pearson = left_constant && right_constant && left_scale_bits == right_scale_bits ? 1.0 : 0.0;
  } else if (scale_left_variance <= 0.0 || scale_right_variance <= 0.0) {
    pearson = scale_mismatches == 0U ? 1.0 : 0.0;
  } else {
    pearson = std::clamp(scale_covariance / std::sqrt(scale_left_variance * scale_right_variance), -1.0, 1.0);
  }
  const auto relative = [](double error, double reference) -> Result<Value> {
    if (reference == 0.0) return error == 0.0 ? Result<Value>(D(0.0)) : Result<Value>(Value(nullptr));
    const double value = std::sqrt(error / reference);
    if (!std::isfinite(value)) return NumericFailure("comparison relative L2 is non-finite");
    return D(value);
  };
  auto reconstruction_relative = relative(difference_value, right_energy_value);
  auto scale_relative = relative(scale_difference_value, right_scale_energy_value);
  if (!reconstruction_relative.ok()) return reconstruction_relative.status();
  if (!scale_relative.ok()) return scale_relative.status();
  const auto cosine = [&] {
    if (left_energy_value == 0.0 && right_energy_value == 0.0) return 1.0;
    if (left_energy_value == 0.0 || right_energy_value == 0.0) return 0.0;
    return std::clamp(dot_value / std::sqrt(left_energy_value * right_energy_value), -1.0, 1.0);
  }();
  Value::Object result;
  result.emplace("columns", I(matrix.columns));
  result.emplace("cosine_similarity", D(cosine));
  result.emplace("difference_sum_squares", D(difference_value));
  result.emplace("elements", I(matrix.rows * matrix.columns));
  result.emplace("layer", I(matrix.layer));
  result.emplace("left_endpoint_7e", I(left_7e)); result.emplace("left_endpoint_fe", I(left_fe)); result.emplace("left_nan_count", I(left_nan));
  result.emplace("left_max", D(left_max)); result.emplace("left_min", D(left_min)); result.emplace("left_scale_max", D(left_scale_max)); result.emplace("left_scale_min", D(left_scale_min));
  result.emplace("left_scale_sha256", S(left_scale_digest)); result.emplace("left_scale_sum", D(left_scale_sum_value)); result.emplace("left_scale_sum_squares", D(left_scale_energy_value));
  result.emplace("left_sum_squares", D(left_energy_value)); result.emplace("left_weight_sha256", S(left_weight_digest));
  result.emplace("max_absolute_error", D(max_error));
  result.emplace("name", S(matrix.name));
  result.emplace("perfect_reconstruction", difference_value == 0.0);
  result.emplace("raw_mismatch_count", I(raw_mismatches));
  result.emplace("reconstruction_dot", D(dot_value)); result.emplace("reconstruction_relative_l2", std::move(reconstruction_relative).value());
  result.emplace("right_endpoint_7e", I(right_7e)); result.emplace("right_endpoint_fe", I(right_fe)); result.emplace("right_nan_count", I(right_nan));
  result.emplace("right_max", D(right_max)); result.emplace("right_min", D(right_min)); result.emplace("right_scale_max", D(right_scale_max)); result.emplace("right_scale_min", D(right_scale_min));
  result.emplace("right_scale_sha256", S(right_scale_digest)); result.emplace("right_scale_sum", D(right_scale_sum_value)); result.emplace("right_scale_sum_squares", D(right_scale_energy_value));
  result.emplace("right_sum_squares", D(right_energy_value)); result.emplace("right_weight_sha256", S(right_weight_digest));
  result.emplace("role", S(std::string(1, matrix.role)));
  result.emplace("rows", I(matrix.rows));
  result.emplace("scale_difference_sum_squares", D(scale_difference_value)); result.emplace("scale_dot", D(scale_dot_value));
  result.emplace("scale_mismatch_count", I(scale_mismatches)); result.emplace("scale_pearson_correlation", D(pearson));
  result.emplace("scale_relative_l2", std::move(scale_relative).value());
  if (difference_value == 0.0 || right_energy_value == 0.0) result.emplace("sqnr_db", Value(nullptr));
  else {
    const double sqnr = 10.0 * std::log10(right_energy_value / difference_value);
    if (!std::isfinite(sqnr)) return NumericFailure("comparison SQNR is non-finite");
    result.emplace("sqnr_db", D(sqnr));
  }
  result.emplace("zero_reference", right_energy_value == 0.0);
  return Value(std::move(result));
#endif
}

Status WriteCompareMetrics(const std::filesystem::path& path, const CompareJob& job,
                           std::vector<Value>& results, std::uint64_t maximum_chunk_bytes) {
  Value::Array matrices;
  matrices.reserve(results.size());
  for (auto& result : results) matrices.emplace_back(std::move(result));
  Value::Object root;
  root.emplace("contract_id", S("gem16.fp8_attention_compare"));
  root.emplace("contract_version", I(1));
  root.emplace("native_build", NativeBuildValue());
  root.emplace("matrices", Value(std::move(matrices)));
  root.emplace("maximum_chunk_bytes", I(maximum_chunk_bytes));
  root.emplace("schema_version", I(1));
  root.emplace("threads", I(job.threads));
  const std::string serialized = json::Stringify(Value(std::move(root))) + "\n";
#ifdef _WIN32
  (void)path; (void)serialized;
  return Status(StatusCode::kUnsupported, "native comparison requires descriptor I/O on Linux");
#else
  const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
  if (fd < 0) return IoFailure("cannot exclusively create comparison metrics: " + path.string());
  struct stat identity {};
  if (::fstat(fd, &identity) != 0) { ::close(fd); return IoFailure("cannot inspect comparison metrics"); }
  const OwnedIdentity ownership{static_cast<std::uint64_t>(identity.st_dev), static_cast<std::uint64_t>(identity.st_ino), true};
  auto cleanup = [&] { RemoveOwnedPath(path, ownership); };
  std::size_t total = 0;
  while (total < serialized.size()) {
    const ssize_t count = ::write(fd, serialized.data() + total, serialized.size() - total);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) { ::close(fd); cleanup(); return IoFailure("cannot write comparison metrics"); }
    total += static_cast<std::size_t>(count);
  }
  if (::fsync(fd) != 0) { ::close(fd); cleanup(); return IoFailure("cannot fsync comparison metrics"); }
  ::close(fd);
  const auto parent = path.parent_path().empty() ? std::filesystem::path(".") : path.parent_path();
  const auto parent_status = SyncParent(parent);
  if (!parent_status.ok()) { cleanup(); return parent_status; }
  return Status::Ok();
#endif
}

}  // namespace

float DecodeE4M3Fn(std::uint8_t bits) noexcept {
  const std::uint8_t magnitude = bits & 0x7FU;
  if (magnitude == kE4M3NaN) return std::numeric_limits<float>::quiet_NaN();
  const std::uint32_t exponent = (magnitude >> 3U) & 0xFU;
  const std::uint32_t mantissa = magnitude & 7U;
  const float value = exponent == 0U ? std::ldexp(static_cast<float>(mantissa), -9) : std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F, static_cast<int>(exponent) - 7);
  return (bits & 0x80U) == 0U ? value : -value;
}

Result<std::uint8_t> EncodeE4M3Fn(float value) {
  if (!std::isfinite(value)) return Invalid("E4M3FN input must be finite");
  const bool negative = std::signbit(value);
  const float magnitude = std::fabs(value);
  std::uint8_t code = 0;
  if (magnitude >= kE4M3Max) code = 0x7EU;
  else {
    const auto& midpoints = PositiveMidpoints();
    const auto iterator = std::lower_bound(midpoints.begin(), midpoints.end(), magnitude);
    const auto index = static_cast<std::size_t>(iterator - midpoints.begin());
    code = static_cast<std::uint8_t>(iterator != midpoints.end() && magnitude == *iterator && (index & 1U) != 0U ? index + 1U : index);
  }
  return static_cast<std::uint8_t>(code | (negative ? 0x80U : 0U));
}

Result<std::uint16_t> RoundBf16Rne(float value) {
  if (!std::isfinite(value) || value < 0.0F) return Invalid("BF16 RNE input must be finite and non-negative");
  return Bf16RneUnchecked(value);
}

int ExitCodeForStatus(StatusCode code) noexcept {
  switch (code) {
    case StatusCode::kInvalidArgument: return 2;
    case StatusCode::kDataLoss: return 3;
    case StatusCode::kInternal:
    case StatusCode::kResourceExhausted: return 4;
    case StatusCode::kIoError: return 5;
    default: return 4;
  }
}

Status EncodeJobFile(const std::filesystem::path& job_path, const std::filesystem::path& payload_path, const std::filesystem::path& telemetry_path) {
#ifdef _WIN32
  (void)job_path; (void)payload_path; (void)telemetry_path;
  return Status(StatusCode::kUnsupported, "native M05 compiler requires Linux descriptor I/O");
#else
  std::optional<OutputFile> output;
  OwnedIdentity telemetry_identity;
  const auto cleanup_owned_outputs = [&] {
    if (output.has_value()) output->RemoveIfOwned();
    if (telemetry_identity.valid) {
      RemoveOwnedPath(telemetry_path, telemetry_identity);
      telemetry_identity.valid = false;
    }
  };
  try {
    if (std::fegetround() != FE_TONEAREST && std::fesetround(FE_TONEAREST) != 0) return NumericFailure("native FP8 compiler requires FE_TONEAREST");
    if (std::fegetround() != FE_TONEAREST) return NumericFailure("native FP8 compiler could not establish FE_TONEAREST");
    auto parsed = ParseJob(job_path);
    if (!parsed.ok()) return parsed.status();
    const Job job = std::move(parsed).value();
    auto output_status = PrepareOutputs(payload_path, telemetry_path);
    if (!output_status.ok()) return output_status;
    auto output_result = OutputFile::Create(payload_path, job.payload_bytes);
    if (!output_result.ok()) return output_result.status();
    output.emplace(std::move(output_result).value());
    std::vector<MatrixTelemetry> telemetry(job.matrices.size());
    std::vector<Status> statuses(job.matrices.size(), Status::Ok());
    std::atomic<std::size_t> next{0};
    std::atomic<bool> failed{false};
    std::mutex status_mutex;
    const std::size_t workers = std::min<std::size_t>(job.threads, job.matrices.size());
    std::vector<std::thread> threads;
    threads.reserve(workers);
    try {
      for (std::size_t index = 0; index < workers; ++index) {
        threads.emplace_back([&] {
          try {
            while (!failed.load(std::memory_order_relaxed)) {
              const std::size_t matrix_index = next.fetch_add(1, std::memory_order_relaxed);
              if (matrix_index >= job.matrices.size()) break;
              const Status status = EncodeMatrix(job.matrices[matrix_index], *output, telemetry[matrix_index]);
              if (!status.ok()) {
                std::lock_guard lock(status_mutex);
                statuses[matrix_index] = status;
                failed.store(true, std::memory_order_relaxed);
                break;
              }
            }
          } catch (const std::exception& exception) {
            std::lock_guard lock(status_mutex);
            statuses[0] = NumericFailure(std::string("native worker exception: ") + exception.what());
            failed.store(true, std::memory_order_relaxed);
          } catch (...) {
            std::lock_guard lock(status_mutex);
            statuses[0] = NumericFailure("native worker exception");
            failed.store(true, std::memory_order_relaxed);
          }
        });
      }
    } catch (const std::exception& exception) {
      failed.store(true);
      for (auto& thread : threads) if (thread.joinable()) thread.join();
      cleanup_owned_outputs();
      return IoFailure(std::string("cannot create native worker: ") + exception.what());
    }
    for (auto& thread : threads) thread.join();
    if (failed.load()) {
      cleanup_owned_outputs();
      for (const auto& status : statuses) if (!status.ok()) return status;
      return NumericFailure("native FP8 matrix conversion failed");
    }
    const auto widest = std::max_element(job.matrices.begin(), job.matrices.end(), [](const MatrixJob& left, const MatrixJob& right) { return left.columns < right.columns; });
    auto sync_status = output->Sync();
    if (!sync_status.ok()) { cleanup_owned_outputs(); return sync_status; }
    auto telemetry_status = WriteTelemetry(telemetry_path, job, widest->columns * 2U, telemetry, &telemetry_identity);
    if (!telemetry_status.ok()) { cleanup_owned_outputs(); return telemetry_status; }
    auto parent_status = SyncParent(payload_path.parent_path().empty() ? std::filesystem::path(".") : payload_path.parent_path());
    if (!parent_status.ok()) { cleanup_owned_outputs(); return parent_status; }
    parent_status = SyncParent(telemetry_path.parent_path().empty() ? std::filesystem::path(".") : telemetry_path.parent_path());
    if (!parent_status.ok()) { cleanup_owned_outputs(); return parent_status; }
    return Status::Ok();
  } catch (const std::bad_alloc&) {
    cleanup_owned_outputs();
    return Status(StatusCode::kResourceExhausted, "native FP8 compiler allocation limit exceeded");
  } catch (const std::exception& exception) {
    cleanup_owned_outputs();
    return IoFailure(std::string("native FP8 compiler exception: ") + exception.what());
  }
#endif
}

Status CompareJobFile(const std::filesystem::path& job_path,
                      const std::filesystem::path& metrics_path) {
#ifdef _WIN32
  (void)job_path;
  (void)metrics_path;
  return Status(StatusCode::kUnsupported, "native comparison requires Linux descriptor I/O");
#else
  try {
    if (std::fegetround() != FE_TONEAREST && std::fesetround(FE_TONEAREST) != 0) {
      return NumericFailure("native comparison requires FE_TONEAREST");
    }
    if (std::fegetround() != FE_TONEAREST) return NumericFailure("native comparison could not establish FE_TONEAREST");
    auto parsed = ParseCompareJob(job_path);
    if (!parsed.ok()) return parsed.status();
    const CompareJob job = std::move(parsed).value();
    if (metrics_path.empty()) return Invalid("comparison metrics path must be non-empty");
    if (Existing(metrics_path)) return Invalid("comparison metrics output must not already exist");
    std::vector<std::optional<Value>> results(job.matrices.size());
    std::vector<Status> statuses(job.matrices.size(), Status::Ok());
    std::atomic<std::size_t> next{0};
    std::atomic<bool> failed{false};
    std::mutex status_mutex;
    const std::size_t workers = std::min<std::size_t>(job.threads, job.matrices.size());
    std::vector<std::thread> threads;
    threads.reserve(workers);
    try {
      for (std::size_t index = 0; index < workers; ++index) {
        threads.emplace_back([&] {
          try {
            while (!failed.load(std::memory_order_relaxed)) {
              const std::size_t matrix_index = next.fetch_add(1, std::memory_order_relaxed);
              if (matrix_index >= job.matrices.size()) break;
              auto result = CompareMatrixValues(job.matrices[matrix_index]);
              if (!result.ok()) {
                std::lock_guard lock(status_mutex);
                statuses[matrix_index] = result.status();
                failed.store(true, std::memory_order_relaxed);
                break;
              }
              results[matrix_index] = std::move(result).value();
            }
          } catch (const std::bad_alloc&) {
            std::lock_guard lock(status_mutex);
            statuses[0] = Status(StatusCode::kResourceExhausted, "native comparison allocation limit exceeded");
            failed.store(true, std::memory_order_relaxed);
          } catch (const std::exception& exception) {
            std::lock_guard lock(status_mutex);
            statuses[0] = NumericFailure(std::string("native comparison worker exception: ") + exception.what());
            failed.store(true, std::memory_order_relaxed);
          } catch (...) {
            std::lock_guard lock(status_mutex);
            statuses[0] = NumericFailure("native comparison worker exception");
            failed.store(true, std::memory_order_relaxed);
          }
        });
      }
    } catch (const std::exception& exception) {
      failed.store(true);
      for (auto& thread : threads) if (thread.joinable()) thread.join();
      return IoFailure(std::string("cannot create native comparison worker: ") + exception.what());
    }
    for (auto& thread : threads) thread.join();
    if (failed.load()) {
      for (const auto& status : statuses) if (!status.ok()) return status;
      return NumericFailure("native comparison failed");
    }
    std::vector<Value> complete;
    complete.reserve(results.size());
    std::uint64_t maximum_chunk_bytes = 0;
    for (std::size_t index = 0; index < results.size(); ++index) {
      if (!results[index].has_value()) return NumericFailure("native comparison did not produce all matrix results");
      complete.emplace_back(std::move(*results[index]));
      maximum_chunk_bytes = std::max(maximum_chunk_bytes, job.matrices[index].columns);
    }
    return WriteCompareMetrics(metrics_path, job, complete, maximum_chunk_bytes);
  } catch (const std::bad_alloc&) {
    return Status(StatusCode::kResourceExhausted, "native comparison allocation limit exceeded");
  } catch (const std::exception& exception) {
    return IoFailure(std::string("native comparison exception: ") + exception.what());
  }
#endif
}

}  // namespace gem16::compiler
