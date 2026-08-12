#include "gem16/nvfp4_head.h"

#include "compiler/sha256.h"
#include "cli/nvfp4_head_diagnostic_reference.h"
#include "model/safetensors.h"
#include "platform/mapped_file.h"
#include "util/json.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#ifndef _WIN32
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {
using gem16::Status;
using gem16::StatusCode;
using gem16::Result;
using gem16::json::Value;

constexpr std::uint64_t kVocabulary = 262144U;
constexpr std::uint64_t kHidden = 2816U;
constexpr std::uint64_t kPackedBytes = 369098752U;
constexpr std::uint64_t kScaleBytes = 46137344U;
constexpr std::uint64_t kScalarBytes = 4U;
constexpr std::array<std::uint32_t, 4> kLookupTokens = {0U, 1U, 131072U, 262143U};
constexpr std::array<float, 8> kE2M1 = {0.0F, 0.5F, 1.0F, 1.5F,
                                          2.0F, 3.0F, 4.0F, 6.0F};
constexpr std::string_view kProfile = "nvfp4-tied-head-partial-v1";
constexpr std::string_view kArtifactStatus =
    "m07_nvfp4_tied_head_partial_not_runtime_loadable";
constexpr std::string_view kSourceLockSha256 =
    "3d9441fdebef33785e33181c335338208b8bf868cb8c7da692fd9c765cca8230";
constexpr std::string_view kPlanSha256 =
    "e549a43864e2e64b4b0783de2337631c5b5989fb3c25f0dc94b762442ded6c27";
constexpr std::string_view kDependencyLockSha256 =
    "a6df4681ee1d2cb34a914e0a4aabe492daf8f12d50b80c70f98bdd7b7389de37";
constexpr std::string_view kSourceRepository =
    "google/gemma-4-26B-A4B-it-qat-q4_0-unquantized";
constexpr std::string_view kSourceRevision =
    "f1e06dc520982d9b9edd76859fdb7ab209449949";
constexpr std::string_view kSourceResolvedAt = "2026-08-06T10:25:52Z";
constexpr std::string_view kSourceTensorSha256 =
    "b2527415ab8a8881bf78ad11832e3e417d8e112479c6fdfb490d926c28788ff2";
constexpr std::string_view kSourceShardSha256 =
    "d57e8bde0feda8b0ebf51a81d04b5d988f82fead7f9242d9231625929421453e";
constexpr std::string_view kSourceRangeIdentitySha256 =
    "72d9b01b939a88bf25cb96c4c1df7afa1d86beb5959bca6cf549b0dbc3a93ae6";
constexpr std::string_view kFileHashScope =
    "all artifact files except gem16_compilation.json; its self-hash is supplied by the external artifact lock in M08";
constexpr std::uint64_t kTargetShardBytes = 1U << 30U;
constexpr std::uint64_t kMaxManifestBytes = 64U * 1024U * 1024U;
constexpr std::uint64_t kCanonicalIndexBytes = 448U;
constexpr std::uint64_t kCanonicalShardBytes = 415236648U;

Status Invalid(std::string message) {
  return Status(StatusCode::kInvalidArgument, std::move(message));
}
Status DataLoss(std::string message) {
  return Status(StatusCode::kDataLoss, std::move(message));
}
Status Io(std::string message) {
  return Status(StatusCode::kIoError, std::move(message));
}

bool AddFits(std::uint64_t first, std::uint64_t second, std::uint64_t* result) {
  if (first > std::numeric_limits<std::uint64_t>::max() - second) return false;
  *result = first + second;
  return true;
}

bool IsFinitePositive(float value) {
  return std::isfinite(value) && value > 0.0F;
}

std::uint8_t Nibble(std::span<const std::uint8_t> bytes, std::uint64_t index) {
  const std::uint8_t byte = bytes[static_cast<std::size_t>(index / 2U)];
  return static_cast<std::uint8_t>((byte >> ((index & 1U) == 0U ? 0U : 4U)) & 0x0FU);
}

float DecodeE2M1Manual(std::uint8_t code) {
  const float magnitude = kE2M1[code & 7U];
  return (code & 8U) == 0U ? magnitude : -magnitude;
}

float DecodeE4M3Manual(std::uint8_t code) {
  const int exponent = static_cast<int>((code >> 3U) & 0x0FU);
  const int mantissa = static_cast<int>(code & 7U);
  const float magnitude = exponent == 0
                              ? std::ldexp(static_cast<float>(mantissa), -9)
                              : std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F,
                                           exponent - 7);
  return (code & 0x80U) == 0U ? magnitude : -magnitude;
}

float Bf16Rne(float value) {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  const std::uint32_t rounded = bits + 0x7FFFU + ((bits >> 16U) & 1U);
  return std::bit_cast<float>(rounded & 0xFFFF0000U);
}

std::uint8_t EncodeE2M1Manual(float value) {
  const float magnitude = std::fabs(value);
  std::uint8_t best = 0U;
  float best_error = std::numeric_limits<float>::infinity();
  for (std::uint8_t candidate = 0U; candidate < 8U; ++candidate) {
    const float error = std::fabs(magnitude - kE2M1[candidate]);
    if (error < best_error ||
        (error == best_error && (candidate & 1U) == 0U && (best & 1U) != 0U)) {
      best = candidate;
      best_error = error;
    }
  }
  return static_cast<std::uint8_t>(best | (std::signbit(value) ? 8U : 0U));
}

std::uint8_t EncodeE4M3Manual(float value) {
  const float magnitude = std::fabs(value);
  if (magnitude >= 448.0F) return 0x7EU;
  std::uint8_t best = 0U;
  float best_error = std::numeric_limits<float>::infinity();
  for (std::uint16_t candidate = 0U; candidate <= 0x7EU; ++candidate) {
    const auto code = static_cast<std::uint8_t>(candidate);
    const float error = std::fabs(magnitude - DecodeE4M3Manual(code));
    if (error < best_error ||
        (error == best_error && (code & 1U) == 0U && (best & 1U) != 0U)) {
      best = code;
      best_error = error;
    }
  }
  return best;
}

float ReadF32(std::span<const std::uint8_t> bytes) {
  const std::uint32_t bits = static_cast<std::uint32_t>(bytes[0]) |
                             (static_cast<std::uint32_t>(bytes[1]) << 8U) |
                             (static_cast<std::uint32_t>(bytes[2]) << 16U) |
                             (static_cast<std::uint32_t>(bytes[3]) << 24U);
  return std::bit_cast<float>(bits);
}

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

bool TextValue(const Value* value, std::string_view expected = {}) {
  return value != nullptr && value->is_string() &&
         (expected.empty() || value->as_string() == expected);
}

bool HexSha256(std::string_view value) {
  if (value.size() != 64U) return false;
  return std::all_of(value.begin(), value.end(), [](char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
  });
}

bool HexCommit(std::string_view value) {
  if (value.size() != 40U) return false;
  return std::all_of(value.begin(), value.end(), [](char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
  });
}

bool ExactString(const Value::Object& object, std::string_view key,
                 std::string_view expected) {
  return TextValue(Required(object, key), expected);
}

bool ExactInteger(const Value::Object& object, std::string_view key,
                  std::int64_t expected) {
  const auto* value = Required(object, key);
  return value != nullptr && value->is_integer() && value->as_integer() == expected;
}

bool ExactNumber(const Value::Object& object, std::string_view key,
                 double expected) {
  const auto* value = Required(object, key);
  return value != nullptr && value->is_number() && value->as_number() == expected;
}

std::uint64_t Fnv1a(std::span<const std::uint8_t> bytes) {
  std::uint64_t hash = 1469598103934665603ULL;
  for (const std::uint8_t byte : bytes) {
    hash ^= byte;
    hash *= 1099511628211ULL;
  }
  return hash;
}

std::vector<float> MakeHidden() {
  std::vector<float> hidden(static_cast<std::size_t>(kHidden));
  for (std::size_t index = 0; index < hidden.size(); ++index) {
    const int a = static_cast<int>(index % 29U) - 14;
    const int b = static_cast<int>((index / 29U) % 17U) - 8;
    hidden[index] = static_cast<float>(a) / 64.0F + static_cast<float>(b) / 512.0F;
  }
  return hidden;
}

struct ManualActivation {
  std::vector<std::uint8_t> packed;
  std::vector<std::uint8_t> scales;
};

ManualActivation QuantizeManual(std::span<const float> hidden, float divisor) {
  ManualActivation result;
  result.packed.assign(hidden.size() / 2U, 0U);
  result.scales.assign(hidden.size() / 16U, 0U);
  std::vector<float> rounded(hidden.size());
  for (std::size_t index = 0; index < hidden.size(); ++index) {
    rounded[index] = Bf16Rne(hidden[index]);
  }
  for (std::size_t block = 0; block < result.scales.size(); ++block) {
    float amax = 0.0F;
    for (std::size_t local = 0; local < 16U; ++local) {
      amax = std::max(amax, std::fabs(rounded[block * 16U + local]));
    }
    const auto scale_code = EncodeE4M3Manual((amax / 6.0F) * divisor);
    result.scales[block] = scale_code;
    const float scale = DecodeE4M3Manual(scale_code);
    for (std::size_t local = 0; local < 16U; ++local) {
      const float scaled = rounded[block * 16U + local] * divisor;
      const float normalized = scale == 0.0F ? 0.0F : scaled / scale;
      const auto code = EncodeE2M1Manual(normalized);
      const std::size_t index = block * 16U + local;
      const std::size_t byte = index / 2U;
      const unsigned shift = (index & 1U) == 0U ? 0U : 4U;
      result.packed[byte] = static_cast<std::uint8_t>(
          (result.packed[byte] & static_cast<std::uint8_t>(0x0FU << (4U - shift))) |
          static_cast<std::uint8_t>(code << shift));
    }
  }
  return result;
}

float ManualLogit(std::span<const std::uint8_t> packed,
                  std::span<const std::uint8_t> scales,
                  std::uint64_t row,
                  float input_divisor, float weight_divisor, float softcap,
                  const ManualActivation& activation) {
  double dot = 0.0;
  const std::uint64_t scale_row = kHidden / 16U;
  for (std::uint64_t index = 0; index < kHidden; ++index) {
    const std::size_t block = static_cast<std::size_t>(index / 16U);
    const float activation_value =
        DecodeE2M1Manual(Nibble(activation.packed, index)) *
        DecodeE4M3Manual(activation.scales[block]);
    const float weight_value =
        DecodeE2M1Manual(Nibble(packed, row * kHidden + index)) *
        DecodeE4M3Manual(scales[row * scale_row + index / 16U]);
    dot += static_cast<double>(activation_value) * static_cast<double>(weight_value);
  }
  dot /= static_cast<double>(input_divisor) * static_cast<double>(weight_divisor);
  return std::tanh(static_cast<float>(dot) / softcap) * softcap;
}

struct TensorSet {
  const gem16::internal::StoredTensor* packed = nullptr;
  const gem16::internal::StoredTensor* scales = nullptr;
  const gem16::internal::StoredTensor* weight = nullptr;
  const gem16::internal::StoredTensor* input = nullptr;
};

Result<TensorSet> ValidateTensors(const std::vector<gem16::internal::StoredTensor>& tensors) {
  if (tensors.size() != 4U) return Invalid("M07 artifact must contain exactly four tensors");
  TensorSet result;
  for (const auto& tensor : tensors) {
    const gem16::internal::StoredTensor** slot = nullptr;
    if (tensor.name == "model.language_model.embed_tokens.weight_packed") slot = &result.packed;
    if (tensor.name == "model.language_model.embed_tokens.weight_scale") slot = &result.scales;
    if (tensor.name == "model.language_model.embed_tokens.weight_global_scale") slot = &result.weight;
    if (tensor.name == "model.language_model.embed_tokens.input_global_scale") slot = &result.input;
    if (slot == nullptr || *slot != nullptr) return Invalid("unexpected or duplicate M07 tensor name");
    *slot = &tensor;
  }
  if (result.packed == nullptr || result.scales == nullptr || result.weight == nullptr || result.input == nullptr) {
    return Invalid("M07 artifact is missing a canonical tied-head tensor");
  }
  const auto check = [](const gem16::internal::StoredTensor& tensor,
                        std::string_view dtype, std::initializer_list<std::uint64_t> shape,
                        std::uint64_t bytes) -> Status {
    if (tensor.dtype != dtype || tensor.shape != std::vector<std::uint64_t>(shape) || tensor.length != bytes) {
      return Invalid("M07 tied-head tensor dtype, shape, or byte length mismatch: " + tensor.name);
    }
    return Status::Ok();
  };
  for (const auto& item : {
           std::pair<const gem16::internal::StoredTensor*, Status>{result.packed, check(*result.packed, "U8", {kVocabulary, 1408U}, kPackedBytes)},
           std::pair<const gem16::internal::StoredTensor*, Status>{result.scales, check(*result.scales, "F8_E4M3", {kVocabulary, 176U}, kScaleBytes)},
           std::pair<const gem16::internal::StoredTensor*, Status>{result.weight, check(*result.weight, "F32", {1U}, kScalarBytes)},
           std::pair<const gem16::internal::StoredTensor*, Status>{result.input, check(*result.input, "F32", {1U}, kScalarBytes)}}) {
    if (!item.second.ok()) return item.second;
  }
  const std::string shard = result.packed->shard;
  if (result.scales->shard != shard || result.weight->shard != shard || result.input->shard != shard) {
    return Invalid("M07 tied-head tensors must share one Safetensors shard");
  }
  return result;
}

Result<gem16::internal::MappedFile> MapShard(const std::filesystem::path& root,
                                              const TensorSet& tensors) {
  if (tensors.packed->shard.empty() || std::filesystem::path(tensors.packed->shard).has_parent_path()) {
    return Invalid("M07 shard name is not a simple relative filename");
  }
  const auto path = root / tensors.packed->shard;
  std::error_code error;
  if (std::filesystem::is_symlink(path, error) || error || !std::filesystem::is_regular_file(path, error) || error) {
    return Invalid("M07 shard must be a non-symlink regular file");
  }
  const auto canonical_root = std::filesystem::canonical(root, error);
  if (error) return Io("cannot canonicalize M07 artifact root: " + error.message());
  const auto canonical_shard = std::filesystem::canonical(path, error);
  if (error || canonical_shard.parent_path() != canonical_root) {
    return DataLoss("M07 shard escapes artifact root");
  }
  auto mapped = gem16::internal::MappedFile::Open(canonical_shard);
  if (!mapped.ok()) return mapped.status();
  for (const auto* tensor : {tensors.packed, tensors.scales, tensors.weight, tensors.input}) {
    std::uint64_t end = 0;
    if (!AddFits(tensor->absolute_offset, tensor->length, &end) || end > mapped.value().size()) {
      return DataLoss("M07 tensor range exceeds mapped shard: " + tensor->name);
    }
  }
  return mapped;
}

Status ValidateManifest(
    const Value& manifest, const TensorSet& tensor_set, const gem16::internal::MappedFile& mapped_shard,
    std::uint64_t index_size, std::string_view shard_hash, std::string_view index_hash,
    std::string_view manifest_hash, std::string* binary_hash,
    std::string* compiler_commit, std::string* dependency_hash) {
  if (!manifest.is_object() ||
      !ExactKeys(manifest.as_object(), {
          "artifact_profile", "artifact_status", "byte_totals", "compiler",
          "compiler_settings", "excluded_tensors", "file_hash_scope", "files",
          "head_format", "omitted_families", "omitted_tensor_groups", "plan",
          "quantization", "schema_version", "source", "text_only", "tensors"})) {
    return Invalid("M07 compilation manifest top-level schema mismatch");
  }
  const auto& object = manifest.as_object();
  const auto* schema = Required(object, "schema_version");
  if (schema == nullptr || !schema->is_integer() || schema->as_integer() != 1 ||
      !TextValue(Required(object, "artifact_profile"), kProfile) ||
      !TextValue(Required(object, "artifact_status"), kArtifactStatus) ||
      !TextValue(Required(object, "head_format"), "nvfp4") ||
      Required(object, "text_only") == nullptr ||
      !Required(object, "text_only")->is_bool() ||
      !Required(object, "text_only")->as_bool()) {
    return Invalid("M07 compilation manifest profile/status mismatch");
  }
  const auto* plan = Required(object, "plan");
  if (plan == nullptr || !plan->is_object() ||
      !ExactKeys(plan->as_object(), {"schema_version", "compiler_manifest_sha256",
                                     "resolved_plan_sha256", "source_contract",
                                     "target_shard_bytes"}) ||
      !TextValue(Required(plan->as_object(), "source_contract"),
                 "gemma4-26b-source-bf16-tied-head-v1") ||
      !ExactInteger(plan->as_object(), "schema_version", 1) ||
      !ExactString(plan->as_object(), "compiler_manifest_sha256", kPlanSha256) ||
      !ExactString(plan->as_object(), "resolved_plan_sha256", kPlanSha256) ||
      !ExactInteger(plan->as_object(), "target_shard_bytes",
                    static_cast<std::int64_t>(kTargetShardBytes))) {
    return Invalid("M07 compilation manifest plan provenance mismatch");
  }
  const auto* compiler = Required(object, "compiler");
  if (compiler == nullptr || !compiler->is_object() ||
      !ExactKeys(compiler->as_object(), {"repository", "commit", "dirty", "python",
                                         "platform", "dependencies_lock_sha256",
                                         "implementation", "native_encoder"}) ||
      !TextValue(Required(compiler->as_object(), "repository"), "Danmoreng/gem16") ||
      !TextValue(Required(compiler->as_object(), "implementation"),
                 "gem16_compile_m07_native_v1") ||
      Required(compiler->as_object(), "dirty") == nullptr ||
      !Required(compiler->as_object(), "dirty")->is_bool() ||
      Required(compiler->as_object(), "dirty")->as_bool() ||
      !TextValue(Required(compiler->as_object(), "python"), "3.14.6") ||
      !HexCommit(Required(compiler->as_object(), "commit") != nullptr &&
                         Required(compiler->as_object(), "commit")->is_string()
                     ? Required(compiler->as_object(), "commit")->as_string()
                     : std::string_view{}) ||
      !ExactString(compiler->as_object(), "dependencies_lock_sha256",
                   kDependencyLockSha256)) {
    return Invalid("M07 compiler provenance is not a clean native build");
  }
  const auto* platform = Required(compiler->as_object(), "platform");
  if (platform == nullptr || !platform->is_object() ||
      !ExactKeys(platform->as_object(), {"byteorder", "locale", "machine",
                                         "python_implementation", "python_major_minor",
                                         "python_version", "system"}) ||
      !ExactString(platform->as_object(), "byteorder", "little") ||
      !ExactString(platform->as_object(), "locale", "C.UTF-8") ||
      !ExactString(platform->as_object(), "machine", "x86_64") ||
      !ExactString(platform->as_object(), "python_implementation", "CPython") ||
      !ExactString(platform->as_object(), "python_major_minor", "3.14") ||
      !ExactString(platform->as_object(), "python_version", "3.14.6") ||
      !ExactString(platform->as_object(), "system", "Linux")) {
    return Invalid("M07 compiler platform identity mismatch");
  }
  *compiler_commit = Required(compiler->as_object(), "commit")->as_string();
  const auto* native = Required(compiler->as_object(), "native_encoder");
  if (native == nullptr || !native->is_object() ||
      !ExactKeys(native->as_object(), {"build", "protocol", "sha256", "threads"}) ||
      !TextValue(Required(native->as_object(), "protocol"),
                 "gem16-nvfp4-direct-v1") ||
      !TextValue(Required(native->as_object(), "sha256")) ||
      !HexSha256(Required(native->as_object(), "sha256")->as_string()) ||
      !ExactInteger(native->as_object(), "threads", 16)) {
    return Invalid("M07 native encoder provenance is invalid");
  }
  *binary_hash = Required(native->as_object(), "sha256")->as_string();
  *dependency_hash = Required(compiler->as_object(), "dependencies_lock_sha256")->as_string();
  const auto* build = Required(native->as_object(), "build");
  if (build == nullptr || !build->is_object() ||
      !ExactKeys(build->as_object(), {"build_type", "compiler_id", "compiler_version",
                                       "cxx_standard", "processor", "system"}) ||
      !ExactString(build->as_object(), "build_type", "Release") ||
      !ExactString(build->as_object(), "compiler_id", "GNU") ||
      !ExactString(build->as_object(), "compiler_version", "16.1.1") ||
      !ExactString(build->as_object(), "cxx_standard", "20") ||
      !ExactString(build->as_object(), "processor", "x86_64") ||
      !ExactString(build->as_object(), "system", "Linux")) {
    return Invalid("M07 native encoder is not a Release build");
  }
  const auto* source = Required(object, "source");
  if (source == nullptr || !source->is_object() ||
      !ExactKeys(source->as_object(), {"lock_sha256", "repository", "revision", "resolved_at_utc"}) ||
      !ExactString(source->as_object(), "lock_sha256", kSourceLockSha256) ||
      !ExactString(source->as_object(), "repository", kSourceRepository) ||
      !ExactString(source->as_object(), "revision", kSourceRevision) ||
      !ExactString(source->as_object(), "resolved_at_utc", kSourceResolvedAt)) {
    return Invalid("M07 source provenance is not the accepted QAT lock");
  }

  const auto* totals = Required(object, "byte_totals");
  const auto total_integer = [](const Value::Object& values, std::string_view key) {
    const auto* value = Required(values, key);
    return value != nullptr && value->is_integer();
  };
  if (totals == nullptr || !totals->is_object() ||
      !ExactKeys(totals->as_object(), {"excluded_tensor_bytes", "excluded_tensor_count",
                                       "output_tensor_bytes", "output_tensor_count",
                                       "source_tensor_count"}) ||
      !total_integer(totals->as_object(), "excluded_tensor_count") ||
      !total_integer(totals->as_object(), "output_tensor_count") ||
      !total_integer(totals->as_object(), "output_tensor_bytes") ||
      !total_integer(totals->as_object(), "source_tensor_count") ||
      Required(totals->as_object(), "excluded_tensor_count")->as_integer() != 1012 ||
      Required(totals->as_object(), "output_tensor_count")->as_integer() != 4 ||
      Required(totals->as_object(), "output_tensor_bytes")->as_integer() != 415236104 ||
      Required(totals->as_object(), "source_tensor_count")->as_integer() != 1013) {
    return Invalid("M07 compilation manifest byte totals mismatch");
  }
  const auto* files = Required(object, "files");
  if (files == nullptr || !files->is_array() || files->as_array().size() != 2U) {
    return Invalid("M07 compilation manifest must contain one shard and one index file");
  }
  std::string manifest_shard;
  std::string manifest_index;
  for (const auto& item : files->as_array()) {
    if (!item.is_object() || !ExactKeys(item.as_object(), {"kind", "path", "sha256", "size"}) ||
        !TextValue(Required(item.as_object(), "path")) ||
        !TextValue(Required(item.as_object(), "sha256")) ||
        !HexSha256(Required(item.as_object(), "sha256")->as_string()) ||
        Required(item.as_object(), "size") == nullptr ||
        !Required(item.as_object(), "size")->is_integer() ||
        Required(item.as_object(), "size")->as_integer() < 0) {
      return Invalid("M07 compilation manifest file record is invalid");
    }
    const auto kind = Required(item.as_object(), "kind")->as_string();
    if (kind == "safetensors_shard" && manifest_shard.empty()) {
      manifest_shard = Required(item.as_object(), "path")->as_string();
      if (Required(item.as_object(), "sha256")->as_string() != shard_hash ||
          static_cast<std::uint64_t>(Required(item.as_object(), "size")->as_integer()) != mapped_shard.size()) {
        return DataLoss("M07 shard file hash or size mismatch");
      }
    } else if (kind == "safetensors_index" && manifest_index.empty()) {
      manifest_index = Required(item.as_object(), "path")->as_string();
      if (Required(item.as_object(), "sha256")->as_string() != index_hash ||
          static_cast<std::uint64_t>(Required(item.as_object(), "size")->as_integer()) != index_size) {
        return DataLoss("M07 index file hash mismatch");
      }
    } else {
      return Invalid("M07 compilation manifest has duplicate or unsupported file kinds");
    }
  }
  if (manifest_shard != tensor_set.packed->shard || manifest_index != "model.safetensors.index.json") {
    return Invalid("M07 compilation manifest file names are not canonical");
  }
  const auto* quantization = Required(object, "quantization");
  if (quantization == nullptr || !quantization->is_object() ||
      !ExactKeys(quantization->as_object(), {"attention", "embedding_head", "experts",
                                             "production_quantization_implemented", "profile"}) ||
      !ExactString(quantization->as_object(), "profile", kProfile) ||
      !ExactString(quantization->as_object(), "attention", "deferred-to-m08") ||
      !ExactString(quantization->as_object(), "experts", "deferred-to-m08") ||
      !ExactString(quantization->as_object(), "embedding_head", "nvfp4-group16-divisor-v1") ||
      Required(quantization->as_object(), "production_quantization_implemented") == nullptr ||
      !Required(quantization->as_object(), "production_quantization_implemented")->is_bool() ||
      Required(quantization->as_object(), "production_quantization_implemented")->as_bool()) {
    return Invalid("M07 quantization contract mismatch");
  }
  if (!ExactString(object, "file_hash_scope", kFileHashScope)) {
    return Invalid("M07 compilation manifest file hash scope mismatch");
  }
  const auto* omitted = Required(object, "omitted_families");
  if (omitted == nullptr || !omitted->is_array() || omitted->as_array().size() != 4U ||
      !TextValue(&omitted->as_array()[0], "audio") ||
      !TextValue(&omitted->as_array()[1], "mtp") ||
      !TextValue(&omitted->as_array()[2], "video") ||
      !TextValue(&omitted->as_array()[3], "vision")) {
    return Invalid("M07 omitted family contract mismatch");
  }
  const auto* settings = Required(object, "compiler_settings");
  if (settings == nullptr || !settings->is_object() ||
      !ExactKeys(settings->as_object(), {"host_memory_cap_bytes", "reference_platform_strict",
                                         "resume_policy", "staging_buffer_bytes", "threads"}) ||
      !ExactInteger(settings->as_object(), "host_memory_cap_bytes", 8589934592LL) ||
      !ExactInteger(settings->as_object(), "staging_buffer_bytes", 1048576LL) ||
      !ExactInteger(settings->as_object(), "threads", 16) ||
      !ExactString(settings->as_object(), "resume_policy", "restart-only") ||
      Required(settings->as_object(), "reference_platform_strict") == nullptr ||
      !Required(settings->as_object(), "reference_platform_strict")->is_bool() ||
      !Required(settings->as_object(), "reference_platform_strict")->as_bool()) {
    return Invalid("M07 compiler settings mismatch");
  }
  if (manifest_hash.empty()) return Invalid("M07 compilation manifest hash is empty");

  const std::array<std::string, 4> names = {
      "model.language_model.embed_tokens.weight_packed",
      "model.language_model.embed_tokens.weight_scale",
      "model.language_model.embed_tokens.weight_global_scale",
      "model.language_model.embed_tokens.input_global_scale"};
  const auto* tensors = Required(object, "tensors");
  if (tensors == nullptr || !tensors->is_array() || tensors->as_array().size() != names.size()) {
    return Invalid("M07 compilation manifest must contain exactly four output tensors");
  }
  const std::set<std::string> expected_names(names.begin(), names.end());
  std::set<std::string> actual_names;
  std::uint64_t payload_base = 0;
  bool payload_base_set = false;
  for (const auto& item : tensors->as_array()) {
    if (!item.is_object() ||
        !ExactKeys(item.as_object(), {"aliased", "axis_transformation", "byte_length",
          "dequantization_equation", "disk_layout", "logical_dtype", "logical_shape",
          "operation_id", "output_data_offsets", "output_dtype", "output_name",
          "output_shard", "physical_shape", "quantizer_parameters", "residency_class",
          "role", "runtime_layout", "sha256", "sources", "transformation",
          "transformation_version"})) {
      return Invalid("M07 output tensor record schema mismatch");
    }
    const auto* output_name = Required(item.as_object(), "output_name");
    if (!TextValue(output_name) || !actual_names.insert(output_name->as_string()).second ||
        !expected_names.contains(output_name->as_string()) ||
        !TextValue(Required(item.as_object(), "operation_id"),
                   "nvfp4-head:model.language_model.embed_tokens") ||
        !TextValue(Required(item.as_object(), "output_shard"), manifest_shard) ||
        !TextValue(Required(item.as_object(), "role"), "tied_embedding_and_output") ||
        !TextValue(Required(item.as_object(), "axis_transformation"), "vocabulary,hidden") ||
        Required(item.as_object(), "aliased") == nullptr ||
        !Required(item.as_object(), "aliased")->is_bool() ||
        !Required(item.as_object(), "aliased")->as_bool() ||
        !TextValue(Required(item.as_object(), "transformation")) ||
        Required(item.as_object(), "transformation_version") == nullptr ||
        !Required(item.as_object(), "transformation_version")->is_integer() ||
        Required(item.as_object(), "transformation_version")->as_integer() != 1 ||
        !TextValue(Required(item.as_object(), "sha256")) ||
        !HexSha256(Required(item.as_object(), "sha256")->as_string())) {
      return Invalid("M07 output tensor identity/alias/hash mismatch");
    }
    const auto* sources = Required(item.as_object(), "sources");
    if (sources == nullptr || !sources->is_array() || sources->as_array().size() != 1U ||
        !sources->as_array()[0].is_object() ||
        !ExactKeys(sources->as_array()[0].as_object(), {"name", "range", "sha256"}) ||
        !TextValue(Required(sources->as_array()[0].as_object(), "name"),
                   "model.language_model.embed_tokens.weight") ||
        !ExactString(sources->as_array()[0].as_object(), "sha256", kSourceTensorSha256)) {
      return Invalid("M07 output source identity mismatch");
    }
    const auto* range = Required(sources->as_array()[0].as_object(), "range");
    if (range == nullptr || !range->is_object() ||
        !ExactKeys(range->as_object(), {"absolute_offset", "byte_length", "range_identity_sha256", "shard", "shard_sha256"}) ||
        !TextValue(Required(range->as_object(), "shard"), "model-00001-of-00002.safetensors") ||
        Required(range->as_object(), "absolute_offset") == nullptr ||
        !Required(range->as_object(), "absolute_offset")->is_integer() ||
        Required(range->as_object(), "absolute_offset")->as_integer() != 6617872 ||
        Required(range->as_object(), "byte_length") == nullptr ||
        !Required(range->as_object(), "byte_length")->is_integer() ||
        Required(range->as_object(), "byte_length")->as_integer() != 1476395008 ||
        !ExactString(range->as_object(), "range_identity_sha256", kSourceRangeIdentitySha256) ||
        !ExactString(range->as_object(), "shard_sha256", kSourceShardSha256)) {
      return Invalid("M07 output source range is not canonical");
    }
    const auto& descriptor = output_name->as_string() == names[0] ? *tensor_set.packed :
        output_name->as_string() == names[1] ? *tensor_set.scales :
        output_name->as_string() == names[2] ? *tensor_set.weight : *tensor_set.input;
    const auto* bytes = Required(item.as_object(), "byte_length");
    const auto* dtype = Required(item.as_object(), "output_dtype");
    const auto* physical_shape = Required(item.as_object(), "physical_shape");
    const std::string_view expected_dtype = output_name->as_string() == names[0] ? "U8" :
        output_name->as_string() == names[1] ? "F8_E4M3" : "F32";
    const auto* offsets = Required(item.as_object(), "output_data_offsets");
    const auto* disk_layout = Required(item.as_object(), "disk_layout");
    const auto* runtime_layout = Required(item.as_object(), "runtime_layout");
    const auto* transformation = Required(item.as_object(), "transformation");
    const auto* quantizer = Required(item.as_object(), "quantizer_parameters");
    const std::vector<std::uint64_t> expected_shape = output_name->as_string() == names[0]
        ? std::vector<std::uint64_t>{kVocabulary, 1408U}
        : output_name->as_string() == names[1]
            ? std::vector<std::uint64_t>{kVocabulary, 176U}
            : std::vector<std::uint64_t>{1U};
    const std::string_view expected_layout = output_name->as_string() == names[0]
        ? "canonical_row_major_low_nibble_first"
        : output_name->as_string() == names[1]
            ? "canonical_row_major_group16_e4m3" : "scalar_f32";
    const std::string_view expected_runtime = output_name->as_string() == names[0]
        ? "sm120_row8_k64"
        : output_name->as_string() == names[1]
            ? "sm120_row8_group16_e4m3" : "scalar_f32";
    const std::string_view expected_transformation = output_name->as_string() == names[0]
        ? "nvfp4-packed"
        : output_name->as_string() == names[1]
            ? "nvfp4-local-scale"
            : output_name->as_string() == names[2]
                ? "nvfp4-weight-divisor" : "nvfp4-input-divisor";
    const std::string_view expected_component = output_name->as_string() == names[0]
        ? "weight_packed"
        : output_name->as_string() == names[1]
            ? "weight_scale"
            : output_name->as_string() == names[2]
                ? "weight_global_scale" : "input_global_scale";
    if (!TextValue(dtype, expected_dtype) || physical_shape == nullptr ||
        !physical_shape->is_array() || physical_shape->as_array().size() != expected_shape.size() ||
        !std::equal(physical_shape->as_array().begin(), physical_shape->as_array().end(),
                    expected_shape.begin(), [](const Value& value, std::uint64_t expected) {
                      return value.is_integer() && value.as_integer() >= 0 &&
                             static_cast<std::uint64_t>(value.as_integer()) == expected;
                    }) ||
        !TextValue(Required(item.as_object(), "logical_dtype"), "BF16") ||
        Required(item.as_object(), "logical_shape") == nullptr ||
        !Required(item.as_object(), "logical_shape")->is_array() ||
        Required(item.as_object(), "logical_shape")->as_array().size() != 2U ||
        !std::equal(Required(item.as_object(), "logical_shape")->as_array().begin(),
                    Required(item.as_object(), "logical_shape")->as_array().end(),
                    std::array<std::uint64_t, 2>{kVocabulary, kHidden}.begin(),
                    [](const Value& value, std::uint64_t expected) {
                      return value.is_integer() && value.as_integer() >= 0 &&
                             static_cast<std::uint64_t>(value.as_integer()) == expected;
                    }) ||
        !TextValue(Required(item.as_object(), "residency_class"), "immutable_device_text") ||
        !TextValue(disk_layout, expected_layout) || !TextValue(runtime_layout, expected_runtime) ||
        !TextValue(transformation, expected_transformation) || quantizer == nullptr ||
        !quantizer->is_object() ||
        !ExactString(quantizer->as_object(), "contract_id", "gem16.nvfp4_bf16_group16") ||
        !ExactInteger(quantizer->as_object(), "contract_version", 1) ||
        !ExactString(quantizer->as_object(), "source_dtype", "BF16") ||
        !ExactString(quantizer->as_object(), "packed_dtype", "U8") ||
        !ExactString(quantizer->as_object(), "local_scale_dtype", "F8_E4M3") ||
        !ExactString(quantizer->as_object(), "global_scale_dtype", "F32") ||
        !ExactInteger(quantizer->as_object(), "group_size", 16) ||
        !ExactString(quantizer->as_object(), "value_codec", "E2M1") ||
        !ExactString(quantizer->as_object(), "scale_codec", "E4M3FN") ||
        !ExactString(quantizer->as_object(), "global_scale_role", "divisor") ||
        !ExactString(quantizer->as_object(), "weight_divisor", "bf16_rne((448*6)/tensor_amax)") ||
        !ExactNumber(quantizer->as_object(), "zero_tensor_divisor", 1.0) ||
        !ExactNumber(quantizer->as_object(), "input_divisor", 1.0) ||
        !ExactString(quantizer->as_object(), "local_scale", "e4m3fn(binary32((block_amax*(1/6))*weight_divisor))") ||
        !ExactString(quantizer->as_object(), "rounding", "nearest_even") ||
        !ExactString(quantizer->as_object(), "saturation", "finite_saturation") ||
        !ExactString(quantizer->as_object(), "signed_zero", "preserve_in_nonzero_blocks_canonicalize_zero_blocks") ||
        !ExactString(quantizer->as_object(), "zero_behavior", "zero_or_underflow_block_emits_zero_scale_and_zero_payload") ||
        !ExactString(quantizer->as_object(), "packing", "low_nibble_first") ||
        !ExactString(quantizer->as_object(), "tensor_granularity", "one_weight_and_input_divisor_per_source_tensor") ||
        !ExactString(quantizer->as_object(), "component", expected_component) ||
        bytes == nullptr || !bytes->is_integer() || bytes->as_integer() < 0 ||
        static_cast<std::uint64_t>(bytes->as_integer()) != descriptor.length ||
        offsets == nullptr || !offsets->is_array() || offsets->as_array().size() != 2U ||
        !offsets->as_array()[0].is_integer() || !offsets->as_array()[1].is_integer() ||
        offsets->as_array()[0].as_integer() < 0 || offsets->as_array()[1].as_integer() < 0 ||
        offsets->as_array()[1].as_integer() < offsets->as_array()[0].as_integer()) {
      return Invalid("M07 output tensor range or contract does not match Safetensors");
    }
    const std::uint64_t begin = static_cast<std::uint64_t>(offsets->as_array()[0].as_integer());
    const std::uint64_t end = static_cast<std::uint64_t>(offsets->as_array()[1].as_integer());
    if (end - begin != descriptor.length) {
      return Invalid("M07 output tensor byte range does not match Safetensors");
    }
    if (descriptor.absolute_offset < begin) return Invalid("M07 output offset underflow");
    const std::uint64_t current_base = descriptor.absolute_offset - begin;
    if (!payload_base_set) {
      payload_base = current_base;
      payload_base_set = true;
    }
    if (payload_base != current_base) {
      return DataLoss("M07 output tensors do not share one Safetensors payload base");
    }
    const auto* actual_hash = Required(item.as_object(), "sha256");
    const auto actual = gem16::compiler::Sha256Hex(
        mapped_shard.data() + descriptor.absolute_offset,
        static_cast<std::size_t>(descriptor.length));
    if (actual != actual_hash->as_string()) return DataLoss("M07 output tensor hash mismatch");
  }
  if (actual_names != expected_names) return Invalid("M07 output tensor names are incomplete");
  return Status::Ok();
}

std::span<const std::uint8_t> SpanFor(const gem16::internal::MappedFile& mapped,
                                      const gem16::internal::StoredTensor& tensor) {
  return {reinterpret_cast<const std::uint8_t*>(mapped.data() + tensor.absolute_offset),
          static_cast<std::size_t>(tensor.length)};
}

Value I64(std::int64_t value) { return Value(value); }
Value F64(double value) { return Value(value); }
Value Text(std::string value) { return Value(std::move(value)); }

#ifndef _WIN32
struct FileStamp {
  dev_t device = 0;
  ino_t inode = 0;
  std::uint64_t size = 0;
};

bool Stamp(const std::filesystem::path& path, FileStamp* result) {
  struct stat info {};
  if (stat(path.c_str(), &info) != 0 || !S_ISREG(info.st_mode) || info.st_size < 0) return false;
  result->device = info.st_dev;
  result->inode = info.st_ino;
  result->size = static_cast<std::uint64_t>(info.st_size);
  return true;
}

bool SameStamp(const FileStamp& left, const FileStamp& right) {
  return left.device == right.device && left.inode == right.inode && left.size == right.size;
}
#endif

bool IsDirectRegular(const std::filesystem::path& root,
                     std::string_view name, std::filesystem::path* result) {
  if (name.empty() || std::filesystem::path(name).is_absolute() ||
      std::filesystem::path(name).has_parent_path()) return false;
  const auto path = root / std::filesystem::path(name);
  std::error_code error;
  if (std::filesystem::is_symlink(path, error) || error ||
      !std::filesystem::is_regular_file(path, error) || error) return false;
  *result = path;
  return true;
}

bool PathHasSymlink(const std::filesystem::path& path) {
  std::error_code error;
  std::filesystem::path current = path.root_path();
  for (const auto& component : path.relative_path()) {
    current /= component;
    if (std::filesystem::is_symlink(current, error) || error) return true;
  }
  return false;
}

bool IsWithin(const std::filesystem::path& child,
             const std::filesystem::path& parent) {
  const auto child_text = child.lexically_normal().string();
  const auto parent_text = parent.lexically_normal().string();
  return child_text == parent_text ||
         (child_text.size() > parent_text.size() &&
          child_text.compare(0, parent_text.size(), parent_text) == 0 &&
          child_text[parent_text.size()] == std::filesystem::path::preferred_separator);
}

bool ExactArtifactFiles(const std::filesystem::path& root,
                        std::string_view shard_name) {
  const std::set<std::string> expected = {
      "gem16_compilation.json", "model.safetensors.index.json",
      std::string(shard_name)};
  std::set<std::string> actual;
  std::error_code error;
  for (std::filesystem::directory_iterator it(root, error), end;
       !error && it != end; it.increment(error)) {
    if (error || it->is_symlink(error) || error || !it->is_regular_file(error) || error ||
        it->path().parent_path() != root) {
      return false;
    }
    actual.insert(it->path().filename().string());
  }
  return !error && actual == expected;
}

bool Better(float value, std::uint32_t token, float best, std::uint32_t best_token) {
  return value > best || (value == best && token < best_token);
}

std::pair<std::uint32_t, float> ScanBest(std::span<const float> logits,
                                         std::uint32_t suppressed) {
  std::uint32_t best_token = 0;
  float best_value = -std::numeric_limits<float>::infinity();
  bool found = false;
  for (std::uint32_t token = 0; token < logits.size(); ++token) {
    if (token == suppressed) continue;
    if (!found || Better(logits[token], token, best_value, best_token)) {
      found = true;
      best_token = token;
      best_value = logits[token];
    }
  }
  return {best_token, best_value};
}

#ifndef _WIN32
Status Publish(const std::filesystem::path& output, const std::string& text) {
  const auto temporary = output.string() + ".incomplete";
  const int fd = open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
  if (fd < 0) return errno == EEXIST ? Invalid("diagnostic staging output already exists") : Io("cannot create diagnostic output");
  auto cleanup = [&] { (void)unlink(temporary.c_str()); };
  std::size_t written = 0;
  while (written < text.size()) {
    const ssize_t count = write(fd, text.data() + written, text.size() - written);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) { (void)close(fd); cleanup(); return Io("cannot write diagnostic output"); }
    written += static_cast<std::size_t>(count);
  }
  if (fsync(fd) != 0 || close(fd) != 0) { cleanup(); return Io("cannot fsync diagnostic output"); }
  if (link(temporary.c_str(), output.c_str()) != 0) {
    cleanup();
    return errno == EEXIST ? Invalid("diagnostic output already exists") : Io("cannot publish diagnostic output");
  }
  const int verify_fd = open(output.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
  if (verify_fd < 0) {
    (void)unlink(output.c_str());
    cleanup();
    return Io("cannot reopen diagnostic output");
  }
  struct stat verify_info {};
  if (fstat(verify_fd, &verify_info) != 0 || verify_info.st_size != static_cast<off_t>(text.size())) {
    (void)close(verify_fd);
    (void)unlink(output.c_str());
    cleanup();
    return DataLoss("published diagnostic output size changed");
  }
  std::string verify_text(text.size(), '\0');
  std::size_t verify_read = 0;
  while (verify_read < verify_text.size()) {
    const ssize_t count = read(verify_fd, verify_text.data() + verify_read,
                               verify_text.size() - verify_read);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) {
      (void)close(verify_fd);
      (void)unlink(output.c_str());
      cleanup();
      return Io("cannot read back diagnostic output");
    }
    verify_read += static_cast<std::size_t>(count);
  }
  (void)close(verify_fd);
  if (gem16::compiler::Sha256Hex(verify_text.data(), verify_text.size()) !=
      gem16::compiler::Sha256Hex(text.data(), text.size())) {
    (void)unlink(output.c_str());
    cleanup();
    return DataLoss("published diagnostic output hash changed");
  }
  const int parent_fd = open(output.parent_path().c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (parent_fd < 0 || fsync(parent_fd) != 0) {
    if (parent_fd >= 0) (void)close(parent_fd);
    (void)unlink(output.c_str());
    cleanup();
    return Io("cannot fsync diagnostic output directory");
  }
  (void)close(parent_fd);
  cleanup();
  return Status::Ok();
}
#else
Status Publish(const std::filesystem::path& output, const std::string& text) {
  const auto temporary = output.string() + ".incomplete";
  std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
  if (!stream) return Io("cannot create diagnostic output");
  stream << text;
  stream.close();
  if (!stream) { (void)std::filesystem::remove(temporary); return Io("cannot write diagnostic output"); }
  std::error_code error;
  std::filesystem::rename(temporary, output, error);
  if (error) { (void)std::filesystem::remove(temporary); return Io("cannot publish diagnostic output: " + error.message()); }
  return Status::Ok();
}
#endif

void Usage() {
  std::cerr << "usage: gem16-nvfp4-head-diagnostic --model ARTIFACT_DIR --output REPORT.json\n";
}

Result<std::pair<std::filesystem::path, std::filesystem::path>> ParseArgs(int argc, char** argv) {
  std::filesystem::path model;
  std::filesystem::path output;
  for (int index = 1; index < argc; ++index) {
    const std::string arg = argv[index];
    if ((arg == "--model" || arg == "--output") && index + 1 < argc) {
      auto& destination = arg == "--model" ? model : output;
      if (!destination.empty()) return Invalid("duplicate argument: " + arg);
      destination = argv[++index];
    } else {
      return Invalid("unknown or incomplete argument: " + arg);
    }
  }
  if (model.empty() || output.empty()) return Invalid("--model and --output are required");
  return std::make_pair(model, output);
}

int Run(int argc, char** argv) {
  const auto arguments = ParseArgs(argc, argv);
  if (!arguments.ok()) { Usage(); std::cerr << arguments.status().message() << '\n'; return 2; }
#ifdef _WIN32
  std::cerr << "M07 tied-head diagnostic requires POSIX no-follow publication\n";
  return 1;
#else
  const auto model = std::filesystem::absolute(arguments.value().first);
  const auto output = std::filesystem::absolute(arguments.value().second);
  std::error_code error;
  if (std::filesystem::is_symlink(model, error) || error ||
      !std::filesystem::is_directory(model, error) || error) {
    std::cerr << "invalid M07 artifact directory\n"; return 2;
  }
  const auto canonical_model = std::filesystem::canonical(model, error);
  if (error || PathHasSymlink(model)) {
    std::cerr << "M07 artifact directory path contains a symlink\n"; return 2;
  }
  error.clear();
  const auto output_status = std::filesystem::symlink_status(output, error);
  const bool output_exists = output_status.type() != std::filesystem::file_type::not_found &&
                             output_status.type() != std::filesystem::file_type::none;
  if ((error && error != std::make_error_code(std::errc::no_such_file_or_directory)) ||
      output_exists || PathHasSymlink(output.parent_path())) {
    std::cerr << "diagnostic output already exists or uses a symlink\n"; return 2;
  }
  const auto canonical_parent = std::filesystem::canonical(output.parent_path(), error);
  if (error || IsWithin(canonical_parent, canonical_model)) {
    std::cerr << "diagnostic output must be outside the artifact directory\n"; return 2;
  }
  const auto manifest_path = model / "gem16_compilation.json";
  const auto index_path = model / "model.safetensors.index.json";
  std::filesystem::path ignored;
#ifndef _WIN32
  FileStamp manifest_before;
  FileStamp index_before;
#endif
  if (!IsDirectRegular(model, "gem16_compilation.json", &ignored) ||
      !IsDirectRegular(model, "model.safetensors.index.json", &ignored)) {
    std::cerr << "M07 artifact manifest/index must be direct regular files\n"; return 1;
  }
#ifndef _WIN32
  if (!Stamp(manifest_path, &manifest_before) || !Stamp(index_path, &index_before) ||
      (manifest_before.inode == index_before.inode &&
       manifest_before.device == index_before.device)) {
    std::cerr << "M07 artifact manifest/index identity is unsafe\n"; return 1;
  }
#endif
  for (std::filesystem::recursive_directory_iterator it(model, error), end;
       !error && it != end; it.increment(error)) {
    if (error || it->is_symlink(error) || error || !it->is_regular_file(error) || error) {
      std::cerr << "M07 artifact contains an unsafe file entry\n"; return 1;
    }
    if (it->path().parent_path() != model) {
      std::cerr << "M07 artifact must contain only direct files\n"; return 1;
    }
  }
  // Bind work before hashing: the canonical M07 partial has one exact index
  // and shard size, so trailing or sparse attacker-controlled bytes are not
  // accepted or scanned.
  if (index_before.size != kCanonicalIndexBytes) {
    std::cerr << "M07 artifact index size is not canonical\n"; return 1;
  }
  auto mapped_manifest = gem16::internal::MappedFile::Open(manifest_path);
  auto mapped_index = gem16::internal::MappedFile::Open(index_path);
  if (!mapped_manifest.ok() || !mapped_index.ok() ||
      mapped_manifest.value().size() > kMaxManifestBytes ||
      mapped_index.value().size() != kCanonicalIndexBytes) {
    std::cerr << "M07 artifact manifest/index cannot be safely mapped\n"; return 1;
  }
  const auto manifest_hash = gem16::compiler::Sha256Hex(
      mapped_manifest.value().data(), static_cast<std::size_t>(mapped_manifest.value().size()));
  const auto index_hash = gem16::compiler::Sha256Hex(
      mapped_index.value().data(), static_cast<std::size_t>(mapped_index.value().size()));
  const std::string_view manifest_text(
      reinterpret_cast<const char*>(mapped_manifest.value().data()),
      static_cast<std::size_t>(mapped_manifest.value().size()));
  const auto parsed_manifest = gem16::json::Parse(
      manifest_text, {128, 2'000'000, kMaxManifestBytes});
  if (!parsed_manifest.ok()) {
    std::cerr << "invalid M07 compilation manifest: " << parsed_manifest.status().message() << '\n'; return 1;
  }
  const auto tensors = gem16::internal::LoadSafetensorsDirectory(model);
  if (!tensors.ok()) { std::cerr << tensors.status().message() << '\n'; return 1; }
#ifndef _WIN32
  FileStamp manifest_after;
  FileStamp index_after;
  if (!Stamp(manifest_path, &manifest_after) || !Stamp(index_path, &index_after) ||
      !SameStamp(manifest_before, manifest_after) || !SameStamp(index_before, index_after) ||
      gem16::compiler::Sha256Hex(mapped_manifest.value().data(), static_cast<std::size_t>(mapped_manifest.value().size())) != manifest_hash ||
      gem16::compiler::Sha256Hex(mapped_index.value().data(), static_cast<std::size_t>(mapped_index.value().size())) != index_hash) {
    std::cerr << "M07 artifact manifest/index changed during validation\n"; return 1;
  }
#endif
  const auto tensor_set = ValidateTensors(tensors.value());
  if (!tensor_set.ok()) { std::cerr << tensor_set.status().message() << '\n'; return 1; }
#ifndef _WIN32
  FileStamp shard_before;
  const auto shard_path = model / tensor_set.value().packed->shard;
  if (!Stamp(shard_path, &shard_before) ||
      shard_before.size != kCanonicalShardBytes ||
      (shard_before.device == manifest_before.device && shard_before.inode == manifest_before.inode) ||
      (shard_before.device == index_before.device && shard_before.inode == index_before.inode)) {
    std::cerr << "M07 artifact files must not be hardlink aliases\n"; return 1;
  }
#endif
  auto mapped = MapShard(model, tensor_set.value());
  if (!mapped.ok()) { std::cerr << mapped.status().message() << '\n'; return 1; }
  if (mapped.value().size() != kCanonicalShardBytes) {
    std::cerr << "M07 artifact shard size is not canonical\n"; return 1;
  }
  const auto shard_hash = gem16::compiler::Sha256Hex(
      mapped.value().data(), static_cast<std::size_t>(mapped.value().size()));
#ifndef _WIN32
  FileStamp shard_after;
  if (!Stamp(shard_path, &shard_after) || !SameStamp(shard_before, shard_after)) {
    std::cerr << "M07 artifact shard changed during validation\n"; return 1;
  }
#endif
  std::string binary_hash;
  std::string compiler_commit;
  std::string dependency_hash;
  const auto manifest_status = ValidateManifest(
      parsed_manifest.value(), tensor_set.value(), mapped.value(), mapped_index.value().size(),
      shard_hash, index_hash, manifest_hash, &binary_hash, &compiler_commit,
      &dependency_hash);
  if (!manifest_status.ok()) { std::cerr << manifest_status.message() << '\n'; return 1; }
  if (!ExactArtifactFiles(model, tensor_set.value().packed->shard)) {
    std::cerr << "M07 artifact file set is not exactly manifest, index, and one shard\n";
    return 1;
  }
  const bool one_physical_payload =
      tensor_set.value().packed->shard == tensor_set.value().scales->shard &&
      tensor_set.value().packed->shard == tensor_set.value().weight->shard &&
      tensor_set.value().packed->shard == tensor_set.value().input->shard;
  const bool one_mapped_shard = one_physical_payload;
  const auto packed = SpanFor(mapped.value(), *tensor_set.value().packed);
  const auto scales = SpanFor(mapped.value(), *tensor_set.value().scales);
  const auto weight_scalar = SpanFor(mapped.value(), *tensor_set.value().weight);
  const auto input_scalar = SpanFor(mapped.value(), *tensor_set.value().input);
  if constexpr (std::endian::native != std::endian::little) {
    std::cerr << "M07 diagnostic requires a little-endian host\n"; return 1;
  }
  const float weight_divisor = ReadF32(weight_scalar);
  const float input_divisor = ReadF32(input_scalar);
  if (!IsFinitePositive(weight_divisor) || !IsFinitePositive(input_divisor)) {
    std::cerr << "M07 artifact divisors must be positive finite F32 values\n"; return 1;
  }
  auto view = gem16::nvfp4::TiedNvfp4HeadView::Create(
      packed, scales, kVocabulary, kHidden, weight_divisor, input_divisor);
  if (!view.ok()) { std::cerr << view.status().message() << '\n'; return 1; }

  const auto hidden = MakeHidden();
  const auto hidden_bytes = std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t*>(hidden.data()), hidden.size() * sizeof(float));
  const auto hidden_checksum = Fnv1a(hidden_bytes);
  const float embedding_scale = Bf16Rne(std::sqrt(static_cast<float>(kHidden)));
  auto lookup_start = std::chrono::steady_clock::now();
  double lookup_max_error = 0.0;
  Value::Array lookup_records;
  for (const std::uint32_t token : kLookupTokens) {
    const auto actual = view.value().Lookup(token);
    if (!actual.ok()) { std::cerr << actual.status().message() << '\n'; return 1; }
    const auto row_offset = static_cast<std::uint64_t>(token) * kHidden;
    double max_error = 0.0;
    for (std::uint64_t index = 0; index < kHidden; ++index) {
      const float expected = Bf16Rne(
          DecodeE2M1Manual(Nibble(packed, row_offset + index)) *
          DecodeE4M3Manual(scales[static_cast<std::size_t>(token) * 176U + index / 16U]) /
          weight_divisor * embedding_scale);
      if (!std::isfinite(expected) || actual.value()[static_cast<std::size_t>(index)] != expected) {
        std::cerr << "M07 lookup reference mismatch\n"; return 1;
      }
      max_error = std::max(max_error, static_cast<double>(std::fabs(actual.value()[static_cast<std::size_t>(index)] - expected)));
    }
    lookup_max_error = std::max(lookup_max_error, max_error);
    lookup_records.emplace_back(Value(Value::Object{{"token", I64(token)}, {"max_abs_error", F64(max_error)}}));
  }
  const double lookup_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - lookup_start).count();

  const auto activation = QuantizeManual(hidden, input_divisor);
  const std::array<std::uint32_t, 4> fixed_rows = {0U, 1U, 131072U, 262143U};
  auto project_start = std::chrono::steady_clock::now();
  gem16::nvfp4::TiedNvfp4HeadView::ProjectionOptions options;
  options.return_diagnostic_logits = true;
  const auto first = view.value().ProjectT1(hidden, options);
  if (!first.ok() || first.value().diagnostic_logits.size() != kVocabulary) {
    std::cerr << (first.ok() ? "M07 projection did not return full logits" : first.status().message()) << '\n'; return 1;
  }
  for (const float value : first.value().diagnostic_logits) {
    if (!std::isfinite(value)) { std::cerr << "M07 projection produced a non-finite logit\n"; return 1; }
  }
  const auto manual_first = ScanBest(first.value().diagnostic_logits, std::numeric_limits<std::uint32_t>::max());
  if (first.value().token != manual_first.first || first.value().value != manual_first.second) {
    std::cerr << "M07 projection selection mismatch\n"; return 1;
  }
  double projection_max_error = 0.0;
  Value::Array fixed_records;
  std::set<std::uint32_t> rows(fixed_rows.begin(), fixed_rows.end());
  rows.insert(first.value().token);
  for (const std::uint32_t row : rows) {
    const float expected = ManualLogit(packed, scales, row, input_divisor,
                                       weight_divisor, 30.0F, activation);
    const double error_value = std::fabs(static_cast<double>(first.value().diagnostic_logits[row]) - expected);
    projection_max_error = std::max(projection_max_error, error_value);
    if (error_value > 1.0e-5) { std::cerr << "M07 projection reference mismatch\n"; return 1; }
    fixed_records.emplace_back(Value(Value::Object{{"token", I64(row)}, {"max_abs_error", F64(error_value)}}));
  }
  const double first_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - project_start).count();

  // The second pass intentionally exercises the public suppression handoff;
  // its timing is reported separately and is not a performance claim.
  const auto second_start = std::chrono::steady_clock::now();
  gem16::nvfp4::TiedNvfp4HeadView::ProjectionOptions suppressed_options;
  const std::array<std::uint32_t, 1> suppression = {first.value().token};
  suppressed_options.suppressed_tokens = suppression;
  suppressed_options.return_diagnostic_logits = true;
  const auto second = view.value().ProjectT1(hidden, suppressed_options);
  if (!second.ok() || second.value().diagnostic_logits.size() != kVocabulary) {
    std::cerr << (second.ok() ? "M07 suppressed projection did not return full logits" : second.status().message()) << '\n'; return 1;
  }
  const auto manual_second = ScanBest(first.value().diagnostic_logits, first.value().token);
  if (second.value().token != manual_second.first || second.value().value != manual_second.second) {
    std::cerr << "M07 suppression selection mismatch\n"; return 1;
  }
  for (std::uint32_t token = 0; token < kVocabulary; ++token) {
    if (second.value().diagnostic_logits[token] != first.value().diagnostic_logits[token]) {
      std::cerr << "M07 suppression changed diagnostic logits\n"; return 1;
    }
  }
  const double second_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - second_start).count();

  // Bind the report to the same mapped bytes and filesystem identities used by
  // the checks above. This closes the long CPU diagnostic window against a
  // replacement of any artifact input before publication.
  const auto final_model = std::filesystem::canonical(model, error);
  FileStamp manifest_final;
  FileStamp index_final;
  FileStamp shard_final;
  if (error || final_model != canonical_model ||
      !Stamp(manifest_path, &manifest_final) || !Stamp(index_path, &index_final) ||
      !Stamp(shard_path, &shard_final) ||
      !SameStamp(manifest_before, manifest_final) ||
      !SameStamp(index_before, index_final) ||
      !SameStamp(shard_before, shard_final) ||
      gem16::compiler::Sha256Hex(
          mapped_manifest.value().data(), static_cast<std::size_t>(mapped_manifest.value().size())) != manifest_hash ||
      gem16::compiler::Sha256Hex(
          mapped_index.value().data(), static_cast<std::size_t>(mapped_index.value().size())) != index_hash ||
      gem16::compiler::Sha256Hex(
          mapped.value().data(), static_cast<std::size_t>(mapped.value().size())) != shard_hash) {
    std::cerr << "M07 artifact changed before diagnostic publication\n";
    return 1;
  }

  Value report(Value::Object{
      {"schema_version", I64(1)},
      {"status", Text("passed")},
      {"artifact_manifest_sha256", Text(manifest_hash)},
      {"artifact_shard_sha256", Text(shard_hash)},
      {"artifact_index_sha256", Text(index_hash)},
      {"native_encoder_sha256", Text(binary_hash)},
      {"compiler_commit", Text(compiler_commit)},
      {"dependencies_lock_sha256", Text(dependency_hash)},
      {"compiler_manifest_sha256", Text(std::string(kPlanSha256))},
      {"diagnostic", Text("m07_nvfp4_tied_head_actual_artifact")},
      {"artifact_profile", Text("nvfp4-tied-head-partial-v1")},
      {"dimensions", Value(Value::Object{{"vocabulary", I64(kVocabulary)}, {"hidden", I64(kHidden)}})},
      {"bytes", Value(Value::Object{{"packed", I64(kPackedBytes)}, {"scales", I64(kScaleBytes)}, {"scalars", I64(8)}, {"total", I64(kPackedBytes + kScaleBytes + 8)}})},
      {"divisors", Value(Value::Object{{"weight", F64(weight_divisor)}, {"input", F64(input_divisor)}})},
      {"physical_storage", Value(Value::Object{{"one_physical_payload", Value(one_physical_payload)}, {"one_mapped_shard", Value(one_mapped_shard)}, {"shared_mapping_for_all_tensors", Value(one_mapped_shard)}, {"packed_range_offset", I64(static_cast<std::int64_t>(tensor_set.value().packed->absolute_offset))}, {"scale_range_offset", I64(static_cast<std::int64_t>(tensor_set.value().scales->absolute_offset))}, {"weight_scalar_range_offset", I64(static_cast<std::int64_t>(tensor_set.value().weight->absolute_offset))}, {"input_scalar_range_offset", I64(static_cast<std::int64_t>(tensor_set.value().input->absolute_offset))}, {"payload_copy", Value(false)}, {"row_cache", Value(false)}, {"shard", Text(tensor_set.value().packed->shard)}})},
      {"lookup", Value(Value::Object{{"tokens", Value(Value::Array{I64(kLookupTokens[0]), I64(kLookupTokens[1]), I64(kLookupTokens[2]), I64(kLookupTokens[3])})}, {"records", Value(std::move(lookup_records))}, {"max_abs_error", F64(lookup_max_error)}, {"seconds", F64(lookup_seconds)}})},
      {"hidden_fixture", Value(Value::Object{{"elements", I64(kHidden)}, {"fnv1a64", Text(std::to_string(hidden_checksum))}, {"post_final_norm_bf16_rne", Value(true)}})},
      {"projection", Value(Value::Object{{"softcap", F64(30.0)}, {"full_vocab_logits", Value(true)}, {"reference_mode", Text("sampled_manual_row_reference")}, {"fixed_row_records", Value(std::move(fixed_records))}, {"max_abs_error", F64(projection_max_error)}, {"winner", I64(first.value().token)}, {"winner_value", F64(first.value().value)}, {"seconds", F64(first_seconds)}})},
      {"suppression", Value(Value::Object{{"suppressed_token", I64(first.value().token)}, {"winner", I64(second.value().token)}, {"winner_value", F64(second.value().value)}, {"seconds", F64(second_seconds)}})},
      {"limitations", Value(Value::Array{Text("reference-only CPU diagnostic; no quality or performance qualification"), Text("synthetic deterministic hidden fixture; unit tests own synthetic tie coverage"), Text("projection numerical comparison uses sampled manual rows; argmax and suppression scan all vocabulary logits"), Text("M07 partial artifact is not runtime-loadable")})},
  });
  const auto status = Publish(output, gem16::json::Stringify(report));
  if (!status.ok()) { std::cerr << status.message() << '\n'; return 1; }
  return 0;
#endif
}
}  // namespace

int main(int argc, char** argv) { return Run(argc, argv); }
