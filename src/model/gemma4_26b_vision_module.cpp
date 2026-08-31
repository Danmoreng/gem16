#include "model/gemma4_26b_vision_module.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "compiler/sha256.h"
#include "model/safetensors.h"
#include "platform/mapped_file.h"
#include "util/json.h"

namespace gem16::internal {
namespace {

constexpr std::uint64_t kMaximumMetadataBytes = 4U * 1024U * 1024U;
constexpr std::string_view kSourceRepository =
    "google/gemma-4-26B-A4B-it-qat-q4_0-unquantized";
constexpr std::string_view kSourceRevision =
    "f1e06dc520982d9b9edd76859fdb7ab209449949";

struct TensorSpec {
  std::string dtype;
  std::vector<std::uint64_t> shape;
};

const json::Value* Field(const json::Value& value, std::string_view name) {
  return value.is_object() ? value.find(name) : nullptr;
}

bool StringIs(const json::Value* value, std::string_view expected) {
  return value != nullptr && value->is_string() &&
         value->as_string() == expected;
}

Result<std::uint64_t> Unsigned(const json::Value* value,
                               std::string_view description) {
  if (value == nullptr || !value->is_integer() || value->as_integer() < 0) {
    return Status(StatusCode::kDataLoss,
                  "invalid Vision module integer: " +
                      std::string(description));
  }
  return static_cast<std::uint64_t>(value->as_integer());
}

Result<json::Value> LoadJson(const std::filesystem::path& path) {
  std::error_code error;
  const auto status = std::filesystem::symlink_status(path, error);
  const auto bytes = std::filesystem::file_size(path, error);
  if (error || std::filesystem::is_symlink(status) ||
      !std::filesystem::is_regular_file(status) ||
      bytes > kMaximumMetadataBytes ||
      bytes > std::numeric_limits<std::size_t>::max()) {
    return Status(StatusCode::kDataLoss,
                  "Vision module metadata is missing, unsafe or oversized: " +
                      path.string());
  }
  std::string payload(static_cast<std::size_t>(bytes), '\0');
  std::ifstream input(path, std::ios::binary);
  if (!input ||
      (bytes != 0U &&
       !input.read(payload.data(), static_cast<std::streamsize>(bytes)))) {
    return Status(StatusCode::kIoError,
                  "cannot read Vision module metadata: " + path.string());
  }
  auto parsed = json::Parse(
      payload, {.max_depth = 64, .max_values = 2'000'000,
                .max_string_bytes = kMaximumMetadataBytes});
  if (!parsed.ok() || !parsed.value().is_object()) {
    return Status(StatusCode::kDataLoss,
                  "invalid Vision module JSON object: " + path.string());
  }
  return std::move(parsed).value();
}

Result<std::string> Sha256File(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return Status(StatusCode::kIoError,
                  "cannot hash Vision module file: " + path.string());
  }
  std::vector<std::byte> buffer(8U * 1024U * 1024U);
  compiler::Sha256 digest;
  while (input) {
    input.read(reinterpret_cast<char*>(buffer.data()),
               static_cast<std::streamsize>(buffer.size()));
    const auto count = input.gcount();
    if (count > 0) digest.Update(buffer.data(), static_cast<std::size_t>(count));
  }
  if (!input.eof()) {
    return Status(StatusCode::kIoError,
                  "failed while hashing Vision module file: " +
                      path.string());
  }
  return digest.HexDigest();
}

void Add(std::map<std::string, TensorSpec, std::less<>>* result,
         std::string name, std::string dtype,
         std::initializer_list<std::uint64_t> shape) {
  result->emplace(std::move(name),
                  TensorSpec{std::move(dtype), std::vector(shape)});
}

void AddFp8(std::map<std::string, TensorSpec, std::less<>>* result,
            const std::string& weight, std::uint64_t rows,
            std::uint64_t columns) {
  Add(result, weight, "F8_E4M3", {rows, columns});
  Add(result, weight.substr(0U, weight.size() - 7U) + ".weight_scale",
      "BF16", {rows, 1U});
}

std::map<std::string, TensorSpec, std::less<>> ExpectedTensors() {
  std::map<std::string, TensorSpec, std::less<>> result;
  AddFp8(&result, "model.embed_vision.embedding_projection.weight", 2816U,
         1152U);
  AddFp8(&result, "model.vision_tower.patch_embedder.input_proj.weight",
         1152U, 768U);
  Add(&result, "model.vision_tower.patch_embedder.position_embedding_table",
      "BF16", {2U, 10240U, 1152U});
  Add(&result, "model.vision_tower.std_bias", "BF16", {1152U});
  Add(&result, "model.vision_tower.std_scale", "BF16", {1152U});
  for (std::uint32_t layer = 0U; layer < 27U; ++layer) {
    const std::string prefix = "model.vision_tower.encoder.layers." +
                               std::to_string(layer) + ".";
    for (const std::string_view name :
         {"input_layernorm.weight", "post_attention_layernorm.weight",
          "post_feedforward_layernorm.weight",
          "pre_feedforward_layernorm.weight"}) {
      Add(&result, prefix + std::string(name), "BF16", {1152U});
    }
    Add(&result, prefix + "self_attn.k_norm.weight", "BF16", {72U});
    Add(&result, prefix + "self_attn.q_norm.weight", "BF16", {72U});
    for (const std::string_view projection : {"k", "o", "q", "v"}) {
      AddFp8(&result, prefix + "self_attn." + std::string(projection) +
                           "_proj.linear.weight",
             1152U, 1152U);
    }
    AddFp8(&result, prefix + "mlp.down_proj.linear.weight", 1152U, 4304U);
    AddFp8(&result, prefix + "mlp.gate_proj.linear.weight", 4304U, 1152U);
    AddFp8(&result, prefix + "mlp.up_proj.linear.weight", 4304U, 1152U);
  }
  return result;
}

Status ValidateFileSet(const std::filesystem::path& root) {
  const std::set<std::string> expected = {
      "gem16_vision.json", "vision.gem16", "vision.lock.json",
      "vision_compilation.json"};
  std::set<std::string> actual;
  std::error_code error;
  for (std::filesystem::directory_iterator iterator(root, error), end;
       !error && iterator != end; iterator.increment(error)) {
    actual.insert(iterator->path().filename().string());
  }
  if (error || actual != expected) {
    return Status(StatusCode::kDataLoss,
                  "Vision module directory file set is invalid");
  }
  return Status::Ok();
}

}  // namespace

Result<Gemma4Moe26BVisionModulePlan>
LoadGemma4Moe26BVisionModulePlan(
    const std::filesystem::path& module_root) {
  std::error_code error;
  const auto root_status = std::filesystem::symlink_status(module_root, error);
  if (error || std::filesystem::is_symlink(root_status) ||
      !std::filesystem::is_directory(root_status)) {
    return Status(StatusCode::kDataLoss,
                  "Vision module root must be a real directory");
  }
  const auto root = std::filesystem::canonical(module_root, error);
  if (error) {
    return Status(StatusCode::kIoError,
                  "cannot resolve Vision module root");
  }
  Status status = ValidateFileSet(root);
  if (!status.ok()) return status;
  const auto artifact_path = root / "vision.gem16";
  const auto descriptor_path = root / "gem16_vision.json";
  const auto compilation_path = root / "vision_compilation.json";
  const auto lock_path = root / "vision.lock.json";
  auto descriptor = LoadJson(descriptor_path);
  auto compilation = LoadJson(compilation_path);
  auto lock = LoadJson(lock_path);
  if (!descriptor.ok()) return descriptor.status();
  if (!compilation.ok()) return compilation.status();
  if (!lock.ok()) return lock.status();
  auto artifact_hash = Sha256File(artifact_path);
  auto descriptor_hash = Sha256File(descriptor_path);
  auto compilation_hash = Sha256File(compilation_path);
  if (!artifact_hash.ok()) return artifact_hash.status();
  if (!descriptor_hash.ok()) return descriptor_hash.status();
  if (!compilation_hash.ok()) return compilation_hash.status();

  const auto artifact_bytes = std::filesystem::file_size(artifact_path, error);
  if (error ||
      !StringIs(Field(descriptor.value(), "artifact"), "vision.gem16") ||
      !StringIs(Field(descriptor.value(), "artifact_sha256"),
                artifact_hash.value()) ||
      !StringIs(Field(descriptor.value(), "capability_profile"),
                kGemma4Moe26BVisionProfile) ||
      !StringIs(Field(descriptor.value(), "enablement"),
                "explicit-profile-selection-only") ||
      !StringIs(Field(descriptor.value(), "required_text_artifact_profile"),
                kGemma4Moe26BVisionRequiredTextProfile) ||
      !StringIs(Field(descriptor.value(), "compilation_manifest_sha256"),
                compilation_hash.value()) ||
      !StringIs(Field(lock.value(), "artifact_sha256"),
                artifact_hash.value()) ||
      !StringIs(Field(lock.value(), "descriptor_sha256"),
                descriptor_hash.value()) ||
      !StringIs(Field(lock.value(), "compilation_manifest_sha256"),
                compilation_hash.value()) ||
      !StringIs(Field(lock.value(), "capability_profile"),
                kGemma4Moe26BVisionProfile) ||
      !StringIs(Field(lock.value(), "required_text_artifact_profile"),
                kGemma4Moe26BVisionRequiredTextProfile) ||
      !StringIs(Field(lock.value(), "source_lock_sha256"),
                kGemma4Moe26BVisionSourceLock) ||
      !StringIs(Field(lock.value(), "source_repository"),
                kSourceRepository) ||
      !StringIs(Field(lock.value(), "source_revision"), kSourceRevision)) {
    return Status(StatusCode::kDataLoss,
                  "Vision module descriptor or lock identity is invalid");
  }
  for (const auto& [object, field, expected] :
       std::array<std::tuple<const json::Value*, std::string_view,
                             std::uint64_t>, 4>{
           std::tuple{&descriptor.value(), "artifact_size", artifact_bytes},
           std::tuple{&lock.value(), "artifact_size", artifact_bytes},
           std::tuple{&descriptor.value(), "schema_version", 1U},
           std::tuple{&lock.value(), "schema_version", 1U}}) {
    auto value = Unsigned(Field(*object, field), field);
    if (!value.ok() || value.value() != expected) {
      return Status(StatusCode::kDataLoss,
                    "Vision module descriptor extent is invalid");
    }
  }

  const auto* artifact = Field(compilation.value(), "artifact");
  if (artifact == nullptr ||
      !StringIs(Field(compilation.value(), "capability_profile"),
                kGemma4Moe26BVisionProfile) ||
      !StringIs(Field(compilation.value(), "required_text_artifact_profile"),
                kGemma4Moe26BVisionRequiredTextProfile) ||
      !StringIs(Field(compilation.value(), "contract_id"),
                "gem16.gemma4_26b_vision_fp8") ||
      !StringIs(Field(*artifact, "filename"), "vision.gem16") ||
      !StringIs(Field(*artifact, "sha256"), artifact_hash.value())) {
    return Status(StatusCode::kDataLoss,
                  "Vision compilation profile identity is invalid");
  }
  auto payload_offset = Unsigned(Field(*artifact, "payload_offset"),
                                 "artifact.payload_offset");
  auto payload_bytes = Unsigned(Field(*artifact, "payload_bytes"),
                                "artifact.payload_bytes");
  auto tensor_count = Unsigned(Field(*artifact, "tensor_count"),
                               "artifact.tensor_count");
  if (!payload_offset.ok() || !payload_bytes.ok() || !tensor_count.ok() ||
      payload_bytes.value() != kGemma4Moe26BVisionPayloadBytes ||
      tensor_count.value() != kGemma4Moe26BVisionTensorCount ||
      payload_offset.value() > artifact_bytes ||
      artifact_bytes - payload_offset.value() != payload_bytes.value()) {
    return Status(StatusCode::kDataLoss,
                  "Vision compilation payload extent is invalid");
  }

  auto stored = LoadSafetensorsFile(artifact_path);
  if (!stored.ok()) return stored.status();
  const auto expected = ExpectedTensors();
  if (stored.value().size() != expected.size() ||
      expected.size() != kGemma4Moe26BVisionTensorCount) {
    return Status(StatusCode::kDataLoss,
                  "Vision module tensor count is invalid");
  }
  Gemma4Moe26BVisionModulePlan plan;
  plan.root = root;
  plan.artifact = artifact_path;
  plan.artifact_sha256 = artifact_hash.value();
  plan.artifact_bytes = artifact_bytes;
  plan.payload_file_offset = payload_offset.value();
  std::uint64_t cursor = 0U;
  std::uint64_t tensor_bytes = 0U;
  auto mapped = MappedFile::Open(artifact_path);
  if (!mapped.ok()) return mapped.status();
  for (const auto& tensor : stored.value()) {
    const auto spec = expected.find(tensor.name);
    if (spec == expected.end() || tensor.dtype != spec->second.dtype ||
        tensor.shape != spec->second.shape ||
        tensor.absolute_offset < plan.payload_file_offset) {
      return Status(StatusCode::kDataLoss,
                    "Vision module tensor contract mismatch: " + tensor.name);
    }
    const std::uint64_t relative =
        tensor.absolute_offset - plan.payload_file_offset;
    const std::uint64_t aligned =
        (cursor + kGemma4Moe26BVisionAlignment - 1U) &
        ~(kGemma4Moe26BVisionAlignment - 1U);
    if (relative != aligned || relative % kGemma4Moe26BVisionAlignment != 0U ||
        relative > kGemma4Moe26BVisionPayloadBytes - tensor.length) {
      return Status(StatusCode::kDataLoss,
                    "Vision module tensor alignment mismatch: " + tensor.name);
    }
    for (std::uint64_t gap = cursor; gap < relative; ++gap) {
      if (mapped.value().data()[plan.payload_file_offset + gap] !=
          std::byte{0}) {
        return Status(StatusCode::kDataLoss,
                      "Vision module contains nonzero alignment padding");
      }
    }
    if (!plan.tensors
             .emplace(tensor.name,
                      Gemma4Moe26BVisionTensorPlan{
                          relative, tensor.length, tensor.dtype, tensor.shape})
             .second) {
      return Status(StatusCode::kDataLoss,
                    "Vision module contains duplicate tensor bindings");
    }
    cursor = relative + tensor.length;
    tensor_bytes += tensor.length;
  }
  if (cursor != kGemma4Moe26BVisionPayloadBytes ||
      tensor_bytes != kGemma4Moe26BVisionTensorBytes ||
      cursor - tensor_bytes != kGemma4Moe26BVisionPaddingBytes) {
    return Status(StatusCode::kDataLoss,
                  "Vision module byte or padding balance is invalid");
  }
  return plan;
}

}  // namespace gem16::internal
