#include "model/gemma4_26b_compiled_loader.h"

#include <array>
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <string_view>

#include "compiler/sha256.h"
#include "model/gemma4_26b_manifest.h"
#include "util/json.h"

namespace gem16::internal {
namespace {

constexpr std::uint64_t kMaximumMetadataBytes = 64U * 1024U * 1024U;
constexpr std::string_view kAcceptedM08ArtifactContentSha256 =
    "471805f7dad8abb84300be78b2822a63dcb1d35bff5aa98426a162cc8532ee17";
constexpr std::string_view kAcceptedM08SourceLockSha256 =
    "3d9441fdebef33785e33181c335338208b8bf868cb8c7da692fd9c765cca8230";
constexpr std::string_view kAcceptedM08CompilerCommit =
    "f433358b8e2c1250b95801fc898faee4fcedcbe5";

Result<std::string> ReadRegularFile(const std::filesystem::path& path,
                                    std::uint64_t maximum_bytes) {
  std::error_code error;
  const auto status = std::filesystem::symlink_status(path, error);
  if (error || !std::filesystem::is_regular_file(status) ||
      std::filesystem::is_symlink(status)) {
    return Status(StatusCode::kDataLoss,
                  "compiled metadata is missing or unsafe: " + path.string());
  }
  const auto size = std::filesystem::file_size(path, error);
  if (error || size > maximum_bytes ||
      size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return Status(StatusCode::kDataLoss,
                  "compiled metadata exceeds its size limit: " + path.string());
  }
  std::string payload(static_cast<std::size_t>(size), '\0');
  std::ifstream input(path, std::ios::binary);
  if (!input || (size != 0 && !input.read(payload.data(), payload.size()))) {
    return Status(StatusCode::kIoError,
                  "cannot read compiled metadata: " + path.string());
  }
  return payload;
}

Result<json::Value> LoadJson(const std::filesystem::path& path) {
  auto payload = ReadRegularFile(path, kMaximumMetadataBytes);
  if (!payload.ok()) return payload.status();
  auto document = json::Parse(payload.value());
  if (!document.ok()) {
    return Status(StatusCode::kDataLoss,
                  "invalid compiled JSON " + path.string() + ": " +
                      document.status().message());
  }
  if (!document.value().is_object()) {
    return Status(StatusCode::kDataLoss,
                  "compiled JSON root must be an object: " + path.string());
  }
  return std::move(document).value();
}

const json::Value* Field(const json::Value& object, std::string_view name) {
  return object.is_object() ? object.find(name) : nullptr;
}

bool IsString(const json::Value* value, std::string_view expected) {
  return value != nullptr && value->is_string() && value->as_string() == expected;
}

std::filesystem::path ExternalLockPath(const std::filesystem::path& root) {
  return root.parent_path() / (root.filename().string() + ".lock.json");
}

Result<Gemma4Moe26BCompiledIdentity> AcceptedM08Identity(
    const json::Value& compilation, const json::Value& lock) {
  const auto* profile = Field(compilation, "artifact_profile");
  if (!IsString(profile, "sm120-text-hybrid-v1")) {
    return Status(StatusCode::kDataLoss,
                  "M08 product artifact_profile does not match the accepted artifact");
  }
  const auto* head = Field(compilation, "head_format");
  if (!IsString(head, "nvfp4")) {
    return Status(StatusCode::kDataLoss,
                  "M08 product head_format does not match the accepted artifact");
  }
  const auto* source = Field(compilation, "source");
  const auto* compiler = Field(compilation, "compiler");
  if (source == nullptr || !source->is_object()) {
    return Status(StatusCode::kDataLoss,
                  "M08 product source record is missing or invalid");
  }
  if (compiler == nullptr || !compiler->is_object()) {
    return Status(StatusCode::kDataLoss,
                  "M08 product compiler record is missing or invalid");
  }
  const auto* artifact_hash = Field(lock, "artifact_content_sha256");
  if (!IsString(artifact_hash, kAcceptedM08ArtifactContentSha256)) {
    return Status(
        StatusCode::kDataLoss,
        "M08 product artifact_content_sha256 does not match the accepted artifact");
  }
  const auto* source_hash = Field(lock, "source_lock_sha256");
  if (!IsString(source_hash, kAcceptedM08SourceLockSha256)) {
    return Status(
        StatusCode::kDataLoss,
        "M08 product source_lock_sha256 does not match the accepted artifact");
  }
  const auto* compiler_commit = Field(lock, "compiler_commit");
  if (!IsString(compiler_commit, kAcceptedM08CompilerCommit)) {
    return Status(StatusCode::kDataLoss,
                  "M08 product compiler_commit does not match the accepted artifact");
  }
  if (!IsString(Field(*source, "lock_sha256"), source_hash->as_string())) {
    return Status(StatusCode::kDataLoss,
                  "M08 product source.lock_sha256 is not bound to the external lock");
  }
  if (!IsString(Field(*compiler, "commit"), compiler_commit->as_string())) {
    return Status(StatusCode::kDataLoss,
                  "M08 product compiler.commit is not bound to the external lock");
  }
  return Gemma4Moe26BCompiledIdentity{
      profile->as_string(), head->as_string(), artifact_hash->as_string(),
      source_hash->as_string(), compiler_commit->as_string()};
}

bool AppendCanonicalJson(const json::Value& value, std::size_t depth,
                         std::string* output) {
  if (value.is_null()) {
    output->append("null");
  } else if (value.is_bool()) {
    output->append(value.as_bool() ? "true" : "false");
  } else if (value.is_integer()) {
    output->append(std::to_string(value.as_integer()));
  } else if (value.is_string()) {
    output->append(json::Quote(value.as_string()));
  } else if (value.is_array()) {
    output->push_back('[');
    const auto& array = value.as_array();
    for (std::size_t index = 0; index < array.size(); ++index) {
      output->append(index == 0 ? "\n" : ",\n");
      output->append((depth + 1U) * 2U, ' ');
      if (!AppendCanonicalJson(array[index], depth + 1U, output)) return false;
    }
    if (!array.empty()) {
      output->push_back('\n');
      output->append(depth * 2U, ' ');
    }
    output->push_back(']');
  } else if (value.is_object()) {
    output->push_back('{');
    std::size_t index = 0;
    for (const auto& [key, child] : value.as_object()) {
      output->append(index++ == 0 ? "\n" : ",\n");
      output->append((depth + 1U) * 2U, ' ');
      output->append(json::Quote(key));
      output->append(": ");
      if (!AppendCanonicalJson(child, depth + 1U, output)) return false;
    }
    if (!value.as_object().empty()) {
      output->push_back('\n');
      output->append(depth * 2U, ' ');
    }
    output->push_back('}');
  } else {
    // The external lock contains no floating-point values. Reject them rather
    // than risk a cross-language formatting mismatch.
    return false;
  }
  return true;
}

Result<std::string> CanonicalJsonDocument(const json::Value& value) {
  std::string output;
  if (!AppendCanonicalJson(value, 0, &output)) {
    return Status(StatusCode::kDataLoss,
                  "M08 external lock contains a noncanonical JSON value");
  }
  output.push_back('\n');
  return output;
}

Result<std::string> Sha256Range(const std::filesystem::path& path,
                                std::uint64_t offset, std::uint64_t length) {
  std::ifstream input(path, std::ios::binary);
  if (!input || offset > static_cast<std::uint64_t>(
                             std::numeric_limits<std::streamoff>::max())) {
    return Status(StatusCode::kIoError, "cannot open artifact file: " + path.string());
  }
  input.seekg(static_cast<std::streamoff>(offset));
  if (!input) {
    return Status(StatusCode::kIoError, "cannot seek artifact file: " + path.string());
  }
  compiler::Sha256 digest;
  std::array<char, 1024U * 1024U> buffer{};
  std::uint64_t remaining = length;
  while (remaining != 0) {
    const auto count = static_cast<std::streamsize>(
        std::min<std::uint64_t>(remaining, buffer.size()));
    input.read(buffer.data(), count);
    if (input.gcount() != count) {
      return Status(StatusCode::kDataLoss,
                    "short read while hashing artifact file: " + path.string());
    }
    digest.Update(buffer.data(), static_cast<std::size_t>(count));
    remaining -= static_cast<std::uint64_t>(count);
  }
  return digest.HexDigest();
}

Result<std::uint64_t> Unsigned(const json::Value* value,
                               std::string_view description) {
  if (value == nullptr || !value->is_integer() || value->as_integer() < 0) {
    return Status(StatusCode::kDataLoss,
                  "invalid nonnegative integer in compiled metadata: " +
                      std::string(description));
  }
  return static_cast<std::uint64_t>(value->as_integer());
}

Result<std::vector<std::uint64_t>> Shape(const json::Value* value,
                                         std::string_view description) {
  if (value == nullptr || !value->is_array()) {
    return Status(StatusCode::kDataLoss,
                  "invalid shape in compiled metadata: " + std::string(description));
  }
  std::vector<std::uint64_t> shape;
  for (const auto& extent : value->as_array()) {
    auto parsed = Unsigned(&extent, description);
    if (!parsed.ok()) return parsed.status();
    shape.push_back(parsed.value());
  }
  return shape;
}

Result<std::filesystem::path> SafeArtifactPath(
    const std::filesystem::path& root, const json::Value* value) {
  if (value == nullptr || !value->is_string()) {
    return Status(StatusCode::kDataLoss, "artifact lock path must be a string");
  }
  const std::filesystem::path relative(value->as_string());
  if (relative.empty() || relative.is_absolute() || relative.has_parent_path()) {
    return Status(StatusCode::kDataLoss,
                  "artifact lock contains an unsafe path: " + value->as_string());
  }
  return root / relative;
}

Status ValidateConfigExtension(const std::filesystem::path& root) {
  auto config = LoadJson(root / "config.json");
  if (!config.ok()) return config.status();
  const auto* gem16 = Field(config.value(), "gem16");
  if (gem16 == nullptr || !gem16->is_object() ||
      !IsString(Field(*gem16, "profile"), "sm120-text-hybrid-v1") ||
      !IsString(Field(*gem16, "variant"), "gemma4-26b-a4b") ||
      !IsString(Field(*gem16, "head_format"), "nvfp4-group16-divisor-v1")) {
    return Status(StatusCode::kUnsupported,
                  "compiled Gemma 4 26B config lacks the exact M08 gem16 block");
  }
  const auto schema = Unsigned(Field(*gem16, "schema_version"), "gem16.schema_version");
  const auto* text_only = Field(*gem16, "text_only");
  const auto* vision = Field(*gem16, "supports_vision");
  const auto* audio = Field(*gem16, "supports_audio");
  const auto* video = Field(*gem16, "supports_video");
  const auto* mtp = Field(*gem16, "supports_mtp");
  if (!schema.ok() || schema.value() != 1 || text_only == nullptr ||
      !text_only->is_bool() || !text_only->as_bool() || vision == nullptr ||
      !vision->is_bool() || vision->as_bool() || audio == nullptr ||
      !audio->is_bool() || audio->as_bool() || video == nullptr ||
      !video->is_bool() || video->as_bool() || mtp == nullptr ||
      !mtp->is_bool() || mtp->as_bool()) {
    return Status(StatusCode::kUnsupported,
                  "compiled Gemma 4 26B capability block is invalid");
  }
  return Status::Ok();
}

Status ValidateExternalLock(const std::filesystem::path& root,
                            const json::Value& compilation) {
  const auto lock_path = ExternalLockPath(root);
  auto lock_payload = ReadRegularFile(lock_path, kMaximumMetadataBytes);
  if (!lock_payload.ok()) return lock_payload.status();
  auto lock = LoadJson(lock_path);
  if (!lock.ok()) return lock.status();
  const std::set<std::string> expected_lock_fields = {
      "artifact_content_sha256", "artifact_profile", "artifact_status",
      "compiler_commit", "files", "schema_version", "source_lock_sha256"};
  std::set<std::string> actual_lock_fields;
  for (const auto& [name, unused] : lock.value().as_object()) {
    (void)unused;
    actual_lock_fields.insert(name);
  }
  auto canonical_lock = CanonicalJsonDocument(lock.value());
  if (actual_lock_fields != expected_lock_fields || !canonical_lock.ok() ||
      canonical_lock.value() != lock_payload.value()) {
    return Status(StatusCode::kDataLoss,
                  "M08 external lock is not the exact canonical schema");
  }
  if (!IsString(Field(lock.value(), "artifact_profile"),
                "sm120-text-hybrid-v1") ||
      !IsString(Field(lock.value(), "artifact_status"),
                "m08_complete_runtime_loadable_experimental")) {
    return Status(StatusCode::kDataLoss, "M08 external lock profile mismatch");
  }
  auto identity = AcceptedM08Identity(compilation, lock.value());
  if (!identity.ok()) return identity.status();
  const auto schema = Unsigned(Field(lock.value(), "schema_version"), "lock schema");
  const auto* files = Field(lock.value(), "files");
  if (!schema.ok() || schema.value() != 1 || files == nullptr ||
      !files->is_array()) {
    return Status(StatusCode::kDataLoss, "M08 external lock schema mismatch");
  }
  std::set<std::string> locked_names;
  std::string previous_name;
  for (const auto& record : files->as_array()) {
    const auto* relative = Field(record, "path");
    auto path = SafeArtifactPath(root, relative);
    auto size = Unsigned(Field(record, "size"), "artifact file size");
    const auto* hash = Field(record, "sha256");
    if (!record.is_object() || record.as_object().size() != 3U ||
        !path.ok() || !size.ok() || hash == nullptr || !hash->is_string() ||
        hash->as_string().size() != 64 ||
        (!previous_name.empty() && relative->as_string() <= previous_name) ||
        !locked_names.insert(relative->as_string()).second) {
      return Status(StatusCode::kDataLoss, "invalid M08 external lock file record");
    }
    previous_name = relative->as_string();
    std::error_code error;
    const auto status = std::filesystem::symlink_status(path.value(), error);
    if (error || std::filesystem::is_symlink(status) ||
        !std::filesystem::is_regular_file(status) ||
        std::filesystem::file_size(path.value(), error) != size.value()) {
      return Status(StatusCode::kDataLoss,
                    "M08 locked artifact file is missing or changed");
    }
    auto actual = Sha256Range(path.value(), 0, size.value());
    if (!actual.ok() || actual.value() != hash->as_string()) {
      return Status(StatusCode::kDataLoss,
                    "M08 locked artifact file hash mismatch: " +
                        relative->as_string());
    }
  }
  std::set<std::string> actual_names;
  std::error_code error;
  for (std::filesystem::directory_iterator iterator(root, error), end;
       !error && iterator != end; iterator.increment(error)) {
    const auto status = iterator->symlink_status(error);
    if (error || std::filesystem::is_symlink(status) ||
        !std::filesystem::is_regular_file(status)) {
      return Status(StatusCode::kDataLoss,
                    "M08 artifact contains a non-regular entry");
    }
    actual_names.insert(iterator->path().filename().string());
  }
  if (error || actual_names != locked_names) {
    return Status(StatusCode::kDataLoss,
                  "M08 external lock does not cover the exact artifact file set");
  }
  auto content_object = lock.value().as_object();
  const auto* recorded_content_hash = Field(lock.value(), "artifact_content_sha256");
  if (recorded_content_hash == nullptr || !recorded_content_hash->is_string()) {
    return Status(StatusCode::kDataLoss,
                  "M08 external lock content hash is invalid");
  }
  content_object.erase("artifact_content_sha256");
  content_object.erase("schema_version");
  auto canonical_content = CanonicalJsonDocument(json::Value(std::move(content_object)));
  if (!canonical_content.ok() ||
      compiler::Sha256Hex(canonical_content.value().data(),
                          canonical_content.value().size()) !=
          recorded_content_hash->as_string()) {
    return Status(StatusCode::kDataLoss,
                  "M08 external lock aggregate content hash mismatch");
  }
  (void)compilation;
  return Status::Ok();
}

Status ValidateTensorRecords(const std::filesystem::path& root,
                             const json::Value& compilation,
                             std::vector<TensorInfo>* tensors) {
  const auto* records = Field(compilation, "tensors");
  if (records == nullptr || !records->is_array() ||
      records->as_array().size() != 1285 || tensors == nullptr) {
    return Status(StatusCode::kDataLoss, "M08 tensor provenance count mismatch");
  }
  std::map<std::string, TensorInfo*, std::less<>> by_name;
  for (auto& tensor : *tensors) {
    if (!by_name.emplace(tensor.name, &tensor).second) {
      return Status(StatusCode::kDataLoss, "duplicate M08 tensor");
    }
  }
  if (by_name.size() != records->as_array().size()) {
    return Status(StatusCode::kDataLoss, "M08 Safetensors inventory count mismatch");
  }
  std::map<std::string, const json::Value*, std::less<>> record_by_name;
  for (const auto& record : records->as_array()) {
    const auto* name = Field(record, "output_name");
    const auto* dtype = Field(record, "output_dtype");
    const auto* shard = Field(record, "output_shard");
    auto bytes = Unsigned(Field(record, "byte_length"), "tensor byte length");
    auto shape = Shape(Field(record, "physical_shape"), "tensor physical shape");
    const auto* hash = Field(record, "sha256");
    if (name == nullptr || !name->is_string() || dtype == nullptr ||
        !dtype->is_string() || shard == nullptr || !shard->is_string() ||
        !bytes.ok() || !shape.ok() || hash == nullptr || !hash->is_string() ||
        !record_by_name.emplace(name->as_string(), &record).second) {
      return Status(StatusCode::kDataLoss, "invalid M08 tensor provenance record");
    }
    const auto found = by_name.find(name->as_string());
    if (found == by_name.end() || found->second->storage_dtype != dtype->as_string() ||
        found->second->shape != shape.value() ||
        found->second->byte_length != bytes.value() ||
        found->second->source_shard != shard->as_string()) {
      return Status(StatusCode::kDataLoss,
                    "M08 tensor record differs from Safetensors: " +
                        name->as_string());
    }
    auto actual_hash = Sha256Range(
        root / found->second->source_shard, found->second->byte_offset,
        found->second->byte_length);
    if (!actual_hash.ok() || actual_hash.value() != hash->as_string()) {
      return Status(StatusCode::kDataLoss,
                    "M08 tensor payload hash mismatch: " + name->as_string());
    }
  }
  auto annotation = ValidateAndAnnotateGemma4Moe26BCompiledHybridInventory(
      tensors, Gemma4Moe26BHeadFormat::kNvfp4);
  if (!annotation.ok()) return annotation;
  for (const auto& [name, record] : record_by_name) {
    const auto& tensor = *by_name.at(name);
    const auto* role = Field(*record, "role");
    const auto* residency = Field(*record, "residency_class");
    const auto* runtime_layout = Field(*record, "runtime_layout");
    const auto* aliased = Field(*record, "aliased");
    if (role == nullptr || !role->is_string() || residency == nullptr ||
        !residency->is_string() || runtime_layout == nullptr ||
        !runtime_layout->is_string() || aliased == nullptr || !aliased->is_bool() ||
        tensor.tensor_role != role->as_string() ||
        tensor.residency_class != residency->as_string() ||
        tensor.final_gpu_layout != runtime_layout->as_string() ||
        tensor.aliased != aliased->as_bool()) {
      return Status(StatusCode::kDataLoss,
                    "M08 tensor semantic provenance mismatch: " + name);
    }
  }
  return Status::Ok();
}

}  // namespace

Status ValidateAndBindGemma4Moe26BCompiledArtifact(
    const std::filesystem::path& model_directory,
    std::vector<TensorInfo>* tensors) {
  auto compilation = LoadJson(model_directory / "gem16_compilation.json");
  if (!compilation.ok()) return compilation.status();
  const auto schema = Unsigned(Field(compilation.value(), "schema_version"),
                               "compilation schema");
  const auto* text_only = Field(compilation.value(), "text_only");
  if (!schema.ok() || schema.value() != 1 ||
      !IsString(Field(compilation.value(), "artifact_profile"),
                "sm120-text-hybrid-v1") ||
      !IsString(Field(compilation.value(), "artifact_status"),
                "m08_complete_runtime_loadable_experimental") ||
      !IsString(Field(compilation.value(), "head_format"), "nvfp4") ||
      text_only == nullptr || !text_only->is_bool() || !text_only->as_bool()) {
    return Status(StatusCode::kUnsupported,
                  "unsupported Gemma 4 26B compiled artifact contract");
  }
  const auto* omitted = Field(compilation.value(), "omitted_families");
  const auto* excluded = Field(compilation.value(), "excluded_tensors");
  constexpr std::array<std::string_view, 4> expected_omitted = {
      "audio", "mtp", "video", "vision"};
  if (omitted == nullptr || !omitted->is_array() ||
      omitted->as_array().size() != expected_omitted.size() ||
      excluded == nullptr || !excluded->is_array() ||
      excluded->as_array().size() != 356) {
    return Status(StatusCode::kDataLoss,
                  "M08 modality omission contract mismatch");
  }
  for (std::size_t index = 0; index < expected_omitted.size(); ++index) {
    if (!IsString(&omitted->as_array()[index], expected_omitted[index])) {
      return Status(StatusCode::kDataLoss,
                    "M08 omitted modality list is not canonical");
    }
  }
  for (const auto& record : excluded->as_array()) {
    if (!IsString(Field(record, "family"), "vision") ||
        !IsString(Field(record, "residency_class"),
                  "compile_excluded_vision")) {
      return Status(StatusCode::kDataLoss,
                    "M08 exclusion contains a non-vision tensor");
    }
  }
  auto status = ValidateConfigExtension(model_directory);
  if (!status.ok()) return status;
  status = ValidateExternalLock(model_directory, compilation.value());
  if (!status.ok()) return status;
  return ValidateTensorRecords(model_directory, compilation.value(), tensors);
}

Result<Gemma4Moe26BCompiledIdentity> LoadGemma4Moe26BCompiledIdentity(
    const std::filesystem::path& model_directory) {
  auto compilation = LoadJson(model_directory / "gem16_compilation.json");
  if (!compilation.ok()) return compilation.status();
  auto lock = LoadJson(ExternalLockPath(model_directory));
  if (!lock.ok()) return lock.status();
  return AcceptedM08Identity(compilation.value(), lock.value());
}

}  // namespace gem16::internal
