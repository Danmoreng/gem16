#include "model/gemma4_26b_trellis35.h"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <set>
#include <span>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "compiler/sha256.h"
#include "platform/mapped_file.h"
#include "util/json.h"

namespace gem16::internal {
namespace {

constexpr std::uint64_t kMaximumMetadataBytes = 4U * 1024U * 1024U;
constexpr std::string_view kFormat = "GEM16-Trellis35";
constexpr std::string_view kSourceRepository =
    "google/gemma-4-26B-A4B-it-qat-q4_0-unquantized";
constexpr std::string_view kSourceRevision =
    "f1e06dc520982d9b9edd76859fdb7ab209449949";
constexpr std::string_view kM08Artifact =
    "471805f7dad8abb84300be78b2822a63dcb1d35bff5aa98426a162cc8532ee17";
constexpr std::string_view kM08Image =
    "1ed73cf105b68db937ac0992283d31fdb2225474204341440721f41fe871bb72";
constexpr std::uint64_t kM08ImageBytes = 14'696'668'160ULL;

const json::Value* Field(const json::Value& object, std::string_view name) {
  return object.is_object() ? object.find(name) : nullptr;
}

bool StringIs(const json::Value* value, std::string_view expected) {
  return value != nullptr && value->is_string() &&
         value->as_string() == expected;
}

bool BoolIs(const json::Value* value, bool expected) {
  return value != nullptr && value->is_bool() &&
         value->as_bool() == expected;
}

Result<std::uint64_t> Unsigned(const json::Value* value,
                               std::string_view description) {
  if (value == nullptr || !value->is_integer() || value->as_integer() < 0) {
    return Status(StatusCode::kDataLoss,
                  "invalid Trellis35 nonnegative integer: " +
                      std::string(description));
  }
  return static_cast<std::uint64_t>(value->as_integer());
}

bool ExactKeys(const json::Value& value,
               std::initializer_list<std::string_view> expected) {
  if (!value.is_object() || value.as_object().size() != expected.size()) {
    return false;
  }
  for (const std::string_view key : expected) {
    if (Field(value, key) == nullptr) return false;
  }
  return true;
}

bool IsSha256(const json::Value* value) {
  if (value == nullptr || !value->is_string() ||
      value->as_string().size() != 64U) {
    return false;
  }
  return std::all_of(value->as_string().begin(), value->as_string().end(),
                     [](char character) {
                       return (character >= '0' && character <= '9') ||
                              (character >= 'a' && character <= 'f');
                     });
}

bool UnsignedArrayIs(const json::Value* value,
                     std::initializer_list<std::uint64_t> expected) {
  if (value == nullptr || !value->is_array() ||
      value->as_array().size() != expected.size()) {
    return false;
  }
  std::size_t index = 0U;
  for (const std::uint64_t extent : expected) {
    auto actual = Unsigned(&value->as_array()[index++], "array extent");
    if (!actual.ok() || actual.value() != extent) return false;
  }
  return true;
}

Result<std::vector<std::uint64_t>> BoundedShape(
    const json::Value* value, std::string_view description) {
  if (value == nullptr || !value->is_array() || value->as_array().empty() ||
      value->as_array().size() > 2U) {
    return Status(StatusCode::kDataLoss,
                  "invalid Trellis35 v2 shape: " + std::string(description));
  }
  std::vector<std::uint64_t> shape;
  shape.reserve(value->as_array().size());
  std::uint64_t product = 1U;
  for (const auto& item : value->as_array()) {
    auto dimension = Unsigned(&item, description);
    if (!dimension.ok() || dimension.value() == 0U ||
        product > kTrellis35CheckpointBytes / dimension.value()) {
      return Status(StatusCode::kDataLoss,
                    "invalid Trellis35 v2 shape extent: " +
                        std::string(description));
    }
    product *= dimension.value();
    shape.push_back(dimension.value());
  }
  return shape;
}

Status ValidateDeviceImageTensorLayout(
    std::string_view dtype, std::string_view layout,
    const std::vector<std::uint64_t>& logical,
    const std::vector<std::uint64_t>& physical, std::uint64_t bytes) {
  std::uint64_t element_bytes = 0U;
  if ((dtype == "U8" && layout == "sm120_row8_k64") ||
      (dtype == "F8_E4M3" &&
       (layout == "source_nk_fp8" ||
        layout == "sm120_row8_group16_e4m3"))) {
    element_bytes = 1U;
  } else if (dtype == "BF16" &&
             (layout == "source_bf16" || layout == "row_bf16" ||
              layout == "scalar_bf16")) {
    element_bytes = 2U;
  } else if (dtype == "F32" && layout == "scalar_f32") {
    element_bytes = 4U;
  } else {
    return Status(StatusCode::kDataLoss,
                  "unsupported Trellis35 v2 tensor dtype/layout");
  }
  std::uint64_t physical_bytes = element_bytes;
  for (const std::uint64_t dimension : physical) {
    if (physical_bytes > kTrellis35CheckpointBytes / dimension) {
      return Status(StatusCode::kDataLoss,
                    "Trellis35 v2 physical shape overflows");
    }
    physical_bytes *= dimension;
  }
  bool shape_valid = false;
  if (layout == "source_bf16" || layout == "row_bf16" ||
      layout == "source_nk_fp8") {
    shape_valid = physical == logical;
  } else if (layout == "scalar_bf16" || layout == "scalar_f32") {
    shape_valid = physical == std::vector<std::uint64_t>{1U};
  } else if (logical.size() == 2U) {
    const std::uint64_t divisor =
        layout == "sm120_row8_group16_e4m3" ? 16U : 2U;
    shape_valid = logical[1] % divisor == 0U && physical.size() == 2U &&
                  physical[0] == logical[0] &&
                  physical[1] == logical[1] / divisor;
  }
  if (!shape_valid || physical_bytes != bytes) {
    return Status(StatusCode::kDataLoss,
                  "Trellis35 v2 tensor shape disagrees with its layout");
  }
  return Status::Ok();
}

Result<std::string> ReadRegularFile(const std::filesystem::path& path,
                                    std::uint64_t maximum_bytes) {
  std::error_code error;
  const auto status = std::filesystem::symlink_status(path, error);
  const auto bytes = std::filesystem::file_size(path, error);
  if (error || std::filesystem::is_symlink(status) ||
      !std::filesystem::is_regular_file(status) || bytes > maximum_bytes ||
      bytes > std::numeric_limits<std::size_t>::max()) {
    return Status(StatusCode::kDataLoss,
                  "Trellis35 metadata is missing, unsafe or oversized: " +
                      path.string());
  }
  std::string payload(static_cast<std::size_t>(bytes), '\0');
  std::ifstream input(path, std::ios::binary);
  if (!input ||
      (bytes != 0U &&
       !input.read(payload.data(), static_cast<std::streamsize>(bytes)))) {
    return Status(StatusCode::kIoError,
                  "cannot read Trellis35 metadata: " + path.string());
  }
  return payload;
}

Result<json::Value> LoadJson(const std::filesystem::path& path) {
  auto payload = ReadRegularFile(path, kMaximumMetadataBytes);
  if (!payload.ok()) return payload.status();
  auto parsed = json::Parse(payload.value());
  if (!parsed.ok() || !parsed.value().is_object()) {
    return Status(StatusCode::kDataLoss,
                  "invalid Trellis35 JSON object: " + path.string());
  }
  return std::move(parsed).value();
}

Status AppendCanonicalJson(const json::Value& value, std::size_t depth,
                           std::string* output) {
  if (value.is_null()) {
    output->append("null");
  } else if (value.is_bool()) {
    output->append(value.as_bool() ? "true" : "false");
  } else if (value.is_integer()) {
    output->append(std::to_string(value.as_integer()));
  } else if (value.is_number()) {
    std::array<char, 64> text{};
    const auto result = std::to_chars(text.data(), text.data() + text.size(),
                                      value.as_number());
    if (result.ec != std::errc{}) {
      return Status(StatusCode::kDataLoss,
                    "Trellis35 metadata contains an invalid number");
    }
    output->append(text.data(), result.ptr);
  } else if (value.is_string()) {
    output->append(json::Quote(value.as_string()));
  } else if (value.is_array()) {
    output->push_back('[');
    const auto& values = value.as_array();
    for (std::size_t index = 0; index < values.size(); ++index) {
      output->append(index == 0U ? "\n" : ",\n");
      output->append((depth + 1U) * 2U, ' ');
      Status status = AppendCanonicalJson(values[index], depth + 1U, output);
      if (!status.ok()) return status;
    }
    if (!values.empty()) {
      output->push_back('\n');
      output->append(depth * 2U, ' ');
    }
    output->push_back(']');
  } else if (value.is_object()) {
    output->push_back('{');
    std::size_t index = 0U;
    for (const auto& [name, child] : value.as_object()) {
      output->append(index++ == 0U ? "\n" : ",\n");
      output->append((depth + 1U) * 2U, ' ');
      output->append(json::Quote(name));
      output->append(": ");
      Status status = AppendCanonicalJson(child, depth + 1U, output);
      if (!status.ok()) return status;
    }
    if (!value.as_object().empty()) {
      output->push_back('\n');
      output->append(depth * 2U, ' ');
    }
    output->push_back('}');
  } else {
    return Status(StatusCode::kDataLoss,
                  "Trellis35 metadata contains an unsupported JSON value");
  }
  return Status::Ok();
}

Result<std::string> DocumentContentSha256(const json::Value& document,
                                          std::string_view field) {
  if (!document.is_object()) {
    return Status(StatusCode::kDataLoss,
                  "Trellis35 content hash requires an object");
  }
  auto object = document.as_object();
  if (object.erase(std::string(field)) != 1U) {
    return Status(StatusCode::kDataLoss,
                  "Trellis35 content hash field is missing");
  }
  std::string canonical;
  Status status = AppendCanonicalJson(json::Value(std::move(object)), 0U,
                                      &canonical);
  if (!status.ok()) return status;
  canonical.push_back('\n');
  return compiler::Sha256Hex(canonical.data(), canonical.size());
}

Result<std::string> ContentSha256(const json::Value& document) {
  return DocumentContentSha256(document, "checkpoint_content_sha256");
}

Result<std::string> Sha256File(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return Status(StatusCode::kIoError,
                  "cannot hash Trellis35 file: " + path.string());
  }
  compiler::Sha256 digest;
  std::array<char, 1024U * 1024U> buffer{};
  while (input) {
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    if (input.gcount() > 0) {
      digest.Update(buffer.data(), static_cast<std::size_t>(input.gcount()));
    }
  }
  if (!input.eof()) {
    return Status(StatusCode::kIoError,
                  "failed while hashing Trellis35 file: " + path.string());
  }
  return digest.HexDigest();
}

Result<std::filesystem::path> SafeFile(const std::filesystem::path& root,
                                       const json::Value* value,
                                       std::string_view description) {
  if (value == nullptr || !value->is_string()) {
    return Status(StatusCode::kDataLoss,
                  "Trellis35 path must be a string: " +
                      std::string(description));
  }
  const std::filesystem::path relative(value->as_string());
  if (relative.empty() || relative.is_absolute()) {
    return Status(StatusCode::kDataLoss,
                  "unsafe Trellis35 path: " + value->as_string());
  }
  std::filesystem::path candidate = root;
  for (const auto& component : relative) {
    if (component == "." || component == ".." || component.empty()) {
      return Status(StatusCode::kDataLoss,
                    "unsafe Trellis35 path: " + value->as_string());
    }
    candidate /= component;
    std::error_code error;
    const auto status = std::filesystem::symlink_status(candidate, error);
    if (error || std::filesystem::is_symlink(status)) {
      return Status(StatusCode::kDataLoss,
                    "unsafe Trellis35 path component: " +
                        candidate.string());
    }
  }
  std::error_code error;
  if (!std::filesystem::is_regular_file(candidate, error) || error) {
    return Status(StatusCode::kDataLoss,
                  "Trellis35 file is missing or not regular: " +
                      candidate.string());
  }
  return candidate;
}

Status ValidateFileExtent(const std::filesystem::path& path,
                          std::uint64_t expected) {
  std::error_code error;
  const auto size = std::filesystem::file_size(path, error);
  if (error || size != expected) {
    return Status(StatusCode::kDataLoss,
                  "Trellis35 file extent mismatch: " + path.string());
  }
  return Status::Ok();
}

constexpr std::uint64_t Align(std::uint64_t value) {
  return (value + kTrellis35Alignment - 1U) &
         ~(kTrellis35Alignment - 1U);
}

Result<std::array<std::uint16_t, kTrellis35ExpertCount>> ParseRateMap(
    const json::Value* value, std::string_view description) {
  if (value == nullptr || !value->is_array() ||
      value->as_array().size() != kTrellis35ExpertCount) {
    return Status(StatusCode::kDataLoss,
                  "invalid Trellis35 rate map: " + std::string(description));
  }
  std::array<std::uint16_t, kTrellis35ExpertCount> rates{};
  std::size_t k3 = 0U;
  std::size_t k4 = 0U;
  for (std::size_t expert = 0; expert < rates.size(); ++expert) {
    const auto& item = value->as_array()[expert];
    if (!item.is_integer() ||
        (item.as_integer() != 3 && item.as_integer() != 4)) {
      return Status(StatusCode::kDataLoss,
                    "invalid Trellis35 expert rate: " +
                        std::string(description));
    }
    rates[expert] = static_cast<std::uint16_t>(item.as_integer());
    k3 += rates[expert] == 3U ? 1U : 0U;
    k4 += rates[expert] == 4U ? 1U : 0U;
  }
  if (k3 != 64U || k4 != 64U) {
    return Status(StatusCode::kDataLoss,
                  "Trellis35 rate map must contain exactly 64 K3 and 64 K4");
  }
  return rates;
}

struct FamilyGeometry {
  std::uint64_t rows;
  std::uint64_t columns;
  std::uint64_t suh_elements;
  std::uint64_t svh_elements;
};

constexpr FamilyGeometry Geometry(bool gate_up) {
  return gate_up ? FamilyGeometry{2816U, 1408U, 2816U, 1408U}
                 : FamilyGeometry{768U, 2816U, 768U, 2816U};
}

Result<Trellis35Region> ParseRegion(const json::Value& regions,
                                    std::string_view name,
                                    std::uint64_t expected_offset,
                                    std::uint64_t expected_bytes) {
  const auto* region = Field(regions, name);
  if (region == nullptr ||
      !ExactKeys(*region, {"bytes", "offset"})) {
    return Status(StatusCode::kDataLoss,
                  "invalid Trellis35 region: " + std::string(name));
  }
  auto offset = Unsigned(Field(*region, "offset"), name);
  auto bytes = Unsigned(Field(*region, "bytes"), name);
  if (!offset.ok() || !bytes.ok() || offset.value() != expected_offset ||
      bytes.value() != expected_bytes) {
    return Status(StatusCode::kDataLoss,
                  "Trellis35 region violates the v1 layout: " +
                      std::string(name));
  }
  return Trellis35Region{offset.value(), bytes.value()};
}

Status ParseFamily(const json::Value& manifest, bool gate_up,
                   Trellis35FamilyPlan* plan, std::uint64_t* cursor) {
  const std::string name = gate_up ? "gate_up" : "down";
  const auto* maps = Field(manifest, "rate_maps");
  const auto* regions = Field(manifest, "regions");
  if (maps == nullptr ||
      !ExactKeys(*maps, {"down", "gate_up"}) || regions == nullptr ||
      !ExactKeys(*regions,
                 {"down_descriptor", "down_k3_payload_pool",
                  "down_k4_payload_pool", "down_suh", "down_svh",
                  "gate_up_descriptor", "gate_up_k3_payload_pool",
                  "gate_up_k4_payload_pool", "gate_up_suh", "gate_up_svh"}) ||
      plan == nullptr || cursor == nullptr) {
    return Status(StatusCode::kDataLoss,
                  "Trellis35 family metadata is missing: " + name);
  }
  auto rates = ParseRateMap(Field(*maps, name), name);
  if (!rates.ok()) return rates.status();
  plan->rate_map = rates.value();
  const FamilyGeometry geometry = Geometry(gate_up);
  const std::uint64_t coefficients = geometry.rows * geometry.columns;
  const std::array<std::pair<std::string, std::uint64_t>, 5> expected = {{
      {name + "_k3_payload_pool", coefficients * 3U / 8U * 64U},
      {name + "_k4_payload_pool", coefficients * 4U / 8U * 64U},
      {name + "_descriptor", kTrellis35ExpertCount * 8U},
      {name + "_suh", kTrellis35ExpertCount * geometry.suh_elements * 2U},
      {name + "_svh", kTrellis35ExpertCount * geometry.svh_elements * 2U},
  }};
  std::array<Trellis35Region*, 5> destinations = {
      &plan->k3_payload_pool, &plan->k4_payload_pool, &plan->descriptor,
      &plan->suh, &plan->svh};
  for (std::size_t index = 0; index < expected.size(); ++index) {
    *cursor = Align(*cursor);
    auto region = ParseRegion(*regions, expected[index].first, *cursor,
                              expected[index].second);
    if (!region.ok()) return region.status();
    *destinations[index] = region.value();
    *cursor += region.value().bytes;
  }
  return Status::Ok();
}

std::uint16_t LittleU16(const std::byte* source) {
  return static_cast<std::uint16_t>(
      std::to_integer<std::uint8_t>(source[0]) |
      (static_cast<std::uint16_t>(std::to_integer<std::uint8_t>(source[1]))
       << 8U));
}

std::uint32_t LittleU32(const std::byte* source) {
  return static_cast<std::uint32_t>(LittleU16(source)) |
         (static_cast<std::uint32_t>(LittleU16(source + 2U)) << 16U);
}

Status ValidateFamilyPayload(const MappedFile& mapped,
                             Trellis35FamilyPlan* plan,
                             const FamilyGeometry& geometry,
                             std::string_view description) {
  std::array<std::uint64_t, 5> offsets = {
      plan->k3_payload_pool.offset, plan->k4_payload_pool.offset,
      plan->descriptor.offset, plan->suh.offset, plan->svh.offset};
  std::array<std::uint64_t, 5> bytes = {
      plan->k3_payload_pool.bytes, plan->k4_payload_pool.bytes,
      plan->descriptor.bytes, plan->suh.bytes, plan->svh.bytes};
  for (std::size_t index = 1; index < offsets.size(); ++index) {
    const std::uint64_t prior_end = offsets[index - 1U] + bytes[index - 1U];
    for (std::uint64_t offset = prior_end; offset < offsets[index]; ++offset) {
      if (mapped.data()[offset] != std::byte{0}) {
        return Status(StatusCode::kDataLoss,
                      "nonzero Trellis35 alignment gap: " +
                          std::string(description));
      }
    }
  }
  std::array<std::uint64_t, 5> next{};
  for (std::size_t expert = 0; expert < kTrellis35ExpertCount; ++expert) {
    const std::byte* encoded =
        mapped.data() + plan->descriptor.offset + expert * 8U;
    Trellis35ExpertDescriptor descriptor{
        LittleU32(encoded), LittleU16(encoded + 4U), LittleU16(encoded + 6U)};
    const std::uint16_t rate = plan->rate_map[expert];
    if (descriptor.rate_bits != rate ||
        descriptor.codebook_id != kTrellis35CodebookId ||
        descriptor.pool_offset != next[rate]) {
      return Status(StatusCode::kDataLoss,
                    "invalid Trellis35 expert descriptor: " +
                        std::string(description));
    }
    next[rate] += geometry.rows * geometry.columns * rate / 8U;
    plan->descriptors[expert] = descriptor;
  }
  if (next[3] != plan->k3_payload_pool.bytes ||
      next[4] != plan->k4_payload_pool.bytes) {
    return Status(StatusCode::kDataLoss,
                  "Trellis35 descriptor pool coverage is incomplete: " +
                      std::string(description));
  }
  for (const Trellis35Region sidecar : {plan->suh, plan->svh}) {
    for (std::uint64_t offset = sidecar.offset;
         offset < sidecar.offset + sidecar.bytes; offset += 2U) {
      const std::uint16_t bits = LittleU16(mapped.data() + offset);
      if ((bits & 0x7c00U) == 0x7c00U) {
        return Status(StatusCode::kDataLoss,
                      "non-finite Trellis35 F16 sidecar: " +
                          std::string(description));
      }
    }
  }
  return Status::Ok();
}

Result<Trellis35LayerPlan> ParseLayer(
    const std::filesystem::path& root, const json::Value& record,
    std::uint32_t layer) {
  if (!ExactKeys(record, {"artifact", "artifact_bytes", "artifact_sha256",
                          "layer", "manifest", "manifest_sha256",
                          "verification", "verification_sha256"})) {
    return Status(StatusCode::kDataLoss,
                  "invalid Trellis35 layer index record");
  }
  auto record_layer = Unsigned(Field(record, "layer"), "layer index");
  auto artifact_bytes =
      Unsigned(Field(record, "artifact_bytes"), "layer artifact bytes");
  if (!record_layer.ok() || record_layer.value() != layer ||
      !artifact_bytes.ok() ||
      artifact_bytes.value() != kTrellis35LayerArtifactBytes ||
      !IsSha256(Field(record, "artifact_sha256")) ||
      !IsSha256(Field(record, "manifest_sha256")) ||
      !IsSha256(Field(record, "verification_sha256"))) {
    return Status(StatusCode::kDataLoss,
                  "Trellis35 layer index identity is invalid");
  }
  auto artifact = SafeFile(root, Field(record, "artifact"), "layer artifact");
  auto manifest_path =
      SafeFile(root, Field(record, "manifest"), "layer manifest");
  auto verification =
      SafeFile(root, Field(record, "verification"), "layer verification");
  if (!artifact.ok()) return artifact.status();
  if (!manifest_path.ok()) return manifest_path.status();
  if (!verification.ok()) return verification.status();
  Status extent =
      ValidateFileExtent(artifact.value(), kTrellis35LayerArtifactBytes);
  if (!extent.ok()) return extent;
  auto manifest_hash = Sha256File(manifest_path.value());
  auto verification_hash = Sha256File(verification.value());
  if (!manifest_hash.ok() ||
      manifest_hash.value() != Field(record, "manifest_sha256")->as_string() ||
      !verification_hash.ok() ||
      verification_hash.value() !=
          Field(record, "verification_sha256")->as_string()) {
    return Status(StatusCode::kDataLoss,
                  "Trellis35 layer metadata hash mismatch");
  }
  auto manifest = LoadJson(manifest_path.value());
  if (!manifest.ok()) return manifest.status();
  if (!ExactKeys(
          manifest.value(),
          {"alignment_bytes", "artifact", "calibration", "candidate_proxy",
           "checkpoint_profile", "codebook_id", "compiler", "format",
           "format_version", "gate_up_boundary",
           "gate_up_inverse_before_split", "hadamard_block", "layer",
           "logical_axis_order", "logical_shapes", "padding_contract",
           "physical_shapes", "producer_revision", "rate_maps", "regions",
           "schema_version", "source_lock_sha256", "source_repository",
           "source_revision", "source_tensors", "status", "trellis_tile"}) ||
      !StringIs(Field(manifest.value(), "format"), kFormat) ||
      !StringIs(Field(manifest.value(), "checkpoint_profile"),
                kGemma4Moe26BTrellis35Profile) ||
      !StringIs(Field(manifest.value(), "source_lock_sha256"),
                kGemma4Moe26BTrellis35SourceLock) ||
      !StringIs(Field(manifest.value(), "source_repository"),
                kSourceRepository) ||
      !StringIs(Field(manifest.value(), "source_revision"), kSourceRevision) ||
      !StringIs(Field(manifest.value(), "status"),
                "experimental_single_layer_wp2") ||
      !BoolIs(Field(manifest.value(), "gate_up_inverse_before_split"), true)) {
    return Status(StatusCode::kDataLoss,
                  "Trellis35 layer manifest profile is invalid");
  }
  const std::array<std::pair<std::string_view, std::uint64_t>, 7>
      fixed_fields = {{{"schema_version", 1U},
                       {"format_version", 1U},
                       {"layer", layer},
                       {"alignment_bytes", kTrellis35Alignment},
                       {"codebook_id", 2U},
                       {"hadamard_block", 128U},
                       {"gate_up_boundary", 704U}}};
  for (const auto& [name, expected] : fixed_fields) {
    auto actual = Unsigned(Field(manifest.value(), name), name);
    if (!actual.ok() || actual.value() != expected) {
      return Status(StatusCode::kDataLoss,
                    "Trellis35 fixed layer field is invalid: " +
                        std::string(name));
    }
  }
  const auto* padding = Field(manifest.value(), "padding_contract");
  const auto* tile = Field(manifest.value(), "trellis_tile");
  const auto* artifact_meta = Field(manifest.value(), "artifact");
  const auto* logical_shapes = Field(manifest.value(), "logical_shapes");
  const auto* physical_shapes = Field(manifest.value(), "physical_shapes");
  const auto artifact_meta_bytes =
      artifact_meta == nullptr
          ? Result<std::uint64_t>(Status(StatusCode::kDataLoss,
                                         "layer artifact metadata is missing"))
          : Unsigned(Field(*artifact_meta, "bytes"), "layer artifact bytes");
  if (padding == nullptr ||
      !ExactKeys(*padding, {"down", "gate_up"}) ||
      !StringIs(Field(*padding, "gate_up"), "none") ||
      !StringIs(Field(*padding, "down"), "input_zero_pad_704_to_768") ||
      !UnsignedArrayIs(tile, {16U, 16U}) || logical_shapes == nullptr ||
      !ExactKeys(*logical_shapes, {"down", "gate_up"}) ||
      !UnsignedArrayIs(Field(*logical_shapes, "gate_up"),
                       {128U, 1408U, 2816U}) ||
      !UnsignedArrayIs(Field(*logical_shapes, "down"),
                       {128U, 2816U, 704U}) ||
      physical_shapes == nullptr ||
      !ExactKeys(*physical_shapes, {"down", "gate_up"}) ||
      !UnsignedArrayIs(Field(*physical_shapes, "gate_up"),
                       {128U, 2816U, 1408U}) ||
      !UnsignedArrayIs(Field(*physical_shapes, "down"),
                       {128U, 768U, 2816U}) ||
      !artifact_meta ||
      !ExactKeys(*artifact_meta, {"bytes", "path", "sha256"}) ||
      !artifact_meta_bytes.ok() ||
      artifact_meta_bytes.value() != kTrellis35LayerArtifactBytes ||
      !StringIs(Field(*artifact_meta, "path"), artifact.value().filename().string()) ||
      !StringIs(Field(*artifact_meta, "sha256"),
                Field(record, "artifact_sha256")->as_string())) {
    return Status(StatusCode::kDataLoss,
                  "Trellis35 layer layout identity is invalid");
  }
  Trellis35LayerPlan result;
  result.layer = layer;
  result.arena_offset =
      kTrellis35NonRoutedBytes + layer * kTrellis35LayerArtifactBytes;
  result.artifact = {artifact.value(), kTrellis35LayerArtifactBytes,
                     Field(record, "artifact_sha256")->as_string()};
  std::uint64_t cursor = 0U;
  Status status = ParseFamily(manifest.value(), true, &result.gate_up, &cursor);
  if (!status.ok()) return status;
  status = ParseFamily(manifest.value(), false, &result.down, &cursor);
  if (!status.ok()) return status;
  if (Align(cursor) != kTrellis35LayerArtifactBytes) {
    return Status(StatusCode::kDataLoss,
                  "Trellis35 layer extent does not close exactly");
  }
  status = ValidateGemma4Moe26BTrellis35LayerPayload(&result);
  if (!status.ok()) return status;
  return result;
}

Status ParseNonRouted(const std::filesystem::path& root,
                      const json::Value& identity,
                      Gemma4Moe26BTrellis35CheckpointPlan* plan) {
  if (!ExactKeys(identity,
                 {"artifact", "artifact_bytes", "artifact_sha256", "manifest",
                  "manifest_sha256", "tensor_count"}) ||
      !IsSha256(Field(identity, "artifact_sha256")) ||
      !IsSha256(Field(identity, "manifest_sha256"))) {
    return Status(StatusCode::kDataLoss,
                  "invalid Trellis35 non-routed identity");
  }
  auto bytes = Unsigned(Field(identity, "artifact_bytes"), "non-routed bytes");
  auto count = Unsigned(Field(identity, "tensor_count"), "non-routed count");
  auto artifact =
      SafeFile(root, Field(identity, "artifact"), "non-routed artifact");
  auto manifest_path =
      SafeFile(root, Field(identity, "manifest"), "non-routed manifest");
  if (!bytes.ok() || bytes.value() != kTrellis35NonRoutedBytes ||
      !count.ok() || count.value() != 1045U || !artifact.ok() ||
      !manifest_path.ok()) {
    return Status(StatusCode::kDataLoss,
                  "Trellis35 non-routed file identity is invalid");
  }
  Status extent = ValidateFileExtent(artifact.value(), bytes.value());
  if (!extent.ok()) return extent;
  auto manifest_hash = Sha256File(manifest_path.value());
  if (!manifest_hash.ok() ||
      manifest_hash.value() !=
          Field(identity, "manifest_sha256")->as_string()) {
    return Status(StatusCode::kDataLoss,
                  "Trellis35 non-routed manifest hash mismatch");
  }
  auto manifest = LoadJson(manifest_path.value());
  if (!manifest.ok()) return manifest.status();
  const auto* tensors = Field(manifest.value(), "tensors");
  const auto* excluded = Field(manifest.value(), "excluded_source_roles");
  if (!ExactKeys(manifest.value(),
                 {"bytes", "checkpoint_profile", "excluded_source_roles",
                  "schema_version", "sha256",
                  "source_artifact_content_sha256", "source_image_bytes",
                  "source_image_sha256", "source_lock_sha256", "status",
                  "tensor_count", "tensors"}) ||
      !StringIs(Field(manifest.value(), "checkpoint_profile"),
                kGemma4Moe26BTrellis35Profile) ||
      !StringIs(Field(manifest.value(), "source_lock_sha256"),
                kGemma4Moe26BTrellis35SourceLock) ||
      !StringIs(Field(manifest.value(), "status"),
                "wp2_non_routed_import_from_accepted_direct_bf16_derivative") ||
      !StringIs(Field(manifest.value(), "source_artifact_content_sha256"),
                kM08Artifact) ||
      !StringIs(Field(manifest.value(), "source_image_sha256"), kM08Image) ||
      !StringIs(Field(manifest.value(), "sha256"),
                Field(identity, "artifact_sha256")->as_string()) ||
      excluded == nullptr || !excluded->is_array() ||
      excluded->as_array().size() != 2U ||
      !StringIs(&excluded->as_array()[0], "routed_expert_down") ||
      !StringIs(&excluded->as_array()[1], "routed_expert_gate_up") ||
      tensors == nullptr || !tensors->is_array() ||
      tensors->as_array().size() != 1045U) {
    return Status(StatusCode::kDataLoss,
                  "Trellis35 non-routed manifest contract is invalid");
  }
  for (const auto& [name, expected] :
       std::array<std::pair<std::string_view, std::uint64_t>, 4>{{
           {"schema_version", 1U},
           {"tensor_count", 1045U},
           {"bytes", kTrellis35NonRoutedBytes},
           {"source_image_bytes", kM08ImageBytes},
       }}) {
    auto actual = Unsigned(Field(manifest.value(), name), name);
    if (!actual.ok() || actual.value() != expected) {
      return Status(StatusCode::kDataLoss,
                    "Trellis35 non-routed fixed field is invalid: " +
                        std::string(name));
    }
  }
  std::uint64_t cursor = 0U;
  std::string previous_name;
  for (const auto& tensor : tensors->as_array()) {
    if (!ExactKeys(tensor,
                   {"bytes", "destination_offset", "logical_shape", "name",
                    "physical_shape", "role", "runtime_layout", "sha256",
                    "source_image_offset", "storage_dtype"})) {
      return Status(StatusCode::kDataLoss,
                    "Trellis35 non-routed tensor schema is invalid");
    }
    const auto* name = Field(tensor, "name");
    const auto* role = Field(tensor, "role");
    auto offset =
        Unsigned(Field(tensor, "destination_offset"), "tensor offset");
    auto tensor_bytes = Unsigned(Field(tensor, "bytes"), "tensor bytes");
    if (name == nullptr || !name->is_string() || name->as_string().empty() ||
        (!previous_name.empty() && name->as_string() <= previous_name) ||
        role == nullptr || !role->is_string() ||
        role->as_string().starts_with("routed_expert_") || !offset.ok() ||
        offset.value() != Align(cursor) || !tensor_bytes.ok() ||
        tensor_bytes.value() == 0U || !IsSha256(Field(tensor, "sha256")) ||
        tensor_bytes.value() > kTrellis35NonRoutedBytes - offset.value()) {
      return Status(StatusCode::kDataLoss,
                    "Trellis35 non-routed tensor identity is invalid");
    }
    if (!plan->non_routed_tensors
             .emplace(name->as_string(),
                      Trellis35NonRoutedTensor{offset.value(),
                                               tensor_bytes.value()})
             .second) {
      return Status(StatusCode::kDataLoss,
                    "duplicate Trellis35 non-routed tensor");
    }
    previous_name = name->as_string();
    cursor = offset.value() + tensor_bytes.value();
  }
  if (Align(cursor) != kTrellis35NonRoutedBytes) {
    return Status(StatusCode::kDataLoss,
                  "Trellis35 non-routed tensor extent is incomplete");
  }
  plan->non_routed = {artifact.value(), kTrellis35NonRoutedBytes,
                      Field(identity, "artifact_sha256")->as_string()};
  return Status::Ok();
}

Status ParseDeviceImageFamily(const json::Value& value, bool gate_up,
                              Trellis35FamilyPlan* plan,
                              std::uint64_t* cursor) {
  const std::string name = gate_up ? "gate_up" : "down";
  const auto* regions = Field(value, "regions");
  if (!ExactKeys(value, {"rate_map", "regions"}) ||
      regions == nullptr ||
      !ExactKeys(*regions,
                 {"descriptor", "k3_payload_pool", "k4_payload_pool",
                  "suh", "svh"}) ||
      plan == nullptr || cursor == nullptr) {
    return Status(StatusCode::kDataLoss,
                  "Trellis35 v2 family metadata is invalid: " + name);
  }
  auto rates = ParseRateMap(Field(value, "rate_map"), name);
  if (!rates.ok()) return rates.status();
  plan->rate_map = rates.value();
  const FamilyGeometry geometry = Geometry(gate_up);
  const std::uint64_t coefficients = geometry.rows * geometry.columns;
  const std::array<std::pair<std::string_view, std::uint64_t>, 5> expected = {{
      {"k3_payload_pool", coefficients * 3U / 8U * 64U},
      {"k4_payload_pool", coefficients * 4U / 8U * 64U},
      {"descriptor", kTrellis35ExpertCount * 8U},
      {"suh", kTrellis35ExpertCount * geometry.suh_elements * 2U},
      {"svh", kTrellis35ExpertCount * geometry.svh_elements * 2U},
  }};
  std::array<Trellis35Region*, 5> destinations = {
      &plan->k3_payload_pool, &plan->k4_payload_pool, &plan->descriptor,
      &plan->suh, &plan->svh};
  for (std::size_t index = 0U; index < expected.size(); ++index) {
    *cursor = Align(*cursor);
    auto region = ParseRegion(*regions, expected[index].first, *cursor,
                              expected[index].second);
    if (!region.ok()) return region.status();
    *destinations[index] = region.value();
    *cursor += region.value().bytes;
  }
  return Status::Ok();
}

Status ParseDeviceImageNonRouted(
    const json::Value& compilation,
    Gemma4Moe26BTrellis35CheckpointPlan* plan) {
  const auto* tensors = Field(compilation, "non_routed_tensors");
  if (plan == nullptr || tensors == nullptr || !tensors->is_array() ||
      tensors->as_array().size() != 1045U) {
    return Status(StatusCode::kDataLoss,
                  "Trellis35 v2 non-routed tensor table is invalid");
  }
  std::uint64_t cursor = 0U;
  std::string previous_name;
  for (const auto& tensor : tensors->as_array()) {
    if (!ExactKeys(tensor,
                   {"bytes", "logical_shape", "name", "offset",
                    "physical_shape", "runtime_layout", "storage_dtype"})) {
      return Status(StatusCode::kDataLoss,
                    "Trellis35 v2 non-routed tensor schema is invalid");
    }
    const auto* name = Field(tensor, "name");
    const auto* storage_dtype = Field(tensor, "storage_dtype");
    const auto* runtime_layout = Field(tensor, "runtime_layout");
    auto logical_shape =
        BoundedShape(Field(tensor, "logical_shape"), "logical shape");
    auto physical_shape =
        BoundedShape(Field(tensor, "physical_shape"), "physical shape");
    auto offset = Unsigned(Field(tensor, "offset"), "v2 tensor offset");
    auto bytes = Unsigned(Field(tensor, "bytes"), "v2 tensor bytes");
    if (name == nullptr || !name->is_string() || name->as_string().empty() ||
        (!previous_name.empty() && name->as_string() <= previous_name) ||
        storage_dtype == nullptr || !storage_dtype->is_string() ||
        runtime_layout == nullptr || !runtime_layout->is_string() ||
        !logical_shape.ok() || !physical_shape.ok() ||
        !offset.ok() || offset.value() != Align(cursor) || !bytes.ok() ||
        bytes.value() == 0U || offset.value() > kTrellis35NonRoutedBytes ||
        bytes.value() > kTrellis35NonRoutedBytes - offset.value() ||
        !plan->non_routed_tensors
             .emplace(name->as_string(),
                      Trellis35NonRoutedTensor{offset.value(), bytes.value()})
             .second) {
      return Status(StatusCode::kDataLoss,
                    "Trellis35 v2 non-routed tensor identity is invalid");
    }
    Status layout = ValidateDeviceImageTensorLayout(
        storage_dtype->as_string(), runtime_layout->as_string(),
        logical_shape.value(), physical_shape.value(), bytes.value());
    if (!layout.ok()) return layout;
    previous_name = name->as_string();
    cursor = offset.value() + bytes.value();
  }
  if (Align(cursor) != kTrellis35NonRoutedBytes) {
    return Status(StatusCode::kDataLoss,
                  "Trellis35 v2 non-routed tensor extent is incomplete");
  }
  return Status::Ok();
}

Result<Trellis35LayerPlan> ParseDeviceImageLayer(
    const json::Value& value, std::uint32_t expected_layer,
    const std::filesystem::path& image) {
  if (!ExactKeys(value, {"arena_offset", "down", "gate_up", "layer"})) {
    return Status(StatusCode::kDataLoss,
                  "Trellis35 v2 layer schema is invalid");
  }
  auto layer = Unsigned(Field(value, "layer"), "v2 layer");
  auto arena_offset =
      Unsigned(Field(value, "arena_offset"), "v2 layer arena offset");
  const std::uint64_t expected_offset =
      kTrellis35NonRoutedBytes +
      static_cast<std::uint64_t>(expected_layer) *
          kTrellis35LayerArtifactBytes;
  if (!layer.ok() || layer.value() != expected_layer || !arena_offset.ok() ||
      arena_offset.value() != expected_offset) {
    return Status(StatusCode::kDataLoss,
                  "Trellis35 v2 layer ordering is invalid");
  }
  Trellis35LayerPlan result;
  result.layer = expected_layer;
  result.arena_offset = expected_offset;
  result.artifact = {image, kTrellis35LayerArtifactBytes, {}};
  std::uint64_t cursor = 0U;
  const auto* gate_up = Field(value, "gate_up");
  const auto* down = Field(value, "down");
  if (gate_up == nullptr || down == nullptr) {
    return Status(StatusCode::kDataLoss,
                  "Trellis35 v2 layer family is missing");
  }
  Status status =
      ParseDeviceImageFamily(*gate_up, true, &result.gate_up, &cursor);
  if (!status.ok()) return status;
  status = ParseDeviceImageFamily(*down, false, &result.down, &cursor);
  if (!status.ok()) return status;
  if (Align(cursor) != kTrellis35LayerArtifactBytes) {
    return Status(StatusCode::kDataLoss,
                  "Trellis35 v2 layer extent is incomplete");
  }
  return result;
}

Result<std::filesystem::path> NamedSafeFile(
    const std::filesystem::path& root, std::string_view name,
    std::string_view description) {
  const json::Value value{std::string(name)};
  return SafeFile(root, &value, description);
}

Status RejectMixedDeviceImageLayout(const std::filesystem::path& root) {
  for (const std::string_view name :
       {"trellis35-checkpoint.json", "trellis35-experts.json",
        "non-routed.gem16", "non-routed.json"}) {
    std::error_code error;
    const auto status =
        std::filesystem::symlink_status(root / name, error);
    if (!error &&
        status.type() != std::filesystem::file_type::not_found) {
      return Status(StatusCode::kDataLoss,
                    "mixed Trellis35 v1/v2 checkpoint is forbidden");
    }
  }
  for (std::uint32_t layer = 0U; layer < kTrellis35LayerCount; ++layer) {
    std::array<char, 9> name{};
    std::snprintf(name.data(), name.size(), "layer-%02u", layer);
    std::error_code error;
    const auto status =
        std::filesystem::symlink_status(root / name.data(), error);
    if (!error &&
        status.type() != std::filesystem::file_type::not_found) {
      return Status(StatusCode::kDataLoss,
                    "mixed Trellis35 v1/v2 layer directory is forbidden");
    }
  }
  return Status::Ok();
}

}  // namespace

Status ValidateGemma4Moe26BTrellis35LayerPayload(
    Trellis35LayerPlan* layer) {
  if (layer == nullptr ||
      layer->artifact.bytes != kTrellis35LayerArtifactBytes) {
    return Status(StatusCode::kInvalidArgument,
                  "invalid Trellis35 layer payload plan");
  }
  Status extent = ValidateFileExtent(layer->artifact.path,
                                     kTrellis35LayerArtifactBytes);
  if (!extent.ok()) return extent;
  std::uint64_t cursor = 0U;
  std::array<const Trellis35Region*, 10> ordered{};
  std::size_t region_index = 0U;
  for (auto* family : {&layer->gate_up, &layer->down}) {
    const bool gate_up = family == &layer->gate_up;
    const FamilyGeometry geometry = Geometry(gate_up);
    const std::uint64_t coefficients = geometry.rows * geometry.columns;
    const std::array<std::pair<Trellis35Region*, std::uint64_t>, 5> regions = {{
        {&family->k3_payload_pool, coefficients * 3U / 8U * 64U},
        {&family->k4_payload_pool, coefficients * 4U / 8U * 64U},
        {&family->descriptor, kTrellis35ExpertCount * 8U},
        {&family->suh, kTrellis35ExpertCount * geometry.suh_elements * 2U},
        {&family->svh, kTrellis35ExpertCount * geometry.svh_elements * 2U},
    }};
    for (const auto& [region, expected_bytes] : regions) {
      cursor = Align(cursor);
      if (region->offset != cursor || region->bytes != expected_bytes) {
        return Status(StatusCode::kDataLoss,
                      "Trellis35 layer plan violates the exact v1 regions");
      }
      ordered[region_index++] = region;
      cursor += region->bytes;
    }
  }
  if (Align(cursor) != kTrellis35LayerArtifactBytes) {
    return Status(StatusCode::kDataLoss,
                  "Trellis35 layer plan does not close at the exact extent");
  }
  auto mapped = MappedFile::Open(layer->artifact.path);
  if (!mapped.ok()) return mapped.status();
  std::uint64_t prior_end = 0U;
  for (const Trellis35Region* region : ordered) {
    for (std::uint64_t offset = prior_end; offset < region->offset; ++offset) {
      if (mapped.value().data()[offset] != std::byte{0}) {
        return Status(StatusCode::kDataLoss,
                      "Trellis35 layer has a nonzero alignment gap");
      }
    }
    prior_end = region->offset + region->bytes;
  }
  for (std::uint64_t offset = prior_end; offset < mapped.value().size();
       ++offset) {
    if (mapped.value().data()[offset] != std::byte{0}) {
      return Status(StatusCode::kDataLoss,
                    "Trellis35 layer has a nonzero final alignment gap");
    }
  }
  Status status = ValidateFamilyPayload(mapped.value(), &layer->gate_up,
                                        Geometry(true), "gate_up");
  if (!status.ok()) return status;
  return ValidateFamilyPayload(mapped.value(), &layer->down, Geometry(false),
                               "down");
}

Result<Gemma4Moe26BTrellis35CheckpointPlan>
LoadGemma4Moe26BTrellis35CheckpointPlan(
    const std::filesystem::path& checkpoint_root) {
  std::error_code error;
  const auto root_status =
      std::filesystem::symlink_status(checkpoint_root, error);
  if (error || std::filesystem::is_symlink(root_status) ||
      !std::filesystem::is_directory(root_status)) {
    return Status(StatusCode::kDataLoss,
                  "Trellis35 checkpoint root must be a real directory");
  }
  const auto root = std::filesystem::canonical(checkpoint_root, error);
  if (error) {
    return Status(StatusCode::kIoError,
                  "cannot resolve Trellis35 checkpoint root");
  }
  const json::Value checkpoint_name(std::string("trellis35-checkpoint.json"));
  const json::Value experts_name(std::string("trellis35-experts.json"));
  auto checkpoint_path =
      SafeFile(root, &checkpoint_name, "checkpoint manifest");
  auto experts_path = SafeFile(root, &experts_name, "expert index");
  if (!checkpoint_path.ok()) return checkpoint_path.status();
  if (!experts_path.ok()) return experts_path.status();
  auto checkpoint = LoadJson(checkpoint_path.value());
  auto experts = LoadJson(experts_path.value());
  if (!checkpoint.ok()) return checkpoint.status();
  if (!experts.ok()) return experts.status();
  auto checkpoint_hash = ContentSha256(checkpoint.value());
  auto experts_hash = ContentSha256(experts.value());
  if (!checkpoint_hash.ok()) return checkpoint_hash.status();
  if (!experts_hash.ok()) return experts_hash.status();
  if (!ExactKeys(checkpoint.value(),
                 {"arena", "checkpoint_content_sha256", "checkpoint_profile",
                  "format", "format_version", "non_routed",
                  "routed_experts", "runtime_supported", "schema_version",
                  "source_lock_sha256", "source_repository",
                  "source_revision", "status"}) ||
      !StringIs(Field(checkpoint.value(), "checkpoint_content_sha256"),
                checkpoint_hash.value()) ||
      !StringIs(Field(checkpoint.value(), "checkpoint_profile"),
                kGemma4Moe26BTrellis35Profile) ||
      !StringIs(Field(checkpoint.value(), "format"), kFormat) ||
      !StringIs(Field(checkpoint.value(), "source_lock_sha256"),
                kGemma4Moe26BTrellis35SourceLock) ||
      !StringIs(Field(checkpoint.value(), "source_repository"),
                kSourceRepository) ||
      !StringIs(Field(checkpoint.value(), "source_revision"),
                kSourceRevision) ||
      !StringIs(
          Field(checkpoint.value(), "status"),
          "wp7_complete_text_only_runtime_characterized") ||
      !BoolIs(Field(checkpoint.value(), "runtime_supported"), true)) {
    return Status(StatusCode::kDataLoss,
                  "Trellis35 checkpoint profile is invalid");
  }
  for (const auto& [name, expected] :
       std::array<std::pair<std::string_view, std::uint64_t>, 2>{{
           {"schema_version", 1U}, {"format_version", 1U}}}) {
    auto actual = Unsigned(Field(checkpoint.value(), name), name);
    if (!actual.ok() || actual.value() != expected) {
      return Status(StatusCode::kDataLoss,
                    "Trellis35 checkpoint schema field is invalid: " +
                        std::string(name));
    }
  }
  const auto* arena = Field(checkpoint.value(), "arena");
  const auto* routed = Field(checkpoint.value(), "routed_experts");
  const auto* non_routed = Field(checkpoint.value(), "non_routed");
  if (arena == nullptr ||
      !ExactKeys(*arena,
                 {"alignment_bytes", "non_routed_bytes",
                  "nvfp4_routed_expert_bytes",
                  "one_immutable_device_representation", "total_bytes",
                  "trellis35_routed_expert_bytes"}) ||
      routed == nullptr || non_routed == nullptr ||
      !BoolIs(Field(*arena, "one_immutable_device_representation"), true)) {
    return Status(StatusCode::kDataLoss,
                  "Trellis35 arena contract is invalid");
  }
  const std::array<std::pair<std::string_view, std::uint64_t>, 5>
      arena_fields = {{{"alignment_bytes", kTrellis35Alignment},
                       {"non_routed_bytes", kTrellis35NonRoutedBytes},
                       {"nvfp4_routed_expert_bytes", 0U},
                       {"total_bytes", kTrellis35CheckpointBytes},
                       {"trellis35_routed_expert_bytes",
                        kTrellis35RoutedExpertBytes}}};
  for (const auto& [name, expected] : arena_fields) {
    auto actual = Unsigned(Field(*arena, name), name);
    if (!actual.ok() || actual.value() != expected) {
      return Status(StatusCode::kDataLoss,
                    "Trellis35 arena field is invalid: " +
                        std::string(name));
    }
  }
  auto index_sha = Sha256File(experts_path.value());
  if (!ExactKeys(*routed,
                 {"bytes", "checkpoint_content_sha256", "index",
                  "index_sha256", "layer_count", "layers"}) ||
      !StringIs(Field(*routed, "index"), "trellis35-experts.json") ||
      !index_sha.ok() ||
      !StringIs(Field(*routed, "index_sha256"), index_sha.value()) ||
      !StringIs(Field(*routed, "checkpoint_content_sha256"),
                experts_hash.value()) ||
      !ExactKeys(experts.value(),
                 {"calibration_corpus_sha256", "checkpoint_content_sha256",
                  "checkpoint_profile", "down_padding", "format",
                  "format_version", "gate_up_padding", "layer_count",
                  "layers", "payload_bpw_encoded", "routed_expert_bytes",
                  "schema_version", "source_lock_sha256", "status"}) ||
      !StringIs(Field(experts.value(), "checkpoint_content_sha256"),
                experts_hash.value()) ||
      !StringIs(Field(experts.value(), "checkpoint_profile"),
                kGemma4Moe26BTrellis35Profile) ||
      !StringIs(Field(experts.value(), "format"), kFormat) ||
      !StringIs(Field(experts.value(), "source_lock_sha256"),
                kGemma4Moe26BTrellis35SourceLock) ||
      !StringIs(Field(experts.value(), "status"),
                "wp2_complete_30_layer_routed_expert_artifact") ||
      !StringIs(Field(experts.value(), "gate_up_padding"), "none") ||
      !StringIs(Field(experts.value(), "down_padding"),
                "input_zero_pad_704_to_768")) {
    return Status(StatusCode::kDataLoss,
                  "Trellis35 routed-expert index is invalid");
  }
  for (const auto& [name, expected] :
       std::array<std::pair<std::string_view, std::uint64_t>, 4>{{
           {"schema_version", 1U},
           {"format_version", 1U},
           {"layer_count", kTrellis35LayerCount},
           {"routed_expert_bytes", kTrellis35RoutedExpertBytes},
       }}) {
    auto actual = Unsigned(Field(experts.value(), name), name);
    if (!actual.ok() || actual.value() != expected) {
      return Status(StatusCode::kDataLoss,
                    "Trellis35 expert-index field is invalid: " +
                        std::string(name));
    }
  }
  for (const auto& [name, expected] :
       std::array<std::pair<std::string_view, std::uint64_t>, 2>{{
           {"layer_count", kTrellis35LayerCount},
           {"bytes", kTrellis35RoutedExpertBytes},
       }}) {
    auto actual = Unsigned(Field(*routed, name), name);
    if (!actual.ok() || actual.value() != expected) {
      return Status(StatusCode::kDataLoss,
                    "Trellis35 routed checkpoint field is invalid: " +
                        std::string(name));
    }
  }
  const auto* bpw = Field(experts.value(), "payload_bpw_encoded");
  if (bpw == nullptr || !bpw->is_number() || bpw->as_number() != 3.5 ||
      !IsSha256(Field(experts.value(), "calibration_corpus_sha256"))) {
    return Status(StatusCode::kDataLoss,
                  "Trellis35 expert rate or calibration identity is invalid");
  }
  const auto* layers = Field(experts.value(), "layers");
  const auto* routed_layers = Field(*routed, "layers");
  if (layers == nullptr || !layers->is_array() ||
      layers->as_array().size() != kTrellis35LayerCount ||
      routed_layers == nullptr ||
      json::Stringify(*routed_layers) != json::Stringify(*layers)) {
    return Status(StatusCode::kDataLoss,
                  "Trellis35 layer index is incomplete or inconsistent");
  }
  Gemma4Moe26BTrellis35CheckpointPlan plan;
  plan.checkpoint_root = root;
  plan.checkpoint_content_sha256 = checkpoint_hash.value();
  plan.arena_bytes = kTrellis35CheckpointBytes;
  plan.nvfp4_routed_expert_bytes = 0U;
  Status non_routed_status = ParseNonRouted(root, *non_routed, &plan);
  if (!non_routed_status.ok()) return non_routed_status;
  for (std::uint32_t layer = 0; layer < kTrellis35LayerCount; ++layer) {
    auto parsed = ParseLayer(root, layers->as_array()[layer], layer);
    if (!parsed.ok()) return parsed.status();
    plan.layers[layer] = std::move(parsed).value();
  }
  return plan;
}

Result<Gemma4Moe26BTrellis35CheckpointPlan>
LoadGemma4Moe26BTrellis35DeviceImagePlan(
    const std::filesystem::path& checkpoint_root) {
  std::error_code error;
  const auto root_status =
      std::filesystem::symlink_status(checkpoint_root, error);
  if (error || std::filesystem::is_symlink(root_status) ||
      !std::filesystem::is_directory(root_status)) {
    return Status(StatusCode::kDataLoss,
                  "Trellis35 v2 root must be a real directory");
  }
  const auto root = std::filesystem::canonical(checkpoint_root, error);
  if (error) {
    return Status(StatusCode::kIoError,
                  "cannot resolve Trellis35 v2 root");
  }
  Status mixed = RejectMixedDeviceImageLayout(root);
  if (!mixed.ok()) return mixed;
  auto model_path =
      NamedSafeFile(root, "gem16_model.json", "v2 model metadata");
  auto compilation_path = NamedSafeFile(
      root, "gem16_compilation.json", "v2 compilation metadata");
  auto lock_path = NamedSafeFile(root, "gem16.lock.json", "v2 lock");
  if (!model_path.ok()) return model_path.status();
  if (!compilation_path.ok()) return compilation_path.status();
  if (!lock_path.ok()) return lock_path.status();
  auto model = LoadJson(model_path.value());
  auto compilation = LoadJson(compilation_path.value());
  auto lock = LoadJson(lock_path.value());
  if (!model.ok()) return model.status();
  if (!compilation.ok()) return compilation.status();
  if (!lock.ok()) return lock.status();
  auto model_content =
      DocumentContentSha256(model.value(), "model_content_sha256");
  auto compilation_content = DocumentContentSha256(
      compilation.value(), "compilation_content_sha256");
  auto lock_content =
      DocumentContentSha256(lock.value(), "artifact_content_sha256");
  if (!model_content.ok() || !compilation_content.ok() ||
      !lock_content.ok() ||
      !StringIs(Field(model.value(), "model_content_sha256"),
                model_content.ok() ? model_content.value() : "") ||
      !StringIs(Field(compilation.value(), "compilation_content_sha256"),
                compilation_content.ok() ? compilation_content.value() : "") ||
      !StringIs(Field(lock.value(), "artifact_content_sha256"),
                lock_content.ok() ? lock_content.value() : "")) {
    return Status(StatusCode::kDataLoss,
                  "Trellis35 v2 metadata content hash mismatch");
  }
  constexpr std::string_view kPayloadPolicy =
      "verified_at_build_download_install_not_runtime_load";
  if (!ExactKeys(
          model.value(),
          {"arena", "artifact_profile", "format", "format_version",
           "model_bytes", "model_content_sha256", "model_file",
           "model_sha256", "qualification_state",
           "runtime_payload_sha256_policy", "runtime_supported",
           "schema_version", "source_checkpoint_content_sha256",
           "source_lock_sha256"}) ||
      !ExactKeys(
          compilation.value(),
          {"arena_bytes", "artifact_profile",
           "compilation_content_sha256", "format", "format_version",
           "layer_bytes", "layer_count", "layers", "non_routed_bytes",
           "non_routed_tensors", "schema_version",
           "source_checkpoint_content_sha256"}) ||
      !ExactKeys(lock.value(),
                 {"artifact_content_sha256", "artifact_profile", "files",
                  "format", "format_version",
                  "runtime_payload_sha256_policy", "schema_version",
                  "source_checkpoint_content_sha256",
                  "source_lock_sha256"})) {
    return Status(StatusCode::kDataLoss,
                  "Trellis35 v2 metadata schema is invalid");
  }
  const auto* source_identity =
      Field(model.value(), "source_checkpoint_content_sha256");
  if (!StringIs(Field(model.value(), "format"),
                kGemma4Moe26BTrellis35DeviceImageFormat) ||
      !StringIs(Field(compilation.value(), "format"),
                kGemma4Moe26BTrellis35DeviceImageFormat) ||
      !StringIs(Field(lock.value(), "format"),
                kGemma4Moe26BTrellis35DeviceImageFormat) ||
      !StringIs(Field(model.value(), "artifact_profile"),
                kGemma4Moe26BTrellis35Profile) ||
      !StringIs(Field(compilation.value(), "artifact_profile"),
                kGemma4Moe26BTrellis35Profile) ||
      !StringIs(Field(lock.value(), "artifact_profile"),
                kGemma4Moe26BTrellis35Profile) ||
      !StringIs(Field(model.value(), "source_lock_sha256"),
                kGemma4Moe26BTrellis35SourceLock) ||
      !StringIs(Field(lock.value(), "source_lock_sha256"),
                kGemma4Moe26BTrellis35SourceLock) ||
      !StringIs(Field(model.value(), "qualification_state"),
                "production_candidate") ||
      !StringIs(Field(model.value(), "runtime_payload_sha256_policy"),
                kPayloadPolicy) ||
      !StringIs(Field(lock.value(), "runtime_payload_sha256_policy"),
                kPayloadPolicy) ||
      !BoolIs(Field(model.value(), "runtime_supported"), true) ||
      !IsSha256(source_identity) ||
      !StringIs(Field(compilation.value(),
                      "source_checkpoint_content_sha256"),
                source_identity->as_string()) ||
      !StringIs(Field(lock.value(), "source_checkpoint_content_sha256"),
                source_identity->as_string()) ||
      !IsSha256(Field(model.value(), "model_sha256")) ||
      !StringIs(Field(model.value(), "model_file"), "model.gem16")) {
    return Status(StatusCode::kDataLoss,
                  "Trellis35 v2 component identity is invalid");
  }
  for (const json::Value* document :
       {&model.value(), &compilation.value(), &lock.value()}) {
    auto schema = Unsigned(Field(*document, "schema_version"), "v2 schema");
    auto version = Unsigned(Field(*document, "format_version"), "v2 format");
    if (!schema.ok() || schema.value() != 1U || !version.ok() ||
        version.value() != 2U) {
      return Status(StatusCode::kDataLoss,
                    "Trellis35 v2 schema version is invalid");
    }
  }
  for (const auto& [name, expected] :
       std::array<std::pair<std::string_view, std::uint64_t>, 5>{
           {{"arena_bytes", kTrellis35CheckpointBytes},
            {"non_routed_bytes", kTrellis35NonRoutedBytes},
            {"layer_count", kTrellis35LayerCount},
            {"layer_bytes", kTrellis35LayerArtifactBytes},
            {"model_bytes", kTrellis35CheckpointBytes}}}) {
    const json::Value& document =
        name == "model_bytes" ? model.value() : compilation.value();
    auto value = Unsigned(Field(document, name), name);
    if (!value.ok() || value.value() != expected) {
      return Status(StatusCode::kDataLoss,
                    "Trellis35 v2 fixed extent is invalid: " +
                        std::string(name));
    }
  }
  const auto* arena = Field(model.value(), "arena");
  if (arena == nullptr ||
      !ExactKeys(*arena,
                 {"alignment_bytes", "layer_base_offset", "layer_count",
                  "layer_stride_bytes", "non_routed_bytes",
                  "non_routed_offset", "total_bytes"})) {
    return Status(StatusCode::kDataLoss,
                  "Trellis35 v2 arena schema is invalid");
  }
  for (const auto& [name, expected] :
       std::array<std::pair<std::string_view, std::uint64_t>, 7>{
           {{"alignment_bytes", kTrellis35Alignment},
            {"layer_base_offset", kTrellis35NonRoutedBytes},
            {"layer_count", kTrellis35LayerCount},
            {"layer_stride_bytes", kTrellis35LayerArtifactBytes},
            {"non_routed_bytes", kTrellis35NonRoutedBytes},
            {"non_routed_offset", 0U},
            {"total_bytes", kTrellis35CheckpointBytes}}}) {
    auto value = Unsigned(Field(*arena, name), name);
    if (!value.ok() || value.value() != expected) {
      return Status(StatusCode::kDataLoss,
                    "Trellis35 v2 arena extent is invalid: " +
                        std::string(name));
    }
  }
  auto image =
      NamedSafeFile(root, "model.gem16", "Trellis35 v2 model payload");
  if (!image.ok()) return image.status();
  Status image_extent =
      ValidateFileExtent(image.value(), kTrellis35CheckpointBytes);
  if (!image_extent.ok()) return image_extent;

  const auto* files = Field(lock.value(), "files");
  if (files == nullptr || !files->is_array() ||
      files->as_array().size() != 3U) {
    return Status(StatusCode::kDataLoss,
                  "Trellis35 v2 lock file table is invalid");
  }
  std::map<std::string, std::pair<std::uint64_t, std::string>, std::less<>>
      locked_files;
  for (const auto& file : files->as_array()) {
    if (!ExactKeys(file, {"path", "sha256", "size"}) ||
        !IsSha256(Field(file, "sha256"))) {
      return Status(StatusCode::kDataLoss,
                    "Trellis35 v2 lock record is invalid");
    }
    const auto* path = Field(file, "path");
    auto size = Unsigned(Field(file, "size"), "v2 locked file size");
    if (path == nullptr || !path->is_string() || !size.ok() ||
        !locked_files
             .emplace(path->as_string(),
                      std::make_pair(size.value(),
                                     Field(file, "sha256")->as_string()))
             .second) {
      return Status(StatusCode::kDataLoss,
                    "Trellis35 v2 lock file identity is invalid");
    }
  }
  const auto locked_model = locked_files.find("model.gem16");
  const auto locked_model_metadata = locked_files.find("gem16_model.json");
  const auto locked_compilation =
      locked_files.find("gem16_compilation.json");
  auto model_metadata_sha = Sha256File(model_path.value());
  auto compilation_sha = Sha256File(compilation_path.value());
  if (locked_model == locked_files.end() ||
      locked_model_metadata == locked_files.end() ||
      locked_compilation == locked_files.end() ||
      locked_model->second.first != kTrellis35CheckpointBytes ||
      locked_model->second.second !=
          Field(model.value(), "model_sha256")->as_string() ||
      locked_model_metadata->second.first !=
          std::filesystem::file_size(model_path.value(), error) || error ||
      !model_metadata_sha.ok() ||
      model_metadata_sha.value() != locked_model_metadata->second.second ||
      locked_compilation->second.first !=
          std::filesystem::file_size(compilation_path.value(), error) ||
      error || !compilation_sha.ok() ||
      compilation_sha.value() != locked_compilation->second.second) {
    return Status(StatusCode::kDataLoss,
                  "Trellis35 v2 lock metadata mismatch");
  }

  Gemma4Moe26BTrellis35CheckpointPlan plan;
  plan.checkpoint_root = root;
  plan.checkpoint_content_sha256 = source_identity->as_string();
  plan.non_routed =
      {image.value(), kTrellis35CheckpointBytes,
       Field(model.value(), "model_sha256")->as_string()};
  plan.arena_bytes = kTrellis35CheckpointBytes;
  plan.nvfp4_routed_expert_bytes = 0U;
  Status non_routed = ParseDeviceImageNonRouted(compilation.value(), &plan);
  if (!non_routed.ok()) return non_routed;
  const auto* layers = Field(compilation.value(), "layers");
  if (layers == nullptr || !layers->is_array() ||
      layers->as_array().size() != kTrellis35LayerCount) {
    return Status(StatusCode::kDataLoss,
                  "Trellis35 v2 layer table is invalid");
  }
  for (std::uint32_t layer = 0U; layer < kTrellis35LayerCount; ++layer) {
    auto parsed =
        ParseDeviceImageLayer(layers->as_array()[layer], layer, image.value());
    if (!parsed.ok()) return parsed.status();
    plan.layers[layer] = std::move(parsed).value();
  }
  return plan;
}

Status Gemma4Moe26BTrellis35EngineDispatchStatus() {
  return Status::Ok();
}

}  // namespace gem16::internal
