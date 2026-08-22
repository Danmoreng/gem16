#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "cuda/nvfp4/sm120_layout.h"
#include "gem16/model.h"
#include "model/gemma4_26b_residency.h"
#include "platform/mapped_file.h"

namespace {

constexpr std::uint64_t kMiB = 1024U * 1024U;
constexpr std::uint64_t kPrimaryWeightLimitBytes = 14100U * kMiB;
constexpr std::uint64_t kHardWeightStopBytes = 14300U * kMiB;
constexpr std::uint64_t kMaximumIdleBaselineBytes = 512U * kMiB;
constexpr std::uint64_t kUploadStagingBytes = 4U * kMiB;

struct Options {
  int device = 0;
  std::filesystem::path model;
  std::filesystem::path output;
  std::string code_revision = "unknown";
};

struct DeviceRegion {
  std::string name;
  std::uint64_t requested_bytes = 0;
  void* pointer = nullptr;
  std::uint64_t free_after_bytes = 0;
};

struct UploadStats {
  std::uint64_t tensor_count = 0;
  std::uint64_t payload_bytes = 0;
  std::uint64_t shard_count = 0;
  std::uint64_t direct_tensor_count = 0;
  std::uint64_t nvfp4_weight_tensor_count = 0;
  std::uint64_t nvfp4_scale_tensor_count = 0;
  std::uint64_t host_staging_peak_bytes = 0;
  double duration_seconds = 0.0;
};

struct ProfileMeasurement {
  std::uint64_t context_tokens = 0;
  std::uint64_t kv_bytes = 0;
  std::uint64_t required_margin_bytes = 0;
  std::uint64_t free_before_bytes = 0;
  std::uint64_t projected_final_free_bytes = 0;
  std::uint64_t margin_shortfall_bytes = 0;
  std::uint64_t final_free_bytes = 0;
  std::uint64_t free_after_release_bytes = 0;
  bool preflight_admitted = false;
  bool allocations_complete = false;
  bool margin_pass = false;
  std::string error;
  std::vector<DeviceRegion> regions;
};

void Usage(std::ostream& output) {
  output << "Usage: gem16-26b-memory-probe --model <artifact> "
            "--output <report.json> [--device <index>] "
            "[--code-revision <sha>]\n";
}

bool ParseInteger(std::string_view text, int* value) {
  try {
    std::size_t consumed = 0;
    const int parsed = std::stoi(std::string(text), &consumed);
    if (consumed != text.size() || parsed < 0) return false;
    *value = parsed;
    return true;
  } catch (...) {
    return false;
  }
}

bool ParseOptions(int argc, char** argv, Options* options) {
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--help") {
      Usage(std::cout);
      return false;
    }
    if (index + 1 >= argc) {
      std::cerr << "error: missing value for " << argument << '\n';
      return false;
    }
    const std::string_view value(argv[++index]);
    if (argument == "--model") {
      options->model = value;
    } else if (argument == "--output") {
      options->output = value;
    } else if (argument == "--device") {
      if (!ParseInteger(value, &options->device)) {
        std::cerr << "error: invalid CUDA device index\n";
        return false;
      }
    } else if (argument == "--code-revision") {
      options->code_revision = value;
    } else {
      std::cerr << "error: unknown argument: " << argument << '\n';
      return false;
    }
  }
  if (options->model.empty() || options->output.empty()) {
    std::cerr << "error: --model and --output are required\n";
    return false;
  }
  return true;
}

std::string JsonEscape(std::string_view value) {
  std::ostringstream output;
  for (const unsigned char byte : value) {
    switch (byte) {
      case '"': output << "\\\""; break;
      case '\\': output << "\\\\"; break;
      case '\b': output << "\\b"; break;
      case '\f': output << "\\f"; break;
      case '\n': output << "\\n"; break;
      case '\r': output << "\\r"; break;
      case '\t': output << "\\t"; break;
      default:
        if (byte < 0x20U) {
          output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                 << static_cast<unsigned int>(byte) << std::dec;
        } else {
          output << static_cast<char>(byte);
        }
    }
  }
  return output.str();
}

std::string UuidText(const cudaUUID_t& uuid) {
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (std::size_t index = 0; index < sizeof(uuid.bytes); ++index) {
    if (index == 4 || index == 6 || index == 8 || index == 10) output << '-';
    output << std::setw(2)
           << static_cast<unsigned int>(
                  static_cast<unsigned char>(uuid.bytes[index]));
  }
  return output.str();
}

bool Cuda(cudaError_t result, std::string_view operation,
          std::string* error_message) {
  if (result == cudaSuccess) return true;
  *error_message = std::string(operation) + ": " + cudaGetErrorName(result) +
                   ": " + cudaGetErrorString(result);
  return false;
}

bool Measure(std::uint64_t* free_bytes, std::uint64_t* total_bytes,
             std::string_view operation, std::string* error_message) {
  std::size_t free = 0;
  std::size_t total = 0;
  if (!Cuda(cudaMemGetInfo(&free, &total), operation, error_message)) return false;
  *free_bytes = free;
  *total_bytes = total;
  return true;
}

bool Release(std::vector<DeviceRegion>* regions, std::string* error_message) {
  bool released = true;
  for (auto region = regions->rbegin(); region != regions->rend(); ++region) {
    if (region->pointer == nullptr) continue;
    const auto result = cudaFree(region->pointer);
    if (result != cudaSuccess) {
      if (!error_message->empty()) *error_message += "; ";
      *error_message += "cudaFree(" + region->name + "): " +
                        cudaGetErrorString(result);
      released = false;
    }
    region->pointer = nullptr;
  }
  return released;
}

gem16::Result<std::uint64_t> CheckedMultiply(
    std::span<const std::uint64_t> factors, std::string_view label) {
  std::uint64_t result = 1U;
  for (const std::uint64_t factor : factors) {
    if (factor != 0U &&
        result > std::numeric_limits<std::uint64_t>::max() / factor) {
      return gem16::Status(gem16::StatusCode::kDataLoss,
                           std::string(label) + " geometry overflows uint64");
    }
    result *= factor;
  }
  return result;
}

gem16::Result<std::pair<std::uint64_t, std::uint64_t>> FlattenGeometry(
    const std::vector<std::uint64_t>& shape, std::uint64_t contracting_factor,
    std::string_view tensor_name) {
  if (shape.size() < 2U || contracting_factor == 0U) {
    return gem16::Status(gem16::StatusCode::kDataLoss,
                         "invalid NVFP4 geometry: " + std::string(tensor_name));
  }
  auto rows = CheckedMultiply(
      std::span<const std::uint64_t>(shape.data(), shape.size() - 1U),
      tensor_name);
  if (!rows.ok()) return rows.status();
  const std::array<std::uint64_t, 2> contracting_factors = {
      shape.back(), contracting_factor};
  auto contracting = CheckedMultiply(contracting_factors, tensor_name);
  if (!contracting.ok()) return contracting.status();
  return std::make_pair(rows.value(), contracting.value());
}

gem16::Status UploadSm120TiledBlocks(
    std::byte* destination, std::span<const std::uint8_t> source,
    const gem16::internal::Sm120Nvfp4SourceLayout& layout,
    std::uint64_t bytes_per_k_block, std::uint64_t source_row_bytes,
    std::string_view operation, std::uint64_t* host_staging_peak_bytes) {
  if (destination == nullptr || bytes_per_k_block == 0U ||
      source_row_bytes != layout.k_blocks * bytes_per_k_block ||
      source.size() != layout.rows * source_row_bytes) {
    return gem16::Status(gem16::StatusCode::kDataLoss,
                         "invalid source buffer for " + std::string(operation));
  }
  constexpr std::uint64_t kRowsPerTile = 8U;
  const std::uint64_t full_tile_bytes =
      kRowsPerTile * layout.k_blocks * bytes_per_k_block;
  const std::uint64_t tiles_per_batch = std::max<std::uint64_t>(
      1U, kUploadStagingBytes / std::max<std::uint64_t>(1U, full_tile_bytes));
  const std::uint64_t staging_bytes = std::min<std::uint64_t>(
      source.size(), tiles_per_batch * full_tile_bytes);
  std::vector<std::uint8_t> staging(static_cast<std::size_t>(staging_bytes));
  *host_staging_peak_bytes =
      std::max(*host_staging_peak_bytes, staging_bytes);

  for (std::uint64_t first_tile = 0; first_tile < layout.row_tiles;
       first_tile += tiles_per_batch) {
    const std::uint64_t end_tile =
        std::min(layout.row_tiles, first_tile + tiles_per_batch);
    const std::uint64_t first_row = first_tile * kRowsPerTile;
    std::uint64_t cursor = 0U;
    for (std::uint64_t row_tile = first_tile; row_tile < end_tile; ++row_tile) {
      const std::uint64_t tile_first_row = row_tile * kRowsPerTile;
      const std::uint64_t tile_rows =
          std::min(kRowsPerTile, layout.rows - tile_first_row);
      for (std::uint64_t k_block = 0; k_block < layout.k_blocks; ++k_block) {
        for (std::uint64_t row = 0; row < tile_rows; ++row) {
          const std::uint64_t source_offset =
              (tile_first_row + row) * source_row_bytes +
              k_block * bytes_per_k_block;
          std::copy_n(source.data() + source_offset, bytes_per_k_block,
                      staging.data() + cursor);
          cursor += bytes_per_k_block;
        }
      }
    }
    const std::uint64_t destination_offset =
        first_row * layout.k_blocks * bytes_per_k_block;
    const cudaError_t error = cudaMemcpy(
        destination + destination_offset, staging.data(),
        static_cast<std::size_t>(cursor), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
      return gem16::Status(
          gem16::StatusCode::kInternal,
          std::string(operation) + ": " + cudaGetErrorName(error) + ": " +
              cudaGetErrorString(error));
    }
  }
  return gem16::Status::Ok();
}

gem16::Status UploadTensor(
    std::byte* arena, const gem16::TensorInfo& tensor,
    const gem16::internal::Gemma4Moe26BUploadRange& range,
    const gem16::internal::MappedFile& mapped, UploadStats* stats) {
  if (range.source_offset > mapped.size() ||
      range.bytes > mapped.size() - range.source_offset ||
      range.bytes > std::numeric_limits<std::size_t>::max()) {
    return gem16::Status(gem16::StatusCode::kDataLoss,
                         "invalid M09 source range: " + tensor.name);
  }
  const auto* source = reinterpret_cast<const std::uint8_t*>(
      mapped.data() + range.source_offset);
  std::byte* destination = arena + range.destination_offset;

  if (range.runtime_layout.ends_with("sm120_row8_k64")) {
    auto geometry = FlattenGeometry(tensor.logical_shape, 1U, tensor.name);
    if (!geometry.ok()) return geometry.status();
    const auto [rows, contracting] = geometry.value();
    auto layout = gem16::internal::PlanSm120Nvfp4SourceLayout(rows, contracting);
    if (!layout.ok()) return layout.status();
    const std::uint64_t source_row_bytes = contracting / 2U;
    if (range.bytes != rows * source_row_bytes) {
      return gem16::Status(gem16::StatusCode::kDataLoss,
                           "M09 packed NVFP4 byte count mismatch: " +
                               tensor.name);
    }
    auto status = UploadSm120TiledBlocks(
        destination, {source, static_cast<std::size_t>(range.bytes)},
        layout.value(), 32U, source_row_bytes, "upload tiled NVFP4 weight",
        &stats->host_staging_peak_bytes);
    if (!status.ok()) return status;
    ++stats->nvfp4_weight_tensor_count;
  } else if (range.runtime_layout.ends_with("sm120_row8_group16_e4m3")) {
    auto geometry = FlattenGeometry(tensor.shape, 16U, tensor.name);
    if (!geometry.ok()) return geometry.status();
    const auto [rows, contracting] = geometry.value();
    auto layout = gem16::internal::PlanSm120Nvfp4SourceLayout(rows, contracting);
    if (!layout.ok()) return layout.status();
    const std::uint64_t source_row_bytes = tensor.shape.back();
    if (range.bytes != rows * source_row_bytes) {
      return gem16::Status(gem16::StatusCode::kDataLoss,
                           "M09 NVFP4 scale byte count mismatch: " +
                               tensor.name);
    }
    auto status = UploadSm120TiledBlocks(
        destination, {source, static_cast<std::size_t>(range.bytes)},
        layout.value(), 4U, source_row_bytes, "upload tiled NVFP4 scale",
        &stats->host_staging_peak_bytes);
    if (!status.ok()) return status;
    ++stats->nvfp4_scale_tensor_count;
  } else {
    const cudaError_t error = cudaMemcpy(
        destination, source, static_cast<std::size_t>(range.bytes),
        cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
      return gem16::Status(
          gem16::StatusCode::kInternal,
          "upload checkpoint tensor " + tensor.name + ": " +
              cudaGetErrorName(error) + ": " + cudaGetErrorString(error));
    }
    ++stats->direct_tensor_count;
  }
  ++stats->tensor_count;
  stats->payload_bytes += range.bytes;
  return gem16::Status::Ok();
}

gem16::Status UploadArtifact(
    const std::filesystem::path& model, const gem16::ModelManifest& manifest,
    const gem16::internal::Gemma4Moe26BResidencyPlan& plan,
    std::byte* arena, UploadStats* stats) {
  std::map<std::string, const gem16::TensorInfo*, std::less<>> tensors;
  for (const auto& tensor : manifest.tensors) {
    tensors.emplace(tensor.name, &tensor);
  }
  std::set<std::string> shards;
  for (const auto& range : plan.upload_ranges) shards.insert(range.source_shard);
  stats->shard_count = shards.size();
  const auto started = std::chrono::steady_clock::now();
  for (const auto& shard : shards) {
    auto mapped = gem16::internal::MappedFile::Open(model / shard);
    if (!mapped.ok()) return mapped.status();
    for (const auto& range : plan.upload_ranges) {
      if (range.source_shard != shard) continue;
      const auto tensor = tensors.find(range.tensor_name);
      if (tensor == tensors.end()) {
        return gem16::Status(gem16::StatusCode::kDataLoss,
                             "M09 upload tensor is missing from manifest");
      }
      auto status = UploadTensor(arena, *tensor->second, range, mapped.value(),
                                 stats);
      if (!status.ok()) return status;
    }
  }
  const auto finished = std::chrono::steady_clock::now();
  stats->duration_seconds =
      std::chrono::duration<double>(finished - started).count();
  if (stats->tensor_count != plan.upload_ranges.size() ||
      stats->payload_bytes != plan.artifact_payload_bytes) {
    return gem16::Status(gem16::StatusCode::kDataLoss,
                         "M09 upload did not consume the exact artifact");
  }
  const cudaError_t synchronized = cudaDeviceSynchronize();
  if (synchronized != cudaSuccess) {
    return gem16::Status(
        gem16::StatusCode::kInternal,
        "synchronize uploaded artifact: " +
            std::string(cudaGetErrorName(synchronized)) + ": " +
            cudaGetErrorString(synchronized));
  }
  return gem16::Status::Ok();
}

ProfileMeasurement RunProfile(
    const gem16::internal::Gemma4Moe26BResidencyPlan& plan,
    const gem16::internal::Gemma4Moe26BContextResidency& profile) {
  ProfileMeasurement measurement;
  measurement.context_tokens = profile.context_tokens;
  measurement.kv_bytes = profile.fp8_kv_bytes;
  measurement.required_margin_bytes = profile.required_free_margin_bytes;
  std::uint64_t ignored_total = 0U;
  if (!Measure(&measurement.free_before_bytes, &ignored_total,
               "cudaMemGetInfo(before profile)", &measurement.error)) {
    return measurement;
  }
  const std::uint64_t variable_bytes =
      profile.fp8_kv_bytes + plan.fixed_region_bytes;
  measurement.projected_final_free_bytes =
      measurement.free_before_bytes > variable_bytes
          ? measurement.free_before_bytes - variable_bytes
          : 0U;
  measurement.margin_shortfall_bytes =
      measurement.projected_final_free_bytes < measurement.required_margin_bytes
          ? measurement.required_margin_bytes -
                measurement.projected_final_free_bytes
          : 0U;
  auto admission = gem16::internal::CheckGemma4Moe26BAdmission(
      plan, profile.context_tokens, measurement.free_before_bytes, false);
  measurement.preflight_admitted = admission.ok();
  if (!admission.ok()) {
    measurement.error = admission.message();
    measurement.free_after_release_bytes = measurement.free_before_bytes;
    return measurement;
  }

  measurement.regions.push_back({"fp8_kv", profile.fp8_kv_bytes});
  for (const auto& fixed : plan.fixed_regions) {
    measurement.regions.push_back({fixed.name, fixed.bytes});
  }
  std::uint64_t total_bytes = 0U;
  for (auto& region : measurement.regions) {
    if (region.requested_bytes > std::numeric_limits<std::size_t>::max() ||
        !Cuda(cudaMalloc(&region.pointer,
                         static_cast<std::size_t>(region.requested_bytes)),
              "cudaMalloc(" + region.name + ')', &measurement.error) ||
        !Cuda(cudaMemset(region.pointer, 0,
                         static_cast<std::size_t>(region.requested_bytes)),
              "cudaMemset(" + region.name + ')', &measurement.error) ||
        !Cuda(cudaDeviceSynchronize(),
              "cudaDeviceSynchronize(" + region.name + ')',
              &measurement.error) ||
        !Measure(&region.free_after_bytes, &total_bytes,
                 "cudaMemGetInfo(" + region.name + ')', &measurement.error)) {
      break;
    }
    measurement.final_free_bytes = region.free_after_bytes;
  }
  measurement.allocations_complete =
      measurement.error.empty() &&
      std::all_of(measurement.regions.begin(), measurement.regions.end(),
                  [](const DeviceRegion& region) {
                    return region.pointer != nullptr &&
                           region.free_after_bytes != 0U;
                  });
  measurement.margin_pass =
      measurement.allocations_complete &&
      measurement.final_free_bytes >= measurement.required_margin_bytes;

  const bool released = Release(&measurement.regions, &measurement.error);
  if (!Measure(&measurement.free_after_release_bytes, &total_bytes,
               "cudaMemGetInfo(after profile release)", &measurement.error)) {
    measurement.margin_pass = false;
  }
  if (!released || measurement.free_after_release_bytes <
                       measurement.free_before_bytes) {
    measurement.margin_pass = false;
    if (measurement.error.empty()) {
      measurement.error = "profile release did not restore CUDA-visible memory";
    }
  }
  return measurement;
}

bool WriteReport(
    const Options& options, const cudaDeviceProp& properties,
    int runtime_version, int driver_version,
    const gem16::ModelManifest& manifest,
    const gem16::internal::Gemma4Moe26BResidencyPlan& plan,
    const UploadStats& upload, std::uint64_t visible_total_bytes,
    std::uint64_t free_after_context_bytes,
    std::uint64_t free_after_weights_bytes,
    std::uint64_t free_before_second_slot_bytes,
    std::uint64_t free_after_second_slot_check_bytes,
    std::uint64_t free_after_release_bytes,
    const std::vector<ProfileMeasurement>& profiles,
    std::uint64_t max_context_candidate_tokens,
    bool second_slot_rejected, bool passed, std::string_view error_message) {
  std::ofstream output(options.output, std::ios::binary | std::ios::trunc);
  if (!output) {
    std::cerr << "error: cannot create report: " << options.output << '\n';
    return false;
  }
  output << "{\n"
         << "  \"schema_version\": 1,\n"
         << "  \"milestone\": \"M09\",\n"
         << "  \"status\": \"" << (passed ? "pass" : "fail") << "\",\n"
         << "  \"code_revision\": \"" << JsonEscape(options.code_revision)
         << "\",\n"
         << "  \"probe_kind\": \"real_artifact_residency\",\n"
         << "  \"not_model_execution\": true,\n"
         << "  \"model_directory\": \"" << JsonEscape(options.model.string())
         << "\",\n"
         << "  \"device\": {\"index\":" << options.device
         << ",\"name\":\"" << JsonEscape(properties.name)
         << "\",\"uuid\":\"" << UuidText(properties.uuid)
         << "\",\"compute_capability\":\"" << properties.major << '.'
         << properties.minor << "\",\"runtime_version\":" << runtime_version
         << ",\"driver_version\":" << driver_version << "},\n"
         << "  \"artifact\": {\n"
         << "    \"validation_contract\": \""
         << JsonEscape(manifest.validation_contract) << "\",\n"
         << "    \"tensor_count\": " << manifest.tensors.size() << ",\n"
         << "    \"shard_count\": " << upload.shard_count << ",\n"
         << "    \"payload_bytes\": " << plan.artifact_payload_bytes << ",\n"
         << "    \"immutable_weight_arena_bytes\": "
         << plan.immutable_weight_arena_bytes << ",\n"
         << "    \"one_device_weight_arena\": true,\n"
         << "    \"persistent_device_repack_bytes\": 0,\n"
         << "    \"mapped_shards_one_at_a_time\": true\n"
         << "  },\n"
         << "  \"upload\": {\n"
         << "    \"status\": \""
         << (upload.tensor_count == manifest.tensors.size() ? "pass" : "fail")
         << "\",\n"
         << "    \"tensor_count\": " << upload.tensor_count << ",\n"
         << "    \"payload_bytes\": " << upload.payload_bytes << ",\n"
         << "    \"direct_tensor_count\": " << upload.direct_tensor_count
         << ",\n"
         << "    \"nvfp4_weight_tensor_count\": "
         << upload.nvfp4_weight_tensor_count << ",\n"
         << "    \"nvfp4_scale_tensor_count\": "
         << upload.nvfp4_scale_tensor_count << ",\n"
         << "    \"host_staging_peak_bytes\": "
         << upload.host_staging_peak_bytes
         << ",\n"
         << "    \"duration_seconds\": " << std::fixed
         << std::setprecision(6) << upload.duration_seconds << "\n"
         << "  },\n"
         << "  \"measurement\": {\n"
         << "    \"cuda_visible_total_bytes\": " << visible_total_bytes << ",\n"
         << "    \"free_after_context_bytes\": " << free_after_context_bytes
         << ",\n"
         << "    \"free_after_weights_bytes\": " << free_after_weights_bytes
         << ",\n"
         << "    \"free_before_second_slot_bytes\": "
         << free_before_second_slot_bytes << ",\n"
         << "    \"free_after_second_slot_check_bytes\": "
         << free_after_second_slot_check_bytes << ",\n"
         << "    \"free_after_release_bytes\": " << free_after_release_bytes
         << ",\n"
         << "    \"measured_max_context_candidate_tokens\": "
         << max_context_candidate_tokens << "\n"
         << "  },\n"
         << "  \"profiles\": [\n";
  for (std::size_t profile_index = 0; profile_index < profiles.size();
       ++profile_index) {
    const auto& profile = profiles[profile_index];
    output << "    {\"context_tokens\":" << profile.context_tokens
           << ",\"fp8_kv_bytes\":" << profile.kv_bytes
           << ",\"required_margin_bytes\":" << profile.required_margin_bytes
           << ",\"free_before_bytes\":" << profile.free_before_bytes
           << ",\"projected_final_free_bytes\":"
           << profile.projected_final_free_bytes
           << ",\"margin_shortfall_bytes\":"
           << profile.margin_shortfall_bytes
           << ",\"final_free_bytes\":" << profile.final_free_bytes
           << ",\"free_after_release_bytes\":"
           << profile.free_after_release_bytes
           << ",\"preflight_admitted\":"
           << (profile.preflight_admitted ? "true" : "false")
           << ",\"allocations_complete\":"
           << (profile.allocations_complete ? "true" : "false")
           << ",\"margin_pass\":"
           << (profile.margin_pass ? "true" : "false")
           << ",\"error\":\"" << JsonEscape(profile.error)
           << "\",\"regions\":[";
    for (std::size_t region_index = 0;
         region_index < profile.regions.size(); ++region_index) {
      const auto& region = profile.regions[region_index];
      if (region_index != 0U) output << ',';
      output << "{\"name\":\"" << JsonEscape(region.name)
             << "\",\"requested_bytes\":" << region.requested_bytes
             << ",\"free_after_bytes\":" << region.free_after_bytes << '}';
    }
    output << "]}" << (profile_index + 1U == profiles.size() ? "\n" : ",\n");
  }
  output << "  ],\n"
         << "  \"gates\": {\n"
         << "    \"idle_baseline_at_most_512_mib\": "
         << (visible_total_bytes - free_after_context_bytes <=
                     kMaximumIdleBaselineBytes
                 ? "true"
                 : "false")
         << ",\n"
         << "    \"immutable_weights_at_most_14100_mib\": "
         << (plan.immutable_weight_arena_bytes <= kPrimaryWeightLimitBytes
                 ? "true"
                 : "false")
         << ",\n"
         << "    \"immutable_weights_below_14300_mib_hard_stop\": "
         << (plan.immutable_weight_arena_bytes <= kHardWeightStopBytes
                 ? "true"
                 : "false")
         << ",\n"
         << "    \"real_32k_slot_at_least_700_mib\": "
         << (profiles.size() >= 3U && profiles[2].margin_pass ? "true" : "false")
         << ",\n"
         << "    \"base_profiles_measured\": "
         << (profiles.size() >= 3U && profiles[0].margin_pass &&
                     profiles[1].margin_pass && profiles[2].margin_pass
                 ? "true"
                 : "false")
         << ",\n"
         << "    \"64k_classified\": "
         << (profiles.size() >= 4U &&
                     (profiles[3].margin_pass ||
                      (!profiles[3].preflight_admitted &&
                       !profiles[3].allocations_complete &&
                       profiles[3].margin_shortfall_bytes > 0U))
                 ? "true"
                 : "false")
         << ",\n"
         << "    \"real_64k_slot_at_least_400_mib\": "
         << (profiles.size() >= 4U && profiles[3].margin_pass ? "true"
                                                              : "false")
         << ",\n"
         << "    \"max_context_candidate_measured\": "
         << (max_context_candidate_tokens > 32768U &&
                     !profiles.empty() && profiles.back().context_tokens ==
                                              max_context_candidate_tokens &&
                     profiles.back().margin_pass
                 ? "true"
                 : "false")
         << ",\n"
         << "    \"second_slot_rejected_before_allocation\": "
         << (second_slot_rejected ? "true" : "false") << ",\n"
         << "    \"second_slot_check_has_zero_allocation_delta\": "
         << (free_before_second_slot_bytes == free_after_second_slot_check_bytes
                 ? "true"
                 : "false")
         << ",\n"
         << "    \"free_after_release_at_least_context_baseline\": "
         << (free_after_release_bytes >= free_after_context_bytes ? "true"
                                                                  : "false")
         << "\n"
         << "  },\n"
         << "  \"limitations\": [\"fixed graph/workspace regions are touched reserves, not captured execution graphs\",\"warm model execution begins with M11/M12 and is not claimed here\"],\n"
         << "  \"error\": \"" << JsonEscape(error_message) << "\"\n"
         << "}\n";
  return static_cast<bool>(output);
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!ParseOptions(argc, argv, &options)) return 2;

  auto inspected = gem16::InspectCheckpoint({options.model, true});
  if (!inspected.ok()) {
    std::cerr << "error: " << inspected.status().message() << '\n';
    return 3;
  }
  auto config = gem16::internal::LoadModelConfig(options.model / "config.json");
  if (!config.ok()) {
    std::cerr << "error: " << config.status().message() << '\n';
    return 3;
  }
  auto plan = gem16::internal::BuildGemma4Moe26BResidencyPlan(
      inspected.value(), config.value());
  if (!plan.ok()) {
    std::cerr << "error: " << plan.status().message() << '\n';
    return 3;
  }

  cudaDeviceProp properties{};
  int runtime_version = 0;
  int driver_version = 0;
  std::uint64_t free_after_context_bytes = 0U;
  std::uint64_t visible_total_bytes = 0U;
  std::string error_message;
  const bool initialized =
      Cuda(cudaSetDevice(options.device), "cudaSetDevice", &error_message) &&
      Cuda(cudaFree(nullptr), "cudaFree(0)", &error_message) &&
      Cuda(cudaGetDeviceProperties(&properties, options.device),
           "cudaGetDeviceProperties", &error_message) &&
      Cuda(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion",
           &error_message) &&
      Cuda(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion",
           &error_message) &&
      Measure(&free_after_context_bytes, &visible_total_bytes,
              "cudaMemGetInfo(after context)", &error_message);
  if (!initialized) {
    std::cerr << "error: " << error_message << '\n';
    return 4;
  }

  auto initial_admission = gem16::internal::CheckGemma4Moe26BAdmission(
      plan.value(), 32768U, free_after_context_bytes, true);
  DeviceRegion weights{"immutable_weights",
                       plan.value().immutable_weight_arena_bytes};
  UploadStats upload;
  std::uint64_t free_after_weights_bytes = 0U;
  std::uint64_t measured_total = 0U;
  if (!initial_admission.ok()) {
    error_message = initial_admission.message();
  } else if (weights.requested_bytes >
                 std::numeric_limits<std::size_t>::max() ||
             !Cuda(cudaMalloc(&weights.pointer,
                              static_cast<std::size_t>(weights.requested_bytes)),
                   "cudaMalloc(immutable_weights)", &error_message)) {
    if (error_message.empty()) error_message = "invalid immutable weight size";
  } else {
    auto upload_status = UploadArtifact(
        options.model, inspected.value(), plan.value(),
        static_cast<std::byte*>(weights.pointer), &upload);
    if (!upload_status.ok()) {
      error_message = upload_status.message();
    } else if (!Measure(&free_after_weights_bytes, &measured_total,
                        "cudaMemGetInfo(after weights)", &error_message)) {
      // Error already recorded.
    } else {
      weights.free_after_bytes = free_after_weights_bytes;
    }
  }

  std::vector<ProfileMeasurement> profiles;
  std::uint64_t max_context_candidate_tokens = 0U;
  if (error_message.empty()) {
    for (const auto& profile : plan.value().context_profiles) {
      profiles.push_back(RunProfile(plan.value(), profile));
    }
    if (profiles.size() == 4U) {
      if (profiles[3].margin_pass) {
        max_context_candidate_tokens = 65536U;
      } else {
        const auto& standard = plan.value().context_profiles[2];
        const auto& long_profile = plan.value().context_profiles[3];
        const std::uint64_t context_delta =
            long_profile.context_tokens - standard.context_tokens;
        const std::uint64_t kv_delta =
            long_profile.fp8_kv_bytes - standard.fp8_kv_bytes;
        const std::uint64_t kv_bytes_per_token = kv_delta / context_delta;
        const std::uint64_t fixed_local_kv_bytes =
            standard.fp8_kv_bytes -
            standard.context_tokens * kv_bytes_per_token;
        const std::uint64_t long_margin =
            long_profile.required_free_margin_bytes;
        const std::uint64_t non_kv_reserve =
            plan.value().fixed_region_bytes + long_margin;
        if (kv_bytes_per_token != 0U &&
            free_after_weights_bytes >
                non_kv_reserve + fixed_local_kv_bytes) {
          const std::uint64_t raw_candidate =
              (free_after_weights_bytes - non_kv_reserve -
               fixed_local_kv_bytes) /
              kv_bytes_per_token;
          max_context_candidate_tokens =
              std::min<std::uint64_t>(65536U,
                                      raw_candidate & ~std::uint64_t{1023U});
        }
        if (max_context_candidate_tokens > 32768U) {
          const std::uint64_t candidate_kv =
              fixed_local_kv_bytes +
              max_context_candidate_tokens * kv_bytes_per_token;
          gem16::internal::Gemma4Moe26BContextResidency candidate{
              max_context_candidate_tokens,
              candidate_kv,
              long_margin,
              plan.value().immutable_weight_arena_bytes + candidate_kv +
                  plan.value().fixed_region_bytes,
              plan.value().immutable_weight_arena_bytes + candidate_kv +
                  plan.value().fixed_region_bytes + long_margin,
          };
          auto candidate_plan = plan.value();
          candidate_plan.context_profiles.push_back(candidate);
          profiles.push_back(RunProfile(candidate_plan, candidate));
        }
      }
    }
  }

  std::uint64_t free_before_second_slot_bytes = 0U;
  std::uint64_t free_after_second_slot_check_bytes = 0U;
  bool second_slot_rejected = false;
  if (error_message.empty() &&
      Measure(&free_before_second_slot_bytes, &measured_total,
              "cudaMemGetInfo(before second slot check)", &error_message)) {
    auto second_slot = gem16::internal::CheckGemma4Moe26BAdmission(
        plan.value(), 32768U, free_before_second_slot_bytes, true);
    second_slot_rejected =
        !second_slot.ok() &&
        second_slot.code() == gem16::StatusCode::kResourceExhausted;
    if (!Measure(&free_after_second_slot_check_bytes, &measured_total,
                 "cudaMemGetInfo(after second slot check)", &error_message)) {
      second_slot_rejected = false;
    }
  }

  if (weights.pointer != nullptr) {
    const cudaError_t release = cudaFree(weights.pointer);
    weights.pointer = nullptr;
    if (release != cudaSuccess && error_message.empty()) {
      error_message = "cudaFree(immutable_weights): " +
                      std::string(cudaGetErrorString(release));
    }
  }
  std::uint64_t free_after_release_bytes = 0U;
  if (!Measure(&free_after_release_bytes, &measured_total,
               "cudaMemGetInfo(after release)", &error_message)) {
    free_after_release_bytes = 0U;
  }

  const bool upload_pass =
      upload.tensor_count == inspected.value().tensors.size() &&
      upload.payload_bytes == plan.value().artifact_payload_bytes &&
      upload.direct_tensor_count == 983U &&
      upload.nvfp4_weight_tensor_count == 151U &&
      upload.nvfp4_scale_tensor_count == 151U &&
      upload.host_staging_peak_bytes <= kUploadStagingBytes;
  const bool profile_count = profiles.size() >= 4U;
  const bool base_profiles_pass =
      profile_count && profiles[0].margin_pass && profiles[1].margin_pass &&
      profiles[2].margin_pass;
  const bool long_profile_classified =
      profile_count &&
      (profiles[3].margin_pass ||
       (!profiles[3].preflight_admitted &&
        !profiles[3].allocations_complete &&
        profiles[3].margin_shortfall_bytes > 0U));
  const bool max_candidate_pass =
      max_context_candidate_tokens > 32768U && !profiles.empty() &&
      profiles.back().context_tokens == max_context_candidate_tokens &&
      profiles.back().margin_pass;
  const bool baseline_pass =
      visible_total_bytes - free_after_context_bytes <=
          kMaximumIdleBaselineBytes &&
      plan.value().immutable_weight_arena_bytes <= kPrimaryWeightLimitBytes &&
      plan.value().immutable_weight_arena_bytes <= kHardWeightStopBytes;
  const bool release_pass =
      free_after_release_bytes >= free_after_context_bytes;
  const bool passed = error_message.empty() && upload_pass &&
                      base_profiles_pass && long_profile_classified &&
                      max_candidate_pass &&
                      second_slot_rejected &&
                      free_before_second_slot_bytes ==
                          free_after_second_slot_check_bytes &&
                      baseline_pass && release_pass;

  if (!WriteReport(options, properties, runtime_version, driver_version,
                   inspected.value(), plan.value(), upload,
                   visible_total_bytes, free_after_context_bytes,
                   free_after_weights_bytes, free_before_second_slot_bytes,
                   free_after_second_slot_check_bytes,
                   free_after_release_bytes, profiles,
                   max_context_candidate_tokens, second_slot_rejected, passed,
                   error_message)) {
    std::cerr << "error: cannot write M09 report\n";
    return 5;
  }
  const std::uint64_t primary_free =
      profile_count ? profiles[2].final_free_bytes : 0U;
  std::cout << (passed ? "PASS" : "FAIL")
            << ": real 32K CUDA-visible free bytes=" << primary_free
            << " required=" << 700U * kMiB << " report=" << options.output
            << '\n';
  return passed ? 0 : 1;
}
