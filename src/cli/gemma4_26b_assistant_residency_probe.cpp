#include <cuda_runtime_api.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>

#include "cuda/engine/gemma4_26b_artifact.h"
#include "gem16/model.h"
#include "model/gemma4_26b_residency.h"

namespace {

struct Options {
  std::filesystem::path model;
  std::filesystem::path output;
  int device = 0;
};

bool ParseDevice(std::string_view text, int* value) {
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

bool Parse(int argc, char** argv, Options* options) {
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (index + 1 >= argc) return false;
    const std::string_view value(argv[++index]);
    if (argument == "--model") {
      options->model = value;
    } else if (argument == "--output") {
      options->output = value;
    } else if (argument == "--device") {
      if (!ParseDevice(value, &options->device)) return false;
    } else {
      return false;
    }
  }
  return !options->model.empty() && !options->output.empty();
}

bool Memory(std::uint64_t* free_bytes, std::uint64_t* total_bytes) {
  std::size_t free = 0;
  std::size_t total = 0;
  if (cudaMemGetInfo(&free, &total) != cudaSuccess) return false;
  *free_bytes = free;
  *total_bytes = total;
  return true;
}

int Fail(std::string_view message) {
  std::cerr << "error: " << message << '\n';
  return 1;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!Parse(argc, argv, &options)) {
    std::cerr << "Usage: gem16-26b-assistant-residency-probe --model <dir> "
                 "--output <report.json> [--device <index>]\n";
    return 2;
  }
  if (std::filesystem::exists(options.output)) {
    return Fail("refusing to overwrite the residency report");
  }
  if (cudaSetDevice(options.device) != cudaSuccess) {
    return Fail("cannot select the CUDA device");
  }
  auto manifest = gem16::InspectCheckpoint({options.model, true});
  if (!manifest.ok()) return Fail(manifest.status().message());
  auto plan = gem16::internal::BuildGemma4Moe26BAssistantResidencyPlan(
      manifest.value());
  if (!plan.ok()) return Fail(plan.status().message());

  std::uint64_t free_before = 0;
  std::uint64_t total = 0;
  if (!Memory(&free_before, &total)) return Fail("cannot query CUDA memory");
  auto admission = gem16::internal::CheckGemma4Moe26BAdmission(
      plan.value(), 32768U, free_before, true);
  if (!admission.ok()) return Fail(admission.message());

  gem16::internal::Gemma4Moe26BDeviceArtifactStats stats;
  std::uint64_t arena_bytes = 0;
  std::uint64_t free_resident = 0;
  void* fixed = nullptr;
  {
    auto artifact = gem16::internal::Gemma4Moe26BDeviceArtifact::Load(
        options.model, manifest.value(), plan.value());
    if (!artifact.ok()) return Fail(artifact.status().message());
    arena_bytes = artifact.value().arena_bytes();
    stats = artifact.value().stats();
    const auto allocation = cudaMalloc(
        &fixed, static_cast<std::size_t>(plan.value().fixed_region_bytes));
    if (allocation != cudaSuccess) {
      return Fail("cannot allocate the fixed 64K Assistant workspace reserve");
    }
    if (!Memory(&free_resident, &total)) return Fail("cannot measure residency");
    if (cudaFree(fixed) != cudaSuccess) return Fail("cannot release fixed reserve");
    fixed = nullptr;
  }
  std::uint64_t free_after = 0;
  if (!Memory(&free_after, &total)) return Fail("cannot measure released memory");

  std::ofstream output(options.output, std::ios::binary | std::ios::trunc);
  if (!output) return Fail("cannot create the residency report");
  output << "{\n"
         << "  \"schema_version\": 1,\n"
         << "  \"status\": \"diagnostic_pass_not_acceptance\",\n"
         << "  \"profile\": \"sm120-mtp-assistant-hybrid-v1\",\n"
         << "  \"device\": " << options.device << ",\n"
         << "  \"tensor_count\": " << stats.tensors << ",\n"
         << "  \"payload_bytes\": " << stats.payload_bytes << ",\n"
         << "  \"arena_bytes\": " << arena_bytes << ",\n"
         << "  \"fixed_workspace_reserve_bytes\": "
         << plan.value().fixed_region_bytes << ",\n"
         << "  \"direct_tensor_count\": " << stats.direct_tensors << ",\n"
         << "  \"tiled_weight_tensor_count\": "
         << stats.tiled_weight_tensors << ",\n"
         << "  \"tiled_scale_tensor_count\": "
         << stats.tiled_scale_tensors << ",\n"
         << "  \"host_staging_peak_bytes\": "
         << stats.host_staging_peak_bytes << ",\n"
         << "  \"free_before_bytes\": " << free_before << ",\n"
         << "  \"free_resident_bytes\": " << free_resident << ",\n"
         << "  \"free_after_release_bytes\": " << free_after << ",\n"
         << "  \"target_kv_cache_bytes\": 0,\n"
         << "  \"target_kv_semantics\": \"shared_view_no_duplicate_allocation\"\n"
         << "}\n";
  if (!output) return Fail("cannot write the residency report");
  std::cout << options.output << '\n';
  return 0;
}
