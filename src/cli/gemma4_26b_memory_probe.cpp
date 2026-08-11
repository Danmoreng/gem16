#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "model/gemma4_26b_manifest.h"

namespace {

constexpr std::uint64_t kMiB = 1024U * 1024U;
constexpr std::uint64_t kRequiredFreeBytes = 700U * kMiB;
constexpr std::uint64_t kPrimaryWeightLimitBytes = 14100U * kMiB;
constexpr std::uint64_t kHardWeightStopBytes = 14300U * kMiB;
constexpr std::uint64_t kMaximumIdleBaselineBytes = 512U * kMiB;

struct Options {
  int device = 0;
  std::filesystem::path output;
  std::string code_revision = "unknown";
};

struct Region {
  std::string name;
  std::uint64_t requested_bytes = 0;
  void* pointer = nullptr;
  std::uint64_t free_after_bytes = 0;
};

void Usage(std::ostream& output) {
  output << "Usage: gem16-26b-memory-probe --output <report.json> "
            "[--device <index>] [--code-revision <sha>]\n";
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
    if (argument == "--output") {
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
  if (options->output.empty()) {
    std::cerr << "error: --output is required\n";
    return false;
  }
  return true;
}

std::string JsonEscape(std::string_view value) {
  std::ostringstream output;
  for (const unsigned char byte : value) {
    switch (byte) {
      case '"':
        output << "\\\"";
        break;
      case '\\':
        output << "\\\\";
        break;
      case '\b':
        output << "\\b";
        break;
      case '\f':
        output << "\\f";
        break;
      case '\n':
        output << "\\n";
        break;
      case '\r':
        output << "\\r";
        break;
      case '\t':
        output << "\\t";
        break;
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
  *error_message = std::string(operation) + ": " + cudaGetErrorString(result);
  return false;
}

bool Release(std::vector<Region>* regions, std::string* error_message) {
  bool released = true;
  for (auto region = regions->rbegin(); region != regions->rend(); ++region) {
    if (region->pointer != nullptr) {
      const auto result = cudaFree(region->pointer);
      if (result != cudaSuccess) {
        if (!error_message->empty()) *error_message += "; ";
        *error_message += "cudaFree(" + region->name + "): " +
                          cudaGetErrorString(result);
        released = false;
      }
      region->pointer = nullptr;
    }
  }
  return released;
}

bool WriteReport(const Options& options, const cudaDeviceProp& properties,
                 int runtime_version, int driver_version,
                 std::uint64_t q4_arena_bytes,
                 std::uint64_t nvfp4_arena_bytes,
                 std::uint64_t immutable_weight_bytes,
                 std::uint64_t kv_bytes, std::uint64_t visible_total_bytes,
                 std::uint64_t free_after_context_bytes,
                 std::uint64_t final_free_bytes,
                 std::uint64_t free_after_release_bytes,
                 const std::vector<Region>& regions, bool passed,
                 std::string_view error_message) {
  std::ofstream output(options.output, std::ios::binary | std::ios::trunc);
  if (!output) {
    std::cerr << "error: cannot create report: " << options.output << '\n';
    return false;
  }
  output << "{\n"
         << "  \"schema_version\": 1,\n"
         << "  \"milestone\": \"M03\",\n"
         << "  \"status\": \"" << (passed ? "pass" : "fail") << "\",\n"
         << "  \"code_revision\": \"" << JsonEscape(options.code_revision)
         << "\",\n"
         << "  \"probe_kind\": \"synthetic_device_admission\",\n"
         << "  \"not_model_execution\": true,\n"
         << "  \"device\": {\n"
         << "    \"index\": " << options.device << ",\n"
         << "    \"name\": \"" << JsonEscape(properties.name) << "\",\n"
         << "    \"uuid\": \"" << UuidText(properties.uuid) << "\",\n"
         << "    \"compute_capability\": \"" << properties.major << '.'
         << properties.minor << "\",\n"
         << "    \"runtime_version\": " << runtime_version << ",\n"
         << "    \"driver_version\": " << driver_version << "\n"
         << "  },\n"
         << "  \"contract\": {\n"
         << "    \"profile\": \"gemma4_26b_compiled_hybrid_v1\",\n"
         << "    \"head_assumption\": \"max_of_q4_0_and_nvfp4\",\n"
         << "    \"q4_0_aligned_weight_arena_bytes\": " << q4_arena_bytes
         << ",\n"
         << "    \"nvfp4_aligned_weight_arena_bytes\": "
         << nvfp4_arena_bytes << ",\n"
         << "    \"selected_conservative_weight_arena_bytes\": "
         << immutable_weight_bytes << ",\n"
         << "    \"fp8_kv_32k_bytes\": " << kv_bytes << ",\n"
         << "    \"arena_alignment_bytes\": 256,\n"
         << "    \"primary_weight_limit_bytes\": "
         << kPrimaryWeightLimitBytes << ",\n"
         << "    \"hard_weight_stop_bytes\": " << kHardWeightStopBytes
         << "\n"
         << "  },\n"
         << "  \"measurement\": {\n"
         << "    \"cuda_visible_total_bytes\": " << visible_total_bytes
         << ",\n"
         << "    \"free_after_context_bytes\": "
         << free_after_context_bytes << ",\n"
         << "    \"context_and_baseline_bytes\": "
         << visible_total_bytes - free_after_context_bytes << ",\n"
         << "    \"required_free_margin_bytes\": " << kRequiredFreeBytes
         << ",\n"
         << "    \"final_direct_free_bytes\": " << final_free_bytes
         << ",\n"
         << "    \"free_after_release_bytes\": "
         << free_after_release_bytes << "\n"
         << "  },\n"
         << "  \"regions\": [\n";
  for (std::size_t index = 0; index < regions.size(); ++index) {
    const auto& region = regions[index];
    output << "    {\"name\":\"" << JsonEscape(region.name)
           << "\",\"requested_bytes\":" << region.requested_bytes
           << ",\"free_after_bytes\":" << region.free_after_bytes << '}';
    output << (index + 1U == regions.size() ? "\n" : ",\n");
  }
  output << "  ],\n"
         << "  \"gates\": {\n"
         << "    \"idle_baseline_at_most_512_mib\": "
         << ((visible_total_bytes - free_after_context_bytes <=
              kMaximumIdleBaselineBytes)
                 ? "true"
                 : "false")
         << ",\n"
         << "    \"immutable_weights_at_most_14100_mib\": "
         << (immutable_weight_bytes <= kPrimaryWeightLimitBytes ? "true"
                                                                : "false")
         << ",\n"
         << "    \"immutable_weights_below_14300_mib_hard_stop\": "
         << (immutable_weight_bytes <= kHardWeightStopBytes ? "true"
                                                            : "false")
         << ",\n"
         << "    \"direct_free_at_least_700_mib\": "
         << (final_free_bytes >= kRequiredFreeBytes ? "true" : "false")
         << ",\n"
         << "    \"free_after_release_at_least_context_baseline\": "
         << (free_after_release_bytes >= free_after_context_bytes ? "true"
                                                                  : "false")
         << "\n"
         << "  },\n"
         << "  \"error\": \"" << JsonEscape(error_message) << "\"\n"
         << "}\n";
  if (!output) {
    std::cerr << "error: failed while writing report: " << options.output
              << '\n';
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!ParseOptions(argc, argv, &options)) return 2;

  auto q4_contract =
      gem16::internal::BuildGemma4Moe26BCompiledHybridContract(
          gem16::internal::Gemma4Moe26BHeadFormat::kQ4_0);
  auto nvfp4_contract =
      gem16::internal::BuildGemma4Moe26BCompiledHybridContract(
          gem16::internal::Gemma4Moe26BHeadFormat::kNvfp4);
  if (!q4_contract.ok() || !nvfp4_contract.ok()) {
    std::cerr << "error: cannot build compiled 26B tensor contract\n";
    return 3;
  }
  auto q4_arena = gem16::internal::Gemma4Moe26BAlignedArenaBytes(
      q4_contract.value());
  auto nvfp4_arena = gem16::internal::Gemma4Moe26BAlignedArenaBytes(
      nvfp4_contract.value());
  auto kv_bytes = gem16::internal::Gemma4Moe26B32KFp8KvBytes();
  if (!q4_arena.ok() || !nvfp4_arena.ok() || !kv_bytes.ok()) {
    std::cerr << "error: cannot calculate compiled 26B memory contract\n";
    return 3;
  }
  const std::uint64_t immutable_weight_bytes =
      std::max(q4_arena.value(), nvfp4_arena.value());

  cudaDeviceProp properties{};
  int runtime_version = 0;
  int driver_version = 0;
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  std::string error_message;
  bool cuda_ok =
      Cuda(cudaSetDevice(options.device), "cudaSetDevice", &error_message) &&
      Cuda(cudaFree(nullptr), "cudaFree(0)", &error_message) &&
      Cuda(cudaGetDeviceProperties(&properties, options.device),
           "cudaGetDeviceProperties", &error_message) &&
      Cuda(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion",
           &error_message) &&
      Cuda(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion",
           &error_message) &&
      Cuda(cudaMemGetInfo(&free_bytes, &total_bytes),
           "cudaMemGetInfo(after context)", &error_message);
  if (!cuda_ok) {
    std::cerr << "error: " << error_message << '\n';
    return 4;
  }
  const std::uint64_t visible_total_bytes = total_bytes;
  const std::uint64_t free_after_context_bytes = free_bytes;

  std::vector<Region> regions = {
      {"immutable_weights", immutable_weight_bytes},
      {"fp8_kv_32k", kv_bytes.value()},
      {"moe_prefill_workspace", 256U * kMiB},
      {"activation_output_workspace", 128U * kMiB},
      {"graph_private_reserve", 32U * kMiB},
      {"allocator_metadata_guard", 32U * kMiB},
  };
  std::uint64_t final_free_bytes = free_after_context_bytes;
  for (auto& region : regions) {
    if (region.requested_bytes > std::numeric_limits<std::size_t>::max()) {
      error_message = "requested region exceeds size_t: " + region.name;
      break;
    }
    if (!Cuda(cudaMalloc(&region.pointer,
                         static_cast<std::size_t>(region.requested_bytes)),
              "cudaMalloc(" + region.name + ')', &error_message) ||
        !Cuda(cudaMemset(region.pointer, 0,
                         static_cast<std::size_t>(region.requested_bytes)),
              "cudaMemset(" + region.name + ')', &error_message) ||
        !Cuda(cudaDeviceSynchronize(),
              "cudaDeviceSynchronize(" + region.name + ')', &error_message) ||
        !Cuda(cudaMemGetInfo(&free_bytes, &total_bytes),
              "cudaMemGetInfo(" + region.name + ')', &error_message)) {
      break;
    }
    region.free_after_bytes = free_bytes;
    final_free_bytes = free_bytes;
  }
  const bool all_regions_allocated =
      std::all_of(regions.begin(), regions.end(), [](const Region& region) {
        return region.pointer != nullptr && region.free_after_bytes != 0;
      });
  const bool allocation_gates_pass =
      error_message.empty() && all_regions_allocated &&
      visible_total_bytes - free_after_context_bytes <=
          kMaximumIdleBaselineBytes &&
      immutable_weight_bytes <= kPrimaryWeightLimitBytes &&
      immutable_weight_bytes <= kHardWeightStopBytes &&
      final_free_bytes >= kRequiredFreeBytes;

  const bool released = Release(&regions, &error_message);
  std::uint64_t free_after_release_bytes = 0;
  const bool measured_after_release =
      Cuda(cudaMemGetInfo(&free_bytes, &total_bytes),
           "cudaMemGetInfo(after release)", &error_message);
  if (measured_after_release) free_after_release_bytes = free_bytes;
  const bool passed = allocation_gates_pass && released &&
                      measured_after_release &&
                      free_after_release_bytes >= free_after_context_bytes;
  if (!WriteReport(options, properties, runtime_version, driver_version,
                   q4_arena.value(), nvfp4_arena.value(),
                   immutable_weight_bytes, kv_bytes.value(),
                   visible_total_bytes, free_after_context_bytes,
                   final_free_bytes, free_after_release_bytes, regions, passed,
                   error_message)) {
    return 5;
  }
  std::cout << (passed ? "PASS" : "FAIL")
            << ": final CUDA-visible free bytes=" << final_free_bytes
            << " required=" << kRequiredFreeBytes << " report="
            << options.output << '\n';
  return passed ? 0 : 1;
}
