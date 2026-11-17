// Explicit test-only AP1 snapshots. Never preload into a primary benchmark.
// All device storage is reserved before capture; mutable inputs are copied
// before their operator and outputs immediately after it. Production source
// and weight representations are unchanged. Three phases are saved by replay.
#include "cuda/layer/reference.h"
#include "fp8_projection_lab_abi.h"
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <array>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Launch = cudaError_t (*)(const void*, dim3, dim3, void**, std::size_t, cudaStream_t);
using Begin = cudaError_t (*)(cudaStream_t, cudaStreamCaptureMode);
using Control = gem16::internal::DecodeControl;
Launch launch = nullptr;
Begin begin = nullptr;
bool enabled = false, ready = false, injecting = false, assistant = false;
unsigned step = 0, target_global_calls = 0;
int pending_global_slot = -1;
constexpr std::size_t MiB = 1024U * 1024U;
constexpr std::size_t kGlobalStride = 48U * MiB, kSmallStride = 256U * 1024U;
constexpr std::size_t kSetBytes = 3U * kGlobalStride + 4U * kSmallStride;
constexpr std::size_t kBytes = 4U * kSetBytes;
constexpr std::size_t kCache = 17519U * 2U * 512U;
std::byte* arena = nullptr;
struct Part { const char* name = nullptr; std::size_t offset = 0, bytes = 0; };
struct Record {
  bool valid = false;
  const void* function = nullptr;
  dim3 grid{}, block{};
  std::size_t shared = 0, used = 0;
  const char* symbol = nullptr;
  std::uint32_t epsilon_bits = 0;
  bool input_output_alias = false, residual_output_alias = false;
  unsigned count = 0;
  std::array<Part, 12> parts{};
};
std::array<std::array<Record, 7>, 4> records{};
std::array<bool, 3> saved{};
std::array<const void*, 2> last_quant_input{};
constexpr std::array<const char*, 7> names{
    "target-global-r1", "assistant-step1-global-r1", "assistant-step2-global-r1",
    "target-postnorm-r1", "target-postnorm-r3", "assistant-step1-feedback", "assistant-step2-feedback"};
std::size_t Offset(unsigned slot) {
  return slot < 3 ? slot * kGlobalStride : 3U * kGlobalStride + (slot - 3U) * kSmallStride;
}
std::byte* Buffer(unsigned set, unsigned slot) { return arena + set * kSetBytes + Offset(slot); }
void Check(cudaError_t error) {
  if (error != cudaSuccess) {
    std::fprintf(stderr, "AP1 input observer: %s\n", cudaGetErrorString(error));
    std::abort();
  }
}
template <typename T> T Arg(void** args, unsigned i) {
  T value{}; std::memcpy(&value, args[i], sizeof(value)); return value;
}
__global__ void Ap1SnapshotKernel(const std::byte* from, std::byte* to, std::size_t n) {
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += std::size_t(gridDim.x) * blockDim.x) to[i] = from[i];
}
__global__ void Ap1MarkSnapshotKernel(unsigned* marker) {
  if (!blockIdx.x && !threadIdx.x) *marker = 1U;
}
unsigned* Marker(unsigned set, unsigned slot) {
  const auto capacity = slot < 3 ? kGlobalStride : kSmallStride;
  return reinterpret_cast<unsigned*>(Buffer(set, slot) + capacity - sizeof(unsigned));
}
void Copy(const void* from, void* to, std::size_t n, cudaStream_t stream) {
  if (n == 0) return;
  Ap1SnapshotKernel<<<static_cast<unsigned>((n + 255U) / 256U), 256U, 0U, stream>>>(
      static_cast<const std::byte*>(from), static_cast<std::byte*>(to), n);
  Check(cudaGetLastError());
}
void Add(unsigned slot, Record& record, const char* name, const void* ptr,
         std::size_t bytes, cudaStream_t stream) {
  if (!ptr || bytes == 0) return;
  const auto offset = (record.used + 255U) & ~std::size_t(255U);
  const auto capacity = slot < 3 ? kGlobalStride : kSmallStride;
  if (record.count >= record.parts.size() || offset + bytes > capacity - sizeof(unsigned)) std::abort();
  record.parts[record.count++] = {name, offset, bytes};
  record.used = offset + bytes;
  Copy(ptr, Buffer(0, slot) + offset, bytes, stream);
}
__attribute__((constructor)) void Initialize() {
  launch = reinterpret_cast<Launch>(dlsym(RTLD_NEXT, "cudaLaunchKernel"));
  begin = reinterpret_cast<Begin>(dlsym(RTLD_NEXT, "cudaStreamBeginCapture"));
  const char* value = std::getenv("GEM16_AP1_INPUT_OBSERVER");
  enabled = value && std::strcmp(value, "1") == 0;
}
}

extern "C" cudaError_t cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode) {
  if (enabled && !ready) {
    injecting = true;
    Check(cudaMalloc(&arena, kBytes)); Check(cudaMemset(arena, 0, kBytes));
    ready = true; injecting = false;
  }
  return begin(stream, mode);
}
extern "C" cudaError_t cudaLaunchKernel(const void* function, dim3 grid, dim3 block,
    void** args, std::size_t shared, cudaStream_t stream) {
  if (!enabled || !ready || injecting) return launch(function, grid, block, args, shared, stream);
  const char* name = nullptr; Check(cudaFuncGetName(&name, function));
  if (!name) return launch(function, grid, block, args, shared, stream);
  if (pending_global_slot >= 0 && std::strstr(name, "MergeOnlineDecodeAttentionKernelILi512E")) {
    injecting = true;
    const auto result = launch(function, grid, block, args, shared, stream);
    if (result == cudaSuccess)
      Add(pending_global_slot, records[0][pending_global_slot], "merged-output",
          Arg<const void*>(args, 2), 16U * 512U * 4U, stream);
    pending_global_slot = -1;
    injecting = false;
    return result;
  }
  if (std::strstr(name, "InitializeControlledAssistantKernel") ||
      std::strstr(name, "SetControlKernel")) { assistant = true; step = 0; }
  if (assistant && std::strstr(name, "PrepareInputKernel")) ++step;
  if (std::strstr(name, "BuildControlledMtpInputsKernel") ||
      std::strstr(name, "BuildMtpVerificationInputsKernel")) assistant = false;
  // Quantization immediately preceding the post projection. The immutable
  // source tells us its input extent; save its pointer until the matching
  // projection is encountered during this same capture.
  if (assistant && step >= 1U && step <= 2U && std::strstr(name, "QuantizeTokenReferenceKernel") &&
      Arg<std::uint64_t>(args, 3) == 1024U) last_quant_input[step - 1U] = Arg<const void*>(args, 0);
  int slot = -1;
  if (std::strstr(name, "SplitOnlineDecodeAttentionFp8GlobalKvh2Kernel") &&
      Arg<int>(args, 11) == 137 && Arg<const void*>(args, 10) == nullptr) {
    pending_global_slot = -1;
    if (assistant && step >= 1U && step <= 2U) slot = static_cast<int>(step);
    else if (!assistant && target_global_calls++ % 5U == 4U) slot = 0;
  }
  const bool norm = !assistant && std::strstr(name, "RmsNormResidualBf16KernelIfE") &&
      Arg<std::uint64_t>(args, 5) == 2816U && (grid.x == 1U || grid.x == 3U);
  if (norm) slot = grid.x == 1U ? 3 : 4;
  bool feedback = false;
  Fp8LabBinding binding{};
  if (assistant && step >= 1U && step <= 2U && std::strstr(name, "Sm120DirectProjectionKernel") &&
      std::strstr(name, "Fp8MatrixBinding")) {
    binding = Arg<Fp8LabBinding>(args, 2);
    feedback = binding.rows == 2816U && Arg<std::uint64_t>(args, 7) == 1024U;
    if (feedback) slot = 4 + static_cast<int>(step);
  }
  if (slot < 0) return launch(function, grid, block, args, shared, stream);
  if (slot < 3) pending_global_slot = slot;
  injecting = true;
  Record record{}; record.valid = true; record.function = function;
  record.symbol = name;
  record.grid = grid; record.block = block; record.shared = shared;
  if (slot < 3) {
    Add(slot, record, "query", Arg<const void*>(args, 0), 16U * 512U * 4U, stream);
    Add(slot, record, "keys", Arg<const void*>(args, 1), kCache, stream);
    Add(slot, record, "values", Arg<const void*>(args, 2), kCache, stream);
    Add(slot, record, "key-scale", Arg<const void*>(args, 3), 2U, stream);
    Add(slot, record, "value-scale", Arg<const void*>(args, 4), 2U, stream);
    Add(slot, record, "control", Arg<const void*>(args, 8), sizeof(Control), stream);
  } else if (norm) {
    record.epsilon_bits = Arg<std::uint32_t>(args, 6);
    record.input_output_alias = Arg<const void*>(args, 0) == Arg<const void*>(args, 4);
    record.residual_output_alias = Arg<const void*>(args, 2) == Arg<const void*>(args, 4);
    const auto bytes = grid.x * 2816U * 4U;
    Add(slot, record, "input", Arg<const void*>(args, 0), bytes, stream);
    Add(slot, record, "weight", Arg<const void*>(args, 1), 2816U * 2U, stream);
    Add(slot, record, "residual", Arg<const void*>(args, 2), bytes, stream);
    Add(slot, record, "scalar", Arg<const void*>(args, 7), 2U, stream);
  } else if (feedback) {
    Add(slot, record, "normalized-fp32", last_quant_input[step - 1U], 1024U * 4U, stream);
    Add(slot, record, "activation", Arg<const void*>(args, 0), 1024U, stream);
    Add(slot, record, "scale", Arg<const void*>(args, 1), 4U, stream);
  }
  const auto result = launch(function, grid, block, args, shared, stream);
  if (result == cudaSuccess) {
    if (slot < 3) {
      Add(slot, record, "partial", Arg<const void*>(args, 5), 137U * 16U * 512U * 4U, stream);
      Add(slot, record, "lse", Arg<const void*>(args, 6), 137U * 16U * 4U, stream);
    } else if (norm) {
      Add(slot, record, "normalized", Arg<const void*>(args, 3), grid.x * 2816U * 4U, stream);
      Add(slot, record, "output", Arg<const void*>(args, 4), grid.x * 2816U * 4U, stream);
    } else if (feedback) {
      Add(slot, record, "output", binding.output, 2816U * 4U, stream);
    }
    records[0][slot] = record;
    Ap1MarkSnapshotKernel<<<1U, 1U, 0U, stream>>>(Marker(0, slot)); Check(cudaGetLastError());
  }
  injecting = false;
  return result;
}
extern "C" cudaError_t __cudaLaunchKernel(cudaKernel_t kernel, dim3 grid, dim3 block,
    void** args, std::size_t shared, cudaStream_t stream) {
  return cudaLaunchKernel(reinterpret_cast<const void*>(kernel), grid, block, args, shared, stream);
}
extern "C" int gem16_lab_save_phase(unsigned phase) {
  if (!enabled || !ready || phase >= 3 || saved[phase]) return 1;
  injecting = true;
  for (unsigned slot = 0; slot < 7; ++slot) {
    const auto& record = records[0][slot];
    if (!record.valid) continue; // absence is explicitly exported, never fabricated.
    Copy(Buffer(0, slot), Buffer(phase + 1U, slot), record.used, nullptr);
    Copy(Marker(0, slot), Marker(phase + 1U, slot), sizeof(unsigned), nullptr);
    records[phase + 1U][slot] = record;
  }
  Check(cudaDeviceSynchronize()); saved[phase] = true; injecting = false;
  return 0;
}
extern "C" int gem16_lab_finish(const char* directory) {
  if (!enabled || !ready || !directory) return 1;
  try {
    injecting = true;
    const std::filesystem::path root(directory);
    if (!std::filesystem::create_directory(root)) throw std::runtime_error("AP1 snapshot directory exists");
    std::ofstream summary(root / "inputs.json");
    summary << "{\"diagnostic_only\":true,\"production_source_modified\":false,\"extra_device_bytes\":"
            << kBytes << ",\"cases\":[";
    unsigned count = 0;
    for (unsigned phase = 0; phase < 3; ++phase) {
      if (!saved[phase]) throw std::runtime_error("AP1 phase missing");
      for (unsigned slot = 0; slot < 7; ++slot) {
        const auto& record = records[phase + 1U][slot];
        unsigned executed = 0U;
        Check(cudaMemcpy(&executed, Marker(phase + 1U, slot), sizeof(unsigned), cudaMemcpyDeviceToHost));
        const bool captured = record.valid && executed == 1U;
        summary << (count++ ? "," : "") << "{\"phase\":" << phase << ",\"operator\":\"" << names[slot]
                << "\",\"captured\":" << (captured ? "true" : "false");
        if (!captured) { summary << '}'; continue; }
        cudaFuncAttributes attr{}; Check(cudaFuncGetAttributes(&attr, record.function));
        summary << ",\"grid\":[" << record.grid.x << ',' << record.grid.y << ',' << record.grid.z
                << "],\"block\":[" << record.block.x << ',' << record.block.y << ',' << record.block.z
                << "],\"registers\":" << attr.numRegs << ",\"static_shared_bytes\":" << attr.sharedSizeBytes
                << ",\"dynamic_shared_bytes\":" << record.shared
                << ",\"symbol\":\"" << record.symbol << "\",\"epsilon_bits\":" << record.epsilon_bits
                << ",\"input_output_alias\":" << (record.input_output_alias ? "true" : "false")
                << ",\"residual_output_alias\":" << (record.residual_output_alias ? "true" : "false")
                << ",\"parts\":[";
        for (unsigned i = 0; i < record.count; ++i) {
          const auto& part = record.parts[i];
          const auto file = "phase" + std::to_string(phase) + "-" + names[slot] + "-" + part.name + ".bin";
          std::vector<std::byte> host(part.bytes);
          Check(cudaMemcpy(host.data(), Buffer(phase + 1U, slot) + part.offset, part.bytes, cudaMemcpyDeviceToHost));
          std::ofstream stream(root / file, std::ios::binary); stream.write(reinterpret_cast<const char*>(host.data()), part.bytes);
          if (!stream) throw std::runtime_error("AP1 snapshot write failed");
          summary << (i ? "," : "") << "{\"name\":\"" << part.name << "\",\"file\":\"" << file
                  << "\",\"bytes\":" << part.bytes << '}';
        }
        summary << "]}";
      }
    }
    summary << "]}\n";
    if (!summary) throw std::runtime_error("AP1 snapshot summary failed");
    injecting = false; return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "AP1 input observer: %s\n", error.what()); return 1;
  }
}
