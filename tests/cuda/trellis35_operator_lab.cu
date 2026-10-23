// Explicit LD_PRELOAD diagnostic for the pinned native engine. Captured copy
// kernels snapshot real operator inputs; no weight is copied or repacked.
// Never use this library for primary throughput measurements.
#include "cuda/engine/gemma4_26b_trellis35_artifact.h"
#include "cuda/layer/reference.h"
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <dlfcn.h>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

extern "C" cudaError_t gem16_lab_launch_m1_eight(
    const std::uint8_t*, const float*,
    const gem16::internal::Trellis35DeviceFamilyBinding*, const std::uint32_t*,
    float*, std::uint64_t, std::uint64_t, cudaStream_t);
extern "C" cudaError_t gem16_lab_launch_t3_pair(
    const std::uint8_t*, const float*,
    const gem16::internal::Trellis35DeviceFamilyBinding*, const std::uint32_t*,
    float*, std::uint64_t, std::uint64_t, cudaStream_t);
extern "C" cudaError_t gem16_lab_launch_gate_fused(
    const float*, const gem16::internal::Trellis35DeviceFamilyBinding*,
    const std::uint32_t*, std::uint8_t*, float*, float*, unsigned, bool, cudaStream_t);
extern "C" cudaError_t gem16_lab_launch_kvh2_stage32(
    const float*, const std::uint8_t*, const std::uint8_t*, const std::uint16_t*,
    const std::uint16_t*, float*, float*, float*,
    const gem16::internal::DecodeControl*, int, dim3, dim3, cudaStream_t);

namespace {
using Family = gem16::internal::Trellis35DeviceFamilyBinding;
using Launch = cudaError_t (*)(const void*, dim3, dim3, void**, std::size_t, cudaStream_t);
using Begin = cudaError_t (*)(cudaStream_t, cudaStreamCaptureMode);
Launch real_launch = nullptr;
Begin real_begin = nullptr;
bool enabled = false, injecting = false, ready = false, t3 = false;
bool attention = false;
unsigned attention_calls = 0U;
unsigned gate_count = 0U;
int pending_transform = -1;
std::byte* arena = nullptr;
constexpr std::size_t kElements = 24U * 2816U;
constexpr std::size_t kStride = 1152U * 1024U;
constexpr std::size_t kArenaBytes = 32U * kStride;
constexpr std::array<unsigned, 4> kLayers{0U, 5U, 15U, 29U};
struct Record {
  const void* function = nullptr;
  const void* transform_function = nullptr;
  const void* quantize_function = nullptr;
  const void* transform_output = nullptr;
  Family family{};
  dim3 grid{}, block{};
  dim3 transform_grid{}, transform_block{}, quantize_grid{}, quantize_block{};
  std::uint64_t input_elements = 0U, output_elements = 0U;
  unsigned assignments = 0U;
  bool valid = false, transformed_valid = false;
  std::size_t raw_bytes = 0U;
};
std::array<Record, 32> records{};
std::array<bool, 3> saved{};
constexpr std::size_t kAttentionStride = 5U * 1024U * 1024U;
constexpr std::size_t kPartialBytes = 137U * 16U * 512U * 4U;
struct AttentionRecord {
  const void* function = nullptr;
  const std::uint8_t *keys = nullptr, *values = nullptr;
  const std::uint16_t *key_scale = nullptr, *value_scale = nullptr;
  dim3 grid{}, block{};
  int splits = 0;
  bool valid = false;
};
std::array<AttentionRecord, 4> attention_records{};
struct AttentionBuffers {
  float *query, *partial, *lse, *output;
  gem16::internal::DecodeControl* control;
};
AttentionBuffers AttentionBuffer(unsigned i) {
  auto* p = arena + kArenaBytes + i * kAttentionStride;
  return {reinterpret_cast<float*>(p), reinterpret_cast<float*>(p + 33024U),
          reinterpret_cast<float*>(p + 33024U + kPartialBytes),
          reinterpret_cast<float*>(p + 33024U + kPartialBytes + 137U * 16U * 4U),
          reinterpret_cast<gem16::internal::DecodeControl*>(p + 32768U)};
}

struct Buffers {
  std::uint8_t* activation;
  float* scales;
  std::uint32_t* ids;
  float *expected, *temporary, *transformed, *raw;
};
Buffers Buffer(unsigned record) {
  auto* p = arena + record * kStride;
  Buffers b{};
  b.activation = reinterpret_cast<std::uint8_t*>(p); p += kElements;
  b.scales = reinterpret_cast<float*>(p); p += 24U * 4U;
  b.ids = reinterpret_cast<std::uint32_t*>(p); p += 24U * 4U;
  b.expected = reinterpret_cast<float*>(p); p += kElements * 4U;
  b.temporary = reinterpret_cast<float*>(p); p += kElements * 4U;
  b.transformed = reinterpret_cast<float*>(p); p += kElements * 4U;
  b.raw = reinterpret_cast<float*>(p); p += kElements * 4U;
  if (p > arena + (record + 1U) * kStride) std::abort();
  return b;
}
void Check(cudaError_t error) {
  if (error != cudaSuccess) {
    std::fprintf(stderr, "operator lab CUDA failure: %s\n", cudaGetErrorString(error));
    std::abort();
  }
}
__global__ void DiagnosticCopyKernel(const std::byte* from, std::byte* to, std::size_t bytes) {
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < bytes; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) to[i] = from[i];
}
void Copy(const void* from, void* to, std::size_t bytes, cudaStream_t stream) {
  DiagnosticCopyKernel<<<static_cast<unsigned>((bytes + 255U) / 256U), 256U, 0, stream>>>(
      static_cast<const std::byte*>(from), static_cast<std::byte*>(to), bytes);
  Check(cudaGetLastError());
}
int Slot(unsigned layer) {
  for (unsigned i = 0U; i < kLayers.size(); ++i) if (layer == kLayers[i]) return static_cast<int>(i);
  return -1;
}
template <typename T> T Arg(void** args, unsigned i) {
  T result{}; std::memcpy(&result, args[i], sizeof(result)); return result;
}
__attribute__((constructor)) void Initialize() {
  real_launch = reinterpret_cast<Launch>(dlsym(RTLD_NEXT, "cudaLaunchKernel"));
  real_begin = reinterpret_cast<Begin>(dlsym(RTLD_NEXT, "cudaStreamBeginCapture"));
  const char* setting = std::getenv("GEM16_TRELLIS35_OPERATOR_LAB");
  enabled = setting && std::strcmp(setting, "1") == 0;
  const char* mode = std::getenv("GEM16_LAB_MODE");
  t3 = mode && std::strcmp(mode, "t3") == 0;
  const char* candidate = std::getenv("GEM16_LAB_CANDIDATE");
  attention = candidate && std::strcmp(candidate, "attention") == 0;
}
void Export(const std::filesystem::path& path, const void* device, std::size_t bytes) {
  std::vector<std::byte> host(bytes);
  Check(cudaMemcpy(host.data(), device, bytes, cudaMemcpyDeviceToHost));
  if (std::filesystem::exists(path)) throw std::runtime_error("operator export exists");
  std::ofstream file(path, std::ios::binary);
  file.write(reinterpret_cast<const char*>(host.data()), static_cast<std::streamsize>(bytes));
  if (!file) throw std::runtime_error("operator export write failed");
}
template <typename Function>
cudaGraphExec_t ReplayGraph(cudaStream_t stream, Function launch) {
  cudaGraph_t graph{};
  Check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  for (unsigned repeat = 0U; repeat < 16U; ++repeat) launch();
  Check(cudaStreamEndCapture(stream, &graph));
  cudaGraphExec_t executable{};
  Check(cudaGraphInstantiate(&executable, graph, 0ULL));
  Check(cudaGraphDestroy(graph));
  return executable;
}
}  // namespace

extern "C" cudaError_t cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode) {
  if (enabled && !ready) {
    injecting = true;
    const auto bytes = kArenaBytes + (attention ? 5U * kAttentionStride : 0U);
    Check(cudaMalloc(&arena, bytes));
    Check(cudaMemset(arena, 0, bytes));
    ready = true;
    injecting = false;
  }
  return real_begin(stream, mode);
}
extern "C" cudaError_t cudaLaunchKernel(const void* function, dim3 grid, dim3 block,
                                       void** args, std::size_t shared, cudaStream_t stream) {
  const cudaError_t result = real_launch(function, grid, block, args, shared, stream);
  if (!enabled || !ready || injecting || result != cudaSuccess) return result;
  const char* name = nullptr;
  Check(cudaFuncGetName(&name, function));
  if (!name) return result;
  injecting = true;
  if (attention && std::strstr(name, "SplitOnlineDecodeAttentionFp8GlobalKvh2Kernel")) {
    if (attention_calls++ % 5U == 4U) {
      auto& r = attention_records[0]; const auto b = AttentionBuffer(0U);
      r.function = function; r.grid = grid; r.block = block;
      r.keys = Arg<const std::uint8_t*>(args, 1U); r.values = Arg<const std::uint8_t*>(args, 2U);
      r.key_scale = Arg<const std::uint16_t*>(args, 3U); r.value_scale = Arg<const std::uint16_t*>(args, 4U);
      r.splits = Arg<int>(args, 11U); r.valid = true;
      if (r.splits <= 0 || r.splits > 137 || Arg<const void*>(args, 10U) != nullptr) std::abort();
      Copy(Arg<const void*>(args, 0U), b.query, 16U * 512U * 4U, stream);
      Copy(Arg<const void*>(args, 8U), b.control, sizeof(gem16::internal::DecodeControl), stream);
      Copy(Arg<const void*>(args, 5U), b.partial, static_cast<std::size_t>(r.splits) * 16U * 512U * 4U, stream);
      Copy(Arg<const void*>(args, 6U), b.lse, static_cast<std::size_t>(r.splits) * 16U * 4U, stream);
    }
  }
  if (attention) { injecting = false; return result; }
  const bool transform = std::strstr(name, "SelectedInputTransformWarpKernel") != nullptr;
  const bool projection = std::strstr(name, t3 ? "MmaW4A8ProjectionGroupedT3M16N128InverseKernel"
                                             : "MmaW4A8ProjectionSelectedN128InverseKernel") != nullptr;
  const unsigned assignments = t3 ? 24U : 8U;
  if (transform && Arg<unsigned>(args, 4U) == assignments && Arg<std::uint64_t>(args, 7U) == 2816U) {
    const int slot = Slot(gate_count % 30U);
    pending_transform = slot >= 0 ? 2 * slot : -1;
    if (slot >= 0) {
      auto& r = records[2U * static_cast<unsigned>(slot)];
      r.transform_function = function; r.transform_grid = grid; r.transform_block = block;
      r.transform_output = Arg<const void*>(args, 3U);
      const auto b = Buffer(2U * static_cast<unsigned>(slot));
      const auto stride = Arg<std::uint64_t>(args, 5U);
      const unsigned per_input = Arg<unsigned>(args, 6U);
      if (!per_input || assignments % per_input) std::abort();
      r.raw_bytes = static_cast<std::size_t>(stride ? stride * (assignments / per_input) : 2816U) * 4U;
      if (r.raw_bytes > kElements * 4U) std::abort();
      Copy(Arg<const void*>(args, 0U), b.raw, r.raw_bytes, stream);
      Copy(Arg<const void*>(args, 3U), b.transformed, assignments * 2816U * 4U, stream);
      r.transformed_valid = true;
    }
  }
  if (pending_transform >= 0 && std::strstr(name, "QuantizeTokenReferenceKernel") &&
      Arg<const void*>(args, 0U) == records[static_cast<unsigned>(pending_transform)].transform_output) {
    auto& r = records[static_cast<unsigned>(pending_transform)];
    if (Arg<std::uint64_t>(args, 3U) != 2816U) std::abort();
    r.quantize_function = function; r.quantize_grid = grid; r.quantize_block = block;
  }
  if (projection) {
    const auto input = Arg<std::uint64_t>(args, 5U);
    const auto output = Arg<std::uint64_t>(args, 6U);
    const bool gate = input == 2816U && output == 1408U;
    if (!gate && !(input == 768U && output == 2816U)) std::abort();
    const unsigned layer = gate ? gate_count++ % 30U : (gate_count + 29U) % 30U;
    const int slot = Slot(layer);
    if (slot >= 0) {
      const unsigned index = 2U * static_cast<unsigned>(slot) + (gate ? 0U : 1U);
      auto& r = records[index]; const auto b = Buffer(index);
      r.function = function; r.family = Arg<Family>(args, 2U);
      r.grid = grid; r.block = block; r.input_elements = input;
      r.output_elements = output; r.assignments = assignments; r.valid = true;
      Copy(Arg<const void*>(args, 0U), b.activation, assignments * input, stream);
      Copy(Arg<const void*>(args, 1U), b.scales, assignments * 4U, stream);
      Copy(Arg<const void*>(args, 3U), b.ids, assignments * 4U, stream);
      Copy(Arg<const void*>(args, 4U), b.expected, assignments * output * 4U, stream);
    }
  }
  injecting = false;
  return result;
}

// CUDA 13 compiler stubs use a cudaKernel_t entry point. The documented C API
// also accepts that handle; route both forms through the same observer.
extern "C" cudaError_t __cudaLaunchKernel(cudaKernel_t kernel, dim3 grid, dim3 block,
                                          void** args, std::size_t shared, cudaStream_t stream) {
  return cudaLaunchKernel(reinterpret_cast<const void*>(kernel), grid, block, args, shared, stream);
}

extern "C" int gem16_lab_save_phase(unsigned phase) {
  if (!enabled || !ready || phase >= 3U || saved[phase]) return 1;
  injecting = true;
  if (attention) {
    if (!attention_records[0].valid) return 1;
    Copy(arena + kArenaBytes, arena + kArenaBytes + (phase + 1U) * kAttentionStride,
         kAttentionStride, nullptr);
    attention_records[phase + 1U] = attention_records[0];
    Check(cudaDeviceSynchronize()); saved[phase] = true; injecting = false;
    return 0;
  }
  for (unsigned i = 0U; i < 8U; ++i) {
    if (!records[i].valid) {
      std::fprintf(stderr, "operator lab missing record %u, gate calls %u\n", i, gate_count);
      std::abort();
    }
    const unsigned destination = 8U + phase * 8U + i;
    Copy(arena + i * kStride, arena + destination * kStride, kStride, nullptr);
    records[destination] = records[i];
  }
  Check(cudaDeviceSynchronize());
  saved[phase] = true;
  injecting = false;
  return 0;
}

extern "C" int gem16_lab_finish(const char* directory) {
  if (!enabled || !ready || !directory) return 1;
  try {
    injecting = true;
    const char* candidate_setting = std::getenv("GEM16_LAB_CANDIDATE");
    const bool test_m1 = candidate_setting && std::strcmp(candidate_setting, "m1") == 0;
    const bool test_t3 = candidate_setting && std::strcmp(candidate_setting, "t3-pair") == 0;
    const bool test_projection = (!t3 && test_m1) || (t3 && test_t3);
    const auto candidate_projection = t3 ? gem16_lab_launch_t3_pair : gem16_lab_launch_m1_eight;
    const bool test_fusion = candidate_setting && std::strcmp(candidate_setting, "fusion") == 0;
    cudaStream_t replay_stream{};
    Check(cudaStreamCreateWithFlags(&replay_stream, cudaStreamNonBlocking));
    const std::filesystem::path root(directory);
    if (!std::filesystem::create_directory(root)) throw std::runtime_error("operator directory exists");
    if (attention) {
      const bool parent_only = std::getenv("GEM16_LAB_ATTENTION_PARENT_ONLY") != nullptr;
      std::ofstream summary(root / "attention.json"); summary.precision(9);
      summary << "{\"diagnostic_only\":true,\"global_kv_borrowed\":true,\"extra_device_bytes\":"
              << kArenaBytes + 5U * kAttentionStride << ",\"cases\":[";
      for (unsigned phase = 0U; phase < 3U; ++phase) {
        if (!saved[phase]) throw std::runtime_error("attention phase absent");
        const auto& r = attention_records[phase + 1U];
        const auto b = AttentionBuffer(phase + 1U), tmp = AttentionBuffer(4U);
        gem16::internal::DecodeControl state{};
        Check(cudaMemcpy(&state, b.control, sizeof(state), cudaMemcpyDeviceToHost));
        if (state.position != std::array<std::uint64_t, 3>{16384U, 16951U, 17517U}[phase])
          throw std::runtime_error("attention control position mismatch");
        const auto active = static_cast<std::size_t>((state.position + 128U) / 128U);
        const auto output_bytes = active * 16U * 512U * 4U, lse_bytes = active * 16U * 4U;
        const float* query = b.query; const auto* keys = r.keys; const auto* values = r.values;
        const auto* key_scale = r.key_scale; const auto* value_scale = r.value_scale;
        float *partial = tmp.partial, *lse = tmp.lse, *output = tmp.output;
        const auto* control = b.control; const gem16::internal::DecodeControl* rows = nullptr;
        std::uint64_t start = 0U; int splits = r.splits;
        void* args[]{&query, &keys, &values, &key_scale, &value_scale, &partial,
                     &lse, &output, &control, &start, &rows, &splits};
        auto parent = [&] { Check(real_launch(r.function, r.grid, r.block, args, 0U, replay_stream)); };
        auto candidate = [&] { Check(gem16_lab_launch_kvh2_stage32(query, keys, values,
            key_scale, value_scale, partial, lse, output, control, splits, r.grid, r.block, replay_stream)); };
        std::vector<std::byte> expected(output_bytes + lse_bytes), actual(expected.size());
        Check(cudaMemcpy(expected.data(), b.partial, output_bytes, cudaMemcpyDeviceToHost));
        Check(cudaMemcpy(expected.data() + output_bytes, b.lse, lse_bytes, cudaMemcpyDeviceToHost));
        auto equal = [&] {
          Check(cudaStreamSynchronize(replay_stream));
          Check(cudaMemcpy(actual.data(), tmp.partial, output_bytes, cudaMemcpyDeviceToHost));
          Check(cudaMemcpy(actual.data() + output_bytes, tmp.lse, lse_bytes, cudaMemcpyDeviceToHost));
          return actual == expected;
        };
        const char* profile_case = std::getenv("GEM16_LAB_PROFILE_CASE");
        const char* profile_kind = std::getenv("GEM16_LAB_PROFILE_KIND");
        const bool profile = profile_case && ("attention-phase" + std::to_string(phase)) == profile_case;
        const bool profile_parent = profile && profile_kind && std::strcmp(profile_kind, "parent") == 0;
        const bool profile_candidate = profile && profile_kind && std::strcmp(profile_kind, "candidate") == 0;
        if (profile_parent) Check(cudaProfilerStart());
        parent(); Check(cudaStreamSynchronize(replay_stream));
        if (profile_parent) Check(cudaProfilerStop());
        if (!equal()) throw std::runtime_error("borrowed-KV parent attention replay mismatch");
        bool exact = false;
        if (!parent_only) {
          if (profile_candidate) Check(cudaProfilerStart());
          candidate(); Check(cudaStreamSynchronize(replay_stream));
          if (profile_candidate) Check(cudaProfilerStop());
          exact = equal();
        }
        const auto pg = ReplayGraph(replay_stream, parent);
        const auto cg = parent_only ? nullptr : ReplayGraph(replay_stream, candidate);
        cudaEvent_t begin{}, end{}; Check(cudaEventCreate(&begin)); Check(cudaEventCreate(&end));
        auto measure = [&](cudaGraphExec_t graph) {
          Check(cudaEventRecord(begin, replay_stream)); Check(cudaGraphLaunch(graph, replay_stream));
          Check(cudaEventRecord(end, replay_stream)); Check(cudaEventSynchronize(end));
          float ms = 0.0F; Check(cudaEventElapsedTime(&ms, begin, end)); return ms * 1000.0F / 16.0F;
        };
        for (unsigned warm = 0U; warm < 3U; ++warm) { (void)measure(pg); if (cg) (void)measure(cg); }
        std::array<float, 10> p{}, c{};
        for (unsigned i = 0U; i < 10U; ++i) {
          if (cg && i % 2U) c[i] = measure(cg);
          p[i] = measure(pg); if (cg && i % 2U == 0U) c[i] = measure(cg);
        }
        if (cg) { Check(cudaGraphLaunch(cg, replay_stream)); exact = exact && equal(); }
        summary << (phase ? "," : "") << "{\"phase\":" << phase << ",\"layer\":29,\"position\":" << state.position
                << ",\"candidate_tested\":" << (cg ? "true" : "false")
                << ",\"partial_and_lse_bit_exact\":" << (exact ? "true" : "false") << ",\"parent_us_samples\":[";
        for (unsigned i = 0U; i < 10U; ++i) summary << (i ? "," : "") << p[i];
        summary << "],\"candidate_us_samples\":[";
        if (cg) for (unsigned i = 0U; i < 10U; ++i) summary << (i ? "," : "") << c[i];
        summary << "]}";
        Check(cudaGraphExecDestroy(pg)); if (cg) Check(cudaGraphExecDestroy(cg));
        Check(cudaEventDestroy(begin)); Check(cudaEventDestroy(end));
      }
      summary << "]}\n"; Check(cudaStreamDestroy(replay_stream)); injecting = false;
      return 0;
    }
    std::ofstream summary(root / "captures.json");
    summary.precision(9);
    summary << "{\"diagnostic_only\":true,\"extra_device_bytes\":" << kArenaBytes
            << ",\"weights_copied\":false,\"mode\":\"" << (t3 ? "t3" : "m1") << "\",\"cases\":[";
    unsigned count = 0U;
    for (unsigned phase = 0U; phase < 3U; ++phase) {
      if (!saved[phase]) throw std::runtime_error("operator phase not saved");
      for (unsigned i = 0U; i < 8U; ++i) {
        const unsigned index = 8U + phase * 8U + i;
        const auto& r = records[index]; const auto b = Buffer(index);
        const auto stem = "phase" + std::to_string(phase) + "-layer" + std::to_string(kLayers[i / 2U]) + (i % 2U ? "-down" : "-gate");
        Export(root / (stem + ".activation.bin"), b.activation, r.assignments * r.input_elements);
        Export(root / (stem + ".scales.bin"), b.scales, r.assignments * 4U);
        Export(root / (stem + ".ids.bin"), b.ids, r.assignments * 4U);
        Export(root / (stem + ".expected.bin"), b.expected, r.assignments * r.output_elements * 4U);
        if (i % 2U == 0U && r.transformed_valid) {
          Export(root / (stem + ".transformed.bin"), b.transformed, r.assignments * r.input_elements * 4U);
          Export(root / (stem + ".raw.bin"), b.raw, r.raw_bytes);
        }
        std::array<std::uint32_t, 24> ids{};
        Check(cudaMemcpy(ids.data(), b.ids, r.assignments * 4U, cudaMemcpyDeviceToHost));
        unsigned k3 = 0U, k4 = 0U, unique = 0U;
        for (unsigned j = 0U; j < r.assignments; ++j) {
          if (ids[j] >= 128U) throw std::runtime_error("invalid captured expert");
          const auto rate = r.family.rate_map[ids[j]];
          if (rate == 3U) ++k3; else if (rate == 4U) ++k4;
          else throw std::runtime_error("invalid captured rate");
          bool prior = false; for (unsigned k = 0U; k < j; ++k) prior |= ids[k] == ids[j];
          if (!prior) ++unique;
        }
        // Replay exactly the executed projection against its captured output,
        // while borrowing only the original resident immutable weight binding.
        const std::uint8_t* activation = b.activation;
        const float* scales = b.scales;
        const std::uint32_t* selected = b.ids;
        auto family = r.family; auto input = r.input_elements, output = r.output_elements;
        float* temporary = b.temporary;
        void* args[]{&activation, &scales, &family, &selected, &temporary, &input, &output};
        const char* profile_case = std::getenv("GEM16_LAB_PROFILE_CASE");
        const char* profile_kind = std::getenv("GEM16_LAB_PROFILE_KIND");
        const bool profile = profile_case && stem == profile_case;
        if (profile && profile_kind && std::strcmp(profile_kind, "parent") == 0) Check(cudaProfilerStart());
        Check(real_launch(r.function, r.grid, r.block, args, 0U, nullptr));
        Check(cudaDeviceSynchronize());
        if (profile && profile_kind && std::strcmp(profile_kind, "parent") == 0) Check(cudaProfilerStop());
        std::vector<float> expected(r.assignments * output), actual(expected.size());
        Check(cudaMemcpy(expected.data(), b.expected, expected.size() * 4U, cudaMemcpyDeviceToHost));
        Check(cudaMemcpy(actual.data(), b.temporary, actual.size() * 4U, cudaMemcpyDeviceToHost));
        if (std::memcmp(expected.data(), actual.data(), actual.size() * 4U)) throw std::runtime_error("real-input projection replay mismatch");
        bool candidate_exact = false;
        std::array<float, 10> parent_us{}, candidate_us{};
        if (test_projection) {
          if (profile && profile_kind && std::strcmp(profile_kind, "candidate") == 0) Check(cudaProfilerStart());
          Check(candidate_projection(activation, scales, &family, selected, temporary, input, output, nullptr));
          Check(cudaDeviceSynchronize());
          if (profile && profile_kind && std::strcmp(profile_kind, "candidate") == 0) Check(cudaProfilerStop());
          Check(cudaMemcpy(actual.data(), b.temporary, actual.size() * 4U, cudaMemcpyDeviceToHost));
          candidate_exact = std::memcmp(expected.data(), actual.data(), actual.size() * 4U) == 0;
        }
        cudaEvent_t begin{}, end{};
        Check(cudaEventCreate(&begin)); Check(cudaEventCreate(&end));
        const auto parent_graph = ReplayGraph(replay_stream, [&] {
          Check(real_launch(r.function, r.grid, r.block, args, 0U, replay_stream));
        });
        const auto candidate_graph = test_projection ? ReplayGraph(replay_stream, [&] {
          Check(candidate_projection(activation, scales, &family, selected, temporary, input, output, replay_stream));
        }) : nullptr;
        auto measure = [&](bool candidate) {
          Check(cudaEventRecord(begin, replay_stream));
          Check(cudaGraphLaunch(candidate ? candidate_graph : parent_graph, replay_stream));
          Check(cudaEventRecord(end, replay_stream)); Check(cudaEventSynchronize(end));
          float ms = 0.0F; Check(cudaEventElapsedTime(&ms, begin, end));
          return ms * 1000.0F / 16.0F;
        };
        for (unsigned warmup = 0U; warmup < 3U; ++warmup) {
          (void)measure(false); if (test_projection) (void)measure(true);
        }
        for (unsigned sample = 0U; sample < 10U; ++sample) {
          // Alternate order rather than measuring all parents then candidates.
          if (test_projection && sample % 2U) candidate_us[sample] = measure(true);
          parent_us[sample] = measure(false);
          if (test_projection && sample % 2U == 0U) candidate_us[sample] = measure(true);
        }
        if (candidate_graph) {
          Check(cudaGraphLaunch(candidate_graph, replay_stream)); Check(cudaStreamSynchronize(replay_stream));
          Check(cudaMemcpy(actual.data(), b.temporary, actual.size() * 4U, cudaMemcpyDeviceToHost));
          candidate_exact = candidate_exact && std::memcmp(expected.data(), actual.data(), actual.size() * 4U) == 0;
          Check(cudaGraphExecDestroy(candidate_graph));
        }
        Check(cudaGraphExecDestroy(parent_graph));
        bool fusion_exact = false, transform_exact = false;
        std::array<float, 10> fusion_parent_us{}, fusion_candidate_us{};
        const bool fusion_available = test_fusion && i % 2U == 0U && r.transformed_valid && r.transform_function && r.quantize_function;
        if (fusion_available) {
          const float* raw = b.raw;
          const std::uint16_t* suh = family.suh_f16;
          unsigned assignments = r.assignments, per_input = 8U;
          std::uint64_t stride = t3 ? 2816U : 0U, logical = 2816U, physical = 2816U;
          float* transformed = b.temporary;
          auto* quantized = reinterpret_cast<std::uint8_t*>(b.raw) + ((r.raw_bytes + 255U) / 256U) * 256U;
          auto* output_scales = reinterpret_cast<float*>(quantized + kElements);
          void* transform_args[]{&raw, &suh, &selected, &transformed, &assignments, &stride, &per_input, &logical, &physical};
          void* quantize_args[]{&transformed, &quantized, &output_scales, &physical};
          Check(gem16_lab_launch_gate_fused(raw, &family, selected, quantized, output_scales, transformed, assignments, true, nullptr));
          Check(cudaDeviceSynchronize());
          std::vector<float> reference_transform(assignments * input), candidate_transform(reference_transform.size());
          std::vector<std::uint8_t> reference_fp8(assignments * input), candidate_fp8(reference_fp8.size());
          std::vector<float> reference_scales(assignments), candidate_scales(assignments);
          Check(cudaMemcpy(reference_transform.data(), b.transformed, reference_transform.size() * 4U, cudaMemcpyDeviceToHost));
          Check(cudaMemcpy(candidate_transform.data(), transformed, candidate_transform.size() * 4U, cudaMemcpyDeviceToHost));
          Check(cudaMemcpy(reference_fp8.data(), b.activation, reference_fp8.size(), cudaMemcpyDeviceToHost));
          Check(cudaMemcpy(candidate_fp8.data(), quantized, candidate_fp8.size(), cudaMemcpyDeviceToHost));
          Check(cudaMemcpy(reference_scales.data(), b.scales, assignments * 4U, cudaMemcpyDeviceToHost));
          Check(cudaMemcpy(candidate_scales.data(), output_scales, assignments * 4U, cudaMemcpyDeviceToHost));
          transform_exact = std::memcmp(reference_transform.data(), candidate_transform.data(), reference_transform.size() * 4U) == 0;
          fusion_exact = transform_exact && reference_fp8 == candidate_fp8 &&
              std::memcmp(reference_scales.data(), candidate_scales.data(), assignments * 4U) == 0;
          const auto fusion_parent_graph = ReplayGraph(replay_stream, [&] {
            Check(real_launch(r.transform_function, r.transform_grid, r.transform_block, transform_args, 0U, replay_stream));
            Check(real_launch(r.quantize_function, r.quantize_grid, r.quantize_block, quantize_args, 0U, replay_stream));
          });
          const auto fusion_candidate_graph = ReplayGraph(replay_stream, [&] {
            Check(gem16_lab_launch_gate_fused(raw, &family, selected, quantized, output_scales, nullptr, assignments, false, replay_stream));
          });
          auto fusion_measure = [&](bool candidate) {
            Check(cudaEventRecord(begin, replay_stream));
            Check(cudaGraphLaunch(candidate ? fusion_candidate_graph : fusion_parent_graph, replay_stream));
            Check(cudaEventRecord(end, replay_stream)); Check(cudaEventSynchronize(end));
            float ms = 0.0F; Check(cudaEventElapsedTime(&ms, begin, end));
            return ms * 1000.0F / 16.0F;
          };
          for (unsigned warmup = 0U; warmup < 3U; ++warmup) { (void)fusion_measure(false); (void)fusion_measure(true); }
          for (unsigned sample = 0U; sample < 10U; ++sample) {
            if (sample % 2U) fusion_candidate_us[sample] = fusion_measure(true);
            fusion_parent_us[sample] = fusion_measure(false);
            if (sample % 2U == 0U) fusion_candidate_us[sample] = fusion_measure(true);
          }
          Check(cudaGraphLaunch(fusion_candidate_graph, replay_stream)); Check(cudaStreamSynchronize(replay_stream));
          Check(cudaMemcpy(candidate_fp8.data(), quantized, candidate_fp8.size(), cudaMemcpyDeviceToHost));
          Check(cudaMemcpy(candidate_scales.data(), output_scales, assignments * 4U, cudaMemcpyDeviceToHost));
          fusion_exact = fusion_exact && reference_fp8 == candidate_fp8 &&
              std::memcmp(reference_scales.data(), candidate_scales.data(), assignments * 4U) == 0;
          Check(cudaGraphExecDestroy(fusion_parent_graph)); Check(cudaGraphExecDestroy(fusion_candidate_graph));
        }
        Check(cudaEventDestroy(begin)); Check(cudaEventDestroy(end));
        summary << (count++ ? "," : "") << "{\"stem\":\"" << stem << "\",\"phase\":" << phase
                << ",\"layer\":" << kLayers[i / 2U] << ",\"assignments\":" << r.assignments
                << ",\"input\":" << input << ",\"output\":" << output << ",\"k3\":" << k3
                << ",\"k4\":" << k4 << ",\"unique\":" << unique << ",\"projection_bit_exact\":true"
                << ",\"candidate_eight_warp_bit_exact\":" << (candidate_exact ? "true" : "false")
                << ",\"timing_scope\":\"diagnostic warm-resident CUDA graph replay, 16 operations/sample\",\"parent_us_samples\":[";
        for (unsigned sample = 0U; sample < 10U; ++sample) summary << (sample ? "," : "") << parent_us[sample];
        summary << "],\"candidate_us_samples\":[";
        if (test_projection) for (unsigned sample = 0U; sample < 10U; ++sample) summary << (sample ? "," : "") << candidate_us[sample];
        summary << "],\"fusion_available\":" << (fusion_available ? "true" : "false")
                << ",\"fusion_transform_bit_exact\":" << (transform_exact ? "true" : "false")
                << ",\"fusion_fp8_scale_bit_exact\":" << (fusion_exact ? "true" : "false")
                << ",\"fusion_parent_us_samples\":[";
        if (fusion_available) for (unsigned sample = 0U; sample < 10U; ++sample) summary << (sample ? "," : "") << fusion_parent_us[sample];
        summary << "],\"fusion_candidate_us_samples\":[";
        if (fusion_available) for (unsigned sample = 0U; sample < 10U; ++sample) summary << (sample ? "," : "") << fusion_candidate_us[sample];
        summary << "]}";
      }
    }
    summary << "]}\n";
    Check(cudaStreamDestroy(replay_stream));
    injecting = false;
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "operator lab: %s\n", e.what());
    return 1;
  }
}
