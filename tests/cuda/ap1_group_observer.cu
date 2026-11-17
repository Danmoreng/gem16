// AP1 test-only group/routing observer. Explicit opt-in; never a primary run.
// Injected nodes write only this observer's preallocated device arena. Export
// happens after the production chain result has been copied back. This adds
// diagnostic work and must be measured separately from the unchanged graph.
#include "cuda/mtp/verify.h"
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
using namespace gem16::internal;
using Launch = cudaError_t (*)(const void*, dim3, dim3, void**, std::size_t, cudaStream_t);
using Begin = cudaError_t (*)(cudaStream_t, cudaStreamCaptureMode);
using Copy = cudaError_t (*)(void*, const void*, std::size_t, cudaMemcpyKind);
constexpr unsigned kGroups = 1134, kLayers = 30, kAssignments = 24;
struct Group {
  MtpGroupResult result{};
  unsigned long long position = 0;
  unsigned routes = 0;
  unsigned ids[kLayers][kAssignments]{};
};
struct State { unsigned group = 0, layer = 0, fault = 0; };
Launch launch = nullptr;
Begin begin = nullptr;
Copy copy = nullptr;
bool enabled = false, ready = false, injecting = false, captured = false;
State* state = nullptr;
Group* groups = nullptr;
const void* chain_pointer = nullptr;
std::array<Group, kGroups> host_groups{};
State host_state{};
MtpChainResult host_chain{};
char output[4096]{};
void Check(cudaError_t error) {
  if (error != cudaSuccess) { std::fprintf(stderr, "AP1 group observer: %s\n", cudaGetErrorString(error)); std::abort(); }
}
template <typename T> T Arg(void** args, unsigned index) {
  T value{}; std::memcpy(&value, args[index], sizeof(value)); return value;
}
__global__ void Ap1BeginGroupKernel(State* state) {
  if (!blockIdx.x && !threadIdx.x) state->layer = 0;
}
__global__ void Ap1ObserveRoutingKernel(const unsigned* ids, State* state, Group* groups) {
  if (blockIdx.x || threadIdx.x) return;
  if (state->group >= kGroups || state->layer >= kLayers) { state->fault = 1; return; }
  auto& group = groups[state->group];
  for (unsigned i = 0; i < kAssignments; ++i) {
    group.ids[state->layer][i] = ids[i];
    if (ids[i] >= 128) state->fault = 1;
  }
  ++state->layer;
}
__global__ void Ap1ObserveGroupKernel(const MtpGroupTransaction* transaction,
    const MtpChainResult* chain, State* state, Group* groups) {
  if (blockIdx.x || threadIdx.x) return;
  if (state->group >= kGroups || chain->group_count != state->group + 1U || state->layer != kLayers) {
    state->fault = 1; return;
  }
  auto& group = groups[state->group];
  group.result = transaction->result;
  // processed_position is the last processed index; expose the pending input's
  // context position, matching prompt length + previously verified outputs.
  group.position = transaction->control.current.processed_position - group.result.output_count + 1U;
  group.routes = state->layer;
  ++state->group;
}
void Save() {
  if (!enabled || !output[0]) return;
  FILE* file = std::fopen(output, "wx");
  if (!file) { std::fprintf(stderr, "AP1 group export exists or cannot be written\n"); return; }
  std::fprintf(file, "{\"diagnostic_only\":true,\"graph_mutated\":true,\"captured\":%s,\"fault\":%u,\"extra_device_bytes\":%zu,\"chain_group_count\":%llu,\"groups\":[",
      captured ? "true" : "false", host_state.fault, sizeof(State) + sizeof(Group) * kGroups,
      static_cast<unsigned long long>(host_chain.group_count));
  if (captured && host_state.group <= kGroups) for (unsigned g = 0; g < host_state.group; ++g) {
    const auto& group = host_groups[g]; const auto& r = group.result;
    std::fprintf(file, "%s{\"group\":%u,\"position\":%llu,\"depth\":%u,\"accepted\":%u,\"output_count\":%u,\"proposals\":[",
        g ? "," : "", g, group.position, r.proposal_count, r.accepted_count, r.output_count);
    for (unsigned i = 0; i < r.proposal_count && i < 4; ++i) std::fprintf(file, "%s%u", i ? "," : "", r.proposed[i]);
    std::fputs("],\"outputs\":[", file);
    for (unsigned i = 0; i < r.output_count && i < 5; ++i) std::fprintf(file, "%s%u", i ? "," : "", r.verified[i]);
    std::fputs("],\"layer_expert_ids\":[", file);
    for (unsigned layer = 0; layer < group.routes && layer < kLayers; ++layer) {
      std::fputs(layer ? ",[" : "[", file);
      for (unsigned i = 0; i < kAssignments; ++i) std::fprintf(file, "%s%u", i ? "," : "", group.ids[layer][i]);
      std::fputc(']', file);
    }
    std::fputs("]}", file);
  }
  std::fputs("]}\n", file); std::fclose(file);
}
__attribute__((constructor)) void Initialize() {
  launch = reinterpret_cast<Launch>(dlsym(RTLD_NEXT, "cudaLaunchKernel"));
  begin = reinterpret_cast<Begin>(dlsym(RTLD_NEXT, "cudaStreamBeginCapture"));
  copy = reinterpret_cast<Copy>(dlsym(RTLD_NEXT, "cudaMemcpy"));
  const char* value = std::getenv("GEM16_AP1_GROUP_OUTPUT");
  enabled = value && std::strlen(value) < sizeof(output);
  if (enabled) { std::strcpy(output, value); std::atexit(Save); }
}
}
extern "C" cudaError_t cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode) {
  if (enabled && !ready) {
    injecting = true;
    Check(cudaMalloc(&state, sizeof(State))); Check(cudaMemset(state, 0, sizeof(State)));
    Check(cudaMalloc(&groups, sizeof(Group) * kGroups)); Check(cudaMemset(groups, 0, sizeof(Group) * kGroups));
    ready = true; injecting = false;
  }
  return begin(stream, mode);
}
extern "C" cudaError_t cudaLaunchKernel(const void* function, dim3 grid, dim3 block,
    void** args, std::size_t shared, cudaStream_t stream) {
  const auto result = launch(function, grid, block, args, shared, stream);
  if (!enabled || !ready || injecting || result != cudaSuccess) return result;
  const char* name = nullptr; Check(cudaFuncGetName(&name, function));
  if (!name) return result;
  injecting = true;
  if (std::strstr(name, "InitializeControlledAssistantKernel"))
    Ap1BeginGroupKernel<<<1, 1, 0, stream>>>(state);
  if (std::strstr(name, "RouterTopKKernel") && grid.x == 3U &&
      Arg<unsigned>(args, 5) == 128U && Arg<unsigned>(args, 6) == 8U)
    Ap1ObserveRoutingKernel<<<1, 1, 0, stream>>>(Arg<const unsigned*>(args, 3), state, groups);
  if (std::strstr(name, "AdvanceMtpD2ChainKernel")) {
    chain_pointer = Arg<const void*>(args, 1);
    Ap1ObserveGroupKernel<<<1, 1, 0, stream>>>(Arg<const MtpGroupTransaction*>(args, 0),
        Arg<const MtpChainResult*>(args, 1), state, groups);
  }
  Check(cudaGetLastError()); injecting = false;
  return result;
}
extern "C" cudaError_t __cudaLaunchKernel(cudaKernel_t kernel, dim3 grid, dim3 block,
    void** args, std::size_t shared, cudaStream_t stream) {
  return cudaLaunchKernel(reinterpret_cast<const void*>(kernel), grid, block, args, shared, stream);
}
extern "C" cudaError_t cudaMemcpy(void* dst, const void* src, std::size_t bytes, cudaMemcpyKind kind) {
  const auto result = copy(dst, src, bytes, kind);
  if (enabled && !captured && result == cudaSuccess && src == chain_pointer &&
      bytes == sizeof(host_chain) && kind == cudaMemcpyDeviceToHost) {
    std::memcpy(&host_chain, dst, bytes);
    Check(copy(&host_state, state, sizeof(State), kind));
    Check(copy(host_groups.data(), groups, sizeof(Group) * kGroups, kind));
    captured = true;
  }
  return result;
}
