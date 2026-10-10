#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void CheckCuda(cudaError_t error, const char* operation) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(error));
  }
}

#define CHECK_CUDA(expression) CheckCuda((expression), #expression)

struct Resources {
  cudaStream_t stream = nullptr;
  cudaStream_t fork_stream = nullptr;
  cudaEvent_t fork = nullptr;
  cudaEvent_t join = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  std::uint32_t* memory = nullptr;
  ~Resources() {
    if (executable != nullptr) (void)cudaGraphExecDestroy(executable);
    if (graph != nullptr) (void)cudaGraphDestroy(graph);
    if (fork != nullptr) (void)cudaEventDestroy(fork);
    if (join != nullptr) (void)cudaEventDestroy(join);
    if (fork_stream != nullptr) (void)cudaStreamDestroy(fork_stream);
    if (stream != nullptr) (void)cudaStreamDestroy(stream);
    if (memory != nullptr) (void)cudaFree(memory);
  }
};

__global__ void SelectBranch(const std::uint32_t* state,
                             cudaGraphConditionalHandle first,
                             cudaGraphConditionalHandle second) {
  cudaGraphSetConditional(first, state[0] == 0U);
  cudaGraphSetConditional(second, state[0] == 1U);
}

__global__ void ContinueLoop(std::uint32_t* state,
                             cudaGraphConditionalHandle condition) {
  ++state[1];
  cudaGraphSetConditional(condition, state[1] < 3U);
}

void TestCopiesInNestedConditionalBodies() {
  constexpr std::size_t kWords = 257U;
  constexpr std::size_t kBytes = kWords * sizeof(std::uint32_t);
  Resources r;
  CHECK_CUDA(cudaStreamCreate(&r.stream));
  CHECK_CUDA(cudaStreamCreate(&r.fork_stream));
  CHECK_CUDA(cudaEventCreateWithFlags(&r.fork, cudaEventDisableTiming));
  CHECK_CUDA(cudaEventCreateWithFlags(&r.join, cudaEventDisableTiming));
  CHECK_CUDA(cudaMalloc(&r.memory, (4U * kWords + 2U) * sizeof(std::uint32_t)));
  std::array<std::uint32_t, 2U * kWords> input{};
  for (std::size_t index = 0U; index < input.size(); ++index) {
    // Includes signed zero, NaN payloads and arbitrary integer bit patterns.
    input[index] = static_cast<std::uint32_t>(index) * 0x9e3779b9U;
  }
  input[0] = 0x80000000U;
  input[1] = 0x7fc12345U;
  auto* output = r.memory + 2U * kWords;
  auto* state = r.memory + 4U * kWords;
  CHECK_CUDA(cudaMemcpy(r.memory, input.data(), input.size() * sizeof(input[0]),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaGraphCreate(&r.graph, 0U));
  cudaGraphConditionalHandle loop = 0U, first = 0U, second = 0U;
  CHECK_CUDA(cudaGraphConditionalHandleCreate(
      &loop, r.graph, 1U, cudaGraphCondAssignDefault));
  CHECK_CUDA(cudaGraphConditionalHandleCreate(
      &first, r.graph, 0U, cudaGraphCondAssignDefault));
  CHECK_CUDA(cudaGraphConditionalHandleCreate(
      &second, r.graph, 0U, cudaGraphCondAssignDefault));
  cudaGraphNodeParams parameters{};
  parameters.type = cudaGraphNodeTypeConditional;
  parameters.conditional.handle = loop;
  parameters.conditional.type = cudaGraphCondTypeWhile;
  parameters.conditional.size = 1U;
  cudaGraphNode_t node = nullptr;
  CHECK_CUDA(cudaGraphAddNode(&node, r.graph, nullptr, nullptr, 0U, &parameters));
  cudaGraph_t loop_body = parameters.conditional.phGraph_out[0];
  cudaGraph_t captured = nullptr;
  CHECK_CUDA(cudaStreamBeginCaptureToGraph(
      r.stream, loop_body, nullptr, nullptr, 0U, cudaStreamCaptureModeThreadLocal));
  SelectBranch<<<1U, 1U, 0U, r.stream>>>(state, first, second);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaStreamEndCapture(r.stream, &captured));
  cudaGraphNode_t route = nullptr;
  std::size_t count = 1U;
  CHECK_CUDA(cudaGraphGetNodes(loop_body, &route, &count));
  if (count != 1U) throw std::runtime_error("route graph node count differs");
  std::array<cudaGraphNode_t, 2U> branches{};
  for (std::size_t branch = 0U; branch < branches.size(); ++branch) {
    parameters = {};
    parameters.type = cudaGraphNodeTypeConditional;
    parameters.conditional.handle = branch == 0U ? first : second;
    parameters.conditional.type = cudaGraphCondTypeIf;
    parameters.conditional.size = 1U;
    CHECK_CUDA(cudaGraphAddNode(&branches[branch], loop_body, &route, nullptr,
                                1U, &parameters));
    cudaGraph_t body = parameters.conditional.phGraph_out[0];
    CHECK_CUDA(cudaStreamBeginCaptureToGraph(
        r.stream, body, nullptr, nullptr, 0U, cudaStreamCaptureModeThreadLocal));
    CHECK_CUDA(cudaMemcpyAsync(output, r.memory + branch * kWords, kBytes,
                               cudaMemcpyDeviceToDevice, r.stream));
    CHECK_CUDA(cudaEventRecord(r.fork, r.stream));
    CHECK_CUDA(cudaStreamWaitEvent(r.fork_stream, r.fork, 0U));
    CHECK_CUDA(cudaMemcpyAsync(output + kWords, output, kBytes,
                               cudaMemcpyDeviceToDevice, r.fork_stream));
    CHECK_CUDA(cudaEventRecord(r.join, r.fork_stream));
    CHECK_CUDA(cudaStreamWaitEvent(r.stream, r.join, 0U));
    CHECK_CUDA(cudaStreamEndCapture(r.stream, &captured));
    if (captured != body) throw std::runtime_error("capture changed body ownership");
  }
  CHECK_CUDA(cudaStreamBeginCaptureToGraph(
      r.stream, loop_body, branches.data(), nullptr, branches.size(),
      cudaStreamCaptureModeThreadLocal));
  ContinueLoop<<<1U, 1U, 0U, r.stream>>>(state, loop);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaStreamEndCapture(r.stream, &captured));
  CHECK_CUDA(cudaGraphInstantiate(&r.executable, r.graph, 0U));
  std::array<std::uint32_t, 2U * kWords> observed{};
  for (std::uint32_t replay = 0U; replay < 20U; ++replay) {
    const std::array<std::uint32_t, 2U> initial{replay % 2U, 0U};
    CHECK_CUDA(cudaMemcpyAsync(state, initial.data(), sizeof(initial),
                               cudaMemcpyHostToDevice, r.stream));
    CHECK_CUDA(cudaMemsetAsync(output, 0x55, observed.size() * sizeof(observed[0]), r.stream));
    CHECK_CUDA(cudaGraphLaunch(r.executable, r.stream));
    CHECK_CUDA(cudaStreamSynchronize(r.stream));
    CHECK_CUDA(cudaMemcpy(observed.data(), output, sizeof(observed), cudaMemcpyDeviceToHost));
    std::array<std::uint32_t, 2U> final{};
    CHECK_CUDA(cudaMemcpy(final.data(), state, sizeof(final), cudaMemcpyDeviceToHost));
    if (final[1] != 3U) throw std::runtime_error("conditional loop iteration count differs");
    for (std::size_t index = 0U; index < observed.size(); ++index) {
      if (observed[index] != input[initial[0] * kWords + index % kWords]) {
        throw std::runtime_error("conditional copy bits or stream dependency differ");
      }
    }
  }
}

}  // namespace

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
  try {
    TestCopiesInNestedConditionalBodies();
    std::cout << "conditional graph: both branches, three iterations, 20 replays, "
                 "bit-exact fork/join copies passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
