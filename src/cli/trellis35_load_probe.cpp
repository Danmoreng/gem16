#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>

#include "cuda/engine/gemma4_26b_trellis35_artifact.h"
#include "gem16/engine.h"
#include "model/gemma4_26b_trellis35.h"

namespace {

int Fail(const gem16::Status& status) {
  std::cerr << status.message() << '\n';
  return 1;
}

gem16::Status CudaFailure(const char* operation, cudaError_t error) {
  return gem16::Status(
      gem16::StatusCode::kInternal,
      std::string(operation) + ": " + cudaGetErrorName(error) + ": " +
          cudaGetErrorString(error));
}

bool IsWithin(const std::byte* pointer, const std::byte* arena,
              std::uint64_t arena_bytes) {
  const auto address = reinterpret_cast<std::uintptr_t>(pointer);
  const auto first = reinterpret_cast<std::uintptr_t>(arena);
  return address >= first && address < first + arena_bytes;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: gem16-trellis35-load-probe CHECKPOINT\n";
    return 2;
  }
  int device = 0;
  cudaError_t error = cudaGetDevice(&device);
  if (error != cudaSuccess) return Fail(CudaFailure("query CUDA device", error));
  cudaDeviceProp properties{};
  error = cudaGetDeviceProperties(&properties, device);
  if (error != cudaSuccess) {
    return Fail(CudaFailure("query CUDA device properties", error));
  }
  if (properties.major != 12 || properties.minor != 0) {
    return Fail(gem16::Status(gem16::StatusCode::kUnsupported,
                              "Trellis35 WP3 probe requires SM120"));
  }
  std::size_t free_before = 0U;
  std::size_t total = 0U;
  error = cudaMemGetInfo(&free_before, &total);
  if (error != cudaSuccess) {
    return Fail(CudaFailure("query CUDA memory before Trellis35 load", error));
  }
  auto artifact =
      gem16::internal::Gemma4Moe26BTrellis35DeviceArtifact::Load(argv[1]);
  if (!artifact.ok()) return Fail(artifact.status());
  std::size_t free_loaded = 0U;
  error = cudaMemGetInfo(&free_loaded, &total);
  if (error != cudaSuccess) {
    return Fail(CudaFailure("query CUDA memory after Trellis35 load", error));
  }
  const auto embedding = artifact.value().NonRoutedPointer(
      "model.language_model.embed_tokens.input_global_scale");
  if (!embedding.ok()) return Fail(embedding.status());
  std::uint64_t k3 = 0U;
  std::uint64_t k4 = 0U;
  for (const auto& layer : artifact.value().layers()) {
    for (const auto* family : {&layer.gate_up, &layer.down}) {
      if (!IsWithin(family->k3_payload_pool, artifact.value().arena(),
                    artifact.value().arena_bytes()) ||
          !IsWithin(family->k4_payload_pool, artifact.value().arena(),
                    artifact.value().arena_bytes()) ||
          !IsWithin(reinterpret_cast<const std::byte*>(family->descriptors),
                    artifact.value().arena(),
                    artifact.value().arena_bytes()) ||
          !IsWithin(reinterpret_cast<const std::byte*>(family->suh_f16),
                    artifact.value().arena(),
                    artifact.value().arena_bytes()) ||
          !IsWithin(reinterpret_cast<const std::byte*>(family->svh_f16),
                    artifact.value().arena(),
                    artifact.value().arena_bytes())) {
        return Fail(gem16::Status(
            gem16::StatusCode::kDataLoss,
            "Trellis35 initialization left an out-of-arena binding"));
      }
      k3 += static_cast<std::uint64_t>(
          std::count(family->rate_map.begin(), family->rate_map.end(), 3U));
      k4 += static_cast<std::uint64_t>(
          std::count(family->rate_map.begin(), family->rate_map.end(), 4U));
    }
  }
  if (k3 != 3840U || k4 != 3840U) {
    return Fail(gem16::Status(gem16::StatusCode::kDataLoss,
                              "Trellis35 device rate maps are incomplete"));
  }
  const auto& first = artifact.value().layers().front().gate_up;
  gem16::internal::Trellis35ExpertDescriptor descriptor{};
  error = cudaMemcpy(&descriptor, first.descriptors, sizeof(descriptor),
                     cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    return Fail(CudaFailure("read Trellis35 device descriptor", error));
  }
  if (descriptor.rate_bits != first.rate_map.front() ||
      descriptor.codebook_id != gem16::internal::kTrellis35CodebookId) {
    return Fail(gem16::Status(gem16::StatusCode::kDataLoss,
                              "Trellis35 uploaded descriptor differs"));
  }
  const auto* stable_embedding = embedding.value();
  const auto* stable_descriptor = first.descriptors;
  for (std::uint32_t iteration = 0U; iteration < 100'000U; ++iteration) {
    if (artifact.value().layers().front().gate_up.descriptors !=
            stable_descriptor ||
        embedding.value() != stable_embedding) {
      return Fail(gem16::Status(gem16::StatusCode::kInternal,
                                "Trellis35 fixed binding changed"));
    }
  }
  std::size_t free_rebound = 0U;
  error = cudaMemGetInfo(&free_rebound, &total);
  if (error != cudaSuccess) {
    return Fail(CudaFailure("query CUDA memory after binding replay", error));
  }
  if (free_rebound != free_loaded) {
    return Fail(gem16::Status(
        gem16::StatusCode::kInternal,
        "Trellis35 binding replay caused a device allocation delta"));
  }
  const auto dispatch =
      gem16::internal::Gemma4Moe26BTrellis35EngineDispatchStatus();
  if (dispatch.ok() || dispatch.code() != gem16::StatusCode::kUnsupported) {
    return Fail(gem16::Status(
        gem16::StatusCode::kInternal,
        "Trellis35 WP3 dispatch did not reject the missing kernel"));
  }
  gem16::ModelRuntimeOptions runtime_options;
  runtime_options.model_directory = argv[1];
  runtime_options.max_context_tokens = 1024U;
  auto runtime = gem16::ModelRuntime::Load(runtime_options);
  if (runtime.ok() || runtime.status().code() != gem16::StatusCode::kUnsupported ||
      runtime.status().message() != dispatch.message()) {
    return Fail(gem16::Status(
        gem16::StatusCode::kInternal,
        "full runtime did not visibly reject the Trellis35 WP3 profile"));
  }
  std::size_t free_runtime_reject = 0U;
  error = cudaMemGetInfo(&free_runtime_reject, &total);
  if (error != cudaSuccess) {
    return Fail(CudaFailure("query CUDA memory after runtime rejection", error));
  }
  if (free_runtime_reject != free_loaded) {
    return Fail(gem16::Status(
        gem16::StatusCode::kInternal,
        "Trellis35 runtime rejection caused a device allocation delta"));
  }
  const auto& stats = artifact.value().stats();
  std::cout << "{\"status\":\"wp3_device_loader_pass\""
            << ",\"profile\":\""
            << gem16::internal::kGemma4Moe26BTrellis35Profile << "\""
            << ",\"checkpoint_content_sha256\":\""
            << stats.checkpoint_content_sha256 << "\""
            << ",\"arena_bytes\":" << stats.arena_bytes
            << ",\"uploaded_bytes\":" << stats.uploaded_bytes
            << ",\"files\":" << stats.files
            << ",\"non_routed_tensors\":" << stats.non_routed_tensors
            << ",\"device_allocations\":" << stats.device_allocations
            << ",\"nvfp4_routed_expert_bytes\":0"
            << ",\"free_before\":" << free_before
            << ",\"free_loaded\":" << free_loaded
            << ",\"free_after_binding_replay\":" << free_rebound
            << ",\"free_after_runtime_reject\":" << free_runtime_reject
            << ",\"binding_replay_iterations\":100000"
            << ",\"k3_descriptors\":" << k3
            << ",\"k4_descriptors\":" << k4
            << ",\"dispatch_status\":\"unsupported_kernel_not_implemented\""
            << ",\"load_path\":\"" << stats.load_path << "\"}\n";
  return 0;
}
