#include <cuda_runtime_api.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "cuda/engine/gemma4_26b_vision_artifact.h"
#include "cuda/vision/gemma4_26b.h"
#include "gem16/image.h"

namespace {

gem16::Status CudaFailure(const char* operation, cudaError_t error) {
  return gem16::Status(
      gem16::StatusCode::kInternal,
      std::string(operation) + ": " + cudaGetErrorName(error) + ": " +
          cudaGetErrorString(error));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4 && argc != 5) {
    std::cerr << "usage: gem16-26b-vision-oracle-dump <vision-module> <image> <output.f32> [input-prefix]\n";
    return 64;
  }
  auto image = gem16::LoadGemma4Moe26BVisionImage(
      argv[2], gem16::Gemma4Moe26BVisionImageOptions{140U});
  if (!image.ok()) {
    std::cerr << "error: " << image.status().message() << '\n';
    return 2;
  }
  auto artifact =
      gem16::internal::Gemma4Moe26BVisionDeviceArtifact::Load(argv[1]);
  if (!artifact.ok()) {
    std::cerr << "error: " << artifact.status().message() << '\n';
    return 2;
  }
  auto runtime =
      gem16::internal::Gemma4Moe26BVisionRuntime::Create(artifact.value());
  if (!runtime.ok()) {
    std::cerr << "error: " << runtime.status().message() << '\n';
    return 2;
  }
  cudaStream_t stream = nullptr;
  cudaError_t error = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (error != cudaSuccess) {
    std::cerr << "error: " << CudaFailure("create stream", error).message()
              << '\n';
    return 2;
  }
  const auto& value = image.value();
  gem16::Gemma4Moe26BVisionInputSegment segment{
      0U, value.soft_token_count, value.soft_token_budget,
      value.raw_patch_count, value.patches, value.positions};
  std::size_t free_before = 0;
  std::size_t total_before = 0;
  gem16::Status status = runtime.value().Encode(segment, stream);
  std::vector<float> output(
      static_cast<std::size_t>(value.soft_token_count) * 2816U);
  if (status.ok()) {
    error = cudaMemcpyAsync(output.data(), runtime.value().output(),
                            output.size() * sizeof(float),
                            cudaMemcpyDeviceToHost, stream);
    if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
    if (error != cudaSuccess) {
      (void)(status = CudaFailure("copy Vision output", error));
    }
  }
  if (status.ok()) {
    error = cudaMemGetInfo(&free_before, &total_before);
    if (error != cudaSuccess) {
      (void)(status = CudaFailure("query recurring free memory", error));
    }
  }
  std::vector<float> repeated(output.size());
  if (status.ok()) (void)(status = runtime.value().Encode(segment, stream));
  if (status.ok()) {
    error = cudaMemcpyAsync(repeated.data(), runtime.value().output(),
                            repeated.size() * sizeof(float),
                            cudaMemcpyDeviceToHost, stream);
    if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
    if (error != cudaSuccess) {
      (void)(status = CudaFailure("copy repeated Vision output", error));
    }
  }
  std::size_t free_after = 0;
  std::size_t total_after = 0;
  if (status.ok()) {
    error = cudaMemGetInfo(&free_after, &total_after);
    if (error != cudaSuccess) {
      (void)(status = CudaFailure("query repeated free memory", error));
    }
  }
  (void)cudaStreamDestroy(stream);
  if (!status.ok()) {
    std::cerr << "error: " << status.message() << '\n';
    return 2;
  }
  std::ofstream file(argv[3], std::ios::binary | std::ios::trunc);
  file.write(reinterpret_cast<const char*>(output.data()),
             static_cast<std::streamsize>(output.size() * sizeof(float)));
  if (!file) {
    std::cerr << "error: cannot write output\n";
    return 2;
  }
  if (argc == 5) {
    std::ofstream patches(std::string(argv[4]) + ".patches.f32",
                          std::ios::binary | std::ios::trunc);
    patches.write(reinterpret_cast<const char*>(value.patches.data()),
                  static_cast<std::streamsize>(value.patches.size() *
                                               sizeof(float)));
    std::ofstream positions(std::string(argv[4]) + ".positions.i32",
                            std::ios::binary | std::ios::trunc);
    positions.write(reinterpret_cast<const char*>(value.positions.data()),
                    static_cast<std::streamsize>(value.positions.size() *
                                                 sizeof(std::int32_t)));
    if (!patches || !positions) {
      std::cerr << "error: cannot write diagnostic Vision input\n";
      return 2;
    }
  }
  std::size_t repeat_mismatches = 0;
  for (std::size_t index = 0; index < output.size(); ++index) {
    repeat_mismatches += output[index] != repeated[index] ? 1U : 0U;
  }
  std::cout << "raw_patches=" << value.raw_patch_count
            << " soft_tokens=" << value.soft_token_count
            << " elements=" << output.size()
            << " repeat_mismatches=" << repeat_mismatches
            << " recurring_device_free_delta="
            << static_cast<std::int64_t>(free_after) -
                   static_cast<std::int64_t>(free_before)
            << '\n';
  return 0;
}
