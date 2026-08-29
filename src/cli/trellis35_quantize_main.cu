#include <cuda_runtime.h>

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "third_party/exllamav3_quant/quant/quantize_tiles_kernel.cuh"

namespace {

constexpr std::size_t kTileValues = 256;
constexpr std::size_t kBatchTiles = 64;
constexpr std::uint64_t kMaximumTiles = 1ULL << 24;

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

std::uint64_t parse_count(const char* text) {
  if (text == nullptr || *text == '\0' || *text == '-') {
    throw std::runtime_error("tile count must be a positive integer");
  }
  errno = 0;
  char* end = nullptr;
  const unsigned long long value = std::strtoull(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || value == 0 || value > kMaximumTiles) {
    throw std::runtime_error("tile count is outside the supported range");
  }
  return static_cast<std::uint64_t>(value);
}

template <int K>
void run(
    const std::filesystem::path& input_path,
    const std::filesystem::path& reconstructed_path,
    const std::filesystem::path& encoded_path,
    std::uint64_t tile_count) {
  constexpr int edges = 65536 >> K;
  constexpr int dynamic_shared = 2 * edges * static_cast<int>(sizeof(half)) + 512 + 64 + 128;
  check_cuda(cudaFuncSetAttribute(
      quantize_tiles_kernel<K, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_shared),
      "setting Trellis35 dynamic shared memory");

  std::ifstream input(input_path, std::ios::binary);
  std::ofstream reconstructed(reconstructed_path, std::ios::binary | std::ios::trunc);
  std::ofstream encoded(encoded_path, std::ios::binary | std::ios::trunc);
  if (!input || !reconstructed || !encoded) {
    throw std::runtime_error("cannot open Trellis35 input/output files");
  }

  const std::size_t max_values = kBatchTiles * kTileValues;
  std::vector<float> host_input(max_values);
  std::vector<float> host_reconstructed(max_values);
  std::vector<std::uint16_t> host_encoded(max_values);
  float* device_input = nullptr;
  float* device_reconstructed = nullptr;
  std::uint16_t* device_encoded = nullptr;
  half* device_costs = nullptr;
  std::uint16_t* device_edges = nullptr;
  check_cuda(cudaMalloc(&device_input, max_values * sizeof(float)), "allocating Trellis35 input");
  check_cuda(cudaMalloc(&device_reconstructed, max_values * sizeof(float)), "allocating Trellis35 reconstruction");
  check_cuda(cudaMalloc(&device_encoded, max_values * sizeof(std::uint16_t)), "allocating Trellis35 indices");
  check_cuda(cudaMalloc(&device_costs, sizeof(half)), "allocating Trellis35 cost sentinel");
  check_cuda(cudaMalloc(
      &device_edges, kBatchTiles * kTileValues * static_cast<std::size_t>(edges) * sizeof(std::uint16_t)),
      "allocating Trellis35 Viterbi edges");

  try {
    for (std::uint64_t first = 0; first < tile_count; first += kBatchTiles) {
      const std::size_t batch = static_cast<std::size_t>(
          std::min<std::uint64_t>(kBatchTiles, tile_count - first));
      const std::size_t values = batch * kTileValues;
      const std::size_t input_bytes = values * sizeof(float);
      input.read(reinterpret_cast<char*>(host_input.data()), static_cast<std::streamsize>(input_bytes));
      if (input.gcount() != static_cast<std::streamsize>(input_bytes)) {
        throw std::runtime_error("short Trellis35 tile input");
      }
      check_cuda(cudaMemcpy(device_input, host_input.data(), input_bytes, cudaMemcpyHostToDevice),
                 "copying Trellis35 input");
      quantize_tiles_kernel<K, 2><<<static_cast<unsigned>(batch), 512, dynamic_shared>>>(
          device_input, device_reconstructed, device_encoded, device_costs, device_edges);
      check_cuda(cudaGetLastError(), "launching Trellis35 quantizer");
      check_cuda(cudaMemcpy(host_reconstructed.data(), device_reconstructed, input_bytes, cudaMemcpyDeviceToHost),
                 "copying Trellis35 reconstruction");
      check_cuda(cudaMemcpy(host_encoded.data(), device_encoded, values * sizeof(std::uint16_t), cudaMemcpyDeviceToHost),
                 "copying Trellis35 indices");
      reconstructed.write(reinterpret_cast<const char*>(host_reconstructed.data()),
                          static_cast<std::streamsize>(input_bytes));
      encoded.write(reinterpret_cast<const char*>(host_encoded.data()),
                    static_cast<std::streamsize>(values * sizeof(std::uint16_t)));
      if (!reconstructed || !encoded) {
        throw std::runtime_error("cannot write Trellis35 quantizer output");
      }
    }
    char trailing = 0;
    if (input.read(&trailing, 1)) {
      throw std::runtime_error("Trellis35 tile input contains trailing bytes");
    }
  } catch (...) {
    cudaFree(device_edges);
    cudaFree(device_costs);
    cudaFree(device_encoded);
    cudaFree(device_reconstructed);
    cudaFree(device_input);
    throw;
  }
  check_cuda(cudaFree(device_edges), "freeing Trellis35 Viterbi edges");
  check_cuda(cudaFree(device_costs), "freeing Trellis35 costs");
  check_cuda(cudaFree(device_encoded), "freeing Trellis35 indices");
  check_cuda(cudaFree(device_reconstructed), "freeing Trellis35 reconstruction");
  check_cuda(cudaFree(device_input), "freeing Trellis35 input");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--describe") {
      std::cout << "{\"codebook_id\":2,\"implementation\":\"gem16_trellis35_quantize_v1\","
                   "\"rates\":[3,4],\"upstream_revision\":"
                   "\"0c49587a7c235e6303a6bbedc8b665272ad3a2ea\"}\n";
      return 0;
    }
    if (argc != 6) {
      std::cerr << "usage: gem16-trellis35-quantize <K3|K4> <tile-count> <input-f32> "
                   "<reconstructed-f32> <encoded-u16>\n";
      return 2;
    }
    const std::string rate(argv[1]);
    const std::uint64_t tile_count = parse_count(argv[2]);
    const std::filesystem::path input(argv[3]);
    const std::filesystem::path reconstructed(argv[4]);
    const std::filesystem::path encoded(argv[5]);
    if (input == reconstructed || input == encoded || reconstructed == encoded) {
      throw std::runtime_error("Trellis35 input and output paths must be distinct");
    }
    const std::uint64_t expected_bytes = tile_count * kTileValues * sizeof(float);
    std::error_code error;
    const std::uint64_t actual_bytes = std::filesystem::file_size(input, error);
    if (error || actual_bytes != expected_bytes) {
      throw std::runtime_error("Trellis35 input file size does not match tile count");
    }
    if (rate == "K3") {
      run<3>(input, reconstructed, encoded, tile_count);
    } else if (rate == "K4") {
      run<4>(input, reconstructed, encoded, tile_count);
    } else {
      throw std::runtime_error("Trellis35 rate must be K3 or K4");
    }
    std::cout << "trellis35_quantize_ok rate=" << rate << " tiles=" << tile_count << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "trellis35_quantize_error: " << error.what() << '\n';
    return 1;
  }
}
