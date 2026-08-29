#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::uint64_t kExperts = 128U;
constexpr std::uint64_t kInput = 2816U;
constexpr std::uint64_t kGateUp = 1408U;
constexpr std::uint64_t kExpert = 704U;
constexpr std::uint64_t kMaximumRecords = 4096U;

void Cuda(cudaError_t value, const char* operation) {
  if (value != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(value));
  }
}

void Blas(cublasStatus_t value, const char* operation) {
  if (value != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(operation) + ": cuBLAS status " +
                             std::to_string(static_cast<int>(value)));
  }
}

std::uint64_t Unsigned(const char* text, const char* name) {
  if (text == nullptr || *text == '\0' || *text == '-') {
    throw std::runtime_error(std::string(name) + " must be an unsigned integer");
  }
  errno = 0;
  char* end = nullptr;
  const unsigned long long value = std::strtoull(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0') {
    throw std::runtime_error(std::string(name) + " is invalid");
  }
  return static_cast<std::uint64_t>(value);
}

__global__ void GatedGelu(const float* gate_up, float* product,
                          std::uint64_t values) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= values) return;
  const std::uint64_t row = index / kExpert;
  const std::uint64_t column = index % kExpert;
  const float gate = gate_up[row * kGateUp + column];
  const float up = gate_up[row * kGateUp + kExpert + column];
  constexpr float coefficient = 0.7978845608028654F;
  const float inner = coefficient * (gate + 0.044715F * gate * gate * gate);
  product[index] = 0.5F * gate * (1.0F + tanhf(inner)) * up;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 6) {
      std::cerr << "usage: gem16-trellis35-expand-down <records> <gate-input-f32> "
                   "<source-bf16-shard> <tensor-absolute-offset> <products-f32>\n";
      return 2;
    }
    const std::uint64_t records = Unsigned(argv[1], "records");
    const std::uint64_t source_offset = Unsigned(argv[4], "source offset");
    if (records == 0U || records > kMaximumRecords) {
      throw std::runtime_error("records exceed the bounded Trellis35 extent");
    }
    const std::filesystem::path input_path(argv[2]);
    const std::filesystem::path source_path(argv[3]);
    const std::filesystem::path output_path(argv[5]);
    if (input_path == source_path || input_path == output_path || source_path == output_path) {
      throw std::runtime_error("Trellis35 expansion paths must be distinct");
    }
    std::error_code error;
    const std::uint64_t input_bytes = records * kInput * sizeof(float);
    if (std::filesystem::file_size(input_path, error) != input_bytes || error) {
      throw std::runtime_error("Gate+Up calibration input has the wrong byte count");
    }
    const std::uint64_t tensor_bytes = kExperts * kGateUp * kInput * sizeof(std::uint16_t);
    const std::uint64_t source_bytes = std::filesystem::file_size(source_path, error);
    if (error || source_offset > source_bytes || tensor_bytes > source_bytes - source_offset) {
      throw std::runtime_error("BF16 Gate+Up tensor range exceeds the source shard");
    }

    std::vector<float> input(records * kInput);
    std::ifstream input_file(input_path, std::ios::binary);
    input_file.read(reinterpret_cast<char*>(input.data()),
                    static_cast<std::streamsize>(input_bytes));
    if (!input_file || !std::all_of(input.begin(), input.end(),
                                    [](float value) { return std::isfinite(value); })) {
      throw std::runtime_error("Gate+Up calibration input is short or non-finite");
    }
    std::ifstream source(source_path, std::ios::binary);
    source.seekg(static_cast<std::streamoff>(source_offset));
    std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
    if (!source || !output) throw std::runtime_error("cannot open Trellis35 expansion files");

    std::vector<std::uint16_t> weight_bf16(kGateUp * kInput);
    std::vector<float> weight(weight_bf16.size());
    std::vector<float> products(records * kExpert);
    float* device_input = nullptr;
    float* device_weight = nullptr;
    float* device_gate_up = nullptr;
    float* device_product = nullptr;
    Cuda(cudaMalloc(&device_input, input_bytes), "allocate calibration input");
    Cuda(cudaMalloc(&device_weight, weight.size() * sizeof(float)), "allocate BF16 expert expansion");
    Cuda(cudaMalloc(&device_gate_up, records * kGateUp * sizeof(float)), "allocate Gate+Up output");
    Cuda(cudaMalloc(&device_product, products.size() * sizeof(float)), "allocate Down input");
    Cuda(cudaMemcpy(device_input, input.data(), input_bytes, cudaMemcpyHostToDevice),
         "copy calibration input");
    cublasHandle_t handle = nullptr;
    Blas(cublasCreate(&handle), "create cuBLAS handle");
    Blas(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH), "set pedantic cuBLAS math");
    const float alpha = 1.0F;
    const float beta = 0.0F;
    for (std::uint64_t expert = 0; expert < kExperts; ++expert) {
      source.read(reinterpret_cast<char*>(weight_bf16.data()),
                  static_cast<std::streamsize>(weight_bf16.size() * sizeof(std::uint16_t)));
      if (!source) throw std::runtime_error("short BF16 Gate+Up expert read");
      for (std::size_t index = 0; index < weight.size(); ++index) {
        weight[index] = std::bit_cast<float>(static_cast<std::uint32_t>(weight_bf16[index]) << 16U);
      }
      if (!std::all_of(weight.begin(), weight.end(),
                       [](float value) { return std::isfinite(value); })) {
        throw std::runtime_error("BF16 Gate+Up expert contains non-finite values");
      }
      Cuda(cudaMemcpy(device_weight, weight.data(), weight.size() * sizeof(float),
                      cudaMemcpyHostToDevice), "copy BF16 Gate+Up expert");
      Blas(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                       static_cast<int>(kGateUp), static_cast<int>(records),
                       static_cast<int>(kInput), &alpha, device_weight,
                       static_cast<int>(kInput), device_input,
                       static_cast<int>(kInput), &beta, device_gate_up,
                       static_cast<int>(kGateUp)), "project BF16 Gate+Up expert");
      const std::uint64_t values = records * kExpert;
      GatedGelu<<<static_cast<unsigned>((values + 255U) / 256U), 256>>>(
          device_gate_up, device_product, values);
      Cuda(cudaGetLastError(), "launch fused Gate+Up GELU expansion");
      Cuda(cudaMemcpy(products.data(), device_product, products.size() * sizeof(float),
                      cudaMemcpyDeviceToHost), "copy Down calibration inputs");
      if (!std::all_of(products.begin(), products.end(),
                       [](float value) { return std::isfinite(value); })) {
        throw std::runtime_error("expanded Down calibration input is non-finite");
      }
      output.write(reinterpret_cast<const char*>(products.data()),
                   static_cast<std::streamsize>(products.size() * sizeof(float)));
      if (!output) throw std::runtime_error("cannot write expanded Down inputs");
    }
    cublasDestroy(handle);
    cudaFree(device_product);
    cudaFree(device_gate_up);
    cudaFree(device_weight);
    cudaFree(device_input);
    std::cout << "trellis35_expand_down_ok records=" << records
              << " experts=" << kExperts << '\n';
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "trellis35_expand_down_error: " << exception.what() << '\n';
    return 1;
  }
}
