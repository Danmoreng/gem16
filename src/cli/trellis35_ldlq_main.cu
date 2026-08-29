#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "third_party/exllamav3_quant/quant/quantize_tiles_kernel.cuh"

namespace {

constexpr std::uint64_t kMaximumRows = 4096U;
constexpr std::uint64_t kMaximumColumns = 4096U;

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
    throw std::runtime_error(std::string(name) + " must be unsigned");
  }
  errno = 0;
  char* end = nullptr;
  const unsigned long long value = std::strtoull(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0') {
    throw std::runtime_error(std::string(name) + " is invalid");
  }
  return static_cast<std::uint64_t>(value);
}

template <typename T>
std::vector<T> ReadExact(const std::filesystem::path& path, std::size_t elements) {
  std::error_code error;
  if (std::filesystem::file_size(path, error) != elements * sizeof(T) || error) {
    throw std::runtime_error("native LDLQ input has the wrong byte count");
  }
  std::vector<T> values(elements);
  std::ifstream input(path, std::ios::binary);
  input.read(reinterpret_cast<char*>(values.data()),
             static_cast<std::streamsize>(values.size() * sizeof(T)));
  if (!input) throw std::runtime_error("short native LDLQ input read");
  return values;
}

void WriteExact(const std::filesystem::path& path, const void* data,
                std::size_t bytes) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<const char*>(data),
               static_cast<std::streamsize>(bytes));
  if (!output) throw std::runtime_error("native LDLQ output write failed");
}

__device__ __forceinline__ int TensorCorePermutation(int index) {
  const int lane = index / 8;
  const int element = index % 8;
  const int row_base = (lane % 4) * 2;
  const int row = row_base + (element % 2 == 1 ? 1 : 0) +
                  (element % 4 >= 2 ? 8 : 0);
  const int column = lane / 4 + (element >= 4 ? 8 : 0);
  return row * 16 + column;
}

__global__ void DifferenceBelow(const float* weight, const float* quantized,
                                float* error, std::uint64_t first,
                                std::uint64_t elements) {
  const std::uint64_t i =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < elements) {
    error[first + i] = weight[first + i] - quantized[first + i];
  }
}

__global__ void PrepareTiles(const float* weight, const float* compensation,
                             float* tiles, std::uint64_t row_start,
                             std::uint64_t columns) {
  const std::uint64_t p =
      static_cast<std::uint64_t>(blockIdx.x) * 256U + threadIdx.x;
  if (threadIdx.x >= 256U || p >= (columns / 16U) * 256U) return;
  const std::uint64_t tile = p / 256U;
  const int original = TensorCorePermutation(static_cast<int>(p % 256U));
  const std::uint64_t row = static_cast<std::uint64_t>(original / 16);
  const std::uint64_t column =
      tile * 16U + static_cast<std::uint64_t>(original % 16);
  tiles[p] = weight[(row_start + row) * columns + column] +
             compensation[row * columns + column];
}

__global__ void StoreTiles(const float* tiles, float* quantized,
                           std::uint64_t row_start, std::uint64_t columns) {
  const std::uint64_t p =
      static_cast<std::uint64_t>(blockIdx.x) * 256U + threadIdx.x;
  if (threadIdx.x >= 256U || p >= (columns / 16U) * 256U) return;
  const std::uint64_t tile = p / 256U;
  const int original = TensorCorePermutation(static_cast<int>(p % 256U));
  const std::uint64_t row = static_cast<std::uint64_t>(original / 16);
  const std::uint64_t column =
      tile * 16U + static_cast<std::uint64_t>(original % 16);
  quantized[(row_start + row) * columns + column] = tiles[p];
}

template <int K>
void Run(std::uint64_t rows, std::uint64_t columns,
         const std::vector<float>& weight, const std::vector<float>& ldl,
         const std::vector<float>* hessian,
         std::vector<float>* reconstructed, std::vector<std::uint16_t>* encoded,
         std::vector<float>* proxy_metrics) {
  constexpr int edges = 65536 >> K;
  constexpr int dynamic_shared = 2 * edges * sizeof(half) + 512 + 64 + 128;
  const std::uint64_t matrix_elements = rows * columns;
  const std::uint64_t tile_columns = columns / 16U;
  const std::uint64_t tile_values = tile_columns * 256U;
  float* d_weight = nullptr;
  float* d_ldl = nullptr;
  float* d_q = nullptr;
  float* d_error = nullptr;
  float* d_comp = nullptr;
  float* d_tiles = nullptr;
  float* d_qtiles = nullptr;
  float* d_hessian = nullptr;
  float* d_product = nullptr;
  std::uint16_t* d_encoded = nullptr;
  std::uint16_t* d_edges = nullptr;
  half* d_costs = nullptr;
  Cuda(cudaMalloc(&d_weight, matrix_elements * sizeof(float)),
       "allocate LDLQ weight");
  Cuda(cudaMalloc(&d_ldl, rows * rows * sizeof(float)),
       "allocate LDLQ factor");
  Cuda(cudaMalloc(&d_q, matrix_elements * sizeof(float)),
       "allocate LDLQ reconstruction");
  Cuda(cudaMalloc(&d_error, matrix_elements * sizeof(float)),
       "allocate LDLQ error");
  Cuda(cudaMalloc(&d_comp, 16U * columns * sizeof(float)),
       "allocate LDLQ compensation");
  Cuda(cudaMalloc(&d_tiles, tile_values * sizeof(float)),
       "allocate LDLQ tiles");
  Cuda(cudaMalloc(&d_qtiles, tile_values * sizeof(float)),
       "allocate LDLQ quantized tiles");
  Cuda(cudaMalloc(&d_encoded,
                  (rows / 16U) * tile_values * sizeof(std::uint16_t)),
       "allocate LDLQ indices");
  Cuda(cudaMalloc(&d_edges,
                  tile_columns * 256U * edges * sizeof(std::uint16_t)),
       "allocate LDLQ Viterbi edges");
  Cuda(cudaMalloc(&d_costs, sizeof(half)), "allocate LDLQ cost sentinel");
  Cuda(cudaMemcpy(d_weight, weight.data(), matrix_elements * sizeof(float),
                  cudaMemcpyHostToDevice),
       "copy LDLQ weight");
  Cuda(cudaMemcpy(d_ldl, ldl.data(), rows * rows * sizeof(float),
                  cudaMemcpyHostToDevice),
       "copy LDLQ factor");
  if (hessian != nullptr) {
    Cuda(cudaMalloc(&d_hessian, rows * rows * sizeof(float)),
         "allocate LDLQ Hessian");
    Cuda(cudaMalloc(&d_product, matrix_elements * sizeof(float)),
         "allocate LDLQ proxy product");
    Cuda(cudaMemcpy(d_hessian, hessian->data(), rows * rows * sizeof(float),
                    cudaMemcpyHostToDevice),
         "copy LDLQ Hessian");
  }
  Cuda(cudaMemset(d_q, 0, matrix_elements * sizeof(float)),
       "clear LDLQ reconstruction");
  Cuda(cudaFuncSetAttribute(
           quantize_tiles_kernel<K, 2>,
           cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_shared),
       "set LDLQ shared memory");
  cublasHandle_t handle = nullptr;
  Blas(cublasCreate(&handle), "create LDLQ cuBLAS");
  Blas(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH),
       "set LDLQ pedantic math");
  const float alpha = 1.0F;
  const float beta = 0.0F;
  for (std::uint64_t end = rows; end > 0; end -= 16U) {
    const std::uint64_t start = end - 16U;
    const std::uint64_t below = rows - end;
    if (below == 0U) {
      Cuda(cudaMemset(d_comp, 0, 16U * columns * sizeof(float)),
           "clear LDLQ compensation");
    } else {
      const std::uint64_t count = below * columns;
      DifferenceBelow<<<static_cast<unsigned>((count + 255U) / 256U), 256>>>(
          d_weight, d_q, d_error, end * columns, count);
      Cuda(cudaGetLastError(), "launch LDLQ error");
      Blas(cublasSgemm(
               handle, CUBLAS_OP_N, CUBLAS_OP_T,
               static_cast<int>(columns), 16, static_cast<int>(below), &alpha,
               d_error + end * columns, static_cast<int>(columns),
               d_ldl + end * rows + start, static_cast<int>(rows), &beta,
               d_comp, static_cast<int>(columns)),
           "compute LDLQ compensation");
    }
    PrepareTiles<<<static_cast<unsigned>(tile_columns), 256>>>(
        d_weight, d_comp, d_tiles, start, columns);
    Cuda(cudaGetLastError(), "prepare LDLQ tiles");
    quantize_tiles_kernel<K, 2>
        <<<static_cast<unsigned>(tile_columns), 512, dynamic_shared>>>(
            d_tiles, d_qtiles, d_encoded + (start / 16U) * tile_values,
            d_costs, d_edges);
    Cuda(cudaGetLastError(), "launch LDLQ Trellis encoder");
    StoreTiles<<<static_cast<unsigned>(tile_columns), 256>>>(
        d_qtiles, d_q, start, columns);
    Cuda(cudaGetLastError(), "store LDLQ tiles");
  }
  Cuda(cudaDeviceSynchronize(), "synchronize native LDLQ");
  if (hessian != nullptr) {
    DifferenceBelow<<<static_cast<unsigned>((matrix_elements + 255U) / 256U),
                      256>>>(d_weight, d_q, d_error, 0, matrix_elements);
    Cuda(cudaGetLastError(), "launch LDLQ proxy error");
    Blas(cublasSgemm(
             handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(columns),
             static_cast<int>(rows), static_cast<int>(rows), &alpha, d_error,
             static_cast<int>(columns), d_hessian, static_cast<int>(rows),
             &beta, d_product, static_cast<int>(columns)),
         "compute LDLQ error proxy product");
    float numerator = 0.0F;
    Blas(cublasSdot(handle, static_cast<int>(matrix_elements), d_error, 1,
                    d_product, 1, &numerator),
         "reduce LDLQ proxy numerator");
    Blas(cublasSgemm(
             handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(columns),
             static_cast<int>(rows), static_cast<int>(rows), &alpha, d_weight,
             static_cast<int>(columns), d_hessian, static_cast<int>(rows),
             &beta, d_product, static_cast<int>(columns)),
         "compute LDLQ source proxy product");
    float denominator = 0.0F;
    Blas(cublasSdot(handle, static_cast<int>(matrix_elements), d_weight, 1,
                    d_product, 1, &denominator),
         "reduce LDLQ proxy denominator");
    (*proxy_metrics)[0] = numerator;
    (*proxy_metrics)[1] = denominator;
  }
  Cuda(cudaMemcpy(reconstructed->data(), d_q,
                  matrix_elements * sizeof(float), cudaMemcpyDeviceToHost),
       "copy LDLQ reconstruction");
  Cuda(cudaMemcpy(encoded->data(), d_encoded,
                  encoded->size() * sizeof(std::uint16_t),
                  cudaMemcpyDeviceToHost),
       "copy LDLQ indices");
  cublasDestroy(handle);
  cudaFree(d_product);
  cudaFree(d_hessian);
  cudaFree(d_costs);
  cudaFree(d_edges);
  cudaFree(d_encoded);
  cudaFree(d_qtiles);
  cudaFree(d_tiles);
  cudaFree(d_comp);
  cudaFree(d_error);
  cudaFree(d_q);
  cudaFree(d_ldl);
  cudaFree(d_weight);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 8 && argc != 10) {
      std::cerr << "usage: gem16-trellis35-ldlq <K3|K4> <rows> <columns> "
                   "<weight-f32> <ldl-f32> <reconstructed-f32> <encoded-u16> "
                   "[hessian-f32 proxy-metrics-f32]\n";
      return 2;
    }
    const std::string rate(argv[1]);
    const auto rows = Unsigned(argv[2], "rows");
    const auto columns = Unsigned(argv[3], "columns");
    if (rows == 0 || columns == 0 || rows > kMaximumRows ||
        columns > kMaximumColumns || rows % 16U || columns % 16U) {
      throw std::runtime_error("LDLQ dimensions are invalid");
    }
    auto weight = ReadExact<float>(argv[4], rows * columns);
    auto ldl = ReadExact<float>(argv[5], rows * rows);
    std::vector<float> hessian;
    if (argc == 10) hessian = ReadExact<float>(argv[8], rows * rows);
    const auto finite = [](float value) { return std::isfinite(value); };
    if (!std::all_of(weight.begin(), weight.end(), finite) ||
        !std::all_of(ldl.begin(), ldl.end(), finite) ||
        !std::all_of(hessian.begin(), hessian.end(), finite)) {
      throw std::runtime_error("LDLQ inputs are non-finite");
    }
    std::vector<float> q(rows * columns);
    std::vector<std::uint16_t> encoded(
        (rows / 16U) * (columns / 16U) * 256U);
    std::vector<float> metrics(2);
    const auto* hessian_ptr = argc == 10 ? &hessian : nullptr;
    if (rate == "K3") {
      Run<3>(rows, columns, weight, ldl, hessian_ptr, &q, &encoded, &metrics);
    } else if (rate == "K4") {
      Run<4>(rows, columns, weight, ldl, hessian_ptr, &q, &encoded, &metrics);
    } else {
      throw std::runtime_error("LDLQ rate must be K3 or K4");
    }
    if (!std::all_of(q.begin(), q.end(), finite)) {
      throw std::runtime_error("LDLQ reconstruction is non-finite");
    }
    WriteExact(argv[6], q.data(), q.size() * sizeof(float));
    WriteExact(argv[7], encoded.data(),
               encoded.size() * sizeof(std::uint16_t));
    if (argc == 10) {
      if (!std::isfinite(metrics[0]) || !std::isfinite(metrics[1]) ||
          metrics[0] < 0.0F || metrics[1] <= 0.0F) {
        throw std::runtime_error("LDLQ proxy metrics are invalid");
      }
      WriteExact(argv[9], metrics.data(), metrics.size() * sizeof(float));
    }
    std::cout << "trellis35_ldlq_ok rate=" << rate << " rows=" << rows
              << " columns=" << columns << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "trellis35_ldlq_error: " << error.what() << '\n';
    return 1;
  }
}
