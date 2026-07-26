#include <cuda_runtime_api.h>

#include <sstream>
#include <string>

namespace gem16::internal {

std::string CudaCapabilityReport() {
  std::ostringstream output;
  output << "compiled_architectures=" << GEM16_COMPILED_CUDA_ARCH << '\n';
  int runtime_version = 0;
  int driver_version = 0;
  const cudaError_t runtime_status = cudaRuntimeGetVersion(&runtime_version);
  const cudaError_t driver_status = cudaDriverGetVersion(&driver_version);
  int device_count = 0;
  const cudaError_t count_status = cudaGetDeviceCount(&device_count);
  output << "cuda_runtime_version=" << (runtime_status == cudaSuccess ? std::to_string(runtime_version) : cudaGetErrorString(runtime_status)) << '\n';
  output << "cuda_driver_version=" << (driver_status == cudaSuccess ? std::to_string(driver_version) : cudaGetErrorString(driver_status)) << '\n';
  output << "device_count=" << (count_status == cudaSuccess ? std::to_string(device_count) : cudaGetErrorString(count_status)) << '\n';
  if (count_status == cudaSuccess && device_count > 0) {
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, 0) == cudaSuccess) {
      output << "gpu_name=" << properties.name << '\n'
             << "compute_capability=" << properties.major << '.' << properties.minor << '\n'
             << "vram_total_bytes=" << properties.totalGlobalMem << '\n'
             << "cuda_graphs=true\n";
    }
  } else {
    output << "cuda_graphs=false\n";
  }
  output << "nvfp4_correctness_cuda=true\n"
         << "nvfp4_sm120_direct_experimental=true\n"
         << "native_nvfp4_kernels=false\n"
         << "fp8_correctness_cuda=true\n"
         << "fp8_sm120_direct_experimental=true\n"
         << "fp8_kernels=false\n"
         << "status=NVFP4 and FP8 real-shape operator bring-up; layer-golden and model qualification pending\n";
  return output.str();
}

}  // namespace gem16::internal
