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
  bool native_sm120_available = false;
  output << "cuda_runtime_version=" << (runtime_status == cudaSuccess ? std::to_string(runtime_version) : cudaGetErrorString(runtime_status)) << '\n';
  output << "cuda_driver_version=" << (driver_status == cudaSuccess ? std::to_string(driver_version) : cudaGetErrorString(driver_status)) << '\n';
  output << "device_count=" << (count_status == cudaSuccess ? std::to_string(device_count) : cudaGetErrorString(count_status)) << '\n';
  if (count_status == cudaSuccess && device_count > 0) {
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, 0) == cudaSuccess) {
      native_sm120_available = properties.major == 12 && properties.minor == 0;
      output << "gpu_name=" << properties.name << '\n'
             << "compute_capability=" << properties.major << '.' << properties.minor << '\n'
             << "vram_total_bytes=" << properties.totalGlobalMem << '\n';
    }
  }
  const char* enabled = native_sm120_available ? "true" : "false";
  output << "cuda_graphs=" << enabled << '\n'
         << "nvfp4_correctness_cuda=" << enabled << '\n'
         << "nvfp4_sm120_direct=" << enabled << '\n'
         << "native_nvfp4_kernels=" << enabled << '\n'
         << "fp8_correctness_cuda=" << enabled << '\n'
         << "fp8_sm120_direct=" << enabled << '\n'
         << "fp8_kernels=" << enabled << '\n'
         << "gemma4_12b_path="
         << (native_sm120_available ? "qualified" : "unavailable_incompatible_gpu")
         << '\n'
         << "gemma4_26b_a4b_path="
         << (native_sm120_available ? "experimental_native_sm120"
                                    : "unavailable_incompatible_gpu")
         << '\n'
         << "status="
         << (native_sm120_available
                 ? "profile-specific native SM120 NVFP4/FP8 dispatch available; consult model acceptance evidence"
                 : "this CUDA build requires an SM120 device for native model execution")
         << '\n';
  return output.str();
}

}  // namespace gem16::internal
