#include "gem16/engine.h"

#include <ostream>
#include <string>

namespace gem16 {
namespace internal {
#if GEM16_HAS_CUDA
std::string CudaCapabilityReport();
#endif
}  // namespace internal

void PrintKernelCapabilities(std::ostream& output) {
  output << "compiled_cuda=" << (GEM16_HAS_CUDA ? "true" : "false") << '\n';
#if GEM16_HAS_CUDA
  output << internal::CudaCapabilityReport();
#else
  output << "compiled_architectures=none\n"
         << "native_nvfp4_kernels=false\n"
         << "fp8_kernels=false\n"
         << "cuda_graphs=false\n"
         << "status=host-only build\n";
#endif
}

}  // namespace gem16

