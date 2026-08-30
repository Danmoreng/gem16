#include "cuda/trellis35/reference.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <utility>

#include "cuda/fp8/reference.h"
#include "cuda/moe/prefill.h"
#include "exllamav3_quant/util.cuh"
#include "exllamav3_quant/quant/codebook.cuh"

namespace gem16::internal {
namespace {

constexpr unsigned kThreads = 256U;
constexpr unsigned kMmaWarps = 4U;
constexpr unsigned kMmaThreads = 32U * kMmaWarps;
constexpr unsigned kMmaN128Warps = 16U;
constexpr unsigned kMmaN128Threads = 32U * kMmaN128Warps;
constexpr float kHadamardScale = 0.08838834764831845F;
constexpr float kGeluScale = 0.7978845608028654F;
constexpr float kGeluCubic = 0.044715F;
constexpr float kE4M3Maximum = 448.0F;
constexpr unsigned kPrefillLegacyRowsPerTile = 4U;
constexpr unsigned kPrefillGroupedRowsPerTile = 32U;
constexpr unsigned kPrefillM64RowsPerTile = 64U;
constexpr unsigned kPrefillOutputBlock = 128U;


#include "cuda/trellis35/detail/codec.cuh"
#include "cuda/trellis35/detail/mma_w4a8.cuh"
#include "cuda/trellis35/detail/h128_warp.cuh"
#include "cuda/trellis35/detail/transform_common.cuh"
#include "cuda/trellis35/detail/m1_kernels.inc.cuh"
#include "cuda/trellis35/detail/t3_kernels.inc.cuh"
#include "cuda/trellis35/detail/prefill_kernels.inc.cuh"
#include "cuda/trellis35/detail/transform_epilogue.inc.cuh"
#include "cuda/trellis35/detail/launchers.inc.cuh"
