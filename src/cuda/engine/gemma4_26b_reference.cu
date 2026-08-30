#include "cuda/engine/gemma4_26b_reference.h"

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "cuda/attention/gemma4_26b_reference.h"
#include "cuda/attention/sm120.h"
#include "cuda/engine/gemma4_26b_artifact.h"
#include "cuda/engine/gemma4_26b_trellis35_artifact.h"
#include "cuda/layer/reference.h"
#include "cuda/moe/prefill.h"
#include "cuda/moe/reference.h"
#include "cuda/mtp/gemma4_26b_assistant.h"
#include "cuda/mtp/verify.h"
#include "cuda/nvfp4/reference.h"
#include "cuda/nvfp4/sm120.h"
#include "cuda/output_head.h"
#include "cuda/sampling/sampling.h"
#include "cuda/trellis35/reference.h"
#include "gem16/model.h"
#include "model/config.h"
#include "model/gemma4_26b_attention.h"
#include "model/gemma4_26b_residency.h"

namespace gem16::internal {
namespace {


#include "cuda/engine/detail/gemma4_26b_common.inc"
#include "cuda/engine/detail/gemma4_26b_state.inc"
#include "cuda/engine/detail/gemma4_26b_decode_graphs.inc"
#include "cuda/engine/detail/gemma4_26b_mtp_graphs.inc"
#include "cuda/engine/detail/gemma4_26b_create.inc"
#include "cuda/engine/detail/gemma4_26b_decode.inc"
#include "cuda/engine/detail/gemma4_26b_prefill.inc"
#include "cuda/engine/detail/gemma4_26b_mtp.inc"
#include "cuda/engine/detail/gemma4_26b_metrics.inc"
