#include "cuda/moe/prefill.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

#include "cuda/layer/reference.h"
#include "cuda/nvfp4/reference.h"
#include "cuda/nvfp4/sm120.h"

namespace gem16::internal {
namespace {

constexpr unsigned kThreads = 128U;
constexpr unsigned kRouterExpertsPerBlock = 32U;
constexpr unsigned kRouterTokensPerBlock = 8U;
constexpr unsigned kRouterThreads =
    kRouterExpertsPerBlock * kRouterTokensPerBlock;
constexpr unsigned kRouterKTile = 32U;
constexpr unsigned kRouterWeightVectorsPerExpert =
    kRouterKTile * sizeof(std::uint16_t) / sizeof(uint4);
static_assert(kRouterExpertsPerBlock * kRouterWeightVectorsPerExpert ==
              kThreads);
constexpr unsigned kGroupingThreads = 256U;
constexpr unsigned kGroupingAssignmentsPerChunk = kGroupingThreads;
constexpr unsigned kMaxParallelGroupingExperts = kThreads;
constexpr std::uint64_t kSm120KBlock = 64U;

Status Invalid(std::string message) {
  return Status(StatusCode::kInvalidArgument, std::move(message));
}

Status CudaFailure(const char* operation, cudaError_t error) {
  return Status(StatusCode::kInternal,
                std::string(operation) + ": " + cudaGetErrorName(error) +
                    ": " + cudaGetErrorString(error));
}

bool PositiveFinite(float value) {
  return std::isfinite(value) && value > 0.0F;
}

bool MatrixValid(const Gemma4MoeNvfp4Matrix& matrix, std::uint64_t rows,
                 std::uint64_t columns) {
  return matrix.packed_e2m1 != nullptr && matrix.scales_e4m3fn != nullptr &&
         matrix.rows == rows && matrix.columns == columns && rows != 0U &&
         rows % 8U == 0U && columns != 0U &&
         columns % kSm120KBlock == 0U &&
         PositiveFinite(matrix.activation_global_divisor) &&
         PositiveFinite(matrix.weight_global_divisor) &&
         PositiveFinite(matrix.activation_global_divisor *
                        matrix.weight_global_divisor);
}

__device__ __forceinline__ float Bf16(std::uint16_t value) {
  return static_cast<float>(__ushort_as_bfloat16(value));
}

__device__ __forceinline__ float RoundBf16(float value) {
  return static_cast<float>(__float2bfloat16_rn(value));
}

__global__ void RoundBf16BatchKernel(float* values, std::uint64_t elements) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) values[index] = RoundBf16(values[index]);
}

__global__ void RouterTransformBatchKernel(
    const float* normalized, const std::uint16_t* scale, float* transformed,
    std::uint64_t tokens, std::uint64_t width) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t total = tokens * width;
  if (index >= total) return;
  const std::uint64_t column = index % width;
  transformed[index] = RoundBf16(normalized[index] * Bf16(scale[column]) *
                                 rsqrtf(static_cast<float>(width)));
}

__global__ void RouterProjectionBatchCoalescedKernel(
    const float* input, const std::uint16_t* weights, float* logits,
    std::uint32_t experts, std::uint64_t width, std::uint64_t tokens) {
  const unsigned token_in_block =
      threadIdx.x / kRouterExpertsPerBlock;
  const std::uint64_t token =
      static_cast<std::uint64_t>(blockIdx.y) * kRouterTokensPerBlock +
      token_in_block;
  __shared__ alignas(16) uint4
      staged_weights[kRouterExpertsPerBlock]
                    [kRouterWeightVectorsPerExpert];
  __shared__ float staged_input[kRouterTokensPerBlock][kRouterKTile];
  const unsigned local_expert =
      threadIdx.x % kRouterExpertsPerBlock;
  const std::uint32_t expert =
      blockIdx.x * kRouterExpertsPerBlock + local_expert;
  const std::uint64_t input_base = token * width;
  float accumulator = 0.0F;
  for (std::uint64_t k_base = 0U; k_base < width;
       k_base += kRouterKTile) {
    if (threadIdx.x <
        kRouterExpertsPerBlock * kRouterWeightVectorsPerExpert) {
      const unsigned load_expert =
          threadIdx.x / kRouterWeightVectorsPerExpert;
      const unsigned weight_vector =
          threadIdx.x % kRouterWeightVectorsPerExpert;
      const std::uint32_t global_expert =
          blockIdx.x * kRouterExpertsPerBlock + load_expert;
      if (global_expert < experts) {
        const std::uint64_t load_weight_base =
            static_cast<std::uint64_t>(global_expert) * width;
        staged_weights[load_expert][weight_vector] =
            reinterpret_cast<const uint4*>(
                weights + load_weight_base + k_base)[weight_vector];
      }
    }
    staged_input[token_in_block][local_expert] =
        token < tokens ? input[input_base + k_base + local_expert] : 0.0F;
    __syncthreads();
    if (token < tokens && expert < experts) {
      const auto* staged_row = reinterpret_cast<const std::uint16_t*>(
          staged_weights[local_expert]);
#pragma unroll 1
      for (unsigned index = 0U; index < kRouterKTile; ++index) {
        accumulator =
            fmaf(Bf16(staged_row[index]),
                 staged_input[token_in_block][index], accumulator);
      }
    }
    __syncthreads();
  }
  if (token < tokens && expert < experts) {
    logits[token * experts + expert] = RoundBf16(accumulator);
  }
}

__global__ void RouterAssignmentsKernel(
    const float* logits, const std::uint16_t* per_expert_scale,
    float* probabilities, Gemma4MoePrefillAssignment* assignments,
    std::uint32_t experts, std::uint32_t top_k, std::uint64_t tokens,
    int* routing_finite) {
  const std::uint64_t token = blockIdx.x;
  if (token >= tokens) return;
  logits += token * experts;
  probabilities += token * experts;
  assignments += token * top_k;
  __shared__ float maximum;
  __shared__ float total;
  __shared__ int valid;
  __shared__ float candidate_probability[kThreads];
  __shared__ std::uint32_t candidate_id[kThreads];
  if (threadIdx.x == 0U) {
    maximum = -3.402823466e+38F;
    valid = 1;
    for (std::uint32_t expert = 0; expert < experts; ++expert) {
      const float logit = logits[expert];
      if (!isfinite(logit)) valid = 0;
      maximum = fmaxf(maximum, logit);
    }
  }
  __syncthreads();
  for (std::uint32_t expert = threadIdx.x; expert < experts;
       expert += blockDim.x) {
    probabilities[expert] = expf(logits[expert] - maximum);
  }
  __syncthreads();
  if (threadIdx.x == 0U && valid) {
    total = 0.0F;
    for (std::uint32_t expert = 0; expert < experts; ++expert) {
      total += probabilities[expert];
    }
    if (!isfinite(total) || total <= 0.0F) valid = 0;
  }
  __syncthreads();
  for (std::uint32_t expert = threadIdx.x; expert < experts;
       expert += blockDim.x) {
    if (!valid) probabilities[expert] = 0.0F;
    else
    probabilities[expert] /= total;
  }
  __syncthreads();
  for (std::uint32_t slot = 0; slot < top_k; ++slot) {
    std::uint32_t best_id = experts;
    float best_probability = -1.0F;
    for (std::uint32_t expert = threadIdx.x; expert < experts;
         expert += blockDim.x) {
      bool used = false;
      for (std::uint32_t previous = 0; previous < slot; ++previous) {
        used = used || assignments[previous].expert_id == expert;
      }
      const float probability = probabilities[expert];
      if (!used && (probability > best_probability ||
                    (probability == best_probability && expert < best_id))) {
        best_probability = probability;
        best_id = expert;
      }
    }
    candidate_probability[threadIdx.x] = best_probability;
    candidate_id[threadIdx.x] = best_id;
    __syncthreads();
    for (unsigned stride = kThreads / 2U; stride != 0U; stride /= 2U) {
      if (threadIdx.x < stride) {
        const float right_probability =
            candidate_probability[threadIdx.x + stride];
        const std::uint32_t right_id = candidate_id[threadIdx.x + stride];
        if (right_probability > candidate_probability[threadIdx.x] ||
            (right_probability == candidate_probability[threadIdx.x] &&
             right_id < candidate_id[threadIdx.x])) {
          candidate_probability[threadIdx.x] = right_probability;
          candidate_id[threadIdx.x] = right_id;
        }
      }
      __syncthreads();
    }
    if (threadIdx.x == 0U) {
      if (candidate_id[0] >= experts) {
        valid = 0;
      } else {
        assignments[slot] = {
            static_cast<std::uint16_t>(candidate_id[0]),
            static_cast<std::uint16_t>(slot), static_cast<std::uint32_t>(token),
            candidate_probability[0]};
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0U) {
    float selected_total = 0.0F;
    if (valid) {
      for (std::uint32_t slot = 0; slot < top_k; ++slot) {
        selected_total += assignments[slot].weight;
      }
      if (!isfinite(selected_total) || selected_total <= 0.0F) valid = 0;
    }
    if (valid) {
      for (std::uint32_t slot = 0; slot < top_k; ++slot) {
        auto& assignment = assignments[slot];
        const float scale = Bf16(per_expert_scale[assignment.expert_id]);
        if (!isfinite(scale)) {
          valid = 0;
          break;
        }
        assignment.weight = (assignment.weight / selected_total) * scale;
      }
    }
    if (!valid) {
      for (std::uint32_t slot = 0; slot < top_k; ++slot) {
        assignments[slot] = {0U, static_cast<std::uint16_t>(slot),
                             static_cast<std::uint32_t>(token), 0.0F};
      }
      if (routing_finite != nullptr) atomicExch(routing_finite, 0);
    }
  }
}

__global__ void StableGroupAssignmentsSerialKernel(
    const Gemma4MoePrefillAssignment* assignments,
    std::uint32_t* histogram, std::uint32_t* prefix,
    std::uint32_t* permutation, std::uint32_t* inverse,
    std::uint32_t experts, std::uint64_t assignment_count) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) return;
  for (std::uint32_t expert = 0; expert < experts; ++expert) {
    histogram[expert] = 0U;
  }
  for (std::uint64_t index = 0; index < assignment_count; ++index) {
    const std::uint32_t expert = assignments[index].expert_id;
    if (expert < experts) ++histogram[expert];
  }
  prefix[0] = 0U;
  for (std::uint32_t expert = 0; expert < experts; ++expert) {
    prefix[expert + 1U] = prefix[expert] + histogram[expert];
    histogram[expert] = 0U;
  }
  // Original assignment order is token ascending, then slot ascending. The
  // cursor therefore creates the required stable order inside each expert.
  for (std::uint32_t original = 0U; original < assignment_count; ++original) {
    const std::uint32_t expert = assignments[original].expert_id;
    if (expert >= experts) continue;
    const std::uint32_t grouped = prefix[expert] + histogram[expert]++;
    permutation[grouped] = original;
    inverse[original] = grouped;
  }
}

// Count each 256-assignment chunk once. The shared atomics only compute exact
// integer counts; they never determine output order.
__global__ void CountGroupAssignmentsByChunkKernel(
    const Gemma4MoePrefillAssignment* assignments,
    std::uint32_t* chunk_offsets,
    std::uint32_t experts, std::uint64_t assignment_count) {
  __shared__ std::uint32_t counts[kMaxParallelGroupingExperts];
  if (threadIdx.x < experts) counts[threadIdx.x] = 0U;
  __syncthreads();
  const std::uint64_t original =
      static_cast<std::uint64_t>(blockIdx.x) *
          kGroupingAssignmentsPerChunk +
      threadIdx.x;
  if (original < assignment_count) {
    const std::uint32_t expert = assignments[original].expert_id;
    if (expert < experts) atomicAdd(counts + expert, 1U);
  }
  __syncthreads();
  if (threadIdx.x < experts) {
    chunk_offsets[static_cast<std::uint64_t>(blockIdx.x) * experts +
                  threadIdx.x] = counts[threadIdx.x];
  }
}

// Convert chunk counts into absolute stable output offsets. The final
// histogram and expert prefix remain byte-for-byte identical to the serial
// implementation.
__global__ void BuildGroupChunkOffsetsKernel(
    std::uint32_t* chunk_offsets, std::uint32_t* histogram,
    std::uint32_t* prefix, std::uint32_t experts,
    std::uint32_t chunk_count) {
  const std::uint32_t expert = threadIdx.x;
  if (expert < experts) {
    std::uint32_t count = 0U;
    for (std::uint32_t chunk = 0U; chunk < chunk_count; ++chunk) {
      count += chunk_offsets[static_cast<std::uint64_t>(chunk) * experts +
                             expert];
    }
    histogram[expert] = count;
  }
  __syncthreads();
  if (threadIdx.x == 0U) {
    prefix[0] = 0U;
    for (std::uint32_t index = 0U; index < experts; ++index) {
      prefix[index + 1U] = prefix[index] + histogram[index];
    }
  }
  __syncthreads();
  if (expert < experts) {
    std::uint32_t offset = prefix[expert];
    for (std::uint32_t chunk = 0U; chunk < chunk_count; ++chunk) {
      const std::uint64_t index =
          static_cast<std::uint64_t>(chunk) * experts + expert;
      const std::uint32_t count = chunk_offsets[index];
      chunk_offsets[index] = offset;
      offset += count;
    }
  }
}

// Every chunk is independent after its absolute offsets are known. Counting
// equal expert IDs before the current lane reconstructs the exact original
// token-major/slot-major rank, so no scheduling-dependent atomic cursor can
// perturb the stable permutation.
__global__ void ScatterStableGroupAssignmentsKernel(
    const Gemma4MoePrefillAssignment* assignments,
    const std::uint32_t* chunk_offsets, std::uint32_t* permutation,
    std::uint32_t* inverse, std::uint32_t experts,
    std::uint64_t assignment_count) {
  __shared__ std::uint32_t expert_ids[kGroupingAssignmentsPerChunk];
  const std::uint64_t original =
      static_cast<std::uint64_t>(blockIdx.x) *
          kGroupingAssignmentsPerChunk +
      threadIdx.x;
  const bool valid = original < assignment_count;
  const std::uint32_t expert =
      valid ? assignments[original].expert_id : 0xffffffffU;
  expert_ids[threadIdx.x] = expert;
  __syncthreads();
  if (!valid || expert >= experts) return;
  std::uint32_t rank = 0U;
  for (unsigned prior = 0U; prior < threadIdx.x; ++prior) {
    rank += expert_ids[prior] == expert ? 1U : 0U;
  }
  const std::uint32_t grouped =
      chunk_offsets[static_cast<std::uint64_t>(blockIdx.x) * experts +
                    expert] +
      rank;
  permutation[grouped] = static_cast<std::uint32_t>(original);
  inverse[original] = grouped;
}

// Once routing is complete, router logits are dead for the remainder of the
// layer. Reuse their FP32 storage as compact expert/tile descriptors
// schedule so the SM120 kernels launch only O(assignments / 16 + experts)
// token tiles. histogram[0] temporarily holds the tile count and is restored
// after W2. Each descriptor packs expert_id in the high 16 bits and the first
// grouped assignment in the low 16 bits; tile_count never exceeds 8T, which
// fits the documented T*experts router-logit region because top_k <= experts.
__global__ void BuildExpertTileScheduleKernel(const std::uint32_t* prefix,
                                              std::uint32_t* tile_count,
                                              std::uint32_t* tiles,
                                              std::uint32_t experts) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) return;
  std::uint32_t count = 0U;
  for (std::uint32_t expert = 0; expert < experts; ++expert) {
    for (std::uint32_t grouped = prefix[expert];
         grouped < prefix[expert + 1U]; grouped += 16U) {
      tiles[count++] = (expert << 16U) | grouped;
    }
  }
  tile_count[0] = count;
}

__global__ void RestoreHistogramZeroKernel(std::uint32_t* histogram,
                                           const std::uint32_t* prefix) {
  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    histogram[0] = prefix[1] - prefix[0];
  }
}

__device__ __forceinline__ float LoadExpertDown(const float* expert_down,
                                                std::uint64_t index) {
  return expert_down[index];
}

__device__ __forceinline__ float LoadExpertDown(
    const std::uint16_t* expert_down, std::uint64_t index) {
  return Bf16(expert_down[index]);
}

template <typename ExpertDown>
__global__ void ReduceAssignmentsKernel(
    const ExpertDown* expert_down,
    const Gemma4MoePrefillAssignment* assignments, float* routed_sum,
    std::uint64_t width, std::uint32_t top_k, std::uint64_t tokens) {
  const std::uint64_t column =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t token = blockIdx.y;
  if (column >= width || token >= tokens) return;
  const std::uint64_t assignment_base = token * top_k;
  float sum = 0.0F;
  for (std::uint32_t slot = 0; slot < top_k; ++slot) {
    const std::uint64_t assignment = assignment_base + slot;
    const float weighted =
        RoundBf16(LoadExpertDown(expert_down,
                                 assignment * width + column) *
                   assignments[assignment].weight);
    sum += weighted;
  }
  routed_sum[token * width + column] = RoundBf16(sum);
}

__global__ void CombineBatchKernel(const float* left, const float* right,
                                   float* output, std::uint64_t elements) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) output[index] = RoundBf16(left[index] + right[index]);
}

std::uint64_t Blocks(std::uint64_t elements) {
  return (elements + kThreads - 1U) / kThreads;
}

Status CheckLaunch(const char* label) {
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess ? Status::Ok() : CudaFailure(label, error);
}

}  // namespace

Status LaunchGemma4MoeReduceAssignments(
    const float* expert_down,
    const Gemma4MoePrefillAssignment* assignments, float* routed_sum,
    std::uint64_t width, std::uint32_t top_k, std::uint64_t tokens,
    cudaStream_t stream) {
  if (expert_down == nullptr || assignments == nullptr ||
      routed_sum == nullptr || width == 0U || top_k == 0U || tokens == 0U ||
      tokens > std::numeric_limits<std::uint64_t>::max() / top_k ||
      tokens * top_k >
          std::numeric_limits<std::uint64_t>::max() / width ||
      tokens > static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max()) ||
      Blocks(width) >
          static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max())) {
    return Invalid("M15 float-container expert reduction contract is invalid");
  }
  ReduceAssignmentsKernel<float><<<
      dim3(static_cast<unsigned>(Blocks(width)),
           static_cast<unsigned>(tokens)),
      kThreads, 0, stream>>>(expert_down, assignments, routed_sum, width,
                             top_k, tokens);
  return CheckLaunch("launch M15 float-container expert reduction");
}

Status LaunchGemma4MoeReduceAssignmentsBf16(
    const std::uint16_t* expert_down_bf16,
    const Gemma4MoePrefillAssignment* assignments, float* routed_sum,
    std::uint64_t width, std::uint32_t top_k, std::uint64_t tokens,
    cudaStream_t stream) {
  if (expert_down_bf16 == nullptr || assignments == nullptr ||
      routed_sum == nullptr || width == 0U || top_k == 0U || tokens == 0U ||
      tokens > std::numeric_limits<std::uint64_t>::max() / top_k ||
      tokens * top_k >
          std::numeric_limits<std::uint64_t>::max() / width ||
      tokens > static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max()) ||
      Blocks(width) >
          static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max())) {
    return Invalid("M15 physical-BF16 expert reduction contract is invalid");
  }
  ReduceAssignmentsKernel<std::uint16_t><<<
      dim3(static_cast<unsigned>(Blocks(width)),
           static_cast<unsigned>(tokens)),
      kThreads, 0, stream>>>(expert_down_bf16, assignments, routed_sum, width,
                             top_k, tokens);
  return CheckLaunch("launch M15 physical-BF16 expert reduction");
}

Status LaunchGemma4MoeSm120PrefillLayer(
    const float* hidden, float* output, std::uint64_t tokens,
    const Gemma4MoeReferenceConfig& c,
    const Gemma4MoeReferenceWeights& w,
    const Gemma4MoePrefillWorkspace& x, cudaStream_t stream) {
  if (hidden == nullptr || output == nullptr || hidden == output ||
      tokens == 0U || tokens > 1024U || c.width == 0U ||
      c.width % kSm120KBlock != 0U || c.shared_intermediate == 0U ||
      c.shared_intermediate % kSm120KBlock != 0U ||
      c.expert_intermediate == 0U ||
      c.expert_intermediate % kSm120KBlock != 0U || c.experts == 0U ||
      c.experts > std::numeric_limits<std::uint16_t>::max() ||
      c.top_k != 8U || c.top_k > c.experts || !PositiveFinite(c.epsilon)) {
    return Invalid("M15 grouped MoE geometry is invalid");
  }
  if (w.pre_shared_norm_bf16 == nullptr ||
      w.post_shared_norm_bf16 == nullptr ||
      w.pre_expert_norm_bf16 == nullptr ||
      w.post_expert_norm_bf16 == nullptr ||
      w.post_combined_norm_bf16 == nullptr || w.router_scale_bf16 == nullptr ||
      w.router_projection_bf16 == nullptr ||
      w.per_expert_scale_bf16 == nullptr || w.layer_scalar_bf16 == nullptr ||
      !MatrixValid(w.shared_gate, c.shared_intermediate, c.width) ||
      !MatrixValid(w.shared_up, c.shared_intermediate, c.width) ||
      !MatrixValid(w.shared_down, c.width, c.shared_intermediate) ||
      !MatrixValid(w.expert_gate_up,
                   static_cast<std::uint64_t>(c.experts) * 2U *
                       c.expert_intermediate,
                   c.width) ||
      !MatrixValid(w.expert_down,
                   static_cast<std::uint64_t>(c.experts) * c.width,
                   c.expert_intermediate) ||
      w.shared_gate.activation_global_divisor !=
          w.shared_up.activation_global_divisor) {
    return Invalid("M15 grouped MoE weight contract is invalid");
  }
  const bool float_expert_boundaries =
      x.expert_product != nullptr && x.expert_down != nullptr &&
      x.expert_product_bf16 == nullptr && x.expert_down_bf16 == nullptr;
  const bool physical_expert_boundaries =
      x.expert_product == nullptr && x.expert_down == nullptr &&
      x.expert_product_bf16 != nullptr && x.expert_down_bf16 != nullptr;
  if (x.router_logits == nullptr || x.router_probabilities == nullptr ||
      x.token_hidden == nullptr || x.token_packed == nullptr ||
      x.token_scales == nullptr ||
      (!float_expert_boundaries && !physical_expert_boundaries) ||
      x.expert_product_packed == nullptr ||
      x.expert_product_scales == nullptr ||
      x.shared_product == nullptr || x.shared_product_packed == nullptr ||
      x.shared_product_scales == nullptr || x.shared_output == nullptr ||
      x.reduced_output == nullptr || x.assignments == nullptr ||
      x.histogram == nullptr || x.prefix == nullptr ||
      x.permutation == nullptr || x.inverse_permutation == nullptr) {
    return Invalid("M15 grouped MoE workspace is incomplete");
  }
  const std::uint64_t assignments = tokens * c.top_k;
  if (assignments > 65535U) {
    return Invalid("M15 assignment count exceeds CUDA grid limits");
  }

  Status status = LaunchRmsNormNvfp4ActivationQuantizationBatch(
      hidden, w.pre_shared_norm_bf16, x.token_packed, x.token_scales, tokens,
      c.width, c.epsilon, w.shared_gate.activation_global_divisor, stream);
  if (!status.ok()) return status;
  status = LaunchNvfp4Sm120FusedGateUpBatch(
      x.token_packed, x.token_scales, w.shared_gate.packed_e2m1,
      w.shared_gate.scales_e4m3fn, w.shared_up.packed_e2m1,
      w.shared_up.scales_e4m3fn, nullptr, nullptr, x.shared_product, tokens,
      c.shared_intermediate, c.width,
      w.shared_gate.activation_global_divisor,
      w.shared_gate.weight_global_divisor,
      w.shared_up.activation_global_divisor,
      w.shared_up.weight_global_divisor, stream);
  if (!status.ok()) return status;
  status = LaunchNvfp4ReferenceActivationQuantization(
      x.shared_product, x.shared_product_packed, x.shared_product_scales,
      tokens * c.shared_intermediate,
      w.shared_down.activation_global_divisor, stream);
  if (!status.ok()) return status;
  status = LaunchNvfp4Sm120DirectProjectionBatch(
      x.shared_product_packed, x.shared_product_scales,
      w.shared_down.packed_e2m1, w.shared_down.scales_e4m3fn,
      x.shared_output, tokens, c.width, c.shared_intermediate,
      w.shared_down.activation_global_divisor,
      w.shared_down.weight_global_divisor, stream);
  if (!status.ok()) return status;
  const std::uint64_t shared_output_elements = tokens * c.width;
  RoundBf16BatchKernel<<<
      static_cast<unsigned>(Blocks(shared_output_elements)), kThreads, 0,
      stream>>>(x.shared_output, shared_output_elements);
  status = CheckLaunch("launch M15 shared-output BF16 rounding");
  if (!status.ok()) return status;
  status = LaunchRmsNormBf16(x.shared_output, w.post_shared_norm_bf16,
                             x.reduced_output, tokens, c.width, c.epsilon,
                             stream);
  if (!status.ok()) return status;

  // Fixed-arena lifetime alias: shared_output's shared-branch value is now
  // preserved in reduced_output, so the same storage may hold the router
  // transform until assignments are materialized.
  status = LaunchRmsNormBf16(hidden, nullptr, x.token_hidden, tokens, c.width,
                             c.epsilon, stream);
  if (!status.ok()) return status;
  RouterTransformBatchKernel<<<
      static_cast<unsigned>(Blocks(tokens * c.width)), kThreads, 0, stream>>>(
      x.token_hidden, w.router_scale_bf16, x.shared_output, tokens, c.width);
  status = CheckLaunch("launch M15 router transform");
  if (!status.ok()) return status;
  const unsigned router_blocks =
      (c.experts + kRouterExpertsPerBlock - 1U) /
      kRouterExpertsPerBlock;
  const unsigned router_token_blocks = static_cast<unsigned>(
      (tokens + kRouterTokensPerBlock - 1U) / kRouterTokensPerBlock);
  RouterProjectionBatchCoalescedKernel<<<
      dim3(router_blocks, router_token_blocks), kRouterThreads, 0,
      stream>>>(x.shared_output, w.router_projection_bf16, x.router_logits,
                c.experts, c.width, tokens);
  status = CheckLaunch("launch M15 router projection");
  if (!status.ok()) return status;
  RouterAssignmentsKernel<<<static_cast<unsigned>(tokens), kThreads, 0, stream>>>(
      x.router_logits, w.per_expert_scale_bf16, x.router_probabilities,
      x.assignments, c.experts, c.top_k, tokens, x.routing_finite);
  status = CheckLaunch("launch M15 router assignments");
  if (!status.ok()) return status;
  auto* expert_tile_schedule =
      reinterpret_cast<std::uint32_t*>(x.router_logits);
  if (c.experts <= kMaxParallelGroupingExperts) {
    const unsigned grouping_chunks = static_cast<unsigned>(
        (assignments + kGroupingAssignmentsPerChunk - 1U) /
        kGroupingAssignmentsPerChunk);
    CountGroupAssignmentsByChunkKernel<<<grouping_chunks, kGroupingThreads, 0,
                                         stream>>>(
        x.assignments, expert_tile_schedule, c.experts, assignments);
    status = CheckLaunch("launch M15 chunk assignment counts");
    if (!status.ok()) return status;
    BuildGroupChunkOffsetsKernel<<<1, kThreads, 0, stream>>>(
        expert_tile_schedule, x.histogram, x.prefix, c.experts,
        grouping_chunks);
    status = CheckLaunch("launch M15 group chunk offsets");
    if (!status.ok()) return status;
    ScatterStableGroupAssignmentsKernel<<<grouping_chunks, kGroupingThreads,
                                           0, stream>>>(
        x.assignments, expert_tile_schedule, x.permutation,
        x.inverse_permutation, c.experts, assignments);
  } else {
    StableGroupAssignmentsSerialKernel<<<1, 1, 0, stream>>>(
        x.assignments, x.histogram, x.prefix, x.permutation,
        x.inverse_permutation, c.experts, assignments);
  }
  status = CheckLaunch("launch M15 stable grouping");
  if (!status.ok()) return status;
  BuildExpertTileScheduleKernel<<<1, 1, 0, stream>>>(
      x.prefix, x.histogram, expert_tile_schedule, c.experts);
  status = CheckLaunch("launch M15 expert tile schedule");
  if (!status.ok()) return status;

  status = LaunchRmsNormNvfp4ActivationQuantizationBatch(
      hidden, w.pre_expert_norm_bf16, x.token_packed, x.token_scales, tokens,
      c.width, c.epsilon, w.expert_gate_up.activation_global_divisor, stream);
  if (!status.ok()) return status;
  status = physical_expert_boundaries
               ? LaunchNvfp4Sm120GroupedExpertFusedGateUpBf16(
                     x.token_packed, x.token_scales,
                     w.expert_gate_up.packed_e2m1,
                     w.expert_gate_up.scales_e4m3fn, x.assignments,
                     x.permutation, x.prefix, expert_tile_schedule,
                     x.histogram, x.expert_product_bf16, assignments,
                     c.expert_intermediate, c.width, c.experts,
                     w.expert_gate_up.activation_global_divisor,
                     w.expert_gate_up.weight_global_divisor, stream)
               : LaunchNvfp4Sm120GroupedExpertFusedGateUp(
                     x.token_packed, x.token_scales,
                     w.expert_gate_up.packed_e2m1,
                     w.expert_gate_up.scales_e4m3fn, x.assignments,
                     x.permutation, x.prefix, expert_tile_schedule,
                     x.histogram, x.expert_product, assignments,
                     c.expert_intermediate, c.width, c.experts,
                     w.expert_gate_up.activation_global_divisor,
                     w.expert_gate_up.weight_global_divisor, stream);
  if (!status.ok()) return status;
  status = physical_expert_boundaries
               ? LaunchNvfp4ReferenceActivationQuantizationBf16(
                     x.expert_product_bf16, x.expert_product_packed,
                     x.expert_product_scales,
                     assignments * c.expert_intermediate,
                     w.expert_down.activation_global_divisor, stream)
               : LaunchNvfp4ReferenceActivationQuantization(
                     x.expert_product, x.expert_product_packed,
                     x.expert_product_scales,
                     assignments * c.expert_intermediate,
                     w.expert_down.activation_global_divisor, stream);
  if (!status.ok()) return status;
  status = physical_expert_boundaries
               ? LaunchNvfp4Sm120GroupedExpertDownBf16(
                     x.expert_product_packed, x.expert_product_scales,
                     w.expert_down.packed_e2m1,
                     w.expert_down.scales_e4m3fn, x.assignments,
                     x.permutation, x.prefix, expert_tile_schedule,
                     x.histogram, x.expert_down_bf16, assignments, c.width,
                     c.expert_intermediate, c.experts,
                     w.expert_down.activation_global_divisor,
                     w.expert_down.weight_global_divisor, stream)
               : LaunchNvfp4Sm120GroupedExpertDown(
                     x.expert_product_packed, x.expert_product_scales,
                     w.expert_down.packed_e2m1,
                     w.expert_down.scales_e4m3fn, x.assignments,
                     x.permutation, x.prefix, expert_tile_schedule,
                     x.histogram, x.expert_down, assignments, c.width,
                     c.expert_intermediate, c.experts,
                     w.expert_down.activation_global_divisor,
                     w.expert_down.weight_global_divisor, stream);
  if (!status.ok()) return status;
  RestoreHistogramZeroKernel<<<1, 1, 0, stream>>>(x.histogram, x.prefix);
  status = CheckLaunch("restore M15 expert-zero histogram");
  if (!status.ok()) return status;
  status = physical_expert_boundaries
               ? LaunchGemma4MoeReduceAssignmentsBf16(
                     x.expert_down_bf16, x.assignments, x.token_hidden,
                     c.width, c.top_k, tokens, stream)
               : LaunchGemma4MoeReduceAssignments(
                     x.expert_down, x.assignments, x.token_hidden, c.width,
                     c.top_k, tokens, stream);
  if (!status.ok()) return status;
  // Assignments and expert_down no longer consume the router transform, so
  // shared_output is reused once more for post-expert normalization.
  status = LaunchRmsNormBf16(x.token_hidden, w.post_expert_norm_bf16,
                             x.shared_output, tokens, c.width, c.epsilon,
                             stream);
  if (!status.ok()) return status;
  CombineBatchKernel<<<static_cast<unsigned>(Blocks(tokens * c.width)),
                       kThreads, 0, stream>>>(
      x.reduced_output, x.shared_output, x.token_hidden, tokens * c.width);
  status = CheckLaunch("launch M15 shared/routed combination");
  if (!status.ok()) return status;
  return LaunchRmsNormResidualBf16(
      x.token_hidden, w.post_combined_norm_bf16, hidden, nullptr, output,
      tokens, c.width, c.epsilon, w.layer_scalar_bf16, stream);
}

}  // namespace gem16::internal
