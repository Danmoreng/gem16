"""Offline fixed-D2 local/global softmax derivative; production stays untouched."""
from pathlib import Path
import argparse
import hashlib
import json

root = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("destination", type=Path)
parser.add_argument("--source", type=Path, default=root / "src/cuda/attention/decode_sm120.cu",
                    help="trusted immutable parent CUDA source; use the recorded snapshot after promotion")
parser.add_argument("--synchronized-reference", action="store_true")
options = parser.parse_args()
source = options.source.resolve(strict=True)
destination = options.destination
synchronized_reference = options.synchronized_reference
text = source.read_text()
start = text.index("void SplitOnlineDecodeAttentionFp8GlobalKvh2FixedRowsKernel(")
end = text.index("template <int kRows, bool kVectorized>", start)
original = text[start:end]
begin_norm = original.index("#pragma unroll\n  for (int row = 0; row < kRows; ++row) {\n    for (int local_head")
end_norm = original.index("  float accumulators[kRows][16] = {};", begin_norm)
replacement = r'''
  // Each warp owns the same query head as in QK/PV. Preserve the old
  // two-level block tree exactly: four active 32-token chunks, four empty
  // chunks, then the full 32-lane shuffle-down tree over their lane-0 values.
  // Do not sum four probabilities per lane before reducing (different tree).
#pragma unroll
  for (int row = 0; row < kRows; ++row) {
    float chunk_maximum[4];
#pragma unroll
    for (int chunk = 0; chunk < 4; ++chunk) {
      const int token = chunk * 32 + lane;
      float value = token < split_tokens[row]
          ? fmaxf(-CUDART_INF_F, scores[(row * kQueriesPerKv + warp) *
                                      kDecodeGlobalChunk + token])
          : -CUDART_INF_F;
      for (int offset = 16; offset > 0; offset >>= 1)
        value = fmaxf(value, __shfl_down_sync(kFullWarpMask, value, offset));
      chunk_maximum[chunk] = __shfl_sync(kFullWarpMask, value, 0);
    }
    float maximum = lane == 0 ? chunk_maximum[0] :
                    lane == 1 ? chunk_maximum[1] :
                    lane == 2 ? chunk_maximum[2] :
                    lane == 3 ? chunk_maximum[3] : -CUDART_INF_F;
    for (int offset = 16; offset > 0; offset >>= 1)
      maximum = fmaxf(maximum, __shfl_down_sync(kFullWarpMask, maximum, offset));
    maximum = __shfl_sync(kFullWarpMask, maximum, 0);
    float chunk_sum[4];
#pragma unroll
    for (int chunk = 0; chunk < 4; ++chunk) {
      const int token = chunk * 32 + lane;
      float value = 0.0F;
      if (token < split_tokens[row]) {
        const int index = (row * kQueriesPerKv + warp) * kDecodeGlobalChunk + token;
        const float probability = expf(scores[index] - maximum);
        scores[index] = probability;
        value += probability;
      }
      for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(kFullWarpMask, value, offset);
      chunk_sum[chunk] = __shfl_sync(kFullWarpMask, value, 0);
    }
    float denominator = lane == 0 ? chunk_sum[0] :
                        lane == 1 ? chunk_sum[1] :
                        lane == 2 ? chunk_sum[2] :
                        lane == 3 ? chunk_sum[3] : 0.0F;
    for (int offset = 16; offset > 0; offset >>= 1)
      denominator += __shfl_down_sync(kFullWarpMask, denominator, offset);
    denominator = __shfl_sync(kFullWarpMask, denominator, 0);
    if (lane == 0) {
      inverse_sum_shared[row][warp] = denominator > 0.0F ? __frcp_rn(denominator) : 0.0F;
      partial_lse[(static_cast<std::uint64_t>(row) * max_splits + split) *
                      kDecodeQueryHeads + query_head] = maximum + logf(denominator);
    }
  }
  __syncwarp(kFullWarpMask);

'''
candidate = original[:begin_norm] + replacement + original[end_norm:]
candidate = candidate.replace("SplitOnlineDecodeAttentionFp8GlobalKvh2FixedRowsKernel",
                              "SplitOnlineDecodeAttentionFp8GlobalKvh2WarpNormCandidateKernel")
assert candidate.count("  __shared__ float reduction[kDecodeWarps];") == 1
candidate = candidate.replace("  __shared__ float reduction[kDecodeWarps];\n", "")
text = (text[:start] + original + "template <int kRows>\n"
        "__launch_bounds__(kDecodeThreads, 1) __global__\n" + candidate + text[end:])
local_start = text.index("void SplitOnlineDecodeAttentionFp8LocalFixedSharedKvKernel(")
local_end = text.index("__launch_bounds__(kDecodeThreads, 1) __global__\nvoid MergeOnlineDecodeAttentionFp8LocalFixedKernel", local_start)
local_original = text[local_start:local_end]
local_begin_norm = local_original.index("  float local_maximum[kReductionItems];")
local_end_norm = local_original.index("  stage(value_cache, speculative_value);", local_begin_norm)
local_norm = replacement.replace(
    "#pragma unroll\n  for (int row = 0; row < kRows; ++row) {",
    "  if (warp < kReductionItems) {\n    const int row = warp / kDecodeLocalGroup;\n    const int group = warp % kDecodeLocalGroup;")
local_norm = local_norm.replace("(row * kQueriesPerKv + warp)", "warp")
local_norm = local_norm.replace("kDecodeGlobalChunk", "kDecodeLocalChunk")
local_norm = local_norm.replace("inverse_sum_shared[row][warp]", "inverse_sum[row][group]")
local_norm = local_norm.replace("kDecodeQueryHeads + query_head] = maximum + logf(denominator)",
                                "kDecodeQueryHeads + query_head_base + group] = split_tokens[row] != 0 ? maximum + logf(denominator) : -CUDART_INF_F")
local_norm = local_norm.replace("__syncwarp(kFullWarpMask)", "__syncthreads()")
local_probability_before = """      float value = 0.0F;
      if (token < split_tokens[row]) {
        const int index = warp * kDecodeLocalChunk + token;
        const float probability = expf(scores[index] - maximum);
        scores[index] = probability;
        value += probability;
      }"""
local_probability_after = """      const int index = warp * kDecodeLocalChunk + token;
      const float probability = token < split_tokens[row]
          ? expf(scores[index] - maximum) : 0.0F;
      // Unlike global PV, local PV reads the entire fixed 128-token chunk.
      // Its inactive score slots must be explicitly zeroed for short contexts.
      scores[index] = probability;
      float value = 0.0F;
      value += probability;"""
assert local_norm.count(local_probability_before) == 1
local_norm = local_norm.replace(local_probability_before, local_probability_after)
local_norm = local_norm.replace("Each warp owns the same query head as in QK/PV.",
                                "Each active warp owns one row/head normalization item.")
local_candidate = local_original[:local_begin_norm] + local_norm + local_original[local_end_norm:]
local_candidate = local_candidate.replace("SplitOnlineDecodeAttentionFp8LocalFixedSharedKvKernel",
                                          "SplitOnlineDecodeAttentionFp8LocalWarpNormCandidateKernel")
local_candidate = local_candidate.replace("  __shared__ float reduction[kReductionItems][kDecodeWarps];\n", "")
text = (text[:local_start] + local_original + "template <int kRows>\n"
        "__launch_bounds__(kDecodeThreads, 1) __global__\n" + local_candidate + text[local_end:])
text += r'''
extern "C" cudaError_t gem16_lab_launch_t3_global_warp_norm(
    const float* query, const std::uint8_t* keys, const std::uint8_t* values,
    const std::uint16_t* key_scale, const std::uint16_t* value_scale,
    float* partial, float* lse, const gem16::internal::DecodeControl* controls,
    int splits, dim3 grid, dim3 block, cudaStream_t stream) {
  using namespace gem16::internal;
  SplitOnlineDecodeAttentionFp8GlobalKvh2WarpNormCandidateKernel<3><<<grid, block, 0U, stream>>>(
      query, keys, values, key_scale, value_scale, partial, lse, controls, splits);
  return cudaGetLastError();
}
extern "C" cudaError_t gem16_lab_launch_t3_local_warp_norm(
    const float* query, const std::uint8_t* keys, const std::uint8_t* values,
    const std::uint8_t* speculative_keys, const std::uint8_t* speculative_values,
    const std::uint16_t* key_scale, const std::uint16_t* value_scale,
    float* partial, float* lse, const gem16::internal::DecodeControl* controls,
    int splits, dim3 grid, dim3 block, cudaStream_t stream) {
  using namespace gem16::internal;
  SplitOnlineDecodeAttentionFp8LocalWarpNormCandidateKernel<3><<<grid, block, 0U, stream>>>(
      query, keys, values, speculative_keys, speculative_values, key_scale, value_scale,
      partial, lse, controls, splits);
  return cudaGetLastError();
}
extern "C" cudaError_t gem16_lab_launch_t3_global_parent(
    const float* query, const std::uint8_t* keys, const std::uint8_t* values,
    const std::uint16_t* key_scale, const std::uint16_t* value_scale,
    float* partial, float* lse, const gem16::internal::DecodeControl* controls,
    int splits, dim3 grid, dim3 block, cudaStream_t stream) {
  using namespace gem16::internal;
  SplitOnlineDecodeAttentionFp8GlobalKvh2FixedRowsKernel<3><<<grid, block, 0U, stream>>>(
      query, keys, values, key_scale, value_scale, partial, lse, controls, splits);
  return cudaGetLastError();
}
extern "C" cudaError_t gem16_lab_launch_t3_local_parent(
    const float* query, const std::uint8_t* keys, const std::uint8_t* values,
    const std::uint8_t* speculative_keys, const std::uint8_t* speculative_values,
    const std::uint16_t* key_scale, const std::uint16_t* value_scale,
    float* partial, float* lse, const gem16::internal::DecodeControl* controls,
    int splits, dim3 grid, dim3 block, cudaStream_t stream) {
  using namespace gem16::internal;
  SplitOnlineDecodeAttentionFp8LocalFixedSharedKvKernel<3><<<grid, block, 0U, stream>>>(
      query, keys, values, speculative_keys, speculative_values, key_scale, value_scale,
      partial, lse, controls, splits);
  return cudaGetLastError();
}
'''
if synchronized_reference:
    # Diagnostic reference only: protect reads of reduction[item][0] before
    # the sum phase reuses that slot. No arithmetic, input or MMA change.
    reference_start = text.index("void SplitOnlineDecodeAttentionFp8LocalFixedSharedKvKernel(")
    reference_end = text.index("void SplitOnlineDecodeAttentionFp8LocalWarpNormCandidateKernel(", reference_start)
    reference = text[reference_start:reference_end]
    needle = "  float local_sum[kReductionItems] = {};"
    assert reference.count(needle) == 1
    reference = reference.replace(needle, "  __syncthreads();  // Diagnostic reference: finish maximum reads before reuse.\n" + needle)
    text = text[:reference_start] + reference + text[reference_end:]
assert not destination.exists()
destination.write_text(text)
destination.with_suffix(".provenance.json").write_text(json.dumps({
    "diagnostic_only": True, "source": str(source),
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "transform": "local/global fixed-D2 warp-owned normalization, exact original 32-lane/two-level sum tree; no KV/QK/PV or merge change",
    "output_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    "production_source_modified": False,
    "diagnostic_original_reference_sync_patch": synchronized_reference,
}, indent=2) + "\n")
