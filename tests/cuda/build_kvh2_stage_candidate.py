"""Offline test-only derivative; only its explicit KVH2 wrapper is exercised."""
from pathlib import Path
import hashlib
import json
import sys

root = Path(__file__).resolve().parents[2]
source = root / "src/cuda/attention/decode_sm120.cu"
destination = Path(sys.argv[1])
text = source.read_text()
before = "constexpr int kDecodeGlobalGqaTileTokens = 16;"
assert text.count(before) == 1
text = text.replace(before, "constexpr int kDecodeGlobalGqaTileTokens = 32;")
text += r'''
// Diagnostic entry point only. Other emitted functions are not exercised.
extern "C" cudaError_t gem16_lab_launch_kvh2_stage32(
    const float* query, const std::uint8_t* keys, const std::uint8_t* values,
    const std::uint16_t* key_scale, const std::uint16_t* value_scale,
    float* partial, float* lse, float* output,
    const gem16::internal::DecodeControl* control, int splits,
    dim3 grid, dim3 block, cudaStream_t stream) {
  using namespace gem16::internal;
  SplitOnlineDecodeAttentionFp8GlobalKvh2Kernel<false><<<grid, block, 0U, stream>>>(
      query, keys, values, key_scale, value_scale, partial, lse, output,
      control, 0U, nullptr, splits);
  return cudaGetLastError();
}
'''
assert not destination.exists()
destination.write_text(text)
destination.with_suffix(".provenance.json").write_text(json.dumps({
    "diagnostic_only": True,
    "source": str(source),
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "transform": "16 to 32 staging constant in a standalone derivative; exercise only the explicit ordinary KVH2 wrapper",
    "output_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    "production_source_modified": False,
}, indent=2) + "\n")
