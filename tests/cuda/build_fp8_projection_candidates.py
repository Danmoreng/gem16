"""Offline FP8 M1/T3 projection candidates over trusted immutable project source."""
from pathlib import Path
import argparse
import hashlib
import json

root = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("destination", type=Path)
parser.add_argument("--source", type=Path, default=root / "src/cuda/fp8/sm120.cu")
options = parser.parse_args()
source = options.source.resolve(strict=True)
header = root / "tests/cuda/fp8_projection_candidates.cuh"
abi = root / "tests/cuda/fp8_projection_lab_abi.h"
text = source.read_text()
marker = "}  // namespace\n\nStatus LaunchFp8Sm120DirectProjection("
assert text.count(marker) == 1
text = text.replace(marker, '#include "fp8_projection_candidates.cuh"\n\n' + marker)
text += r'''
#include "fp8_projection_lab_abi.h"
namespace {
gem16::internal::Fp8MatrixBinding NativeBinding(Fp8LabBinding b) {
  return {b.weight,b.weight_scales,b.output,b.rows};
}
}
extern "C" cudaError_t gem16_lab_launch_fp8_qkv_pack3(
    const std::uint8_t* a,const float* scale,Fp8LabBinding q,Fp8LabBinding k,
    Fp8LabBinding v,unsigned count,std::uint64_t width,unsigned variant,
    dim3 grid,dim3 block,cudaStream_t stream) {
  using namespace gem16::internal;
  if(variant==0U) Sm120QkvPackedT3CandidateKernel<0U><<<grid,block,3U*width,stream>>>(
      a,scale,NativeBinding(q),NativeBinding(k),NativeBinding(v),count,width);
  else if(variant==1U) Sm120QkvPackedT3CandidateKernel<32U><<<grid,block,3U*(width+32U),stream>>>(
      a,scale,NativeBinding(q),NativeBinding(k),NativeBinding(v),count,width);
  else return cudaErrorInvalidValue;
  return cudaGetLastError();
}
extern "C" cudaError_t gem16_lab_launch_fp8_o_pack3(
    const std::uint8_t* a,const float* scale,Fp8LabBinding w,std::uint64_t width,
    dim3 grid,dim3 block,cudaStream_t stream) {
  using namespace gem16::internal;
  Sm120OPackedT3CandidateKernel<<<grid,block,0U,stream>>>(a,scale,w.weight,w.weight_scales,w.output,w.rows,width);
  return cudaGetLastError();
}
extern "C" cudaError_t gem16_lab_launch_fp8_qkv_m1_pipeline(
    const std::uint8_t* a,const float* scale,Fp8LabBinding q,Fp8LabBinding k,
    Fp8LabBinding v,unsigned count,std::uint64_t width,unsigned local,
    dim3 grid,dim3 block,cudaStream_t stream) {
  using namespace gem16::internal;
  if(local==1U) Sm120QkvM1PipelineCandidateKernel<true><<<grid,block,0U,stream>>>(
      a,scale,NativeBinding(q),NativeBinding(k),NativeBinding(v),count,width);
  else if(local==0U) Sm120QkvM1PipelineCandidateKernel<false><<<grid,block,0U,stream>>>(
      a,scale,NativeBinding(q),NativeBinding(k),NativeBinding(v),count,width);
  else return cudaErrorInvalidValue;
  return cudaGetLastError();
}
extern "C" cudaError_t gem16_lab_launch_fp8_o_m1_pipeline(
    const std::uint8_t* a,const float* scale,Fp8LabBinding w,std::uint64_t width,
    dim3 grid,dim3 block,cudaStream_t stream) {
  using namespace gem16::internal;
  Sm120OM1PipelineCandidateKernel<<<grid,block,0U,stream>>>(a,scale,w.weight,w.weight_scales,w.output,w.rows,width);
  return cudaGetLastError();
}
extern "C" cudaError_t gem16_lab_launch_fp8_qkv_t3_parent(
    const std::uint8_t* a,const float* scale,Fp8LabBinding q,Fp8LabBinding k,
    Fp8LabBinding v,unsigned count,std::uint64_t width,unsigned,
    dim3 grid,dim3 block,cudaStream_t stream) {
  using namespace gem16::internal;
  Sm120DirectProjectionFixedBatchKernel<3U><<<grid,block,3U*width,stream>>>(
      a,scale,NativeBinding(q),NativeBinding(k),NativeBinding(v),count,width);
  return cudaGetLastError();
}
extern "C" cudaError_t gem16_lab_launch_fp8_o_t3_parent(
    const std::uint8_t* a,const float* scale,Fp8LabBinding w,std::uint64_t width,
    dim3 grid,dim3 block,cudaStream_t stream) {
  using namespace gem16::internal;
  Sm120DirectProjectionSplitK2FixedT3Kernel<<<grid,block,0U,stream>>>(a,scale,w.weight,w.weight_scales,w.output,w.rows,width);
  return cudaGetLastError();
}
'''
assert not options.destination.exists()
options.destination.write_text(text)
options.destination.with_suffix(".provenance.json").write_text(json.dumps({
    "diagnostic_only": True, "production_source_modified": False,
    "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "candidate_header": str(header), "candidate_header_sha256": hashlib.sha256(header.read_bytes()).hexdigest(),
    "abi_header_sha256": hashlib.sha256(abi.read_bytes()).hexdigest(),
    "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "transform": "append native R3 M16 row packing and bounded M1 two-entry register pipeline; explicitly exercise only chosen wrappers",
    "output_sha256": hashlib.sha256(options.destination.read_bytes()).hexdigest(),
}, indent=2) + "\n")
