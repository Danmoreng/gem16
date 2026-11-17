"""Offline fixed-T3 vs packed-T3 replay, preserving 12B FP8 K order."""
from pathlib import Path
import argparse,hashlib,json
root=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('destination',type=Path);a=p.parse_args()
source=root/'src/cuda/fp8/sm120.cu';header=root/'src/cuda/fp8/detail/decode_t3_rows.cuh';text=source.read_text()
text+=r'''
#include "fp8_projection_lab_abi.h"
extern "C" cudaError_t gem16_12b_fp8(unsigned candidate,
 const std::uint8_t*a,const float*s,Fp8LabBinding q,Fp8LabBinding k,Fp8LabBinding v,
 unsigned count,std::uint64_t width,dim3 grid,dim3 block,cudaStream_t stream){
 using namespace gem16::internal;
 const Fp8MatrixBinding bq{q.weight,q.weight_scales,q.output,q.rows};
 const Fp8MatrixBinding bk{k.weight,k.weight_scales,k.output,k.rows};
 const Fp8MatrixBinding bv{v.weight,v.weight_scales,v.output,v.rows};
 if(candidate==0U)Sm120DirectProjectionFixedBatchKernel<3U><<<grid,block,3U*width,stream>>>(a,s,bq,bk,bv,count,width);
 else if(candidate==1U)Sm120DirectProjectionPackedT3Kernel<32U><<<grid,block,3U*(width+32U),stream>>>(a,s,bq,bk,bv,count,width);
 else return cudaErrorInvalidValue;
 return cudaGetLastError();
}
'''
with a.destination.open('x') as f:f.write(text)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a.destination.with_suffix('.provenance.json').write_text(json.dumps({'inputs':{str(p.relative_to(root)):sha(p) for p in (source,header,Path(__file__))},'output_sha256':sha(a.destination),'transformation':'Append exact fixed-T3 / packed-T3 wrappers; identical grid, K order, row layout, weights and FP32 boundaries.'},indent=2)+'\n')
