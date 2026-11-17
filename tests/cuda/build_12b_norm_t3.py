"""Offline three-row post-attention fusion, preserving each row's exact tree."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('destination', type=Path)
a = p.parse_args()
a.destination.mkdir(parents=True, exist_ok=True)
base = a.destination / 'base.cu'
subprocess.run([sys.executable, str(ROOT / 'tests/cuda/build_12b_norm_candidate.py'), str(base)], check=True)
s = base.read_text()
header = (ROOT / 'src/cuda/nvfp4/detail/twelve_b_post_attention.cuh').read_text()
begin = header.index('__global__ void TwelveBPostAttentionQuantizeKernel(')
k = header[begin:header.index('\n}', begin) + 2]
k = k.replace('TwelveBPostAttentionQuantizeKernel', 'TwelveBPostAttentionQuantizeT3Kernel', 1)
marker = 'float global_divisor){'
assert k.count(marker) == 1
k = k.replace(marker, marker + '''
 const std::uint64_t row=blockIdx.x;
 projection+=row*3840U;residual+=row*3840U;
 if(post_output!=nullptr)post_output+=row*3840U;
 hidden_output+=row*3840U;packed_e2m1+=row*1920U;
 block_scales_e4m3fn+=row*240U;
''')
s += '\nnamespace gem16::internal { namespace {\n' + k + '\n}}\n' + r'''
extern "C" cudaError_t gem16_12b_norm_t3(unsigned candidate,
 float*projection,const std::uint16_t*post_weight,const float*residual,
 float*post_output,float*hidden_output,const std::uint16_t*pre_weight,
 std::uint8_t*packed,std::uint8_t*scales,float divisor,cudaStream_t stream){
 using namespace gem16::internal;
 if(candidate==0U){
  for(unsigned row=0;row<3;++row)TwelveBRoundProjectionKernel<<<15U,256U,0U,stream>>>(projection+row*3840U);
  TwelveBParentNormKernel<float><<<3U,256U,0U,stream>>>(projection,post_weight,residual,post_output,hidden_output,3840U,1.0e-6F,nullptr);
  RmsNormQuantizeActivationKernel<float><<<3U,256U,0U,stream>>>(hidden_output,pre_weight,nullptr,packed,scales,3840U,1.0e-6F,divisor);
 }else if(candidate==1U){
  TwelveBPostAttentionQuantizeT3Kernel<<<3U,256U,0U,stream>>>(projection,post_weight,residual,post_output,hidden_output,pre_weight,packed,scales,divisor);
 }else return cudaErrorInvalidValue;
 return cudaGetLastError();
}
'''
# Exact production rounding uses a single 45-CTA launch, not three launches.
rounder = r'''
namespace gem16::internal { namespace {
__global__ void TwelveBRoundProjectionT3Kernel(float*p){const unsigned i=blockIdx.x*256U+threadIdx.x;if(i<3U*3840U)p[i]=static_cast<float>(__float2bfloat16_rn(p[i]));}
}}
'''
at = s.index('extern "C" cudaError_t gem16_12b_norm_t3(')
s = s[:at] + rounder + s[at:]
s = s.replace('for(unsigned row=0;row<3;++row)TwelveBRoundProjectionKernel<<<15U,256U,0U,stream>>>(projection+row*3840U);',
              'TwelveBRoundProjectionT3Kernel<<<45U,256U,0U,stream>>>(projection);')
out = a.destination / 'candidate.cu'
with out.open('x') as f: f.write(s)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
(a.destination / 'derivatives.json').write_text(json.dumps({
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), ROOT / 'src/cuda/nvfp4/detail/twelve_b_post_attention.cuh', ROOT / 'tests/cuda/build_12b_norm_candidate.py']},
    'outputs': {out.name: sha(out), base.name: sha(base)},
    'scope': 'T3 independent CTAs; same T1 256-thread RMS trees/rounding/quantization; production unchanged'}, indent=2) + '\n')
