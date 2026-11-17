"""Exact-arithmetic12B T1 scheduling derivatives; no weight/precision transform."""
from pathlib import Path
import argparse,hashlib,json
R=Path(__file__).resolve().parents[2]

def offline_source(path):
 s=path.read_text()
 for header in ['cuda/fp8/detail/twelve_b_ordinary_o.cuh','cuda/attention/detail/twelve_b_ordinary_global_gqa4.cuh']:
  s=s.replace('#include "'+header+'"\n','')
 for function in ['Status LaunchTwelveBOrdinaryFp8OProjection(', 'Status LaunchTwelveBOrdinaryGlobalAttentionFp8Sm120(']:
  if function in s:
   start=s.index(function);end=matching(s,s.index('{',start))+1;s=s[:start]+s[end:]
 return s

def matching(s,start):
 depth=0
 for i in range(start,len(s)):
  depth+=(s[i]=='{')-(s[i]=='}')
  if depth==0:return i
 raise ValueError('unbalanced CUDA body')

def fp8_candidate():
 s=(R/'src/cuda/fp8/sm120.cu').read_text();start=s.index('template <bool kRoundBf16>\n__global__ void Sm120DirectProjectionKernel(');brace=s.index('{',start);end=matching(s,brace)+1;k=s[start:end]
 k=k.replace('template <bool kRoundBf16>','template <int kOutputRowsPerWarp>')
 k=k.replace('Sm120DirectProjectionKernel','TwelveBOrdinaryOProjectionKernel',1)
 # Preserve exactly the row's B fragment and ordered K32 MMA chain; unused
 # fragment rows are zero-filled, not re-packed into another weight layout.
 k=k.replace('kRowsPerWarp','kOutputRowsPerWarp')
 k=k.replace('if (source_row < rows) {','if (source_row_in_tile < kOutputRowsPerWarp && source_row < rows) {')
 k=k.replace('if (lane < 4U) {','if (lane < kOutputRowsPerWarp / 2U) {')
 k=k.replace('kRoundBf16\n              ? static_cast<float>(__float2bfloat16_rn(value))\n              : value','value')
 assert 'kRoundBf16' not in k and k.count('mma.sync')==1
 return k

def fp8_pipeline_candidate():
 import re
 s=(R/'src/cuda/fp8/sm120.cu').read_text();start=s.index('template <bool kRoundBf16>\n__global__ void Sm120DirectProjectionKernel(');end=matching(s,s.index('{',start))+1;k=s[start:end]
 k=k.replace('template <bool kRoundBf16>','template <unsigned kDepth>').replace('Sm120DirectProjectionKernel','TwelveBOrdinaryOPipelineKernel',1)
 k=k.replace('kRoundBf16\n              ? static_cast<float>(__float2bfloat16_rn(value))\n              : value','value')
 loop='  for (std::uint64_t k_block = 0; k_block < k_blocks; ++k_block) {';begin=k.index(loop);body=begin+len(loop);end=matching(k,k.index('{',begin));middle=k.index('    float next0',body)
 load=k[body:middle];mma=k[middle:end]
 for name in ['a_first','a_second']:
  load=load.replace('const std::uint32_t '+name+' =',name+' =')
 for name in ['b_first','b_second']:
  load=load.replace('std::uint32_t '+name+' =',name+' =')
 for name in ['a_first','a_second','b_first','b_second']:
  load=re.sub(r'\b'+name+r'\b',name+'[pipe]',load);mma=re.sub(r'\b'+name+r'\b',name+'[pipe]',mma)
 load=load.replace('k_block *','(k_base + pipe) *')
 replacement='''  static_assert(kDepth==4U || kDepth==8U);
  for (std::uint64_t k_base=0; k_base<k_blocks; k_base+=kDepth) {
    std::uint32_t a_first[kDepth],a_second[kDepth],b_first[kDepth],b_second[kDepth];
#pragma unroll
    for (unsigned pipe=0;pipe<kDepth;++pipe) {'''+load+'''
    }
#pragma unroll
    for (unsigned pipe=0;pipe<kDepth;++pipe) {
'''+mma+'''
    }
  }'''
 k=k[:begin]+replacement+k[end+1:];assert 'k_block *' not in k and k.count('mma.sync')==1 and 'kRoundBf16' not in k
 return k

def attention_candidate():
 s=(R/'src/cuda/attention/decode_sm120.cu').read_text();name=s.index('void SplitOnlineDecodeAttentionFp8GlobalGqaKernel(');start=s.rfind('template <',0,name);brace=s.index('{',name);end=matching(s,brace)+1;k=s[start:end]
 k=k.replace('SplitOnlineDecodeAttentionFp8GlobalGqaKernel','TwelveBOrdinaryGlobalGqa4Kernel',1)
 k=k.replace('  static_assert(kDecodeWarps', '  static_assert(!kBatch && kVectorized);\n  constexpr int kHeadsPerGroup=4;\n  constexpr int kActiveWarps=kHeadsPerGroup/kDecodeGlobalGqaHeadsPerWarp;\n  const int head_group_base=static_cast<int>(blockIdx.y)*kHeadsPerGroup;\n  static_assert(kDecodeWarps',1)
 k=k.replace('scores[kDecodeQueryHeads *','scores[kHeadsPerGroup *').replace('inverse_sum_shared[kDecodeQueryHeads]','inverse_sum_shared[kHeadsPerGroup]')
 k=k.replace('query[(query_head_base + head) *','query[(head_group_base + query_head_base + head) *')
 # Cooperatively stage K/V and normalize with all256 threads. Only the
 # two active warps perform the four heads' unchanged dot/context chains.
 begins=[];at=0
 marker='    for (int token = 0; token < tile_tokens; ++token) {'
 while (at:=k.find(marker,at))!=-1:
  begins.append(at);at+=len(marker)
 assert len(begins)==2
 for at in reversed(begins):
  end=matching(k,k.index('{',at))+1;k=k[:end]+'\n    }'+k[end:];k=k[:at]+'    if (warp < kActiveWarps) {\n'+k[at:]
 at=k.index('#pragma unroll\n  for (int head = 0;');end=k.index('\n\n  for (int tile_start',at)
 k=k[:at]+'  if (warp < kActiveWarps) {\n'+k[at:end]+'\n  }'+k[end:]
 k=k.replace('for (int query_head = 0; query_head < kDecodeQueryHeads;', 'for (int query_head = 0; query_head < kHeadsPerGroup;')
 # Shared scores/inverse use local heads; only the external head indices shift.
 k=k.replace('             query_head] = maximum + logf(denominator);','             head_group_base + query_head] = maximum + logf(denominator);')
 k=k.replace('row) * kDecodeQueryHeads + query_head)', 'row) * kDecodeQueryHeads + head_group_base + query_head)')
 k=k.replace('              query_head) *\n                 kDecodeGlobalHeadDimension', '              head_group_base + query_head) *\n                 kDecodeGlobalHeadDimension')
 epilogue='#pragma unroll\n  for (int head = 0; head < kDecodeGlobalGqaHeadsPerWarp; ++head) {\n    const int query_head = query_head_base + head;'
 assert k.count(epilogue)==1;k=k.replace(epilogue,'  if (warp >= kActiveWarps) return;\n'+epilogue)
 assert k.count('fmaf(')==4 and k.count('head_group_base + query_head')==4
 return k

def emit(directory, pipeline=False):
 directory.mkdir(parents=True,exist_ok=True);o=fp8_pipeline_candidate() if pipeline else fp8_candidate();g=attention_candidate()
 for name,text in [('ordinary_o.cuh',o),('ordinary_global_gqa4.cuh',g)]:
  (directory/name).write_text('// Generated by tests/cuda/build_12b_ordinary_o_global.py; do not edit.\n'+'\n'.join(line.rstrip() for line in text.splitlines())+'\n')
 fp=offline_source(R/'src/cuda/fp8/sm120.cu')+'\nnamespace gem16::internal { namespace {\n'+o+'\n}}\n'+r'''
extern "C" cudaError_t gem16_12b_o(unsigned variant,const std::uint8_t*a,
 const float*s,Fp8LabBinding w,std::uint64_t width,cudaStream_t stream){
 using namespace gem16::internal;
 if(variant>2||w.rows!=3840U||(width!=4096U&&width!=8192U))return cudaErrorInvalidValue;
 const Fp8MatrixBinding b{w.weight,w.weight_scales,w.output,w.rows},empty{};
 if(variant==0)Sm120DirectProjectionKernel<false><<<120U,128U,0U,stream>>>(a,s,b,empty,empty,1U,1U,width);
 else if(variant==1)TwelveBOrdinaryOProjectionKernel<4><<<240U,128U,0U,stream>>>(a,s,b,empty,empty,1U,1U,width);
 else TwelveBOrdinaryOProjectionKernel<2><<<480U,128U,0U,stream>>>(a,s,b,empty,empty,1U,1U,width);
 return cudaGetLastError();
}
'''
 fp=fp.replace('extern "C" cudaError_t gem16_12b_o','\n#include "fp8_projection_lab_abi.h"\nextern "C" cudaError_t gem16_12b_o')
 if pipeline:
  fp=fp.replace('TwelveBOrdinaryOProjectionKernel<4><<<240U','TwelveBOrdinaryOPipelineKernel<4><<<120U').replace('TwelveBOrdinaryOProjectionKernel<2><<<480U','TwelveBOrdinaryOPipelineKernel<8><<<120U')
 (directory/'fp8.cu').write_text(fp)
 att=offline_source(R/'src/cuda/attention/decode_sm120.cu')+'\nnamespace gem16::internal { namespace {\n'+g+'\n}}\n'+r'''
extern "C" cudaError_t gem16_12b_global_t1(unsigned variant,
 const float*q,const std::uint8_t*k,const std::uint8_t*v,const std::uint16_t*ks,
 const std::uint16_t*vs,float*p,float*l,const gem16::internal::DecodeControl*c,
 std::uint64_t capacity,unsigned splits,cudaStream_t stream){
 using namespace gem16::internal;
 if(variant>1||capacity<16384U||capacity>17519U||splits!=(capacity+511U)/512U)return cudaErrorInvalidValue;
 if(variant==0)SplitOnlineDecodeAttentionFp8GlobalGqaKernel<false><<<splits,256U,0U,stream>>>(q,k,v,ks,vs,p,l,nullptr,c,0U,nullptr,capacity,splits);
 else TwelveBOrdinaryGlobalGqa4Kernel<false><<<dim3(splits,4U),256U,0U,stream>>>(q,k,v,ks,vs,p,l,nullptr,c,0U,nullptr,capacity,splits);
 return cudaGetLastError();
}
extern "C" cudaError_t gem16_12b_global_t1_merge(const float*p,float*l,
 float*out,const gem16::internal::DecodeControl*c,unsigned splits,cudaStream_t stream){
 using namespace gem16::internal;
 MergeOnlineDecodeAttentionKernel<512,512><<<16U,256U,0U,stream>>>(p,l,out,c,17519U,false,splits);
 return cudaGetLastError();
}
'''
 (directory/'attention.cu').write_text(att)
 inputs=[Path(__file__),R/'src/cuda/fp8/sm120.cu',R/'src/cuda/attention/decode_sm120.cu']
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 (directory/'derivatives.json').write_text(json.dumps({'inputs':{str(p.relative_to(R)):sha(p) for p in inputs},'outputs':{p.name:sha(p) for p in directory.iterdir() if p.is_file()},'o_pipeline':pipeline,'transforms':'12B T1 O rows4/2 or ordered K-load bundles4/8; global GQA4 with unchanged per-head arithmetic and256-thread reductions; no production edit'},indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('--o-pipeline',action='store_true');a=p.parse_args();emit(a.directory,a.o_pipeline)
