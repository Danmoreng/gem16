"""One bounded Assistant GEMV reduction with the identical FP32 tree."""
from pathlib import Path
import argparse,hashlib,json
root=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('destination',type=Path);a=p.parse_args()
source=root/'src/cuda/mtp/assistant.cu';text=source.read_text()
start=text.index('__global__ void Bf16GemvKernel(')
# Current production also contains the accepted exact-tree kernel. Extract
# only the original kernel so regeneration does not duplicate that definition.
end=text.index('\n}\n',start)+len('\n}\n')
kernel=text[start:end].replace('Bf16GemvKernel','Bf16GemvExactTreeCandidateKernel',1)
old='const float reduced = BlockSum(sum);';assert kernel.count(old)==1
kernel=kernel.replace(old,'''// Same stride128/64/32/16/8/4/2/1 tree as BlockSum256.
  __shared__ float partial[kThreads];
  partial[threadIdx.x] = sum;
  __syncthreads();
  float reduced = 0.0F;
  if (threadIdx.x < 32U) {
    const unsigned lane = threadIdx.x;
    float s0 = partial[lane] + partial[lane + 128U];
    const float s1 = partial[lane + 64U] + partial[lane + 192U];
    float s2 = partial[lane + 32U] + partial[lane + 160U];
    const float s3 = partial[lane + 96U] + partial[lane + 224U];
    s0 += s1;
    s2 += s3;
    s0 += s2;
    for (unsigned offset = 16U; offset != 0U; offset >>= 1U) {
      s0 += __shfl_down_sync(0xFFFFFFFFU, s0, offset);
    }
    reduced = s0;
  }''')
# Keep only the private helpers/kernels preceding host Assistant definitions.
# This diagnostic derivative therefore has no model loader/runtime dependencies.
cut=text.index('struct AssistantModel::Impl')
private_end=text.rfind('}  // namespace',0,cut)
assert private_end>start
text=text[:private_end]+kernel+'\n'+text[private_end:cut]
# The enclosing internal namespace remains open at the cut.
text+='\n}  // namespace gem16::internal\n'
text+=r'''
extern "C" cudaError_t gem16_12b_assistant(unsigned candidate,
 const float*input,const std::uint16_t*weight,float*output,
 std::uint64_t rows,std::uint64_t contracting,cudaStream_t stream){
 using namespace gem16::internal;
 if(candidate==0U)Bf16GemvKernel<<<static_cast<unsigned>(rows),256U,0U,stream>>>(input,weight,output,rows,contracting);
 else if(candidate==1U)Bf16GemvExactTreeCandidateKernel<<<static_cast<unsigned>(rows),256U,0U,stream>>>(input,weight,output,rows,contracting);
 else return cudaErrorInvalidValue;
 return cudaGetLastError();
}
'''
with a.destination.open('x') as f:f.write(text)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a.destination.with_suffix('.provenance.json').write_text(json.dumps({'source_sha256':sha(source),'generator_sha256':sha(Path(__file__)),'output_sha256':sha(a.destination),'transformation':'Assistant GEMV reduction only: exact original FP32 addition tree via one shared-memory barrier and warp shuffles; pair loads/FMA and BF16 boundary unchanged'},indent=2)+'\n')
