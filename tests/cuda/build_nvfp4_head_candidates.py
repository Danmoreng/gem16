"""Build an offline derivative from immutable project NVFP4 source, never runtime JIT."""
from pathlib import Path
import argparse, hashlib, json
root=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('destination',type=Path);p.add_argument('--source',type=Path,default=root/'src/cuda/nvfp4/sm120.cu');o=p.parse_args()
source=o.source.resolve(strict=True);header=root/'tests/cuda/nvfp4_head_candidates.cuh';text=source.read_text()
marker='}  // namespace\n\nStatus LaunchNvfp4Sm120DirectProjection('
assert text.count(marker)==1
text=text.replace(marker,'#include "nvfp4_head_candidates.cuh"\n\n'+marker)
text+=r'''
extern "C" cudaError_t gem16_lab_launch_nvfp4_head(unsigned kind,
 const std::uint8_t*a,const std::uint8_t*sa,const std::uint8_t*b,const std::uint8_t*sb,
 void*out,std::uint64_t rows,std::uint64_t width,float divisor,std::uint32_t m0,std::uint32_t m1,dim3 grid,dim3 block,cudaStream_t stream){
 using namespace gem16::internal;
 if(kind==1U)Sm120HeadPackedT3CandidateKernel<<<grid,block,0,stream>>>(a,sa,b,sb,static_cast<float*>(out),rows,width,divisor);
 else if(kind==2U&&rows%128U==0U)Sm120HeadPairedCandidateKernel<<<grid.x/2U,block,0,stream>>>(a,sa,b,sb,static_cast<Nvfp4ProjectionCandidate*>(out),rows,width,divisor,m0,m1);
 else return cudaErrorInvalidValue;
 return cudaGetLastError();
}
extern "C" cudaError_t gem16_lab_launch_nvfp4_head_parent(unsigned kind,
 const std::uint8_t*a,const std::uint8_t*sa,const std::uint8_t*b,const std::uint8_t*sb,
 void*out,std::uint64_t rows,std::uint64_t width,float divisor,std::uint32_t m0,std::uint32_t m1,dim3 grid,dim3 block,cudaStream_t stream){
 using namespace gem16::internal;
 if(kind==1U)Sm120DirectProjectionFixedT3Kernel<<<grid,block,0,stream>>>(a,sa,b,sb,static_cast<float*>(out),rows,width,divisor);
 else if(kind==2U)Sm120ProjectionArgmaxCandidatesBf16Kernel<<<grid,block,0,stream>>>(a,sa,b,sb,static_cast<Nvfp4ProjectionCandidate*>(out),rows,width,divisor,m0,m1);
 else return cudaErrorInvalidValue;
 return cudaGetLastError();
}
'''
assert not o.destination.exists();o.destination.write_text(text)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
o.destination.with_suffix('.provenance.json').write_text(json.dumps({'diagnostic_only':True,'source':str(source),'source_sha256':sha(source),'header_sha256':sha(header),'generator_sha256':sha(Path(__file__)),'transform':'native M16 R3 scale-aware A packing; adjacent N8 paired Assistant tiles preserving64-token candidate partitions; original source kernels retained','output_sha256':sha(o.destination)},indent=2)+'\n')
