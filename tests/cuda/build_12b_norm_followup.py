"""Rebuild the single existing norm candidate plus exact O producer/trace ABI."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('destination',type=Path);a=p.parse_args();out=a.destination;out.mkdir(parents=True,exist_ok=True)
base=out/'candidate-base.cu';subprocess.run([sys.executable,str(ROOT/'tests/cuda/build_12b_norm_candidate.py'),str(base)],check=True)
s=base.read_text();start=s.index('__global__ void TwelveBPostAttentionQuantizeCandidateKernel(');end=s.index('\n} } // private / internal',start)
trace=s[start:end].replace('TwelveBPostAttentionQuantizeCandidateKernel','TwelveBPostAttentionQuantizeTraceKernel',1)
trace=trace.replace('float global_divisor){','float global_divisor,float* normalized_output){').replace(' float*const normalized_output=nullptr;\n','')
s+='\nnamespace gem16::internal { namespace {\n'+trace+'\n} }\n'+r'''
extern "C" cudaError_t gem16_12b_norm_trace(unsigned candidate,
 float*projection,const std::uint16_t*post_weight,const float*residual,
 float*post_output,float*hidden_output,const std::uint16_t*pre_weight,
 std::uint8_t*packed,std::uint8_t*scales,float divisor,float*normalized,cudaStream_t stream){
 using namespace gem16::internal;
 if(candidate==0U){
  TwelveBRoundProjectionKernel<<<15U,256U,0U,stream>>>(projection);
  TwelveBParentNormKernel<float><<<1U,256U,0U,stream>>>(projection,post_weight,residual,post_output,hidden_output,3840U,1.0e-6F,nullptr);
  RmsNormQuantizeActivationKernel<float><<<1U,256U,0U,stream>>>(hidden_output,pre_weight,normalized,packed,scales,3840U,1.0e-6F,divisor);
 }else if(candidate==1U){
  TwelveBPostAttentionQuantizeTraceKernel<<<1U,256U,0U,stream>>>(projection,post_weight,residual,post_output,hidden_output,pre_weight,packed,scales,divisor,normalized);
 }else return cudaErrorInvalidValue;
 return cudaGetLastError();
}
'''
with (out/'candidate.cu').open('x') as f:f.write(s)
s=(ROOT/'src/cuda/fp8/sm120.cu').read_text()+r'''
extern "C" cudaError_t gem16_12b_producer(const std::uint8_t*a,const float*s,
 const std::uint8_t*w,const std::uint16_t*ws,float*output,std::uint64_t width,cudaStream_t stream){
 using namespace gem16::internal;
 const Fp8MatrixBinding binding{w,ws,output,3840U};
 Sm120DirectProjectionKernel<false><<<dim3(120U,1U,1U),128U,0U,stream>>>(a,s,binding,{}, {},1U,1U,width);
 return cudaGetLastError();
}
'''
with (out/'producer.cu').open('x') as f:f.write(s)
identity=lambda x:{'bytes':x.stat().st_size,'sha256':hashlib.sha256(x.read_bytes()).hexdigest()}
inputs=[ROOT/'tests/cuda/build_12b_norm_candidate.py',ROOT/'src/cuda/nvfp4/reference.cu',ROOT/'src/cuda/norm/reference.cu',ROOT/'src/cuda/fp8/sm120.cu',Path(__file__)]
with (out/'derivatives.json').open('x') as f:json.dump({'scope':'unchanged prior fusion candidate; added optional trace ABI for pre-MLP normalized bits and unchanged native FP8 O producer; no runtime integration','inputs':{str(p.relative_to(ROOT)):identity(p) for p in inputs},'outputs':{p.name:identity(p) for p in [base,out/'candidate.cu',out/'producer.cu']}},f,indent=2)
