"""One bounded Ordinary12B round/postnorm/residual/pre-MLP quantization fusion."""
from pathlib import Path
import argparse,hashlib,json
root=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('destination',type=Path);a=p.parse_args()
source=root/'src/cuda/nvfp4/reference.cu';norm=root/'src/cuda/norm/reference.cu';text=source.read_text();ns=norm.read_text()
b=ns[ns.index('__device__ float BlockSum('):ns.index('template <typename Input>\n__device__ float LoadBoundaryValue')].replace('BlockSum','TwelveBParentBlockSum')
k=ns[ns.index('template <typename Input>\n__global__ void RmsNormResidualBf16Kernel('):ns.index('__global__ void RmsNormResidualPhysicalBf16Kernel(')].replace('RmsNormResidualBf16Kernel','TwelveBParentNormKernel').replace('BlockSum','TwelveBParentBlockSum').replace('LoadBoundaryValue','LoadRmsNormInput')
quant_start=text.index('  const std::uint64_t blocks = elements / kBlockElements;',text.index('__global__ void RmsNormQuantizeActivationKernel('))
# Stop at the kernel's own closing brace, before any production include.
# The accepted fusion header is included after this kernel in current sources.
quant_end=text.index('\n}\n',quant_start)+len('\n}\n')
quant=text[quant_start:quant_end]
# This original tail closes its per-token quantization kernel. Reuse verbatim,
# replacing only the second RMSNorm's input pointer with its preserved hidden.
quant=quant.replace('LoadRmsNormInput(input, index)','hidden_output[index]')
text+='\nnamespace gem16::internal { namespace {\nconstexpr unsigned kThreads=256U;\n'+b+k+r'''
__global__ void TwelveBRoundProjectionKernel(float*p){const unsigned i=blockIdx.x*256U+threadIdx.x;if(i<3840U)p[i]=static_cast<float>(__float2bfloat16_rn(p[i]));}
__global__ void TwelveBPostAttentionQuantizeCandidateKernel(
 float*projection,const std::uint16_t*post_weight,const float*residual,
 float*post_output,float*hidden_output,const std::uint16_t*weight,
 std::uint8_t*packed_e2m1,std::uint8_t*block_scales_e4m3fn,
 float global_divisor){
 constexpr std::uint64_t elements=3840U;constexpr float epsilon=1.0e-6F;
 float*const normalized_output=nullptr;
 float squared_sum=0.0F;
 for(std::uint64_t index=threadIdx.x;index<elements;index+=blockDim.x){
  const float value=static_cast<float>(__float2bfloat16_rn(projection[index]));
  projection[index]=value;
  squared_sum=fmaf(value,value,squared_sum);
 }
 const float first_inverse=rsqrtf(TwelveBParentBlockSum(squared_sum)/static_cast<float>(elements)+epsilon);
 // Every lane consumes the first result before its shared reduction is reused.
 __syncthreads();
 squared_sum=0.0F;
 for(std::uint64_t index=threadIdx.x;index<elements;index+=blockDim.x){
  const float normalized=static_cast<float>(__float2bfloat16_rn(projection[index]*first_inverse*static_cast<float>(__ushort_as_bfloat16(post_weight[index]))));
  if(post_output!=nullptr)post_output[index]=normalized;
  const float value=static_cast<float>(__float2bfloat16_rn(normalized+residual[index]));
  hidden_output[index]=value;
  squared_sum=fmaf(value,value,squared_sum);
 }
 const float inverse_rms=rsqrtf(TwelveBParentBlockSum(squared_sum)/static_cast<float>(elements)+epsilon);
'''+quant+r'''
} } // private / internal
extern "C" cudaError_t gem16_12b_norm(unsigned candidate,
 float*projection,const std::uint16_t*post_weight,const float*residual,
 float*post_output,float*hidden_output,const std::uint16_t*pre_weight,
 std::uint8_t*packed,std::uint8_t*scales,float divisor,cudaStream_t stream){
 using namespace gem16::internal;
 if(candidate==0U){
  TwelveBRoundProjectionKernel<<<15U,256U,0U,stream>>>(projection);
  TwelveBParentNormKernel<float><<<1U,256U,0U,stream>>>(projection,post_weight,residual,post_output,hidden_output,3840U,1.0e-6F,nullptr);
  RmsNormQuantizeActivationKernel<float><<<1U,256U,0U,stream>>>(hidden_output,pre_weight,nullptr,packed,scales,3840U,1.0e-6F,divisor);
 }else if(candidate==1U){
  TwelveBPostAttentionQuantizeCandidateKernel<<<1U,256U,0U,stream>>>(projection,post_weight,residual,post_output,hidden_output,pre_weight,packed,scales,divisor);
 }else return cudaErrorInvalidValue;
 return cudaGetLastError();
}
'''
with a.destination.open('x') as f:f.write(text)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a.destination.with_suffix('.provenance.json').write_text(json.dumps({'inputs':{str(p.relative_to(root)):sha(p) for p in (source,norm,Path(__file__))},'output_sha256':sha(a.destination),'transformation':'one T1/H3840 fusion preserving original BF16 projection, postnorm, residual, pre-norm rounds,256-lane reduction tree and unchanged NVFP4 quantization tail'},indent=2)+'\n')
