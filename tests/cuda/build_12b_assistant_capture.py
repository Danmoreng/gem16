"""Prepare parent-only Assistant captures before the historical/counter gate.
No candidate execution or timing is performed by this observer.
"""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('destination',type=Path);a=p.parse_args();out=a.destination;out.mkdir(parents=True,exist_ok=True)
base=out/'observer-base.cu';subprocess.run([sys.executable,str(ROOT/'tests/cuda/build_12b_assistant_observer.py'),str(base)],check=True)
s=base.read_text();start=s.index('void Finish(){');end=s.index('__attribute__((constructor))',start)
s=s[:start]+r'''void Finish(){
 finished=true;injecting=true;
 if(calls!=42U)throw std::runtime_error("expected exactly42 static GEMV calls");
 if(ignored_standalone_gemv!=42U||directed_captures!=2U)throw std::runtime_error("unexpected standalone/conditional capture topology");
 unsigned groups=0;Check(cudaMemcpy(&groups,counter,4,cudaMemcpyDeviceToHost));
 if(groups!=507U)throw std::runtime_error("expected507 native D2 groups");
 const auto dir=std::filesystem::path(std::getenv("GEM16_12B_ASSISTANT_LAB_DIR"));
 if(!std::filesystem::create_directory(dir))throw std::runtime_error("directory exists");
 constexpr std::array<std::array<std::uint64_t,2>,21> shapes{{
 {4096,1024},{1024,4096},{8192,1024},{8192,1024},{1024,8192},
 {4096,1024},{1024,4096},{8192,1024},{8192,1024},{1024,8192},
 {4096,1024},{1024,4096},{8192,1024},{8192,1024},{1024,8192},
 {8192,1024},{1024,8192},{8192,1024},{8192,1024},{1024,8192},
 {3840,1024}}};
 for(unsigned slot=0;slot<42;++slot){
  const auto&r=records[slot];
  if(r.rows!=shapes[slot%21][0]||r.width!=shapes[slot%21][1])throw std::runtime_error("unexpected shape/call order");
  if(slot>=21&&r.weight!=records[slot-21].weight)throw std::runtime_error("drafts do not share expected weights");
 }
 Check(cudaMalloc(&outputs[0],32768));cudaStream_t stream{};Check(cudaStreamCreate(&stream));
 std::ofstream f(dir/"captures.json");
 f<<"{\"diagnostic_only\":true,\"parent_only\":true,\"captured_static_calls\":"<<calls<<",\"ignored_standalone_gemv\":"<<ignored_standalone_gemv<<",\"directed_captures\":"<<directed_captures<<",\"groups\":"<<groups<<",\"actual_groups\":[0,250,506],\"cases\":[";
 for(unsigned i=0;i<phases*slots;++i){
  const auto&r=records[i%slots];const auto n=r.rows*sizeof(float);
  const auto expected=Download(Output(i,0),n);Run(0,i,stream);Check(cudaStreamSynchronize(stream));
  if(expected!=Download(outputs[0],n))throw std::runtime_error("parent replay differs from actual capture");
  Export(dir/("a-"+std::to_string(i)+".bin"),Base(i),r.width*sizeof(float));
  Export(dir/("expected-"+std::to_string(i)+".bin"),Output(i,0),n);
  if(i<slots)Export(dir/("w-"+std::to_string(i)+".bin"),r.weight,r.rows*r.width*2);
  if(i)f<<',';
  f<<"{\"slot\":"<<i%slots<<",\"phase\":"<<i/slots<<",\"rows\":"<<r.rows<<",\"width\":"<<r.width<<",\"parent_bit_equal\":true}";
 }
 f<<"]}\n";f.close();Check(cudaStreamDestroy(stream));Check(cudaFree(outputs[0]));injecting=false;
}
'''+s[end:]
# The runtime constructs both a standalone D2 graph and a conditional-owned
# D2 body. Only the latter executes in the measured chained request. Capture
# its42 calls explicitly rather than counting the two constructions as84.
s=s.replace('bool enabled=false,ready=false',
            'bool capture_into_graph=false;unsigned ignored_standalone_gemv=0,directed_captures=0;bool enabled=false,ready=false')
marker='if(!n||!std::strstr(n,"Bf16GemvKernel"))return e;'
assert s.count(marker)==1
s=s.replace(marker,marker+'if(!capture_into_graph){++ignored_standalone_gemv;return e;}')
s+=r'''
extern "C" cudaError_t cudaStreamBeginCaptureToGraph(cudaStream_t stream,cudaGraph_t graph,const cudaGraphNode_t*dependencies,const cudaGraphEdgeData*edges,std::size_t count,cudaStreamCaptureMode mode){
 using Fn=cudaError_t(*)(cudaStream_t,cudaGraph_t,const cudaGraphNode_t*,const cudaGraphEdgeData*,std::size_t,cudaStreamCaptureMode);
 static auto real=reinterpret_cast<Fn>(dlsym(RTLD_NEXT,"cudaStreamBeginCaptureToGraph"));
 const auto e=real(stream,graph,dependencies,edges,count,mode);
 if(enabled&&ready&&!injecting&&!finished&&e==cudaSuccess){capture_into_graph=true;++directed_captures;}
 return e;
}
extern "C" cudaError_t cudaStreamEndCapture(cudaStream_t stream,cudaGraph_t*graph){
 using Fn=cudaError_t(*)(cudaStream_t,cudaGraph_t*);static auto real=reinterpret_cast<Fn>(dlsym(RTLD_NEXT,"cudaStreamEndCapture"));
 const auto e=real(stream,graph);capture_into_graph=false;return e;
}
'''
with (out/'observer.cu').open('x') as f:f.write(s)
s=(ROOT/'src/cuda/mtp/assistant.cu').read_text();cut=s.index('struct AssistantModel::Impl');s=s[:cut]+'\n} // namespace gem16::internal\n'+r'''
extern "C" cudaError_t gem16_12b_assistant(unsigned variant,
 const float*input,const std::uint16_t*weight,float*output,
 std::uint64_t rows,std::uint64_t width,cudaStream_t stream){
 using namespace gem16::internal;
 if(variant!=0U)return cudaErrorInvalidValue;
 Bf16GemvKernel<<<static_cast<unsigned>(rows),256U,0U,stream>>>(input,weight,output,rows,width);
 return cudaGetLastError();
}
'''
with (out/'parent.cu').open('x') as f:f.write(s)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
with (out/'capture-derivatives.json').open('x') as f:json.dump({'scope':'parent-only actual capture/validation; no optimization candidate before historical/counter gate','inputs':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'tests/cuda/build_12b_assistant_observer.py',ROOT/'src/cuda/mtp/assistant.cu']},'outputs':{p.name:sha(p) for p in [out/'observer.cu',out/'parent.cu']}},f,indent=2)
