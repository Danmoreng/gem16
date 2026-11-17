"""Generate export-only Ordinary observer from the reviewed T3 observer."""
from pathlib import Path
import argparse,hashlib,json
root=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('destination',type=Path);a=p.parse_args()
src=root/'tests/cuda/twelve_b_fp8_operator_lab.cu';s=src.read_text()
s=s.replace('GEM16_12B_FP8_LAB_DIR','GEM16_12B_ORDINARY_LAB_DIR')
s=s.replace('group==250?1:(group==506?2:-1)','group==566?1:(group==1133?2:-1)')
a1=s.index('void Finish(){');a2=s.index('__attribute__((constructor))',a1)
s=s[:a1]+r'''void Finish(){finished=true;injecting=true;const auto dir=std::filesystem::path(std::getenv("GEM16_12B_ORDINARY_LAB_DIR"));if(!std::filesystem::create_directory(dir))throw std::runtime_error("directory exists");
std::ofstream f(dir/"fixtures.json");f<<"{\"diagnostic_only\":true,\"tokens\":1,\"actual_decode_steps\":[0,566,1133],\"cases\":[";
for(unsigned i=0;i<phases*slots;++i){const auto&r=records[i%slots];if(!r.count||r.width>8192)throw std::runtime_error("missing/invalid capture");
Export(dir/("a-"+std::to_string(i)+".bin"),Base(i),r.width);Export(dir/("s-"+std::to_string(i)+".bin"),Scale(i),4);for(unsigned j=0;j<r.count;++j){auto stem=std::to_string(i)+"-"+std::to_string(j);Export(dir/("expected-"+stem+".bin"),Output(i,j),r.b[j].rows*sizeof(float));if(i<slots){Export(dir/("w-"+stem+".bin"),r.b[j].weight,r.b[j].rows*r.width);Export(dir/("ws-"+stem+".bin"),r.b[j].weight_scales,2*r.b[j].rows);}}
if(i)f<<',';f<<"{\"slot\":"<<i%slots<<",\"phase\":"<<i/slots<<",\"layer\":"<<layers[(i%slots)/2]<<",\"family\":\""<<(i%2?"O":"QKV")<<"\",\"width\":"<<r.width<<",\"count\":"<<r.count<<",\"rows\":[";for(unsigned j=0;j<r.count;++j){if(j)f<<',';f<<r.b[j].rows;}f<<"],\"grid\":["<<r.grid.x<<','<<r.grid.y<<','<<r.grid.z<<"],\"block\":["<<r.block.x<<','<<r.block.y<<','<<r.block.z<<"]}";}
f<<"]}\n";f.close();injecting=false;}
'''+s[a2:]
s=s.replace('gate=reinterpret_cast<Gate>(dlsym(RTLD_DEFAULT,"gem16_12b_fp8"));if(!gate)std::abort();','')
s=s.replace('!std::strstr(n,"Sm120DirectProjectionFixedBatchKernel")','!std::strstr(n,"Sm120DirectProjectionKernelILb0EE")')
s=s.replace('r.width=Arg<std::uint64_t>(args,6)','r.width=Arg<std::uint64_t>(args,7)')
s=s.replace('Base(slot),3*r.width','Base(slot),r.width').replace('Scale(slot)),12','Scale(slot)),4').replace('3*r.b[j].rows*sizeof(float),counter','r.b[j].rows*sizeof(float),counter')
s=s.replace('if(count>=507)','if(count>=1134)')
s+=r'''
extern "C" cudaError_t cudaStreamSynchronize(cudaStream_t stream){using Sync=cudaError_t(*)(cudaStream_t);static auto real=reinterpret_cast<Sync>(dlsym(RTLD_NEXT,"cudaStreamSynchronize"));const auto e=real(stream);if(enabled&&ready&&!injecting&&!finished&&calls>=96&&e==cudaSuccess){unsigned count=0;injecting=true;Check(cudaMemcpy(&count,counter,4,cudaMemcpyDeviceToHost));injecting=false;if(count>=1134){try{Finish();}catch(const std::exception&x){std::fprintf(stderr,"12B ordinary lab: %s\n",x.what());std::abort();}}}return e;}
'''
with a.destination.open('x') as f:f.write(s)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a.destination.with_suffix('.provenance.json').write_text(json.dumps({'observer_source_sha256':sha(src),'generator_sha256':sha(Path(__file__)),'output_sha256':sha(a.destination),'transformation':'export-only T1 FP8 observer; original production outputs at steps0/566/1133, no timed candidate'},indent=2)+'\n')
