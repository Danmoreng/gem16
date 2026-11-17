"""Parent-only actual T3 norm/quant boundary capture; offline LD_PRELOAD only."""
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
base = a.destination / 'base'
subprocess.run([sys.executable, str(ROOT / 'tests/cuda/build_12b_down_capture.py'), str(base)], check=True)
s = (base / 'observer.cu').read_text().replace('GEM16_12B_DOWN_CAPTURE_DIR', 'GEM16_12B_NORM_T3_CAPTURE_DIR')
s = s.replace('stride=65536,slots=4,phases=3', 'stride=196608,slots=4,phases=3')
x = s.index('struct Record{'); y = s.index('std::array<Record,slots>', x)
s = s[:x] + 'struct Record{const std::uint16_t*post=nullptr;const std::uint16_t*pre=nullptr;float divisor=0;};int pending=-1;\n' + s[y:]
x = s.index('void Finish(){'); y = s.index('__attribute__((constructor))', x)
s = s[:x] + r'''
void Finish(){
 finished=true;injecting=true;
 if(calls!=48||ignored!=48||directed_captures!=2)throw std::runtime_error("unexpected T3 norm graph topology");
 unsigned groups=0;Check(cudaMemcpy(&groups,counter,4,cudaMemcpyDeviceToHost));
 if(groups!=507)throw std::runtime_error("expected507 D2 groups");
 const auto dir=std::filesystem::path(std::getenv("GEM16_12B_NORM_T3_CAPTURE_DIR"));
 if(!std::filesystem::create_directory(dir))throw std::runtime_error("capture directory exists");
 std::ofstream f(dir/"captures.json");f.precision(10);
 f<<"{\"diagnostic_only\":true,\"parent_only\":true,\"tokens\":3,\"width\":3840,\"groups\":"<<groups<<",\"calls\":"<<calls<<",\"ignored\":"<<ignored<<",\"actual_groups\":[0,250,506],\"cases\":[";
 for(unsigned i=0;i<phases*slots;++i){
  const auto&r=records[i%slots];if(!r.post||!r.pre||!(r.divisor>0))throw std::runtime_error("missing norm binding");
  for(auto part:std::array<std::array<std::size_t,2>,5>{{{0,46080},{49152,46080},{98304,46080},{147456,5760},{155648,720}}})
   Export(dir/("part-"+std::to_string(i)+"-"+std::to_string(part[0])+".bin"),Base(i)+part[0],part[1]);
  if(i<slots){Export(dir/("post-"+std::to_string(i)+".bin"),r.post,7680);Export(dir/("pre-"+std::to_string(i)+".bin"),r.pre,7680);}
  if(i)f<<',';f<<"{\"slot\":"<<i%slots<<",\"phase\":"<<i/slots<<",\"layer\":"<<layers[i%slots]<<",\"divisor\":"<<r.divisor<<"}";
 }
 f<<"]}\n";f.close();injecting=false;
}
''' + s[y:]
s = s.replace('parent=reinterpret_cast<Parent>(dlsym(RTLD_DEFAULT,"gem16_12b_down_parent"));\n  if(!parent)std::abort();', '')
# The copied Down observer creates its buffers at the same already-qualified
# stream-capture boundary. It needs no replay kernel or candidate during capture.
x = s.index('extern "C" cudaError_t cudaLaunchKernel(')
y = s.index('extern "C" cudaError_t __cudaLaunchKernel', x)
s = s[:x] + r'''
extern "C" cudaError_t cudaLaunchKernel(const void*fn,dim3 grid,dim3 block,void**args,std::size_t shared,cudaStream_t stream){
 const auto e=launch(fn,grid,block,args,shared,stream);
 if(!enabled||!ready||injecting||finished||e!=cudaSuccess)return e;
 const char*n=nullptr;Check(cudaFuncGetName(&n,fn));if(!n)return e;
 if(std::strstr(n,"RmsNormResidualBf16KernelIfEE")&&Arg<std::uint64_t>(args,5)==3840&&Arg<const void*>(args,7)==nullptr){
  if(!directed){++ignored;return e;}
  if(grid.x!=3||block.x!=256||Arg<const void*>(args,3)!=nullptr)std::abort();
  const unsigned layer=calls++;pending=-1;for(unsigned j=0;j<slots;++j)if(layer==layers[j])pending=j;
  if(pending<0)return e;const unsigned slot=pending;records[slot].post=Arg<const std::uint16_t*>(args,1);
  injecting=true;
  Snapshot<<<1,256,0,stream>>>(reinterpret_cast<const std::uint8_t*>(Arg<const void*>(args,0)),Base(slot),46080,counter,slot==0);
  Snapshot<<<1,256,0,stream>>>(reinterpret_cast<const std::uint8_t*>(Arg<const void*>(args,2)),Base(slot)+49152,46080,counter,false);
  Snapshot<<<1,256,0,stream>>>(reinterpret_cast<const std::uint8_t*>(Arg<const void*>(args,4)),Base(slot)+98304,46080,counter,false);
  Check(cudaGetLastError());injecting=false;
 }else if(directed&&pending>=0&&std::strstr(n,"RmsNormQuantizeActivationKernelIfEE")&&Arg<std::uint64_t>(args,5)==3840){
  const unsigned slot=pending;records[slot].pre=Arg<const std::uint16_t*>(args,1);records[slot].divisor=Arg<float>(args,7);
  injecting=true;
  Snapshot<<<1,256,0,stream>>>(Arg<const std::uint8_t*>(args,3),Base(slot)+147456,5760,counter,false);
  Snapshot<<<1,256,0,stream>>>(Arg<const std::uint8_t*>(args,4),Base(slot)+155648,720,counter,false);
  Check(cudaGetLastError());injecting=false;pending=-1;
 }
 return e;
}
''' + s[y:]
assert 'if(!parent)std::abort()' not in s
out = a.destination / 'observer.cu'
with out.open('x') as f: f.write(s)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
(a.destination / 'derivatives.json').write_text(json.dumps({
    'source_sha256': sha(base / 'observer.cu'), 'generator_sha256': sha(Path(__file__)),
    'output_sha256': sha(out), 'scope': 'Parent T3 post-attention residual/pre-MLP quantization capture; all three rows, four layers, three phases; no candidate timing'}, indent=2) + '\n')
