"""Real-input FP8 replay and bounded synthetic checks, with optional NCU target."""
from pathlib import Path
import argparse,ctypes as C,json,random
from twelve_b_device_lab import Device,Binding,Dim3,P,U,Q,read,finite
p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('library',type=Path);p.add_argument('--ordinary',action='store_true');p.add_argument('--profile',action='store_true');p.add_argument('--case',type=int);p.add_argument('--variant',type=int,default=0);p.add_argument('--parent-only',action='store_true');p.add_argument('--timing-output',type=Path);a=p.parse_args()
manifest=json.loads((a.directory/('fixtures.json' if a.ordinary else 'screen.json')).read_text());cases=manifest['cases'];assert len(cases)==24
lib=C.CDLL(str(a.library.resolve()));fn=getattr(lib,'gem16_12b_ordinary' if a.ordinary else 'gem16_12b_fp8');fn.argtypes=[U,P,P,Binding,Binding,Binding,U,Q,Dim3,Dim3,P];fn.restype=C.c_int
selected=list(enumerate(cases)) if a.case is None else [(a.case,cases[a.case])]
d=Device();activ=d.buffer(3*8192);scale=d.buffer(12);weights=[d.buffer(8192*8192) for _ in range(3)];ws=[d.buffer(16384) for _ in range(3)];out=[d.buffer(3*8192*4) for _ in range(3)];result=[];tests=0
try:
 for i,c in selected:
  width=c['width'];rows=c['rows'];count=c['count'];tokens=1 if a.ordinary else 3
  assert width in (3840,4096,8192) and count==len(rows) and rows in ([4096,2048,2048],[8192,512],[3840]);assert c['block']==[128,1,1] and c['grid'][1:]==[1,count]
  av=read(a.directory/f'a-{i}.bin',tokens*width);sv=read(a.directory/f's-{i}.bin',tokens*4);activ.put(av+bytes(activ.n-len(av)));scale.put(sv+bytes(scale.n-len(sv)))
  bindings=[];expected=[]
  for j,n in enumerate(rows):
   w=read(a.directory/f'w-{c["slot"]}-{j}.bin',n*width);s=read(a.directory/f'ws-{c["slot"]}-{j}.bin',n*2);weights[j].put(w+bytes(weights[j].n-len(w)));ws[j].put(s+bytes(ws[j].n-len(s)));bindings.append(Binding(weights[j].ptr,ws[j].ptr,out[j].ptr,n));expected.append(read(a.directory/f'expected-{i}-{j}.bin',tokens*n*4))
  bindings.extend(Binding() for _ in range(3-count))
  def run(v):
   code=fn(v,activ.ptr,scale.ptr,*bindings,count,width,Dim3(*c['grid']),Dim3(*c['block']),d.stream)
   assert code==0,code
  if a.profile:
   scratch=d.buffer(256*1024*1024);d.call('cudaMemsetAsync',scratch.ptr,C.c_int(17),C.c_size_t(scratch.n),d.stream);d.synchronize();run(a.variant);d.synchronize();print('PROFILE',i,c,flush=True);break
  for variant in ((0,) if a.parent_only else (0,1)):
   run(variant);d.synchronize()
   for j,n in enumerate(rows):
    actual=out[j].get()[:tokens*n*4];finite(actual);assert actual==expected[j],(i,j,variant,'actual fixture differs')
   tests+=1
  if not a.parent_only:
   for edge in range(4):
    rng=random.Random(12000+edge);av=bytes([0])*tokens*width if edge==0 else (bytes([0x7e if edge==1 else 0xfe])*tokens*width if edge<3 else bytes(rng.choice([0,1,0x38,0x7e,0x80,0xb8,0xfe]) for _ in range(tokens*width)))
    activ.put(av+bytes(activ.n-len(av)));run(0);d.synchronize();reference=[out[j].get()[:tokens*n*4] for j,n in enumerate(rows)];run(1);d.synchronize()
    for j,n in enumerate(rows):actual=out[j].get()[:tokens*n*4];finite(actual);assert actual==reference[j],(i,j,edge,'edge differs')
    tests+=1
   activ.put(read(a.directory/f'a-{i}.bin',tokens*width)+bytes(activ.n-tokens*width))
   if a.timing_output:result.append({**{k:v for k,v in c.items() if k!='samples'},'bit_equal':True,'samples':d.paired(run)})
  d.guards()
 if a.timing_output:
  with a.timing_output.open('x') as f:json.dump({'diagnostic_only':True,'cases':result},f,indent=2)
 print('PASS',tests,'actual/edge cases; finite and device guards',flush=True)
finally:d.close()
