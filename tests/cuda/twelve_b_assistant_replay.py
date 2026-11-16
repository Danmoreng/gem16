"""Offline actual Assistant GEMV fixtures and exact-reduction boundary tests."""
from pathlib import Path
import argparse,ctypes as C,json,random,struct
from twelve_b_device_lab import Device,P,U,Q,read,finite
p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('library',type=Path);p.add_argument('--case',type=int);a=p.parse_args()
cases=json.loads((a.directory/'screen.json').read_text())['cases'];assert len(cases)==126
lib=C.CDLL(str(a.library.resolve()));fn=lib.gem16_12b_assistant;fn.argtypes=[U,P,P,P,Q,Q,P];fn.restype=C.c_int
d=Device();activ=d.buffer(8192*4);weight=d.buffer(8192*1024*2);output=d.buffer(8192*4);tests=0
try:
 for i,c in (enumerate(cases) if a.case is None else [(a.case,cases[a.case])]):
  rows=c['rows'];width=c['width'];assert rows in (1024,3840,4096,8192) and width in (1024,4096,8192);assert rows*width*2<=weight.n
  w=read(a.directory/f'w-{c["slot"]}.bin',rows*width*2);weight.put(w+bytes(weight.n-len(w)))
  av=read(a.directory/f'a-{i}.bin',width*4);activ.put(av+bytes(activ.n-len(av)));expected=read(a.directory/f'expected-{i}.bin',rows*4)
  def run(v):
   code=fn(v,activ.ptr,weight.ptr,output.ptr,rows,width,d.stream);assert code==0,code;d.synchronize();out=output.get()[:rows*4];finite(out);return out
  assert run(0)==expected and run(1)==expected,(i,'actual differs');tests+=2
  for edge in range(5):
   rng=random.Random(1200500+edge)
   vals=[0.0 if edge==0 else (1.0 if edge==1 else ((-1.0 if j%2 else 1.0)*1000.0 if edge==2 else (2.0**-120 if edge==3 else rng.choice((-1.00390625,-1,0,1,1.00390625))*rng.choice((.001,1,1024))))) for j in range(width)]
   av=struct.pack(f'{width}f',*vals);activ.put(av+bytes(activ.n-len(av)));assert run(0)==run(1),(i,edge,'boundary differs');tests+=1
  d.guards()
 print('PASS',tests,'bit-exact actual/boundary cases; finite outputs and guarded buffers',flush=True)
finally:d.close()
