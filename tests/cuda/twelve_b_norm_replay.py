"""Offline real-input and finite boundary replay for one12B norm fusion."""
from pathlib import Path
import argparse,ctypes as C,json,random,struct
from twelve_b_device_lab import Device,P,U,F,read,finite
p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('library',type=Path);a=p.parse_args()
cases=json.loads((a.directory/'screen.json').read_text())['cases'];assert len(cases)==12
lib=C.CDLL(str(a.library.resolve()));fn=lib.gem16_12b_norm;fn.argtypes=[U,P,P,P,P,P,P,P,P,F,P];fn.restype=C.c_int
d=Device();projection=d.buffer(15360);postweight=d.buffer(7680);residual=d.buffer(15360);post=d.buffer(15360);hidden=d.buffer(15360);preweight=d.buffer(7680);packed=d.buffer(1920);scales=d.buffer(240);tests=0
try:
 for i,c in enumerate(cases):
  slot=c['slot'];assert slot in range(4) and 0<c['divisor']<1e38
  postweight.put(read(a.directory/f'post-{slot}.bin',7680));preweight.put(read(a.directory/f'pre-{slot}.bin',7680))
  original=read(a.directory/f'part-{i}-0.bin',15360);rv=read(a.directory/f'part-{i}-16384.bin',15360)
  expected=[read(a.directory/f'part-{i}-{offset}.bin',n) for offset,n in ((32768,15360),(49152,15360),(65536,1920),(67584,240))]
  def run(v,pv,rv):
   projection.put(pv);residual.put(rv);code=fn(v,projection.ptr,postweight.ptr,residual.ptr,post.ptr,hidden.ptr,preweight.ptr,packed.ptr,scales.ptr,F(c['divisor']),d.stream);assert code==0,code;d.synchronize();result=[post.get(),hidden.get(),packed.get(),scales.get()];finite(result[0]);finite(result[1]);return result
  assert run(0,original,rv)==expected and run(1,original,rv)==expected,(i,'actual differs');tests+=2
  for edge in range(6):
   rng=random.Random(1200400+edge)
   vals=[0.0 if edge==0 else ((-1 if j%2 else 1)*1e-30 if edge==1 else (rng.uniform(-100,100) if edge==2 else rng.choice((0,.99609375,1.00390625,-1.00390625,3.14159265))*rng.choice((.0001,1,10000)))) for j in range(3840)]
   pv=struct.pack('3840f',*vals);rr=struct.pack('3840f',*([0.0]*3840 if edge<3 else [-v for v in vals]));assert run(0,pv,rr)==run(1,pv,rr),(i,edge,'boundary differs');tests+=1
  d.guards()
 print('PASS',tests,'bit-exact actual/boundary cases, all float intermediates and FP4/scales, finite and guards',flush=True)
finally:d.close()
