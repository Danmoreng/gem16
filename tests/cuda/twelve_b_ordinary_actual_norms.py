"""Replay retained real12B T1 post-attention boundaries, no throughput claim."""
from pathlib import Path
import argparse,ctypes as C,json
from twelve_b_device_lab import Device,P,F,read,finite
p=argparse.ArgumentParser(description=__doc__);p.add_argument('fixtures',type=Path);p.add_argument('library',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
m=json.load(open(a.fixtures/'screen.json'));assert len(m['cases'])==12
lib=C.CDLL(str(a.library.resolve()));fn=lib.gem16_12b_ordinary_post_attention;fn.argtypes=[P,P,P,P,P,P,P,P,F,P];fn.restype=C.c_int
d=Device();checks=0
try:
 projection=d.buffer(15360);postweight=d.buffer(7680);residual=d.buffer(15360);post=d.buffer(15360);hidden=d.buffer(15360);preweight=d.buffer(7680);packed=d.buffer(1920);scales=d.buffer(240)
 for i,c in enumerate(m['cases']):
  postweight.put(read(a.fixtures/f'post-{c["slot"]}.bin',7680));preweight.put(read(a.fixtures/f'pre-{c["slot"]}.bin',7680));pv=read(a.fixtures/f'part-{i}-0.bin',15360);rv=read(a.fixtures/f'part-{i}-16384.bin',15360)
  expected=[read(a.fixtures/f'part-{i}-{offset}.bin',n) for offset,n in ((32768,15360),(49152,15360),(65536,1920),(67584,240))]
  for alias in (False,True):
   for null_post in (False,True):
    projection.put(pv);residual.put(rv);out=residual if alias else hidden
    assert fn(projection.ptr,postweight.ptr,residual.ptr,P() if null_post else post.ptr,out.ptr,preweight.ptr,packed.ptr,scales.ptr,F(c['divisor']),d.stream)==0
    d.synchronize();actual=[post.get(),out.get(),packed.get(),scales.get()]
    assert actual[1:]==expected[1:],(i,alias,null_post)
    if not null_post:assert actual[0]==expected[0],i
    finite(actual[1]);d.guards();checks+=1
 a.output.write_text(json.dumps({'diagnostic_only':True,'real_fixture_cases':12,'alias_optional_post_combinations':checks,'post_hidden_fp4_scales_bit_exact':True,'guards_and_finite':True},indent=2)+'\n');print('PASS',checks,'actual12B boundary combinations')
finally:d.close()
