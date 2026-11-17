"""Real12B T1 O/global-attention exactness and one-warmup/three-sample lab."""
from pathlib import Path
import argparse,ctypes as C,json,random,statistics,struct
from twelve_b_device_lab import Device,P,U,Q,F,Z,Binding,read,finite

def timing(d,launch,variants):
    evict=d.buffer(256*1024*1024);graphs={v:d.graph(lambda v=v:launch(v),1) for v in variants}
    start=P();end=P();d.call('cudaEventCreate',C.byref(start));d.call('cudaEventCreate',C.byref(end));d.events.extend([start,end])
    for v in variants:d.call('cudaGraphLaunch',graphs[v],d.stream)
    d.synchronize();samples={v:[] for v in variants}
    for sample in range(3):
        for v in variants if sample%2==0 else list(reversed(variants)):
            d.call('cudaMemsetAsync',evict.ptr,C.c_int(127),Z(evict.n),d.stream)
            d.call('cudaEventRecord',start,d.stream);d.call('cudaGraphLaunch',graphs[v],d.stream);d.call('cudaEventRecord',end,d.stream);d.call('cudaEventSynchronize',end)
            ms=F();d.call('cudaEventElapsedTime',C.byref(ms),start,end);samples[v].append(ms.value*1000)
    d.guards();return {str(v):{'samples_us':xs,'median_us':statistics.median(xs)} for v,xs in samples.items()}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--family',choices=['o','global'],required=True);p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--library',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--timing',action='store_true');p.add_argument('--case',type=int);a=p.parse_args();lib=C.CDLL(str(a.library.resolve()));d=Device();results=[];checks=0
 try:
  if a.family=='o':
   m=json.load(open(a.fixtures/'fixtures.json'));assert len(m['cases'])==24
   fn=lib.gem16_12b_o;fn.argtypes=[U,P,P,Binding,Q,P];fn.restype=C.c_int
   act=d.buffer(8192);scale=d.buffer(4);weight=d.buffer(3840*8192);ws=d.buffer(7680);out=d.buffer(15360)
   for i,c in enumerate(m['cases']):
    if c['family']!='O' or (a.case is not None and a.case!=i):continue
    width=c['width'];assert width in [4096,8192] and c['rows']==[3840] and c['count']==1
    av=read(a.fixtures/f'a-{i}.bin',width);act.put(av+bytes(8192-width));scale.put(read(a.fixtures/f's-{i}.bin',4));w=read(a.fixtures/f'w-{c["slot"]}-0.bin',3840*width);weight.put(w+bytes(weight.n-len(w)));ws.put(read(a.fixtures/f'ws-{c["slot"]}-0.bin',7680));expected=read(a.fixtures/f'expected-{i}-0.bin',15360);binding=Binding(weight.ptr,ws.ptr,out.ptr,3840)
    def launch(v):assert fn(v,act.ptr,scale.ptr,binding,Q(width),d.stream)==0
    for v in [0,1,2]:launch(v);d.synchronize();actual=out.get();finite(actual);assert actual==expected,(i,v,'actual O mismatch');checks+=1
    for edge in range(4):
     rng=random.Random(171117+edge);vals=bytes(width) if edge==0 else bytes([0x7e if edge==1 else 0xfe])*width if edge<3 else bytes(rng.choice([0,1,0x38,0x7e,0x80,0xb8,0xfe]) for _ in range(width));act.put(vals+bytes(8192-width));launch(0);d.synchronize();parent=out.get();finite(parent)
     for v in [1,2]:launch(v);d.synchronize();actual=out.get();finite(actual);assert actual==parent,(i,v,edge,'finite O boundary mismatch');checks+=1
    act.put(av+bytes(8192-width));sample=timing(d,launch,[0,1,2]) if a.timing and i in [1,7] else None;d.guards();results.append({'case':i,'width':width,'bit_exact':True,'timing':sample});print('PASS O',i,flush=True)
  else:
   m=json.load(open(a.fixtures/'captures.json'));assert m['kv_heads']==1 and m['rows']==3 and len(m['cases'])==24
   fn=lib.gem16_12b_global_t1;fn.argtypes=[U]+[P]*8+[Q,U,P];fn.restype=C.c_int
   merge=lib.gem16_12b_global_t1_merge;merge.argtypes=[P,P,P,P,U,P];merge.restype=C.c_int
   sizes={'q':16*512*4,'k':17519*512,'v':17519*512,'ks':2,'vs':2,'p':35*16*512*4,'l':35*16*4,'c':24,'out':16*512*4};b={k:d.buffer(n) for k,n in sizes.items()};capacity=17519;splits=35
   for i,c in enumerate(m['cases']):
    if a.case is not None and a.case!=i:continue
    for key in ['q','c']:
     data=read(a.fixtures/f'{key}-{i}.bin',sizes[key]*3)[:sizes[key]];b[key].put(data)
    for key in ['k','v','ks','vs']:b[key].put(read(a.fixtures/f'{key}-{i}.bin',sizes[key]))
    expected={key:read(a.fixtures/f'{key}-{i}.bin',sizes[key]*3)[:sizes[key]] for key in ['p','l']}
    def launch(v):assert fn(v,*(b[k].ptr for k in ['q','k','v','ks','vs','p','l','c']),Q(capacity),U(splits),d.stream)==0
    def run(v,seed=expected):
     for key in ['p','l']:b[key].put(seed[key])
     launch(v);d.synchronize();partial=b['p'].get();lse=b['l'].get();assert merge(b['p'].ptr,b['l'].ptr,b['out'].ptr,b['c'].ptr,U(splits),d.stream)==0;d.synchronize();output=b['out'].get();finite(output);return partial,lse,output,b['l'].get()
    parent=run(0);assert parent[:2]==(expected['p'],expected['l']),(i,'parent row0 differs from real capture');assert run(1)==parent,(i,'global partial/LSE/merge mismatch');checks+=2
    # Cover active/inactive splits, first/last token and split boundaries with
    # finite cache values, retaining the control's other fields.
    control=bytearray(read(a.fixtures/f'c-{i}.bin',72)[:24])
    if i==0 and a.case is None:
     valid=(c['position']+3)*512
     for key in ['k','v']:
      raw=read(a.fixtures/f'{key}-{i}.bin',sizes[key]);b[key].put(raw[:valid]+bytes(len(raw)-valid))
     seed={k:bytes([165])*sizes[k] for k in ['p','l']}
     for cap,position in [(17519,x) for x in [0,510,511,512,1023,16383,16384,17518]] + [(16384,0),(16384,16383),(16385,16384),(17408,17407)]:
      capacity=cap;splits=(capacity+511)//512
      struct.pack_into('Q',control,8,position);b['c'].put(bytes(control));assert run(0,seed)==run(1,seed),('global boundary',position);d.guards();checks+=1
     capacity=17519;splits=35
     b['c'].put(read(a.fixtures/f'c-{i}.bin',72)[:24])
     for key in ['k','v']:b[key].put(read(a.fixtures/f'{key}-{i}.bin',sizes[key]))
    sample=timing(d,launch,[0,1]) if a.timing and i in [0,16] else None;d.guards();results.append({'case':i,'position':c['position'],'bit_exact':True,'timing':sample});print('PASS global',i,flush=True)
  a.output.write_text(json.dumps({'diagnostic_only':True,'family':a.family,'checks':checks,'cases':results,'warmups_per_timed_variant':1,'retained_samples_per_timed_variant':3,'timing_boundary':'single CUDA graph event interval;256MiB cache eviction before event; no primary throughput claim','finite_and_guards':True},indent=2)+'\n');print('PASS',a.family,checks,'checks',flush=True)
 finally:d.close()
if __name__=='__main__':main()
