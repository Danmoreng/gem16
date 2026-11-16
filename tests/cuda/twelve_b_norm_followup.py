"""Raw-O boundary and unchanged native O-producer graph-chain replay (diagnostic)."""
from pathlib import Path
import argparse,ctypes as C,hashlib,json,random,struct,time
from twelve_b_device_lab import Device,P,U,F,Q,Z,read,finite
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--ordinary',type=Path,required=True);p.add_argument('--norm',type=Path,required=True)
p.add_argument('--candidate',type=Path,required=True);p.add_argument('--producer',type=Path,required=True)
p.add_argument('--output',type=Path);p.add_argument('--case',type=int);a=p.parse_args()
old=json.loads((a.norm/'screen.json').read_text())['cases'];ordinary=json.loads((a.ordinary/'fixtures.json').read_text())['cases']
lib=C.CDLL(str(a.candidate.resolve()));fn=lib.gem16_12b_norm;fn.argtypes=[U,P,P,P,P,P,P,P,P,F,P];fn.restype=C.c_int
trace=lib.gem16_12b_norm_trace;trace.argtypes=[U,P,P,P,P,P,P,P,P,F,P,P];trace.restype=C.c_int
prod=C.CDLL(str(a.producer.resolve())).gem16_12b_producer;prod.argtypes=[P,P,P,P,P,Q,P];prod.restype=C.c_int
selected=list(enumerate(old)) if a.case is None else [(a.case,old[a.case])]
d=Device();projection=d.buffer(15360);original=d.buffer(15360);postweight=d.buffer(7680);residual=d.buffer(15360);post=d.buffer(15360);hidden=d.buffer(15360);preweight=d.buffer(7680);packed=d.buffer(1920);scales=d.buffer(240);normalized=d.buffer(15360)
if a.output:evict=d.buffer(256*1024*1024)
results=[];checks=0

def event():
 e=P();d.call('cudaEventCreate',C.byref(e));d.events.append(e);return e

def paired(run,chain):
 # Restore before EVERY timed invocation, not just before a 16-node graph.
 # Unique event pairs within the graph exclude restoration/eviction exactly.
 schemes={}
 for cold,n in [(False,16),(True,1)]:
  for variant in [0,1]:
   pairs=[(event(),event()) for _ in range(n)]
   graph=P();execute=P();d.call('cudaStreamBeginCapture',d.stream,U(1))
   for start,end in pairs:
    if not chain:d.call('cudaMemcpyAsync',projection.ptr,original.ptr,Z(15360),C.c_int(3),d.stream)
    if cold:d.call('cudaMemsetAsync',evict.ptr,C.c_int(123),Z(evict.n),d.stream)
    d.call('cudaEventRecordWithFlags',start,d.stream,U(1))
    if chain:produce()
    run(variant)
    d.call('cudaEventRecordWithFlags',end,d.stream,U(1))
   d.call('cudaStreamEndCapture',d.stream,C.byref(graph));d.call('cudaGraphInstantiateWithFlags',C.byref(execute),graph,Q(0));d.call('cudaGraphDestroy',graph);d.graphs.append(execute)
   schemes[cold,variant]=(execute,pairs)
 for _ in range(8):
  for variant in [0,1]:d.call('cudaGraphLaunch',schemes[False,variant][0],d.stream)
 d.synchronize()
 def measure(cold,variant):
  execute,pairs=schemes[cold,variant];d.call('cudaGraphLaunch',execute,d.stream);d.synchronize();total=0.0
  for start,end in pairs:
   elapsed=F();d.call('cudaEventElapsedTime',C.byref(elapsed),start,end);total+=elapsed.value*1000
  return total/len(pairs)
 samples=[]
 for sample in range(25):
  row=[0.0]*4
  for cold in [False,True]:
   for variant in ([0,1] if sample%2==0 else [1,0]):row[2*int(cold)+variant]=measure(cold,variant)
  samples.append(row)
 return samples

try:
 for i,c in selected:
  matches=[(j,x) for j,x in enumerate(ordinary) if (x['layer'],x['phase'],x['family'])==(c['layer'],c['phase'],'O')];assert len(matches)==1
  j,o=matches[0];raw=read(a.ordinary/f'expected-{j}-0.bin',15360);rounded=read(a.norm/f'part-{i}-0.bin',15360)
  words=struct.unpack('<3840I',raw);assert struct.pack('<3840I',*[((x+32767+((x>>16)&1))&0xffff0000) for x in words])==rounded
  assert raw!=rounded
  rv=read(a.norm/f'part-{i}-16384.bin',15360);postweight.put(read(a.norm/f'post-{c["slot"]}.bin',7680));preweight.put(read(a.norm/f'pre-{c["slot"]}.bin',7680))
  expected=[read(a.norm/f'part-{i}-{offset}.bin',n) for offset,n in [(32768,15360),(49152,15360),(65536,1920),(67584,240)]]
  def run(variant,with_trace=False,alias=False):
   h=residual if alias else hidden
   arguments=[variant,projection.ptr,postweight.ptr,residual.ptr,post.ptr,h.ptr,preweight.ptr,packed.ptr,scales.ptr,F(c['divisor'])]
   code=trace(*arguments,normalized.ptr,d.stream) if with_trace else fn(*arguments,d.stream)
   assert code==0,code
  def evaluate(variant,pv,rr,alias=False):
   projection.put(pv);residual.put(rr);run(variant,True,alias);d.synchronize()
   h=residual if alias else hidden
   value=[projection.get(),post.get(),h.get(),normalized.get(),packed.get(),scales.get()]
   for x in value[:4]:finite(x)
   return value
  parent=evaluate(0,raw,rv);candidate=evaluate(1,raw,rv);assert parent==candidate,(i,'raw actual mismatch')
  assert parent[0]==rounded and [parent[1],parent[2],parent[4],parent[5]]==expected,(i,'captured boundaries mismatch');checks+=1
  assert evaluate(0,raw,rv,True)==evaluate(1,raw,rv,True)==parent,(i,'residual/hidden alias mismatch');checks+=1
  for edge in range(6):
   rng=random.Random(1201500+edge)
   values=[0.0 if edge==0 else ((-1 if k%2 else 1)*1e-30 if edge==1 else (rng.uniform(-100,100) if edge==2 else rng.choice([0.0,.99609375,1.00390625,1.0039061307907104,1.0039063692092896,-1.00390625,3.14159265])*rng.choice([.0001,1.,10000.]))) for k in range(3840)]
   pv=struct.pack('<3840f',*values);rr=struct.pack('<3840f',*([0.0]*3840 if edge<3 else [-x for x in values]))
   assert evaluate(0,pv,rr)==evaluate(1,pv,rr),(i,edge,'finite boundary mismatch');checks+=1
  # Exact native FP8 O producer and bindings, same for parent and candidate.
  width=o['width'];assert o['rows']==[3840] and o['count']==1 and o['grid']==[120,1,1] and o['block']==[128,1,1]
  activ=d.buffer(width);activ.put(read(a.ordinary/f'a-{j}.bin',width));scale=d.buffer(4);scale.put(read(a.ordinary/f's-{j}.bin',4))
  weight=d.buffer(3840*width);weight.put(read(a.ordinary/f'w-{o["slot"]}-0.bin',3840*width));ws=d.buffer(7680);ws.put(read(a.ordinary/f'ws-{o["slot"]}-0.bin',7680))
  def produce():
   code=prod(activ.ptr,scale.ptr,weight.ptr,ws.ptr,projection.ptr,width,d.stream);assert code==0,code
  produce();d.synchronize();assert projection.get()==raw,(i,'native producer output mismatch');checks+=1
  for variant in [0,1]:
   residual.put(rv);produce();run(variant);d.synchronize();assert [post.get(),hidden.get(),packed.get(),scales.get()]==expected,(i,variant,'producer chain mismatch');checks+=1
  d.guards()
  if a.output:
   residual.put(rv);original.put(raw)
   # Warm GPU before this case's paired timing. Primary benchmark is separate.
   until=time.monotonic()+.15
   while time.monotonic()<until:
    for _ in range(16):produce();run(0)
    d.synchronize()
   isolated=paired(run,False);chain=paired(run,True)
   results.append({'layer':c['layer'],'phase':c['phase'],'decode_step':[0,566,1133][c['phase']],'ordinary_case':j,'norm_case':i,'width':width,'weight':8.0 if c['layer']==47 else 40.0/3,'bit_equal':True,'raw_projection_sha256':hashlib.sha256(raw).hexdigest(),'samples':isolated,'chain_samples':chain})
  d.guards();print('PASS case',i,'layer',c['layer'],'phase',c['phase'],'raw O/intermediates/alias/edges/producer',flush=True)
 if a.output:
  with a.output.open('x') as f:json.dump({'diagnostic_only':True,'candidate':'unchanged prior single fusion candidate','numeric_checks':checks,'timing':'unique per-invocation event pairs in CUDA graphs; raw restore before each isolated invocation outside event interval, then256MiB eviction if cold; chain includes unchanged native O producer and boundary; same graph controls for both variants','samples_per_case':25,'warm_invocations_per_graph':16,'cold_invocations_per_graph':1,'cases':results},f,indent=2)
 print('PASS',checks,'bit-exact actual/finite/alias/native-producer checks; device guards',flush=True)
finally:d.close()
