"""Validated actual-input Assistant GEMV replay for the bounded Pro experiment."""
from pathlib import Path
import argparse
import ctypes as C
import json
import random
import struct
from twelve_b_device_lab import Device,P,U,Q,Z,read,finite

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('directory',type=Path);p.add_argument('library',type=Path)
p.add_argument('--mode',choices=['numeric','timing'],default='numeric')
p.add_argument('--output',type=Path)
p.add_argument('--actual-only',action='store_true',help='skip finite arithmetic repeats for value-independent race topology')
p.add_argument('--progress',action='store_true')
a=p.parse_args()
cases=json.loads((a.directory/'captures.json').read_text())['cases']
assert len(cases)==126
lib=C.CDLL(str(a.library.resolve()));fn=lib.gem16_12b_assistant
fn.argtypes=[U,P,P,P,Q,Q,P];fn.restype=C.c_int
evict=lib.gem16_12b_assistant_evict;evict.argtypes=[P,Z,P];evict.restype=C.c_int
d=Device();activation=d.buffer(8192*4);weight=d.buffer(8192*1024*2);output=d.buffer(8192*4)
tests=0;result=[];canary=bytes([165])*output.n
try:
    eviction=d.buffer(256*1024*1024) if a.mode=='timing' else None
    start=P();end=P()
    if a.mode=='timing':
        d.call('cudaEventCreate',C.byref(start));d.call('cudaEventCreate',C.byref(end))
        d.events.extend([start,end])
    for i,c in enumerate(cases):
        rows,width=c['rows'],c['width']
        assert (rows,width) in {(4096,1024),(1024,4096),(8192,1024),(1024,8192),(3840,1024)}
        w=read(a.directory/f'w-{c["slot"]}.bin',rows*width*2)
        weight.put(w+bytes(weight.n-len(w)))
        actual_input=read(a.directory/f'a-{i}.bin',width*4)
        padded_input=actual_input+bytes(activation.n-len(actual_input));activation.put(padded_input)
        expected=read(a.directory/f'expected-{i}.bin',rows*4)
        def launch(v):
            code=fn(v,activation.ptr,weight.ptr,output.ptr,rows,width,d.stream)
            assert code==0,code
        def checked(v):
            output.put(canary);launch(v);d.synchronize()
            entire=output.get();out=entire[:rows*4];finite(out)
            assert entire[rows*4:]==canary[rows*4:],(i,'logical output tail overwritten')
            return out
        assert checked(0)==expected and checked(1)==expected,(i,'actual mismatch');tests+=2
        if a.mode=='numeric' and not a.actual_only:
            for edge in range(5):
                rng=random.Random(1200500+edge)
                values=[0.0 if edge==0 else 1.0 if edge==1 else
                        (-1.0 if j%2 else 1.0)*1000.0 if edge==2 else
                        2.0**-120 if edge==3 else
                        rng.choice((-1.00390625,-1,0,1,1.00390625))*rng.choice((.001,1,1024))
                        for j in range(width)]
                packed=struct.pack(f'<{width}f',*values)
                activation.put(packed+bytes(activation.n-len(packed)))
                assert checked(0)==checked(1),(i,edge,'finite/rounding/cancellation mismatch');tests+=1
            activation.put(padded_input)
        elif a.mode=='timing':
            # Input/weights are immutable in GEMV; each repetition sees the
            # original captured input. Eviction follows data upload and lies
            # outside the timed graph interval.
            graphs=[[d.graph(lambda v=v:launch(v),n) for v in [0,1]] for n in [16,1]]
            for _ in range(4):
                for graph in graphs[0]:d.call('cudaGraphLaunch',graph,d.stream)
            d.synchronize()
            def measure(v,cold):
                if cold:assert evict(eviction.ptr,eviction.n,d.stream)==0
                d.call('cudaEventRecord',start,d.stream)
                d.call('cudaGraphLaunch',graphs[int(cold)][v],d.stream)
                d.call('cudaEventRecord',end,d.stream);d.call('cudaEventSynchronize',end)
                ms=C.c_float();d.call('cudaEventElapsedTime',C.byref(ms),start,end)
                return ms.value*1000/(1 if cold else 16)
            samples=[]
            for j in range(25):
                sample=[0.0]*4
                for cold in [False,True]:
                    for v in ([0,1] if j%2==0 else [1,0]):sample[2*int(cold)+v]=measure(v,cold)
                samples.append(sample)
            result.append({**c,'samples':samples})
        assert activation.get()==padded_input,(i,'input mutated')
        d.guards()
        if a.progress:print('CASE',i,'PASS',flush=True)
    if a.output:
        with a.output.open('x') as f:
            json.dump({'diagnostic_only':True,'mode':a.mode,'checks':tests,'cases':result,
                       'weighting':'all42 real calls per group at3equal phases, summed operator latency',
                       'cache':'16warm invocations or1invocation after256MiB read-modify-write eviction; eviction outside timing'},f,indent=2)
    print('PASS',tests,'bit-exact actual/boundary checks; finite, logical tails, immutable inputs, guards',flush=True)
finally:d.close()
