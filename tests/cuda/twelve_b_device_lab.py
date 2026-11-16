"""Explicit diagnostic CUDA buffers and paired graph timing; never inference code."""
from __future__ import annotations
import ctypes as C
import math
from pathlib import Path
import statistics
import time

P=C.c_void_p;U=C.c_uint;Z=C.c_size_t;F=C.c_float;Q=C.c_uint64
class Dim3(C.Structure):
    _fields_=[('x',U),('y',U),('z',U)]
class Binding(C.Structure):
    _fields_=[('weight',P),('weight_scales',P),('output',P),('rows',Q)]
assert C.sizeof(Binding)==32
class Device:
    def __init__(self):
        self.lib=C.CDLL('/opt/cuda/lib64/libcudart.so.13');self.buffers=[];self.graphs=[];self.events=[];self.stream=P();self.evict=None;self.warmed=False
        self.call('cudaStreamCreate',C.byref(self.stream))
    def call(self,name,*args):
        code=getattr(self.lib,name)(*args)
        if code:raise RuntimeError(f'{name}: CUDA error {code}')
    def buffer(self,n):
        b=Buffer(self,n);self.buffers.append(b);return b
    def graph(self,run,n):
        graph=P();execute=P();self.call('cudaStreamBeginCapture',self.stream,U(1))
        for _ in range(n):run()
        self.call('cudaStreamEndCapture',self.stream,C.byref(graph));self.call('cudaGraphInstantiateWithFlags',C.byref(execute),graph,Q(0));self.call('cudaGraphDestroy',graph);self.graphs.append(execute);return execute
    def synchronize(self):self.call('cudaStreamSynchronize',self.stream)
    def guards(self):
        for b in self.buffers:b.guards()
    def close(self):
        self.synchronize()
        for g in self.graphs:self.call('cudaGraphExecDestroy',g)
        for e in self.events:self.call('cudaEventDestroy',e)
        for b in reversed(self.buffers):b.close()
        self.call('cudaStreamDestroy',self.stream)
    def paired(self,run,samples=25):
        # Same graph-based timing as the native observer; cold memset is excluded
        # from the timed interval. It deliberately touches256MiB before each call.
        if self.evict is None:self.evict=self.buffer(256*1024*1024)
        evict=self.evict
        graphs=[[self.graph(lambda v=v:run(v),n) for v in (0,1)] for n in (16,1)]
        if not self.warmed:
            until=time.monotonic()+1.0
            while time.monotonic()<until:
                for _ in range(32):self.call('cudaGraphLaunch',graphs[0][0],self.stream)
                self.synchronize()
            self.warmed=True
        start=P();end=P();self.call('cudaEventCreate',C.byref(start));self.call('cudaEventCreate',C.byref(end));self.events.extend((start,end))
        for _ in range(4):
            for g in graphs[0]:self.call('cudaGraphLaunch',g,self.stream)
        self.synchronize()
        def measure(v,cold):
            if cold:self.call('cudaMemsetAsync',evict.ptr,C.c_int(123),Z(evict.n),self.stream)
            self.call('cudaEventRecord',start,self.stream);self.call('cudaGraphLaunch',graphs[int(cold)][v],self.stream);self.call('cudaEventRecord',end,self.stream);self.call('cudaEventSynchronize',end)
            ms=F();self.call('cudaEventElapsedTime',C.byref(ms),start,end);return ms.value*1000/(1 if cold else 16)
        result=[]
        for j in range(samples):
            row=[0.0]*4
            for cold in (False,True):
                for v in ((0,1) if j%2==0 else (1,0)):row[2*int(cold)+v]=measure(v,cold)
            result.append(row)
        evict.guards();return result
class Buffer:
    def __init__(self,d,n):
        assert 0<n<=1024*1024*1024;self.d=d;self.n=n;self.base=P();d.call('cudaMalloc',C.byref(self.base),Z(n+512));self.ptr=P(self.base.value+256);d.call('cudaMemset',self.base,C.c_int(165),Z(n+512))
    def put(self,data):
        assert len(data)==self.n;src=C.create_string_buffer(data);self.d.call('cudaMemcpy',self.ptr,C.cast(src,P),Z(self.n),C.c_int(1))
    def get(self):
        dst=C.create_string_buffer(self.n);self.d.call('cudaMemcpy',C.cast(dst,P),self.ptr,Z(self.n),C.c_int(2));return dst.raw
    def guards(self):
        for ptr in (self.base,P(self.ptr.value+self.n)):
            dst=C.create_string_buffer(256);self.d.call('cudaMemcpy',C.cast(dst,P),ptr,Z(256),C.c_int(2));assert dst.raw==bytes([165])*256,'device guard modified'
    def close(self):self.d.call('cudaFree',self.base)
def read(path,n):
    path=Path(path);assert not path.is_symlink() and path.is_file() and path.stat().st_size==n,(path,n);return path.read_bytes()
def finite(data):
    assert len(data)%4==0
    values=C.cast(C.create_string_buffer(data),C.POINTER(F))
    assert all(math.isfinite(values[i]) for i in range(len(data)//4)),'nonfinite output'
