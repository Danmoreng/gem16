#!/usr/bin/env python3
"""Read-only accounting of matched 12B Ordinary diagnostic GPU windows.

Nsight times are ns; PyTorch times are us. Counts are actual dispatched kernels,
not source operations. Synchronize API durations overlap device work.
"""
import argparse, collections, gzip, hashlib, json, pathlib, re, sqlite3

def union_us(events):
    total = 0.0
    end = float('-inf')
    for e in sorted(events, key=lambda x: x['start']):
        total += max(0.0, e['end'] - max(end, e['start']))
        end = max(end, e['end'])
    return total

def native_family(n):
    if 'Sm120DirectProjectionKernel' in n:
        return 'FP8 QKV/O' if 'Fp8MatrixBinding' in n else 'NVFP4 Gate/Up/Down'
    if 'Attention' in n and ('SplitOnline' in n or 'MergeOnline' in n): return 'attention'
    if 'OutputHead' in n or 'Candidates' in n: return 'head and argmax'
    if 'GatedGelu' in n: return 'FFN GELU/quant'
    if 'AppendKv' in n: return 'KV write'
    return 'norm/quant/rotary/round/embedding'

def llama_family(n):
    if 'mul_mat_vec_q' in n: return 'NVFP4 Gate/Up/Down' if '(ggml_type)40' in n else 'Q8 Q/K/V/O'
    if 'flash_attn' in n or 'dequantize_block_q8_0_f16' in n: return 'attention including KV dequant'
    if 'mul_mat_vec_f' in n or 'softcap' in n: return 'head and softcap'
    if 'k_set_rows' in n: return 'KV write'
    return 'norm/quant/rotary/FWHT/elementwise'

def vllm_family(n):
    if 'BlockScaled' in n: return 'NVFP4 GateUp/Down'
    if 'device_kernel' in n and 'e4m3' in n: return 'FP8 QKV/O'
    if n in ('kernel_unified_attention','reduce_segments'): return 'attention'
    if 'reshape_and_cache' in n: return 'KV write'
    if 'gelu_mul_scaled_fp4' in n: return 'FFN GELU/quant'
    if 'gemvx' in n: return 'BF16 head GEMV'
    if n.startswith('triton_') or 'cvt_fp16_to_fp4' in n: return 'norm/quant/rotary/embedding'
    return 'metadata/sampling/other'

def summarize(kernels, copies, apis, first, last, classify, steps=16):
    families=collections.defaultdict(list); signatures=collections.defaultdict(list)
    for k in kernels:
        families[classify(k['name'])].append(k)
        signatures[(k['name'],tuple(k['grid']),tuple(k['block']),k.get('registers'),k.get('shared'))].append(k)
    span=last-first; ku=union_us(kernels); allu=union_us(kernels+copies)
    return {'steps':steps,'window_start_us':first,'window_end_us':last,'span_ms_per_step':span/steps/1000,
      'kernel_union_ms_per_step':ku/steps/1000,'all_gpu_activity_union_ms_per_step':allu/steps/1000,
      'summed_kernel_ms_per_step':sum(k['end']-k['start'] for k in kernels)/steps/1000,
      'api_timing_note':'Host API durations overlap GPU work and must not be added to device costs. Profiler drain synchronization is diagnostic only.',
      'non_kernel_interval_ms_per_step':(span-ku)/steps/1000,
      'kernel_count':len(kernels),'kernels_per_step':len(kernels)/steps,
      'families':{f:{'count':len(es),'count_per_step':len(es)/steps,'sum_ms_per_step':sum(e['end']-e['start'] for e in es)/steps/1000} for f,es in sorted(families.items())},
      'copies':{'count':len(copies),'bytes':sum(e.get('bytes',0) for e in copies),'sum_ms_per_step':sum(e['end']-e['start'] for e in copies)/steps/1000},
      'apis':{n:{'count':len(es),'sum_ms':sum(e['end']-e['start'] for e in es)/1000} for n,es in apis.items()},
      'dispatches':[{'name':sig[0],'grid':sig[1],'block':sig[2],'registers':sig[3],'shared':sig[4],'count':len(es),'sum_ms_per_step':sum(e['end']-e['start'] for e in es)/steps/1000} for sig,es in sorted(signatures.items(),key=lambda p:-sum(e['end']-e['start'] for e in p[1]))]}

def nsight(path,engine,index):
    db=sqlite3.connect(f'file:{path}?mode=ro',uri=True);db.row_factory=sqlite3.Row
    names={r['id']:r['value'] for r in db.execute('select * from StringIds')}
    kernels=[]
    for r in db.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL order by start'):
        kernels.append({'start':r['start']/1000,'end':r['end']/1000,'name':names[r['demangledName']], 'short':names[r['shortName']], 'correlation':r['correlationId'],'graph':r['graphId'],'grid':[r['gridX'],r['gridY'],r['gridZ']],'block':[r['blockX'],r['blockY'],r['blockZ']],'registers':r['registersPerThread'],'shared':r['staticSharedMemory']+r['dynamicSharedMemory']})
    groups=collections.defaultdict(list)
    for k in kernels: groups[k['correlation']].append(k)
    if engine=='native':
        steps=[v for v in groups.values() if v[0]['short']=='ControlledEmbeddingKernel'];assert len(steps)==63
        assert all(len(v)==868 for v in steps)
        selected=steps[index:index+16]; correlations={v[0]['correlation'] for v in selected}
        chosen=[k for v in selected for k in v]
        first=min(k['start'] for k in chosen);last=max(k['end'] for k in chosen)
    else:
        # Last 64 head calls are prefill output followed by 63 decode outputs.
        # Startup warmup head calls precede them. Decode0 is eager; decode1..62
        # each replay six graph partitions, with stable actual node counts.
        heads=[k for k in kernels if k['short']=='mul_mat_vec_f'][-64:]; assert len(heads)==64
        replay=[v for v in groups.values() if v[0]['graph']==20];assert len(replay)==62
        parts=[v for v in groups.values() if len(v)>1 and v[0]['graph'] in (20,23,26,29,32,35)]
        assert len(parts)==62*6
        assert [len(v) for v in parts[:6]]==[257,257,257,254,258,69]
        assert all(heads[i+2]['correlation']==parts[i*6+5][0]['correlation'] for i in range(62))
        selected=parts[(index-1)*6:(index-1+16)*6];assert len(selected)==96
        correlations={v[0]['correlation'] for v in selected};chosen=[k for v in selected for k in v]
        first=min(k['start'] for k in chosen);last=max(k['end'] for k in chosen)
    copies=[]
    for r in db.execute('select * from CUPTI_ACTIVITY_KIND_MEMCPY'):
        if (engine=='native' and r['correlationId'] in correlations) or (engine=='llama' and r['start']/1000>=first and r['end']/1000<=last):
            copies.append({'start':r['start']/1000,'end':r['end']/1000,'bytes':r['bytes'],'kind':r['copyKind']})
    if copies:first=min(first,min(e['start'] for e in copies));last=max(last,max(e['end'] for e in copies))
    apis=collections.defaultdict(list)
    for r in db.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME'):
        start=r['start']/1000;end=r['end']/1000
        if first<=start<=last or r['correlationId'] in correlations:
            apis[names[r['nameId']]].append({'start':start,'end':end})
    out=summarize(chosen,copies,apis,first,last,native_family if engine=='native' else llama_family)
    out['decode_index_start_zero_based']=index;out['graph_partition_counts']=dict(collections.Counter(k['graph'] for k in chosen));out['selected_correlations']=sorted(correlations)
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('trace_root',type=pathlib.Path);p.add_argument('output',type=pathlib.Path);a=p.parse_args()
    events=json.load(gzip.open(next((a.trace_root/'vllm-traces').glob('*.json.gz'))))['traceEvents']
    annotations=[e for e in events if e.get('cat')=='user_annotation' and e.get('name','').startswith('execute_')]
    seq=[int(re.search(r'generation_1\(sq1sk(\d+)',e['name'])[1]) for e in annotations]
    assert len(seq)==16 and seq==list(range(seq[0],seq[0]+16));assert all('context_0' in e['name'] for e in annotations)
    index=seq[0]-16385;assert 1<=index<=47
    kernels=[];copies=[];apis=collections.defaultdict(list)
    for e in events:
        if e.get('ph')!='X':continue
        k={'start':e['ts'],'end':e['ts']+e['dur'],'name':e['name']};args=e.get('args',{})
        if e.get('cat')=='kernel':
            k.update(grid=args['grid'],block=args['block'],registers=args.get('registers per thread'),shared=args.get('shared memory'));kernels.append(k)
        elif e.get('cat') in ('gpu_memcpy','gpu_memset'):k['bytes']=args.get('bytes',0);copies.append(k)
        elif e.get('cat')=='cuda_runtime':apis[e['name']].append(k)
    first=min(k['start'] for k in kernels+copies);last=max(k['end'] for k in kernels+copies)
    v=summarize(kernels,copies,apis,first,last,vllm_family);v['sequence_lengths']=seq;v['graph_counts']=dict(collections.Counter(e['args'].get('graph id') for e in events if e.get('cat')=='kernel'))
    assert sum(1 for k in kernels if k['name']=='kernel_unified_attention')==16*48
    out={'status':'diagnostic_profile_not_primary_throughput','sequence_lengths':seq,'boundary':'first GPU activity through last GPU activity of 16 complete decode steps; includes 15 inter-step gaps; profiler timings are not primary throughput','native':nsight(a.trace_root/'native.sqlite','native',index),'llama':nsight(a.trace_root/'llama.sqlite','llama',index),'vllm':v}
    a.output.write_text(json.dumps(out,indent=2)+'\n')
    for engine in ['native','llama','vllm']:
        r=out[engine];print(engine,{k:r[k] for k in ['span_ms_per_step','kernel_union_ms_per_step','non_kernel_interval_ms_per_step','kernels_per_step']});print(json.dumps(r['families'],indent=2))
if __name__=='__main__':main()
