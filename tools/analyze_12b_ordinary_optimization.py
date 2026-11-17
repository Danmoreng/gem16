"""Account for12B T1 norm kernels and actual Gate/Up graph overlap."""
from pathlib import Path
import argparse,collections,json,sqlite3
from analyze_12b_ordinary_kernels import summarize,native_family
p=argparse.ArgumentParser(description=__doc__);p.add_argument('trace_root',type=Path);p.add_argument('output',type=Path);a=p.parse_args();results={}
for name in ['kernel-only','combined']:
 c=sqlite3.connect(f'file:{a.trace_root/(name+".sqlite")}?mode=ro',uri=True);c.row_factory=sqlite3.Row;n={r['id']:r['value'] for r in c.execute('select * from StringIds')};groups=collections.defaultdict(list)
 for r in c.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL order by start'):
  k={'start':r['start']/1000,'end':r['end']/1000,'name':n[r['demangledName']],'short':n[r['shortName']],'mangled':n[r['mangledName']],'correlation':r['correlationId'],'grid':[r['gridX'],r['gridY'],r['gridZ']],'block':[r['blockX'],r['blockY'],r['blockZ']],'registers':r['registersPerThread'],'shared':r['staticSharedMemory']+r['dynamicSharedMemory']};groups[k['correlation']].append(k)
 steps=[v for v in groups.values() if v[0]['short']=='ControlledEmbeddingKernel'];assert len(steps)==63;assert all(len(s)==820 for s in steps);selected=steps[30:46];chosen=[k for s in selected for k in s];correlations={k['correlation'] for k in chosen};overlaps=[]
 for step in selected:
  ff=[k for k in step if 'Sm120DirectProjectionKernel' in k['name'] and 'Fp8MatrixBinding' not in k['name'] and k['grid']==[480,1,1]];assert len(ff)==96
  for x,y in zip(ff[::2],ff[1::2]):overlaps.append(max(0,min(x['end'],y['end'])-max(x['start'],y['start'])))
 copies=[{'start':r['start']/1000,'end':r['end']/1000,'bytes':r['bytes']} for r in c.execute('select * from CUPTI_ACTIVITY_KIND_MEMCPY') if r['correlationId'] in correlations];first=min(k['start'] for k in chosen+copies);last=max(k['end'] for k in chosen+copies);apis=collections.defaultdict(list)
 for r in c.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME'):
  if first<=r['start']/1000<=last or r['correlationId'] in correlations:apis[n[r['nameId']]].append({'start':r['start']/1000,'end':r['end']/1000})
 out=summarize(chosen,copies,apis,first,last,native_family);out['gate_up_pairs']=len(overlaps);out['overlapping_gate_up_pairs']=sum(x>0 for x in overlaps);out['gate_up_overlap_ms_per_step']=sum(overlaps)/16/1000;out['mean_pair_overlap_us']=sum(overlaps)/len(overlaps);out['new_kernel_mangled_names']=sorted({k['mangled'] for k in chosen if 'TwelveBOrdinary' in k['name']});assert len(out['new_kernel_mangled_names'])==5
 if name=='kernel-only':assert not any(overlaps)
 else:assert any(overlaps),'explicit fork must produce observed overlap, not just extra stream APIs'
 assert sum(k['short']=='TwelveBOrdinaryPostAttentionKernel' for k in chosen)==16*48
 results[name]=out
 print(name,'span',out['span_ms_per_step'],'kernelunion',out['kernel_union_ms_per_step'],'overlappairs',out['overlapping_gate_up_pairs'],'/',len(overlaps),'overlap_ms/step',out['gate_up_overlap_ms_per_step'])
a.output.write_text(json.dumps({'status':'diagnostic_only_not_primary_throughput','sequence_lengths':list(range(16415,16431)),'results':results},indent=2)+'\n')
