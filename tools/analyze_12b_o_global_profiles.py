"""Read-only matched16-step12B Ordinary O/global profile accounting."""
from pathlib import Path
import argparse,collections,json,sqlite3
from analyze_12b_ordinary_kernels import summarize,native_family

def category(k):
 n=k['name']
 if 'TwelveBOrdinaryOPipelineKernel' in n:return 'FP8 O'
 if 'Sm120DirectProjectionKernel' in n and 'Fp8MatrixBinding' in n:return 'FP8 O' if k['grid'][2]==1 else 'FP8 QKV'
 if 'TwelveBOrdinaryGlobalGqa4Kernel' in n or 'SplitOnlineDecodeAttentionFp8GlobalGqaKernel' in n:return 'global attention split'
 if 'MergeOnlineDecodeAttentionKernel' in n and '(int)512' in n:return 'global attention merge'
 if 'SplitOnlineDecodeAttentionFp8Kernel' in n or 'MergeOnlineDecodeAttentionKernel' in n:return 'local attention split/merge'
 return native_family(n)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('trace_root',type=Path);p.add_argument('output',type=Path);a=p.parse_args();result={}
 for name in ['parent','combined']:
  c=sqlite3.connect(f'file:{a.trace_root/(name+".sqlite")}?mode=ro',uri=True);c.row_factory=sqlite3.Row;n={r['id']:r['value'] for r in c.execute('select * from StringIds')};groups=collections.defaultdict(list)
  for r in c.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL order by start'):
   k={'start':r['start']/1000,'end':r['end']/1000,'name':n[r['demangledName']],'short':n[r['shortName']],'mangled':n[r['mangledName']],'correlation':r['correlationId'],'grid':[r['gridX'],r['gridY'],r['gridZ']],'block':[r['blockX'],r['blockY'],r['blockZ']],'registers':r['registersPerThread'],'shared':r['staticSharedMemory']+r['dynamicSharedMemory'],'local_bytes_per_thread':r['localMemoryPerThread']};groups[k['correlation']].append(k)
  steps=[v for v in groups.values() if v[0]['short']=='ControlledEmbeddingKernel'];assert len(steps)==63 and all(len(v)==820 for v in steps)
  selected=steps[30:46];chosen=[k for v in selected for k in v];correlations={k['correlation'] for k in chosen};copies=[{'start':r['start']/1000,'end':r['end']/1000,'bytes':r['bytes']} for r in c.execute('select * from CUPTI_ACTIVITY_KIND_MEMCPY') if r['correlationId'] in correlations];first=min(k['start'] for k in chosen+copies);last=max(k['end'] for k in chosen+copies);apis=collections.defaultdict(list)
  for r in c.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME'):
   if first<=r['start']/1000<=last or r['correlationId'] in correlations:apis[n[r['nameId']]].append({'start':r['start']/1000,'end':r['end']/1000})
  out=summarize(chosen,copies,apis,first,last,native_family);families=collections.defaultdict(list)
  for k in chosen:families[category(k)].append(k)
  out['families']={f:{'count':len(v),'count_per_step':len(v)/16,'sum_ms_per_step':sum(k['end']-k['start'] for k in v)/16000} for f,v in families.items()}
  assert out['families']['FP8 O']['count']==768 and out['families']['global attention split']['count']==128
  if name=='combined':
   new=[k for k in chosen if 'TwelveBOrdinaryOPipelineKernel' in k['name'] or 'TwelveBOrdinaryGlobalGqa4Kernel' in k['name']];assert len(new)==896 and all(k['local_bytes_per_thread']==0 for k in new)
   assert all(k['grid']==[35,4,1] for k in new if 'GlobalGqa4' in k['name'])
   out['actual_new_mangled_names']=sorted({k['mangled'] for k in new});assert len(out['actual_new_mangled_names'])==2
  out['decode_indices_zero_based']=[30,45];out['sequence_lengths']=list(range(16415,16431));result[name]=out
  print(name,'GPU union ms/step',out['kernel_union_ms_per_step'],'families',{f:round(v['sum_ms_per_step'],4) for f,v in out['families'].items()},flush=True)
 a.output.write_text(json.dumps({'status':'diagnostic_only_not_primary_throughput','boundary':'first through last GPU activity of16 complete steps;15 inter-step gaps; sums may overlap','profiles':result},indent=2)+'\n')
if __name__=='__main__':main()
