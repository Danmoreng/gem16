"""Export only confirmed, matched-chunk12B Prefill samples and a concise chart."""
from pathlib import Path
import argparse,csv,hashlib,json,re,statistics

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('raw',type=Path);p.add_argument('destination',type=Path)
a=p.parse_args();raw=a.raw.resolve();dest=a.destination.resolve()
def load(p):return json.loads(p.read_text())
def identity(p):
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return {'bytes':p.stat().st_size,'sha256':h}
def clean(x):
 if isinstance(x,dict):return {clean(k):clean(v) for k,v in x.items()}
 if isinstance(x,list):return [clean(v) for v in x]
 if isinstance(x,str):
  for old,new in [(str(ROOT),'$ROOT'),(str(Path.home()/'.cache/huggingface'),'$HF_CACHE'),(str(Path.home()/'Development/_engines/llama.cpp'),'$CENTRAL_LLAMA_CPP'),(str(Path.home()),'$HOME')]:x=x.replace(old,new)
  assert not re.search(r'/home/|/Users/|GPU-[0-9a-f]{8}',x),x
 return x
import numpy as np
rng=np.random.default_rng(121115)
inputs=load(raw/'inputs.json');reference=load(raw/'confirmed-reference.json');cppraw=raw/reference['directory']
inputs['initial_screen_reference']={k:inputs[k] for k in ('central_commit','server','central_libraries')}
inputs.update(central_commit=reference['commit'],server=reference['server'],central_libraries=reference['libraries'])
records=[]
for engine in ('gem16','llama-cpp','vllm'):
 for chunk in (512,1024,2048,4096,8192):
  label=f'confirm-{engine}-{chunk}'+('-mem94' if engine=='vllm' else '')
  pointdir=cppraw if engine=='llama-cpp' else raw
  path=pointdir/(label+'.json');d=load(path);cfg=d['configuration']
  assert cfg['warmups']==3 and cfg['measured_repetitions']==10 and len(d['runs'])==10
  assert cfg['context_tokens_requested']==17519 and d['summary']['deterministic_outputs']
  assert d['workload']['prompt_tokens']==16384 and d['workload']['max_new_tokens']==1
  samples=d['runs'];assert all((r['generated_tokens'],r['measured_decode_intervals'])==(1,0) for r in samples)
  for r in samples:assert abs(r['prompt_tokens_per_second']-16384000/r['prompt_ms'])<1e-7
  rates=np.array([r['prompt_tokens_per_second'] for r in samples]);boot=np.median(rng.choice(rates,(10000,10),replace=True),axis=1)
  telemetry=list(csv.reader((pointdir/(label+'-telemetry.csv')).open()))
  rows=telemetry[1:];active=[r for r in rows if float(r[5].strip().split()[0])>=80]
  stats={'sampled_peak_device_total_mib':max(float(r[6].strip().split()[0]) for r in rows)}
  if active:
   for name,i in [('active_median_power_w',1),('active_median_sm_clock_mhz',2),('active_median_memory_clock_mhz',3),('active_median_temperature_c',4)]:stats[name]=statistics.median(float(r[i].strip().split()[0]) for r in active)
  rec={'engine':engine,'chunk':chunk,'median_prefill_tokens_per_second':float(np.median(rates)),'median_bootstrap_95':[float(x) for x in np.quantile(boot,[.025,.975])],'median_prefill_ms':statistics.median(r['prompt_ms'] for r in samples),'configuration':cfg,'runtime':d['runtime'],'samples':[{k:r[k] for k in ('prompt_ms','prompt_tokens_per_second','output_token_sha256','first_output_token_id','generated_tokens','measured_decode_intervals')} for r in samples],'output_token_ids':d['representative_output_token_ids'],'telemetry':stats,'raw_result':identity(path),'command_template':load(pointdir/(label+'.command.json'))}
  if engine=='gem16':
   assert all(r['engine_result']['prefill_chunk_tokens']==chunk and not r['engine_result']['fallbacks'] and not r['engine_result']['token_loop_allocations'] for r in samples)
   e=samples[0]['engine_result'];rec['native_plan']={k:e[k] for k in ('prefill_chunk_tokens','weight_arena_bytes','workspace_bytes','kv_cache_bytes','precision','kv_cache_storage','fp8_prefill_tile','fp8_prefill_schedule','nvfp4_gate_up_prefill_tile','nvfp4_down_prefill_tile')}
  elif engine=='llama-cpp':
   assert cfg['batch_size']==cfg['ubatch_size']==chunk and cfg['tensor_overrides']==['token_embd.weight=CUDA0']
   assert d['runtime']['llama_cpp_commit']==inputs['central_commit']
   log=(pointdir/(label+'.server.log')).read_text();b=re.findall(r'\bn_batch\s*=\s*(\d+)',log);u=re.findall(r'\bn_ubatch\s*=\s*(\d+)',log)
   assert b and u and set(map(int,b+u))=={chunk}
   progress=list(map(int,re.findall(r'cached n_tokens\s*=\s*(\d+)',log)))
   assert progress and all(v%chunk==0 for v in progress)
   assert 'offloaded 49/49 layers to GPU' in log
   rec['observed_batch_plan']={'logical_batch':chunk,'physical_microbatch':chunk,'progress_values':progress,'gpu_embedding_override':True,'all_layers_offloaded':'offloaded 49/49 layers to GPU' in log}
  else:
   assert cfg['max_num_batched_tokens_requested']==chunk and cfg['gpu_memory_utilization']==.94
   log=(raw/(label+'.stdout.log')).read_text();assert f'Chunked prefill is enabled with max_num_batched_tokens={chunk}' in log
   measured=log[log.index('measured 1/10'):]
   rec['measured_jit_warning']='JIT compilation during inference' in measured
   assert not rec['measured_jit_warning'],'measured sample includes JIT warning'
   rec['default_tactic_warning']='No tuned config covers fp4_gemm' in (raw/(label+'.stderr.log')).read_text()
  records.append(rec)
assert len(records)==15
assert all(r['output_token_ids']==records[0]['output_token_ids'] for r in records), 'first output differs across engines/chunks'
result={'schema_version':1,'date':'2026-11-15','scope':'confirmed12B Prefill chunk curve; no Decode speed claim','protocol':{'prompt_tokens':16384,'requested_context':17519,'requested_output_tokens':1,'warmups':3,'retained_per_point':10,'sequences':1,'prefix_reuse':False,'metric':'prompt processing through first selected token, initialization excluded; engine timing boundaries retain their documented differences','uncertainty':'median bootstrap percentile95%,10000 resamples,seed121115; same-day single-GPU cohorts, not hardware-general confidence'},'inputs':inputs,'records':records,'best_confirmed':{e:max([r for r in records if r['engine']==e],key=lambda r:r['median_prefill_tokens_per_second'])['chunk'] for e in ('gem16','llama-cpp','vllm')},'limitations':['Native/vLLM mixed FP8/NVFP4 target versus Q8/NVFP4 converted GGUF; KV FP8 versus Q8_0; batching match does not establish numerical/quality equivalence.','llama logical batch and physical microbatch both equal chunk; GPU token embedding placement is explicit at all points.','vLLM uses94% budget consistently; the original90%8192 failure is retained separately and does not prove a physical16GB limit.','Native CLI initializes per sample (initialization excluded); references keep their models within a cohort, so cache and thermal history can differ.','One-output probes preserve17,519 requested context; full1135-output native boundary/default/D2 checks are separate and bit-equal.','Best means fastest confirmed among the tested512/1024/2048/4096/8192 settings, not an exhaustive engine tuning search.']}
result['native_full_output_boundaries']={}
for chunk in (512,1024,2048,4096,8192):
 name=f'boundary-gem16-{chunk}.json' if chunk in (512,2048,4096) else f'preflight-native-ordinary-{chunk if chunk==1024 else "default"}-1135.json'
 d=load(raw/name);result['native_full_output_boundaries'][str(chunk)]={'output_sha256':d['runs'][0]['output_token_sha256'],'tokens':len(d['representative_output_token_ids']),'raw_result':identity(raw/name)}
assert all(v['tokens']==1135 for v in result['native_full_output_boundaries'].values())
assert len({v['output_sha256'] for v in result['native_full_output_boundaries'].values()})==1
result['final_control_sources']={name:identity(ROOT/name) for name in ('src/cuda/engine/prefill_chunk_plan.h','src/cuda/engine/inference_engine.cu','src/cuda/engine/inference_engine_initialize.cuh','tools/benchmark_wikipedia_workload.py','tests/python/test_benchmark_wikipedia_workload.py','tests/cuda/twelve_b_prefill_chunk_plan_host.cpp')}
for path in sorted(raw.glob('observed-vllm-*.json')):result.setdefault('scheduler_observations',{})[path.stem]=load(path)
result['failed90_percent_attempt']=load(raw/'screen-vllm-8192.failure.json')
result=clean(result);dest.mkdir(parents=True,exist_ok=True)
serialized=json.dumps(result,indent=2,ensure_ascii=False)+'\n'
if (dest/'results.json').exists():
 assert (dest/'results.json').read_text()==serialized, 'refusing to replace different recorded measurements'
else:
 with (dest/'results.json').open('x') as f:f.write(serialized)
with (dest/'results.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['engine','chunk_tokens','median_prefill_tokens_per_second','lower95','upper95','median_prompt_ms','sampled_peak_device_total_mib'])
 for r in records:w.writerow([r['engine'],r['chunk'],r['median_prefill_tokens_per_second'],*r['median_bootstrap_95'],r['median_prefill_ms'],r['telemetry']['sampled_peak_device_total_mib']])
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.fonttype':'none'})
fig,ax=plt.subplots(figsize=(12.8,7.2));fig.subplots_adjust(left=.10,right=.85,bottom=.15,top=.76)
colors={'gem16':'#007f86','llama-cpp':'#d55e00','vllm':'#7251b5'}
labels={'gem16':'GemSixteen · FP8/NVFP4 · 9.30 GB','llama-cpp':'llama.cpp · Q8/NVFP4 GGUF · 9.37 GB','vllm':'vLLM · FP8/NVFP4 · 9.30 GB'}
for e in colors:
 rows=[r for r in records if r['engine']==e];xs=[r['chunk'] for r in rows];ys=[r['median_prefill_tokens_per_second'] for r in rows];lo=[r['median_bootstrap_95'][0] for r in rows];hi=[r['median_bootstrap_95'][1] for r in rows]
 ax.plot(xs,ys,color=colors[e],lw=2.6,marker='o',ms=7,label=labels[e]);ax.fill_between(xs,lo,hi,color=colors[e],alpha=.13)
 ax.annotate(f'{ys[-1]:,.0f}',(xs[-1],ys[-1]),xytext=(13,0),textcoords='offset points',va='center',color=colors[e],fontsize=13,weight='bold')
ax.set_xscale('log',base=2);ax.set_xticks([512,1024,2048,4096,8192],['512','1,024','2,048','4,096','8,192']);ax.set_xlim(430,10200)
ax.set_xlabel('Prefill chunk (tokens)',labelpad=13);ax.set_ylabel('Prefill throughput (tokens/s)',labelpad=12);ax.grid(axis='y',color='#e6e9ed',lw=.8);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
fig.suptitle('Gemma 4 12B IT — Prefill',x=.10,y=.96,ha='left',fontsize=23,weight='bold');fig.text(.10,.90,'RTX 5080 Laptop · 16,384 prompt tokens',fontsize=12,color='#505963')
ax.legend(loc='lower left',bbox_to_anchor=(-.01,1.03),frameon=False,fontsize=11,ncol=1,handlelength=2.4,labelspacing=.7)
for ext in ('png',):fig.savefig(dest/f'prefill-chunks.{ext}',dpi=200,facecolor='white')
plt.close(fig)
(dest/'chart-notes.txt').write_text('Confirmed medians:3 warmups+10 retained per point. Shading:bootstrap95% CI of the median. One output token;17,519 requested context. No prefix reuse. Native/vLLM target9.304966064GB; converted GGUF9.366658208GB. KV FP8 versus Q8_0. vLLM0.33.0 uses94% GPU budget throughout. llama.cpp revision '+inputs['central_commit']+'. Full method and samples are in results.json; initialization excluded. Best is among the tested chunk settings.\n')
print('WROTE',dest.relative_to(ROOT),'15 confirmed points',result['best_confirmed'],flush=True)
