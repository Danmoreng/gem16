from pathlib import Path
import sys,subprocess,json,os,shutil,sqlite3,collections
R=Path.cwd();O=Path(__file__).resolve().parent;P=R/'benchmarks/results/2026-11-17/12b-mtp-parity'
sys.path.insert(0,str(R/'tools'));from experiment_scratch import experiment_scratch
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
def run(name,c,env):
 (O/(name+'.command.json')).write_text(json.dumps(c,indent=2)+'\n')
 with (O/(name+'.stdout.log')).open('w') as out,(O/(name+'.stderr.log')).open('w') as err:subprocess.run(c,check=True,env=env,stdout=out,stderr=err,timeout=240)
with experiment_scratch('mtp-ordinary-only-trace',4) as S:
 name='d3-dispatch';c=json.load(open(P/'gpu-d3.command.json'));c[c.index('--executable')+1]=str(O/'gem16-run');c[c.index('--fixed-output-tokens')+1]='64';c[c.index('--warmups')+1]='0';c[c.index('--repetitions')+1]='1';c[c.index('--output')+1]=str(O/(name+'.json'));i=c.index('--gpu-telemetry-csv');del c[i:i+2]
 env=dict(os.environ,GEM16_12B_ORDINARY_NORMS='1',GEM16_12B_ORDINARY_OVERLAP='1',GEM16_12B_ORDINARY_O='1',GEM16_12B_ORDINARY_GLOBAL='1')
 run(name,['nsys','profile','--trace=cuda,nvtx,cublas','--sample=none','--cpuctxsw=none','--cuda-graph-trace=node','--wait=all','--output',str(S/name),*c],env)
 j=json.load(open(O/(name+'.json')));old=json.load(open(P/'gpu-d3.json'));assert j['representative_output_token_ids']==old['representative_output_token_ids'][:64]
 run(name+'-export',['nsys','export','--type','sqlite','--output',str(S/(name+'.sqlite')),str(S/(name+'.nsys-rep'))],env)
 db=sqlite3.connect(f'file:{S/(name+".sqlite")}?mode=ro',uri=True);names=dict(db.execute('select id,value from StringIds'));counts=collections.Counter(names[r[0]] for r in db.execute('select shortName from CUPTI_ACTIVITY_KIND_KERNEL'))
 required=['TwelveBOrdinaryOPipelineKernel','TwelveBOrdinaryGlobalGqa4Kernel','TwelveBOrdinaryInputNormFp8Kernel','TwelveBOrdinaryAttentionQuantizeKernel','TwelveBOrdinaryPostAttentionKernel','TwelveBOrdinaryPostMlpKernel','Bf16GemvExactTreeKernel','TwelveBPackedFp8ProjectionKernel']
 assert all(counts[k]>0 for k in required),{k:counts[k] for k in required}
 rejected=['TwelveBMtpPackedFp8OPipelineKernel','TwelveBMtpGlobalGqa4Kernel','TwelveBMtpInputNormFp8Kernel','Bf16GemvLoad4Kernel','TwelveBAssistantNormKernel']
 assert all(counts[k]==0 for k in rejected)
 detail=[]
 for row in db.execute('select demangledName,gridX,gridY,gridZ,blockX,registersPerThread,localMemoryPerThread from CUPTI_ACTIVITY_KIND_KERNEL'):
  if any(k in names[row[0]] for k in required):detail.append((names[row[0]],*row[1:]))
 m=j['runs'][0]['engine_result']['mtp'];summary={'diagnostic_not_primary_throughput':True,'all64output_ids_exact':True,'actual_dispatch_counts':{k:counts[k] for k in required},'rejected_dispatch_counts':{k:counts[k] for k in rejected},'ordinary_tail_steps':m['target_batches']-m['verification_groups'],'verification_groups':m['verification_groups'],'dispatch_geometries':sorted(set(detail))}
 (O/'dispatch-summary.json').write_text(json.dumps(summary,indent=2)+'\n');shutil.copyfile(S/(name+'.nsys-rep'),O/(name+'.nsys-rep'));db.close();print(json.dumps(summary),flush=True)
