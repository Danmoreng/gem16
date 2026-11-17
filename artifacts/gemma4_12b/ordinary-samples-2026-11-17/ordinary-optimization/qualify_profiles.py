from pathlib import Path
import subprocess,json,os,tempfile,gzip,hashlib,shutil
R=Path.cwd();O=Path(__file__).resolve().parent;B=R/'benchmarks/results/2026-11-17/ordinary-kernel-comparison';T=Path(tempfile.mkdtemp(prefix='gem16-ordinary-optimized-',dir='/tmp'))
(O/'profile-storage.json').write_text(json.dumps({'temporary_root':str(T)},indent=2)+'\n')
def run(name,cmd,env=None,timeout=600):
 (O/(name+'.command.json')).write_text(json.dumps(cmd,indent=2)+'\n')
 with (O/(name+'.stdout.log')).open('x') as out,(O/(name+'.stderr.log')).open('x') as err:subprocess.run(cmd,check=True,env=env,stdout=out,stderr=err,timeout=timeout)
 print('PASS',name,flush=True)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
run('actual-norms',['python',str(R/'tests/cuda/twelve_b_ordinary_actual_norms.py'),str(R/'benchmarks/results/2026-11-15/58febf4/12b-five-step-goal/norm-fixtures'),str(O/'actual-norm-lab.so'),'--output',str(O/'actual-norms.json')])
# The bounded diagnostic's math is exactly the same prefix as the full primary.
for name in ['kernel-only','combined']:
 c=json.load(open(O/(name+'.command.json')));c[c.index('--warmups')+1]='0';c[c.index('--repetitions')+1]='1';c[c.index('--fixed-output-tokens')+1]='64';c[c.index('--output')+1]=str(O/(name+'-profile.json'));i=c.index('--gpu-telemetry-csv');del c[i:i+2]
 env=dict(os.environ,**json.load(open(O/(name+'.environment.json'))))
 cmd=['nsys','profile','--trace=cuda,nvtx,cublas','--sample=none','--cpuctxsw=none','--cuda-graph-trace=node','--wait=all','--output',str(T/name),*c]
 run(name+'-profile',cmd,env)
 d=json.load(open(O/(name+'-profile.json')));parent=json.load(open(B/'native.json'));assert d['representative_output_token_ids']==parent['representative_output_token_ids'][:64];d['status']='diagnostic_only_profiled_not_primary_throughput';(O/(name+'-profile.json')).write_text(json.dumps(d,indent=2)+'\n')
 run(name+'-export',['nsys','export','--type','sqlite','--output',str(T/(name+'.sqlite')),str(T/(name+'.nsys-rep'))])
# Functional12B MTP regressions only, not new throughput controls.
for depth in [2,3,4]:
 c=json.load(open(R/f'benchmarks/results/2026-11-17/12b-mtp-parity/gpu-d{depth}.command.json'))
 c[c.index('--executable')+1]=str(O/'candidate/gem16-run');c[c.index('--fixed-output-tokens')+1]='64';c[c.index('--warmups')+1]='0';c[c.index('--repetitions')+1]='1';c[c.index('--output')+1]=str(O/f'mtp-d{depth}-regression.json')
 if '--gpu-telemetry-csv' in c:i=c.index('--gpu-telemetry-csv');del c[i:i+2]
 run(f'mtp-d{depth}-regression',c)
 d=json.load(open(O/f'mtp-d{depth}-regression.json'));mtp_parent=json.load(open(R/f'benchmarks/results/2026-11-17/12b-mtp-parity/gpu-d{depth}.json'));assert d['representative_output_token_ids']==mtp_parent['representative_output_token_ids'][:64];assert not d['runs'][0]['engine_result']['fallbacks'] and not d['runs'][0]['engine_result']['token_loop_allocations']
 d['status']='functional_regression_only_not_primary_throughput';(O/f'mtp-d{depth}-regression.json').write_text(json.dumps(d,indent=2)+'\n')
r=O/'raw-traces';r.mkdir();manifest=[]
for src in T.iterdir():
 if not src.is_file():continue
 dst=r/(src.name+'.gz' if src.suffix=='.sqlite' else src.name);data=src.read_bytes();sha=hashlib.sha256(data).hexdigest()
 if src.suffix=='.sqlite':
  with gzip.open(dst,'wb',compresslevel=6) as f:f.write(data)
  assert hashlib.sha256(gzip.open(dst,'rb').read()).hexdigest()==sha
 else:shutil.copyfile(src,dst)
 manifest.append({'path':str(dst.relative_to(R)),'original_sha256':sha,'stored_sha256':hashlib.sha256(dst.read_bytes()).hexdigest(),'original_bytes':len(data),'stored_bytes':dst.stat().st_size})
(O/'retained-traces.json').write_text(json.dumps(manifest,indent=2)+'\n')
