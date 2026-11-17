from pathlib import Path
import subprocess,json,os,sys,hashlib,shutil
R=Path.cwd();O=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'tools'));from experiment_scratch import experiment_scratch

def run(name,cmd,env=None,timeout=600):
 (O/(name+'.command.json')).write_text(json.dumps(cmd,indent=2)+'\n')
 with (O/(name+'.stdout.log')).open('w') as out,(O/(name+'.stderr.log')).open('w') as err:subprocess.run(cmd,check=True,env=env,stdout=out,stderr=err,timeout=timeout)
 print('PASS',name,flush=True)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip();assert shutil.disk_usage(R).free>2*1024**3
run('host-unit-tests',[str(R/'build/Linux/blackwell-release/bin/gem16-unit-tests')])
with experiment_scratch('ordinary-o-global-qualify',expected_gib=4) as S:
 run('extended-generate',[sys.executable,str(R/'tests/cuda/build_12b_ordinary_o_global.py'),str(S),'--o-pipeline'])
 cmd=['/opt/cuda/bin/nvcc','-std=c++20','-O3','-rdc=true','-shared','-Xcompiler=-fPIC','-gencode=arch=compute_120a,code=sm_120a','-I'+str(R/'include'),'-I'+str(R/'src'),'-I'+str(R/'tests/cuda'),str(S/'attention.cu'),'-o',str(S/'global.so')];run('extended-build',cmd)
 run('global-extended-numeric',[sys.executable,str(R/'tests/cuda/twelve_b_ordinary_o_global_replay.py'),'--family','global','--fixtures',str(R/'benchmarks/results/2026-11-15/58febf4/12b-pro-followup/e5/fixtures'),'--library',str(S/'global.so'),'--output',str(O/'global-extended-numeric.json')])
 for name in ['parent','combined']:
  c=json.load(open(O/(name+'.command.json')));c[c.index('--warmups')+1]='0';c[c.index('--repetitions')+1]='1';c[c.index('--fixed-output-tokens')+1]='64';c[c.index('--output')+1]=str(O/(name+'-profile.json'));csv=O/(name+'-profile.telemetry.csv');c[c.index('--gpu-telemetry-csv')+1]=str(csv);env=dict(os.environ,**json.load(open(O/(name+'.environment.json'))))
  cmd=['nsys','profile','--trace=cuda,nvtx,cublas','--sample=none','--cpuctxsw=none','--cuda-graph-trace=node','--wait=all','--output',str(S/name),*c]
  with csv.open('w') as f:
   monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,memory.used,utilization.gpu','--format=csv','--loop-ms=200'],stdout=f,stderr=subprocess.STDOUT)
   try:run(name+'-profile',cmd,env)
   finally:monitor.terminate();monitor.wait(timeout=10)
  d=json.load(open(O/(name+'-profile.json')));primary=json.load(open(O/(name+'.json')));assert d['representative_output_token_ids']==primary['representative_output_token_ids'][:64];d['status']='diagnostic_only_profiled_not_primary_throughput';(O/(name+'-profile.json')).write_text(json.dumps(d,indent=2)+'\n')
  run(name+'-export',['nsys','export','--type','sqlite','--output',str(S/(name+'.sqlite')),str(S/(name+'.nsys-rep'))])
 run('profile-accounting',[sys.executable,str(R/'tools/analyze_12b_o_global_profiles.py'),str(S),str(O/'kernel-accounting.json')])
 data=json.load(open(O/'kernel-accounting.json'));functions=data['profiles']['combined']['actual_new_mangled_names'];run('actual-sass',['/opt/cuda/bin/cuobjdump','--dump-sass','--function',','.join(functions),str(O/'candidate/gem16-run')]);assert 'MMA' in (O/'actual-sass.stdout.log').read_text()
 run('resources',['/opt/cuda/bin/cuobjdump','--dump-resource-usage',str(O/'candidate/gem16-run')])
 # Deliberately retain only the compact bounded raw reports; SQL stays in RAM.
 manifest=[]
 for name in ['parent','combined']:
  src=S/(name+'.nsys-rep');assert src.stat().st_size<16*1024**2;dst=O/(name+'-bounded.nsys-rep');shutil.copyfile(src,dst);manifest.append({'path':str(dst.relative_to(R)),'bytes':dst.stat().st_size,'sha256':hashlib.sha256(dst.read_bytes()).hexdigest(),'sqlite_export_retained':False,'reason':'bounded actual dispatch evidence; SQLite view remains ephemeral'})
 (O/'retained-traces.json').write_text(json.dumps(manifest,indent=2)+'\n')
for depth in [2,3,4]:
 c=json.load(open(R/f'benchmarks/results/2026-11-17/12b-mtp-parity/gpu-d{depth}.command.json'));c[c.index('--executable')+1]=str(O/'candidate/gem16-run');c[c.index('--fixed-output-tokens')+1]='64';c[c.index('--warmups')+1]='0';c[c.index('--repetitions')+1]='1';c[c.index('--output')+1]=str(O/f'mtp-d{depth}-regression.json');i=c.index('--gpu-telemetry-csv');del c[i:i+2];run(f'mtp-d{depth}-regression',c)
 d=json.load(open(O/f'mtp-d{depth}-regression.json'));parent=json.load(open(R/f'benchmarks/results/2026-11-17/12b-mtp-parity/gpu-d{depth}.json'));assert d['representative_output_token_ids']==parent['representative_output_token_ids'][:64];assert not d['runs'][0]['engine_result']['fallbacks'] and not d['runs'][0]['engine_result']['token_loop_allocations'];d['status']='functional_regression_only_not_primary_throughput';(O/f'mtp-d{depth}-regression.json').write_text(json.dumps(d,indent=2)+'\n')
print('PASS qualification and RAM cleanup',flush=True)
