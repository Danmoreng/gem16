from pathlib import Path
import sys,subprocess,json,hashlib,shutil,os,concurrent.futures
R=Path.cwd();O=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'tools'));from experiment_scratch import experiment_scratch
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
assert shutil.disk_usage(R).free>2*1024**3

def run(name,cmd,timeout=300):
 (O/(name+'.command.json')).write_text(json.dumps(cmd,indent=2)+'\n')
 with (O/(name+'.stdout.log')).open('w') as out,(O/(name+'.stderr.log')).open('w') as err:subprocess.run(cmd,check=True,stdout=out,stderr=err,timeout=timeout)
 print('PASS',name,flush=True)
with experiment_scratch('ordinary-o-global',expected_gib=4) as S:
 (O/'scratch.json').write_text(json.dumps({'ram_directory':str(S),'auto_cleanup':True,'expected_gib':4,'ssd_free_bytes':shutil.disk_usage(R).free},indent=2)+'\n')
 run('generate',[sys.executable,str(R/'tests/cuda/build_12b_ordinary_o_global.py'),str(S)])
 for name in ['ordinary_o.cuh','ordinary_global_gqa4.cuh','derivatives.json']:shutil.copyfile(S/name,O/name)
 base=['/opt/cuda/bin/nvcc','-std=c++20','-O3','-rdc=true','-shared','-Xcompiler=-fPIC','-gencode=arch=compute_120a,code=sm_120a','-I'+str(R/'include'),'-I'+str(R/'src'),'-I'+str(R/'tests/cuda')]
 commands={family:base+[str(S/file),'-o',str(S/(family+'.so'))] for family,file in [('o','fp8.cu'),('global','attention.cu')]}
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  futures=[pool.submit(run,family+'-build',cmd) for family,cmd in commands.items()]
  for f in futures:f.result()
 for family,fixtures in [('o',R/'benchmarks/results/2026-11-15/58febf4/12b-five-step-goal/ordinary-fixtures'),('global',R/'benchmarks/results/2026-11-15/58febf4/12b-pro-followup/e5/fixtures')]:
  library=S/(family+'.so');(O/(family+'-binary.json')).write_text(json.dumps({'library_sha256':hashlib.sha256(library.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256((S/('fp8.cu' if family=='o' else 'attention.cu')).read_bytes()).hexdigest()},indent=2)+'\n')
  cmd=[sys.executable,str(R/'tests/cuda/twelve_b_ordinary_o_global_replay.py'),'--family',family,'--fixtures',str(fixtures),'--library',str(library),'--output',str(O/(family+'-numeric-timing.json')),'--timing'];run(family+'-numeric-timing',cmd)
  # Scoped single-real-case operator sanitizer suite; full real-data numerical
  # coverage is above. Do not turn these into full-model/all-profile claims.
  for tool in ['memcheck','synccheck','racecheck']:
   c=cmd[:-1];c[c.index('--output')+1]=str(O/(family+'-'+tool+'.json'));c+=['--case','1' if family=='o' else '0']
   run(family+'-'+tool,['/opt/cuda/bin/compute-sanitizer','--tool',tool,'--error-exitcode','99',*c],timeout=240)
  run(family+'-resources',['/opt/cuda/bin/cuobjdump','--dump-resource-usage',str(library)])
  # SASS is compact enough to retain after selecting the dispatch names.
  cmd=['/opt/cuda/bin/cuobjdump','--dump-sass','--function','TwelveBOrdinaryOProjectionKernel' if family=='o' else 'TwelveBOrdinaryGlobalGqa4Kernel',str(library)]
  run(family+'-sass',cmd)
print('Offline RAM workspace cleaned',flush=True)
