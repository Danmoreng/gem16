from pathlib import Path
import subprocess,json,os,time,sys,hashlib
R=Path.cwd();O=Path(__file__).resolve().parent;B=R/'benchmarks/results/2026-11-17/ordinary-kernel-comparison';exe=R/'build/Linux/blackwell-release/bin/gem16-run';saved=O/'candidate';saved.mkdir(exist_ok=True)
import shutil
for src in [exe,R/'build/Linux/blackwell-release/lib/libgem16_core.a']:shutil.copyfile(src,saved/src.name)
exe=saved/'gem16-run';exe.chmod(0o755)
(O/'candidate-hashes.json').write_text(json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in saved.iterdir()},indent=2)+'\n')
def run(name,cmd,env=None,timeout=600):
 (O/(name+'.command.json')).write_text(json.dumps(cmd,indent=2)+'\n')
 with (O/(name+'.stdout.log')).open('x') as out,(O/(name+'.stderr.log')).open('x') as err:subprocess.run(cmd,check=True,env=env,stdout=out,stderr=err,timeout=timeout)
 print('PASS',name,flush=True)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
for tool in ['memcheck','synccheck','racecheck']:
 run('norm-'+tool,['/opt/cuda/bin/compute-sanitizer','--tool',tool,'--error-exitcode','99',str(O/'norm-boundaries')],timeout=180)
base=json.loads((B/'native.command.json').read_text());base[base.index('--executable')+1]=str(exe)
# Bounded graph/dependency smoke before any primary cohort.
c=base.copy();c[c.index('--warmups')+1]='0';c[c.index('--repetitions')+1]='1';c[c.index('--fixed-output-tokens')+1]='64';c[c.index('--output')+1]=str(O/'combined-smoke.json');i=c.index('--gpu-telemetry-csv');del c[i:i+2]
run('combined-smoke',c,dict(os.environ,GEM16_12B_ORDINARY_NORMS='1',GEM16_12B_ORDINARY_OVERLAP='1'))
d=json.load(open(O/'combined-smoke.json'));parent=json.load(open(B/'native.json'));assert d['representative_output_token_ids']==parent['representative_output_token_ids'][:64]
for name,norm,overlap in [('kernel-only','1','0'),('overlap-only','0','1'),('combined','1','1')]:
 deadline=time.monotonic()+600
 while float(subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],text=True))>50:
  assert time.monotonic()<deadline;print('cooling',name,flush=True);time.sleep(5)
 env=dict(os.environ,GEM16_12B_ORDINARY_NORMS=norm,GEM16_12B_ORDINARY_OVERLAP=overlap)
 (O/(name+'.environment.json')).write_text(json.dumps({k:env[k] for k in ['GEM16_12B_ORDINARY_NORMS','GEM16_12B_ORDINARY_OVERLAP']},indent=2)+'\n')
 c=base.copy();c[c.index('--output')+1]=str(O/(name+'.json'));csv=O/(name+'.telemetry.csv');c[c.index('--gpu-telemetry-csv')+1]=str(csv)
 print('RUN',name,'1warmup+3retained',flush=True)
 with csv.open('x') as f:
  monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,memory.used,utilization.gpu','--format=csv','--loop-ms=200'],stdout=f,stderr=subprocess.STDOUT)
  try:run(name,c,env,timeout=600)
  finally:monitor.terminate();monitor.wait(timeout=10)
 d=json.load(open(O/(name+'.json')));assert d['summary']['output_token_sha256_values']==parent['summary']['output_token_sha256_values'];assert d['summary']['deterministic_outputs']
 for r in d['runs']:assert r['generated_tokens']==1135 and r['measured_decode_intervals']==1134 and r['engine_result']['token_loop_allocations']==0 and not r['engine_result']['fallbacks']
 print('RESULT',name,d['summary']['decode_tokens_per_second']['median'],d['summary']['decode_tokens_per_second']['samples'],flush=True)
