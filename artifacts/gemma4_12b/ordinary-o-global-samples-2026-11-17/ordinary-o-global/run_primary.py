from pathlib import Path
import subprocess,json,os,time,sys,hashlib,shutil
R=Path.cwd();O=Path(__file__).resolve().parent;B=R/'benchmarks/results/2026-11-17/ordinary-kernel-comparison';old=R/'benchmarks/results/2026-11-17/ordinary-optimization';saved=O/'candidate';saved.mkdir(exist_ok=True)
for src in [R/'build/Linux/blackwell-release/bin/gem16-run',R/'build/Linux/blackwell-release/lib/libgem16_core.a']:shutil.copyfile(src,saved/src.name)
(saved/'gem16-run').chmod(0o755);(O/'binary-hashes.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in saved.iterdir()},indent=2)+'\n')
base=json.load(open(B/'native.command.json'));parent=json.load(open(old/'combined.json'));expected=parent['summary']['output_token_sha256_values']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip();assert shutil.disk_usage(R).free>2*1024**3

def run(name,cmd,env):
 (O/(name+'.command.json')).write_text(json.dumps(cmd,indent=2)+'\n')
 with (O/(name+'.stdout.log')).open('x') as out,(O/(name+'.stderr.log')).open('x') as err:subprocess.run(cmd,check=True,env=env,stdout=out,stderr=err,timeout=600)
 print('PASS',name,flush=True)
c=base.copy();c[c.index('--executable')+1]=str(saved/'gem16-run');c[c.index('--warmups')+1]='0';c[c.index('--repetitions')+1]='1';c[c.index('--fixed-output-tokens')+1]='64';c[c.index('--output')+1]=str(O/'combined-smoke.json');i=c.index('--gpu-telemetry-csv');del c[i:i+2]
env=dict(os.environ,GEM16_12B_ORDINARY_NORMS='1',GEM16_12B_ORDINARY_OVERLAP='1',GEM16_12B_ORDINARY_O='1',GEM16_12B_ORDINARY_GLOBAL='1');run('combined-smoke',c,env)
d=json.load(open(O/'combined-smoke.json'));assert d['representative_output_token_ids']==parent['representative_output_token_ids'][:64]
for name,exe,on in [('parent',old/'candidate/gem16-run','0'),('combined',saved/'gem16-run','1')]:
 deadline=time.monotonic()+600
 while float(subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],text=True))>50:
  assert time.monotonic()<deadline;print('cooling',name,flush=True);time.sleep(5)
 env=dict(os.environ,GEM16_12B_ORDINARY_NORMS='1',GEM16_12B_ORDINARY_OVERLAP='1',GEM16_12B_ORDINARY_O=on,GEM16_12B_ORDINARY_GLOBAL=on);(O/(name+'.environment.json')).write_text(json.dumps({k:env[k] for k in ['GEM16_12B_ORDINARY_NORMS','GEM16_12B_ORDINARY_OVERLAP','GEM16_12B_ORDINARY_O','GEM16_12B_ORDINARY_GLOBAL']},indent=2)+'\n')
 c=base.copy();c[c.index('--executable')+1]=str(exe);c[c.index('--output')+1]=str(O/(name+'.json'));csv=O/(name+'.telemetry.csv');c[c.index('--gpu-telemetry-csv')+1]=str(csv)
 print('RUN',name,'1warmup+3retained',flush=True)
 with csv.open('x') as f:
  monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,memory.used,utilization.gpu','--format=csv','--loop-ms=200'],stdout=f,stderr=subprocess.STDOUT)
  try:run(name,c,env)
  finally:monitor.terminate();monitor.wait(timeout=10)
 d=json.load(open(O/(name+'.json')));assert d['summary']['output_token_sha256_values']==expected and d['summary']['deterministic_outputs']
 for row in d['runs']:assert row['generated_tokens']==1135 and row['measured_decode_intervals']==1134 and not row['engine_result']['token_loop_allocations'] and not row['engine_result']['fallbacks']
 print('RESULT',name,d['summary']['decode_tokens_per_second']['median'],d['summary']['decode_tokens_per_second']['samples'],flush=True)
