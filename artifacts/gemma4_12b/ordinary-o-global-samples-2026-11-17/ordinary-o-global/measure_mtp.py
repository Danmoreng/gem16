from pathlib import Path
import subprocess,json,os,time,hashlib
R=Path.cwd();O=Path(__file__).resolve().parent;P=R/'benchmarks/results/2026-11-17/12b-mtp-parity';exe=O/'candidate/gem16-run';expected=json.load(open(O/'combined.json'))['summary']['output_token_sha256_values']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
for depth in [2,3,4]:
 deadline=time.monotonic()+300
 while float(subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],text=True))>50:
  assert time.monotonic()<deadline;print('cooling D'+str(depth),flush=True);time.sleep(5)
 name=f'mtp-d{depth}-primary';c=json.load(open(P/f'gpu-d{depth}.command.json'));c[c.index('--executable')+1]=str(exe);c[c.index('--output')+1]=str(O/(name+'.json'));csv=O/(name+'.telemetry.csv');c[c.index('--gpu-telemetry-csv')+1]=str(csv);assert c[c.index('--warmups')+1]=='1' and c[c.index('--repetitions')+1]=='3'
 (O/(name+'.command.json')).write_text(json.dumps(c,indent=2)+'\n');print('RUN D'+str(depth),'1warmup+3retained',flush=True)
 with csv.open('x') as f:
  monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,memory.used,utilization.gpu','--format=csv','--loop-ms=200'],stdout=f,stderr=subprocess.STDOUT)
  try:
   with (O/(name+'.stdout.log')).open('x') as out,(O/(name+'.stderr.log')).open('x') as err:subprocess.run(c,check=True,stdout=out,stderr=err,timeout=600)
  finally:monitor.terminate();monitor.wait(timeout=10)
 d=json.load(open(O/(name+'.json')));old=json.load(open(P/f'gpu-d{depth}.json'));assert d['summary']['deterministic_outputs'] and d['summary']['output_token_sha256_values']==old['summary']['output_token_sha256_values']
 for row in d['runs']:assert row['generated_tokens']==1135 and row['measured_decode_intervals']==1134 and not row['engine_result']['fallbacks'] and not row['engine_result']['token_loop_allocations']
 result={'depth':depth,'median_tps':d['summary']['decode_tokens_per_second']['median'],'samples_tps':d['summary']['decode_tokens_per_second']['samples'],'previous_median_tps':old['summary']['decode_tokens_per_second']['median'],'all_outputs_exact':True,'binary_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'new_ordinary_specializations_enabled_in_mtp':False}
 (O/(name+'-comparison.json')).write_text(json.dumps(result,indent=2)+'\n');print('RESULT',json.dumps(result),flush=True)
