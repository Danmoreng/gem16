from pathlib import Path
import subprocess,json,time,sys,hashlib
R=Path.cwd();O=Path(__file__).resolve().parent;engine=sys.argv[1];assert engine in ['native','vllm','llama']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
deadline=time.monotonic()+600
while float(subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],text=True))>50:
 assert time.monotonic()<deadline,'cooldown deadline';print('cooling',flush=True);time.sleep(5)
model=Path('/home/sebastian/.cache/huggingface/hub/.gem16/snapshots/unsloth--gemma-4-12b-it-NVFP4--b1f649734b34aa5575b03d186abd1b9be3d0d5c4');csv=O/(engine+'.telemetry.csv');python=R/'third_party/cache/vllm-0.33.0-env/bin/python' if engine=='vllm' else Path(sys.executable)
c=[str(python),str(R/'tools/benchmark_wikipedia_workload.py'),'--engine',{'native':'gem16','llama':'llama-cpp','vllm':'vllm'}[engine],'--model',str(model),'--workload',str(R/'benchmarks/prompts/wikipedia-summary-16k.json'),'--context-tokens','17519','--fixed-output-tokens','1135','--mtp-draft-tokens','0','--warmups','1','--repetitions','3','--output',str(O/(engine+'.json')),'--gpu-telemetry-csv',str(csv)]
if engine=='native':c+=['--executable',str(R/'benchmarks/results/2026-11-17/12b-mtp-parity/production-candidate/gem16-run')]
if engine=='vllm':c+=['--gpu-memory-utilization','0.90','--vllm-kv-cache-dtype','fp8','--vllm-max-num-batched-tokens','1024']
if engine=='llama':
 executable=(Path('/home/sebastian/Development/_engines/llama.cpp/current')/'build-central-cuda/bin/llama-server').resolve(strict=True);c+=['--executable',str(executable),'--gguf','/home/sebastian/.cache/huggingface/derived/gem16/gemma4-12b-llama-cpp/gemma4-12b-mixed-q8-nvfp4.gguf','--llama-override-tensor','token_embd.weight=CUDA0','--llama-kv-cache-type','q8_0','--llama-batch-size','2048','--llama-ubatch-size','512','--llama-threads','8','--llama-poll','100','--llama-port','18097','--llama-log-verbosity','3']
(O/(engine+'.command.json')).write_text(json.dumps(c,indent=2)+'\n');env=dict(__import__('os').environ,CUDA_HOME='/opt/cuda',MAX_JOBS='4',TORCHINDUCTOR_COMPILE_THREADS='1',FLASHINFER_NVCC_THREADS='1');(O/(engine+'.environment.json')).write_text(json.dumps({k:env[k] for k in ['CUDA_HOME','MAX_JOBS','TORCHINDUCTOR_COMPILE_THREADS','FLASHINFER_NVCC_THREADS']},indent=2)+'\n')
with csv.open('x') as f:
 monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,memory.used,utilization.gpu','--format=csv','--loop-ms=200'],stdout=f,stderr=subprocess.STDOUT)
 print('RUN',engine,'Ordinary:1warmup+3retained',flush=True)
 try:
  with (O/(engine+'.stdout.log')).open('x') as out,(O/(engine+'.stderr.log')).open('x') as err:subprocess.run(c,check=True,env=env,stdout=out,stderr=err,timeout=900)
 finally:monitor.terminate();monitor.wait(timeout=10)
d=json.loads((O/(engine+'.json')).read_text());assert d['configuration']['mtp_draft_tokens']==0 and d['summary']['deterministic_outputs'];assert d['configuration']['warmups']==1 and d['configuration']['measured_repetitions']==3
for row in d['runs']:
 assert row['generated_tokens']==1135 and row['measured_decode_intervals']==1134
 if engine=='native':assert not row['engine_result']['mtp']['enabled'] and not row['engine_result']['fallbacks'] and not row['engine_result']['token_loop_allocations']
print('RESULT',engine,d['summary']['decode_tokens_per_second']['median'],d['summary']['decode_tokens_per_second']['samples'],flush=True)
