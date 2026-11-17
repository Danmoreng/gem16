from pathlib import Path
import json,csv,hashlib,statistics
R=Path.cwd();O=Path(__file__).resolve().parent;P=R/'benchmarks/results/2026-11-17/ordinary-o-global'
load=lambda p:json.loads(p.read_text())
env={f'GEM16_12B_ORDINARY_{k}':'1' for k in ['NORMS','OVERLAP','O','GLOBAL']}
(O/'environment.json').write_text(json.dumps(env,indent=2)+'\n')
def telemetry(p):
 rows=list(csv.DictReader(p.open()));active=[r for r in rows if float(r[' utilization.gpu [%]'].split()[0])>90]
 return {'sampled_whole_process_peak_mib':max(float(r[' memory.used [MiB]'].split()[0]) for r in rows),'high_utilization_sm_median_mhz':statistics.median(float(r[' clocks.current.sm [MHz]'].split()[0]) for r in active)}
cohorts={}
for d in [2,3,4]:
 a=load(P/f'mtp-d{d}-primary.json');b=load(O/f'mtp-d{d}-primary.json');c=load(O/f'mtp-d{d}-primary-comparison.json')
 assert len(b['runs'])==3 and b['summary']['output_token_sha256_values']==a['summary']['output_token_sha256_values']
 parent=a['runs'][0]['engine_result']['mtp'];counters=[]
 for run in b['runs']:
  e=run['engine_result'];m=e['mtp'];assert run['generated_tokens']==1135 and run['measured_decode_intervals']==1134 and not e['fallbacks'] and not e['token_loop_allocations']
  for k in ['accepted_tokens','proposed_tokens','proposed_token_ids','verification_groups','target_batches','target_forwards','group_execution','short_batch_projection_path']:
   assert m[k]==parent[k],(d,k)
  counters.append({k:m[k] for k in ['accepted_tokens','proposed_tokens','verification_groups','target_batches','target_forwards']})
 assert all(x==counters[0] for x in counters)
 cohorts[d]={**c,'delta_percent':(c['median_tps']/c['previous_median_tps']-1)*100,'ordinary_tail_steps':counters[0]['target_batches']-counters[0]['verification_groups'],'mtp_counters':counters[0],'proposed_ids_exact_to_parent':True,'output_sha256_values':b['summary']['output_token_sha256_values'],'parent_telemetry':telemetry(P/f'mtp-d{d}-primary.telemetry.csv'),'candidate_telemetry':telemetry(O/f'mtp-d{d}-primary.telemetry.csv'),'raw_result':str((O/f'mtp-d{d}-primary.json').relative_to(R))}
trace=load(O/'dispatch-summary.json');binary=load(O/'binary.json');assert hashlib.sha256((O/'gem16-run').read_bytes()).hexdigest()==binary['sha256']
source=load(O/'source-manifest.json');assert all(hashlib.sha256((R/p).read_bytes()).hexdigest()==v for p,v in source.items())
result={'scope':'12B optimized Ordinary kernels enabled inside MTP; verifier and Assistant unchanged','warmups_per_depth':1,'retained_per_depth':3,'binary':binary,'source':source,'source_integrity':load(O/'source-integrity.json'),'environment':env,'workload':load(O/'mtp-d3-primary.json')['workload'],'runtime':load(O/'mtp-d3-primary.json')['runtime'],'cohorts':cohorts,'dispatch':trace,'host_units_passed':True,'unchanged_operator_qualification':'benchmarks/results/2026-11-17/ordinary-o-global/qualification-summary.json','limitations':['Separate chronological cohorts; no AB/BA or locked-clock confidence claim. Small deltas do not prove a speedup.','NVML peaks are 200ms sampled whole-process values, not allocator-exact decode peaks.','Ordinary single-row optimizations affect only the small tail in these fixed-D requests.','Rejected MTP row/Assistant ports are completely removed; their measurements are not final runtime evidence.']}
(O/'qualification-summary.json').write_text(json.dumps(result,indent=2)+'\n');(R/'artifacts/gemma4_12b/mtp-ordinary-only-2026-11-17.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# 12B MTP with optimized Ordinary only — 2026-11-17','',
'Only the Ordinary single-row path changes. The four startup MTP exclusions on',
'Ordinary norms, Gate/Up overlap, FP8 O and global attention are removed. All',
'other runtime sources are byte-identical to the previously measured source;',
'Assistant, verifier/prefill and norm CUDA files also match the committed parent.',
'The unwanted verifier/Assistant port experiment was fully removed and remains',
'rejected diagnostic history under `benchmarks/results/2026-11-17/mtp-integrated/`.','',
'Run, chat, server and host units rebuilt successfully. Primary measurements',
'use the final immutable runner, one warmup plus three retained requests per',
'depth, no profiler. Same 16,384 input IDs, 1,135 output IDs, 1,134 timed decode',
'intervals, context 17,519, batch one, greedy, suppression set, pinned models,',
'resident weights, FP8 KV and full fixed-D conditional graphs as the parent.','',
'| Mode | Parent tok/s | Final tok/s | Delta | Samples | Ordinary tail steps |',
'|---|---:|---:|---:|---|---:|']
for d,c in cohorts.items():lines.append(f"| D{d} | {c['previous_median_tps']:.5f} | {c['median_tps']:.5f} | {c['delta_percent']:+.3f}% | {', '.join(f'{x:.5f}' for x in c['samples_tps'])} | {c['ordinary_tail_steps']} |")
lines += ['',
'All output IDs and all proposed draft IDs match the parent exactly in every',
'retained request. Acceptance/group counters are unchanged; runtime fallback',
'and token-loop allocation counters remain zero. Host unit tests pass. Existing',
'unchanged Ordinary operator evidence covers 132 FP8-O and 60 global-attention',
'comparisons plus scoped memcheck/synccheck/racecheck; no new numerical kernels',
'are introduced by this startup-gating correction.','',
'The bounded D3 graph trace confirms actual new Ordinary O/global/norm kernels',
'alongside the old packed verifier projection and BF16 Assistant exact-tree',
'kernel. Rejected port kernel counts are zero. SQL exports use checked tmpfs',
'and are automatically removed. The retained Nsight report is a bounded',
'diagnostic and is excluded from primary throughput.','',
'The Ordinary path executes only at the request tail when too few output slots',
'remain for D+1 verification rows. For example, D3 has 437 verification groups',
'and only three Ordinary steps; these kernels therefore cannot materially',
'accelerate its entire decode. The small measured deltas do not establish a',
'performance gain: these are separate cohorts with dynamic laptop clocks,',
'not AB/BA confidence measurements. Standalone Ordinary remains the previously',
'qualified 53.17228 tok/s; it was not remeasured because its source/dispatch',
'and MTP=0 startup behavior are unchanged.','',
'| Mode | Parent sampled peak MiB | Final sampled peak MiB |',
'|---|---:|---:|']
for d,c in cohorts.items():lines.append(f"| D{d} | {c['parent_telemetry']['sampled_whole_process_peak_mib']:.0f} | {c['candidate_telemetry']['sampled_whole_process_peak_mib']:.0f} |")
lines += ['',
'Peaks are whole-process 200ms NVML samples, not exact decode-only maxima.',
f"Runner SHA256: `{binary['sha256']}`. Base revision `{binary['git_revision']}`",
'plus the source manifest (dirty working tree). No commit or push.','',
'Raw commands, JSON samples, logs, telemetry, dispatch summary, source hashes',
'and final binary: `benchmarks/results/2026-11-17/mtp-ordinary-only/`.',
'Portable summary: `artifacts/gemma4_12b/mtp-ordinary-only-2026-11-17.json`.','']
(R/'docs/evidence/gemma4_12b/mtp-ordinary-only-2026-11-17.md').write_text('\n'.join(lines))
print(json.dumps({d:{k:c[k] for k in ['median_tps','delta_percent','ordinary_tail_steps']} for d,c in cohorts.items()},indent=2))
