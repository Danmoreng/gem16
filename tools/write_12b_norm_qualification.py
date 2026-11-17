"""Publish the completed E1 qualification without local identity paths."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import statistics

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('raw',type=Path);p.add_argument('destination',type=Path)
a=p.parse_args();raw=a.raw; e1=raw/'e1'
load=lambda path:json.loads(path.read_text())
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
primary=load(e1/'model-primary-stats.json')
secondary_dir=e1/'secondary-2026-11-16-r2'
secondary=load(secondary_dir/'stats.json')
state=load(e1/'state-audit-2026-11-16-r2/result.json')
protected=load(e1/'protected-2026-11-16-r2/result.json')
assert primary['passes_primary_amended_gate'] and primary['passes_prefill_inverse_ttft_gate']
assert secondary['passes_secondary'] and secondary['passes_prefill_inverse_ttft']
assert state['all1135outputs_exact'] and state['all1014proposals_exact']
assert state['parent_candidate_and_ordinary_d2_full_kv_exact']
assert protected['all_parent_candidate_cases_exact']
assert protected['identities']['parent']['core']['sha256']!=protected['identities']['candidate']['core']['sha256']
samples={}; peaks={}; absolute={}
for mode,directory in [('ordinary',e1),('d2',secondary_dir)]:
    samples[mode]={};peaks[mode]={};both={'parent':[],'candidate':[]}
    for i,role in enumerate(['parent','candidate','candidate','parent']):
        label=f'{mode}-{i}-{role}';d=load(directory/(label+'.json'))
        samples[mode][label]=[{k:row[k] for k in ['decode_ms','decode_tokens_per_second','prompt_ms','prompt_tokens_per_second','output_token_sha256']} for row in d['runs']]
        both[role]+=[row['decode_tokens_per_second'] for row in d['runs']]
        with (directory/(label+'-telemetry.csv')).open() as f:
            rows=list(csv.DictReader(f))
        peak=max(float(row[' memory.used [MiB]'].split()[0]) for row in rows)
        peaks[mode][label]=peak
    absolute[mode]={role:statistics.median(values) for role,values in both.items()}
identities=load(e1/'model-inputs.json');parent=load(raw/'parent/inputs.json')
case_hashes={key:hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
             for key,value in protected['cases'].items()}
result={'schema_version':1,'date':'2026-11-16','status':'qualified_linux_local_gain',
        'scope':'12B T1 checkpoint-FP8 post-attention fusion only; no changed D2 T3 verifier or other-profile arithmetic',
        'base_commit':parent['base_commit'],
        'primary':primary,'secondary':secondary,'median_tokens_per_second':absolute,
        'retained_samples':samples,'telemetry_peak_total_gpu_used_mib':peaks,
        'memory_note':'nvidia-smi200ms process-wide GPU-use samples; includes context/display overhead, not a byte-exact allocation high-water mark. Engine arena sizes match per request.',
        'state_audit':state,'protected_case_hashes':case_hashes,
        'protected_scope':protected['scope'],
        'binary_sha256':{role:identities[role]['sha256'] for role in ['parent','candidate']},
        'candidate_source_sha256':{name:row['sha256'] for name,row in identities['sources'].items()},
        'model_sha256':{name:row['sha256'] for name,row in parent['models'].items()},
        'dispatch':load(e1/'production-dispatch-facts.json'),
        'build':'CUDA13.4.92/GCC16.2.1/CMake4.5, Release-O3/C++20/SM120a/RDC=true',
        'limitations':['Linux SM120a execution only; no new Windows execution evidence',
                       'one fixed16K/1135-output workload; no updated llama.cpp comparison',
                       'D2 effectively unchanged; no qualified D2 improvement claimed',
                       'static shared section2KiB versus loaded runtime1KiB; conservative2KiB budget'],
        'git_publication':'not part of this Goal',
        'raw_evidence_sha256':{str(path.relative_to(raw)):sha(path) for path in [e1/'model-primary-stats.json',secondary_dir/'stats.json',e1/'state-audit-2026-11-16-r2/result.json',e1/'protected-2026-11-16-r2/result.json',e1/'production-kernel.sass']}}
text=json.dumps(result,indent=2)+'\n'
assert '/home/' not in text and '/Users/' not in text
with a.destination.open('x') as f:f.write(text)
print(a.destination)
