"""Publish E2's full fixed-protocol outcome, including any rejected gain."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import statistics

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('raw',type=Path);p.add_argument('destination',type=Path)
a=p.parse_args();out=a.raw
load=lambda p:json.loads(p.read_text())
stats=load(out/'model-stats.json');inputs=load(out/'production-inputs.json')
retained=(out/'promotion.json').exists();restored=(out/'restoration.json').exists()
assert retained != restored, 'must first retain or restore the candidate'
samples={};medians={};peaks={}
for mode in ['d2','ordinary']:
    samples[mode]={};peaks[mode]={};pooled={'parent':[],'candidate':[]}
    for i,role in enumerate(['parent','candidate','candidate','parent']):
        label=f'{mode}-{i}-{role}';d=load(out/(label+'.json'))
        assert len(d['runs'])==10 and d['configuration']['warmups']==3
        samples[mode][label]=[{key:row[key] for key in ['decode_ms','decode_tokens_per_second','prompt_ms','prompt_tokens_per_second','output_token_sha256']} for row in d['runs']]
        pooled[role]+=[row['decode_tokens_per_second'] for row in d['runs']]
        with (out/(label+'-telemetry.csv')).open() as f:
            peaks[mode][label]=max(float(row[' memory.used [MiB]'].split()[0]) for row in csv.DictReader(f))
    medians[mode]={role:statistics.median(values) for role,values in pooled.items()}
result={'schema_version':1,'date':'2026-11-16',
        'status':'correctness_qualified_provisional_combined_package_component' if retained else 'measured_effect_not_retained',
        'scope':'one exact-tree12B Assistant GEMV candidate; five declared shapes; fullD2 primary and Ordinary secondary, unchanged outputs/proposals/counters',
        'full_model_statistics':stats,'median_tokens_per_second':medians,
        'retained_samples':samples,'operator_statistics':load(out/'operator-stats.json'),
        'safety':load(out/'candidate-safety-complete.json'),'counters':load(out/'counter-comparison.json'),
        'binary_sha256':{role:inputs[role]['gem16-run']['sha256'] for role in ['parent','candidate']},
        'source_sha256':{role:inputs[role+'_source']['sha256'] for role in ['parent','candidate']},
        'model_sha256':{role:info['sha256'] for role,info in inputs['models'].items()},
        'telemetry_peak_total_gpu_used_mib':peaks,
        'memory_note':'200ms nvidia-smi device samples include context/display overhead and are not exact byte-allocation high-water marks; per-request arenas match parent',
        'minimum_final_package_measured_gain':.005,'decode_lower95_margin':-.02,
        'owner_amendment':'explicit2026-11-16 reply moves0.5% threshold to combined package of current five experiments; original single-trial assessment preserved',
        'limitations':['Linux SM120a only; no new Windows execution evidence',
                       'one fixed16Kprompt/1135output workload; separate fresh parent not pooled with E1',
                       'operator gains are not model gains; profiled timings excluded from primary',
                       'two bounded all-arithmetic-repeat race attempts timed out; completed race pass covers all126real full-grid changed-kernel calls; numeric/memcheck/synccheck cover all edges'],
        'git_publication':'not authorized by this Goal'}
if (out/'native-sequence.json').exists():result['native_sequence']=load(out/'native-sequence.json')
if (out/'state.json').exists():
    state=load(out/'state.json');result['full_state_audit']={key:value for key,value in state.items() if key!='interposer'}
if retained:
    protected=load(out/'protected.json');assert protected['all_parent_candidate_cases_exact']
    result['protected_cases_sha256']={key:hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest() for key,value in protected['cases'].items()}
    result['protected_scope']=protected['scope']
else:
    restoration=load(out/'restoration.json')
    result['restoration']={key:value for key,value in restoration.items() if key!='canonical_measured_parent_binaries'}
    result['acceptance_only_gates_skipped']='Any absent protected/native-state gate is unrun, not passed; no positive claim/promotion relies on it.'
text=json.dumps(result,indent=2)+'\n'
assert '/home/' not in text and '/Users/' not in text
with a.destination.open('x') as f:f.write(text)
print(a.destination)
