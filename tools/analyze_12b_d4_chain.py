"""Offline D2/D3/D4 chain dispatch accounting; profiler times are diagnostic."""
from pathlib import Path
from collections import Counter
import argparse
import csv
import json
import sqlite3
from analyze_12b_vllm_alignment import stats

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('root', type=Path)
p.add_argument('--reference-root', type=Path)
p.add_argument('--depths', type=int, nargs='+', default=[2, 4], choices=[2, 3, 4])
p.add_argument('--output-tokens', type=int, default=256)
p.add_argument('--expected-d2-changes', action='store_true')
a = p.parse_args()
assert a.output_tokens > 0
out = {}
for depth in a.depths:
    with sqlite3.connect((a.root / f'dispatch-d{depth}.sqlite').resolve().as_uri() + '?mode=ro', uri=True) as c:
        c.row_factory = sqlite3.Row
        strings = dict(c.execute('select id,value from StringIds'))
        rows = [dict(r) for r in c.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL order by start')]
        apis = Counter({strings[r[0]]: r[1] for r in c.execute('select nameId,count(*) from CUPTI_ACTIVITY_KIND_RUNTIME group by nameId')})
    for r in rows:
        r['name'] = strings[r['shortName']]
        r['full_name'] = strings[r['demangledName']]
    bounds = [i for i, e in enumerate(rows) if e['name'] == 'InitializeControlledAssistantKernel']
    assert len(bounds) > 32
    events = []
    phase = 'assistant'
    for r in rows[bounds[16]:bounds[32]]:
        n = r['name']
        if n == 'InitializeControlledAssistantKernel': phase = 'assistant'
        if n == 'EmbeddingBatchKernel': phase = 'target'
        if n == 'AcceptMtpGroupKernel': phase = 'commit'
        if phase == 'assistant': family = 'assistant_all'
        elif phase == 'commit': family = 'commit_control'
        elif 'OutputHead' in n: family = 'target_head_selection'
        elif 'Sm120D2DownFixedShape' in n or 'Sm120D4DownFixedShape' in n or 'Sm120D3DownFixedShape' in n: family = 'target_down'
        elif 'Sm120MatrixProjectionKernel' in n:
            family = 'target_gateup' if '<(bool)1,' in r['full_name'] else 'target_down'
        elif 'Sm120DirectProjection' in n or 'TwelveBD3PackedFp8Projection' in n or 'TwelveBPackedFp8Projection' in n: family = 'target_fp8_qkv_o'
        elif 'PostAttentionQuantize' in n: family = 'target_norm_quant_rotary'
        elif 'Attention' in n: family = 'target_attention'
        elif 'Kv' in n: family = 'target_kv'
        elif any(x in n for x in ['Norm', 'Quantize', 'RoundBf16', 'Rotary']): family = 'target_norm_quant_rotary'
        else: family = 'target_other'
        events.append({'name': n, 'full_name': r['full_name'], 'family': family,
                       'ts': r['start'] / 1000, 'dur': (r['end'] - r['start']) / 1000,
                       'grid': [r['gridX'], r['gridY'], r['gridZ']],
                       'block': [r['blockX'], r['blockY'], r['blockZ']],
                       'graph': r['graphId'], 'registers': r['registersPerThread'],
                       'shared': r['staticSharedMemory'] + r['dynamicSharedMemory']})
    start = rows[bounds[16]]['start'] / 1000
    result = stats(events, start, rows[bounds[32]]['start'] / 1000, 16)
    result['all_group_boundaries'] = len(bounds)
    result['capture_runtime_api_counts'] = dict(apis)
    assert sum(v for k, v in apis.items() if k.startswith('cudaGraphLaunch')) == 1
    assert not any(v for k, v in apis.items() if any(x in k for x in ['Synchronize', 'Malloc', 'HostAlloc']))
    text = (a.root / f'dispatch-d{depth}.stdout.log').read_text()
    engine = json.JSONDecoder().raw_decode(text[text.index('{\n'):])[0]
    (a.root / f'dispatch-d{depth}-engine.json').write_text(json.dumps(engine, indent=2) + '\n')
    assert len(engine['output_token_ids']) == a.output_tokens
    assert engine['mtp']['group_execution'] == f'gpu_chained_fixed_d{depth}_conditional_graph'
    assert not engine['fallbacks'] and not engine['token_loop_allocations']
    result['engine_mtp'] = {k: v for k, v in engine['mtp'].items() if k != 'proposed_token_ids'}
    with (a.root / f'dispatch-d{depth}-timeline.csv').open('w') as stream:
        w = csv.writer(stream)
        w.writerow(['start_us','duration_us','family','graph','grid','block','registers','shared','kernel'])
        for e in events:
            w.writerow([e['ts'] - start,e['dur'],e['family'],e['graph'],e['grid'],e['block'],e['registers'],e['shared'],e['full_name']])
    out[f'd{depth}'] = result
    print('D' + str(depth), result['family_ms_per_group'])
# The preceding fresh full D2 capture has the exact same first256 output IDs
# and selected groups. Compare invocation identities, not profiler durations.
ref = a.reference_root or a.root.parent / 'vllm-alignment'
baseline = list(csv.DictReader((ref / 'native-candidate-d2-timeline.csv').open()))
current = list(csv.DictReader((a.root / 'dispatch-d2-timeline.csv').open()))
keys = ['family','grid','block','registers','shared','kernel']
previous = Counter(tuple(e[k] for k in keys) for e in baseline)
latest = Counter(tuple(e[k] for k in keys) for e in current)
exact = previous == latest
if not exact:
    assert a.expected_d2_changes, 'unexpected D2 dispatch change'
    delta = list((previous - latest).items()) + list((latest - previous).items())
    assert all(row[0][0] in ('target_fp8_qkv_o', 'target_attention') for row in delta)
    assert Counter(e['family'] for e in baseline) == Counter(e['family'] for e in current)
    out['d2_dispatch_changes'] = {
        'removed': [{'identity': list(k), 'calls': v} for k,v in (previous-latest).items()],
        'added': [{'identity': list(k), 'calls': v} for k,v in (latest-previous).items()]}
expected = json.loads((ref / 'down-fixed/integration/screen-d2-candidate.json').read_text())['representative_output_token_ids']
for depth in a.depths:
    assert json.loads((a.root / f'dispatch-d{depth}-engine.json').read_text())['output_token_ids'] == expected[:a.output_tokens]
out['d2_prior_dispatch_identities_exact'] = exact
(a.root / 'dispatch-comparison.json').write_text(json.dumps(out, indent=2) + '\n')
print('D2 dispatch identities exact:', exact, '; all captures one launch, no sync/allocation APIs')
