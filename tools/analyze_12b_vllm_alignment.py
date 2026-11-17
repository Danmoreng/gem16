"""Offline warmed trace accounting; diagnostic times are never benchmark rates."""
from pathlib import Path
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
import sqlite3


def union(intervals):
    total = 0.0
    end = None
    for a, b in sorted(intervals):
        if end is None or a > end:
            total += b - a
        elif b > end:
            total += b - end
        end = max(end if end is not None else b, b)
    return total


def stats(events, start, end, groups):
    costs = defaultdict(float)
    counts = Counter()
    names = defaultdict(lambda: [0, 0.0])
    for e in events:
        costs[e['family']] += e['dur']
        counts[e['family']] += 1
        names[e['name']][0] += 1
        names[e['name']][1] += e['dur']
    busy = union([(e['ts'], e['ts'] + e['dur']) for e in events])
    return {'diagnostic_only': True, 'groups': groups, 'kernel_events': len(events),
            'wall_ms': (end - start) / 1000, 'kernel_busy_union_ms': busy / 1000,
            'non_kernel_interval_ms': (end - start - busy) / 1000,
            'kernel_duration_sum_ms': sum(e['dur'] for e in events) / 1000,
            'overlapping_kernel_ms': (sum(e['dur'] for e in events) - busy) / 1000,
            'family_ms_per_group': {k: v / groups / 1000 for k, v in sorted(costs.items())},
            'family_kernel_counts': dict(counts),
            'kernels': [{'name': k, 'calls': v[0], 'total_ms': v[1] / 1000}
                        for k, v in sorted(names.items(), key=lambda p: -p[1][1])]}


def native(root, filename='native-parent-d2-capture.sqlite'):
    db = root / filename
    with sqlite3.connect(db.resolve().as_uri() + '?mode=ro', uri=True) as c:
        c.row_factory = sqlite3.Row
        strings = dict(c.execute('select id,value from StringIds'))
        rows = [dict(r) for r in c.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL order by start')]
    for r in rows:
        r['name'] = strings[r['shortName']]
        r['full_name'] = strings[r['demangledName']]
    boundaries = [i for i, e in enumerate(rows) if e['name'] == 'InitializeControlledAssistantKernel']
    assert len(boundaries) == 507
    left, right = boundaries[16], boundaries[32]
    events = []
    phase = 'assistant'
    for r in rows[left:right]:
        n = r['name']
        if n == 'InitializeControlledAssistantKernel': phase = 'assistant'
        if n == 'EmbeddingBatchKernel': phase = 'target'
        if n == 'AcceptMtpGroupKernel': phase = 'commit'
        if phase == 'assistant': f = 'assistant_all'
        elif phase == 'commit': f = 'commit_control'
        elif 'FusedOutputHead' in n or 'OutputHead' in n: f = 'target_head_selection'
        elif 'Sm120D2DownFixedShapeKernel' in n: f = 'target_down'
        elif 'Sm120MatrixProjectionKernel' in n:
            f = 'target_gateup' if '<(bool)1,' in r['full_name'] else 'target_down'
        elif 'Sm120DirectProjection' in n: f = 'target_fp8_qkv_o'
        elif 'PostAttentionQuantize' in n: f = 'target_norm_quant_rotary'
        elif 'Attention' in n: f = 'target_attention'
        elif 'Kv' in n: f = 'target_kv'
        elif any(x in n for x in ('Norm', 'Quantize', 'RoundBf16', 'Rotary')): f = 'target_norm_quant_rotary'
        else: f = 'target_other'
        events.append({'name': r['name'], 'full_name': r['full_name'], 'family': f,
                       'ts': r['start'] / 1000, 'dur': (r['end'] - r['start']) / 1000,
                       'grid': [r['gridX'], r['gridY'], r['gridZ']],
                       'block': [r['blockX'], r['blockY'], r['blockZ']],
                       'graph': r.get('graphId'), 'registers': r.get('registersPerThread'),
                       'shared': r.get('staticSharedMemory', 0) + r.get('dynamicSharedMemory', 0)})
    return events, stats(events, rows[left]['start'] / 1000, rows[right]['start'] / 1000, 16)


def vllm(root, depth):
    path = next((root / f'vllm-d{depth}-traces').glob('*.json.gz'))
    trace = json.load(gzip.open(path, 'rt'))['traceEvents']
    annotations = [e for e in trace if e.get('cat') == 'user_annotation' and e['name'].startswith('execute_')]
    assert len(annotations) == 16 and all('context_0(' in e['name'] and '_generation_1(' in e['name'] for e in annotations)
    ks = [e for e in trace if e.get('cat') == 'kernel']
    graph_counts = Counter(e['args'].get('graph id', 0) for e in ks if e['name'] == 'kernel_unified_attention')
    target_graph = graph_counts.most_common(1)[0][0]
    assert graph_counts[target_graph] == 16 * 48
    events = []
    for e in ks:
        n = e['name']; g = e['args'].get('graph id', 0)
        if g and g != target_graph: f = 'assistant_all'
        elif not g: f = 'target_head_selection' if '128x1_tn_align8' in n else 'outside_graph_other'
        elif 'BlockScaled' in n:
            # Actual geometries match logical N=30720 Gate/Up and N=3840 Down,
            # at128 output channels/CTA (CUTLASS may swap M/N internally).
            # Do not infer operator identity from the pipeline stage count.
            grid = e['args'].get('grid')
            assert grid in ([240, 1, 1], [1, 30, 1]), (n, grid)
            f = 'target_gateup' if grid == [240, 1, 1] else 'target_down'
        elif 'cutlass' in n and 'device_kernel' in n: f = 'target_fp8_qkv_o'
        elif n in ('kernel_unified_attention', 'reduce_segments'): f = 'target_attention'
        elif 'reshape_and_cache' in n: f = 'target_kv'
        elif 'scaled_fp4_quant_slice' in n: f = 'target_gelu_down_quant'
        else: f = 'target_norm_quant_rotary'
        events.append({'name': n, 'full_name': n, 'family': f, 'ts': e['ts'], 'dur': e['dur'],
                       'graph': g, 'grid': e['args'].get('grid'), 'block': e['args'].get('block'),
                       'registers': e['args'].get('registers per thread'),
                       'shared': e['args'].get('shared memory')})
    start = min(e['ts'] for e in events); end = max(e['ts'] + e['dur'] for e in events)
    result = stats(events, start, end, 16)
    result['target_graph_id'] = target_graph
    result['graph_launches'] = sum(e.get('name') == 'cudaGraphLaunch' for e in trace)
    result['graphs'] = {str(g): {'kernels': sum(e['graph'] == g for e in events),
                               'attention_calls': graph_counts[g],
                               'kernel_ms': sum(e['dur'] for e in events if e['graph'] == g) / 1000}
                        for g in sorted({e['graph'] for e in events})}
    result['annotation_names'] = [e['name'] for e in annotations]
    return events, result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    a = p.parse_args()
    out = {}
    collectors = [('native-d2', native), ('vllm-d2', lambda r: vllm(r, 2)), ('vllm-d3', lambda r: vllm(r, 3))]
    if (a.root / 'native-candidate-d2-capture.sqlite').exists():
        collectors.append(('native-candidate-d2', lambda r: native(r, 'native-candidate-d2-capture.sqlite')))
    for label, collect in collectors:
        events, summary = collect(a.root)
        out[label] = summary
        first = min(e['ts'] for e in events)
        with (a.root / f'{label}-timeline.csv').open('w') as stream:
            w = csv.writer(stream)
            w.writerow(['start_us', 'duration_us', 'family', 'graph', 'grid', 'block', 'registers', 'shared', 'kernel'])
            for e in sorted(events, key=lambda e: e['ts']):
                w.writerow([e['ts'] - first, e['dur'], e['family'], e['graph'], e['grid'], e['block'], e['registers'], e['shared'], e['full_name']])
        print(label, json.dumps({k: summary[k] for k in ('wall_ms', 'kernel_busy_union_ms', 'non_kernel_interval_ms', 'family_ms_per_group')}))
    (a.root / 'trace-comparison.json').write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__': main()
