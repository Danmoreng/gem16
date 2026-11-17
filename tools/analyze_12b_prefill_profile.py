"""Attribute an existing12B Prefill SQLite trace without repeating GPU work.

Kernel execution times are joined to launch APIs and then to existing host NVTX
layer ranges. Logical GEMM shapes are source/order attribution, not recorded
device argument values. Actual launch/resource shapes are retained separately.
"""
from collections import defaultdict
from pathlib import Path
import argparse
import json
import sqlite3


def family(name):
    if 'PrepareWeightKernel' in name:
        return 'weight_format_preparation'
    if 'cutlass' in name.lower():
        if any(t in name for t in ['float_e2m1', 'nv_float4', 'mx_float4']):
            return 'NVFP4_CUTLASS'
        if 'float_e4m3' in name:
            return 'FP8_CUTLASS'
        return 'CUTLASS_unclassified'
    if 'Attention' in name:
        if 'Global' in name:
            return 'global_attention'
        if 'Local' in name:
            return 'local_attention'
        return 'attention_unclassified'
    if 'Norm' in name or 'Quant' in name or 'Interleave' in name:
        return 'norm_quant_format_preparation'
    return 'other_explicit_names'


def analyze(path, chunk, *, require_prefill_only=True):
    assert chunk in [4096, 8192]
    with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as db:
        ranges = db.execute('SELECT n.start,n.end,n.globalTid,COALESCE(n.text,s.value) FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id=n.textId WHERE n.end IS NOT NULL ORDER BY n.start').fetchall()
        outer = [r for r in ranges if r[3] == 'gem16.prefill']
        assert len(outer) == 1, outer
        begin, end, tid, _ = outer[0]
        layers = [r for r in ranges if r[3] == 'gem16.prefill.layer' and r[2] == tid and begin <= r[0] <= r[1] <= end]
        assert len(layers) == 48 * (16384 // chunk), len(layers)
        launches = db.execute('SELECT correlationId,start,end,globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE start>=? AND start<=? AND globalTid=?', (begin, end, tid)).fetchall()
        by_correlation = {}
        for correlation, start, stop, thread in launches:
            assert correlation not in by_correlation, correlation
            by_correlation[correlation] = (start, stop, thread)
        rows = db.execute('SELECT s.value,k.start,k.end,k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ,k.registersPerThread,k.staticSharedMemory,k.dynamicSharedMemory,k.localMemoryPerThread,k.correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName ORDER BY k.start').fetchall()

    chosen = [row for row in rows if row[-1] in by_correlation]
    assert chosen, 'no kernels correlated to Prefill launch APIs'
    # The capture must not silently include startup/Decode kernels. A changed
    # profiler schema or graph boundary requires investigation, not guesswork.
    if require_prefill_only:
        assert len(chosen) == len(rows), (len(chosen), len(rows))
    names = defaultdict(lambda: {'calls': 0, 'summed_gpu_us': 0., 'shapes': set()})
    buckets = defaultdict(float)
    intervals, layer_kernels = [], defaultdict(list)
    outside = []
    for row in chosen:
        name, start, stop = row[:3]
        elapsed = (stop - start) / 1000
        names[name]['calls'] += 1
        names[name]['summed_gpu_us'] += elapsed
        names[name]['shapes'].add(row[3:-1])
        buckets[family(name)] += elapsed
        if intervals and start <= intervals[-1][1]:
            intervals[-1][1] = max(intervals[-1][1], stop)
        else:
            intervals.append([start, stop])
        launch = by_correlation[row[-1]][0]
        matches = [i for i, r in enumerate(layers) if r[0] <= launch <= r[1]]
        assert len(matches) <= 1, matches
        (layer_kernels[matches[0]] if matches else outside).append(row)

    phases, logical = [], defaultdict(lambda: {'calls': 0, 'summed_gpu_us': 0.})
    for i, host in enumerate(layers):
        kernels = layer_kernels[i]
        assert kernels, i
        fs = defaultdict(float)
        for row in kernels:
            fs[family(row[0])] += (row[2] - row[1]) / 1000
        global_calls = [r for r in kernels if 'OnlineGlobalAttentionFp8Kernel' in r[0]]
        local_calls = [r for r in kernels if 'OnlineLocalAttentionFp8Kernel' in r[0]]
        assert len(global_calls) + len(local_calls) == 1, (i, len(global_calls), len(local_calls))
        kind = 'global' if global_calls else 'local'
        fp8 = [r for r in kernels if family(r[0]) == 'FP8_CUTLASS']
        nvfp4 = [r for r in kernels if family(r[0]) == 'NVFP4_CUTLASS']
        expected_fp8 = [('Q', 8192, 3840), ('K_reused_as_V', 512, 3840), ('O', 3840, 8192)] if kind == 'global' else [('Q', 4096, 3840), ('K', 2048, 3840), ('V', 2048, 3840), ('O', 3840, 4096)]
        expected_nvfp4 = [('Gate', 15360, 3840), ('Up_GELU_quant', 15360, 3840), ('Down', 3840, 15360)]
        assert len(fp8) == len(expected_fp8) and len(nvfp4) == 3, (i, len(fp8), len(nvfp4))
        for bucket, actual, expected in [('FP8', fp8, expected_fp8), ('NVFP4', nvfp4, expected_nvfp4)]:
            for row, (operation, n, k) in zip(actual, expected):
                key = (bucket, operation, kind, chunk, n, k)
                logical[key]['calls'] += 1
                logical[key]['summed_gpu_us'] += (row[2] - row[1]) / 1000
        phases.append({'chunk_index': i // 48, 'prompt_start_position': (i // 48) * chunk,
                       'tokens': chunk, 'layer_index': i % 48, 'attention_kind': kind,
                       'kernel_calls': len(kernels), 'summed_gpu_us': sum(fs.values()),
                       'host_range_us': (host[1] - host[0]) / 1000,
                       'families_summed_gpu_us': dict(fs)})
    for index in range(16384 // chunk):
        assert sum(p['attention_kind'] == 'global' for p in phases if p['chunk_index'] == index) == 8
    total = sum(buckets.values())
    span = (max(r[2] for r in chosen) - min(r[1] for r in chosen)) / 1000
    covered = sum(stop - start for start, stop in intervals) / 1000
    prepare = sum(n['calls'] for name, n in names.items() if 'PrepareWeightKernel' in name)
    assert prepare == 3 * len(layers), prepare
    return {
        'prefill_kernel_calls': len(chosen), 'summed_kernel_us': total,
        'kernel_wall_span_us': span, 'host_prefill_range_us': (end - begin) / 1000,
        'non_kernel_time_within_kernel_span_us': span - covered,
        'non_kernel_note': 'includes copies/memsets and scheduling gaps; not isolated CPU overhead',
        'all_kernels_correlated_to_prefill_launch_apis': True,
        'other_phase_kernels_excluded': len(rows) - len(chosen),
        'weight_repack_calls': prepare,
        'families': {name: {'summed_gpu_us': value, 'fraction_of_kernel_sum': value / total}
                     for name, value in sorted(buckets.items(), key=lambda kv: -kv[1])},
        'layer_chunk_phases': phases,
        'outside_layer_ranges': {'kernel_calls': len(outside), 'summed_gpu_us': sum((r[2] - r[1]) / 1000 for r in outside),
                                 'kernel_names': sorted(set(r[0] for r in outside))},
        'logical_gemm_shape_attribution': [{
            'family': key[0], 'operation': key[1], 'layer_kind': key[2],
            'M_tokens': key[3], 'N_output': key[4], 'K_contracting': key[5], **data,
        } for key, data in sorted(logical.items())],
        'logical_shape_limit': 'source/order attribution from unchanged12B wrappers plus one attention and expected FP8/NVFP4 launches per NVTX layer; not a capture of kernel argument values',
        'kernels': [{'name': name, 'calls': info['calls'], 'summed_gpu_us': info['summed_gpu_us'],
                     'grid_block_register_static_dynamic_shared_local_shapes': sorted(info['shapes'])}
                    for name, info in sorted(names.items(), key=lambda kv: -kv[1]['summed_gpu_us'])],
    }


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('sqlite', type=Path)
    p.add_argument('chunk', type=int, choices=[4096, 8192])
    p.add_argument('output', type=Path)
    a = p.parse_args()
    with a.output.open('x') as f:
        json.dump(analyze(a.sqlite, a.chunk), f, indent=2)
