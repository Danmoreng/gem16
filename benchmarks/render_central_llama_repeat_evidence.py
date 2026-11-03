"""Render the November3 central llama repeat without editing historical results."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / 'artifacts/llama_cpp/central-repeat-2026-11-03.json'
d = json.loads(RECORD.read_text())
assert d['status'] == 'complete_measured_repeat'
c = d['cohorts']
r = d['comparisons']
old_record = ROOT / 'artifacts/trellis35/decode-five-steps-final-2026-10-24.json'
assert hashlib.sha256(old_record.read_bytes()).hexdigest() == d['historical_reference_record']['sha256']
old = json.loads(old_record.read_text())


def span(values):
    return f'{min(values):.3f}–{max(values):.3f}'


def current(mode, engine):
    return [data['summary']['decode_tokens_per_second']['median'] for key, data in c.items()
            if key.startswith(mode + '-' + ('native' if engine == 'native' else 'central'))]


def replace(path, marker, body, anchor):
    text = path.read_text()
    begin, end = f'<!-- {marker} -->', f'<!-- /{marker} -->'
    block = begin + '\n' + body.rstrip() + '\n' + end
    if begin in text:
        start = text.index(begin)
        stop = text.index(end, start) + len(end)
        text = text[:start] + block + text[stop:]
    else:
        assert anchor in text
        text = text.replace(anchor, block + '\n\n' + anchor, 1)
    path.write_text(text)


rows = []
for mode in ['ordinary', 'd2']:
    x = r[mode]['llama_today_vs_oct24']
    low, high = x['within_campaign_bootstrap_95']
    rows.append(f"| {mode} | {x['old_median']:.3f} | {x['new_median']:.3f} | {100*x['gain_fraction']:+.3f}% | [{100*low:+.3f}%, {100*high:+.3f}%] |")
table = '| llama.cpp mode | Oct24 median tok/s | Nov3 median tok/s | Change | Bootstrap 95% |\n|---|---:|---:|---:|---|\n' + '\n'.join(rows)
controls = '| Mode | Native Oct24 | Native Nov3 | Native change | Change in llama/native ratio |\n|---|---:|---:|---:|---:|\n' + '\n'.join(
    f"| {mode} | {r[mode]['native_today_vs_oct24']['old_median']:.3f} | {r[mode]['native_today_vs_oct24']['new_median']:.3f} | {100*r[mode]['native_today_vs_oct24']['gain_fraction']:+.3f}% | {100*r[mode]['relative_to_native_control']['ratio_of_ratios_minus_one']:+.3f}% |"
    for mode in ['ordinary', 'd2'])
cohort_table = '| Cohort | Median tok/s | Mean | SD | Sampled peak MiB | High-util SM MHz |\n|---|---:|---:|---:|---:|---:|\n' + '\n'.join(
    f"| {label} | {data['summary']['decode_tokens_per_second']['median']:.3f} | {data['summary']['decode_tokens_per_second']['mean']:.3f} | {data['summary']['decode_tokens_per_second']['standard_deviation']:.3f} | {data['monitoring']['sampled_peak_vram_mib']:.0f} | {data['monitoring']['high_util_sm_clock_median_mhz']:.0f} |"
    for label, data in c.items())
memory_table = '| llama.cpp mode | Oct24 sampled peak MiB | Nov3 sampled peak MiB | Change MiB |\n|---|---:|---:|---:|\n' + '\n'.join(
    f"| {mode} | {max(old['cohorts'][f'{mode}-llama-current-{order}']['telemetry']['sampled_peak_vram_mib'] for order in ['c','c2']):.0f} | {max(c[f'{mode}-central-{order}']['monitoring']['sampled_peak_vram_mib'] for order in ['b','b2']):.0f} | {max(c[f'{mode}-central-{order}']['monitoring']['sampled_peak_vram_mib'] for order in ['b','b2'])-max(old['cohorts'][f'{mode}-llama-current-{order}']['telemetry']['sampled_peak_vram_mib'] for order in ['c','c2']):+.0f} |"
    for mode in ['ordinary','d2'])
intro = f'''The **2026-11-03 central llama.cpp repeat** uses official master
`7071ba378afa` (0.8.0-dev), resolved from the shared CUDA installation, against
the retained October24 `3f03b17724ad` measurements. Each median combines two
ten-run cohorts after three discarded warmups per cohort.

{table}

The unchanged gem16 executable was remeasured as a machine control. The NVIDIA
driver changed from 615.71.09 to 615.78.08 and clocks remain firmware-managed;
the central build uses standard Flash Attention quant combinations, where the
old build enabled all combinations. Generated streams also differ despite the
same immutable GGUF files and effective sampling settings. These are measured
changes for the recorded configurations, not exclusive source-code causality or
a new quality-parity claim. See the
[complete repeat, controls and raw evidence](docs/evidence/gemma4_26b/central-llama-repeat-2026-11-03.md).
'''
report_intro = intro.split(' See the\n',1)[0]
report = ROOT / 'docs/evidence/gemma4_26b/central-llama-repeat-2026-11-03.md'
report.write_text(f'''# Central llama.cpp repeat — 2026-11-03

{report_intro}

The [privacy-safe repeat record](../../../artifacts/llama_cpp/central-repeat-2026-11-03.json)
retains exact commands, input/source/executable/library hashes, all rate samples,
output IDs, server counters, telemetry summaries and raw evidence identities.
Local paths are parameterized, and hostnames, device UUIDs and process identifiers
are removed from the published derivative. Hardware models, driver/toolchain
versions and every measurement remain unchanged. File hashes refer to the original
private raw files, which remain intact locally. Regenerate this export with
`python benchmarks/export_central_llama_repeat_evidence.py` before rendering.
Raw root: `benchmarks/results/2026-11-03/6b71ee3/central-llama-7071ba3/`.
Bootstrap intervals use independent resampling of the twenty retained values per
campaign, 10,000 draws with seed 20261103. They describe sample variation and do
not encompass all cross-day driver/firmware/build/output differences. No sample
is discarded or selected by its result.

## Machine control

{controls}

The control is the frozen, qualified `f6de38e` runtime executable, whose source
also matches current gem16 HEAD for all production `src/` files. No native engine
rebuild or code modification is part of this repeat. The ratio-of-ratios helps
detect broad machine drift; it does not isolate an upstream kernel improvement.

## Identical workload and disclosed differences

For ordinary and maximum-D2: 16,384 identical prompt token IDs, 1,135 outputs,
1,134 timed decode intervals, requested context 17,519, batch one, greedy, seed zero,
the same suppressed IDs 1/50/106/258882/258883, no prompt-cache reuse and all
per-token weights on GPU. Llama internally pads context to 17,664. Eight threads,
prefill batch 2048 / microbatch 512, Flash Attention / CUDA graphs, Q8_0 K/V, and the same
explicit token-embedding CUDA overrides are retained. Effective server generation
settings match October24 exactly. Native retains its FP8 KV and Trellis35 W4A8
Target. This is a text-Target measurement; no native Vision tower is loaded.

The exact same Google QAT Q4_0 GGUF (14,439,363,584 bytes) and Assistant GGUF
(251,939,328 bytes) were verified against their immutable prior source hashes;
no model download, conversion or quantization occurred. GGUF embeds tokenizer
metadata; native tokenizer storage is separate. No size or quality equivalence
between the different native/llama quantizations is asserted.

The prior llama build is `3f03b17724ad8b69040825f45c43c59b092a0244` (b11866).
The current central build is `7071ba378afa96dcd69c41e1ec2b4912364dbee4`.
Its shallow checkout reports build 1; the full commit, not that build counter,
identifies this version. CUDA architecture 120a-real, Release and CPU-native OFF
are retained. The central updater removes CMake intermediates after building;
the original per-version manifest and exact linked libraries are retained here.
The previous executable is not run in this campaign. Upstream freshness is
checked by the central resolver at each benchmark start; the versioned resolved
path prevents a later `current` symlink switch from mixing binaries.

Current llama's output sequences differ from October24 at zero-based index 103
(ordinary) and 102 (D2). All repeats match their current mode's preflight; all
native outputs remain bit-exact to the qualified prior fixture. Cross-engine
and cross-version generated expert-routing work is therefore not identical.
There is no new text-quality qualification in this performance repeat.

The current D2 server reports 497 base groups, 993 proposals and 637 accepted tokens;
the prior version used 497/994/637. With maximum 2 / minimum 1 and the independently
verified cumulative server counters, current totals prove 496 depth-two groups and
one depth-one group. Do not interpret the helper's configured `d2_groups` label as an
observed per-depth histogram. Native remains fixed-D2 with 502 groups, 1,004 proposals
and 632 accepted tokens. Output count and decode timing boundaries are unchanged.

## All retained cohorts

{cohort_table}

{memory_table}

The GPU residency logs continue to show full Target/Assistant offload. No GGUF
file or requested KV precision was changed. These are sampled device-total
peaks; the cause of the allocation reduction is not separately isolated here,
and no new maximum-context capacity is qualified by this observation.

Order is native/current/current/native for ordinary, then the same order for D2.
Every cohort starts at or below 50 C with an idle compute GPU. Power policy remains
max-power with active nvidia-powerd Dynamic Boost. Monitoring samples the whole
device every 200 ms and processes every second, including loading/prefill/warmups.
High-util clocks are medians at utilization >=90%, SM >1000 MHz and memory >1000 MiB;
they are not isolated steady-decode clocks. Sampled memory is not an instantaneous
allocator peak. Raw mean per-request ITL is retained; per-token p50/p95/p99 is not
collected and is not claimed.

## Existing uncommitted centralization changes

The initial worktree contains eleven modified files and one new central helper. The
build entry points delegate to the shared updater; comparison/quality scripts
resolve the current central CUDA server, require verified official freshness and
record revision plus executable/shared-library hashes. Resumed quality runs reject
a changed reference or missing provenance. Two new Python tests cover central
resolution/rejection; all 22 workload/wrapper tests and shell syntax checks pass.
The additional Studio packaging and Trellis35 quality-run changes are separate
from llama performance. The original initial patch and file hashes remain in the
private raw record; the published patch has parameterized paths. The benchmark
task preserved the initial dirty files. A subsequent publication privacy check
replaces a user-specific Python bootstrap path and removes GPU UUID collection
from the benchmark wrapper; these changes do not affect the measured inference.
''')
replace(ROOT / 'README.md', 'CENTRAL-LLAMA-REPEAT-2026-11-03', intro,
        '<!-- DECODE-FIVE-STEPS-FINAL-2026-10-24 -->')
replace(ROOT / 'docs/PERFORMANCE.md', 'CENTRAL-LLAMA-REPEAT-2026-11-03',
        '## 2026-11-03 central llama.cpp repeat\n\n' + intro.replace('(docs/evidence/gemma4_26b/', '(evidence/gemma4_26b/'),
        '<!-- DECODE-FIVE-STEPS-FINAL-2026-10-24 -->')
print(report)
