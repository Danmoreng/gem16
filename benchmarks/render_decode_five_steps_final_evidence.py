"""Render the frozen twelve-cohort October24 native/current-llama comparison."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / 'artifacts/trellis35/decode-five-steps-final-2026-10-24.json'
d = json.loads(RECORD.read_text())
assert d['status'] == 'measured' and len(d['cohorts']) == 12
c = d['cohorts']
r = d['comparisons']
scope_path = ROOT / 'artifacts/trellis35/decode-five-steps-final-storage-scope-2026-10-24.json'
storage_scope = {
    'schema_version': 1,
    'kind': 'storage-boundary clarification; no measurement, rate, size or historical record is changed',
    'frozen_record': {'path': str(RECORD), 'sha256': hashlib.sha256(RECORD.read_bytes()).hexdigest()},
    'corrects_field': 'quantization_sizes.size_boundary',
    'reason': 'The frozen boundary label excludes tokenizer too broadly. GGUF includes embedded tokenizer metadata; the native tokenizer is external.',
    'native_model_gem16_bytes': d['quantization_sizes']['native_target']['bytes'],
    'llama_complete_gguf_bytes': d['quantization_sizes']['llama_target']['bytes'],
    'comparison': 'native compiled device image versus complete GGUF file; an approximate target storage comparison, not a tokenizer-normalized pure-weight comparison',
    'embedded_tokenizer_evidence': 'ordinary-llama-current-c.server.log contains tokenizer.ggml.model and tokenizer.ggml.tokens arr[str,262144]; original log hash is pinned in the frozen record',
    'unchanged_exclusions': 'Assistant, KV, workspaces and driver allocations are outside these two target-file sizes',
}
if scope_path.exists():
    assert json.loads(scope_path.read_text()) == storage_scope
else:
    scope_path.write_text(json.dumps(storage_scope, indent=2) + '\n')


def rate(label):
    return c[label]['summary']['decode_tokens_per_second']['median']


def bounds(values, digits=3, suffix=''):
    return f'{min(values):.{digits}f}–{max(values):.{digits}f}{suffix}'


def cohort_range(mode, role):
    return bounds([rate(name) for name in c if name.startswith(f'{mode}-{role}-')])


def gain_range(mode):
    return bounds([100 * r[f'{mode}-cumulative_{order}']['gain_fraction'] for order in ['ab', 'ba']], 2, '%')


def replace_block(path, marker, body, anchor):
    text = path.read_text()
    start, end = f'<!-- {marker} -->', f'<!-- /{marker} -->'
    replacement = start + '\n' + body.rstrip() + '\n' + end
    if start in text:
        first = text.index(start)
        last = text.index(end, first) + len(end)
        text = text[:first] + replacement + text[last:]
    else:
        assert anchor in text
        text = text.replace(anchor, replacement + '\n\n' + anchor, 1)
    path.write_text(text)


cohort_rows = []
for label, cohort in c.items():
    summary = cohort['summary']['decode_tokens_per_second']
    t = cohort['telemetry']
    ci = summary['confidence_interval_95']
    cohort_rows.append(f"| {label} | {summary['median']:.3f} | {summary['mean']:.3f} | {summary['standard_deviation']:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] | {t['sampled_peak_vram_mib']:.0f} | {t['high_util_sm_clock_median_mhz']:.0f} |")
cohort_table = '| Cohort | Median tok/s | Mean | SD | Mean 95% CI | Peak MiB | SM MHz |\n|---|---:|---:|---:|---|---:|---:|\n' + '\n'.join(cohort_rows)
gain_rows = []
for name, item in r.items():
    low, high = item['bootstrap_95_percentile']
    gain_rows.append(f"| {name} | {100*item['gain_fraction']:+.3f}% | [{100*low:+.3f}%, {100*high:+.3f}%] |")
gain_table = '| Candidate / baseline − 1 | Median throughput change | Bootstrap 95% |\n|---|---:|---|\n' + '\n'.join(gain_rows)
comparison_table = '| Mode | Day-start gem16 | Final gem16 | Current llama.cpp b11866 |\n|---|---:|---:|---:|\n' + '\n'.join(
    f'| {mode} | {cohort_range(mode, "day-start")} | {cohort_range(mode, "native-final")} | {cohort_range(mode, "llama-current")} |'
    for mode in ['ordinary', 'd2'])

report = ROOT / 'docs/evidence/gemma4_26b/decode-five-steps-final-2026-10-24.md'
report.write_text(f'''# Five-step decode campaign and current llama.cpp — 2026-10-24

The bounded five experiments are finished. Production retains M1 GateUp codec-window
sharing, fixed-D2 attention normalization, and three-row native FP8 QKV/O packing.
T3 GateUp pipeline variants and NVFP4 output-head variants remain rejected diagnostics.
The final accepted runtime is `f6de38e`; later commits preserve diagnostics and fix
host benchmark group reporting without changing throughput timing or runtime code.

The [frozen record](../../../artifacts/trellis35/decode-five-steps-final-2026-10-24.json)
contains every sample, command, output vector/hash, model/compiler/library identity,
actual source hash, telemetry and raw-file hash. Raw root:
`benchmarks/results/2026-10-24/30a61fc/final-comparison/`.

## Final end-to-end result

{comparison_table}

Rates are tok/s. Each range spans the medians of the two independently retained
ten-run cohorts, not the fastest runs or a cross-day comparison. Direct cumulative
gain against the frozen day-start executable is **{gain_range('ordinary')} ordinary**
and **{gain_range('d2')} D2**. The following table gives both experiment orders,
including the actual gap to current llama.cpp; negative native/llama values mean
native is slower. This is a **native-format/kernel baseline**, with different
quantization, KV formats and generated streams; it is not exact quality/semantic
parity or proof of a generic engine advantage.

{gain_table}

Bootstrap intervals resample independently inside each ten-run cohort (10,000
resamples, sequential seed 20261030), using the ratio of medians. AB/BA describes
experiment order; request samples are not assumed to be statistically paired.
No run is discarded as an outlier and no repetition is selected by its result.
Both native/llama point estimates favor llama in each mode. In D2 the reverse-order
interval includes zero, so that order does not resolve the small remaining gap.
The campaign makes no native superiority claim and does not repeat measurements
to force one.

## Workload, formats and actual behavior

All twelve cohorts run on the RTX 5080 Laptop, driver 615.71.09, CUDA 13.4.92,
AMD Ryzen 9 9955HX3D. Each has three discarded warmups and ten retained requests,
the same 16,384 prompt IDs, 1,135 outputs and 1,134 measured decode intervals,
batch one, greedy sampling, seed zero, suppression of IDs 1/50/106/258882/258883,
and no prompt-cache reuse. First output belongs to prefill; load/prefill are
excluded from the decode timer. Order is A,B,C,C2,B2,A2 for ordinary, then D2;
A is day-start, B final native, C freshly built current llama. GPU is idle and
cooled to at most 50 C before each cohort. No overlapping compute/build jobs.

Requested context is 17,519. Llama internally pads to 17,664; the explicit
context and full GPU-layer settings prevent auto-fit from changing parameters.
Target offload is 31/31 layers, Assistant 5/5, including CUDA token embeddings.
Llama prefill batching is 2048/512, eight inference threads, one request slot,
Flash Attention and CUDA graphs enabled. Its automatic MMQ/cuBLAS selection is
retained; no forced diagnostic fallback. Native prefill batching remains the
qualified model path. This is batch-one decode, not continuous batching.

| Target weights, Assistant excluded | Stored payload | Runtime / KV |
|---|---:|---|
| gem16 Trellis35 W4A8 | 12,204,692,480 B (12.205 GB) | BF16 state, FP8 attention/dense and expert activation boundaries, NVFP4 non-routed families; FP8 E4M3FN KV |
| Official Google QAT Q4_0 GGUF | 14,439,363,584 B (14.439 GB) | 265 Q4_0 tensors, one Q6_K tensor plus unquantized tensors; Q8_0 KV |

Trellis35 target payload is 15.476% smaller relative to GGUF. These are stored
file sizes, not whole-device memory and not identical effective precision.
GGUF includes embedded tokenizer/model metadata; native `model.gem16` is the
compiled device image and its tokenizer is external. This is an approximate
target storage comparison, not a tokenizer-normalized pure-weight comparison.
The [storage-boundary clarification](../../../artifacts/trellis35/decode-five-steps-final-storage-scope-2026-10-24.json)
corrects the overly broad tokenizer exclusion in the frozen record's boundary
label while preserving its verified byte counts and all original measurements.
The native weight arena is exactly the 12,204,692,480-byte compiled device image.
Llama's logged target CUDA model buffer is 14,332.92 MiB; Assistant is 225.21 MiB.
Its Assistant GGUF is 251,939,328 bytes; native source Assistant Safetensors is
258,317,280 bytes. Tensor layouts/metadata and runtime arenas differ, so file
sizes must not be substituted for GPU allocation totals.

All 80 retained native output vectors match exactly across ordinary/D2 and
day-start/final. All 40 retained llama outputs match their own mode's preflight;
llama ordinary and D2 diverge at zero-based output index 197. Old b11552 already
had ordinary/D2 divergence, so this observation does not establish a new upstream
regression. Different outputs imply different subsequent expert routing and work;
no exclusive precision/rounding explanation or new model-quality result is claimed.

Native D2 retains 502 base verification groups, 1,004 proposals and 632 accepted
tokens. Current llama D2 retains 497 groups, 994 proposals and 637 accepted tokens.
Its cumulative server `#calls(b,g,a)` counters independently verify all thirteen
requests per cohort, including warmups; 994 = 2 × 497 proves full depth two for
these requests. No output-token shortening or ordinary tail inflates throughput.
The first prefill token is excluded from inferred groups and throughput intervals.
Historical preflight JSON keeps its original 498-group helper inference; the
separate counter observation explains the corrected 497 count without rewriting it.

## All retained cohorts and environment

{cohort_table}

The table's interval describes the helper's mean estimate; median is primary.
Per-request average inter-token latency is retained in the record. Within-request
per-token p50/p95/p99 latency is not collected by this workload and is not claimed;
speculative bursts must not be presented as a measured per-token distribution.
Whole-device telemetry samples every 200 ms and process monitoring every second.
SM clock is the median among whole-process samples with GPU utilization at least
90%, SM above 1000 MHz and resident memory above 1000 MiB. It includes prefill
and warmups, not an isolated steady-decode frequency. Clocks are not locked:
the unchanged max-power/nvidia-powerd Dynamic Boost policy remains active.
Sampled peaks include initialization/prefill/workspaces and driver allocations;
they are not instantaneous allocator peaks. Raw monitoring must show only the
expected compute executable and Xorg (4 MiB). Native per-mode weight/KV/workspace,
graph/repack accounting, precision, cache and sampling stay exact before/after;
every native request reports zero fallbacks and token-loop allocations.
These are text-Target characterizations, with the Assistant added for D2. The
Vision tower is not loaded; these peaks are not the complete Compact Vision
product's resident footprint.

## Source and qualification

Llama is freshly built official upstream `{d['llama']['commit']}` (b11866),
CUDA `120a-real`, Release, native CPU tuning off, all Flash Attention quants and
graphs enabled, automatic MMQ/cuBLAS choice. The previous b11552 checkout and
binaries are not used. The source archive is outside any parent Git repository;
explicit correct build-number/commit definitions avoid inherited archive metadata.
Exact server and linked ggml/llama library hashes are pinned in the record.
`BLACKWELL_NATIVE_FP4=1` alone denotes compiled capability, not actual activation
precision or an instruction-level explanation of performance.

Separate whole-process Nsight captures observe
{d['llama_actual_dispatch']['ordinary']['kernel_count']:,} CUDA kernels in ordinary and
{d['llama_actual_dispatch']['d2']['kernel_count']:,} in D2, including initialization,
the server's built-in warmup, prefill and one complete 1,135-output request.
Actual demangled kernel families and their counts are retained in
`llama_actual_dispatch` and the original SQLite/reports. Their profiled durations
are diagnostic and never substituted for the unprofiled primary samples.
Observed families include `mul_mat_vec_q` with type 2 (Q4_0), type 14 (Q6_K),
three-row variants and `mul_mat_vec_q_moe` in D2, plus Q8 activation quantization.
No recorded demangled family is instantiated with type 39/40 (MXFP4/NVFP4).
This is function-dispatch evidence, not a complete instruction-by-instruction
causal attribution of the performance gap.
The pinned `ggml_cuda_mmq_get_prec_src1` source allows native W4A4 only for
NVFP4/MXFP4 tensors on Blackwell; Q4_0 uses Q8 activation precision in that MMQ
path. This conditional source fact is distinct from observed per-operator
dispatch and does not identify an exclusive cause of the remaining speed gap.

The three accepted steps each retain actual-input/operator/resource evidence,
bitwise numerical and guarded boundaries, relevant CUDA sanitizers, full 16K
model memory checks, protected 12B/internal NVFP4 regressions, complete generated
IDs, D2 proposals/forced acceptance/rollback continuations, and fresh end-to-end
controls. The current production source hashes match the accepted FP8 record.
The rebuilt final CLI's loaded ELF/CUDA bytes match its qualified parent; only
non-loadable nvcc temporary-PID symbol metadata differs, as the pinned restoration
proof records. No primary run loads diagnostic observers or source repacks.
The final read-only qualification audit rehashes
{d['qualification_audit']['unique_files_rehashed']:,} unique evidence files and checks
actual seven-test CUDA/host logs, 12B product fixtures, native/protected outputs,
full-memory/proposal/forced-continuation results and source restoration. Earlier
failed/rejected experiments remain in their original evidence rather than being
reclassified as passed or silently removed.

- [Step 1: accepted M1 GateUp](m1-gate-window-2026-10-24.md).
- [Step 2: rejected T3 pipeline](t3-gate-pipeline-rejected-2026-10-24.md), including the [production-RDC re-screen](t3-gate-rdc-audit-2026-10-24.md).
- [Step 3: accepted D2 attention](d2-attention-warp-normalization-2026-10-24.md).
- [Step 4: accepted native FP8 projections](fp8-projections-2026-10-24.md).
- [Step 5: rejected Target/Assistant heads](nvfp4-output-heads-2026-10-24.md).

This campaign establishes Linux text performance and preserves prior quality
boundaries. It adds no Windows/media/release qualification, does not reopen
closed broader tuning work, and makes no quality-equivalence claim. The owner's
subsequent instruction authorizes pushing the tested code and documentation to
`main`; a new release tag or downloadable package publication is separate.
''')

compact_table = comparison_table
intro = f'''The **2026-10-24 final Linux text comparison** directly measures the frozen
day-start engine, the final accepted engine, and freshly built official
**llama.cpp b11866 (`3f03b17724ad`)** on the RTX 5080 Laptop. Ranges below are
the medians from both experiment orders, in tok/s.

{compact_table}

The direct cumulative gain is **{gain_range('ordinary')} ordinary** and
**{gain_range('d2')} fixed-D2**. Each cohort retains ten runs after three warmups,
with identical 16,384 prompt IDs / 1,135 outputs / batch one / greedy sampling.
This is a **native-format/kernel baseline**: Trellis35 W4A8 (12.20 GB) versus
official QAT Q4_0 GGUF (14.44 GB), FP8 versus Q8_0 KV, and different generated
streams across engines. It does not prove exact numerical or quality parity.
File sizes include their format metadata; GGUF embeds tokenizer metadata while
the native tokenizer is external.
Requested context is 17,519; llama internally pads to 17,664. Both keep all
weights on GPU without prompt-cache reuse. Dynamic Boost clocks differ;
both modes' point estimates favor llama, with the reverse D2 interval spanning zero.
All samples, confidence intervals, VRAM, formats and limitations are in the
[final comparison report](docs/evidence/gemma4_26b/decode-five-steps-final-2026-10-24.md).
'''
replace_block(ROOT / 'README.md', 'DECODE-FIVE-STEPS-FINAL-2026-10-24', intro,
              'The **2026-10-24 Linux text optimization benchmark**')
readme = ROOT / 'README.md'
text = readme.read_text().replace('in the latest paired campaign.', 'in the earlier FP8 projection qualification campaign.')
text = text.replace('qualification; a fresh llama comparison is separate.',
                    'qualification; the separate final comparison above includes current llama.cpp.')
readme.write_text(text)
perf_intro = '## 2026-10-24 final five-step / current llama.cpp comparison\n\n' + intro.replace(
    '(docs/evidence/gemma4_26b/', '(evidence/gemma4_26b/')
replace_block(ROOT / 'docs/PERFORMANCE.md', 'DECODE-FIVE-STEPS-FINAL-2026-10-24',
              perf_intro, '<!-- T3-GATE-RDC-AUDIT-2026-10-24 -->')
print(report)
