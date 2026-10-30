"""Render the bounded October24 FP8 projection evidence from its immutable record."""
from pathlib import Path
import json,statistics
ROOT=Path(__file__).resolve().parents[1]
d=json.loads((ROOT/'artifacts/trellis35/fp8-projections-2026-10-24.json').read_text());assert d['status']=='accepted'
c=d['cohorts'];r=d['gain_ratios']
def rate(n):return c[n]['summary']['decode_tokens_per_second']['median']
rows=[]
for key in ['ab','ba','ordinary_adjacent']:
 x=r[key];rows.append(f"| {'Fixed-D2 '+key.upper() if key!='ordinary_adjacent' else 'Ordinary adjacent'} | {rate(x['parent']):.3f} | {rate(x['candidate']):.3f} | {100*x['gain_fraction']:+.4f}% |")
table='| Mode / comparison | Parent tok/s | Candidate tok/s | Change |\n|---|---:|---:|---:|\n'+'\n'.join(rows)+'\n'
report=ROOT/'docs/evidence/gemma4_26b/fp8-projections-2026-10-24.md'
report.write_text(f'''# Native fixed-R3 FP8 projections — 2026-10-24

Accepted Linux text change against frozen `{d['source_base_commit']}`. The
[acceptance record](../../../artifacts/trellis35/fp8-projections-2026-10-24.json)
retains all commands, source/binary/model/toolchain identities, positive and
negative samples, telemetry, actual shaders and raw-file hashes. Raw root:
`benchmarks/results/2026-10-24/85fc453/fp8-projections/`.

## End-to-end result

{table}
Each cohort uses three discarded warmups and ten retained process runs, the same
16,384 prompt IDs / 1,135 outputs / 1,134 decode intervals, context 17,519, batch
one, greedy/mask and FP8 KV on the RTX 5080 Laptop, driver 615.71.09, CUDA 13.4.92.
Both D2 bootstrap 95% gain intervals exclude zero: AB
[{100*r['ab']['bootstrap_95_percentile'][0]:.4f}%, {100*r['ab']['bootstrap_95_percentile'][1]:.4f}%],
BA [{100*r['ba']['bootstrap_95_percentile'][0]:.4f}%, {100*r['ba']['bootstrap_95_percentile'][1]:.4f}%]
(10,000 resamples, seed 20261029). Ordinary's adjacent interval includes zero.
This is a **small firmware-policy gain**, not the isolated kernel percentages
below. Whole-process high-util SM clock medians differ: parent 2,565 / 2,557 MHz,
candidate 2,580 / 2,580 MHz. The max-power/Dynamic Boost policy is unchanged;
there is no fixed-clock or exclusive algorithmic-cycle claim.

All 60 retained vectors reproduce the same 1,135 IDs. All 40 D2 runs retain 502
verification groups and 632 accepted / 1,004 proposed tokens. Weight/KV/workspace,
repack and graph accounting, precision/cache layouts, sampling, zero fallbacks
and zero token-loop allocations are unchanged. At 200 ms sampling, device-total
peaks remain 12,455 MiB ordinary / 12,725 MiB D2. One-Hz process monitoring shows
only expected gem16 compute PIDs and Xorg with 4 MiB. These samples include
loading/prefill/warmups and do not measure instantaneous allocation peaks or
isolated steady-decode clocks.

## Selected change

The 26B three-row QKV and split-K2 O projections now use three actual M16 A rows
in one native E4M3 MMA per K step, following the already implemented matrix
fragment map. Upper unused rows are zero. FP32 accumulation and per-row scaling
retain their K and multiplication order; O retains its two contiguous K halves
and exact `(first + second) * input_scale * weight_scale` boundary. Projection
outputs remain FP32 workspace values. No precision, source weight, persistent
tensor representation, prefill, sampling, cache or timing boundary changes.

Only exact 26B target shapes select these kernels: hidden 2,816, local
Q4,096/K2,048/V2,048, global Q8,192/K1,024 with **two bindings**, and O2,816 with
K4,096 or K8,192. Global raw-K-to-V reuse remains intact. M1, R2/R5 and 12B/non-target
launches retain their prior paths. New actual shader resources: QKV 36 registers,
zero stack/local/static shared, 8,544 dynamic shared bytes; O 40 registers, zero
stack/local and 96 static shared bytes. SASS and actual function-handle captures
are retained; all 24 production input/scale/output snapshots match the frozen
parent bit exactly.

The first packed QKV version introduces 405,504 shared-load bank conflicts;
original and naive packed both use 608,256 shared-load wavefronts. One scoped
32-byte gap between the three ephemeral shared activation rows adds 96 bytes/CTA,
removes those conflicts and reduces wavefronts to 202,752. Direct three-way
interleaved measurements show another 14.18–18.21% QKV operator gain versus naive
packing, exact in all 12 QKV cases. Naive T3 gains approximately 14–25% versus
parent in all 24 real cases. Both diagnostic and production use RDC=true.

These repeated-operator results do **not** transfer proportionally. Complete
1,135-output diagnostic traces retain exactly 12,550 local QKV calls, 2,510 global
Q/K calls and 15,060 O calls. Summed local QKV elapsed is 408.592 → 408.527 ms;
global Q/K 90.504 → 90.442 ms; O 306.215 → 298.367 ms. QKV is essentially unchanged
in the full model; O falls about 2.6%. Weights were already reused across three
rows in the parent, so packing reduces instruction work, not weight bytes.
Different repeated-operator cache behavior and full-model bandwidth are plausible
explanations; these traces are not pure same-clock causal accounting.

## Rejected M1 experiment and checks

Fresh M1 counters show substantial long-scoreboard stalls. One bounded two-entry
register fragment pipeline is exact in 24 actual cases but slower in **every**
case: QKV throughput is roughly 2.3–6.6% lower, O throughput 18.8–28.0% lower.
It is retained only in the
offline lab. No M1 candidate memory/full-model performance qualification or
production change is claimed. No further pipeline depth is inferred from its
counter alone.

The initial observer assumes three global bindings and stops before completing
capture. Source inspection identifies the real two-binding global contract;
failed commands/source snapshots remain unchanged. The corrected observer borrows
immutable resident weights and snapshots mutable data in a fixed 12 MiB arena.
It is absent from every primary run.

The selected packed/padded candidate passes 24 additional guarded synthetic
cases with diverse finite FP8 bytes, independent row/channel scales, zero rows,
partial eight-column tiles and split-K boundaries. Every FP32 output bit and
canary matches the original; standalone memcheck and synccheck report zero
errors. The exact measured production binary passes seven CUDA/host tests,
CUDA-suite memcheck, native 8/64 ordinary/D2, protected 12B/internal NVFP4 64-token
vectors identical to the qualified parent, explicit 12B product fixtures and a
full 16K D2 memcheck. Full 1,135-token traces, exact proposals and forced 0/1/2
acceptance/rollback continuations pass.

Reproduction commands and ordered scripts are in the record: parent counters,
`screen.py`, candidate/shared counters, `padding_screen.py`, `boundaries.py`,
`promote.py`, `qualify.py`, `actual_profile.py`, `compare.py`, `post_compare.py`.
The offline generator accepts `--source PARENT_SNAPSHOT` after promotion. No new
full 12.2 GB payload hash is claimed; immutable metadata/locks are inherited and
runtime lock/layout checks remain active. No fresh llama.cpp, Windows, image or
release qualification is claimed by this record. Earlier results remain intact.
''')
perf=ROOT/'docs/PERFORMANCE.md';text=perf.read_text();anchor='## 2026-10-24 fixed-D2 warp normalization';heading='## 2026-10-24 FP8 QKV/O row packing'
if heading in text:text=text[:text.index(heading)]+text[text.index(anchor):]
section=f'''{heading}

Frozen `85fc453` versus qualified three-row native FP8 projections, same16K
prompt /1,135 outputs/context17,519/FP8KV/greedy,3warmups+10retained per cohort:

{table}
This is a small same-policy end-to-end gain. Whole-process GPU clocks differ;
there is no fixed-clock claim. All60 vectors, D2 groups/acceptance, arenas and
precision stay exact; sampled peaks12,455/12,725MiB, process monitoring quiet.
The M1 two-entry pipeline loses all sampled cases and remains diagnostic only.
Actual dispatch/shaders, diverse numerical/sanitizer, full-memory, protected
profiles, proposals and forced continuations pass. Isolated cache-warm gains
are much larger than the whole-model effect; complete traces show QKV nearly
unchanged and O about2.6% shorter. See the
[qualification report](evidence/gemma4_26b/fp8-projections-2026-10-24.md) and
[acceptance record](../artifacts/trellis35/fp8-projections-2026-10-24.json).
No fresh llama or Windows/media/release claim; all earlier rows remain below.

'''
for a,b in {'same16K':'same 16K','/1,135':'/ 1,135','outputs/context17,519/FP8KV/greedy,3warmups+10retained':'outputs / context 17,519 / FP8 KV / greedy, 3 warmups + 10 retained','All60':'All 60','peaks12,455/12,725MiB':'peaks 12,455 / 12,725 MiB','about2.6%':'about 2.6%'}.items():section=section.replace(a,b)
assert anchor in text;perf.write_text(text.replace(anchor,section+anchor,1))
readme=ROOT/'README.md';text=readme.read_text();start=text.index('The **2026-10-24 Linux text optimization benchmark**');end=text.index('Earlier public-profile measurements',start)
replacement=f'''The **2026-10-24 Linux text optimization benchmark** on the RTX 5080 Laptop GPU
records **{rate('candidate-ordinary'):.3f} tok/s ordinary decode** and
**{min(rate('candidate-b'),rate('candidate-b2')):.3f}–{max(rate('candidate-b'),rate('candidate-b2')):.3f} tok/s fixed-D2** in the latest paired campaign.
Native FP8 QKV/O row packing adds **{100*r['ab']['gain_fraction']:.2f}–{100*r['ba']['gain_fraction']:.2f}% D2 throughput** against fresh controls;
ordinary is unchanged within variation. All 60 retained runs reproduce the same
1,135 IDs with unchanged D2 acceptance, arenas and precision. Sampled device-total
peaks are 12,455 / 12,725 MiB. The workload uses 16,384 prompt IDs, context 17,519,
batch one, greedy, FP8 KV, three warmups and ten retained runs per cohort. See the
[qualification and raw evidence](docs/evidence/gemma4_26b/fp8-projections-2026-10-24.md).
Absolute rates vary under firmware-managed Dynamic Boost; compare paired controls.
The preceding [M1 GateUp](docs/evidence/gemma4_26b/m1-gate-window-2026-10-24.md)
and [D2 attention](docs/evidence/gemma4_26b/d2-attention-warp-normalization-2026-10-24.md)
gains remain recorded. These text tests do not extend Windows, image or release
qualification; a fresh llama comparison is separate.

'''
readme.write_text(text[:start]+replacement+text[end:]);print(report)
