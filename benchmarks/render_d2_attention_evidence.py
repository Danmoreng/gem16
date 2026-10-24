"""Render the accepted 2026-10-24 attention record; historical raw data stays immutable."""
from pathlib import Path
import json,statistics
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'benchmarks/results/2026-10-24/c48124d/d2-attention'
record_path=ROOT/'artifacts/trellis35/d2-attention-warp-normalization-qualified-2026-10-24.json';d=json.loads(record_path.read_text());assert d['status']=='accepted'
c=d['cohorts'];r=d['d2_gain_ratios']
def rate(n):return c[n]['summary']['decode_tokens_per_second']['median']
rows=[]
for label in ['ab','ba','ordinary_adjacent']:
 x=r[label];rows.append(f"| {'Fixed-D2 '+label.upper() if label!='ordinary_adjacent' else 'Ordinary adjacent'} | {rate(x['parent']):.3f} | {rate(x['candidate']):.3f} | {100*x['gain_fraction']:+.4f}% |")
table='| Mode / comparison | Parent decode tok/s | Candidate decode tok/s | Change |\n|---|---:|---:|---:|\n'+'\n'.join(rows)+'\n'
ci='; '.join(f"{label.upper()} [{100*r[label]['bootstrap_95_percentile'][0]:.4f}%, {100*r[label]['bootstrap_95_percentile'][1]:.4f}%]" for label in ['ab','ba'])
vram_d2=max(c[n]['telemetry']['sampled_peak_vram_mib'] for n in ['parent-a','candidate-b','candidate-b2','parent-a2']);vram_o=max(c[n]['telemetry']['sampled_peak_vram_mib'] for n in ['parent-ordinary','candidate-ordinary'])
path=ROOT/'docs/evidence/gemma4_26b/d2-attention-warp-normalization-2026-10-24.md'
path.write_text(f'''# Fixed-D2 local/global warp normalization — 2026-10-24

Accepted Linux text optimization against frozen main `{d['source_base_commit']}`.
The parent runtime is the qualified M1 GateUp binary from e0c7fe5; c48124d only
adds rejected diagnostic evidence. Exact binaries, source hashes, toolchain,
commands, raw samples, telemetry and every raw-file identity are retained in
[the acceptance record](../../../artifacts/trellis35/d2-attention-warp-normalization-qualified-2026-10-24.json).
Raw root: `{OUT.relative_to(ROOT)}`.
Monitored confirmation: `benchmarks/results/2026-10-24/c48124d/d2-attention-followup/`.
The qualified record preserves and supersedes the initial acceptance evaluation;
all original data, including the initial BA pair, remain accessible.

## End-to-end result

{table}
Same RTX 5080 Laptop / driver 615.71.09 / CUDA 13.4.92, firmware-managed max-power and
Dynamic Boost, fixed 16,384 prompt IDs, 1,135 output IDs / 1,134 decode intervals,
context 17,519, batch 1, greedy/mask, FP8 KV. Each cohort uses 3 warmups + 10 retained
process runs. D2 uses the original AB plus a process-monitored repeat BA; Ordinary is one
adjacent non-regression pair. The original BA remains supplementary evidence;
no sample is removed. The repeat adds 1 Hz nvidia-smi pmon to both cohorts.
Bootstrap 95% gain intervals (10,000 resamples; per-pair seeds in the record): {ci}.
Maximum sampled device-total VRAM: {vram_o:,.0f} MiB ordinary / {vram_d2:,.0f} MiB D2.
All 80 retained vectors match all 1,135 IDs. All 60 D2 runs retain 502 verification
groups, 632 accepted / 1,004 proposed tokens. All arenas, precision/cache layouts,
sampling, fallback 0, persistent repack 0 and token-loop allocation 0 are unchanged.
Whole-process telemetry includes loading/prefill/warmups; it is not isolated
steady-decode clocks or an instantaneous allocation peak. Clocks are not fixed.
The original candidate-b2 cohort has a transient 13,129 MiB device-total peak and
one 185.309 tok/s retained sample; reported weight/KV/workspace/repack/graph bytes
are unchanged. Allocation source is unknown. The repeated BA pair shows only
the expected gem16 compute processes and 4 MiB Xorg; its device peaks are retained
in the qualified record. The 13,129 MiB maximum above conservatively includes the
original spike rather than silently replacing it with quieter repeats. Both
monitored confirmation cohorts peak at 12,725 MiB. The supplementary original
BA pair remains 192.852 → 195.479 tok/s (+1.3624%), including its slow sample.

## Change and numerical boundary

The fixed 3-row global kernel previously normalized 24 row/head combinations
serially with block reductions. Each warp now owns its query head. Local
normalization assigns its six row/head items to six warps. Both reproduce the
original two-level 256-thread reduction: four active 32-token trees plus four
empty chunks, then the original full 32-lane shuffle-down 16/8/4/2/1 tree over
chunk results. No per-lane four-term regrouping is substituted. expf/logf,
reciprocal, K/FMA order, FP8 conversion/scales, QK/PV, cache capacities,
speculative/ring causality and split/LSE merge remain unchanged.

The observer snapshots **all mutable** global/ring/speculative KV, queries,
controls, scales and expected partial/LSE values at real positions 16,384, 16,951,
17,512 (local layer 0 / global layer 29). Its 304 MiB arena/copy work is diagnostic
only; the library is absent from every primary benchmark. Actual production
function handles reproduce the snapshotted active outputs bit exactly; shader
resources/SASS and native dispatch are retained. Prototype real-input gains
are 14.73–15.16% global / 37.00–38.13% local. Those prototype percentages are not
production/end-to-end claims. The diagnostic library is compiled without RDC;
production uses -rdc=true. Actual RDC-built shaders use 96 registers global and
108 registers local, zero stack/local, shared 45,152 / 36,376 bytes. Single-kernel NCU clocks
and cache traffic differ between profiles; their wall durations cannot isolate
same-clock causality. Full parent/candidate traces retain 2,510 global and 12,550
local fixed 3-row calls and their actual elapsed sums, plus adjacent merge work.

## Failed candidates and shared-memory ordering

The first local prototype leaves invalid score slots untouched; full windows
pass but the standalone context 0 test fails because local PV reads the entire
128-token chunk. It was corrected to write masked probabilities 0 before any
production qualification. Every failed input/source/command is retained.

The unmodified original local shader also produces 3,084 partial/LSE mismatches
under synccheck while the new candidate matches the actual capture. Source
reads reduction[item][0] maxima and then reuses those shared slots without a
block barrier. One diagnostic-reference barrier restores 647 exact cases under
plain execution, memcheck and synccheck. Bounded racecheck (block 0 and all blocks,
one original launch each) reports 0 hazards. This is source/controlled-barrier
evidence for missing shared lifetime ordering, **not** a dynamic racecheck
error report or an assertion of an NVIDIA defect. New R3 normalization avoids
that shared reuse. Legacy local R2/R5 arithmetic receives the extra maximum-read
barrier; R2/R5 performance and a full R2/R5 numerical campaign are not claimed.

## Qualification and reproduction

The standalone test loads immutable captured bytes, independently reproduces
six actual production partial/LSE snapshots, then compares 641 counterfactual
positions (all tail lengths, short contexts, ring-window transition and final
capacity edge). These positions validate arithmetic/masking, not model quality
for invented prefixes. All 647 cases pass exactly under plain execution,
memcheck and synccheck. Two initial full-driver memory attempts are incomplete
(timeout 420 s / stopped 317 s before operator export); neither is a pass.

The final exact measured binary passes 7 CUDA/host tests, attention memcheck,
full 16K D2 memcheck, native 8/64 Ordinary/D2, protected 12B / internal NVFP4 64-token
vectors identical to the accepted parent and explicit-fixture 12B product tests.
Full 1,135-token traces, exact draft vectors, accepted groups, forced 0/1/2 commit
and rollback continuations pass. A blank-line whitespace cleanup changed the
binary identity after initial qualification; complete qualification was repeated
for the final binary. Both attempts/identities remain available.

Commands and ordered reproduction scripts are preserved in the raw root:
`parent_counters.py`, `final_screen.py`, `synchronized_reference_boundaries.py`,
`final_qualify.py`, `actual_profile.py`, `compare.py`, `post_compare.py`.
The offline generator never modifies production; the promotion transformation
and whitespace cleanup are recorded separately. After promotion, reproduce the
derivative with `build_t3_attention_candidate.py OUTPUT --source PARENT_SNAPSHOT`;
the explicit parent override reproduces the generated CUDA byte for byte.
All primary runs execute the frozen original/final CLI directly without LD_PRELOAD or diagnostic flags.
Use immutable model locks/metadata from the preceding accepted M1 record;
no new full 12.2 GB payload hash is claimed. Ordinary alone and model formats are
not changed to obtain this D2 gain. No fresh llama.cpp, Windows, image or release
qualification is claimed in this step. Earlier evidence remains immutable.
''')
perf=ROOT/'docs/PERFORMANCE.md';text=perf.read_text();anchor='## 2026-10-24 bounded T3 GateUp pipeline rejection'
existing='## 2026-10-24 fixed-D2 warp normalization'
if existing in text:text=text[:text.index(existing)]+text[text.index(anchor):]
section=f'''## 2026-10-24 fixed-D2 warp normalization

Frozen `c48124d` runtime versus qualified R3 local/global FP8 normalization,
identical 16K prompt / 1,135 outputs /context 17,519 / FP8 KV/greedy settings,3+10 per cohort:

{table}
All 80 vectors are exact; D2 groups/acceptance, all arenas and native precision
remain unchanged. Sampled peaks {vram_o:,.0f}/{vram_d2:,.0f}MiB. Operator/tail,
memcheck/synccheck, final exact-binary full-model/protected-profile, actual
shader/dispatch, draft and forced-continuation checks pass. Legacy local R2/R5
keeps its arithmetic with a shared maximum-read barrier; details and the original
under-instrumentation countercase are in the
[qualification report](evidence/gemma4_26b/d2-attention-warp-normalization-2026-10-24.md)
and [acceptance record](../artifacts/trellis35/d2-attention-warp-normalization-qualified-2026-10-24.json).
No fresh llama comparison or Windows/media/release claim; prior rows stay below.

'''
assert anchor in text;perf.write_text(text.replace(anchor,section+anchor,1))
readme=ROOT/'README.md';text=readme.read_text();start=text.index('The **2026-10-24 Linux text optimization benchmark**');end=text.index('Earlier public-profile measurements',start)
low=min(rate('candidate-b'),rate(r['ba']['candidate']));high=max(rate('candidate-b'),rate(r['ba']['candidate']))
replacement=f'''The **2026-10-24 Linux text optimization benchmark** on the RTX 5080 Laptop GPU
records **{rate('candidate-ordinary'):.3f} tok/s ordinary decode** and **{low:.3f}–{high:.3f} tok/s fixed-D2**.
Local/global D2 attention normalization adds **{100*min(r['ab']['gain_fraction'],r['ba']['gain_fraction']):.2f}–{100*max(r['ab']['gain_fraction'],r['ba']['gain_fraction']):.2f}% D2 throughput** against its
frozen parent in AB/BA measurements. All retained runs reproduce the same 1,135
output IDs with unchanged D2 acceptance and arenas. Sampled device-total peaks
are 12,455 MiB ordinary and 12,725 MiB in the confirmed D2 pairs. An original
cohort briefly reached 13,129 MiB; all its data and the follow-up are retained
in the report. The workload uses 16,384 prompt IDs, batch 1, greedy,
FP8 KV, three warmups and ten retained runs per cohort. See the
[qualification and raw-evidence record](docs/evidence/gemma4_26b/d2-attention-warp-normalization-2026-10-24.md)
for revisions, settings and checks. The preceding
[M1 GateUp result](docs/evidence/gemma4_26b/m1-gate-window-2026-10-24.md) retains
its 1.87–1.97% ordinary gain. These bounded text measurements do not extend
Windows, image or release qualification; a fresh llama comparison is separate.

'''
readme.write_text(text[:start]+replacement+text[end:])
print(path)
