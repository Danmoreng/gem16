# Fixed-D2 local/global warp normalization — 2026-10-24

Accepted Linux text optimization against frozen main `c48124d974288a3434a5d7ba3fb9e88b64ddeb37`.
The parent runtime is the qualified M1 GateUp binary from e0c7fe5; c48124d only
adds rejected diagnostic evidence. Exact binaries, source hashes, toolchain,
commands, raw samples, telemetry and every raw-file identity are retained in
[the acceptance record](../../../artifacts/trellis35/d2-attention-warp-normalization-qualified-2026-10-24.json).
Raw root: `benchmarks/results/2026-10-24/c48124d/d2-attention`.
Monitored confirmation: `benchmarks/results/2026-10-24/c48124d/d2-attention-followup/`.
The qualified record preserves and supersedes the initial acceptance evaluation;
all original data, including the initial BA pair, remain accessible.

## End-to-end result

| Mode / comparison | Parent decode tok/s | Candidate decode tok/s | Change |
|---|---:|---:|---:|
| Fixed-D2 AB | 193.093 | 195.666 | +1.3324% |
| Fixed-D2 BA | 195.090 | 197.774 | +1.3760% |
| Ordinary adjacent | 139.211 | 139.219 | +0.0056% |

Same RTX 5080 Laptop / driver 615.71.09 / CUDA 13.4.92, firmware-managed max-power and
Dynamic Boost, fixed 16,384 prompt IDs, 1,135 output IDs / 1,134 decode intervals,
context 17,519, batch 1, greedy/mask, FP8 KV. Each cohort uses 3 warmups + 10 retained
process runs. D2 uses the original AB plus a process-monitored repeat BA; Ordinary is one
adjacent non-regression pair. The original BA remains supplementary evidence;
no sample is removed. The repeat adds 1 Hz nvidia-smi pmon to both cohorts.
Bootstrap 95% gain intervals (10,000 resamples; per-pair seeds in the record): AB [1.2100%, 1.4262%]; BA [1.2769%, 1.5434%].
Maximum sampled device-total VRAM: 12,455 MiB ordinary / 13,129 MiB D2.
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
