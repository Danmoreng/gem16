> Historical task contract, closed or superseded. Current disposition is in
> [the archive index](../../evidence/benchmark-archive-2026-11-17.md).
> This file does not authorize a new experiment.

# Five bounded 12B Decode experiments

Owner authorization: 2026-11-15. Implement the five proposed steps carefully;
retain only measured gains with identical outputs, document every result and
commit successful changes. Push and release are not authorized. This supersedes
standing decision 7's tuning freeze only for the five bounded 12B experiments
below; permanent numerical, residency, allocation and profile rules remain.

Initial parent: `58febf4d66bb028a35f6f7f9528d2206cfab2475`. Existing unrelated
dirty files are preserved. Initial source/executable identities and copies:
`benchmarks/results/2026-11-15/58febf4/12b-five-step-goal/initial-state.json`.

1. **D2 NVFP4 Gate/Up:** test one M16 tile instead of two for exactly the
   12B three-row, N15360/K3840 fused projection. Preserve every output/scale
   and K-order/BF16 boundary. Primary: D2.
2. **D2 FP8:** adapt the existing packed-three-row mechanism for 12B QKV/O,
   preserving its actual arithmetic and source representation. Primary: D2.
3. **Ordinary FP8:** diagnose current real shapes, then test one bounded
   memory/instruction candidate; old rejected warp/parallel probes stay closed.
   Primary: Ordinary.
4. **Normalization boundaries:** test one 12B residual/norm/quantization fusion
   at a measured remaining boundary. Preserve all BF16 rounds and reductions.
   Primary: Ordinary, with D2 non-regression; a D2-only specialization instead
   declares D2 primary before measuring it.
5. **Assistant:** test one bounded BF16 GEMV or head candidate after current
   diagnosis, retaining the Assistant model and candidate partition semantics.
   Primary: D2.

Each starts from the last qualified parent. First use an offline derivative
and actual-input replay where practical; synthetic boundary tests supplement
rather than replace actual/output checks. An isolated screen needs a positive
95% bound and at least 1% weighted operator improvement. Account separately
for warm and cache-evicted conditions; do not infer E2E gains from hot replay.
One narrowed corrective revision is allowed for a demonstrated harness or
dispatch defect, not an open-ended parameter search.

Only a passing screen enters production qualification. Keep parent and candidate
executables side by side. Run one 16K/1135-output native preflight, then primary
ABBA, each cohort 3 warmups + 10 retained. Primary needs at least 0.5% pooled
throughput gain and positive bootstrap 95% lower bounds in both AB and BA.
Secondary throughput lower bounds must exceed -0.20% in both orders. Prefill
throughput and inverse TTFT lower bounds must exceed -1% in both orders.
Keep every sample and exact token IDs; native Ordinary/D2 identity, all 1135
outputs and the parent's proposals/acceptance must match. No fallback,
recurring allocation, precision change or added resident weight representation.

Acceptance also needs proportional numerical/boundary/memory checks, actual
dispatch/resource evidence, same planned arenas and sampled peak VRAM (allow
only explained desktop telemetry variation), and applicable 12B text/media,
26B internal NVFP4 and Trellis35 regressions. Linux evidence does not imply
Windows execution. Reject and restore candidates that fail a gate; record
unrun acceptance-only checks as unrun. Do not commit rejected runtime code.

Start wrapping at 22:45 Berlin, save/stop by 22:55, no work after 23:00.
Do not start an experiment without enough safe time. A deadline stops further
work with a continuation note; it does not turn unfinished work into a result.

## Progress

- Step1: complete, rejected at the predeclared0.5% primary gate. Full D2 ABBA
  gives+0.188335% pooled (AB+0.152904%, BA+0.223786%; both95% lower bounds
  positive), identical1135 IDs and all507/1014/626 group/proposal/acceptance
  counts. Offline real-input replay was+15.30% warm /+22.00% evicted,
  56 bit-equal numerical/optional-output/guard cases; scoped memcheck/synccheck
  each0 errors. Tiny positive model gain is recorded, not called zero.
  Runtime source and canonical runner restored to the freshly rebuilt parent.
  Secondary ABBA/protected acceptance tests skipped after primary gate failure;
  separate Ordinary preflights remain equal. Exact commands/raw samples are in
  the task raw root; no primary run used LD_PRELOAD or a profiler.
- Steps2–5: pending; separate offline derivatives are prepared for steps2/4/5.

Before step3 candidate timing: actual-input NCU captures at local/global QKV/O
show40 registers/thread,0.17–0.71 launch waves, low issue rate and dominant
long-scoreboard stalls. LocalQKV DRAM throughput is62.35%, with half-sector
L1 requests but ~55% L1 hits. Clocks were not fixed; this is diagnosis only.
The single step3 candidate is two-way K-loop unrolling, preserving sequential
MMA/K order, grid/block, FP32 scaling and source-row weights. No warp sweep,
precision/reduction change or repacking is authorized by this selection.

Step2 closes at the cache-evicted gate: all24 real cases are bit-equal;
packedT3 QKV/O is about+30.37% warm but-5.46% evicted, with both families
negative evicted. No production edit; acceptance-only checks are unrun.

Step3 diagnostic selection is per operator family: the same unroll2 candidate
is tested at every QKV/O geometry. O fails and remains untouched; only QKV
passes both screens. This is a dispatch restriction of the single tested
candidate, not a second unroll/warp/layout search. Independent E2E qualification
has not yet occurred and retains the original0.5% both-orders gate. Python
replay timings are supplementary; native C++ post-inference replay confirms
QKV+2.3107% warm /+2.2275% evicted, with positive95% bounds. All144 actual/edge
cases are bit-equal/finite/guarded and scoped memcheck/synccheck report0 errors.
The proposed production kernel is separately scoped to the exact12B T1 QKV
shapes; all generic and26B kernels remain unchanged.

Owner's Prefill-batching question during execution exposed a comparison
limitation:12B gem16 uses8,192-token chunks, llama.cpp batch2,048/microbatch512,
and vLLM chunked Prefill1,024. The existing Prefill numbers are configuration
comparisons, not matched-chunk or optimum-setting evidence. The separate
[chunk audit](../../evidence/gemma4_12b/prefill-chunks-2026-11-15.md) records this
without new GPU work. Decode candidate/parent settings remain identical.

## Paused by owner,2026-11-15 18:42 Berlin

The Decode Goal is explicitly paused in favor of a new matched-chunk12B
Prefill comparison. No runtime candidate is retained/committed. Steps1–3
are closed/rejected; step3 Ordinary ABBA is+0.1436075% pooled, below0.5%,
and its exact source/header/canonical-runner restoration is complete.

Step4 has only an offline screen:12 real cases bit-equal,96 actual/finite
boundary/guard checks pass, warm+34.9254%, fully evicted+0.4171% with95%
interval−0.0444% to+0.8897%. No production integration or E2E qualification.
Before resuming, review whether full eviction represents the immediately
preceding O-projection's freshly written input; do not silently redefine a
gate. Also align raw FP8 O outputs from `ordinary-fixtures` with the norm
fixtures to exercise the actual unrounded projection boundary. Step5's single
exact-tree GEMV derivative/observer/replay are prepared, not GPU-tested.

No optimization GPU jobs remain. Canonical gem16-run is the fresh unchanged
parent (`step1-parent-run`,SHA31036cc3ea7acdeb15307d78f7926002f87e34c2fb350cbfb85a8e248824d73b).
The CUDA-test executable/library may still be the rejected step3 build; rebuild
restored sources before using those artifacts. Raw root and all scripts are
preserved. No source/include/CMake runtime diff remains after restoration.
Resume this Goal only on an owner instruction, after the Prefill task.
