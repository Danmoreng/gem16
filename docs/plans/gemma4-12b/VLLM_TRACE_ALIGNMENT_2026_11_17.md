# Trace-guided 12B alignment — 2026-11-17

The owner requests actual vLLM diagnostic traces of the 117 tok/s MTP path,
comparison with native kernel/graph execution, then implementation of measured
opportunities. The owner explicitly skips the additional native D4 baseline.
Use native fixed D2, preserve its exact semantics and the two retained patches.
This new owner request supersedes the tuning freeze and previous campaign's
no-further-benchmark restriction only for this trace-guided 12B slice. It does
not resume the paused five-experiment campaign or reopen old negative searches
without new matching evidence. No commit/push/release is authorized.

1. Pin actual dirty source/binary identities without altering unrelated work.
2. Capture warmed steady-state vLLM D3 (and matched D2 if needed for attribution)
   and native D2. Separate diagnostic timings from unprofiled throughput.
3. Establish actual kernel sequence, graph replay boundaries, shapes/resources,
   family costs, gaps and verification/acceptance work. Never attribute a
   whole-request or prefilling capture to Decode, or D3 acceptance gains to a
   faster per-row kernel. Do not infer invocation facts just from class names.
4. Select one bounded change from the largest evidenced native gap. Preserve
   precision, BF16 boundaries, quantization/scales, weights/residency, KV
   commit/rollback, allocation/repack rules and protected model paths.
5. Validate actual-input numerical and memory behavior, actual dispatch and
   required protected/shared-runtime checks. Screen throughput with the owner's
   **one warmup + three retained requests**; widen only if a material concern
   needs resolution. Keep only correct measured gains, restore rejected code.

Evidence root: `benchmarks/results/2026-11-17/vllm-alignment/`.
Result summary: `docs/evidence/gemma4_12b/vllm-alignment-2026-11-17.md`.
Daily cutoff remains 22:45 wrap-up, saved/stopped by 22:55, stop at 23:00 Berlin.

## First measured hypothesis

Current actual D2 traces put native Down at about 4.1 ms/group versus 2.3 ms
for vLLM, while native FP8 projections are already faster. The first offline
candidate caches only the three Down activation/SFA rows once per CTA in
bank-padded shared memory (26,016 bytes), retaining native weights, K64/MMA
order, division/BF16 epilogue, grid 60 and block 256. This differs from the
rejected two-stage **weight/SFB** pipeline. It is an operand-load hypothesis,
not a proven diagnosis of the entire gap. Replay uses the immutable actual
12-case Down capture after confirming parent replay against its recorded
BF16 outputs. No runtime integration before numerical/cache-condition screen.

The activation-cache trial is closed without runtime integration: warm+0.77%,
evicted−0.49%. Its unchanged-source 12 actual and finite cases are exact;
acceptance-only full-model/protected gates were not run.

The next trace-evidenced candidate is the 12B FP8-KV D2 T3 post-attention
boundary: apply the already qualified per-row T1 fusion independently in
three CTAs. Fresh parent captures cover all three rows, layers0/15/31/47,
groups0/250/506. All120 actual/null-output/alias/finite-boundary checks pass,
as do memcheck/synccheck/racecheck. Paired kernel replay gains49.38% warm
and1.28% evicted. Source integration changes only the exact T3 branch;
Ordinary, D1/D4, BF16-KV, Prefill and 26B retain their paths. Full-model
and protected qualification follow before retention.

First unprofiled parent/candidate D2 cohorts (1+3 each) are94.1376/94.6965
tok/s, +0.5937%, with identical1135 IDs and507/1014/626 group/proposal/acceptance
counts. Because this is a small effect, run the reverse order at1+3 before
retention. No timing uses a profiler/interposer.

The reverse order confirms94.7123 candidate versus94.0743 parent, +0.6782%.
Full-state and protected-profile qualification is in progress.

One further offline Down hypothesis follows the measured Down gap: two
independent N8 accumulator tiles per warp, four warps instead of eight in
the same60 N64 CTAs. Reuse each activation operand for two independent MMA
chains without changing any output's sequential K64 arithmetic, layout,
scales or epilogue. This tests instruction reuse/parallelism, rather than
the previously rejected activation cache or weight pipeline. One geometry,
no parameter sweep; require exact actual-input and finite replay followed
by a positive paired warm/evicted screen before production integration.

The dual-tile hypothesis fails decisively: warm−31.18%, evicted−26.24%,
despite84 exact comparisons and zero memcheck/synccheck/racecheck errors.
It stays offline. A final distinct Down screen retains the parent's exact
60CTA/256thread/N8-per-warp geometry, specializing only fixed T3/N3840/K15360
valid/zero operands and32bit index arithmetic. This isolates fixed-shape
instruction simplification from the failed change to warp/accumulator
parallelism. No further shape/grid variants are part of this trial.

The fixed-shape screen passes: warm+31.0643%, evicted+20.7225%,84 exact
comparisons and zero errors/hazards in all three sanitizer tools. The
generated production kernel matches the tested kernel body byte for byte.
Integration selects only T3/N3840/K15360 in the physical-BF16-output Down
launcher, retaining the parent's60CTA/256thread/N8-per-warp geometry and
all other dispatches. Qualify this component against the already qualified
T3-fusion parent; reuse that fresh parent timing, then1+3 combined requests.

## Completion

Retain the T3 boundary fusion and fixed-shape Down only. Final unprofiled
1+3 cohort99.006443tok/s versus starting pooled94.105942, +5.207430%.
Exact1135 outputs/1014 proposals/whole311287808-byte KV, both-stage
protected profile/session/media checks, final36 protected executions
(18 paired cases, including BF16-KV), CUDA/host tests and all scoped
sanitizers pass. Final actual dispatch has96 fewer kernels/group and
48 fixed Down calls/group. GPU jobs are stopped. No native D4 benchmark,
commit, push, checkpoint conversion or release. Canonical CLI/chat/core,
tests and server built. Evidence and limitations are in the linked report;
remaining attention/norm and D3/D4 graph work are separate next slices,
not unfinished background jobs or a117tok/s native claim.
