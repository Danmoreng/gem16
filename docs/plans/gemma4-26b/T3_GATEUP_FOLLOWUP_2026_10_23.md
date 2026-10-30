# T3 GateUp follow-up — complete, 2026-10-23

Owner: continue after accepted T3 Down `6895ed2`; retain/commit only correct
reproducible end-to-end benefit, no push. All prior accepted production paths,
model/cache/sampling/precision and graph/ring safety remain. Berlin wrap 22:45,
save/stop owned jobs by 22:55, no work after 23:00.

At 21:19: main clean; accepted executable matches previous qualified candidate.
Frozen parent CLI and native replay driver, start/source identity and exact inputs
are in `benchmarks/results/2026-10-23/6895ed2/t3-gateup/`.

First compare the previous eight-warp countercase (phase2/layer29/GateUp,
union20, 13 K3 + 11 K4 assignments) with the good phase0/layer5/union12 case.
Updated test-only observer recognizes the already accepted eight-warp Down;
GateUp still uses the old sixteen-warp production kernel. Exact pre-build source
snapshots are retained. Three real-input warmups precede optional NCU capture;
NCU uses cache-control none and clock-control none, distinct from old cold-cache
profiles and primary throughput. No whole-model speed inferred from counters.

Next: use measured scheduler/memory/instruction evidence to define one narrow
GateUp candidate. Do not reopen rejected generic prefetch, fusion, staging or
route-packing without matching new evidence and scoped active-decision record.
Screen all real inputs, K3/K4/mixed and unions, then strict numerical/memory and
protected regressions before full 3+10 AB/BA D2 plus ordinary non-regression.
Restore a failed production candidate, preserve evidence and stop safely.

## Checkpoint 21:41 CEST

Fresh real-input naive eight-warp screen repeats the late layer29 countercase
(-1.05%) and late layer0 (-0.90%). Warm-cache/no-clock-lock NCU comparison:
layer29 parent/candidate occupancy 59.94/52.36%, eligible warps 2.398/1.835,
issue 71.43/64.99%, short-scoreboard per issue 0.471/1.170, DRAM active
3.03/2.37%. Good layer5 case also loses eligible warps and increases short
scoreboard but saves operator time. Counters motivate examining shuffle/codec
latency, not reopening generic DRAM prefetch. Some cache hit ratios exceed 100%
under multipass/cache-preserving replay; those inconsistent percentages are not
interpreted as physical hit probabilities. Profiler times are not primary rates.

First general paired-window prototype is bit exact over 16,777,216 lane cases
but slower on real inputs; archived and rejected. It has a 16-byte stack frame.
A refinement of the same candidate uses the exact 32 MMA positions and fixed
state windows, preserving each 16-bit state, native half2/FP8 rounding and K order.
All 2,097,152 scoped lane cases pass; all twelve real GateUp cases improve
11.29–48.26% (median 24.93% operator throughput), including former countercases.
Actual pre-build source snapshots are preserved for each prototype.

Tentative production selector: native/vector GateUp (2816 -> 1408) only; Down/M1
unchanged. Two private headers hold the scoped codec and GateUp kernel. Regular
CUDA codec CTest now includes the independent old/new two-million-case matrix;
existing K3/K4/mixed, union8/16/24 projection/final-reduction zero-bit tests pass.
Seven CUDA/host tests (37.06s), targeted Trellis memcheck, native 8/64 parity,
protected 12B/NVFP4 and explicit 12B product checks pass. Production SASS proves
native E4M3/QMMA, 47 registers, stack/local0, shared1536. Static SHFL.IDX count
across both rate branches is 16 old versus12 new, with half as many warps for
same N128 output tile. No primary speed claim yet.

Short Nsight traces incur unexpectedly long outer collection/teardown (ordinary
~107s) while observed engine load/prefill/decode and exact outputs remain normal;
unprofiled ordinary8 process is4.17s. Retain this diagnostic overhead explicitly,
exclude it from primary performance and do not blame model/kernel or driver
without evidence. D2 node trace/full memcheck and primary AB/BA remain pending.

## Qualification checkpoint 21:46 CEST

Regular CUDA codec test now compares the production fixed-window helper against
the old independent decoder over all32 supported MMA positions,1024 word seeds,
32 lanes and both rates (2,097,152 lane cases), including zero/all-one words.
Tolerances remain bit-exact. Native/vector GateUp dispatch is tentatively integrated;
accepted Down/M1, precision and cache/ring/graph code are unchanged.

Seven broad CUDA/host tests pass (37.06s), targeted memcheck0errors; native8/64
and protected12B/NVFP4 ordinary/D2 parity and explicit12B product pass. New actual
GateUp shader has47registers, stack0/local0, shared1536; nativeE4M3/QMMA proven.
Static SHFL.IDX is16old/12new across two rate branches; new CTA has8versus16warps.
A query originally restricted to N128-named functions omitted the new GateWindow
symbol; corrected query confirms60GateUp calls,256threads,grid11x24, with accepted
Down/M1 dispatch intact. This was an analysis-filter assertion, not a kernel fault.
Full 16K/8-output D2 memcheck is running. No primary gain or commit claimed yet.

## Primary checkpoint 21:58 CEST

Full D2 memcheck0errors in249.30s. Actual production countercase input/output bytes
match baseline. New47-register shader has eligible warps2.572 (naive8:1.835),
issue75.20% (64.99%), short-scoreboard per issue0.181 (1.170); NCU79.936us
versus original98.272us. Diagnostic only; source/function handle is the actual
production shader, not separately compiled lab candidate.

Frozen parent/current 3+10 primary: parentA179.102926tok/s -> candidateB
193.284326tok/s (+7.9180%). CandidateB2 repeats193.353700tok/s; parentA2
is running, ordinary pair and final full accounting/forced-D2 checks pending.
All completed retained runs have exact1,135IDs,502groups,632accepted/1004proposed,
no fallback or token-loop allocation. First-pair sampled VRAM12725MiB each,
high-util clocks2617/2595MHz, power171.21/171.19W (whole-run telemetry).
No accepted production commit yet; candidate retained only after full AB/BA and
ordinary non-regression pass. Raw path and safe wrap deadline above still apply.

## Final state 22:10 CEST

The bounded GateUp candidate is accepted. D2 3+10 AB/BA medians179.102926 ->
193.284326 and178.949737 ->193.353700tok/s (+7.9180% / +8.0492%). Ordinary
137.593406 ->137.411329tok/s (-0.1323%, overlapping intervals, adjacent pair).
All60 retained runs match1,135IDs; all40D2 runs retain502groups,632/1004accepted,
identical precision, sampling, KV layout and arena byte counts, no forcing,
fallback/repack/loop allocations. Sampled peak stays12725MiB D2 /12455MiB ordinary.
D2 prefill differences -0.2272% / +0.0137% are separately disclosed.

Final full candidate trace:5921.865726ms,502groups,no tail/unclassified kernels,
GPU interval union98.7813%. Routed branch remains terminal in15060MoE blocks.
GateUp observed path3.242633 ->2.253963ms/group (-30.49%); acceptedDown
1.090925 ->1.089208. Exact concrete proposals, histogram[138,96,268], forced
0/1/2 commit/rollback/continuation all pass. All numerical, sanitizer, protected
profile, dispatch and instruction gates above pass. Qualified binary is built.

Evidence: `docs/evidence/gemma4_26b/t3-gate-window-2026-10-23.md` and
`artifacts/trellis35/t3-gate-window-2026-10-23.json`. Keep only the accepted
fixed-window implementation; generic-window prototype remains rejected/archived.
Narrow commit authorized, no push. Own GPU jobs are finished before stopping.
No required step remains for this slice. Existing Down/M1 and graph/ring fixes
remain. No new Windows/media/everyday-context release qualification or llama
baseline. Native remains below pinned196.55tok/s reference; formats differ.

Possible next bounded slice: investigate whether the same scoped paired-window
codec helps the accepted eight-warp T3 Down. First use its actual counters/inputs,
operator/reference and whole-D2 proof; do not transfer the GateUp gain by assumption.
Down still costs~1.09ms/group, so this is a smaller budget. The current accepted
binary is the next parent. No generic prefetch/packing/fusion/staging reopening
without matching evidence; closed220/250targets remain closed.

Owner follow-up before commit: README performance section and docs/PERFORMANCE.md
now include the qualified October23 result, protocol, ordinary variation and
evidence links, while retaining the historical measurements and release scope.
