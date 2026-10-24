# Five measured decode steps — 2026-10-24

Owner explicitly requests an active Goal to implement/investigate all five steps.
Starting main is clean at `26c6840`; built executable SHA matches the accepted
October23 candidate. Same RTX5080Laptop, driver615.71.09; GPU initially idle36C.
No new push/release authorization. Berlin wrap22:45/save-stop22:55/hard-stop23:00.

1. **M1 GateUp — complete, accepted.** Test paired fixed codec windows while preserving
   the existing four-iteration payload pipeline. Use actual-input old countercases,
   strict operator/reference comparison and standalone/actual-production resources.
   Promote only after full ordinary AB/BA, D2 non-regression and protected gates.
2. **T3 GateUp — pending.** Refresh actual counters and residual cost; define one
   justified candidate rather than assuming M1 transfers. Verify unchanged D2
   proposals, acceptance, rollback/continuation and memory boundaries.
3. **D2 local/global attention — pending.** Verify actual dispatch, inputs, numerical
   reduction order and cache semantics. Earlier KV staging rejection remains;
   only a fresh matching finding can reopen a narrowly scoped mechanism.
4. **QKV/O projections — pending.** Inspect and measure one/three-row native
   specializations, preserving weights/accumulation/precision and protected12B.
5. **Target/Assistant output heads — pending.** Profile actual small-batch matrix
   work (not already tiny token selection), test one justified candidate, retain
   exact vocabulary/softcap/mask/tie semantics and output IDs.

After each accepted change, freeze the next parent and reassess the critical path.
Negative/neutral candidates are archived and removed from production. Primary
claims use3warmups/10retained, exact16384prompt IDs/1135outputs, context17519,
batch1, greedy/mask/FP8KV, same timing boundaries and power policy; collect raw
samples, clocks/power/thermals/VRAM and actual native dispatch/instruction evidence.
Before claiming a lead, rebuild and directly compare current llama.cpp under the
documented native-format baseline, disclosing formats/outputs/quality differences.

Step1 raw root: `benchmarks/results/2026-10-24/26c6840/m1-gate-window/`.
Only diagnostic code is changed initially; production parent remains qualified.


## Step1 checkpoint 14:04

M1 GateUp paired-window screen: all24 real projections and2,097,152 codec lane
cases exact; twelve GateUp cases gain13.57–17.60%, median15.74%. Former naive
countercases are included. NCU actual-parent versus diagnostic prototype shows
lower duration despite lower eligible-warps/issue rates; interpret as diagnostic
work/latency comparison, not occupancy-only causality. Three real-input warmups,
cache-control none, clock-control none. Raw/source snapshots preserved.

Tentative native/vector2816->1408 selector built; M1Down/T3 unchanged. Seven
CUDA/host tests pass34.06s, targeted Trellis memcheck0errors, native8/64 and
protected12B/NVFP4 ordinary/D2 plus explicit-fixture12B product pass. Actual shader
56registers, stack/local0, shared512 and nativeE4M3/QMMA. Full16K ordinary memory
check is running. Next: actual-production counter/input verification, full3+10
ordinary AB/BA plus D2 parent/candidate non-regression, fresh full ordinary traces,
exact D2 proposals and forced0/1/2 checks. No accepted speed claim/commit yet.


## Step1 checkpoint 14:15

Full16K ordinary memcheck0errors240.71s. Actual-production counter capture uses
rebuilt native driver/function handle; activation/scales/ids/expected bytes match
the old real-input screen exactly. First3+10 pair136.480899->139.030651tok/s,
approximately+1.87%, all1135IDs exact. Reverse pair and adjacent D2 still pending;
no production acceptance yet. Protected12B/NVFP4 vectors also exactly match the
previous qualified parent. Whole-run initial parent clock median2565MHz differs
from yesterday; use fresh same-day AB/BA only, no cross-day gain claim.


## Later-step source map (hypotheses only; no implementation or gain claim)

Actual D2 trace uses FP8 `Sm120DirectProjectionFixedBatchKernel<3>` for grouped
QKV and `Sm120DirectProjectionSplitK2FixedT3Kernel` for O. Both currently keep
three accumulator sets and issue one M16 MMA per token, reusing B. A bounded
Step4 candidate could pack the three real A rows into one M16 instruction,
retaining per-row K order, the exact split-K combination and scale products.
Use actual layer-dependent local/global shapes and fresh counters, not assumed
bandwidth or dimensions. Same-source real-input bit parity is mandatory.

Actual heads are NVFP4, not the BF16 reference output_head.cu or FP8 matvec:
`src/cuda/nvfp4/sm120.cu` contains `Sm120DirectProjectionFixedT3Kernel` (Target)
and `Sm120ProjectionArgmaxCandidatesBf16Kernel` (Assistant). `rg --no-ignore`
is needed to see that source under the repository's ignore patterns. The Target
T3 kernel likewise issues three MMAs; M16 A-row packing is a possible Step5
experiment, but native block-scale SF_A lane mapping requires official PTX
verification and diverse per-row-scale tests before any conclusion. Preserve
NVFP4/UE4M3, BF16 output boundary, output_divisor, masks, softcap and exact tie
rules; do not introduce FP8 or change head format. Heads may already be bandwidth
limited; verify counters before tuning or predicting gains. No source changes
for later steps while Step1's frozen source/binary qualification is running.


## Step1 accepted 14:32 CEST

Ordinary AB/BA 136.480899->139.030651 and
136.303319->138.991625tok/s (+1.8682% / +1.9723%).
D2 adjacent 192.599852->193.024563 (+0.2205%).
All numerical/memory/protected/full-vector/dispatch/proposal/forced gates pass.
Fresh full parent/candidate ordinary accounting is preserved. README and results
updated; commit narrowly, no push. Evidence/raw paths above; actual qualified
executable is built and frozen. Next: freeze this accepted parent for Step2,
refresh actual T3 GateUp counters and test one justified candidate. All earlier
closed/rejected mechanisms require matching new evidence before retry.
