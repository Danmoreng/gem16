# Five measured decode steps — 2026-10-24

Owner explicitly requests an active Goal to implement/investigate all five steps.
Starting main is clean at `26c6840`; built executable SHA matches the accepted
October23 candidate. Same RTX5080Laptop, driver615.71.09; GPU initially idle36C.
No new push/release authorization. Berlin wrap22:45/save-stop22:55/hard-stop23:00.

1. **M1 GateUp — complete, accepted.** Test paired fixed codec windows while preserving
   the existing four-iteration payload pipeline. Use actual-input old countercases,
   strict operator/reference comparison and standalone/actual-production resources.
   Promote only after full ordinary AB/BA, D2 non-regression and protected gates.
2. **T3 GateUp — complete, rejected diagnostic variants.** Refresh actual counters and residual cost; define one
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


## Step2 start 14:35

Step1 committed as e0c7fe5; no push. Frozen current parent/driver match accepted
M1 artifact. Raw root: benchmarks/results/2026-10-24/e0c7fe5/t3-gate-pipeline/.
First collect actual current T3 GateUp counters for old late layer29 and early
layer5 cases, using the frozen observer only to call captured production handles.
Previous47-register codec has substantial long-scoreboard ratio; do not assume
a new pipeline helps until current evidence and real-input/resource screens pass.


## Step2 checkpoint 14:48

Fresh parent GateUp long-scoreboard0.8167/0.9589, short0.1809/0.1993,47regs.
Depth2 real-screen range-4.74%..+23.45%, median+8.11%; depth4-3.28%..+23.07%,
median+9.28%. All24 projections per variant and2,097,152 lane cases exact, but
both fail the predeclared all-case>2% gate, so neither is promoted or advertised
as an end-to-end gain. The union24 phase1/layer15 countercase shows47->63regs,
register block limit5->4, long-scoreboard1.1063->0.5881 and slightly slower NCU
wall. This justifies one last depth2 variant bounded to5blocks/256threads; if it
fails, close this mechanism and retain all negatives. Production is still e0c7fe5.

Later Step5 native-scale source: NVIDIA PTX ISA9.4 block-scaling selectors,
https://docs.nvidia.com/cuda/archive/13.4.2/parallel-thread-execution/index.html
(section Block Scaling for mma.sync). For4X UE4M3 all four SF bytes are used;
byte selectors remain0. Existing thread-id-a0 selects the lower two lanes in
each quad; thread-id-b0 selects lane0 of the quad. Preserve those selector
semantics and verify distinct per-row scales with independent native operators
before trying A-row packing. No scale-format or precision change is authorized.


## Step2 closed 14:51

Depth2/depth4 and the one register-bounded depth2 variant are exact, but all fail
the predeclared sampled-case gate. Bounded shader48regs still loses all cases.
No production source/binary changed; no full-model speedup/loss claim or candidate
memory qualification. Positive operator medians are not global performance.
Evidence: docs/evidence/gemma4_26b/t3-gate-pipeline-rejected-2026-10-24.md;
artifacts/trellis35/t3-gate-pipeline-rejected-2026-10-24.json. All raw snapshots,
inputs/outputs, modules, samples and NCU metrics preserved. Commit diagnostic/
evidence changes only; no push. Next Step3 actual local/global T3 attention
capture and counter analysis. Earlier KV staging stays rejected unless a fresh
matching finding justifies a specifically scoped new candidate.
