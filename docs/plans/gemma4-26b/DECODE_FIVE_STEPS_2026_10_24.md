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
3. **D2 local/global attention — complete, accepted.** Verify actual dispatch, inputs, numerical
   reduction order and cache semantics. Earlier KV staging rejection remains;
   only a fresh matching finding can reopen a narrowly scoped mechanism.
4. **QKV/O projections — complete, accepted small D2 gain.** Inspect and measure one/three-row native
   specializations, preserving weights/accumulation/precision and protected12B.
5. **Target/Assistant output heads — complete, no production promotion.** Profile actual small-batch matrix
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

## Step3 checkpoint 15:14

Parent c48124d frozen; runtime binary remains the qualified e0c7fe5 code.
Raw root benchmarks/results/2026-10-24/c48124d/d2-attention/. New explicitly
attached offline observer snapshots all mutable global/ring/speculative KV,
queries, controls, scales and expected partial/LSE bytes; no borrowed mutable
cache. Three actual positions16384/16951/17512, local layer0/global29, all
parent replays bit exact. Extra304MiB is diagnostic only, never primary timing.
Fresh global counters barrier0.8261/short0.8936 perissue; local barrier0.6776/
short1.1447. Test warp-owned normalization retains the old full32 shuffle-down
chunk tree and second-level eight-warp tree including zero/negative-inf empty
chunks, preserving all FP32 association. No K/V staging retry or QK/PV/cache/
merge changes. Global sampled gains14.7–15.2%, local37.3–38.2%, all partials/
LSE bit exact. Prototype global barrier0.2118 and88->91regs. Initial derivative
compile failed because renaming removed the old wrapper's kernel reference;
failed command/source retained; v2 retains the unused original definition.
Now checking257 global and384 local counterfactual-position boundaries over
captured bytes with memcheck before production promotion. These synthetic
positions test arithmetic/masking, not model quality for those prefixes.
No full-model gain/production acceptance yet. Legacy R2/R5 normalization and
all one-row attention remain unchanged in the proposed production selector.

## Step3 checkpoint 15:33

The full-model three-window memcheck exceeded420s before standalone operator
validation; stopped a second kernel-filtered full-driver attempt after317s with
no operator export. Both incomplete attempts retained, no memory pass claimed.
A small offline runner now loads the immutable operator snapshots directly:
647 cases = six actual captures plus641 counterfactual positions. Initial local
prototype failed short context0 because invalid score slots remained negative
infinity while local PV reads the entire chunk. Corrected candidate explicitly
writes masked probability0; plain/memcheck647 cases exact, memcheck5.3s0errors.
Synccheck then perturbed the *unmodified original local* shader: candidate matches
capture, original has3084 partial/LSE mismatches. Source lacks a barrier between
maximum reads and reuse of reduction[item][0]. A diagnostic reference with that
single barrier passes all647 cases plain, memcheck5.4s and synccheck1.6s0errors.
Two bounded racecheck runs (block0 and allblocks/one original launch) show0hazards;
record this limitation rather than claiming a dynamic racecheck report. The
missing shared lifetime ordering plus the controlled barrier comparison supports
the fix; no NVIDIA driver/tool defect is asserted.
Final corrected real-input re-screen: local37.00–38.13%, global14.73–15.16%, all
six partial/LSE outputs exact. Proposed production: kRows3 warp normalization;
legacy R2/R5 local arithmetic retained with the maximum-read barrier. All other
QK/PV/KV/cache/merge semantics and FP32 trees unchanged. Full qualification now
running. No accepted full-model gain yet. Production whitespace-only cleanup
recorded separately; rebuild/freeze again before primary measurement.


## Step3 accepted

D2 AB 193.092926->195.665682tok/s (+1.3324%),
BA 192.851748->197.774083 (+1.3760%).
Ordinary adjacent 139.211053->139.218831 (+0.0056%).
Exact final-binary full-memory/protected/dispatch/shader/output/proposal/forced
gates pass. Both binary identities, all negatives and the synccheck original
countercase retained. README/PERFORMANCE updated; commit narrowly, no push.
Original candidate-b2 device spike/slow sample retained; monitored BA confirms
the result. All80 vectors exact; original and follow-up manifests verified.
Next freeze this accepted parent for Step4 real one-/three-row QKV/O counters
and native row-packing hypothesis, preserving all formats and precision.

## Step3 final audit 16:24

The initial BA cohort has one transient total-device peak13,129MiB and one
185.309tok/s sample. Reported weight/KV/workspace/repack/graph bytes are identical;
allocation source is unknown. Keep all samples and the initial acceptance record.
One additional BA pair with1Hz compute/graphics pmon confirms195.089586->197.774083
(+1.3760%,bootstrap95[1.2769%,1.5434%]); both peak12,725MiB, only expected gem16
compute PIDs plus4MiB Xorg. Original AB gain+1.3324%,95[1.2100%,1.4262%]. All80
retained vectors and60 D2 group/acceptance records exact. Ordinary139.211053->
139.218831 is neutral. Original868raw hashes and all runtime-source identities
reverified. Qualified record supersedes the initial acceptance basis, preserving
all original data. Traces: global584.252->509.485ms, local413.985->393.440ms
(summed diagnostic elapsed, not pure same-clock causal accounting).

Final evidence/docs renderer is benchmarks/render_d2_attention_evidence.py;
idempotence and syntax pass. Source review proves only local/global normalization
regions changed. README distinguishes confirmed12,725MiB D2 device peak from the
retained transient13,129MiB observation. Commit Step3 now, no push.

Next Step4: freeze the accepted binary/driver, inspect actual one-/three-row QKV/O
counters and shapes. Match production -rdc=true in new diagnostic builds; Step3's
non-RDC prototype percentages do not fully transfer to production. Try native
M16 A-row packing for three rows only when fresh counters justify it. Preserve
per-row K/MMA/split-K and scale order, source weights and protected12B. M1 is
an investigation, not a promise to change its already specialized kernels.

## Step4 start 16:36

Step3 committed85fc453, clean; frozen CLIcc956a04..., driver424a3126.... Raw:
benchmarks/results/2026-10-24/85fc453/fp8-projections/. Initial observer wrongly
assumes3bindings globally and fails before capture with only25QKV calls; kept
both failed versions. Source proves five global layers use Q8192/K1024 only,
then raw-K-to-V copy; local layers Q4096/K2048/V2048. Fixed observer captures
all24real M1/T3 cases, immutable weights borrowed, mutable data copied; RDC=true.
Fresh six counters complete. M1 long-score20.65/22.76/12.53 strongly outweighs
short0.206/0.266/0.075; T3 math-throttle7.04/7.89/4.98 supports reducing three
M16 MMAs to one packed3-row operation. Counter wall times have different GPC
clocks and are diagnostic, not same-clock speed evidence.
Narrow active rationale opens T3 A-row packing and one fixed two-entry M1
FP8 fragment pipeline, retaining exact K/split-K/scale/MMA order and all formats.
Do not promote positive medians alone: require per-case gate and full primary
benefit with numerical/memory/protected/draft/continuation evidence.

## Step4 accepted 17:43

M1 two-entry projection pipeline loses every sampled case; kept only offline.
All24 real T3 outputs exact; naive packing gains about14–25% operator throughput.
Packed QKV creates405,504 shared-load conflicts;32B ephemeral row padding removes
them and cuts608,256->202,752 wavefronts, adding14.18–18.21% QKV throughput versus
naive in direct3way measurements. Selected24scale/zero/tail/split-K guarded cases
pass plain/memcheck/synccheck. Production-matching RDC, new actual resources:
QKV36regs/static0/dynamic8544, O40regs/static96, stack/local0. All24 actual native
input/output snapshots match frozen parent. M1/12B/non-target/legacy unchanged;
global Q/K2bindings plus raw-K-to-V copy preserved.

Seven CUDA/host checks, CUDA-suite memcheck, exact native8/64, protected12B/NVFP464,
explicit12Bproduct, full16K D2 memcheck237.52s0errors pass. Full1135outputs,
proposals, forced0/1/2continuations and exact30120target projection calls pass.
Primary D2 AB195.731027->196.555428 (+0.4212%,95CI[0.3610%,0.5460%]);
BA195.354054->196.491573 (+0.5823%,95CI[0.4399%,0.6969%]); Ordinary139.132828->
139.085218 neutral(-0.0342%). All60 vectors exact, all40D2 counts502/632/1004.
Device peaks12455/12725MiB; all pmon only expected compute plus4MiB Xorg.
High-util clocks parent2565/2557 vs candidate2580/2580MHz: same max-power policy,
not fixed-clock or pure cycle claim. Whole-model trace QKV essentially unchanged;
O306.215->298.367ms, explaining much smaller full benefit than repeated operators.
Source weight traffic is unchanged; cache/bandwidth is an inference, not proof.

Record artifacts/trellis35/fp8-projections-2026-10-24.json; all1602raw files and
current sources to verify before narrow commit. README/PERFORMANCE/evidence
rendered via benchmarks/render_fp8_projection_evidence.py (syntax/idempotence pass).
No push. Next Step5 freeze accepted binary, capture actual Target/Assistant NVFP4
heads and counters. Source matrix SF_A mapping already explicitly documents
lane0 lowerM8/lane1 upperM8 under thread-id-a0; use that existing map for any
R3 candidate, preserving nativeE2M1/UE4M3, BF16/divisor/mask/tie boundaries.
No head candidate or gain yet. Fresh latest llama comparison remains after5.


## Step5 started after owner status request, 17:53 CEST

Step4 committed f6de38e; accepted CLI7e79ca58... frozen. Production is unchanged
while the head investigation begins. Raw root:
`benchmarks/results/2026-10-24/f6de38e/output-heads/`.
Initial observer borrowed the earlier replay driver and forced-D2 windows;
those windows never execute Assistant generation. Assistant replay therefore
fails against unexecuted diagnostic slots, not a runtime head regression. Both
failed captures and the initial older-driver M1 counter are retained. Do not use
these initial Assistant slots as valid model inputs or acceptance evidence.

The test-only replay driver now has `d2-head-windows`, calling the actual
`RunMtpAssistantGroup` at the same three controlled positions; existing ordinary,
forced-D2 and rollback modes retain their behavior. Exact current f6de38e core
is linked. Driver/source hashes in parent-head-windows.json. New observer has a
fixed96MiB diagnostic arena, snapshots all mutable A/SF_A and output bytes and
borrows only immutable resident weights. Never preload into primary benchmarks.
Current M1 three cases and actual-D2 nine cases replay byte-exactly. Actual Target
vocab262144/K2816, Assistant vocab262144/K1024, masks258883/258882, candidate
array4096*8bytes; Target preserves full BF16-rounded float logits, divisor4736.

Fresh current-parent Target M1 has54regs, DRAM93.63% of peak, long-score54.28;
Target T3 has106regs, DRAM91.55%, long-score17.72, math-throttle0.403. Full9-case
D2 snapshots pass before profiling. These are cache/clock-uncontrolled diagnostic
counters, not primary timing or a new gain. Assistant counters are next in the
serial current_parent_counters.py run. No head candidate is implemented yet.
Potential bounded Target R3 packing would remove two redundant M16 instructions
and accumulator sets, preserving the existing native scale-lane map/K64 order;
weights/traffic stay unchanged and high DRAM utilization limits expectations.
Only write the narrow active rationale after all counters are reviewed, then
screen diverse per-row scales and all actual outputs before any promotion.

Status for owner:4/5 investigated; accepted M1 +1.8682/+1.9723% ordinary,
attention +1.3324/+1.3760% D2 and QKV/O +0.4212/+0.5823% D2. T3GateUp negatives
retained. D2-specific sequential gains compose to roughly+1.8–2.0%; this is not
a directly measured complete-stack delta. Latest qualified139.085ordinary,
196.492–196.555D2. README/evidence updated; accepted changes committed, no push.
Fresh latest llama comparison remains after Step5, no current superiority claim.

Correction to the earlier Step3 accepted checkpoint's BA line:192.851748 is the
original BA parent, while197.774083 belongs to the separate monitored BA repeat.
The repeat parent is195.089586, as recorded in the final audit and qualified
artifact. The +1.3760% acceptance uses only that matched repeat pair; do not pair
those two different cohorts or reinterpret the earlier shorthand as a new gain.

Step5 fresh counters completed successfully17:54: all four current-parent runs,
three M1 and nine actual-D2 cases each, exact. Assistant56regs, sampled
long-score47.05/42.38, math-throttle0.069/0.070; all raw counters in
current-parent-counters-summary.json. No GPU jobs left running. Production core
remains accepted f6de38e; two diagnostic source changes plus this journal remain
uncommitted. Next review bandwidth/resource counters, write narrowly bounded
mechanism rationale, implement and screen heads; no new gain claimed yet.


## Step5 qualification checkpoint18:14

Fresh counter rationale recorded in ACTIVE_DECISIONS. Native Target R3 packing
and one adjacent-N8 Assistant candidate are exact in all9actual cases and all42
scale/zero/mask/tie/overflow/partial-warp guarded cases; standalone plain,
memcheck0.46s and synccheck0.41s pass. Initial nvcc shorthand emits generic
compute_120 PTX and ptxas rejects block-scale MMA; failed source/log preserved.
V2 explicitly matches production compute_120a/sm_120a and RDC=true.

Assistant pairing loses every real case(-0.01..-0.85%, median-0.154%),95regs
versus56/register block limit2versus4, DRAM95.19%; remains offline. Target packing
has+0.184..0.214% repeated-operator gain, aggregate bootstrap95lower>0,64regs
versus106/block limit8versus4, DRAM94.11%. Only Target enters a provisional
production selector for tokens3/vocab262144/K2816; ordinary/Assistant/legacy and
other shapes unchanged. Source reverse-audit reconstructs parent exactly.

Production binarya2c4529fd37ec20f197dc64c24e6e023551a37e8079665753a731fada2be1b9f
is frozen. Seven CUDA/host tests34.51s, recognized fixedT3 memcheck1.17s,
native8/64 exact, protected12B/internalNVFP464 exact against Step4, explicit12B
product and full16KD2 memcheck238.78s0errors pass. Actual current native9capture
input/scale/output bytes match parent; actual64regs/stack-local-shared0, SASS
OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X confirmed.

Primary job is live in exec session3347, running finish.py (actual_profile.py,
compare.py, post_compare.py sequentially). First AB195.967129->196.289882,
point0.1647%,95CI[0.0218%,0.3158%]. This is provisional,
not acceptance: reverse BA and ordinary cohorts are still running. Do not restart
or overlap GPU work. Re-poll3347; observation timeout is not process completion.

After all6cohorts/postchecks: run audit_comparison.py; inspect clocks/processes/
VRAM and head dispatch costs. decision.py implements the predeclared gate and,
if rejected, restores only owned runtime source/header and rebuilds/compares the
accepted parent binary byte-for-byte. Then make_record.py preserves accepted or
rejected artifact, followed by benchmarks/render_nvfp4_head_evidence.py (not run
yet). Report/README/PERFORMANCE rendering must be checked idempotent, raw/source
hashes audited, journal/ACTIVE closed and narrow commit. No push. No whitespace
cleanup/rebuild of the qualified candidate mid-measurement. Fresh latest llama
and direct cumulative-parent comparison remain afterward.


## Step5 closed18:34 — qualified trial rejected

Full D2 AB195.967129->196.289882(+0.1647%,95CI[+0.0218%,+0.3158%]);
BA195.881483->196.219919(+0.1728%,95CI[-0.0018%,+0.3470%]). Ordinary139.026898->
139.169282(+0.1024%). Both D2 point estimates are positive, but the predeclared
both-orders95lower>0 gate fails. No repeat or sample removal is used to force
acceptance. Whole-process high-util clocks2565vs2572MHz in both D2 orders and
2580vs2587MHz ordinary, sampled peaks12455/12725MiB, only expected compute plus
4MiBXorg. Positive ordinary despite unchanged dispatch further limits attribution.

All60vectors and40D2 count records exact. Full traces502Target heads plus1004
Assistant candidates, all draft arrays and forced0/1/2rollback/continuations pass.
Target summed diagnostic head elapsed259.079->255.606ms, Assistant185.368->187.205ms;
non-head legacyT3 grid88 retains15060calls114.534->114.434ms. These elapsed sums
have uncontrolled clocks/cache and are not same-clock causal accounting.

Owned production selector/include/header removed; runtime source is exactly
accepted f6de38e. Rebuild succeeds. Strict whole-file hash initially differs in
four nvcc PID characters inside non-loadable .strtab; read-only ELF proof shows
every other byte and all PT_LOAD CPU code/data/PTX/SASS/fatbins identical. No
binary is patched. Restored CLI02b65a2e... versus frozen7e79ca58...; both exact full
hashes, offsets, temporary names and all segment hashes in rollback/elf-equivalence.
Initial ELF verifier's uniqueness assumption spans several sm120 source filenames;
v2 isolates the single changed string. Initial record CTest-summary string expects
older wording; v2 verifies every one of the7required Passed lines and100%summary,
not weakening a runtime test. All failed helper versions are retained.

Artifact artifacts/trellis35/nvfp4-output-heads-2026-10-24.json status=rejected,
545raw identities and all final source hashes verified. README/PERFORMANCE/report
rendered via benchmarks/render_nvfp4_head_evidence.py, repeated rendering idempotent;
git diff --check passes. Head trial and Assistant pairing stay only in the offline
lab; no native head production change is committed. Commit this bounded negative
record/diagnostic tooling, no push. Fresh latest llama and direct full-stack parent
comparison remain; audit the existing Step2 non-RDC prototype limitation before
making the final campaign completion claim. No GPU jobs remain active.


## Final Step2 matched-build audit18:46

Main8046990 diagnostic-only, runtime remains accepted f6de38e. The existing
three variants are now re-screened with production compute120a/sm120a,RDC=true.
Depth2 min/median/max-0.764/-0.155/+0.659%; depth4-4.170/+8.592/+21.285%;
bounded2-6.809/-5.213/-4.358%. All24actual projections and2,097,152codec lane
cases per variant exact. All fail original every-real-GateUp-case>2% gate, so
no production candidate/memory/full-model gain/loss is claimed. No new variant
is tested. Source bodies are unchanged; the old first-screen module predates
appending the bounded kernel, so this is not a flags-only causal attribution.

Fresh union24/layer15 actual-function counter resources parent47/blocklimit5,
depth2 48/5, depth4 63/4; long-score1.0556/0.8416/0.6535, eligible warps
2.809/2.915/2.719. NCU wall values have uncontrolled caches/clocks and differ from
standalone screens; no same-clock counter-wall speed claim. An unrelated old
attention link derivative is omitted with a visible unsupported stub, never
called in T3 mode. No attention or runtime precision fallback is executed.

Raw benchmarks/results/2026-10-24/8046990/t3-gate-rdc-audit/; artifact
artifacts/trellis35/t3-gate-rdc-audit-2026-10-24.json. All789raw entries and current
source hashes verified. Read-only native-QMMA disassembly derived from the same
already-hashed module is additionally pinned in t3-gate-rdc-audit-sass-2026-10-24.json;
it adds no measurement and does not overwrite the initial record. Renderer
benchmarks/render_t3_gate_rdc_evidence.py idempotent; README/PERFORMANCE updated.

All five bounded investigations are closed: three accepted production gains,
T3GateUp/head variants rejected with full negative evidence. Remaining goal work:
fresh latest upstream llama build and matched same-machine comparison, plus a
direct day-start/full-stack control rather than multiplying historical ratios.
No active GPU jobs, no push. Commit the audit, then freeze final native binaries.
