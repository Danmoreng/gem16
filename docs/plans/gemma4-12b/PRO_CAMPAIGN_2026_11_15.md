> Historical task contract, closed or superseded. Current disposition is in
> [the archive index](../../evidence/benchmark-archive-2026-11-17.md).
> This file does not authorize a new experiment.

# Five bounded experiments adopted from the new Pro plan

Owner authorization: 2026-11-15, after clearing the previous Goal and receiving
`GEM16_12B_CODEX_PLAN_2026-11-15.txt`. The new Goal adopts that plan's five
experiments in order. This supersedes the tuning freeze and earlier paused/
cleared campaign only for these bounded experiments. It does not reopen prior
rejected searches or authorize Git publication. Unrelated dirty files remain.

Original plan/response copies and their SHA256, current Git/source/binary
snapshots and build flags are retained at
`benchmarks/results/2026-11-15/58febf4/12b-pro-followup/`.
The original instructions are evidence supplied by the owner, adopted within
the repository's permanent runtime/integrity rules. No other Pro attachment is
referenced as a required implementation file. Local binary fixtures, models,
tokenizers, candidate generators and profiler databases have been located;
missing new counters/tests are work to perform, not downloads to request.

## Fixed order and scope

1. **Post-attention boundary fusion, primary Ordinary:** one existing candidate,
   T1/H3840, post-attention only, scalar/capture null. Join unrounded FP8 O
   outputs and norm cases by layer/decode phase/family, restore mutating inputs
   before every replay, and add an identical-O-producer graph-chain comparison.
   Preserve the original cold-screen failure; the chain is supplementary.
2. **Assistant GEMV, primary D2:** first compare the historical warp-reduction
   rejection or establish a new concrete counter hypothesis. Only the prepared
   exact-tree candidate and five specified N/K shapes; validate all 42 calls
   per group and 126 actual cases. No PreProjection/head/warp-count sweep.
3. **FP8 QKV 64-bit loads, primary Ordinary:** a single reversible operand
   reconstruction for the declared T1 and T3 local/global shapes, both required
   to pass their independent screens. Keep O, Prefill and 26B paths unchanged.
4. **D2 NVFP4 Down weight/scale pipeline:** counter filter first; one two-stage
   K64 cp.async candidate for T3/N3840/K15360 physical BF16 output, with the
   existing arithmetic and no split-K or persistent representation change.
5. **Global D2 Target attention:** counter filter first; one two-eight-token
   staging-buffer candidate within existing staging capacity, rows3/KV-head1/
   context>=16384. Same 512-token split, all arithmetic/softmax/merge boundaries.

## Acceptance and stop rules

**Owner amendment, 2026-11-16 during E2:** the0.5% minimum is now required
from the combined package of the five current experiments, rather than each
single patch. Small positive measured primary effects may be carried as
provisional components after all non-performance qualification and existing
loss margins pass. Report the original single-experiment gate separately;
do not rewrite historical protocols. The final original-parent/package ABBA
decides combined performance. Ordinary and D2 gains are never added together.
No older rejected search is reopened by this exception.

**Owner amendment, 2026-11-15 after the E1 screen:** the owner explicitly
requests a full Decode test for each otherwise numerically/memory-valid
candidate despite a slightly negative operator confidence bound. A 95% lower
bound may extend to -2% overall Decode loss; positive-lower-bound operator and
Decode admission rules above/below are superseded for future qualification.
The minimum measured primary gain and exactness/runtime/protected-profile
requirements remain; positive local chains are evidence, not an E2E claim.
In particular E1 is reopened for safety checks, narrow integration and full
Ordinary/D2 comparison. Preserve its original-gate rejection as historical
evidence. A measured negative overall effect is never promoted as a gain.

- Freeze each hypothesis, allowlist, weighting, cache conditions and samples
  before measurement; one candidate, only proven harness/dispatch fixes.
- Bit-equal actual/boundary payloads, scales, intermediates and outputs;
  correct alias/guard/finite limits. Scoped memcheck/synccheck and racecheck
  for shared/pipeline changes, production SASS/dispatch/resources evidence.
- Report weighted operator effects and 95% intervals separately warm and
  evicted, plus native producer/consumer chains. These are diagnostic, not
  positive-confidence admission gates after the owner's amendment. Restore
  inputs outside isolated timing, then evict before the cold interval.
- Full 16K/1135-output preflight, then unprofiled adjacent AB/BA cohorts,
  3 warmups + 10 retained each. Single-trial primary pooled effect must be
  positive to remain a provisional component; the0.5% minimum applies to the
  final combined package. Primary and secondary
  Decode lower 95% bounds >-2% in both orders; Prefill and inverse
  TTFT lower bounds >-1%. Preserve all output IDs and D2/KV commit/rollback.
- Same immutable models/precision/workload/context/sequence/cache/sampling;
  GPU-resident single weights, tied aliases, no recurring allocation or JIT.
  Run proportionate protected 12B media/26B Compact Vision/internal NVFP4 gates
  for eligible runtime candidates. Linux does not establish Windows execution.
- Failures restore only owned candidate sources/binaries; skipped
  acceptance-only gates remain unrun for rejected candidates. Small positive
  results may remain provisional package components under the owner's dated
  amendment; final acceptance requires the measured combined package. No
  parameter search or rewriting original-gate evidence.
- At closure measure the cumulative accepted runtime against the original
  campaign parent. If no runtime candidate survives, report byte identity
  rather than invent a cumulative gain. Afterwards one separate bounded
  4096/8192 Prefill profile with no sixth runtime experiment.

## Initial status

All five new experiments are open. The current native computation sources
match the restored parent plus the accepted-for-benchmark startup chunk knob.
A targeted rebuild invalidates FP8, NVFP4, Assistant, quantization and engine
objects, then rebuilds both runner and CUDA tests before fresh baseline runs.
Older baseline samples will not be pooled with the newly identified parent.

Daily Berlin rule: start wrapping at 22:45, save and stop long jobs by 22:55,
no work after 23:00. Preserve completed evidence and a continuation note;
the deadline does not make incomplete experiments complete.

## Continuation at daily shutdown, 2026-11-15

E1 primary full OrdinaryABBA passes: pooled+0.708245%, AB+0.744559%
(95%+0.721992…+0.764807%), BA+0.671935% (+0.658173…+0.684220%).
Prefill/inference-TTFT non-inferiority passes; all40retained1135IDs exact.
FullD2candidate3+10:93.27998 versus freshparent93.11262tok/s, descriptive
only; formal adjacent secondaryABBA still open. Numeric132cases and scoped
memcheck/synccheck/racecheck pass. Production dispatch48calls/grid1/block256,
64registers, zero localbytes/thread. Offline shared2KiB versus CUPTI1KiB
reporting discrepancy retained for reconciliation; global local-total field
is not called per-kernel spilling. No promotion yet.

Candidate binary, CUDA tests, source/patch and all data are isolated in rawe1/.
All originalsrc/include files and canonical runner/tests are exact pinned
parent again; restored-source rebuild succeeded but had different binary
hashes, so pinned copies were explicitly restored. Do not pool a new build
with old samples. No commit/push.

Resume: verify environment/source/binary/model identities; finish E1 secondary
D2ABBA, protected12Bmedia/26BCompactVision/internalNVFP4 and productionKV/
commit/rollback gates, reconcile resource reporting, then promote only if
fully qualified. E2parent-only capture generator and rawe2/preparation.json
are prepared, not compiled or GPU-tested. E3–E5 and cumulative/Prefill-profile
closure remain open. The owner confidence amendment(-2%lower bound and full
Decode despite isolated-screen uncertainty) supersedes the original positive
bound admission rules; measured gain/bit/runtime constraints remain.

Evidence: docs/evidence/gemma4_12b/pro-campaign-2026-11-15.md; rawroot
benchmarks/results/2026-11-15/58febf4/12b-pro-followup/.

## Current continuation, 2026-11-16

E1 qualification is complete and the narrow runtime change is retained locally.
Full adjacent secondaryD2ABBA is effectively unchanged(+0.06491%); both Decode
and Prefill/inferenceTTFT margins pass. All1135outputIDs/1014proposals and the
entire Target KV arena are exact; protected text/media/session and host/CUDA
checks pass. The stale diagnostic build was detected, invalidated and replaced
with a forced, hash-distinct candidate rebuild. Primary archived-binary samples
were unaffected. Resource budget conservatively uses2KiB (runtime API/CUPTI1KiB).
Canonical runner/tests are the pinned E1candidate, not the original parent now.
See `e1/promotion-2026-11-16.json` and the qualified public artifact.

Next: E2parent-only capture126cases, weight identity, current barrier/reduction
counters and historical start filter. No E2candidate has run. E3–E5 and final
cumulative/4096–8192Prefill diagnosis remain open. No Git publication.

E2 parent capture and current-counter start filter now pass. The observer
separates the standalone graph's42 construction calls from the42 calls in
the conditional-owned chain body. All126 actual cases replay exactly, all42
borrowed matrices match unique pinned Safetensors tensors and the21 bindings
are shared across drafts. Current parent SASS has9 CTA barrier sites; the
five declared shapes show2.38–3.58 barrier-stall warp cycles per issued
instruction. Larger memory stalls are disclosed; this is a reason to test,
not evidence of a gain. `e2/start-filter.json` admits only the existing exact-tree
candidate. No candidate has been compiled or executed yet. Next: generate/build
that one derivative, numeric/finite/guard/sanitizer checks and warm/evicted/
native-sequence measurements, then full D2 and Ordinary qualification.

E2 continuation at11:12 Berlin: the single candidate passes882 numerical/
finite/guard checks, memcheck and synccheck. Racecheck passes all126 actual
full-grid changed-kernel calls with0hazards; two earlier all-arithmetic-repeat
race attempts exceeded300s and are not passes. Arithmetic values do not affect
shared addresses/barrier participation; all finite repeats still pass numeric/
memory/synchronization tests. Operator throughput+31.6587%warm/+12.5831%evicted;
all five current-counter cases reduce barrier stalls at unchanged36registers.
SASS shows9→1 CTA barrier sites and10FFMA sites in both;10static SHFL sites
include5fast-path plus5compiler fallback sites, not10executed shuffles claimed.

Fresh E2parent/candidate runner/tests/chat/core are preserved under
`e2/production-parent/` and `production-candidate/`. Only
`src/cuda/mtp/assistant.cu` is additionally changed over retained E1, and only
the five allowed shapes dispatch to the new kernel. Candidate remains
unqualified/pending. Full1135output Ordinary/D2 preflights pass on both builds.
`e2/model.py` is running D2thenOrdinary ABBA3+10, raw progress `model-progress.log`.
Current exec session15703 was confirmed live after preflights/firstD2cohort;
poll that handle or inspect actual process state, never restart blindly.
Parent runner SHA056f24f7e5641ebd4ae0ea1969928f7a8ea608f385aa59ac6582fa4ce400f732;
candidate e5f5cd7ab436185bf398fedc06a517c09eb63e0bb50f2b33917cbf24a0c032ea.

Next after the live coordinator actually terminates: run `e2/model_stats.py`.
Both modes have full cohorts even if D2 misses0.5%. `e2/native_sequence.py`
prepares a separate short actual Assistant-chain/dispatch audit after timing;
full KV/proposal and protected qualification remain for an eligible gain.
Do not pool profiler timings or E1samples. A failed gain restores only E2,
preserving E1 and unrelated Prefill/benchmark work. No commit/push/release.

Owner explicitly answered during the live E2trial: “Ja,0,5%für das Gesamtpaket”.
The dated amendment above supersedes the single-trial minimum, not numerical/
memory/profile gates. `e2/model_stats.py` retains the original gate result and
adds `eligible_positive_package_component`; protected/promotion guards now
use that component criterion. Final combined performance remains unproven.

Execution update11:32 Berlin: original session15703 is verified terminal after
the120s cooling limit before starting d2-3-parent. No timed cohort failed or
was repeated. `e2/model_resume.py`, current exec session38878, validates/reuses
the four completed preflights and first three D2cohorts and runs only the
missing final D2parent plus all four Ordinarycohorts. Cooling target remains
50°C; only allowed idle wait rises120→600s. At resume actual GPU was idle/P8/
below50°C. Progress: `e2/model-resume-progress.log`; provenance:
`e2/cooling-resume.json`. Poll38878, never restart15703 or repeat recorded data.

Continuation11:48 Berlin: first Ordinary AB pair complete, medians50.34004
parent/50.34194candidate, effectively unchanged. Both reverse-order Ordinary
cohorts remain. New `e2/after_model.py` is running under exec session23213,
watching the exact live coordinator PID29861/start_ticks666003 every30s.
It requires authoritative process termination and all eight successful cohort
records before invoking stats, native-sequence, full-state and protected gates.
Only a positive margin-safe component may be provisionally retained; otherwise
only E2 is restored. It then publishes the scoped E2outcome. It never starts
new benchmark cohorts or repeats completed samples. No new GPU work/source
changes for E3 until both38878 and23213 are confirmed terminal and E2outcome
is inspected. Progress: `e2/after-model-progress.log`; exact watched identity:
`e2/after-model-process.json`. Both daily shutdown guards remain enforced.

At12:09 Berlin E2 is complete; sessions38878 and23213 both verified terminal.
All80retained full requests pass. D2+0.39353%, Ordinary−0.01550%; both loss and
Prefill/inverseTTFT margins pass. Native sequence counts/grid/block match;
fullKV/proposals and32protected cases plus CUDA/host tests pass. E2 is retained
provisionally under the owner's package rule; original individual0.5% result
is still false. Canonical binaries are the measured E2candidate and all other
runtime sources retain E1/prior Prefill work. `e2/promotion.json`, `state.json`,
`protected.json`, `native-sequence.json`, `model-stats.json` and public outcome
`artifacts/gemma4_12b/assistant-exact-tree-2026-11-16.json` contain the evidence.

Next E3: CPU-only generator preparation exists at
`tests/cuda/build_12b_qkv_load64.py`, not generated/compiled/executed yet. A
planning permutation check covers32lanes/256random patterns; it is not CUDA
evidence. Generate the one T1+T3 candidate now, validate actual raw-FP32 outputs,
operand pairs/scales/guards and both declared shape families, then scoped
sanitizers/resources/timings and full Decode. No O/26B/Prefill changes.

E3 continuation12:33 Berlin: generated and built the one T1/T3 load64
candidate, no variant search. All144actual/finite comparisons pass, with
31,943,520 GPU operand-pair comparisons exact, unchanged rawFP32/scales/input
bits/logical tails/NoV/guards. Memcheck/synccheck pass all comparisons; racecheck
passes all24real changed-kernel cases at full grid/block,0hazards. Isolated
warm/evicted effects: Ordinary+10.2782%/+0.4493%(cold95includeszero),
D2−7.7887%/−29.8706%. This mixed result is not promoted; owner requires full
Decode for safe candidates. Actual SASS has LDG.64 loads and4SHFL perunrolled
Kstep, same5T1/15T3 staticMMA sites as parent. T1regs40→44; T3regs46→40;
no stack/local. Use `e3/instruction-resource-facts-corrected.json`: an earlier
parser incorrectly counted64bit address operands as LDG.64 opcodes and is
explicitly superseded by `instruction-parser-note.json`; raw SASS unchanged.

Production uses exactly the tested helper/both kernels via generated
`src/cuda/fp8/detail/twelve_b_qkv_load64.cuh`; only
`src/cuda/fp8/sm120.cu` dispatches the two exact12B shapes to them. Normal prompt
prefill still uses separate CUTLASS physical-BF16 projections; O/26B/other
tokencounts stay old. All other initialsrc/include hashes remain accepted
E1+E2/prior Prefill state. Candidate is temporary/unqualified.

`e3/model.py` is live in exec session19399, Ordinary thenD2ABBA3+10 after four
full1135output preflights. Preflights are exact; D2 single point93.97764→91.09092
is about3%slower, so secondary loss may reject the whole declared T1/T3trial.
No T1-only salvage or broader tuning. First Ordinaryparent cohort was running
at handoff. Progress `e3/model-progress.log`. Do not rerun the coordinator.
`e3/model_resume.py` is prepared for a verified terminal cooling timeout only:
valid recorded cohorts are reused, unchanged50°C target with up to600s idle
wait. Original live model still has120s cooling bound. Daily deadline applies.

Next: observe19399 to actual completion/terminal state, complete missing cohorts
if needed, then `e3/model_stats.py`. Need native dispatch/chain evidence and
proportionate acceptance gates for any margin-safe positive component. If
rejected, restore only E3FP8source/header from `e3/production-parent/`, preserving
E2Assistant and E1norm. Fresh matched parent/candidate runners/tests/chat/core
and source snapshots are in `e3/production-{parent,candidate}/` and
`production-inputs.json`. No E3post-model coordinator is running yet. E4/E5 and
final combined/Prefill-profile closure remain open. No commit/push/release.

At12:50 Berlin the original E3model process PID39115/start_ticks1012279 is
still verified live (exec19399); three Ordinary cohorts are complete and the
final Ordinaryparent cohort runs. An additional serialized coordinator
`e3/after_model.py` runs under exec66593. It observes that exact process
lifetime, refuses incomplete/failed cohorts, then runs stats and separate
native T1/T3 graph/dispatch traces. A rejected component restores only E3's
FP8source/new header and the exact measured E1+E2parent binaries. A positive
component stops for remaining state/protected qualification; it is not promoted.
Only after a rejection/restoration may the coordinator run E4parent-only
capture/counters. It never repeats a benchmark cohort or generates an E4candidate.
Progress: `e3/after-model-progress.log`; live identity: `after-model-process.json`.

CPU-only capture generators now exist for E4Down and E5global Target attention.
Their generated sources and frozen parent-counter protocols are under
`e4/capture-sources/` and `e5/capture-sources/`. Neither has yet compiled or run
on the GPU at this checkpoint. E4captures the four declared layers at groups
0/250/506; E5captures all eight global bindings and all three rows at the same
phases, preserving mutable caches/controls and complete partial/LSE bytes.
Both separate standalone construction calls from the actually executed
conditional-owned graph. E5execution additionally requires explicit E4closure;
no E5GPU job is running. Current-counter decisions remain open for both.

At12:55 Berlin all40Ordinary retained requests are complete. E3Ordinary gain
is+0.151928% pooled (AB+0.103285%,95%−0.001690…+0.174385%;
BA+0.200603%,+0.126789…+0.310544%). Both prompt margins pass. This is an
interim mode result, not candidate acceptance; D2ABBA is now running its first
parent cohort in the same verified live process. Raw:
`e3/ordinary-complete-interim.json`. The coordinator remains live in66593;
do not start another GPU task or rerun completed cohorts. E4's exact production
parent Down SASS is extracted read-only from the archived E3parent core;
counter-start filter is still unmeasured. E4/E5capture generators and19Python
files parse, but parsing is not a CUDA compilation/safety or counter pass.

### Existing Prefill boundary conflict found during closure preparation

Source inspection finds an existing recurring weight-format preparation in
12Bprompt Prefill: `inference_engine_prefill_layers.cuh` calls the CUTLASS
Gate/Up/Down wrappers per layer/chunk; `nvfp4/cutlass_sm120.cu` launches
`PrepareWeightKernel` from both projection wrappers (currently lines542/615),
converting tiled weights/scales to transient CUTLASS scratch. This conflicts
with AGENTS.md's permanent prohibition on recurring weight repack. No owner
exception is inferred. It is inherited by the original parent and all current
patches; the five Decode experiments do not introduce or repair it. Do not
claim that complete recurring inference already satisfies that invariant.
Keep the declared no-sixth-runtime-experiment scope: verify actual dispatch and
include all such preparation time in the final4096/8192Prefill diagnosis,
then report this concrete pre-existing limitation and any separate follow-up.
Decode-relative correctness/performance evidence remains separately scoped.

### Current continuation, 2026-11-16 at13:33 Berlin

E3 is closed and rejected after all80retained full requests: Ordinary+0.151928%,
D2−3.004012%; both D2confidence bounds fail−2%. Numeric/sanitizer/operand and
actual native dispatch/instruction checks pass. Only E3FP8source/header is
restored, exact measured E1+E2runner/tests/chat installed; `e3/closed.json`
audits closure. No subset rescue. Public full outcome and README are updated.
The old exec19399/66593 handles are both verified terminal.

E4current parent-only capture/counter filter is complete and admits the one
weight/SFBpipeline. Twelve real BF16cases plus complete1135outputs/counters
match. DRAM38.53–44.22%, long-scoreboard28.42–32.96cycles/issued instruction;
OMMAconsumers account for93.29–94.29%of sampled not-issued states as long
scoreboard waits. Those consumers depend on both operands; weight dominance
is an explicit footprint/reuse inference, not proven unique causality.

The one E4candidate now passes96actual/finite comparisons and44,236,800
staged operand-word comparisons. Memcheck/synccheck pass the entire suite;
racecheck passes all12actual full-grid pipeline calls plus four full-weight
staging audits,0hazards. One helper stages8rows of32Bweight+4BSFB in each
of two warp-private buffers; all240K64steps, A/SFA, native division and BF16
epilogue remain old. Source __syncwarp and per-copy-thread async wait are
retained; SASS shows LDGSTS/LDGDEPBAR/DEPBAR plus reconvergence, no separate
WARPSYNC opcode is invented. Source shared4608B; labELF5632B and production
ELF4608B are distinct measurements. Actual native trace remains pending;
production61registers,0stack/local (parent64registers).

Isolated Down throughput−38.7189%warm,−17.5654%evicted. Owner requires full
Decode for safe candidates, so this is not a stop/promotion by itself. Fresh
matched production builds under `e4/production-{parent,candidate}/` preserve
E1+E2 and change only `src/cuda/nvfp4/sm120.cu` plus generated
`src/cuda/nvfp4/detail/twelve_b_down_weight_pipeline.cuh`. Canonical binaries
are the temporary unqualified E4candidate, not the accepted package now.
All four full preflights pass (D2single points parent94.1462/candidate92.4560,
descriptive only). Full D2thenOrdinaryABBA3+10 runs in exec23468,
PID51368/start_ticks1381968; progress `e4/model-progress.log`. Cooling remains
<=50°C with600s maximum idle wait from the start; no cohort reruns.

Serialized `e4/after_model.py` is live in exec23715 and observes exactly that
PID lifetime. It requires all eight completed successful cohorts before stats,
actual production SASS and native Ordinary/D2dispatch traces. A positive
component stops for state/protected qualification. Otherwise it restores only
E4, publishes its outcome, audits `e4/closed.json` with pinned canonical runner,
then executes the prepared E5parent-only capture/counters. No E5candidate is
generated by it. Progress `e4/after-model-progress.log`. Do not start another
GPU job or restart these coordinators. E5's filter decision remains open.

The final combined original-parent/package comparison is still open. A bounded
4096/8192Prefill-only profiler script is prepared under `prefill-profile/`,
guarded on all trial closures and a pinned final runtime. It has not run; all
weight preparation is included and the inherited runtime-rule conflict above
must remain disclosed. No sixth runtime experiment, commit/push/release.

At13:47 Berlin all40retained E4D2requests are complete: pooled−1.687957%,
AB−1.735524%(95%−1.832925…−1.659655%), BA−1.640405%
(−1.684216…−1.597989%). The−2% non-inferiority margin passes, but a negative
primary effect cannot be retained as a gain. Ordinary's four full cohorts are
still required by the frozen protocol; original process51368/exec23468 and
post-model exec23715 remain live. Raw `e4/d2-complete-interim.json`. Do not
mark E4closed or begin E5GPU work before actual completion and restoration.

CPU preparation now also covers the final direct comparison: `final/prepare.py`,
`model.py`, `model_stats.py`, `state.py`. They have not run. Preparation requires
all E3–E5closures and E5's `canonical_runner` plus complete
`runtime_source_sha256` inventory, then pins/copies the original parent runner
and selected package, checks immutable models/helper, and snapshots exact
src/include bytes. Final statistics preserve0.5% per-mode flags plus both-mode/
at-least-one-mode flags without adding percentages or making an automatic
promotion decision; whole-state and closure audit remain required. E5closure
must include that source inventory when its outcome is eventually written.
An additional source audit confirms all245expected runtime source/header hashes
match E1+E2 plus only temporary E4, including the new E1header absent from the
original initial inventory. E4restoration now explicitly checks that header too.

E5's parent static SASS/resource record is prepared from the exact archived
E1+E2core:64registers,41,056staticShared bytes,0stack/local. This is no counter
admission; no E5candidate has been generated or executed. Current capture/
counter jobs remain serialized behind E4closure.

### Current continuation, 2026-11-16 at14:21 Berlin

E4 is fully closed and restored after80retained requests. D2−1.687957%,
Ordinary−0.003331%; all loss margins pass but primary gain is negative.
Production instruction/native-trace facts pass, with288narrow Down calls in
six native D2groups, no new Ordinary calls, identical other counts. Actual
production shared4608B/registers61/0local are recorded. Exec23468 is terminal.
The post-model exec23715 ended after successful restoration but a pure report
filename error (`instruction-resource-facts-corrected.json` belonged to E3).
Corrected helper and `e4/after_publication_resume.py` reuse existing data,
publish the outcome, audit closure and run only previously unrun E5stages.
Resume exec65596 is verified terminal/success. No benchmark or trace reruns.
E4public outcome, README/active decisions and evidence are updated.

Canonical runtime is accepted E1+E2source again; E4header removed. Exact
measured E4parent runner SHA
`fdb590e87aac8b88bd758b498bffbaa31337748b185e88c0cee31456c3a8e3b3`
is installed. E4parent binaries have their own identities, not E3's older
parent hashes; do not mix builds. No current GPU job is running at checkpoint.

E5parent capture/counters now pass:24actual partial/LSE cases across all eight
global bindings, all three rows, groups0/250/506; full1135IDs/counters exact.
Mutable caches/query/controls are exported. Actual source-correlated
K/Vcopy-wait barriers0x720/0x23f0 immediately follow async waits0x710/0x23e0.
They account for19.51–21.28%of sampled not-issued states as long-scoreboard
waits, not wall time. DRAM7.13–8.33%, issue60.76–63.04%, math-throttle
0.170–0.180cycles/issued instruction. Conversion/reduction/fixed-latency stalls
remain substantial; data support a bounded overlap experiment, not a claim
that memory dominates. `e5/start-filter.json` admits only the declared two8token
staging candidate before any generation/compilation/execution. Source64regs/
41,056staticShared/0local; NCUallocated Shared42,112B is separately named.

Next: generate the single private clone of
`SplitOnlineDecodeAttentionFp8GlobalGqaKernel<false,true,true>` with two8token
buffers within the existing8192B. Prime/wait/synchronize the first tile;
prefetch the alternate next tile while computing current QK/PV, wait per copy
thread and block-synchronize before reads/reuse. Keep complete512token softmax,
scores/LSE, token/dimension/FMA/shuffle/max/sum/exp/log/reciprocal and merge
order, all other paths and the41,056B source budget. No tile/head/split search.

Use `e5/fixtures/captures.json` and the already built parent helper/observer.
Numerical tests must reproduce captured actual parent partial/LSE first, then
compare candidate bytes, score/probability boundaries and unchanged production
merge outputs. Diagnostic score taps may be separate instrumented clones,
never the timed/production kernel. Seed untouched invalid split outputs from
their expected bytes; test8/16/512tails, empty splits, context-tier/last-capacity
boundaries, priming/drain/reuse/zero-fill and guards. Sanitizers/resources/
actual dispatch/producer-chain and separate warm/RMW-evicted timing follow,
then full D2/Ordinary even for a safe negative candidate. Integrate only
TargetkRows3/kv1/context>=16384 in the fixed-controlled wrapper; Ordinary/
Assistant/26B stay old. No missing Pro attachment or user answer is required.
Final original-parent/package ABBA and4096/8192Prefill profile remain open.

### Current continuation, 2026-11-16 at15:20 Berlin

E5 numerical/safety work passes:166 actual/finite/control comparisons include
raw scores, pre/post-softmax probabilities, partials/LSE and unchanged merge.
Memcheck/synccheck cover the full suite. First whole-topology racecheck exceeded
300s; repeating exactly that command with600s completes in302.62s,0hazards,
covering all24actual and31control-boundary configurations. A diagnostic merge
wrapper initially declared destructive LSE scratch const; corrected wrapper
ABI compiles, with the candidate kernel body hash unchanged. No math variant
or parameter search is introduced. Failed logs are preserved.

Isolated throughput+2.93029%warm/+2.21180%evicted, both positive95bounds.
Actual production70registers versus64parent, same41,056B static shared,
zero stack/local; lab42,080B is a separate packaging measurement. FFMA/SHFL
static sites remain82/40; LDGSTS sites2→4. Actual native trace remains pending.
Only the controlled TargetkRows3/kv1/long-context branch is integrated;
ordinary,Assistant,26B and all other rows/tiers remain old.

Full40retained D2 requests complete: pooled−0.016676%, AB−0.062176%
(95%−0.165518…+0.015566%), BA+0.028823%(−0.006669…+0.065567%).
Thus the isolated gain has not transferred to full D2. This is an interim
result, not a final promotion decision; all four Ordinary cohorts remain
required. Raw `e5/d2-complete-interim.json` is immutable.

Original model processPID62756/start_ticks1887638 (exec28088) and serialized
after-model coordinator(exec85893) are verified live. Ordinary parent0 is
complete and candidate1 is running. The coordinator waits for all eight
successful cohorts, audits stats/production instructions/native dispatch,
then either runs acceptance gates or restores only E5. After audited closure
it starts the prepared final original-parent/package ABBA and full state test.
Progress `e5/model-progress.log`, `after-model-progress.log`; final progress
will be `e5/stage-final-model.stdout.log`. Never start another GPU job or repeat
completed cohorts while these are live.

CPU-only final acceptance script `final/accept.py` is prepared, not executed.
It checks original-parent/package fresh80retained requests, full state,
unchanged arenas and source inventory, qualified E1/E2 and closure decisions.
The0.5% package threshold is checked separately per mode, never added; at
least one mode must reach it and both independently satisfy owner loss margins.
Final acceptance, bounded Prefill diagnosis/public report/README and completion
audit remain open. No commit/push/release is authorized by this Goal.

At15:31 Berlin CPU-only closure preparation is complete. New
`tools/analyze_12b_prefill_profile.py` joins launch correlations to existing
Prefill/layer NVTX ranges, retains actual launch/resources and source/order
logical GEMM shape attribution, explicitly not device-argument capture.
On the historical full Ordinary trace it selects2005Prefill kernels,
96layer phases(16global/80local),656FP8/NVFP4 GEMMs and288weight preparations,
excluding1,093,178kernels outside Prefill. This validates analysis only, not
current-package timing; raw `prefill-profile/analyzer-historical-validation.json`.
The prepared two new captures remain Prefill-only and require all their kernels
to correlate inside that boundary. Historical full4096boundary output also
matches all1135original-parent IDs; one-output throughput cohorts are not
mistaken for full-Decode evidence.

`final/after_model.py` now runs in exec81754, watching exact existing
E5after-model coordinatorPID65002/start_ticks1974822, verified live at start.
It waits for authoritative termination and final model/state evidence, then
runs only unrun `final/accept.py`, the two Prefill captures, and the sanitized
`tools/write_12b_campaign_closeout.py` export. No repeats or runtime changes.
Progress `final/after-model-progress.log`; its execution outputs belong to
`final/accept-package.*`, `prefill-diagnosis.*`, `publish-package.*`.
Any failed stage stops rather than silently rerunning evidence. Final manual
documentation/README and requirement-by-requirement audit remain mandatory.
E5model28088, E5coordinator85893 and finalcoordinator81754 are the three live
serialized handles; only the model uses GPU now. Ordinary candidate2 has begun.

### Current continuation, 2026-11-16 at15:44 Berlin

E5model exec28088 is verified terminal/success. All80retained full requests
complete; D2−0.016676%, Ordinary−0.034206%. Actualproduction SASS and native
Ordinary/D2trace pass;48new T3global calls across six groups,70regs/same41,056B
staticShared/0dynamic/local, all other kernel/grid/block counts exact.
E5 is rejected and only its source/header restored; exact E5parent runner/tests/
chat installed, E1/E2and all other source hashes checked. Public outcome and
README/evidence/active decisions updated. Acceptance-only state/protected
gates after rejection are unrun, not passed. `e5/closed.json` pins246current
runtime sources and canonical runnerSHA
23b9bbf01eea186553015c49005906145c7b55be615d7158a54ff0180733a38c.

All five individual trials are complete. E5coordinator exec85893 is still
live, now running the final direct original-parent/package comparison:
`final/production-inputs.json` pins exact original runner234376...and current
23b9bb...without rebuilding/relabeling either. First Ordinary parent preflight
passes; other full preflights and fresh80retained ABBA remain. Progress
`e5/stage-final-model.stdout.log`. Its child is `final/model.py`; never start
another GPU job or repeat a cohort. Finalcoordinator exec81754 still watches
exact65002/start_ticks1974822 and will serialize acceptance/profile/export
after successful final state. Manual documentation and completion audit remain.

### Owner stop, 2026-11-16 at15:57 Berlin

Owner explicitly requested aborting the current run and ending work. The Goal
is paused, not complete. Do not automatically resume any job.

All owned coordinators and their descendants were stopped with SIGINT/SIGTERM,
without SIGKILL. Authoritative post-stop inspection finds no owned live process
and no GPU compute process. Raw stop record:
`final/owner-stop-2026-11-16.json`. Exec85893/81754 and their model descendants
must not be restarted from old commands.

All five individual experiments remain complete: E1 retained with Ordinary
+0.708245%; E2 correctness-qualified/provisional with D2+0.393532%; E3/E4/E5
rejected and restored. Exact E1+E2canonical runnerSHA23b9bbf01eea186553015c49005906145c7b55be615d7158a54ff0180733a38c
is unchanged. All raw evidence and unrelated dirty files are preserved.

Final comparison is interrupted: four full preflights and only
`final/ordinary-0-parent.json` completed. `ordinary-1-candidate` is interrupted
(execution returncode null), not a valid cohort; its logs/partial output remain.
There is no final/model-stats.json, state.json or selected-runtime.json and no
accepted cumulative gain. The bounded4096/8192Prefill profile has not run.
README and evidence explicitly mark the pause. No commit/push/release occurred.
If the owner later requests resuming, inspect actual artifacts first; preserve
the interruption and use a separately declared continuation/new cohort identity,
never rerun exclusive-output scripts wholesale or pool incomplete samples.
