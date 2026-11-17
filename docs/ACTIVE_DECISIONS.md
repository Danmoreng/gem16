# Active decisions

**Updated:** 2026-11-17 · **Track:** productization · **Status:** stable v0.3.0 publication authorized

Permanent rules in [AGENTS.md](../AGENTS.md) remain binding. Read the
[product contract](PRODUCT_CONTRACT.md) and the narrow task contract next.
The [pre-consolidation record](archive/ACTIVE_DECISIONS_2026-09-04_BEFORE_CONSOLIDATION.md)
preserves the complete decision history, including superseded intermediate stages.
The existing numbered decisions retain their identities below.

### Owner update: 12B fixed D3 sampling and selectable context (2026-11-17)

The owner requests optimized fixed-D3 with checkpoint sampling and selectable
context through the 12B maximum of 262144. This supersedes the previous
greedy-only and capacity-17519 selection limits. Use the existing Target sampler
and transactional verifier with the selected arithmetic. D2 remains the Studio
default. The earlier 16K timing and quality evidence applies to its original
workload. The current selection contract is in [MTP math modes](MTP_MATH_MODES.md).

### Owner update: optimized12B D3 default; reference retained (2026-11-17)

The owner selects the fastest measured12B D3 path as default, retains reference
arithmetic as an explicit choice, and authorizes committing/pushing implementation,
diagnosis and evidence. This supersedes the earlier exact-default, opt-in-only
and no-commit/no-push restrictions for these completed12B D3 slices. It does not
authorize a release or change26B. The [math-mode contract](MTP_MATH_MODES.md)
defines automatic optimized selection for supported fixed-D3/FP8-KV/greedy/
capacity17519, visible native reference selection elsewhere, and visible failure
for unsupported forced optimized mode. Target acceptance/cache semantics stay
intact. The32-question quality screen and numerical differences do not prove
reference-distribution identity or general quality equivalence.

### Historical owner update: separate exact and experimental12B D3 math (2026-11-17)

The owner now authorizes a separate, explicitly selected 12B D3 path with
changed arithmetic/rounding and potentially different output IDs, aiming to
approach vLLM speed. This supersedes earlier identical-output restrictions only
for this experimental path; the exact default and other profiles stay intact.
The [bounded task](plans/gemma4-12b/D3_FAST_ATTENTION_2026_11_17.md) follows the
[kernel diagnosis](plans/gemma4-12b/D3_PATH_DIAGNOSIS_2026_11_17.md).
This is experimental performance work, not quality qualification or public
profile promotion. No new commit, push or release is authorized.

The owner subsequently requests continuing with the next measured kernels and
an initial quality comparison. This supersedes the completed-first-slice boundary
for the [bounded follow-up](plans/gemma4-12b/D3_FAST_FOLLOWUP_2026_11_17.md): paired
T4 norm/FP4, Assistant global FP8 and local Target FP8. It reaches112.10tok/s
versus107.34 previously and117.13 vLLM. Exact outputs/drafts remain unchanged;
the experimental32-question GSM8K screen scores31/32 versus29/32 exact, with
all unscored answers truncated. This is not general quality equivalence or a
promotion. [Evidence](evidence/gemma4_12b/d3-fast-followup-2026-11-17.md).

### Owner update: close and publish the reviewed working tree (2026-11-17)

The owner authorizes retaining final useful measurements/qualification/tools,
removing obsolete candidate/DFlash helpers, intermediate reports and duplicate
plots, then committing and pushing the closed archive. This supersedes older
no-commit/no-push restrictions for these completed records only. It does not
resume stopped campaigns, authorize new optimization work or publish a release.

The current 12B Ordinary/MTP state is committed as `23da107`: fixed D2/D3/D4
verification and Assistant are unchanged; only the Ordinary tail enables the
qualified Ordinary kernels. Storage policy/helper is committed as `d789159`.
DFlash is rejected; both draft weight files are deleted. External MTP depth
references and immutable raw measurement values remain retained.

Use the [archive index](evidence/benchmark-archive-2026-11-17.md) for final
retained/rejected dispositions. Historical intermediate statuses below describe
their dates, not a current queue. The original failed N1 gate is separate from
the later owner-approved RoPE qualification. No additional combined-package
speedup or completed Prefill diagnosis is claimed for the stopped Pro campaign.

### Owner update: standing release publication policy (2026-11-08)

The owner authorizes publishing `v0.3.0` as a stable downloadable release.
For this and future owner-requested versions, publication follows successful
applicable build, host, Studio and archive-integrity checks plus tests
proportional to the actual change. Report the checks run and known limitations.

Full two-platform GPU/model/SDK/agent qualification, clean-machine and complete
model-download campaigns, and a full `verify_release_gates.py` manifest are
removed from the mandatory release process. They are not pending release
blockers or automatically scheduled follow-up work. An owner instruction to
publish a version requires no further qualification exception or confirmation.

This permanently supersedes the former future publication requirements in
standing decisions 1 and 10, the product contract's release boundary, roadmap
C07/C08/C09 and workflow comments, including the restriction of earlier
exceptions to 0.2.x. Proportional correctness/regression tests, truthful evidence,
runtime integrity and performance-claim rules remain binding. Omitted campaigns
are not reported as passed. The
[previous decisions](archive/ACTIVE_DECISIONS_2026-11-08_BEFORE_RELEASE_POLICY.md)
preserve the superseded requirements and historical publication authorizations.

### Owner update: central current llama.cpp benchmarks (2026-11-03)

Future gem16 comparison benchmarks use the shared central llama.cpp CUDA build
at the latest official master, rather than a project-local build or a fixed
llama.cpp commit. Verify the upstream revision before each run, record the
actual executable and commit, and fail visibly when freshness cannot be
verified. This supersedes earlier fixed-revision requirements for **future
benchmark runs and benchmark builds**, not the immutable historical results
or the separate pinned source recipe used for historical GGUF conversion.
A changed reference revision requires a new quality result directory; do not
resume a partially recorded reference with a different executable.

### Owner update: AP1 decode measurement refresh (2026-11-04)

The owner authorizes AP1 from the returned Pro review: verify the source
findings, repair diagnostic cost attribution/speculation telemetry/replay
provenance, and preserve fresh ordinary/D2 baseline and separate diagnostic
captures before starting AP2. This supersedes standing decision 7's freeze
only for this bounded measurement work. Production `src/` remains unchanged;
AP2–AP5 runtime candidates are not started as part of AP1. Historical evidence
is retained. Unobserved work stays unknown, and observer/profile timings are
excluded from primary throughput. The task entry is
`plans/gemma4-26b/AP1_MEASUREMENT_2026_11_04.md`. No commit, push or release is
part of this owner request.

AP1 is complete locally. The
[measurement evidence](evidence/gemma4_26b/decode-ap1-measurement-2026-11-04.md)
records current baselines, corrected attribution, same-capture groups and scoped
memory-check limitations. Its Go concerns one bounded AP2 R3-QKV grid trial;
AP1 itself made no production runtime change.

### Owner update: bounded AP2 R3-QKV trial (2026-11-04)

After AP1's completion the owner explicitly requests proceeding to the AP2
performance experiment. This supersedes decision 7's freeze only for flattening
the useful native 26B fixed-R3 Q/K/V CTA ranges (256 local / 288 global).
Preserve the accepted arithmetic, packing/padding, weights, scales, K ordering,
global Q/K-only behavior and every other dispatch path. Begin with one offline
candidate and current actual-input bit/resource/timing screens. Only a winning
screen may enter production qualification and adjacent unprofiled D2 AB/BA;
Ordinary and protected profiles must pass. No commit, push or wider AP3–AP5
implementation is authorized. Task: `plans/gemma4-26b/AP2_QKV_FLAT_2026_11_04.md`.

The AP2 trial is closed without promotion. Warm QKV replay improves1.323%,
but full D2 improves only0.080633% pooled and BA's lower95bound is negative.
The predeclared both-orders gate fails; Ordinary is skipped and the exact
parent sources/canonical executable are restored. All numerical, scoped memory
and protected-profile checks passed. The
[negative record](evidence/gemma4_26b/qkv-flat-grid-2026-11-04.md) preserves the
candidate, measurements and actual dispatch. No further grid/address/prefetch
search or AP3 implementation follows from this closed exception.

### Owner update: AP3–AP5 experiment Goal (2026-11-04)

The owner now explicitly authorizes implementing/testing AP3, AP4 and AP5,
retaining and committing qualified gains, otherwise restoring the parent and
documenting rejected candidates. This supersedes decision7's tuning freeze and
the earlier AP1/AP2-only scope for these three bounded experiments only:
R1 global26B exact warp normalization (Target and Assistant), final unused
fixed-D2 FP8 feedback projection/quantization, and width2816 rows1/3 attention
postnorm warp tail. Keep each separate, preserve numerical/format/runtime
boundaries and all other paths; no reopening of older negative searches.
Commits are authorized for accepted gains and their necessary evidence/tests;
no push or release is authorized. Follow the predeclared numerical/memory/
protected-profile/ABBA gates in `plans/gemma4-26b/AP3_AP5_GOAL_2026_11_04.md`.

AP3 closes without promotion: warm operator+1.8666%, pooled fullOrdinary
+0.083396%, but BA95lower-0.112272% fails the predeclared both-orders gate.
The exact parent attention source and canonical executable are restored before
AP4 starts. Numerical, scoped memory, protected profiles and full output/proposal
parity passed. [Negative evidence](evidence/gemma4_26b/r1-global-warp-normalization-2026-11-04.md).
No further AP3 tuning is authorized by this result; AP4/AP5 remain in scope.

AP4 also closes without promotion: D2 AB-0.116104% and BA-0.060738%, both
95%intervals include zero. Pooled+0.011921% does not satisfy the independently
predeclared both-orders gate. Exactly502unused quant/projection pairs disappear;
all other kernel counts and all1135outputs/1004proposals remain exact. Nine
regressions and the scoped32output memory screen pass. Remaining acceptance-only
gates are not claimed after failure. Exact parent source/binary restored;
[negative evidence](evidence/gemma4_26b/final-d2-feedback-2026-11-04.md). AP5 remains open.

AP5 closes at the offline operator gate without a production patch:8actual
R1/R3cases, geometric mean-0.030264%,95%[-0.069885%,+0.017713%]. Numerical,
scalar/alias/finite-boundary, memory/synchronization and resource/instruction
screens pass; SASS CTA barrier sites fall9to5 without a winning timing screen.
Production/E2E/protected gates stay unrun. [Evidence](evidence/gemma4_26b/attention-postnorm-warp-2026-11-04.md).
The AP3–AP5 Goal is now closed with three rejections and no retained runtime
change, commit or push. Exact parent source/canonical executable restored;
[final audit](../artifacts/trellis35/ap3-ap5-closeout-2026-11-04.json). No further
search, subset promotion or wider tuning follows from this completed exception.

### Owner update: five bounded Trellis35 decode experiments (2026-11-05)

The owner authorizes testing N1, N3, N2, N4 and N5 in that order, and retaining
qualified wins. This supersedes decision 7's tuning freeze for these five
candidates only. The [task entry](plans/gemma4-26b/N1_N5_DECODE_GOAL_2026_11_05.md)
defines the screen thresholds, ABBA qualification and restoration rules.
Git publication is outside this scope.

All five experiments are closed. N1 passed primary D2 (+1.225% AB,
+1.645% BA) and missed the Ordinary secondary point-estimate requirement
(−0.0731% AB, −0.0518% BA). N3, N2, N4 and N5 failed their operator screens.
The exact qualified parent is restored. See the
[evidence](evidence/gemma4_26b/n1-n5-decode-2026-11-05.md).
This bounded exception is complete.

### Owner update: qualify the Trellis35 Fixed-D2 RoPE candidate (2026-11-05)

The owner authorizes qualifying the measured N1 candidate and publishing it
on `main` if it passes. For this follow-up only, the lower 95% confidence
bound for Ordinary throughput must exceed −0.20% in both orders. This
supersedes the earlier additional requirement for nonnegative point estimates
in both orders. The original experiment and its decision remain historical
evidence. D2, prefill, TTFT, numerical, memory, lifecycle and protected-profile
requirements still apply. This is a bounded exception to decision 7 for the
measured Fixed-D2 RoPE patch.

N1 passed qualification on Linux SM120a. The native 16K Fixed-D2 throughput
rose from 197.73 to 200.62 tok/s (+1.46%). Both Ordinary confidence bounds
passed the follow-up margin. GPU memory, protected profiles and parent/candidate
lifecycle parity passed. The in-flight cancellation probe reproduced the same
pre-existing streaming status issue on both builds. See the
[qualification evidence](evidence/gemma4_26b/n1-rope-qualification-2026-11-05.md).

### Owner update: bounded Trellis35 Ordinary Decode follow-up (2026-11-05)

The owner authorizes measuring the actual inter-graph gap on the N1 baseline.
This supersedes decision 7's freeze for that diagnostic and the conditional
next step in the [task entry](plans/gemma4-26b/ORDINARY_DECODE_FOLLOWUP_2026_11_05.md).
A large reproducible gap permits one bounded Ordinary greedy graph
prototype. A small gap directs the follow-up to real-input Ordinary
Routed-GateUp diagnosis. Previous negative trials remain historical.

The follow-up is complete. Both timing methods place the inter-graph gap
at roughly 10–13 µs and below the prototype gate. The real-route GateUp
measurement separates warm and evicted conditions and records the extra
K3 instruction work. See the
[evidence](evidence/gemma4_26b/ordinary-decode-followup-2026-11-05.md).

### Owner update: Ordinary local Attention and FP8 O Goal (2026-11-05)

The owner authorizes the two sequential optimization experiments in the
[task entry](plans/gemma4-26b/LOCAL_ATTENTION_O_GOAL_2026_11_05.md). This
supersedes decision 7 only for these exact-scope candidates and their
qualification. Preserve quantization, math, cache semantics and protected
profiles. Earlier negative layout and generic prefetch trials remain closed.

Qualification is complete. Keep the staged V split for Trellis35 Ordinary
with a 1,024-slot local ring. The 16K native campaign measured +1.68%
Ordinary with exact outputs and matched peak VRAM. D2 passed the
−0.20% margin in both orders and the protected-profile checks passed.
The FP8 O two-tile CTA missed its primary operator gate and is closed.
See the [evidence](evidence/gemma4_26b/local-attention-o-2026-11-05.md).

### Owner update: five bounded 12B Decode experiments (2026-11-15)

The owner authorizes the five experiments in
[`DECODE_FIVE_STEPS_2026_11_15.md`](plans/gemma4-12b/DECODE_FIVE_STEPS_2026_11_15.md),
retaining and committing only measured gains with identical outputs. This
supersedes standing decision 7's tuning freeze for these five bounded 12B
candidates only. Preserve model/precision, all numerical boundaries, residency,
allocation and protected profiles. Existing unrelated dirty work is separate.
Record rejected candidates and restore their runtime changes. No push or
release is authorized.

### Owner update: matched 12B Prefill comparison (2026-11-15)

The owner explicitly pauses the five-step Decode Goal and requests a current
12B three-engine Prefill comparison with matched chunk sizes, including1024
and larger valid sizes up to8192. This supersedes decision7's tuning freeze
only for explicit benchmark chunk/context controls, proportional validation and
new immutable measurements. Existing numerical/kernel/format/residency rules
remain. Decode optimization is not resumed by this exception. Task entry:
`plans/gemma4-12b/MATCHED_PREFILL_2026_11_15.md`. No push/release is requested.

The comparison is complete locally: all 15 chunk/engine combinations have
three warmups and ten retained samples. Common 1,024-token chunks yield
5,246.93 / 4,839.77 / 6,871.12 tok/s for native / llama.cpp / vLLM.
All final llama.cpp points use current central revision `18b74fa`; the earlier
screen revision remains separate. Full native output checks pass at all five
sizes. [Evidence](evidence/gemma4_12b/matched-prefill-2026-11-15.md).
The Decode Goal remains paused; no commit, push or release was performed.

The owner-requested independent batch/microbatch follow-up is also complete:
batch 8192 with microbatches 512/1024/2048 reaches
4728.51/4798.45/4598.25 tok/s. An adjacent 1024/1024 control reaches
4812.04 tok/s. All four cohorts retain ten samples after three warmups;
logical batch enlargement provides no demonstrated gain at microbatch 1024.
[Follow-up evidence](evidence/gemma4_12b/prefill-batch-microbatch-2026-11-15.md).
No runtime kernel change, Decode Goal resumption or Git publication follows.

### Owner update: new Pro 12B campaign (2026-11-15)

After clearing the earlier Goal, the owner explicitly adopts
`GEM16_12B_CODEX_PLAN_2026-11-15.txt` as a new Goal. The bounded exception to
standing decision 7 now covers corrected post-attention fusion, historically
filtered Assistant GEMV, exact QKV load reconstruction, counter-filtered D2
Down weight pipelining and global D2 Target-attention preloading, in that
order. Existing numerical/cache/operator/E2E gates remain unchanged. This
supersedes the earlier pause only for the newly adopted five experiments;
prior negative results remain evidence, not authorization for a broader search.
Task: [execution contract](plans/gemma4-12b/PRO_CAMPAIGN_2026_11_15.md).
No earlier commit/push/release authorization is inherited.

The owner subsequently changes the new campaign's admission/confidence rule:
always perform a full Decode test for otherwise numerically/memory-valid
candidates, even if an isolated operator interval dips slightly below zero;
95% lower bounds may extend to -2% overall Decode loss. This supersedes the
previous positive-lower-bound admission/Decode rules for future experiments,
including reopening E1; exactness/runtime/profile requirements and measured
gain requirements remain. Original decisions/samples stay immutable.

E1 is qualified locally on2026-11-16 and retained: Ordinary+0.708245%,
secondary D2+0.06491% (effectively unchanged), all full output/proposal/KV bytes
exact, relevant12B/26B text/media/session and host/CUDA regressions pass.
The first rebuilt diagnostic candidate was stale because restored timestamps
prevented compilation; its protection results are invalidated and replaced by
a distinct, actually rebuilt candidate. Primary pinned-binary timings were
unaffected. Static shared2KiB and loaded1KiB are both disclosed with a conservative
2KiB budget. [Qualification](../artifacts/gemma4_12b/norm-fusion-qualified-2026-11-16.json).
E2–E5 and final cumulative/Prefill diagnosis remain open. No Git publication.

### Owner update: cumulative gain threshold for the current five experiments (2026-11-16)

The owner explicitly moves the0.5% minimum from each individual experiment
to the combined package of these five experiments. This supersedes the
per-experiment minimum/stop rule in the adopted Pro plan for future decisions,
including the still-running E2trial. Small positive measured primary effects
may remain provisional components after complete numerical/memory/native-state/
protected qualification and the existing Decode/Prefill margins pass. They
are not advertised as independently meeting the former0.5% gate.

Preserve the original protocols/results and report both the old single-trial
assessment and the new package-component assessment. The final combined
version must be measured directly against the original campaign parent;
Ordinary and D2 are assessed separately, never added across modes. The−2%
Decode confidence margin,−1% Prefill/inverse-TTFT margin, exactness/residency/
allocation rules and bounded candidate count remain. This does not reopen
older campaigns or authorize a parameter sweep or Git publication.

E2 is now correctness-qualified and retained provisionally over E1. Full
primary D2ABBA gains0.39353%; Ordinary−0.01550% is within its margin, as are
Prefill/inverseTTFT. Full1135IDs/1014proposals/wholeKV, actual native Assistant
sequence and32protected text/media/session cases plus host/CUDA tests pass.
The original individual0.5% gate remains recorded as failed; the owner's new
package rule admits this small positive component. E3–E5 and final combined
performance remain open. [Evidence](../artifacts/gemma4_12b/assistant-exact-tree-2026-11-16.json).

E3 is closed without retention: fullOrdinary+0.15193%, D2−3.00401%; both
D2confidence bounds violate the owner's−2% loss margin. Numeric/sanitizer/
operand/actual-dispatch checks pass. Only the joint T1/T3QKVcandidate is
restored, preserving E1+E2; no T1-only salvage/search. E4's current parent
Down counters admit the single predeclared weight/SFBpipeline hypothesis;
E4/E5 and final combined/Prefill closure remain open.
[QKV evidence](../artifacts/gemma4_12b/qkv-load64-full-2026-11-16.json).

E4 is closed without retention: fullD2−1.68796%, Ordinary−0.00333%.
The loss margins pass but the primary effect is negative, so the pipeline
is restored, preserving E1+E2. Numeric/memory/race and actual production
dispatch/instruction gates pass. E5's actual24case parent capture/counters
admit the single predeclared two8token KVpreload trial; no runtime candidate
has run yet. E5 and the final combined/Prefill diagnosis remain open.
[Down evidence](../artifacts/gemma4_12b/down-weight-pipeline-2026-11-16.json).

E5 is closed without retention after all80full requests: D2−0.01668%,
Ordinary−0.03421%. The isolated warm/evicted gain(+2.93%/+2.21%) does not
transfer to a positive full-model effect. Numeric/memory/sync/race checks
and actual native dispatch/instructions pass;48new Targetglobal calls over
six groups, same other kernel/grid/block counts,70versus64registers,
unchanged41,056B staticShared and zero local. Only E5attention source/header
is restored, preserving E1+E2. Acceptance-only full state/protected gates
are unrun after rejection, not claimed passed.
[Attention evidence](../artifacts/gemma4_12b/global-attention-preload-2026-11-16.json).
All five individual experiments are complete. The fresh direct original-parent/
cumulative-package Ordinary/D2 comparison is running; final state, bounded
4096/8192Prefill diagnosis and closure audit remain open. No broader tuning
or Git publication is authorized by these outcomes. The existing recurring
Prefill weight repack remains a disclosed runtime-rule conflict for a separate
task, not an exception to the permanent rule.

The owner explicitly stops this campaign on2026-11-16 at15:56Berlin.
All owned benchmark/coordinator processes are stopped; the Goal is paused.
Individual E1–E5evidence remains valid, but the interrupted cumulative comparison
has no accepted package result and the final Prefill diagnosis is unrun.
No automatic continuation or new benchmark is authorized until the owner resumes.
Continuation state is recorded at the end of the campaign task entry.

### Owner update: publish the two retained 12B Decode patches (2026-11-16)

After stopping and clearing the five-experiment Goal, the owner explicitly
requests committing and pushing the retained post-attention fusion and
Assistant exact-tree GEMV, their existing measurements and README to `main`.
This supersedes the campaign's earlier no-publication restriction for these
two patches only; publication does not wait for the interrupted final package
comparison. No further benchmarks are authorized. Existing per-trial results,
including the Assistant's former provisional assessment, remain immutable;
there is no claim of a qualified cumulative gain or a completed Prefill profile.
Unrelated local work stays outside this commit. This is no release request.
[Published scope and limitations](evidence/gemma4_12b/retained-decode-2026-11-16.md).

## Standing decisions

1. **Equal platforms.** Windows x64 and Linux x86-64 are equal product
   platforms. Releases provide both platform packages with applicable build,
   host, Studio and archive-integrity checks. Use proportional change-specific
   tests under the standing release publication policy above.
2. **Local single machine.** The initial deployment is one local user on one
   supported machine. Loopback is the default and the supported security
   boundary. Authentication/TLS and remote multi-user serving remain outside
   this baseline.
3. **Two equal public model profiles.** Gemma 4 12B Unified and Gemma 4 26B
   A4B Compact Vision are the two user-selectable public profiles and may be
   installed side by side. Neither is preferred. The 12B profile supports
   qualified text, image and audio. Compact Vision is the exact Trellis35 text
   Target plus FP8 Vision composite, accepts images within context capacity, is single-slot, and may
   use its separately pinned fixed-D2 Assistant. The qualified text-only 26B
   NVFP4 path remains implemented and regression-protected as an internal
   rollback profile, but is not offered in the normal Studio selection. This
   owner decision supersedes the earlier requirement that NVFP4 26B remain an
   equal public Studio choice. Studio stores model files in the standard shared
   Hugging Face Hub cache and must not create a second private model store. The
   12B entries continue to reference the existing upstream Hugging Face
   repositories; no GEM16 mirror or duplicate 12B weight repository is created.
4. **One active GUI.** `nativeStudio/` is the only active GUI. `studioApp/` is
   deprecated, receives no new product work, and remains temporarily as
   read-only migration evidence.
5. **Bounded OpenAI compatibility.** The product API is OpenAI Agent Core v1,
   not full OpenAI platform emulation. Supported fields, events, state, tools,
   exclusions, and qualification gates are versioned and fail visibly when a
   request exceeds the contract.
6. **One version source.** Repository, CMake builds, packaged server, native
   Studio, and release automation consume the root `VERSION` file. Release tags
   and manually supplied release versions must match it.
7. **Productization first; tuning frozen.** The NVFP4 and Trellis35 performance
   campaigns are closed after their accepted freezes; the 220/250 tok/s targets
   remain closed. The later bounded K/V exception below remains accepted.
   Earlier experimental-only Vision wording was superseded by decisions 8 and 10.
8. **Compact Vision production gates authorized.** The 2026-09-02 owner decision
   superseded the requirement to end Vision in an experimental V20 freeze and
   authorized bounded PRD00/PERF13/PERF12/FMT01/PUB01/QUAL01/APP01/REL01/P20/P21
   work. Decision 10 subsequently accepted bounded P20; the protected 12B and
   internal NVFP4 paths remain unchanged.
9. **Trellis35 device-image integrity is split between acquisition and
   runtime.** The owner's 2026-09-02 direction for FMT01 supersedes any future
   requirement to recompute the 12.2 GB `model.gem16` SHA-256 while loading the
   model. The offline packager fully verifies the immutable v1 inputs and the
   resulting v2 payload; download/install verification checks the published
   payload hash. Runtime startup hashes only the small self-describing metadata
   and lock files, validates the exact payload size, fixed arena layout, tensor
   table, layer regions, rate maps and uploaded expert descriptors, then
   uploads the payload directly. It neither hashes nor repacks model weights.
   Same-size payload corruption is therefore an acquisition/install failure,
   not a second full-file runtime check.
10. **Bounded P20 is accepted.** The
    owner's 2026-09-04 direction accepts Compact Vision as the second public
    product profile using the existing bounded V19 quality, capacity and
    lifecycle evidence. It expressly waives the larger QUAL01 text/Vision
    campaign for this P20 decision instead of claiming that campaign ran. It
    also supersedes P20's former dependency on REL01, two-platform packaging,
    clean-machine smokes and live Windows SM120 execution. The standing release
    publication policy now governs releases; promotion to `main` does not itself
    publish a release. Evidence is
    `artifacts/vision/p20-owner-acceptance-2026-09-04.json`.
11. **The consolidated 26B Hub layout is normalized in immutable revision
    `5a7a0225b3f23067e082c21312a4c38676cc237f`.** Existing published revisions
    were not rewritten. Payload names contain the full Gemma 4 26B A4B model
    and actual quantization/storage format: Trellis35 W4A8 for the compact text
    Target, FP8 E4M3FN for Vision, and hybrid NVFP4/FP8/BF16 for the Assistant
    and internal Target. `profiles.json` exposes Compact Vision and classifies
    NVFP4 as internal regression/rollback. Anonymous Hub metadata checks prove
    all four copied payloads retain their prior LFS identities and byte sizes.
    Current runtime locks remain on `6de2a057...` until the new remote-to-local
    filename mapping has passed acquisition and loader verification; changing
    the repository's default revision does not alter the pinned product.
    Evidence is `artifacts/vision/hf-layout-normalization-2026-09-04.json`.

### Owner update: multiple images and one Canvas conversation (2026-09-05)

The owner explicitly removes the one-image-per-conversation restriction. This
supersedes the single-image requirement in decision 3, AGENTS.md's profile wording,
the product/API contracts and the earlier Vision production scope. Conversations
may retain successive images and accept multiple images in a turn, bounded by
context, validated request sizes and storage capacity, with no fixed image-count
product limit. This is separate from the unchanged single execution slot.

Canvas checks append the actual screenshot to the same conversation as a tool
result. Images persist with non-temporary chats. No separate visual-review model
call or session eviction is permitted. New image spans reuse the existing Vision
workspace sequentially; historical image spans remain in the resident KV prefix.
Document revisions belong in tool results, not a changing system prompt. Precision,
per-image budgets, model locks and fixed-D2 semantics remain unchanged. Historical
single-image qualification evidence is not reclassified as multi-image evidence;
new multi-image evidence and platform limitations must be reported separately.

### Owner update: single-user C04 scope and client priority (2026-09-12)

The owner reconfirms one local user and normally one active conversation, including
12B. Existing qualified two-slot support and queue/cancellation safeguards remain;
this decision does not change runtime defaults, capacity or supported behavior.

This supersedes ROADMAP C04's future requirement for a broad parallel-media and
aggregate-concurrency RSS campaign in the initial single-user baseline. Keep the
existing input validation, pixel/byte limits and bounded allocation rules. A single
request can still contain oversized or multiple images, so retain a small sequential
multi-image/replay check with process-memory observation. Reuse accepted C03
interruption and recovery evidence rather than repeating its full matrix.

Bound the remaining C04 probe to eight generation requests total across the two
public profiles (1/2/4-image requests and one same-session continuation per profile),
one qualified decode mode per profile and ten minutes wall time. Stop at the cap
and report any unfinished case; these are test limits, not product image limits.
No new model downloads, parallel-load campaign or extended stress runs are needed.
Historical-image CPU caching is a deferred optimization, revisited only if the
bounded probe or practical client use exposes a material problem.

C05/C06 client session continuity, compaction, replay and tool behavior take priority
after this bounded check; deferred C04 optimization/stress work must not block them.
Malformed-input safety, equal-platform applicable checks and proportional
regressions remain required. Existing evidence is retained; publication follows
the standing release policy.

### Owner update: neutral Chat penalty compatibility (2026-09-30)

The owner authorizes Chat Completions to accept numeric zero values for
`frequency_penalty` and `presence_penalty` without changing checkpoint sampling.
This supersedes the blanket rejection of those two explicitly neutral fields
in the existing API subset; it does not authorize ignoring active penalties.
Non-zero values remain unsupported with a specific error, and invalid types or
values outside `[-2, 2]` remain invalid requests. Responses and all other
per-request sampling limitations are unchanged. Full native penalty/sampling
support and its resident RNG/fixed-D2 qualification remain future work.

### Owner update: native response penalties and graph changes (2026-09-30)

The owner's follow-up authorizes real `frequency_penalty` and `presence_penalty`
for Chat Completions, superseding the neutral-only restriction above for these
fields. Both default to zero and accept finite FP32-representable values in
`[-2, 2]`. They apply to emitted Target token counts within the current response,
including reasoning and control tokens, before temperature/top-k/top-p/min-p.
Prompt tokens, previous responses and rejected Assistant drafts are excluded.
The existing multiplicative repetition history remains unchanged.

Keep resident KV and RNG progress across parameter changes. Capture penalty
coefficients by value; use the existing sampling kernels and graph topology
when both are zero. Unchanged coefficients must not cause graph reconstruction.
The owner explicitly accepts reconstruction costs when coefficients change.
This bounded request-boundary graph executable replacement supersedes the
no-recurring-allocation requirement only for CUDA graph infrastructure during
that explicit reconfiguration; arenas and counters remain preallocated and
no token loop allocates, repacks or recompiles. Active penalties may add count
work; the zero path must be checked against the measured parent. Other sampling
parameters remain startup/session configuration, and Responses remains unchanged.
Windows evidence does not establish Linux qualification or a release claim.

### Owner update: native penalty production-main integration (2026-09-30)

The owner explicitly authorizes committing and pushing the native penalty change
directly to production `main` using the bounded Windows evidence. For this change,
this supersedes the requirement to finish the remaining Linux, internal NVFP4
model and everyday-context/media qualification before main integration. The owner
will boot the same machine into Linux, build from main and test there afterward.
Outstanding checks remain documented and are not reclassified as passed. This
decision does not authorize a new release tag, binary publication or a claim that
the general two-platform release gates have passed.

## Current model facts

### Owner update: bounded Trellis35 decode optimization (2026-10-22)

The owner authorizes the five follow-up experiments to the pinned decode audit:
allowed-top1 host selection, test-internal sequence-controlled replay and real
operator inputs, an isolated eight-warp M1 N128 candidate with separate T3
qualification, counter-gated KVH2 staging, and GateUp transform/FP8 quantization
fusion. This supersedes standing decision 7's tuning freeze only for these
bounded experiments; it does not reopen the closed 220/250 tok/s targets or
previously rejected generic prefetch/route-packing work. Keep numerical, memory,
sampling, cache and protected-profile contracts. Retain and commit only correct
production candidates with reproducible end-to-end benefit, preserving raw
positive and negative evidence. The owner authorizes commits, not a push or
release. The starting audit is
`docs/evidence/gemma4_26b/trellis35-decode-cost-audit-2026-10-22.md`.

### Owner update: bounded D2 follow-up (2026-10-23)

The owner authorizes continuing the D2 optimization from the completed five-step
follow-up. This extends the October 22 exception to standing decision 7 only for
refreshing full fixed-D2 cost/acceptance accounting and one independently qualified
T3 projection candidate sharing two N8 fragments across eight warps. GateUp and
Down are screened separately using actual inputs and expert unions; no automatic
transfer from the accepted M1 kernel is authorized as a performance claim. Keep
all precision, accumulation, cache, sampling, graph/ring safety and protected
profile gates. Only correct reproducible end-to-end gains may remain in production
and be committed under the owner's existing instruction; no push or release.
Rejected staging/fusion, generic prefetch and route-packing stay closed without
new matching evidence. The 220/250 tok/s targets remain closed.

### Owner update: T3 GateUp countercase follow-up (2026-10-23)

The owner authorizes continuing from accepted T3 Down `6895ed2` with the
GateUp countercase investigation. This extends standing decision 7's bounded
exception only to actual-input GateUp counter/latency analysis and one candidate
justified by that comparison. Keep the accepted Down/M1 and graph/ring fixes,
precision, K accumulation, cache, sampling and protected profiles. Reopening
any rejected mechanism requires new matching evidence and an explicit scoped
record, not the historical task queue. The counter comparison opens only a paired-fragment codec word-window candidate:
share circular word gathers across both N8 fragments to reduce warp shuffles,
without changing decoded states or accumulation. It does not open generic
prefetch, route packing, quantization fusion or KV staging. Retain/commit only
numerically correct reproducible end-to-end benefit under the existing owner instruction. No push,
release, or reopening of the closed 220/250 tok/s targets.

### Owner update: bounded Down codec follow-up (2026-10-23)

After accepted GateUp `629bbac`, the owner asks to continue toward the pinned
llama reference. This extends standing decision 7's bounded exception only to
actual-input counter/operator investigation of paired codec windows for T3 Down
and, if justified, one separately qualified production candidate. It supersedes
the GateUp-only boundary above for this Down experiment, preserving GateUp/M1,
precision, accumulation, context, sampling, cache, graph/ring and protected gates.
Retain/commit only correct reproducible end-to-end gains; no push or release.
Generic prefetch/packing/fusion/staging and the closed 220/250 targets stay closed.

The fresh actual-production layer0 Down countercase reports long-scoreboard
0.9556 versus short-scoreboard0.1742 per active issue, with a four-block register
occupancy limit. This opens one diagnostic refinement of the Down candidate:
a fixed four-iteration register pipeline for the same resident payload words,
using the existing M1 pipeline pattern while preserving the T3 K order. This
supersedes the generic-prefetch closure only for that counter-backed Down test;
it does not authorize wider prefetch, layout or runtime changes. No production
promotion without separate numerical, memory and end-to-end evidence.

### Owner update: five measured decode follow-ups (2026-10-24)

The owner explicitly sets a Goal to carry out the five proposed steps after
`26c6840`: M1 GateUp, T3 GateUp, local/global fixed-D2 attention, QKV/O projections
for one/three rows, and Target/Assistant output heads. This supersedes standing
decision 7's tuning freeze and the earlier GateUp/Down-only scope solely for
these bounded investigations and counter-justified candidates. It does not
reopen the old 220/250 targets or authorize generic framework changes. Prior
negative evidence remains binding context; a rejected mechanism needs new
matching dispatch/counter evidence and a narrow written rationale before retry.
Preserve all precision, accumulation, format, sampling, context/cache, graph/ring,
resident-weight, memory and protected-profile boundaries. Reassess costs after
each accepted gain; keep/commit only correct reproducible end-to-end benefit.
Retain positive and negative raw evidence and update README/results. No push or
release. A fresh current-llama same-machine comparison is required before a
new superiority claim. Work journal: `plans/gemma4-26b/DECODE_FIVE_STEPS_2026_10_24.md`.

The October24 Step2 fresh actual T3 GateUp captures show long-scoreboard ratios
0.8167/0.9589 versus short-scoreboard0.1809/0.1993 per active issue for late29
and early5, with47registers. This opens only a fixed resident-payload register
pipeline for T3 GateUp, bounded depths2 and4, preserving exact K order/codec/MMA.
It supersedes the generic-prefetch closure solely for this matching counter-backed
experiment; no wider prefetch, routing, packing, staging or fusion is opened.

Step2's union24 countercase shows register occupancy limit5->4 with47->63
registers, despite lower long-scoreboard ratio1.1063->0.5881. Both depth2/4
screens regress and remain diagnostic. This supports one final bounded depth2
variant with `__launch_bounds__(256,5)` to test register-pressure compensation.
Keep the original countercases, require bit parity and the same all-case gate,
and stop this pipeline experiment if it still fails; no production promotion
based on its median operator gain alone.

| Profile | Public selection | Input | Context policy / retained capacity | Slots |
|---|---|---|---|---|
| 12B Unified | Yes | Text, image, audio | Subject to configured capacity | Up to two |
| 26B Compact Vision | Yes | Text, images, 70/140/280 soft tokens per image | Everyday: Linux 220,000 / Windows 170,000 (ordinary and D2) | One |
| 26B NVFP4 | Internal only | Text | 98,304 / 86,016 | One |

Compact Vision requires the exact locked Trellis35 Target + FP8 Vision and,
for D2, the separately pinned Assistant. It reports `production_qualified` and
`experimental=false` following bounded P20. This does not assert the waived
extended QUAL01 or deferred release tests ran.

All current 26B locks use `danmoreng/gemma-4-26B-A4B-it-GEM16` at
`6de2a057f11332420819f8e6efd08e42d7a03bc7`: internal NVFP4 at the root,
Compact Target under `trellis35/`, Vision under `vision/`, Assistant under `assistant/`.
Use the current device images for runtime regression; do not reconstruct the
old M08 shard layout. Locks retain exact hashes and source identities.

Accepted evidence: [V19](../artifacts/vision/v19-acceptance.json),
[P20](../artifacts/vision/p20-owner-acceptance-2026-09-04.json),
[PUB01](../artifacts/trellis35/pub01-consolidated-publication.json),
[normalized Hub layout](../artifacts/vision/hf-layout-normalization-2026-09-04.json).
Prior numerical and performance records remain under `artifacts/` and `benchmarks/`.

### Bounded 26B K/V epilogue optimization (owner direction 2026-09-04)

The owner authorizes a bounded ordinary-decode 26B NVFP4/Trellis35 K/V
epilogue fusion, with
exact Q/output/cache-byte tests and parent/candidate performance measurements.
This supersedes decision 7's prohibition on further NVFP4 tuning only for this
experiment (including the frozen Trellis35 tuning restriction). The 220/250
token/s targets remain closed. Prefill, T3 speculative cache visibility/rollback
and 12B dispatch remain unchanged. Per follow-up owner direction, initial
performance checks use one warmup and three measured runs per variant, beginning
with ordinary decode. A further owner direction authorizes and, after the
positive exact-output screen, accepts the corresponding fixed-D2/T3 staging
fusion for NVFP4 and Trellis35. The speculative cache append, backup, rollback
and per-row visibility remain separate and unchanged.
The owner's follow-up accepts the positive small-sample result and integrates
the exact-output ordinary fusion for both formats. It changes no product
capability, checkpoint or quality claim; wider performance publication still
requires the permanent benchmark contract.


### Owner update: 26B long-context admission reserve (2026-09-05)

Use 200 MiB (209,715,200 bytes) of free device memory for 26B long-context
startup admission at 65,536 tokens and above, including the preliminary engine
slot check. This supersedes the previous 400 MiB long-context requirement;
the existing 200 MiB final fixed-D2 check remains consistent with it. The 700 MiB
short-context and 12B policies are unchanged. Context capacity must still be
explicit and pass actual admission with Vision, Assistant and the desktop active.
No weight offload, precision change or automatic context fallback is authorized.

## Release publication

The standing owner policy above governs current and future releases. Former
0.2.x publication exceptions and superseded full-campaign requirements are
preserved only in the linked historical decision snapshot. They impose no
future release tasks or publication restrictions.

### Owner update: server-first hardening and everyday context (2026-09-05)

Server/CLI and unmodified coding-agent operation are the primary entry and work
priority; native Studio is optional. This supersedes the desktop-first entry
requirement, not the equal public status of 12B Unified and 26B Compact Vision.
Compact Vision (Trellis35 text plus FP8 Vision) is the public 26B agent target;
NVFP4 remains internal regression/rollback, without public promotion.

The public Compact Vision everyday context recommendation and new Studio profile
default are 220,000 tokens on Linux and 170,000 on Windows, with or without D2.
These owner-reported stable operating values supersede Linux's 229,120 default;
they are not new independent GPU qualification. Saved explicit settings remain
explicit. Historical capacity ceilings and evidence remain unchanged. Admission
still requires the existing 200 MiB long-context reserve; no automatic fallback.

Implement practical OpenAI interoperability where semantics can be preserved;
only substantial unsupported behavior remains an explicit limitation. The
server-first backlog in ROADMAP.md tracks review C00–C09. The standing release
publication policy supersedes the former full release campaign; the existing
QUAL01 waiver remains unchanged.

### Bounded Step4 FP8 projection experiments (2026-10-24)

Within the owner's five-step Goal, fresh actual 85fc453 captures show M1
long-scoreboard per-issue20.65 (local QKV),22.76 (global Q/K),12.53 (global O),
versus short0.206/0.266/0.075; T3 math-pipe throttle7.04/7.89 (local/global QKV)
and4.98 (global O), with three same-weight M16 MMAs per K iteration. Raw:
`benchmarks/results/2026-10-24/85fc453/fp8-projections/`.
This permits only (a) native M16 A-row packing for the three actual FP8 rows,
following the existing matrix fragment map, and (b) one fixed two-entry register
load pipeline for the actual M1 QKV/O FP8 fragments. The latter supersedes the
prefetch closure solely for these counter-backed projection shapes; no generic
staging, source-weight repack, K reordering, routing or precision change opens.
Use production-matching RDC builds, all-case exact/output/resource screens and
full numerical, memory, protected and end-to-end acceptance before promotion.
Global layers project Q/K only and reuse raw K for V; retain that behavior.

Step4's exact-input packed-QKV counter shows405,504 shared-load bank conflicts
versus0 in the original; both execute608,256 shared-load wavefronts. This permits
one32-byte padding trial between the three *ephemeral shared activation* rows,
adding96bytes/CTA without changing any persistent tensor/weight layout. Require
bit parity and a direct packed-versus-padded all-case gain before choosing it.
Other staging/prefetch mechanisms remain closed.

Step4 closes the M1 two-entry trial: every sampled case loses, despite bit
parity; it stays test-only. The selected R3 packing/padding passes all numerical,
memory/protected and actual-dispatch gates and yields a small +0.42–0.58% D2
same-policy AB/BA gain, with differing Dynamic Boost clocks disclosed. No wider
prefetch/staging scope is carried forward by that acceptance.

### Bounded Step5 output-head experiments (2026-10-24)

Fresh f6de38e actual-head windows execute Assistant proposals, not forced drafts.
Target R3 NVFP4 head has106registers (112allocated), register block limit4,
long-score17.72 and three same-weight M16 MMAs perK64; its DRAM utilization is
91.55%. This opens one native M16 packing of the three actual A rows, using the
existing matrix SF_A lane map, removing two accumulator sets/MMA instructions.
Target M1 is already at93.63% DRAM and is left unchanged in this step.
Assistant R1 head has56registers, long-score47.05/42.38 and95.52/95.59% DRAM.
This permits one adjacent-two-N8-tiles-per-warp experiment sharing A/SF_A loads,
with two ordered K64 chains. Keep the original64-token candidate partitions,
all4096 candidate values/tokens, finite/BF16/divisor/mask/tie semantics exact;
no weight repack, format change, speculative-depth change or new prefetch opens.
High DRAM utilization bounds expectations; long-score alone is not proof of
avoidable latency. First require exact diverse native-scale boundary tests and
all actual outputs, positive operator medians in every sampled role/position and
positive aggregate bootstrap lower bound before a full-model trial. Primary
promotion still requires exact numerical/memory/protected/dispatch evidence,
fresh AB/BA end-to-end confidence and ordinary non-regression. Failed variants
stay offline. Raw: benchmarks/results/2026-10-24/f6de38e/output-heads/.


Step5 closes both head variants. Assistant pairing loses all sampled operator
cases and remains offline. The Target trial passes numerical/memory/protected/
dispatch gates, but D2 AB+0.1647% and BA+0.1728% do not pass the predeclared
both95lower>0 criterion (BA lower-0.0018%); slightly higher clocks are disclosed.
The production source is restored exactly, loaded binary bytes match the
accepted parent, and only non-loadable nvcc temporary-PID metadata differs.
All raw data and the trial are retained in the negative output-head record.
No new head prefetch/staging/pairing/packing search is opened by this result.

The final five-step audit permits one build-mode verification of the existing
Step2 depth2/depth4/register-bounded-depth2 variants, with production-matching
compute_120a/sm_120a and RDC=true. The earlier non-RDC record remains valid for
its stated diagnostic build but cannot establish the matched-build outcome.
This is only a re-screen of those exact three existing sources and current
actual-input/operator counters, not a new pipeline/depth/staging search. Keep
the original all-real-GateUp-case>2% gate and exact codec/output boundaries;
only a qualifying variant may enter the existing full production acceptance
loop. Omit the unrelated old attention link derivative with a visible
cudaErrorNotSupported diagnostic link stub, never executed in T3 projection mode.


The build-mode audit is closed: all three current-source RDC variants remain
exact but fail the unchanged all-case>2% gate. No production source change or
new pipeline/staging search follows. Matched-build record and disassembly
supplement preserve the result without rewriting the earlier non-RDC evidence.

### Five-step decode campaign closeout (2026-10-24)

All five bounded experiments are complete. Production retains M1 GateUp codec-window
sharing, fixed-D2 attention normalization and native three-row FP8 QKV/O packing;
the T3 GateUp pipelines and Target/Assistant head trials stay rejected diagnostics.
The accepted runtime source remains `f6de38e`. Qualification and restoration
evidence is checked against actual logs, outputs, source/binary hashes and retained
positive and negative records, including the production-RDC re-screen.

The final twelve-cohort day-start/final/current-llama comparison uses freshly built
official llama.cpp b11866 at `3f03b17724ad8b69040825f45c43c59b092a0244`.
Native cumulative gains are 1.22–1.93% ordinary and 1.72–1.79% fixed-D2 under the
same Dynamic Boost policy. Both point estimates favor llama across engines;
the reverse D2 gap interval includes zero. Formats, KV and generated streams
differ, so this is a native-format/kernel baseline with no quality-equivalence or
native-superiority claim. See
[final evidence](evidence/gemma4_26b/decode-five-steps-final-2026-10-24.md) and
the [frozen record](../artifacts/trellis35/decode-five-steps-final-2026-10-24.json).
This closes the bounded exception; no broader tuning, push, Windows/media
qualification or release is authorized by the closeout.

The owner's subsequent October24 instruction explicitly authorizes pushing the
current tested code and documentation to GitHub `main`. This supersedes the
five-step Goal's no-push restriction for this completed campaign only; it does
not authorize a new release tag, downloadable release packages or wider tuning.
