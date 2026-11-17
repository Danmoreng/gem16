# Active decisions

**Updated:** 2026-11-17 · **Track:** productization · **Status:** stable v0.3.0 publication authorized

Permanent rules in [AGENTS.md](../AGENTS.md) remain binding. Read the
[product contract](PRODUCT_CONTRACT.md) and the narrow task contract next.
The [pre-consolidation record](archive/ACTIVE_DECISIONS_2026-09-04_BEFORE_CONSOLIDATION.md)
preserves the complete decision history, including superseded intermediate stages.
The existing numbered decisions retain their identities below.

### Minimal evidence for future work (2026-11-17)

The owner adopts the [testing and evidence rules](../AGENTS.md#testing-and-evidence) for all new work.
Routine checks are reported in the PR or final response. Development runs stay local. Published performance
claims and accepted kernel or model qualifications share one compact record per result across all citations.
This supersedes older instructions to archive every attempt, transcript or export. The checks required to
support a claim still apply.

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
retained/rejected dispositions. The [original decision history](https://github.com/Danmoreng/gem16/blob/9f7f7a6555ac169a9e49580f11824eef0e55fd85/docs/ACTIVE_DECISIONS.md)
records the temporary exceptions to the tuning freeze. Reopening a rejected mechanism requires new
matching dispatch or counter evidence and a narrow rationale. The original failed N1 gate is separate from
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
