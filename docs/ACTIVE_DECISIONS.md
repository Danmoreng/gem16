# Active decisions

**Updated:** 2026-09-01
**Track:** Productization baseline
**Status:** owner-approved product contract; implementation and release gates in progress

This file is the short operational policy for current work. Permanent safety,
security, evidence, and runtime-integrity rules in `AGENTS.md` remain binding.
The detailed product scope is [`PRODUCT_CONTRACT.md`](PRODUCT_CONTRACT.md), and
the API subset is [`OPENAI_AGENT_CORE_V1.md`](OPENAI_AGENT_CORE_V1.md).

The prior Gemma 4 26B fast-track decision history is preserved unchanged at
[`archive/ACTIVE_DECISIONS_2026-08-28_GEMMA4_26B_FAST_TRACK.md`](archive/ACTIVE_DECISIONS_2026-08-28_GEMMA4_26B_FAST_TRACK.md).
Read it only for a concrete historical or evidence question.

## Active product decisions

1. **Equal platforms.** Windows x64 and Linux x86-64 are equal product
   platforms. A product release must pass the same applicable source, host,
   GPU, API, Studio, packaging, and clean-machine gates on both.
2. **Local single machine.** The initial deployment is one local user on one
   supported machine. Loopback is the default and the supported security
   boundary. Authentication/TLS and remote multi-user serving remain outside
   this baseline.
3. **Equal model profiles.** Gemma 4 12B Unified and Gemma 4 26B A4B are equal,
   user-selectable product profiles and may be installed side by side. Neither
   is the preferred product choice. Capability differences remain explicit:
   12B is the qualified multimodal profile; 26B is qualified text-only,
   single-slot, and optionally uses its separately pinned fixed-D2 Assistant.
   Studio stores model files in the standard shared Hugging Face Hub cache and
   must not create a second private model store. Each profile is represented by
   an independently pinned Target and Assistant catalog entry. The 12B entries
   continue to reference the existing upstream Hugging Face repositories; no
   GEM16 mirror or duplicate 12B weight repository is created.
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
7. **Productization remains primary; experimental Trellis35 tuning is frozen
   after PFX31 and the Trellis35-only Vision vertical slice is active.** The owner's 2026-08-31 direction closes the bounded 26B
   Trellis35/W4A8 performance pass reopened on 2026-08-30. PFX28-D is the only
   retained closeout candidate; further Trellis micro-optimization is stopped.
   The owner subsequently approved the separate 26B Vision vertical slice on a
   branch based directly on frozen PFX31 commit `7649a84`. Vision v1 is bound
   exclusively to the compact `gem16-trellis35-w4a8-v1` text artifact and its
   independently locked FP8 Vision module compiled from Google's pinned
   unquantized BF16 QAT checkpoint. It is not a Vision extension for `main`,
   NVFP4, or a generic 26B profile, and file existence never enables the
   capability. The qualified 26B
   NVFP4 checkpoint and evidence remain frozen, its 220/250 token/s targets
   remain closed, and the protected 12B path, product claims, release
   sequencing, and productization priority remain unchanged. Trellis35 stays
   experimental and may not modify or silently fall back to a qualified path.

## Current qualified model facts

- 12B remains regression-protected and qualified for text, image, and audio on
  the existing SM120 product path.
- 26B is a qualified selectable product checkpoint for SM120, batch one, one
  resident slot, fixed-D2 MTP, and bounded published quality claims.
- 26B fixed-D2 MTP supports up to 86,016 context tokens with the accepted
  200 MiB reserve. Target-only execution supports up to 98,304 with its
  separate reserve contract.
- **Consolidated 26B publication.** The owner's 2026-09-01 decision supersedes
  the earlier requirement for separate 26B Target and Assistant repositories.
  The canonical public repository is now
  `danmoreng/gemma-4-26B-A4B-it-GEM16` at immutable revision
  `31842e12882d09bab7109c0ad52a4ee2e945069c`: qualified NVFP4 remains in the
  root, with Trellis35 under `trellis35/`, fixed-D2 Assistant under
  `assistant/`, and FP8 Vision under `vision/`. They remain independently
  locked components and may never be silently substituted. The historical
  Assistant repository remains immutable only for old locks; new catalog
  entries use the consolidated repository.
- **Current 26B NVFP4 regression artifact.** Runtime and non-regression work
  uses the published `danmoreng/gemma-4-26B-A4B-it-GEM16` Target at revision
  `31842e12882d09bab7109c0ad52a4ee2e945069c`, format
  `sm120-device-image-v1`, `model.gem16` SHA-256
  `1ed73cf105b68db937ac0992283d31fdb2225474204341440721f41fe871bb72`,
  and artifact-content identity
  `471805f7dad8abb84300be78b2822a63dcb1d35bff5aa98426a162cc8532ee17`.
  The original 16-shard M08 working artifact and its former local output
  directory are deprecated runtime-test inputs and may be absent. Preserve
  M08 records as historical compiler/provenance evidence, but do not search
  for, reconstruct, or recompile that old layout for routine NVFP4 regression
  testing. Test the current device image with the current engine instead.
- The final retained sampled-D2 characterization reaches 203.842 token/s
  median; the prior 220 and 250 token/s targets remain unmet and closed for the
  frozen decode phase. This does not block the accepted product checkpoint.
- Within the separately experimental Trellis35 Vision slice, V14 qualifies
  Vision plus fixed-D2 only for the exact validated Target+Vision+Assistant
  composite. That exact composite reports `vision_mtp_supported=true`; all
  other combinations continue to fail closed. This narrow enablement does not
  promote Trellis35 or 26B Vision to the released product profile. V15 has closed
  the experimental runtime/server profile, validation, error-code, timing,
  metrics, and cancellation contract; its accepted evidence is
  `artifacts/vision/v15-runtime-server-closure.json`. V16 has closed immutable
  consolidated publication, per-component locks, generated catalog entries,
  anonymous resume/hash verification, and collision-free hardlink views; its
  evidence is `artifacts/vision/v16-consolidated-publication.json`.
  V17 has added the explicit third Native Studio profile, generalized bounded
  component model, persisted Vision settings, exact `--vision-model` launch,
  independent Target/Vision/Assistant state, deduplicated capacity preflight,
  and shared-blob-safe removal. Its evidence is
  `artifacts/vision/v17-native-studio-profile.json`. V18 has closed the native
  chat-UX gate: the explicit Vision profile accepts one image, rejects audio
  and a second image locally, sends the selected 70/140/280 processing budget,
  exposes preview/estimate/remove/retry affordances, gates fixed-D2 on live
  `vision_mtp_supported`, and rejects mismatched external-server profiles. Its
  generated catalog also corrects the V16 Vision runtime view to the four
  strict module files while retaining the canonical publication and its legal
  files; old auxiliary view hardlinks are pruned without deleting Hub blobs.
  Its evidence is `artifacts/vision/v18-native-studio-image-d2.json`. The profile
  remains experimental pending V19 capacity, quality, cross-platform, and
  lifecycle qualification.

Accepted numerical, performance, context, product, and publication evidence
continues to live in `artifacts/`, `benchmarks/`, and the archived fast-track
record. This summary does not replace that evidence.

## Current product gaps

- Native Studio now exposes three explicit profiles and its generated catalog
  contains the pinned 12B Target,
  12B Assistant, consolidated 26B NVFP4 Target and Assistant, and the new
  Trellis35 Target and FP8 Vision components. A fresh installation presents
  a neutral profile choice, checks available storage, and can install either
  or both profiles. Payloads remain in their source repositories' canonical
  Hub blobs and snapshots. Because the 12B Target lock includes one file from
  the separate Google repository, Studio creates a deterministic
  `.gem16/snapshots` runtime view made only from hardlinks to verified Hub
  blobs; it is not a mirror or second blob store. Full large-download and
  clean-machine qualification on both product platforms remains a release
  gate.
- The current OpenAI SDK validators establish a narrow development gate; full
  Agent Core v1 still needs equal Windows/Linux and 12B/26B qualification,
  TypeScript SDK coverage, and an external coding-agent workflow.
- The native Studio has not yet shipped in a release produced after its
  migration from Compose.
- Windows and Linux packaging are not yet equal. Linux runtime dependencies and
  clean-machine installation remain unresolved, and both packages require
  complete notices/manifests and release smoke evidence.
- Version reporting is being centralized in the 0.2.x development line; no
  0.2 product release is implied by the source version alone.

## Permanent product constraints

- No productization change may weaken qualified 12B or 26B model behavior,
  precision, cache semantics, resident-weight ownership, context limits,
  fallback reporting, or allocation rules.
- Model locks, model files, requests, media, tool schemas, tool arguments, and
  tool results remain untrusted inputs.
- Studio and server must disclose the selected model profile and actual
  capabilities. Equal product status never permits capability substitution.
- Historical evidence is preserved rather than rewritten. Compact active docs
  point to archived decisions and accepted immutable records.
