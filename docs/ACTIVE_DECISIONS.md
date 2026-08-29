# Active decisions

**Updated:** 2026-08-29
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
7. **Product work before new decode tuning.** The 26B decode optimization phase
   remains frozen. Current priority is documentation consistency, two-profile
   native onboarding, Agent Core qualification, and equal Windows/Linux
   packaging.

## Current qualified model facts

- 12B remains regression-protected and qualified for text, image, and audio on
  the existing SM120 product path.
- 26B is a qualified selectable product checkpoint for SM120, batch one, one
  resident slot, fixed-D2 MTP, and bounded published quality claims.
- 26B fixed-D2 MTP supports up to 86,016 context tokens with the accepted
  200 MiB reserve. Target-only execution supports up to 98,304 with its
  separate reserve contract.
- The qualified 26B Target and Assistant are separate immutable model
  repositories. Studio must download and verify both when MTP is selected and
  may never silently substitute either component.
- The final retained sampled-D2 characterization reaches 203.842 token/s
  median; the prior 220 and 250 token/s targets remain unmet and closed for the
  frozen decode phase. This does not block the accepted product checkpoint.

Accepted numerical, performance, context, product, and publication evidence
continues to live in `artifacts/`, `benchmarks/`, and the archived fast-track
record. This summary does not replace that evidence.

## Current product gaps

- Native Studio now consumes one generated catalog for the pinned 12B Target,
  12B Assistant, 26B Target, and 26B Assistant. A fresh installation presents
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
