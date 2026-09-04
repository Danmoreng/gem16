# Gemma 4 26B A4B Compact Vision production contract

**Status:** owner-bounded P20 accepted 2026-09-04; REL01 and P21 deferred

**Owner direction:** 2026-09-02

**Source baseline:** `26691d607f7945234a39aed8a98f0d1ed1d904c1`

## Scope

Compact Vision is the explicit Gemma 4 26B A4B profile composed of the locked
Trellis35 text Target, the locked FP8 Vision module, and an optional separately
locked fixed-D2 Assistant. It is not an extension of the NVFP4 profile and file
presence never enables it.

The candidate boundary is:

- NVIDIA Blackwell SM120 in the measured approximately 16 GB class;
- Windows x64 and Linux x86-64 as equal release platforms;
- batch one and one resident execution slot;
- one image and text output;
- 70, 140, or 280 Vision soft tokens;
- Ordinary or fixed-D2 decode;
- no audio or video;
- only the context limits accepted by final post-PERF12/PERF13 capacity gates.

## Stable identity and components

The profile ID is always:

```text
gemma4-26b-a4b-trellis35-vision-fp8
```

Execution reports `decode_mode=ordinary|fixed-d2` separately. Fixed-D2 is
supported only when the exact validated Target, Vision, and Assistant locks are
loaded. Qualification is bound to their immutable hashes; components and
profiles are never inferred from filenames or substituted.

The accepted runtime reports:

```text
qualification_state=production_qualified
experimental=false
```

This status applies only to the exact locked composite. The owner's 2026-09-04
decision replaces the extended QUAL01 requirement with bounded V19 evidence
and separates P20 profile acceptance from REL01, packaging, clean-machine and
live Windows-SM120 release gates. Deferred gates remain open and may not be
reported as completed.

## Precision and runtime integrity

- Trellis35 text weights retain their published W4A8/BF16/FP8 contract.
- Vision linear weights retain E4M3FN plus BF16 scales; copied position, norm,
  and control tensors retain BF16.
- Vision Attention preserves the accepted BF16/FP32 boundaries unless a
  separately gated numerical candidate is accepted.
- No runtime JIT, quantization, repack, CPU weight offload, expert streaming,
  recurring token/image-loop allocation, or silent precision/kernel/budget/
  context/decode fallback is permitted.
- After FMT01, the Trellis35 product payload is one GPU-ready device image with
  one device allocation and no second persistent representation.
- The offline packager and download/install path verify the complete payload
  hash. Runtime startup does not hash or repack the 12.2 GB payload; it verifies
  the small metadata/lock identity, exact file extent, fixed layout and uploaded
  descriptors before execution, as fixed by active decision 9.

## Image and cache boundary

Encoded media is untrusted. The source identity used for resident conversation
reuse is SHA-256 plus encoded byte length. Inputs without an authoritative
encoded-source digest require a complete structural image comparison. The
profile accepts exactly one supported image and rejects malformed, excessive,
or mismatched input before execution.

`vision_max_soft_token_budget` is a fixed startup capacity.
`last_vision_soft_token_budget` is request telemetry. A request may never
exceed the configured maximum, and changing that maximum requires controlled
server restart and capacity replanning.

## Quality claim

The eventual claim is limited to the frozen production suite: one-image
description, OCR, documents/tables, charts, spatial relations, counting,
colors, small details, representative natural images and the tested geometry
and budget matrix. It does not assert general multimodal benchmark parity.

The underlying Trellis35 text Target must independently pass the frozen text,
retrieval, tool, stop, repetition, sampling, Ordinary and fixed-D2 gates before
the Vision profile can become productive.

## Release gates

The original full P20 gate list was:

1. PRD00 security, fixtures, identity and stable reporting;
2. accepted PERF13/PERF12 performance and numerical candidates;
3. byte-exact FMT01 Trellis35 device image v2 and explicit legacy behavior;
4. a verified immutable revision in the existing consolidated repository;
5. final Trellis35 text and expanded Vision quality;
6. fresh Target+Vision and Target+Vision+Assistant capacity at every maximum
   Vision budget, including product default, advanced maximum and first reject;
7. live SM120 Vision+D2 execution on Windows and Linux;
8. Studio, server, OpenAI Agent Core, cancellation and lifecycle qualification;
9. full download/resume/hash/corruption and clean-machine package smokes;
10. protected 12B and qualified 26B NVFP4 non-regression.

Items 5, 7 and 9 remain release work under the owner-bounded acceptance, as do
REL01 and the wider Agent Core qualification. The preserved V19 limitations
remain disclosed. P21 later freezes the source revision, binaries, repository revision, component
and catalog hashes, defaults and maxima, performance and quality panels, known
limitations, and rollback procedure. Historical evidence is retained rather
than rewritten.
