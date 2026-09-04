# Active decisions

**Updated:** 2026-09-04 · **Track:** productization · **Status:** bounded P20 accepted; release gates open

Permanent rules in [AGENTS.md](../AGENTS.md) remain binding. Read the
[product contract](PRODUCT_CONTRACT.md) and the narrow task contract next.
The [pre-consolidation record](archive/ACTIVE_DECISIONS_2026-09-04_BEFORE_CONSOLIDATION.md)
preserves the complete decision history, including superseded intermediate stages.
The existing numbered decisions retain their identities below.

## Standing decisions

1. **Equal platforms.** Windows x64 and Linux x86-64 are equal product
   platforms. A product release must pass the same applicable source, host,
   GPU, API, Studio, packaging, and clean-machine gates on both.
2. **Local single machine.** The initial deployment is one local user on one
   supported machine. Loopback is the default and the supported security
   boundary. Authentication/TLS and remote multi-user serving remain outside
   this baseline.
3. **Two equal public model profiles.** Gemma 4 12B Unified and Gemma 4 26B
   A4B Compact Vision are the two user-selectable public profiles and may be
   installed side by side. Neither is preferred. The 12B profile supports
   qualified text, image and audio. Compact Vision is the exact Trellis35 text
   Target plus FP8 Vision composite, accepts one image, is single-slot, and may
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
10. **Bounded P20 is accepted; release qualification remains separate.** The
    owner's 2026-09-04 direction accepts Compact Vision as the second public
    product profile using the existing bounded V19 quality, capacity and
    lifecycle evidence. It expressly waives the larger QUAL01 text/Vision
    campaign for this P20 decision instead of claiming that campaign ran. It
    also supersedes P20's former dependency on REL01, two-platform packaging,
    clean-machine smokes and live Windows SM120 execution. Those items remain
    mandatory before a release claim, together with P21; promotion to `main`
    does not itself imply a release. Evidence is
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

## Current model facts

| Profile | Public selection | Input | Context Target / fixed-D2 | Slots |
|---|---|---|---|---|
| 12B Unified | Yes | Text, image, audio | Subject to configured capacity | Up to two |
| 26B Compact Vision | Yes | Text, one image, 70/140/280 soft tokens | 229,376 / 229,120 | One |
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


## Open product gates

- Full Agent Core v1 qualification requires official Python and TypeScript SDKs,
  an external coding-agent workflow, both public profiles and both platforms.
- REL01/P21, live Windows SM120 Compact Vision, equal packages, full downloads
  and clean-machine onboarding remain release gates. Native Studio has not
  shipped as a fully qualified two-platform release.
- `VERSION` is the shared version source; the 0.2.x development line alone is no release claim.
- The [roadmap](ROADMAP.md) routes remaining work. Historical evidence is preserved;
  no cleanup changes model precision, behavior, context, allocation or fallback rules.
