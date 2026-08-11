# Gemma 4 26B M00 policy review

Date: 2026-08-06

Branch: `feat/26b-m00-policy`

Parent: `b078a153772d12711032ef1b940221e825ce36e5`

Status: passed; owner accepted 2026-08-06

## Scope review

- [x] M00 contains governance, track documentation and evidence only.
- [x] No source lock or model payload was downloaded.
- [x] No checkpoint compiler, quantizer, loader, runtime or CUDA implementation was added.
- [x] Existing Gemma 4 12B direct-load locks and product defaults are unchanged.
- [x] No incomplete 26B capability is exposed. A default-off experimental boundary is deferred until code exists.

## Artifact contract review

- [x] Source, compiler and outputs require immutable locks and exact hashes.
- [x] Compilation is offline, deterministic, bounded-memory, auditable and never performed by inference/server
  startup.
- [x] The derived artifact remains Safetensors plus versioned per-tensor provenance metadata.
- [x] Every production mathematical tensor derives from one exact QAT BF16 source.
- [x] The artifact is labeled project-built and may not be called official Google NVFP4.
- [x] Runtime quantization, silent requantization and undisclosed persistent conversion are forbidden.
- [x] One resident device representation and physical tied-head aliasing are required.
- [x] Hash mismatch or missing native support fails visibly before inference.
- [x] Artifact distribution remains blocked pending M01 terms/hosting review.

## Product and benchmark review

- [x] Initial profile is text-only.
- [x] Vision, audio, video, MTP, continuous batching, multiple 26B slots and multi-GPU are excluded.
- [x] CPU weight offload and expert streaming are diagnostic only and cannot support a primary claim.
- [x] Direct Unsloth NVFP4, same-compiler ordinary BF16 and official Q4_0 remain mandatory comparisons.
- [x] QAT source provenance is not treated as a quality result.
- [x] Initial immutable-weight target is at most 14,100 MiB; above 14,300 MiB is a hard review stop.
- [x] Required 32K CUDA-visible reserve is at least 700 MiB.
- [x] Existing 12B correctness, memory and performance evidence remains the regression baseline.

## External implementation review

- [x] Pinned imp use is classified as reference/clean-room by default through M13.
- [x] M00 approves the provenance structure in
  [`IMP_LICENSE_AND_PROVENANCE.md`](../../plans/gemma4-26b/references/imp/IMP_LICENSE_AND_PROVENANCE.md), not source
  copying.
- [x] Any later isolated MIT port requires a separate owner decision, exact hashes/notices and independent evidence.
- [x] General executor, paged KV, broad dispatch, continuous batching and host expert offload are rejected.

## Validation

Required before requesting owner acceptance:

- [x] Host configure/build and CTest pass after the M00 documentation changes: 1/1 tests passed.
- [x] Blackwell configure/build and CTest pass after the M00 documentation changes: 2/2 tests passed.
- [x] Relative-link check passes across 150 Markdown files.
- [x] Immutable 26B plan-package integrity passes `sha256sum -c SHA256SUMS.txt`; package-authored files and generated
  integrity metadata remain unchanged.
- [x] `git diff --check` passes.
- [x] Scope diff confirms no CMake, model lock, compiler, runtime, CUDA, test or tool implementation changed from
  M00 parent `b078a153772d12711032ef1b940221e825ce36e5`.

Local ignored logs are retained under `build/m00-policy/`.

## Owner acceptance

The project owner approved these exact points on 2026-08-06:

1. the concrete derived-checkpoint contract in [`docs/GEMMA4_26B.md`](../../GEMMA4_26B.md);
2. the text-only/no-MTP/no-offload initial product boundary;
3. the required comparison set and claim wording;
4. the 14,100 MiB target, 14,300 MiB stop and 700 MiB 32K reserve;
5. imp reference/clean-room use now, with copied MIT code requiring a later separate approval.

- [x] Owner accepted M00 contract.
- [x] M00 is passed.
- [x] M01 is ready.

M02 and every later milestone remain blocked until M01 passes its own exit criteria.
