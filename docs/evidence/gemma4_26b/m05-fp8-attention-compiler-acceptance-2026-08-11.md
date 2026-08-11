# Gemma 4 26B M05 FP8 attention compiler acceptance

Date: 2026-08-11
Branch: `feat/gemma4-26b`
Implementation commit: `d91388113d68974f9ab7cec1a90ef768285c0645`
Status: accepted; M06 is dependency-unblocked but not started

## Acceptance decision

The project owner authorized the implementation commit, clean-revision evidence runs, documentation, commit and
push. M05 is accepted as the deterministic, non-runtime-loadable FP8 attention compiler stage. This acceptance does
not authorize 26B runtime execution, start M06, or make a model-quality claim.

The accepted architecture is a strict hybrid: Python owns source locks, exact plans, bounded orchestration, schemas,
provenance and atomic publication; the versioned C++20 data plane owns all promoted BF16-to-E4M3FN arithmetic and
large Ordinary-versus-Unsloth comparison work. There is no production Python conversion fallback.

Machine-readable acceptance evidence is retained in
[`artifacts/m05/acceptance.json`](../../../artifacts/m05/acceptance.json), SHA-256
`05d1534759dbf6b4cc455f54c252d6a34f4337bacd9c18e18355c6868b329603`.

## Clean implementation identity

Both clean conversions were produced from the same clean implementation commit:

- commit: `d91388113d68974f9ab7cec1a90ef768285c0645`;
- compiler dirty: `false`;
- native executable: `build/Linux/host-release/bin/gem16-fp8-compiler`;
- native SHA-256: `b0f5f85fb4ea75abc7f3bfa78d85508efda0936845f1e6477c647f0a1f1a973f`;
- protocol: `gem16-fp8-batch-v1`;
- build: GNU 16.1.1, Release, C++20, Linux x86-64;
- threads: 8;
- locale: `C.UTF-8`.

The native executable is CUDA-free. The full source locks were verified before interpretation and reverified before
publication. Staged output was structurally reopened and verified before atomic publication.

## Clean full conversions

| Lane | Result | Compiler duration | Outputs | Tensor bytes | Native peak RSS | Control-plane peak RSS | Manifest SHA-256 |
|---|---|---:|---:|---:|---:|---:|---|
| Ordinary BF16 | pass | 137.780519 s | 230 | 1,110,850,560 | 6,279,168 | 38,727,680 | `ba368221b50938de08e8f2e3fa3585f383cd4dae772d1dbac611830e63d34d1e` |
| QAT BF16 | pass | 155.516982 s | 230 | 1,110,850,560 | 6,201,344 | 35,471,360 | `f11bce2907d133a561573ff41c52173033c748c548ad7e777a7fdd8a2c00df94` |

Retained clean reports and complete manifest hash records:

- `artifacts/m05/ordinary-compile.json`, SHA-256
  `90cd98eeb5eb0ce72c8cca58330816d86eb56eb4bbcaa4148d61674a52a51eb1`;
- `artifacts/m05/qat-compile.json`, SHA-256
  `b3c4006e26dfcdc0e1d86b4982b2087edf447880ed65df5b5e901e43ec4ea182`;
- `artifacts/m05/ordinary-gem16-compilation-clean.json`, SHA-256
  `ba368221b50938de08e8f2e3fa3585f383cd4dae772d1dbac611830e63d34d1e`;
- `artifacts/m05/qat-gem16-compilation-clean.json`, SHA-256
  `f11bce2907d133a561573ff41c52173033c748c548ad7e777a7fdd8a2c00df94`.

Each lane contains exactly 115 matrices and 230 outputs: Q/K/O in all 30 layers, V in the 25 local layers, and no
synthetic V in global layers 5, 11, 17, 23 or 29. Ordinary and QAT clean tensor/shard hashes are respectively
identical to their earlier corrected diagnostic payloads; the clean reruns were required for provenance, not solely
for reproducibility.

## Verification and comparison

Standalone verification passed for both clean artifacts and did not reconvert tensors:

| Lane | Report SHA-256 | `transformation_recomputed` |
|---|---|---|
| Ordinary | `4c58b9bff1b4e4db1d9006dafd75b46c99f55c4cac9924cda483d49288d6637e` | `false` |
| QAT | `ba85efcae90933c6cc38970bdd9a9325adcae327ca757598047f5e97a0d50d36` | `false` |

The native clean Ordinary-versus-Unsloth comparison covered all 115 matrices and 1,110,179,840 weight elements.
Report: `artifacts/m05/ordinary-vs-unsloth-fp8.json`, SHA-256
`cc3cc415f114c9a6f578d0e195543becd1c4661de7ae1176d85f496922f8672d`.

- raw FP8 mismatch rate: `0.02980192560513439`;
- BF16 row-scale mismatches: `0`;
- reconstruction relative L2 using Unsloth as reference: `0.015792460337879278`;
- reconstruction cosine similarity: `0.9998753477506377`;
- reconstruction SQNR: `36.03100410290244` dB;
- maximum absolute error: `0.055419921875`.

These are stored-weight and row-scale measurements only. They are not activation/operator-output evidence, a quality
qualification, or QAT attribution. Real-activation comparison remains explicitly downstream M12 work.

## Correctness and regression gates

The final implementation/evidence state passed:

- Python: 147/147;
- host-debug CTest: 1/1;
- host-sanitize ASan/UBSan CTest: 1/1;
- Blackwell-release CTest: 2/2, including CUDA;
- native E4M3FN byte/rounding/saturation/NaN/Inf/signed-zero/zero-row fixtures;
- native row-partial telemetry differential golden against the Python oracle;
- deterministic threads-1-versus-N bounded fixtures;
- source/descriptor/output/SHA-256, resource-limit and failure-cleanup tests;
- report symlink/containment and artifact-root-symlink rejection tests;
- checked FP8 plans, retained semantic reports and Gemma-4-26B package integrity.

The clean 12B regression retained exact output `[9503, 106]`, zero fallbacks, `native_sm120` projection dispatch and
a 9,304,895,488-byte weight arena. Report: `artifacts/m05/12b-validation.json`, SHA-256
`a588b865e981b40c57841495e8c50f886796693923ccb1953aec89df4375e7f8`.

## Exit criteria

- [x] Versioned native C++20 FP8 output is deterministic and fully specified; Python is oracle/control-plane only.
- [x] One clean full Ordinary-BF16 and one clean full QAT-BF16 attention compile pass.
- [x] Runtime schema/binding validation and bounded native tests pass; runtime loading remains M08.
- [x] Ordinary-versus-Unsloth differences are quantified for every attention matrix.
- [x] Standalone verification is structural/hash/source-lock-only and records `transformation_recomputed=false`.
- [x] The mature 12B loader, generation and native dispatch remain unchanged.

M05 is complete. M06 is dependency-unblocked but remains not started pending separate owner direction.
