# Gemma 4 26B M06 native NVFP4 compiler acceptance

Date: 2026-08-12
Branch: `feat/gemma4-26b`
Implementation commit: `81055eb48e05321481a8b63dd0dc5e7e017a7c00`
Status: accepted experimental/reference compiler stage
Acceptance record SHA-256: `6aec6cba051cc7c71b04550a65cdcfba75f577ee69b40718fc51b36fa317c556`

## Scope and boundaries

M06 adds the native C++20 BF16-to-NVFP4 compiler for the 150 QAT shared/routed expert source tensors. The
partial artifact is not runtime-loadable and makes no model-quality or performance-qualification claim. The
protected 12B path remains unchanged. The complete Ordinary-BF16 conversion and causal attribution remain
conditional M18 work.

The frozen contract is E2M1 with low-nibble-first packing, E4M3FN group-16 scales, one BF16-RNE weight divisor
per source tensor, and input divisor `1.0`. The full plan contains 1,013 source tensors: 150 transformed expert/shared
tensors, 600 output tensors, 863 excluded/deferred tensors, and 13,147,454,640 output bytes in 15 Safetensors
shards.

## Clean full-run evidence

The run started from the clean committed worktree at commit `81055eb48e05321481a8b63dd0dc5e7e017a7c00`, using the
host Release native executable:

- protocol: `gem16-nvfp4-direct-v1`;
- native binary SHA-256: `9956c5ed66b6e1624b37256a963586656c5f0a88a672ed78603650c4b13aba76`;
- GNU 16.1.1, C++20, Linux x86-64, Release, 16 worker threads;
- QAT source lock SHA-256: `3d9441fdebef33785e33181c335338208b8bf868cb8c7da692fd9c765cca8230`;
- compiler plan SHA-256: `d303d72a7d4f5afdcc04f989d3dfdbc66f3712fa524e9cf0674083d42b4e3725`.

Results:

| Stage | Result | Duration |
|---|---:|---:|
| plan/source-lock check | pass | 53.146608 s |
| native analysis | pass | 14.141594 s |
| native NVFP4 conversion | pass | 62.464874 s |
| complete compile orchestration | pass | 280.697080 s |
| standalone artifact verification | pass | 183.134406 s |

The native arithmetic time is approximately 76.6 seconds. The larger compile wall time includes source-lock
verification, shard preallocation, direct-write durability, telemetry, manifest construction and canonical artifact
verification. The verification pass is hash/structure verification and does not reconvert tensors. These timings
are evidence for this machine and boundary only, not a performance qualification.

The parent control-plane peak RSS was 37,122,048 bytes during compile; native child peak RSS was 7,798,784 bytes.
The configured host cap was 8 GiB and the maximum native source row was 5,632 bytes. The native compiler reported
two source passes. The resulting artifact was independently verified as 600 tensors and 13,147,454,640 output
bytes, with `compiler_dirty=false` and the committed compiler identity.

## Ordinary/Unsloth convention diagnostic

The frozen bounded diagnostic passed all 48 logical matrices from 20 complete parent source tensors. Parent tensors
were compiled whole so routed fused Gate/Up and Down divisors use the production source-tensor granularity; logical
expert slices then use the exact Gate/Up row offsets. It is a convention screen, not a quality claim.

| Comparison | Maximum relative L2 | Minimum cosine | Minimum SQNR |
|---|---:|---:|---:|
| Ordinary compiled vs source | 0.0951944 | 0.995470 | 20.4278 dB |
| Unsloth reference vs source | 0.0954010 | 0.995456 | 20.4089 dB |
| Ordinary compiled vs Unsloth | 0.102928 | 0.994703 | 19.7494 dB |

All 48 records passed the frozen finite/shape/divisor/split/threshold checks. Ordinary source lock SHA-256 is
`1932b06d568ce8c72f3d2013d39ffd15dd61d37732638c2013f2220eb177a620`; Unsloth source lock SHA-256 is
`6635b15f4f974dec22821c7a9efd4d2404ba4ce9afb1424ca2f3529d20db7941`.

## Tests and regressions

- 43 focused Python tests passed;
- 166-test Python discovery passed (the suite intentionally exercises a CLI parser error as part of one test);
- host Debug, Release and ASan/UBSan unit CTest: 1/1 each;
- Blackwell CTest: 3/3, including real native-compiler output consumed by CPU reference, CUDA reference and SM120 direct/fused paths;
- CUDA `compute-sanitizer` memcheck, initcheck and racecheck: zero errors/hazards;
- NVFP4 plan generator check, plan-integrity check and `git diff --check`: passed;
- protected 12B validation: output token IDs `[9503, 106]`, zero fallbacks, `native_sm120` projection, and
  9,304,895,488-byte weight arena. This characterization report is not a performance qualification.

## Reproduction records

The exact preflight and command transcript is retained in `artifacts/m06/preflight-release.txt`. Compact
verification, 12B, acceptance and hash-summary records remain tracked. The expanded plan, compile manifest/report
and Ordinary/Unsloth diagnostic are pruned from Git; `artifacts/raw-evidence-index.json` and
`artifacts/m06/acceptance.json` retain their original paths, sizes and SHA-256 values. The hash summary still binds
the 15-shard output inventory.

The 13 GiB artifact itself remains in the external run directory under `/tmp`; it is not duplicated in the repository.
The acceptance record therefore binds the verified shard hashes from the compilation manifest without adding those
large payloads to the repository.
