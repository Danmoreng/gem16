# Gemma 4 26B M07 provisional NVFP4 tied head acceptance

Date: 2026-08-12
Branch: `feat/gemma4-26b`
Implementation commit: `60f500b7be567fafd483ebd6f5f9b07988197ca1`
Status: accepted experimental/reference compiler and CPU-reference stage

## Scope and result

M07 compiles the single QAT source tensor
`model.language_model.embed_tokens.weight` (`BF16 [262144, 2816]`) once into the
provisional NVFP4 tied embedding/output-head family. The artifact contains exactly four aliased components:

- packed U8 `[262144, 1408]`: 369,098,752 bytes;
- local E4M3 scales `[262144, 176]`: 46,137,344 bytes;
- weight divisor F32 `[1]`: 4 bytes;
- input divisor F32 `[1]`: 4 bytes.

The payload is 415,236,104 bytes in one 415,236,648-byte Safetensors shard including its header. The plan covers
1,013 source tensors, transforms one tied source, explicitly excludes 1,012 tensors, and contains no `lm_head`
duplicate. The M07 partial artifact is not runtime-loadable; M08 owns complete artifact assembly and loader binding.

## Clean Release run

The run started from implementation commit `60f500b7be567fafd483ebd6f5f9b07988197ca1`, with a clean worktree,
Linux x86-64, GNU 16.1.1, C++20, native Release compiler and 16 threads. The QAT source lock is
`3d9441fdebef33785e33181c335338208b8bf868cb8c7da692fd9c765cca8230`. The compiler plan and resolved-plan SHA-256
are both `e549a43864e2e64b4b0783de2337631c5b5989fb3c25f0dc94b762442ded6c27`; the NVFP4 specification SHA-256 is
`fe3f5d6a3e26eaf54f7a0d98ecef12f6de376b52743f2a6e5603180d0a5b24fa`.

| Stage | Result | Wall time |
|---|---|---:|
| plan/source-lock check | pass | 60.549 s |
| native analysis | pass | 3.938455459 s |
| native conversion | pass | 21.067485441 s |
| complete compile orchestration | pass | 177.420 s |
| standalone artifact verification | pass | 102.288 s |

The configured host cap was 8 GiB. Native source passes: 2; maximum source row: 5,632 bytes; native child peak
RSS: 4,771,840 bytes; parent peak RSS: 34,873,344 bytes. The native compiler binary SHA-256 is
`37d67f709bbed664ebde1f502bde7bb487f886b0337f740095faf37365cbf5a5`.

The verified artifact hashes are:

- compilation manifest: `7bf46ce571ea593c3202819db2a8c2bc2a29ba8bb73a23a0d5571a38c88f089e`;
- Safetensors shard: `3c200aa69710344f0c2f0075aa1c2f408fda888d553e230672ac85da73669725`;
- Safetensors index: `f241a2f2160430a1054429bec73aec60d1c480214cec6c749673f2fbd36ce454`.

The 415 MB artifact payload remains at
`/tmp/m07-release-60f500b7be567fafd483ebd6f5f9b07988197ca1/qat-head-artifact` and is not copied into the repository.

## Real tied-head diagnostic

`gem16-nvfp4-head-diagnostic` passed in 15.829 s using the verified artifact and a deterministic synthetic
post-final-normalization hidden vector. Four lookup rows (tokens 0, 1, 131072 and 262143) matched the independent
manual E2M1/E4M3/divisor/BF16-RNE reference exactly: maximum absolute error 0.

The full-vocabulary T=1 projection produced finite logits and selected token 159254 with value
`0.5457066893577576`. Independent manual checks on sampled rows, including the selected row, had maximum absolute
error 0. The diagnostic exhaustively scanned all logits for deterministic argmax selection. Suppressing token
159254 selected token 14334 with value `0.5314220190048218`; pre-suppression diagnostic logits were unchanged.
Lookup timing was 0.000181821 s, first projection timing 6.377696456 s and suppressed projection timing
6.366903547 s. These are reference-diagnostic timings only.

The diagnostic confirmed one mapped shard and one physical tied payload, with no payload copy and no row cache.
Softcap, suppression limits, lowest-token ties, malformed encodings, divisor validation and BF16 boundaries are also
covered by the host reference tests. This is not a model-quality or performance claim.

## Validation and limitations

- Python discovery: 176/176 passed;
- host Debug, Release and ASan/UBSan CTest: 1/1 passed each;
- M05/M06/M07 generator checks, plan-integrity check and `git diff --check`: passed;
- no shared loader, runtime or CUDA path changed, so the proportional M07 gate did not rerun the 12B execution
  regression;
- the repository still has no executable Gemma 4 26B runtime path.

M07 is accepted. M08 is next and ready to begin, but was not started by this change. See
`artifacts/m07/acceptance.json` for the machine-readable acceptance record and the complete evidence-file hashes.
