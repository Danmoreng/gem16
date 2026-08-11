# Gemma 4 26B M05 native FP8 implementation and diagnostic runs

Date: 2026-08-11
Branch: `feat/gemma4-26b`
Milestone: M05 — deterministic FP8 attention compiler
Status: diagnostic implementation evidence retained; M05 remains current/in progress and unaccepted

## Evidence classification

All artifacts and reports in this record were produced while the implementation worktree was dirty. They are
retained diagnostic evidence, not clean-revision release evidence. The M05 partial artifacts remain
`fp8-attention-partial-v1`, `head_format=deferred`, and `m05_fp8_attention_partial_not_runtime_loadable`.
No 26B runtime execution or quality conclusion is implied.

The native executable was built as:

- path: `build/Linux/host-release/bin/gem16-fp8-compiler`
- SHA-256: `ca0d48d92a76d5e931c19d46a9b0570ea48b37bed77550d3e194f7f740c36c9f`
- GNU 16.1.1, C++20, Release, Linux x86-64, eight threads
- native protocol: `gem16-fp8-batch-v1`

## Short native throughput probe

The retained machine-readable record is
[`artifacts/m05/native-throughput-probe.json`](../../../artifacts/m05/native-throughput-probe.json).
The probe used seven real attention matrices: layer 0 Q/K/V/O and layer 5 Q/K/O. It intentionally used no
full-model conversion and measured direct native arithmetic only.

| Measurement | Result |
|---|---:|
| Matrices | 7 |
| Weight elements | 83,623,936 |
| Output payload | 83,670,016 bytes |
| Source-range hashing | 0.095803 s |
| Native arithmetic | 0.999307 s |
| Native rate | 83,681,904.045 elements/s |
| Projected full native arithmetic | 13.267 s (0.221 min) |
| Peak child RSS | 5,521,408 bytes |
| Maximum source row | 16,384 bytes |

The projection was short enough that no additional long-run approval gate was required. It does not include
source-lock verification or Python orchestration overhead.

## Initial full native run per M05 source

The initial owner-authorized diagnostic runs used the exact 115-matrix plans and produced 230 output tensors (115 FP8
weights plus 115 BF16 row-scale tensors), totaling 1,110,850,560 tensor bytes. No duplicate full conversion was
performed solely for reproducibility.

| Source | Command result | Wall time | Manifest SHA-256 | Child peak RSS | Python process peak RSS |
|---|---|---:|---|---:|---:|
| Ordinary BF16 | pass | 186.157 s | `a283db41…` | 6,205,440 | 38,072,320 |
| QAT BF16 | pass | 184.857 s | `dbe330c8…` | 6,352,896 | 37,908,480 |

The complete reports are `artifacts/m05/ordinary-compile.json` and `artifacts/m05/qat-compile.json`; both record
`compiler_dirty=true` and are diagnostic only. Their artifacts are under
`build/models/ordinary-fp8-attention-partial/` and `build/models/qat-fp8-attention-partial/`.

## Owner-authorized post-review Ordinary conversion

After the native telemetry association and report-path containment review fixes, the owner explicitly authorized one
additional full conversion to exercise the corrected implementation. The parent session built the host-release
native executable and ran the complete Ordinary plan directly with eight threads and `--allow-dirty` into the fresh
path `build/models/ordinary-fp8-attention-partial-post-review/`.

| Measurement | Result |
|---|---:|
| Status | pass |
| Compiler report duration | 157.342282 s |
| Shell wall time | 157.450 s |
| Output tensors | 230 |
| Output tensor bytes | 1,110,850,560 |
| Native executable SHA-256 | `b0f5f85fb4ea75abc7f3bfa78d85508efda0936845f1e6477c647f0a1f1a973f` |
| Native child peak RSS | 6,295,552 bytes |
| Python peak RSS | 38,174,720 bytes |
| Compilation manifest SHA-256 | `f1587fc40d0aae48a025dab5df818b0bb923c87732723753c0e97d2855477502` |
| Compile report SHA-256 | `7cf46b5d0a3f1f6309a4b42748e94487c4f34e41ac9d7ab204e27272767f95c3` |

The report is retained as `artifacts/m05/ordinary-compile-post-review.json`. Compile-time canonical structural/hash
verification and the post-conversion source reverification passed. A metadata comparison against the pre-fix
Ordinary diagnostic artifact found zero mismatching tensor payload hashes and byte-identical Safetensors shard/index
hashes. All 115 weight telemetry records changed only at the expected binary64 association boundary; the maximum
absolute deltas were `2.2686713618824683e-14` source RMS, `1.3520434771763234e-14` relative L2,
`6.543765529443135e-12` cosine, and `4.440892098500626e-12` dB SQNR. No redundant standalone verification or second
post-fix conversion was run. This artifact also records `compiler_dirty=true` and remains diagnostic.

## Structural verification and Ordinary comparison

Standalone verification is hash/source-lock-only and does not reconvert. The retained Ordinary verification
report has SHA-256
`f8b1ee6f2542e11318ec6eb25073cd938baf0f33f4df15e43f466cb65c785a03`. The QAT verification completed in
122.286 s, reported `transformation_recomputed=false`, and bound compilation manifest
`dbe330c8bd69a057f774cc8e0cb9e8b9f81c7e6c1ade2b16c4141bf2812c0260`.

The single native Ordinary-versus-Unsloth comparison covered all 115 matrices. Its retained metrics were:

- raw weight mismatch rate: 2.98019%;
- row-scale mismatches: 0;
- relative L2: `0.015792460337879278`;
- cosine similarity: `0.9998753477506377`;
- maximum absolute error: `0.055419921875`;
- comparator SHA-256: `ca0d48d92a76d5e931c19d46a9b0570ea48b37bed77550d3e194f7f740c36c9f`;
- comparator threads: 8.

The comparison is weight-only diagnostic evidence. It is not activation/operator-output evidence, not model-quality
qualification, and not QAT attribution. Real-activation CUDA comparison is explicitly downstream M12 work. The
canonical retained comparison report SHA-256 after its metadata-only canonical-equation normalization is
`fd48bfc49adb1f1b0b94aff4c741640ed4a1adea0d85c130f424251010c43af3`.

## Run attribution

The short probe and the owner-authorized post-review Ordinary conversion were run by the parent session in a direct
shell. The initial Ordinary and QAT full compiles, QAT verification, and the Ordinary comparison were each run exactly
once by the bounded test-agent workers using the commands retained in their handoff; no substantive failure was
retried. The original host-debug, host-sanitize and Python results were development-worker reports. After the two
review fixes, the parent directly reran focused tests, the 147-test Python suite, host-debug, host-sanitize and
Blackwell CTest. The initial Blackwell build/CTest and the single 12B validation were run once by a test-agent worker.
This attribution distinguishes worker/test-agent evidence from parent-retained review; all outputs remain
dirty-worktree diagnostics.

## Regression gates

The following gates passed once during this implementation slice; they do not accept M05:

- host-debug CTest: 1/1;
- host-sanitize CTest: 1/1;
- Blackwell CTest: 2/2;
- Python suite after the review fixes: 147/147;
- 12B validation: output tokens `[9503, 106]`, fallbacks `0`, weight arena `9,304,895,488` bytes, dispatch
  `native_sm120`.

The exact 12B validation command was:

```text
python3 tools/validate_inference.py \
  --run build/Linux/blackwell-release/bin/gem16-run \
  --model models/checkpoints/unsloth-gemma-4-12b-it-NVFP4-b1f6497 \
  --output artifacts/m05/12b-validation.json
```

The report path is retained above, but its JSON does not contain complete executable/model/commit/command
provenance; this record supplies that invocation. The run was characterization evidence (`benchmark_qualified=false`),
not a promoted performance claim.

## Remaining M05 acceptance blockers

M05 is not yet accepted. The owner authorized committing the implementation and running clean-revision evidence.
The remaining gates are a clean native build from that implementation commit, one clean Ordinary and one clean QAT
partial conversion, retained structural verification and hashes, final review, and final status documentation.

The current diagnostic evidence does not make the partial artifact runtime-loadable, does not waive the
clean-revision policy, and does not authorize M06 or any other downstream production milestone.
