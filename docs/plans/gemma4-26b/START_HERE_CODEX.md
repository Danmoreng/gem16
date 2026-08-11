# Start here: current Codex task

Do not ask Codex to implement all milestones at once. All remaining work stays on `feat/gemma4-26b`, and each
task must stop at its milestone gate.

## Current task: M05 deterministic FP8 attention compiler

The project owner explicitly authorized starting M05 on 2026-08-11 after first pausing it; that authorization
supersedes the pause. M00-M04 remain accepted. M05 is current/in progress, while M06+ remain dependency-gated.
The kickoff record is [M05 evidence](../../evidence/gemma4_26b/m05-kickoff-2026-08-11.md). The cross-milestone
compiler boundary is [NATIVE_CONVERTER_ARCHITECTURE.md](specs/NATIVE_CONVERTER_ARCHITECTURE.md); read the
version-scoped [llama.cpp conversion research](../../evidence/gemma4_26b/m05-llama-cpp-converter-research-2026-08-11.md)
before extending the converter in M06, M07 or M18.

M05 is limited to deterministic, bounded BF16-to-E4M3FN encoding for attention Q/K/V/O, with BF16
per-output-row scales, reports, tests and comparisons. The promoted conversion is a versioned native C++20 batch
backend; the Python codec is oracle/fixture/report support only and never a production fallback. Use the canonical
representation `F8_E4M3 [N,K] + BF16 [N,1]`, rowwise max-abs scale v1, round-to-nearest ties-even, finite
saturation, NaN/Inf rejection, and all-zero scale `1.0`. Exact Ordinary/QAT plan gates pass. The short native probe,
one diagnostic full run per source, structural verification, weight-only comparison and CUDA/12B regression gates are
retained in [M05 diagnostic evidence](../../evidence/gemma4_26b/m05-native-fp8-implementation-and-diagnostic-runs-2026-08-11.md).
All currently retained M05 compiler artifacts record `compiler_dirty=true`; this is diagnostic evidence, not an
accepted M05 gate. The owner has authorized the implementation commit and clean-revision evidence. Remaining gates
are one clean Ordinary plus one clean QAT run, final structural verification/review and final hashes/status.

## Shared converter boundary

M04's accepted Python standard-library scaffold owns control-plane work and byte-identical `copy-v1` synthetic
evidence; `copy-v1` is not numerical conversion. Promoted large tensor arithmetic and billion-element comparisons
must use the shared versioned native C++20 data plane. Python remains limited to locks, exact plans, schemas,
evidence and small independent oracles unless an explicit diagnostic/reference decision says otherwise. Do not add
separate Python NVFP4 or Q4 converters, a large BF16/F16 intermediate GGUF, silent fallback, or an unpinned llama.cpp
cache dependency. The current M05 `gem16-fp8-compiler` is the first native seed; the planned common family is
`gem16-checkpoint-compiler`.

## M05 boundaries and drift

- Preserve the accepted M04 action-first CLI (`plan`, `compile`, `verify`, `compare-reproducibility`) and add an
  explicit M05 profile/plan path; use explicit `--native-encoder` and `--threads`; do not use the old suggested
  `--stage` commands.
- Before implementation, make attention-only partial-artifact handling explicit: every non-attention source tensor
  must have a recorded profile/plan disposition, with no silently dropped coverage.
- C++ loader work, if needed for M05 schema/binding/operator validation, is validation only. M08 owns the first
  complete direct-load artifact and loader integration.
- Do not implement M06 NVFP4, M07 head work, M08 loader/full artifact, runtime activation or accumulation changes,
  26B execution, or any 12B behavior change.

## Native M05 execution policy

Run one native full pass for the 115-matrix Ordinary-BF16 attention family and one for QAT-BF16. Do not run a
Python/native duplicate or a second complete M05 artifact solely for reproducibility. Prove determinism with native
exhaustive codec tests, byte-golden rows, bounded threads-1-versus-N fixtures and complete output hashes. Standalone
M05 verification is structural/hash/source-lock-only, does not reconvert, and reports
`transformation_recomputed=false` until M08's external artifact lock. The short probe and diagnostic full runs are
retained in the M05 evidence link above; they do not waive the clean-revision release-evidence requirement.

## Verified kickoff state

Branch `feat/gemma4-26b` remains `feat/gemma4-26b`; the worktree is intentionally dirty while the native M05 slice
and its documentation are integrated. No milestone branch is used. The ordinary BF16, QAT BF16 and Unsloth NVFP4 prerequisites and the M04 fixture/integrity checks are recorded in the
kickoff evidence. Diagnostic implementation and full-run evidence is retained separately in
[M05 native diagnostic evidence](../../evidence/gemma4_26b/m05-native-fp8-implementation-and-diagnostic-runs-2026-08-11.md);
M05 remains in progress and unaccepted.

## Daily agent loop

```text
verify branch and current HEAD
read M05 scope and contracts
verify prerequisites and source locks
write drift note
add fixtures/tests
implement the bounded host slice
run required gates
store evidence and update status
stop at the M05 gate
```

## Do not do this

```text
“Implement Gemma 4 26B according to this entire package.”
```

That prompt encourages assumptions before tensor discovery, compiler/runtime coupling, unreviewable CUDA changes,
quality leakage and benchmark claims without evidence.
