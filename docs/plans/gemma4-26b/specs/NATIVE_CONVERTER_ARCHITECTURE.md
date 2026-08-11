# Native converter architecture

## Status and scope

This is the binding architecture for the Gemma 4 26B offline converter. It is a compiler design contract, not a
runtime inference design.

| Milestone | Converter responsibility | Current boundary |
|---|---|---|
| M04 | Plan, lock, coverage, bounded copy, sharding, provenance and publication scaffold | Accepted Python standard-library `copy-v1`; byte movement only, not numerical conversion |
| M05 | BF16 attention Q/K/V/O to FP8 E4M3 plus BF16 row scales | Native C++20 batch backend is the promoted implementation; Python is oracle/control-plane support |
| M06 | BF16 shared and routed expert weights to the locked NVFP4 contract | Planned extension of the same native data plane |
| M07 | Tied embedding/output Q4_0 and NVFP4 candidates | Planned native extensions of the same data plane |
| M08 | Assemble the complete immutable text-only artifact and load it | Assembly may initially use the existing Python control plane; it performs no quantization |
| M18 | Converter/source/head attribution and large comparisons | Planned native data-plane comparison and telemetry, with small Python orchestration permitted |

M00-M04 are accepted. M05's native implementation and diagnostic full Ordinary-BF16/QAT-BF16 runs are complete;
owner-authorized clean-revision evidence and final M05 acceptance remain pending. The current executable is
`gem16-fp8-compiler`; before M06 the reusable implementation should evolve toward the planned
`gem16-checkpoint-compiler` family name. This naming evolution must not rewrite accepted M04 evidence.

## Control plane and data plane

### Allowed control-plane responsibilities

The repository-owned control plane may:

- verify immutable source locks and inspect validated Safetensors headers;
- generate and validate the exact model-specific plan and source coverage;
- select a versioned profile and native backend, and construct a strict job protocol;
- coordinate bounded staging, shard assignment, JSON schemas, provenance and error reporting;
- create `.incomplete` output, fsync, reopen/verify and atomically publish it;
- retain small deterministic codec or report oracles and generate evidence;
- invoke one user-facing compiler command which selects the native backend explicitly.

The current Python wrapper is allowed to perform these duties. It is not allowed to become a hidden numerical
backend.

### Mandatory native data-plane responsibilities

The promoted backend must perform all large tensor work:

- BF16/FP8/NVFP4/Q4_0 conversion and dequantization;
- row, block and expert streaming over large tensors;
- coupled output generation, such as weights and scales from one source traversal;
- billion-element error metrics or comparisons;
- native deterministic reductions and codec telemetry.

A missing or incompatible native backend is a visible error. There is no silent Python, NumPy, PyTorch, F16 or
other-precision fallback. Python tensor libraries may be used only under an explicit diagnostic/reference decision,
with the affected tensors and precision recorded; such a path cannot support a promoted conversion claim.

## Shared native data plane

The shared C++20 implementation consists of reusable, versioned components:

1. strict job/profile and tensor-contract validation;
2. bounded Safetensors range readers with checked offsets and source identity;
3. codec kernels for each versioned format;
4. row/block/expert workers with an explicit thread count and deterministic partitioning;
5. fixed-size scratch buffers and deterministic native reductions;
6. native output hashes and telemetry;
7. a backend-neutral result protocol consumed by the publication layer.

Each coupled transformation reads a source matrix once and emits all of its declared components together. The native
path writes directly toward the final logical Safetensors representation. It must not create a giant BF16/F16 GGUF
intermediate, materialize a complete model in host RAM, or retain a second persistent weight representation.
Runtime-specific Row8/K64 placement remains a loader concern unless a later profile explicitly changes the artifact
contract.

M05's `gem16-fp8-compiler` is the seed implementation. M06 must reuse its bounded job, codec, telemetry and failure
boundaries rather than introducing `quantize_nvfp4.py` as a production converter. M07 Q4_0 and NVFP4 head candidates
must use the same native library. M18's full-family comparison must not reintroduce billion-element Python loops.

## Integrity and publication

The native backend and control plane jointly own the following invariants:

- every source file is verified against an immutable lock before interpretation and again before publication;
- every source range, name, dtype, shape, operation and output byte count is checked;
- source, native binary, compiler/toolchain, plan, transformation and output hashes are recorded;
- native protocol/version and explicit thread count are part of provenance;
- invalid precision, layout, scale direction, shape or unsupported producer semantics fail visibly;
- output is written below `<output>.incomplete`, with restrictive permissions, checked writes and fsync;
- the staged artifact is reopened and structurally verified before one same-filesystem atomic rename;
- M08 supplies the external derived-artifact lock that binds the final manifest against mutable-manifest attacks.

Standalone M05 verification is structural/hash/source-lock-only and does not reconvert. It records
`transformation_recomputed=false`. For M05-M07 partial numerical stages, the accepted policy allows one full native
run for each approved source and selected profile or head candidate; native exhaustive codec tests, byte fixtures,
bounded thread-1-versus-N identity, complete output hashes and small independent oracles establish determinism.
M08 complete-artifact reproducibility remains a two-clean-build requirement.

## Lessons from llama.cpp

The local research is version-scoped to the clean checkout at commit
`0b14b87d7c20cb753b94b96854dd7b45306fc696`; retained evidence is
[`m05-llama-cpp-converter-research-2026-08-11.md`](../../../evidence/gemma4_26b/m05-llama-cpp-converter-research-2026-08-11.md).
The desired benchmark pin in `benchmarks/baselines/llama_cpp/commit.txt` is
`153d324bcf86d220b235ca010eeb11213f32b5d1`, so the inspected checkout must not be described as the current benchmark
revision. The local `llama.cpp-mixed` worktree contains a separate uncommitted patch and is not an upstream contract.

llama.cpp uses a useful two-stage pattern:

- `convert_hf_to_gguf.py` uses Python model mapping, PyTorch/NumPy transformations, lazy/memmap source access and a
  Python GGUF writer;
- `llama-quantize` then uses native C++/ggml codecs, mmap and explicit multithreaded block work for post-GGUF
  quantization.

Its upstream converter has no general persistent GGML FP8 type: FP8 input is dequantized, or `--fp8-as-q8` produces
Q8_0. GGML has native MXFP4 and NVFP4 codecs. MXFP4_MOE is exposed as a normal quantization mode; NVFP4 is available
through selective tensor-type overrides and native codecs, but not as the same complete mixed Gemma compiler contract.
The local mixed patch accepts compressed-tensors FP8 plus NVFP4 and produces Q8 attention plus repacked NVFP4. It is
not exact Gem16 FP8 parity, and no local llama.cpp BF16-to-26B conversion was performed.

We adopt conceptually:

- a model-mapping layer separated from native codec/data-plane code;
- native threaded codecs and bounded row/block work;
- independent reference versus optimized codec tests;
- explicit tensor-type policy rather than one universal quantizer.

We reject as Gem16 policy:

- a large BF16/F16 intermediate GGUF solely to reach native quantization;
- permissive F16 fallback when a requested quantizer cannot run;
- direct final-name overwrite without incomplete staging and atomic publication;
- absent source/compiler/output provenance, host caps and full lock verification;
- unsafe path handling or repository-code execution.

The llama.cpp GGUF NVFP4 representation is not Gem16's representation. Its 64-element superblock aggregates four
16-element groups and stores four scale bytes followed by four contiguous 8-byte packed subblocks. It uses a UE4M3
convention with a doubled FP4 lookup compensation and producer-specific scale handling. Gem16's NVFP4 contract, scale
direction, tie rules and runtime layout win. No byte or numerical parity may be claimed without explicit differential
fixtures.

## Reuse and provenance rules

Before adopting an idea or code path, agents must inspect and cite the exact pinned source, commit, file and tests.
Conceptual adoption and source-code copying are separate decisions. Copied llama.cpp code requires its MIT notice,
source hash, exact commit, destination and a recorded code-provenance entry. A pinned copy may enter tracked
`third_party/` only after an explicit decision; ignored cache checkouts are never runtime or build dependencies.
Every adopted codec receives differential tests against the donor and Gem16's own contract tests. Gem16's stronger
lock, memory, no-fallback, security, output and atomicity rules always take precedence.

## Milestone and test policy

M04 keeps its accepted Python scaffold and synthetic byte-reproducibility evidence. M05 uses the native FP8 backend.
M05-M07 partial numerical stages use one full native Ordinary run and one full native QAT run per selected profile or
head candidate, not duplicate Python/native or second full partial-artifact runs solely for reproducibility. Each
retains independent small reference oracles and bounded thread-identity evidence. M08 owns complete artifact assembly,
external locking and two clean complete-artifact builds. M18 uses native large comparisons and retains small Python
orchestration only.

Every native codec requires exhaustive boundary tests, byte fixtures, malformed-input tests, bounded-memory tests,
thread determinism and real-shape samples. Full conversion runs are separated from unit tests and require measured
throughput and owner approval when projected to be long. Benchmark claims record raw samples, peak RSS, actual native
dispatch, toolchain and exact source/artifact identities.
