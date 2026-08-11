# M04 — Deterministic checkpoint compiler scaffold

## Objective

Create the accepted bounded-memory offline compiler scaffold that reads one locked BF16 source, applies an explicit versioned plan, and writes one auditable Safetensors-based text-only artifact plus complete provenance. M04's `copy-v1` encoder moves bytes only; it is not numerical quantization and is not a precedent for promoted Python tensor loops. Read the binding [native converter architecture](../specs/NATIVE_CONVERTER_ARCHITECTURE.md) before extending this scaffold.

## Why this milestone exists

Quantizer details should not be mixed with file discovery, output schema, resumability and hash generation. A stable scaffold enables independent FP8, NVFP4 and head-format milestones and ensures the inference process never performs conversion.

## Prerequisites

- M00 policy permits the derived artifact.
- M01 source locks exist.
- M03 tensor-role mapping is frozen.

## Current status and binding architecture

M04 is accepted. Its Python standard-library implementation is the control-plane scaffold for immutable source locks,
exact coverage, bounded byte movement, Safetensors sharding, provenance and atomic publication. `copy-v1` does not
quantize or transform tensor values. Numerical production conversion begins with the native C++20 M05 backend and
continues through the shared data plane described in the [native converter architecture](../specs/NATIVE_CONVERTER_ARCHITECTURE.md).

## Repository areas to inspect first

- `tools/fetch_model.py`
- `tools/hf_cache.py`
- `src/model/manifest.*`
- `models/*.lock.json`
- `docs/CHECKPOINT_FORMAT.md`
- `third_party/safetensors or current reader`

## Suggested additions or boundaries

- `tools/compile_gemma4_26b.py`
- `tools/gem16_compile/`
- `docs/GEMMA4_26B_CHECKPOINT_COMPILER.md`
- `docs/plans/gemma4-26b/specs/NATIVE_CONVERTER_ARCHITECTURE.md`

## Implementation sequence

1. Define a compiler CLI with required source lock, output directory, profile, compiler manifest path, host-memory cap and verification mode.
2. Implement source verification before opening any tensor payload.
3. Memory-map shards and expose tensor slices without loading the whole model.
4. Create a versioned `QuantizationPlan` mapping roles to encoders and target tensor names. Keep M04's `copy-v1` as byte movement only; numerical M05-M07 encoders belong to the shared native C++ data plane described in the architecture specification.
5. Write output to temporary files and atomically rename only after all tensor hashes and manifests pass.
6. Support deterministic sharding by target byte cap; output shard order and tensor order must be canonical.
7. Copy only approved tokenizer, generation and template metadata from locked sources.
8. Exclude all vision tensors and record excluded names/bytes/hashes in provenance.
9. Write `gem16_compilation.json` containing source repository/revision, source lock hash, compiler git SHA, toolchain versions, plan hash, profile, excluded roles, per-tensor source and output hashes, output file hashes and byte totals.
10. Add action-first `plan`, `compile` and `verify` operations with strict hash and schema checks. M04 verification may recompute byte hashes for its copy scaffold; later native numerical milestones define their own no-reconversion verification boundary.
11. Add resume only if partial state is cryptographically bound to source and plan; otherwise prefer restart.
12. Enforce host-memory cap with bounded staging and report peak RSS in tests.
13. Do not implement production numerical quantization in M04; use `copy-v1` and small synthetic byte fixtures to validate orchestration. M05 is the first promoted native C++ conversion milestone; Python remains control-plane/oracle support only for later numerical work.

## Required tests

- Tiny synthetic multi-shard source compiles deterministically.
- Output tensor order and file hashes repeat exactly.
- Corrupted source, wrong lock, changed plan and interrupted output fail cleanly.
- Vision-role exclusion is exact.
- Atomic publish leaves no valid-looking partial artifact.
- Compiler respects a deliberately small host-memory cap.
- Verification works without importing or executing model repository code.

## Evidence and documentation outputs

- Compiler CLI help and schema documentation.
- Synthetic source/output fixtures and expected hashes.
- Compiler manifest JSON schema.
- Peak host-memory test result.
- Decision on canonical compiler environment/container.

## Suggested commands

```text
python3 tools/compile_gemma4_26b.py plan --source-lock <lock> --profile synthetic-copy-v1 --head-format source --compiler-manifest <plan> --max-host-memory <bytes> --staging-bytes <bytes>
```
```text
python3 tools/compile_gemma4_26b.py verify --source-lock <lock> --profile synthetic-copy-v1 --head-format source --compiler-manifest <plan> --model <dir> --max-host-memory <bytes> --staging-bytes <bytes>
```

## Risks to watch in this milestone

- Python package serialization or dictionary order can introduce nondeterminism in the M04 control plane; canonical JSON and the reference environment close this risk. Promoted numerical conversion must not rely on Python elementwise loops.
- Whole-tensor reads of expert arrays can exceed host RAM.
- Atomic rename semantics differ between Windows and Linux.

## Forbidden shortcuts

- Do not use pickle or execute remote code.
- Do not write output directly over a valid artifact.
- Do not hide tensor transposes in generic callbacks.
- Do not depend on mutable Python package versions.
- Do not add GPU runtime conversion.

## Exit criteria

- [ ] Compiler orchestration is deterministic.
- [ ] Output is Safetensors plus explicit metadata, not an opaque binary.
- [ ] No inference code invokes the compiler.
- [ ] Every output tensor is traceable to one source tensor or declared transform.
- [ ] Vision exclusion is visible and exact.
- [ ] M05–M07 can plug native C++ data-plane encoders into a stable plan/publication API without changing M04 artifact semantics.

## Downstream milestones unblocked

- M05
- M06
- M07

## Codex execution prompt

```text
You are implementing M04: Deterministic checkpoint compiler scaffold in Danmoreng/gem16.

Read repository AGENTS.md, docs/DECISIONS.md, docs/CORRECTNESS.md, docs/BENCHMARKING.md, the package master plan, `../specs/NATIVE_CONVERTER_ARCHITECTURE.md`, the version-scoped llama.cpp research evidence, and this milestone. Work only on M04. Inspect the actual current tree before editing and write a drift note if it differs from the anchored commit.

Implement the narrowest change that satisfies every exit criterion. Add tests before or with implementation. Do not add silent fallbacks, token-loop allocations, CPU weight offload, or unrelated refactors. Preserve all existing Gemma 4 12B exact gates.

At completion, report files changed, tests and exact results, generated evidence paths, memory/performance deltas where relevant, unresolved risks, and whether each M04 exit criterion passed. Stop before starting the next milestone.
```
