# M04 — Deterministic checkpoint compiler scaffold

## Objective

Create a bounded-memory offline compiler that reads one locked BF16 source, applies an explicit versioned quantization plan, and writes one auditable Safetensors-based text-only artifact plus complete provenance.

## Why this milestone exists

Quantizer details should not be mixed with file discovery, output schema, resumability and hash generation. A stable scaffold enables independent FP8, NVFP4 and head-format milestones and ensures the inference process never performs conversion.

## Prerequisites

- M00 policy permits the derived artifact.
- M01 source locks exist.
- M03 tensor-role mapping is frozen.

## Repository areas to inspect first

- `tools/fetch_model.py`
- `tools/hf_cache.py`
- `src/model/manifest.*`
- `models/*.lock.json`
- `docs/CHECKPOINT_FORMAT.md`
- `third_party/safetensors or current reader`

## Suggested additions or boundaries

- `tools/compile_gemma4_26b.py`
- `tools/gem16_quant/compiler.py`
- `tools/gem16_quant/plan.py`
- `tools/gem16_quant/safetensors_writer.py`
- `docs/GEMMA4_26B_CHECKPOINT_COMPILER.md`

## Implementation sequence

1. Define a compiler CLI with required source lock, output directory, profile, compiler manifest path, host-memory cap and verification mode.
2. Implement source verification before opening any tensor payload.
3. Memory-map shards and expose tensor slices without loading the whole model.
4. Create a versioned `QuantizationPlan` mapping roles to encoders and target tensor names.
5. Write output to temporary files and atomically rename only after all tensor hashes and manifests pass.
6. Support deterministic sharding by target byte cap; output shard order and tensor order must be canonical.
7. Copy only approved tokenizer, generation and template metadata from locked sources.
8. Exclude all vision tensors and record excluded names/bytes/hashes in provenance.
9. Write `gem16_compilation.json` containing source repository/revision, source lock hash, compiler git SHA, toolchain versions, plan hash, profile, excluded roles, per-tensor source and output hashes, output file hashes and byte totals.
10. Add `--verify-only` to recompute every hash and schema rule.
11. Add resume only if partial state is cryptographically bound to source and plan; otherwise prefer restart.
12. Enforce host-memory cap with bounded staging and report peak RSS in tests.
13. Do not implement production quantization yet; use copy/small synthetic encoders to validate orchestration.

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
python tools/compile_gemma4_26b.py --source-lock <lock> --profile synthetic-copy --output <dir>
```
```text
python tools/compile_gemma4_26b.py --verify-only --output <dir>
```

## Risks to watch in this milestone

- Python package serialization or dictionary order can introduce nondeterminism.
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
- [ ] M05–M07 can plug encoders into a stable plan API.

## Downstream milestones unblocked

- M05
- M06
- M07

## Codex execution prompt

```text
You are implementing M04: Deterministic checkpoint compiler scaffold in Danmoreng/gem16.

Read repository AGENTS.md, docs/DECISIONS.md, docs/CORRECTNESS.md, docs/BENCHMARKING.md, the package master plan, and this milestone. Work only on M04. Inspect the actual current tree before editing and write a drift note if it differs from the anchored commit.

Implement the narrowest change that satisfies every exit criterion. Add tests before or with implementation. Do not add silent fallbacks, token-loop allocations, CPU weight offload, or unrelated refactors. Preserve all existing Gemma 4 12B exact gates.

At completion, report files changed, tests and exact results, generated evidence paths, memory/performance deltas where relevant, unresolved risks, and whether each M04 exit criterion passed. Stop before starting the next milestone.
```
