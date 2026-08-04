# M08 — Derived checkpoint artifact, schema and direct loader

## Objective

Assemble the compiled FP8/NVFP4/quantized-head tensors into a single immutable, text-only Safetensors artifact and extend gem16 to validate and load it directly into one final GPU weight layout.

## Why this milestone exists

The runtime needs a reproducible deployment artifact that fits the target and preserves complete provenance. It must remain auditable and must not hide a second conversion, duplicate weight copy or architecture fallback at startup.

## Prerequisites

- M05 and M06 complete
- M07 provisional head decision
- M00 derived-artifact policy accepted

## Repository areas to inspect first

- `tools/fetch_model.py`
- `tools/hf_cache.py`
- `src/model/safetensors.cpp`
- `src/model/checkpoint_loader.cpp`
- `src/model/manifest.cpp`
- `include/gem16/types.h`
- `src/cuda/engine/target_model.cu`
- `models/gemma4-12b-nvfp4.lock.json`

## Suggested additions or boundaries

- `models/gemma4-26b-gem16-hybrid.lock.json`
- `docs/GEMMA4_26B_CHECKPOINT.md`
- `tools/gem16_compile/artifact_writer.py`
- `tools/verify_compiled_model.py`
- `src/model/compiled_checkpoint.cpp`

## Implementation sequence

1. Finalize the derived checkpoint schema described in `specs/DERIVED_CHECKPOINT_SCHEMA.md` before writing production shards.
2. Write immutable `config.json`, tokenizer assets, generation metadata, `gem16_compilation.json`, Safetensors index and sharded payloads.
3. Omit vision/audio/video tensors entirely from the production artifact. Record omissions explicitly rather than marking absent tensors as corrupt.
4. Record source repository/revision/file hashes, compiler repository commit, dirty-state flag, toolchain lock, quantizer configuration, per-output-tensor source mapping and SHA-256.
5. Use deterministic shard ordering, tensor ordering, JSON serialization and file naming so two clean builds are byte-identical.
6. Extend `gem16-inspect` to recognize the compiled model family and report source provenance, selected head format, logical/source dtype, runtime layout and text-only status.
7. Extend the loader to allocate only selected resident tensors. Stream source bytes into final allocations; preserve the current exact Row8/K64 load tiling without retaining a raw GPU copy.
8. Bind model traits rather than 12B constants. Fail before allocation on unknown schema versions, wrong tensor counts, incompatible quantizer parameters or missing provenance.
9. Generate a lock file compatible with `tools/fetch_model.py` or a deliberately versioned extension of that lock schema.
10. Run load/unload repetition and failure-injection tests to prove cleanup and error messages.

## Required tests

- Two independent clean compiler runs produce identical file hashes and artifact lock.
- Manifest byte totals equal Safetensors payload totals and planned resident totals.
- Loader rejects modified tensors, config, compilation metadata, scale links and wrong source hashes.
- No modality tensor is present or allocated.
- Only one tied embedding/head allocation and one final expert layout exist.
- Peak startup VRAM and host RSS are recorded; no transient device OOM on the reference 16 GB GPU.
- 12B direct checkpoint loading remains unchanged and green.

## Evidence and documentation outputs

- `models/gemma4-26b-gem16-hybrid.lock.json`
- `artifacts/m08/compiled-manifest.json`
- `artifacts/m08/reproducibility-hashes.json`
- `artifacts/m08/load-memory-telemetry.json`
- `docs/GEMMA4_26B_CHECKPOINT.md`

## Suggested commands

```text
python tools/compile_gemma4_26b.py --source-lock models/gemma4-26b-qat-bf16.lock.json --profile sm120-text-hybrid-v1 --output build/models/gemma4-26b-qat-hybrid
```
```text
python tools/verify_compiled_model.py --model build/models/gemma4-26b-qat-hybrid --lock models/gemma4-26b-gem16-hybrid.lock.json
```
```text
build/blackwell-release/bin/gem16-inspect --model build/models/gemma4-26b-qat-hybrid --validate --json artifacts/m08/compiled-manifest.json
```

## Risks to watch in this milestone

- Writing runtime-tiled weights to disk would couple the artifact to SM120 and weaken auditability.
- Keeping source and final device copies during load can exceed 16 GB.
- A non-deterministic JSON order can invalidate reproducibility despite identical tensors.
- Tokenizer/config assets from mismatched revisions can create silent behavior changes.

## Forbidden shortcuts

- A runtime that downloads or compiles weights automatically during inference startup.
- A private opaque binary container without a tensor manifest.
- Omitting compiler/source provenance.
- Treating missing vision tensors as a generic fallback to another model.
- Keeping duplicate LM-head or source-order expert weights resident.

## Exit criteria

- [ ] A complete text-only derived checkpoint builds reproducibly.
- [ ] `gem16-inspect --validate` accepts it and reports exact byte accounting.
- [ ] The loader binds every 26B tensor without duplicate device layouts.
- [ ] Artifact lock and provenance are complete and immutable.
- [ ] Weight arena prediction is within the M09 planning envelope.
- [ ] All existing 12B download/inspect/load tests remain green.

## Downstream milestones unblocked

- M09 memory planning
- M11 CUDA reference path
- M13 full-model inference

## Codex execution prompt

```text
You are implementing M08: Derived checkpoint artifact, schema and direct loader in Danmoreng/gem16.

Read repository AGENTS.md, docs/DECISIONS.md, docs/CORRECTNESS.md, docs/BENCHMARKING.md, the package master plan, and this milestone. Work only on M08. Inspect the actual current tree before editing and write a drift note if it differs from the anchored commit.

Implement the narrowest change that satisfies every exit criterion. Add tests before or with implementation. Do not add silent fallbacks, token-loop allocations, CPU weight offload, or unrelated refactors. Preserve all existing Gemma 4 12B exact gates.

At completion, report files changed, tests and exact results, generated evidence paths, memory/performance deltas where relevant, unresolved risks, and whether each M08 exit criterion passed. Stop before starting the next milestone.
```
