# M01 — Immutable source locks and golden evidence

## Objective

Pin every upstream model and software reference needed to distinguish source-weight effects, quantizer effects and runtime effects; then capture a small but comprehensive set of BF16/Q4_0/Unsloth reference outputs.

## Why this milestone exists

Without full source locks, any later tensor or quality comparison is ambiguous. The program needs four model sources: QAT BF16, ordinary BF16, official Q4_0, and Unsloth NVFP4. It also needs exact Transformers, llama.cpp, compressed-tensors and CUTLASS revisions used as reference implementations.

## Prerequisites

- M00 decision merged.
- Hugging Face access to gated Google repositories is available.
- Sufficient SSD and host RAM are available for source checkpoints and reference runs.

## Repository areas to inspect first

- `models/gemma4-12b-nvfp4.lock.json`
- `tools/fetch_model.py`
- `tools/hf_cache.py`
- `docs/CORRECTNESS.md`
- `benchmarks/`
- `toolchains/blackwell16gb.lock`
- `src/model/checkpoint_loader.cpp`
- `gem16-inspect tooling`

## Suggested additions or boundaries

- `models/gemma4-26b-qat-bf16.lock.json`
- `models/gemma4-26b-base-bf16.lock.json`
- `models/gemma4-26b-unsloth-nvfp4.lock.json`
- `models/gemma4-26b-qat-q4_0.lock.json`
- `tools/capture_gemma4_26b_goldens.py`
- `benchmarks/corpora/gemma4_26b/`

## Implementation sequence

1. Resolve full immutable commit SHAs and remote file manifests for Google QAT BF16, Google ordinary BF16, Google official Q4_0 GGUF and Unsloth NVFP4. Never store `main` or a short SHA.
2. Before downloading large payloads, calculate required source bytes, cache duplication, derived-artifact space and temporary compiler output. Verify local capacity or record a staged/external-storage plan. The current reference host's approximately 121 GB free space is not assumed sufficient for all variants simultaneously.
3. Create lock files using the existing schema style, including each source file's exact size, SHA-256, git/LFS/Xet identity and any tokenizer/template source override.
4. Pin a Transformers revision whose Gemma 4 implementation will generate goldens; pin llama.cpp for Q4_0 decode/quantization; pin compressed-tensors and CUTLASS versions used for format interpretation.
5. Extend `tools/fetch_model.py` only as needed for multi-shard models, nested files or source-type metadata. Preserve safe-path and immutable-revision checks.
6. Create a source-inventory command that writes tensor names, shapes, dtypes, shard, offset and alias information without loading the full model into RAM.
7. Build a frozen prompt corpus with tiny synthetic prompts, multilingual text, code, reasoning, tool syntax, repeated tokens, 1024-window boundary, local/global boundary and long-context samples. Store token IDs and tokenizer hashes.
8. Capture BF16 reference data for selected tokens and layers: embeddings, pre/post norms, Q/K/V/O outputs, router input, 128 router probabilities, top-8 IDs/weights, shared MLP, routed expert contributions, layer residuals, final logits and generated IDs. Record whether local 64 GiB/no-swap execution is safe or a larger reference host is required.
9. Capture official Q4_0 teacher-forced logits and deterministic outputs through the pinned reference runtime.
10. Capture Unsloth NVFP4 outputs and format inventory through a trusted runtime. Record any fallback or unsupported operator.
11. Split calibration/development/test manifests now. The final quality test manifest must be write-protected and excluded from quantizer tuning.
12. Add checks that every golden includes source lock hash, software revision, dtype, device and exact input token IDs.

## Required tests

- Lock parser rejects mutable or short revisions.
- Fetch/verify-only succeeds from a clean cache and fails after a byte is modified.
- Multi-shard linking does not duplicate large payloads.
- Capacity preflight rejects a download plan larger than the selected filesystem budget before fetching payload shards.
- Golden capture repeated twice produces identical deterministic files where the reference supports determinism.
- Tokenizer/token-ID fixtures match across ordinary BF16, QAT BF16 and official Q4_0.
- Calibration, development and test manifests have zero overlap by document and token-span hash.

## Evidence and documentation outputs

- Four model lock files and software-reference lock metadata.
- A machine-readable tensor inventory for each model source.
- Golden manifest and compact fixtures permitted by the model license.
- A data-split audit report.
- A source/cache/derived-artifact storage plan and BF16-reference host-memory feasibility note.
- A limitations note for reference runtimes and any non-exact timing/arithmetic boundaries.

## Suggested commands

```text
python tools/fetch_model.py --lock models/gemma4-26b-qat-bf16.lock.json
```
```text
python tools/fetch_model.py --lock models/gemma4-26b-unsloth-nvfp4.lock.json --verify-only
```
```text
python tools/capture_gemma4_26b_goldens.py --manifest benchmarks/corpora/gemma4_26b/dev.json
```

## Risks to watch in this milestone

- QAT and ordinary checkpoints may differ in tokenizer/template metadata as well as weights.
- Reference runtimes may silently select different quantization kernels.
- Golden files can become too large; retain compact selected rows and hashed full outputs.

## Forbidden shortcuts

- Do not infer full revisions from short tree hashes.
- Do not store gated full model weights in git.
- Do not use chat impressions as goldens.
- Do not let different tokenizer revisions enter comparison variants.
- Do not run final quality selection yet.

## Exit criteria

- [ ] Every model and software source has a full immutable revision.
- [ ] Required disk/cache bytes and BF16 reference-memory needs are measured, and an adequate local, staged or external resource plan is recorded.
- [ ] Every required file has size and SHA-256.
- [ ] Reference tokenization is identical across compared model formats.
- [ ] At least layer 0, one later local layer, one global layer and final logits have QAT BF16 goldens.
- [ ] Router goldens include all 128 probabilities plus selected IDs and weights.
- [ ] Calibration and final test sets are disjoint.
- [ ] No quantizer tuning has used the final test set.

## Downstream milestones unblocked

- M02
- M03
- M10

## Codex execution prompt

```text
You are implementing M01: Immutable source locks and golden evidence in Danmoreng/gem16.

Read repository AGENTS.md, docs/DECISIONS.md, docs/CORRECTNESS.md, docs/BENCHMARKING.md, the package master plan, and this milestone. Work only on M01. Inspect the actual current tree before editing and write a drift note if it differs from the anchored commit.

Implement the narrowest change that satisfies every exit criterion. Add tests before or with implementation. Do not add silent fallbacks, token-loop allocations, CPU weight offload, or unrelated refactors. Preserve all existing Gemma 4 12B exact gates.

At completion, report files changed, tests and exact results, generated evidence paths, memory/performance deltas where relevant, unresolved risks, and whether each M01 exit criterion passed. Stop before starting the next milestone.
```

## imp reference amendment

Add an immutable lock for `kekzl/imp@a392904d4216388828d0d56317de046f4ca49627`, its MIT license and the selected files in `references/imp/IMP_SOURCE_MAP.md`. Goldens must record whether they come from official Transformers, imp, another runtime or the local oracle. Do not use imp `main` or a 5090 benchmark result as a golden.
