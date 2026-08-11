# M07 — Tied embedding and output-head format experiment

## Objective

Implement exact candidate encoders and reference operators for the tied 262,144 × 2,816 embedding/output matrix, then select Q4_0 or NVFP4 for the first production artifact using quality, speed and memory evidence.

## Why this milestone exists

Leaving the tied matrix in BF16 costs about 1,408 MiB and breaks the 16 GB target. Q4_0 preserves Google's QAT-targeted format; NVFP4 enables the native Blackwell path. The correct choice cannot be made from format names alone.

## Prerequisites

- M04 compiler scaffold
- M01 official Q4_0 source lock
- M05/M06 codec conventions fixed
- Shared native converter architecture and the M06 native NVFP4 extension are available

## Current status and binding architecture

M07 is dependency-gated and has not started. Both candidate encoders are planned as native extensions of the shared
C++ converter data plane; Python may prepare plans and small reference fixtures but must not perform the promoted
738,197,504-element head conversion (about 1.48 GB of BF16 payload). Read the binding [native converter architecture](../specs/NATIVE_CONVERTER_ARCHITECTURE.md)
and the version-scoped [llama.cpp converter research](../../../evidence/gemma4_26b/m05-llama-cpp-converter-research-2026-08-11.md)
before editing. The pinned llama.cpp Q4_0 implementation is an audit/reference source, not a GGUF runtime dependency or
permission to create a BF16 intermediate artifact.

## Repository areas to inspect first

- `src/cuda/output_head.h`
- `src/cuda/output_head.cu`
- `src/model/tokenizer.cpp`
- `src/cuda/engine/target_model.h`
- `src/cuda/engine/target_model.cu`
- `src/numeric/nvfp4.cpp`
- `third_party or pinned llama.cpp Q4_0 reference`

## Suggested additions or boundaries

- `src/compiler/q4_0_batch_encoder.{h,cpp}` shared native converter extension
- `src/compiler/nvfp4_batch_encoder.{h,cpp}` from M06
- `src/cuda/embedding/q4_0_reference.cu`
- `src/cuda/embedding/nvfp4_reference.cu`
- `tests/unit/q4_0_batch_encoder_test.cpp`
- `tests/cuda/embedding_head_test.cu`
- `docs/plans/gemma4-26b/specs/NATIVE_CONVERTER_ARCHITECTURE.md`

## Implementation sequence

1. Implement the exact Q4_0 block contract in the shared native C++ data plane: 32 weights, one FP16 scale, 16 packed bytes, canonical nibble order and deterministic rounding.
2. Implement an NVFP4 encoder for the same matrix using the M06 native codec and a dedicated embedding/head role. Both candidates must stream directly from locked Safetensors to final-format output; neither may create a BF16 intermediate artifact.
3. Extract the official Google Q4_0 embedding tensor from the pinned GGUF into an audit-only test fixture without making GGUF a runtime dependency.
4. Compare project Q4_0 bytes against the official tensor where tensor mapping and source revision permit exact comparison; otherwise compare dequantized values and document why byte identity is not expected. Treat pinned llama.cpp Q4_0 code as a reference/golden source only; copied code requires the MIT notice, exact provenance and frozen differential tests.
5. Implement CPU lookup and output projection references for BF16, Q4_0 and NVFP4.
6. Implement minimal CUDA prototypes: Q4_0 lookup, Q4_0×BF16 candidate reduction, NVFP4 lookup, and native NVFP4 candidate reduction. Keep diagnostic full-logit mode separate.
7. Use one allocation for input embedding and output head in every profile; verify the tied alias explicitly.
8. Measure isolated T=1 head latency, T=3/T=5 verifier latency, lookup cost, full-logit diagnostic cost, and resident bytes.
9. Run teacher-forced logit and token-selection comparison against QAT BF16 and official Q4_0.
10. Record a provisional format decision. The final decision can be reopened in M19 only with held-out model evidence.

## Required tests

- Exact Q4_0 reference quantizer fixtures from the pinned llama.cpp implementation.
- Q4_0 and NVFP4 lookup rows match CPU dequantization for random and edge token IDs.
- Fused softcap/candidate/argmax preserves lowest-token tie behavior and suppression.
- Batch head paths cover 1, 3 and 5 rows.
- The same allocation pointer is bound for embedding and output head.
- No complete dequantized 1.48 GB matrix is materialized on host or device.
- Quantized-head teacher-forcing report includes KL, top-1/top-10 agreement and margin sensitivity.

## Evidence and documentation outputs

- `artifacts/m07/head-format-quality.json`
- `artifacts/m07/head-format-performance.json`
- `artifacts/m07/head-format-memory.json`
- `docs/DECISIONS.md` entry selecting the provisional first profile
- Disassembly and Nsight traces for both CUDA candidates.

## Suggested commands

```text
python3 tools/compile_gemma4_26b.py plan --source-lock models/gemma4-26b-qat-bf16.lock.json --profile embedding-head-candidates-v1 --head-format q4_0 --compiler-manifest benchmarks/goldens/gemma4_26b/head/qat-q4_0-compiler-plan.json --max-host-memory <bytes> --staging-bytes <bytes>
```
```text
python3 tools/compile_gemma4_26b.py compile --source-lock models/gemma4-26b-qat-bf16.lock.json --profile embedding-head-candidates-v1 --head-format nvfp4 --compiler-manifest benchmarks/goldens/gemma4_26b/head/qat-nvfp4-compiler-plan.json --native-encoder <gem16-checkpoint-compiler> --threads <N> --output build/models/qat-head-nvfp4 --max-host-memory <bytes> --staging-bytes <bytes>
```
```text
python3 tools/benchmark_gemma4_26b_head.py --formats bf16,q4_0,nvfp4 --warmups 3 --repetitions 10 --output artifacts/m07/head-format-performance.json
```

These compiler commands describe a future action-first interface and are not runnable M07 commands today. Earlier stage-style examples are retired.

## Risks to watch in this milestone

- The output head is highly sensitive because the vocabulary is large and many logits have small margins.
- A fast NVFP4 W4A4 head may lose tokens because activation quantization adds error beyond weight quantization.
- A Q4_0 W4A16 kernel may be memory-bandwidth competitive at T=1 despite lacking native FP4 MMA.
- Official GGUF tensor order or metadata may differ from the project artifact.

## Forbidden shortcuts

- Selecting NVFP4 solely because it uses Tensor Cores.
- Reusing llama.cpp's GGUF NVFP4 block layout or scale convention as if it were Gem16's contract.
- Selecting Q4_0 solely because Google trained for Q4_0.
- Keeping both full head formats resident in production.
- Using an approximate vocabulary subset or skipping the softcap.
- Duplicating the tied matrix as a separate LM head.

## Exit criteria

- [ ] Both Q4_0 and NVFP4 formats have exact host codecs and usable reference operators.
- [ ] Memory is approximately 396 MiB plus bounded metadata for either candidate.
- [ ] Isolated quality and performance reports exist under identical inputs.
- [ ] One provisional profile is selected with an explicit rollback path.
- [ ] The alternate profile remains buildable for M19 comparison.

## Downstream milestones unblocked

- M08 artifact assembly
- M16 production head kernel
- M19 held-out quality selection

## Codex execution prompt

```text
You are implementing M07: Tied embedding and output-head format experiment in Danmoreng/gem16.

Read repository AGENTS.md, docs/DECISIONS.md, docs/CORRECTNESS.md, docs/BENCHMARKING.md, the package master plan, `../specs/NATIVE_CONVERTER_ARCHITECTURE.md`, the version-scoped llama.cpp research evidence, and this milestone. Work only on M07. Inspect the actual current tree before editing and write a drift note if it differs from the anchored commit.

Implement the narrowest change that satisfies every exit criterion. Add tests before or with implementation. Do not add silent fallbacks, token-loop allocations, CPU weight offload, or unrelated refactors. Preserve all existing Gemma 4 12B exact gates.

At completion, report files changed, tests and exact results, generated evidence paths, memory/performance deltas where relevant, unresolved risks, and whether each M07 exit criterion passed. Stop before starting the next milestone.
```
