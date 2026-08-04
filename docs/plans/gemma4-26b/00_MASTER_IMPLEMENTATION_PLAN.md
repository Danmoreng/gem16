# Master implementation plan

## Mission

Extend `gem16` with a production-quality, text-only Gemma 4 26B A4B path for a single approximately 16 GB Blackwell GPU without weakening the existing Gemma 4 12B Unified path.

The work is successful only when the resulting system is simultaneously:

1. numerically understood;
2. reproducibly compiled from immutable model sources;
3. fully resident in GPU memory during generation;
4. deterministic under deterministic settings;
5. fast in batch-one prefill and decode;
6. fairly compared against direct Unsloth NVFP4 and official Google Q4_0 references;
7. usable through the existing CLI/server product surface;
8. documented so another agent can reproduce every decision.

## Production hypothesis

The primary candidate is a derived text-only checkpoint compiled from Google's unquantized QAT BF16 model:

| Tensor family | First production candidate | Reason |
|---|---|---|
| Routed expert gate/up/down | NVFP4 W4A4 | Dominant weight volume and native Blackwell block-scaled Tensor Core path |
| Always-active dense/shared MLP | NVFP4 W4A4 | Same fast path and approximately 287 MiB resident |
| Attention Q/K/V/O | FP8 W8A8 | Reuse current gem16 arithmetic and native Blackwell path |
| Router projection, router scales | BF16 | Routing discontinuities make this a poor first quantization target |
| Norms and scalar controls | BF16/F32 as source | Tiny memory footprint and quality sensitivity |
| Tied embedding/output head | Q4_0 quality candidate versus NVFP4 speed candidate | Same approximate storage, different quality/performance behavior |
| KV cache | checkpoint-compatible FP8 | Required for 32K/64K within the remaining budget |
| Vision tower and projection | omitted from compiled artifact and runtime residency | Not compatible with the first 16 GB budget |
| MTP assistant | disabled | Must not confound base arithmetic or memory qualification |

This is a hypothesis, not a pre-approved quality result. Milestones M18 and M19 decide whether the QAT-derived candidate is actually better than, equal to or worse than the Unsloth conversion.

## Program stages

### Stage A — Governance and immutable evidence

**Milestones:** M00–M03

Outcomes:

- a reviewed 26B contract for project-built derived checkpoints;
- exact source locks for QAT BF16, ordinary BF16, Unsloth NVFP4 and Google Q4_0;
- a repository-drift report;
- a model-traits design that preserves 12B compile-time specialization;
- complete tensor inventories and naming evidence;
- reference activations, router outputs, expert outputs, logits and token streams.

No CUDA performance work is allowed before this stage closes.

### Stage B — Deterministic checkpoint compiler

**Milestones:** M04–M08

Outcomes:

- a streaming compiler with bounded host memory;
- exact and tested FP8 and NVFP4 encoders;
- a reproducible Q4_0 embedding/head encoder or an explicit decision not to use it;
- a standard Safetensors-based derived checkpoint;
- complete provenance, source hashes, compiler hash and per-tensor hashes;
- a loader that accepts the derived artifact without retaining duplicate device layouts.

The compiler must also be able to quantize the ordinary non-QAT BF16 model. This control is essential for separating "our quantizer differs from Unsloth" from "QAT weights differ from ordinary weights."

### Stage C — Runtime correctness path

**Milestones:** M09–M13

Outcomes:

- text-only residency and context-aware memory planning;
- a CPU MoE oracle;
- a slow CUDA MoE reference path;
- correct 26B local/global attention and FP8 KV;
- a complete end-to-end 26B forward path that is intentionally not yet performance-qualified.

The purpose of this stage is to create trustworthy differential tests. It is acceptable for the first full model to be slow. It is not acceptable for it to be opaque.

### Stage C2 — Early converter and quality attribution gate

**Milestone:** M18, executed after M13

Outcomes:

- converter/source/head effects are separated on the correctness runtime;
- a development-corpus quality screen rejects catastrophic or unexplained NVFP4 loss;
- candidate profiles and held-out thresholds are frozen before native kernel optimization;
- M14–M17 proceed only when the explicit quality kill gate passes.

### Stage D — Native Blackwell performance path

**Milestones:** M14–M17, after M18 passes

Outcomes:

- GPU-resident deterministic top-8 routing;
- native NVFP4 expert decode;
- grouped, bounded-workspace NVFP4 prefill;
- quantized embedding and output-head kernels;
- whole-model CUDA Graph decode;
- no token-loop allocations;
- one resident weight representation.

Hot kernels remain model-shape-specialized. Generic model dispatch occurs once at initialization and may not add per-layer or per-token virtual dispatch.

### Stage E — Final comparison and qualification

**Milestones:** M19–M21, using the M18 attribution evidence

Outcomes:

- converter A/B evidence;
- model-quality evaluation against QAT BF16, ordinary BF16, Unsloth NVFP4 and official Q4_0;
- controlled batch-one prefill/decode benchmarks;
- 32K required and 64K target context qualification;
- power, clocks, thermals, VRAM and native instruction evidence;
- promotion or rejection of each candidate format.

A faster candidate with worse quality than the accepted threshold is rejected. A higher-quality candidate that does not fit the memory budget is not the production profile.

### Stage F — Product and release

**Milestones:** M22–M23

Outcomes:

- CLI/server/Studio model selection;
- immutable model download and verification;
- capability reporting;
- clean failure for unsupported MTP/vision requests;
- release notes, migration notes and rollback;
- retained benchmark and quality artifacts.

### Optional stage G — Later tracks

**Milestones:** M24–M25

- full-model Q4_0 reference/backend;
- QAT-compatible MTP;
- on-demand vision.

These are explicitly outside the critical path.

## Candidate matrix

Every quality and performance report must distinguish these variants:

| ID | Source | Conversion/runtime | Purpose |
|---|---|---|---|
| A | Unsloth published NVFP4 | Direct-load mixed FP8/NVFP4 | External practical NVFP4 baseline |
| B | Ordinary Google IT BF16 | gem16 compiler to the same hybrid recipe | Quantizer control |
| C | Google QAT BF16 | gem16 compiler to FP8/NVFP4 plus selected head | Primary hypothesis |
| D | Official Google QAT Q4_0 GGUF | llama.cpp or exact Q4_0 reference | QAT-target-format quality reference |
| E | Google QAT BF16 | gem16 production hybrid | Final candidate selected from head experiments |
| F | Google QAT BF16 | gem16 slow BF16/reference operators where feasible | Numerical oracle, not a 16 GB deployment |
| G | NVIDIA/ModelOpt Gemma 4 26B NVFP4 | pinned imp or another validated loader | Negative/control quantization-recipe arm; not a production default |
| H | UD-Q4_K_M Gemma 4 26B | pinned imp/llama.cpp | External quality/speed context; not the official QAT reference |

Do not compare only A against C. B is necessary to identify whether differences come from QAT or from a different quantizer.

## Promotion gates

### Gate 1: source and compiler integrity

Required:

- full immutable commit SHAs;
- file sizes and SHA-256 hashes;
- deterministic compiler output on the reference toolchain;
- complete tensor count and byte accounting;
- no vision tensor in the production artifact;
- no unexplained tensor rename, transpose or alias.

### Gate 2: operator correctness

Required:

- CPU dequantization oracles;
- exact byte fixtures for packing and scale encodings;
- real-shape FP8 and NVFP4 operator comparisons;
- deterministic top-8 routing with explicit tie behavior;
- exact residual/norm ordering from the reference implementation;
- separate local and global attention tests;
- output-head softcap and lowest-token tie break.

### Gate 3: full-model numerical evidence

Required:

- teacher-forced logits and state captures;
- per-layer residual drift;
- router probability and selected-expert drift;
- greedy sequence determinism;
- no NaN/Inf;
- quality non-inferiority against the approved reference envelope.

### Gate 4: memory feasibility

Required:

- a preliminary M03 synthetic 32K admission probe before compiler work, using runtime-visible rather than nominal VRAM;
- immutable weight arena target at or below 14,100 MiB;
- no duplicate embedding/head copy;
- no persistent source-order plus runtime-order copy;
- measured 32K process peak with at least 700 MiB directly reported free-device margin against the approximately 15,881 MiB CUDA-visible reference capacity;
- bounded prefill workspace;
- no token-loop allocation;
- exact allocator accounting reconciled with `cudaMemGetInfo` and sampled process telemetry.

If the immutable weight arena exceeds 14,300 MiB, stop and revisit format selection before kernel optimization.

### Gate 5: performance

Required for a performance release:

- native SM120/SM120a NVFP4 instructions verified in the selected kernels;
- median prefill and decode both faster than the accepted Q4_0 baseline on the same machine;
- three warm-ups and ten retained runs;
- non-overlapping confidence intervals for headline wins or appropriately cautious wording;
- no CPU offload, prompt-cache asymmetry or semantic shortcut;
- deterministic outputs for the deterministic benchmark.

Matching or beating direct vLLM NVFP4 is a stretch target, not a reason to falsify timing boundaries.

### Gate 6: product readiness

Required:

- source/compiled lock download path;
- clear model profile naming;
- 12B regressions green;
- clean unsupported-feature errors;
- server session admission respects actual 26B slot size;
- raw evidence retained and release rollback documented.

## Milestone execution discipline

Each milestone must follow this sequence:

1. read repository `AGENTS.md`, this master plan and the milestone file;
2. inspect the actual current source before editing;
3. write or update the decision/experiment record;
4. add failing tests or goldens first where practical;
5. implement the narrowest possible change;
6. run host tests, then CUDA operator tests, then model tests;
7. collect memory or performance evidence only after correctness passes;
8. update documentation and ledger;
9. stop at the milestone exit gate.

A milestone PR may not contain unrelated UI work, opportunistic refactors or multiple arithmetic changes that cannot be isolated in an A/B test.

## First usable checkpoints along the way

The plan intentionally creates several usable but differently qualified points:

- **M08:** compiled artifact can be inspected and loaded;
- **M13:** complete slow 26B text inference for correctness;
- **M17:** optimized batch-one path;
- **M19:** quality-qualified candidate;
- **M21:** 32K/64K context-qualified candidate;
- **M23:** product release candidate.

The project owner can stop after any point without pretending later gates have passed.

## Final expected artifact set in the repository

At completion, expect additions broadly like:

```text
models/
  gemma4-26b-qat-bf16.lock.json
  gemma4-26b-base-bf16.lock.json
  gemma4-26b-unsloth-nvfp4.lock.json
  gemma4-26b-qat-q4_0.lock.json
  gemma4-26b-gem16-hybrid.lock.json

tools/
  compile_gemma4_26b.py
  compare_quantized_checkpoints.py
  capture_gemma4_26b_goldens.py
  evaluate_gemma4_26b_quality.py

src/model/
  model_variant.*
  gemma4_26b_contract.*

src/cuda/moe/
  reference.*
  router.*
  decode_sm120.*
  prefill_sm120.*
  reduction.*

src/cuda/embedding/
  q4_0.*
  nvfp4.*

docs/
  GEMMA4_26B.md
  GEMMA4_26B_CHECKPOINT.md
  GEMMA4_26B_MEMORY.md
  GEMMA4_26B_QUALITY.md
  GEMMA4_26B_BENCHMARKING.md
```

Exact names may change after repository inspection, but responsibilities may not disappear.

## imp reference amendment

A pinned imp reference lane is now part of Stage A and Stage E. It supplies independent MoE failure-mode evidence, producer-specific NVFP4 scale contracts, potential grouped-small-M kernel ideas and stronger engineering gates. See [`13_IMP_REFERENCE_INTEGRATION.md`](13_IMP_REFERENCE_INTEGRATION.md).

The primary checkpoint strategy is unchanged. Imp's documented ModelOpt Gemma 4 quality deficit makes candidate G a required negative/control arm and raises the burden of proof for any NVFP4 promotion.
