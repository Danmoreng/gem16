# M03 — 26B tensor manifest and exact inventory

## Objective

Build a variant-specific tensor inventory and validation contract for the 26B BF16, Unsloth NVFP4 and future gem16-compiled artifacts without guessing names or fusion layouts.

## Why this milestone exists

The MoE checkpoint can store expert weights as fused 3D gate/up plus down tensors, individual expert tensors, or quantizer-expanded families. Kernels and compiler schema must be based on observed, locked tensors rather than regex assumptions copied from another model.

## Prerequisites

- M01 source files are locally available and locked.
- M02 model variant classification is merged.

## Repository areas to inspect first

- `src/model/manifest.h`
- `src/model/manifest.cpp`
- `src/model/checkpoint_loader.cpp`
- `include/gem16/types.h`
- `docs/CHECKPOINT_FORMAT.md`
- `gem16-inspect`
- `pinned QAT BF16 and Unsloth tensor inventories`

## Suggested additions or boundaries

- `src/model/gemma4_26b_manifest.cpp`
- `src/model/tensor_role.h`
- `tests/fixtures/gemma4_26b_inventory.json`

## Implementation sequence

1. Generate and review sorted tensor inventories for QAT BF16, ordinary BF16 and Unsloth NVFP4. Record exact names, shapes, storage dtype, logical dtype, quantization companions, shard and byte totals.
2. Classify tensor roles: tied embedding, final norm, attention, shared dense MLP, router norm/scale/projection/per-expert scale, routed experts, layer norms/scalars, vision and metadata-only.
3. Determine whether QAT BF16 expert weights are fused as `[experts, 2*moe_intermediate, hidden]` and `[experts, hidden, moe_intermediate]`; determine how Unsloth serializes quantized experts.
4. Define `TensorRole` and `ResidencyClass` instead of relying only on name prefixes.
5. Implement a 26B source-BF16 inventory validator and a separate compiled-hybrid validator.
6. Validate tied embedding alias semantics and reject a duplicate `lm_head` payload unless the source format requires it and the compiler proves aliasing.
7. Validate every router tensor and exact shape, including learned hidden scale and per-expert scale.
8. Validate that global layers omit or include V exactly as the pinned source dictates; do not generalize from 12B names.
9. Validate vision tensors for source completeness but mark them `kCompileExcludedVision`; the production compiled artifact must contain none.
10. Make manifest summaries report bytes by role, format and residency class.
11. Add a canonical JSON inventory used by the compiler and runtime tests.
12. From the frozen role inventory, calculate conservative compiled bytes and run a synthetic device-admission probe for immutable weights, 32K FP8 K/V, current CUDA context, minimum realistic fixed arenas and a 700 MiB free margin. Use runtime-visible capacity, not nominal board memory.

## Required tests

- Missing one expert, swapped expert dimension, wrong fused gate/up order and duplicate tensor tests.
- Wrong router tensor shape/dtype tests.
- Wrong local/global attention inventory tests.
- Vision exclusion bytes exactly reconcile with source totals.
- Tied embedding duplicate/alias tests.
- Unsloth inventory parses without being accepted as the project-compiled artifact.
- Every manifest byte is assigned to exactly one role and one residency class.
- Synthetic 32K admission passes with at least 700 MiB directly measured free memory, or M04 remains blocked and the format/head/workspace assumptions are revised.

## Evidence and documentation outputs

- Checked-in compact canonical inventory metadata, not model payload.
- Updated `docs/CHECKPOINT_FORMAT.md` 26B section.
- A source-versus-Unsloth schema comparison table.
- Exact byte totals by role for use in M07/M09.
- Preliminary 32K synthetic admission report with CUDA-visible capacity, context delta, named conservative regions and measured free margin.

## Suggested commands

```text
gem16-inspect --model <qat-source> --json > qat-bf16-inventory.json
```
```text
gem16-inspect --model <unsloth-source> --json > unsloth-nvfp4-inventory.json
```

## Risks to watch in this milestone

- Safetensors viewer and actual headers can disagree if a tool normalizes names.
- Fused versus per-expert serialization may affect compiler streaming strategy.
- The approximately 14,014 MiB planning estimate may leave too little runtime-visible space for the 32K/700 MiB product gate.

## Forbidden shortcuts

- Do not encode guessed tensor names in CUDA.
- Do not accept unknown extras with a warning.
- Do not drop router scale or per-expert scale because they are small.
- Do not use a single dense-layer inventory validator for both models.

## Exit criteria

- [ ] No unknown executable tensor remains.
- [ ] All 30 layers and 128 experts per layer are accounted for.
- [ ] Source and compiled validators are separate and strict.
- [ ] Vision bytes are explicitly classified, not accidentally skipped.
- [ ] Manifest totals reconcile exactly with Safetensors headers.
- [ ] M04 can consume a frozen role mapping.
- [ ] Preliminary 32K synthetic admission passes the 700 MiB direct-free-memory gate; otherwise M04 remains blocked.

## Downstream milestones unblocked

- M04
- M10

## Codex execution prompt

```text
You are implementing M03: 26B tensor manifest and exact inventory in Danmoreng/gem16.

Read repository AGENTS.md, docs/DECISIONS.md, docs/CORRECTNESS.md, docs/BENCHMARKING.md, the package master plan, and this milestone. Work only on M03. Inspect the actual current tree before editing and write a drift note if it differs from the anchored commit.

Implement the narrowest change that satisfies every exit criterion. Add tests before or with implementation. Do not add silent fallbacks, token-loop allocations, CPU weight offload, or unrelated refactors. Preserve all existing Gemma 4 12B exact gates.

At completion, report files changed, tests and exact results, generated evidence paths, memory/performance deltas where relevant, unresolved risks, and whether each M03 exit criterion passed. Stop before starting the next milestone.
```

## imp reference amendment

Each quantized tensor family must record `quantization_producer`, local-scale dtype/vector size, global-scale role (`multiplier` or `divisor`), activation-scale role and final GPU layout. Reject an inventory that identifies “NVFP4” without identifying the producer semantics. Include explicit text-only skip evidence for vision and MTP tensors.
