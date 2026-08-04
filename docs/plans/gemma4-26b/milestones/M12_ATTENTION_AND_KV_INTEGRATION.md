# M12 — 26B attention, RoPE, KV sharing and cache integration

## Objective

Adapt the existing FP8 attention and hybrid KV implementation to the validated 26B model traits, including 25 local layers, 5 global layers, two global KV heads and any declared cross-layer KV sharing.

## Why this milestone exists

The 26B architecture differs from the 12B variant in layer count, hidden size, global KV heads and potentially shared-KV behavior. Reusing kernels without an explicit semantic audit risks incorrect cache ownership or memory formulas.

## Prerequisites

- M02 traits
- M03 attention tensor inventory
- M05 FP8 compiler
- M09 memory plan

## Repository areas to inspect first

- `src/cuda/attention/decode_sm120.cu`
- `src/cuda/attention/prefill_local_sm120.cu`
- `src/cuda/attention/prefill_global_sm120.cu`
- `src/cuda/attention/reference.cu`
- `src/cuda/kv_cache/reference.cu`
- `src/cuda/rope/reference.cu`
- `src/cuda/engine/inference_engine.cu`
- `src/cuda/engine/target_model.cu`

## Suggested additions or boundaries

- `src/cuda/attention/gemma4_26b_traits.h`
- `tests/cuda/gemma4_26b_attention_test.cu`
- `docs/GEMMA4_26B_ATTENTION.md`

## Implementation sequence

1. Derive every attention shape from model traits and per-layer config; specialize kernels at initialization rather than embedding 12B constants.
2. Validate the exact layer type pattern and any `num_kv_shared_layers` behavior against the pinned config and trusted reference code.
3. For layers that omit V projection under `attention_k_eq_v`, reuse the raw projection result only; apply distinct K normalization/RoPE and scale-free V normalization and store separate final K/V states.
4. Implement local D256 attention for 8 KV heads and global D512 attention for 2 KV heads, with 16 query heads.
5. Preserve local RoPE and global proportional/partial RoPE semantics from the checkpoint config, not inherited 12B constants.
6. Bind FP8 Q/K/V/O weights emitted in M05 and retain exact BF16 rounding boundaries required by the model.
7. Implement local circular cache and global contiguous cache for separate FP8 K/V.
8. If cross-layer KV sharing exists, represent ownership and consumers explicitly so shared states are neither duplicated unnecessarily nor overwritten by a consumer layer.
9. Add reference and native comparisons for prompt boundaries, ring wrap and long global extents.
10. Update memory counters from actual cache bindings and validate them against M09.

## Required tests

- Layer-traits table test for all 30 layers.
- Local and global Q/K/V/O shape tests.
- Global missing-V projection test proves distinct final K and V bytes.
- RoPE fixtures at positions 0, 1, 1,023, 1,024, 32K and near maximum context.
- Local ring wrap and global append/read tests.
- Shared-KV producer/consumer tests if declared by config.
- Real-layer reference/native output comparisons for a local and global layer.
- Cache byte totals match M09 at 8K, 32K and 64K.

## Evidence and documentation outputs

- `docs/GEMMA4_26B_ATTENTION.md`
- `artifacts/m12/layer-attention-table.json`
- `artifacts/m12/attention-comparison.json`
- `artifacts/m12/cache-layout.json`

## Suggested commands

```text
build/blackwell-debug/bin/gem16-cuda-tests --filter gemma4_26b_attention
```
```text
python tools/validate_gemma4_26b_attention.py --model "$GEM16_26B" --output artifacts/m12/attention-comparison.json
```

## Risks to watch in this milestone

- Current 12B code derives global layers from `index % 6 == 5`; this must not substitute for config validation.
- Physical K/V aliasing is invalid even when raw K projection supplies V.
- Cross-layer KV sharing can change both tensor inventory and cache ownership.
- D512 global attention may require new kernel geometry for two KV heads.

## Forbidden shortcuts

- Assuming all 26B layers own independent K/V projections without inventory evidence.
- Sharing physical final K and V buffers.
- Using the 12B hidden/head constants in output or workspace indexing.
- Changing attention arithmetic to make a reference comparison easier.
- Advertising maximum context from allocation-only tests.

## Exit criteria

- [ ] All 30 layers have validated attention traits and bindings.
- [ ] Local/global reference tests and ring/global cache tests pass.
- [ ] Separate K/V and any cross-layer sharing semantics are correct.
- [ ] Cache bytes reconcile with the memory planner.
- [ ] No 12B attention or long-context regression is introduced.

## Downstream milestones unblocked

- M13 full-model correctness
- M17 optimized integration
- M21 long-context qualification

## Codex execution prompt

```text
You are implementing M12: 26B attention, RoPE, KV sharing and cache integration in Danmoreng/gem16.

Read repository AGENTS.md, docs/DECISIONS.md, docs/CORRECTNESS.md, docs/BENCHMARKING.md, the package master plan, and this milestone. Work only on M12. Inspect the actual current tree before editing and write a drift note if it differs from the anchored commit.

Implement the narrowest change that satisfies every exit criterion. Add tests before or with implementation. Do not add silent fallbacks, token-loop allocations, CPU weight offload, or unrelated refactors. Preserve all existing Gemma 4 12B exact gates.

At completion, report files changed, tests and exact results, generated evidence paths, memory/performance deltas where relevant, unresolved risks, and whether each M12 exit criterion passed. Stop before starting the next milestone.
```
