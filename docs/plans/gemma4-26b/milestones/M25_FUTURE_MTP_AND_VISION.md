# M25 — Future MTP and on-demand vision tracks

## Objective

Define, but do not prematurely merge, future work for 26B-compatible speculative decoding and vision support after the text-only base release is stable.

## Why this milestone exists

MTP and vision both consume scarce VRAM and add independent arithmetic, quality and product risks. They must not confound the first 26B residency and performance result.

## Prerequisites

- M23 release complete
- Fresh memory and profiling evidence
- Compatible model assets verified

## Repository areas to inspect first

- `docs/MTP.md`
- `src/cuda/mtp/`
- `docs/VISION.md`
- `src/cuda/vision/`
- `src/model/image.cpp`
- `src/cuda/engine/inference_engine.cu`

## Suggested additions or boundaries

- `docs/GEMMA4_26B_MTP_PLAN.md`
- `docs/GEMMA4_26B_VISION_PLAN.md`

## Implementation sequence

1. Verify whether Google publishes an assistant compatible with the exact 26B target architecture and QAT-derived artifact. Do not reuse the 12B assistant.
2. Model assistant weights, workspace, graph and acceptance memory against the measured 26B headroom.
3. If no compatible assistant exists, evaluate n-gram or prompt-lookup proposals only as separate exact-verification research.
4. Require target-verified output identity for any MTP path and count only accepted target tokens.
5. For vision, inventory the exact omitted source tensors and decide between on-demand GPU residency, temporary CPU/GPU swap or a larger-memory profile.
6. Never evict text weights per image in the primary performance profile without explicit user-visible mode and benchmarks.
7. Define separate quality, memory, context and product gates for each track.
8. Keep both tracks off by default until their own release qualification.

## Required tests

- Assistant-target architecture and tokenizer compatibility.
- Exact target verification and transactional KV commit.
- Memory admission with assistant or vision active.
- Image placeholder and attention-mask correctness if vision proceeds.
- Base text-only performance and context regression.
- Explicit unsupported-feature errors while tracks remain disabled.

## Evidence and documentation outputs

- `docs/GEMMA4_26B_MTP_PLAN.md`
- `docs/GEMMA4_26B_VISION_PLAN.md`
- Feasibility memory tables and asset locks
- Separate decision records before implementation

## Risks to watch in this milestone

- 26B base headroom may be insufficient for an assistant.
- Vision tensors and workspaces can exceed the remaining 16 GB budget.
- An assistant trained for a different target can reduce acceptance or violate semantics.
- Swapping weights can destroy latency and complicate server concurrency.

## Forbidden shortcuts

- Using the 12B MTP assistant with the 26B target.
- Counting unverified proposals as output.
- Silently evicting text weights or reducing context.
- Advertising multimodal support from architecture metadata alone.
- Starting these tracks before the base release evidence is frozen.

## Exit criteria

- [ ] Feasibility is documented with exact compatible assets and measured memory.
- [ ] Each track has independent gates and a decision record.
- [ ] Base 26B text-only behavior remains unchanged.
- [ ] No MTP or vision feature is enabled by default without separate qualification.

## Downstream milestones unblocked

- Optional future release programs

## Codex execution prompt

```text
You are implementing M25: Future MTP and on-demand vision tracks in Danmoreng/gem16.

Read repository AGENTS.md, docs/DECISIONS.md, docs/CORRECTNESS.md, docs/BENCHMARKING.md, the package master plan, and this milestone. Work only on M25. Inspect the actual current tree before editing and write a drift note if it differs from the anchored commit.

Implement the narrowest change that satisfies every exit criterion. Add tests before or with implementation. Do not add silent fallbacks, token-loop allocations, CPU weight offload, or unrelated refactors. Preserve all existing Gemma 4 12B exact gates.

At completion, report files changed, tests and exact results, generated evidence paths, memory/performance deltas where relevant, unresolved risks, and whether each M25 exit criterion passed. Stop before starting the next milestone.
```
