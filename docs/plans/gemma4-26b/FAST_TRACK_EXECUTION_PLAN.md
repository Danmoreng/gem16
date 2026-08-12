# Fast-track execution plan from M06

## Goal

Reach a real, directly loadable 26B model and a first correct token as early as possible, then optimize and qualify it without losing the evidence needed for a safe 16 GB release. MTP and maximum safe single-user context are program goals, not mixed into the first artifact gate.

## Execution waves

### F1 — Artifact and residency

Critical lane: `M06 → M07 → M08 → M09`

Parallel prework:

- M10 phase A: BF16 MoE semantics and fixtures;
- M12 phase A: layer traits, attention/RoPE/KV fixtures;
- M09 phase A: checked formulas and one-slot admission tests;
- M13/M19/M20/M21 harness scaffolding;
- M25 phase A: compatible-assistant inventory and memory feasibility.

Exit checkpoint: a complete QAT-derived text artifact loads directly, occupies one final weight layout and passes real 32K admission.

### F2 — First complete correct model

Critical integration: `M10 → M11`, M12 phase B, then M13.

M10 and M12 are independent until their runtime adapters meet in M11/M13. The first full path may be slow. M13 is the single early quality gate.

Exit checkpoint: a deterministic prompt and generation complete with correct MoE, attention, KV, tokenizer and head behavior.

### F3 — Native runtime

After M13 passes, M14, M15 and M16 may run in parallel. M17 integrates each accepted operator as it becomes available rather than waiting for all three to finish.

- M14: batch-one native MoE decode;
- M15: grouped bounded-workspace MoE prefill;
- M16: production tied head, initially NVFP4 and T=1;
- M17: rolling whole-model integration and graph capture.

M18 is invoked only for diagnosis or attribution. It is not between M13 and M17 on the normal path.

Exit checkpoint: the complete optimized path is deterministic, all-resident and free of token-loop allocations.

### F4 — Qualification and maximum base context

Freeze one M17 artifact hash. M19, M20 and M21 may collect evidence in parallel on that exact artifact:

- M19: held-out quality and final format acceptance;
- M20: controlled prefill/decode performance;
- M21: 32K confirmation, 64K attempt and measured maximum safe base context.

M22 integrates CLI/server in parallel once profile metadata is frozen. Studio is a nonblocking subtask. M23 verifies and freezes the already collected evidence; it does not blindly rerun unchanged work.

Exit checkpoint: a base 26B release candidate with a documented default context and measured maximum.

### F5 — MTP final target

M25 turns the frozen base target into the final single-user profile:

1. lock a compatible 26B assistant source or explicitly record the asset blocker;
2. compile/quantize the assistant if BF16 residency is not viable;
3. reuse Target KV where the assistant contract permits and avoid a second long-context cache;
4. implement multi-row Target verification inside M25, not in M07;
5. preserve ordinary Target output exactly under matched controls;
6. qualify base-versus-MTP speed, acceptance, memory and context;
7. pass the 32K MTP minimum, attempt 64K and publish separate `base_max_context` and `mtp_max_context` values.

Vision remains outside this plan.

## Milestone classification

| Milestone | Classification | Normal-path role |
|---|---|---|
| M06 | critical | native expert compiler and QAT full conversion |
| M07 | critical-lite | provisional NVFP4 tied head/reference path |
| M08 | critical | complete artifact and direct loader |
| M09 | critical | one-slot 32K residency; 64K/max feasibility |
| M10 | parallel/critical | CPU MoE semantic oracle |
| M11 | critical | CUDA correctness MoE |
| M12 | parallel/critical | attention, RoPE and KV |
| M13 | critical gate | complete slow model and sole early quality gate |
| M14 | parallel | optimized decode |
| M15 | parallel | optimized prefill |
| M16 | parallel/conditional | optimized head; format revision only on evidence |
| M17 | rolling integration | optimized whole model |
| M18 | conditional | attribution and failure diagnosis |
| M19 | qualification | held-out quality |
| M20 | qualification | performance |
| M21 | qualification | 32K, 64K and maximum safe base context |
| M22 | product | CLI/server; Studio nonblocking |
| M23 | checkpoint | base evidence freeze and rollback |
| M24 | optional | internal Q4_0 backend/reference |
| M25 | final target | 26B MTP and maximum safe MTP context |

## Vertical checkpoints

The owner should judge progress by these outcomes rather than document count:

1. **Artifact:** M08 loads.
2. **Fit:** M09 passes 32K on the real GPU.
3. **Correct token:** M13 passes.
4. **Fast token:** M17 passes.
5. **Qualified base:** M19–M23 pass.
6. **Final target:** M25 passes with exact MTP and measured MTP context.

## Deferred work

The following are not on the base critical path:

- full ordinary-BF16 model conversion and full causal attribution unless M18 is triggered;
- an internal full-model Q4_0 implementation;
- T=3/T=5 output-head work before M25;
- positive multi-slot 26B admission;
- 128K/256K advertising without measured safety margin;
- Studio polish;
- vision.
