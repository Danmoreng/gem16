# Milestone dependency graph

## Required critical path

```text
M00 Policy and track bootstrap
 └─ M01 Source locks and golden evidence
     ├─ M02 Model configuration and variant traits
     │   └─ M03 26B manifest and tensor inventory
     │       ├─ M04 Compiler scaffold
     │       │   └─ M05 FP8 attention compiler and native converter seed
     │       │       └─ M06 NVFP4 expert compiler and shared native data-plane extension
     │       │           └─ M07 Embedding/head format experiment
     │       │               └─ M08 Derived checkpoint and loader
     │       │                   └─ M09 Memory planner and text-only residency
     │       └─ M10 CPU MoE oracle
     │           └─ M11 CUDA MoE reference
     │               └─ M12 26B attention and KV
     │                   └─ M13 Slow full-model reference integration
     │                       └─ M18 Converter A/B and preliminary quality gate
     │                           ├─ M14 Native NVFP4 MoE decode
     │                           ├─ M15 Grouped NVFP4 MoE prefill
     │                           └─ M16 Quantized embedding/output head
     │                               └─ M17 Optimized full-model integration
     │                                   ├─ M19 Model-quality qualification
     │                                   └─ M20 Performance qualification
     │                                       └─ M21 32K/64K context qualification
     │                                           └─ M22 CLI/server/Studio productization
     │                                               └─ M23 Release and rollback
     └─ source evidence also feeds M18 and M19
```

Optional tracks:

```text
M23 ──> M24 Optional full Q4_0 backend
M23 ──> M25 Future QAT MTP and vision
```

## Scheduling within the single feature branch

All M03-M25 implementation is integrated sequentially on `feat/gemma4-26b`; no published parallel milestone
branches are used. After M03, the dependency graph still permits the project owner to choose between independent
next slices:

- M04 compiler scaffolding and M10 CPU MoE oracle may be scheduled in either order.
- M05 FP8 must establish the accepted native converter seed/generalization boundary before M06 NVFP4 work; M05 and M06 may not be implemented in either order.
- M07 head experiments may start only after the M06 native extension and both Q4_0/NVFP4 host oracles exist.
- M12 attention/KV can be selected once its prerequisites are committed, without weakening the M11/M12 gates.
- M14 decode, M15 prefill and M16 head kernels may be scheduled in any evidence-driven order after M13 provides
  stable full-model fixtures and M18 passes the preliminary quality kill gate.

Only one implementation slice is active at a time. Close and commit its contract and evidence before selecting the
next eligible slice. Do not combine incompatible tensor naming or arena schemas in the long-lived branch.

## What must not run in parallel

- M00 policy and compiler implementation;
- tensor inventory and tensor-binding implementation;
- final head-format selection and release memory budgeting;
- quantizer tuning and final quality evaluation on the same held-out corpus;
- arithmetic changes and end-to-end performance promotion;
- 32K/64K qualification before fixed workspace accounting;
- product model download integration before the compiled lock format is stable.

## Gate ownership

| Gate | Owning milestone | Downstream work blocked |
|---|---|---|
| 26B artifact contract reviewed | M00 | all compiler work |
| Sources fully immutable | M01 | all goldens and comparisons |
| Variant contract accepted | M02 | manifest, loader, engine |
| Tensor inventory and preliminary 32K admission exact | M03 | compiler schema, bindings and all compiler work |
| Compiler deterministic | M04 | quantized artifact publication |
| FP8 oracle accepted | M05 | attention compilation |
| NVFP4 oracle accepted | M06 | expert compilation/kernels |
| Head profile selected provisionally | M07 | final byte budget |
| Derived artifact validates | M08 | runtime load |
| Real artifact and final 32K arena plan fit | M09 | full runtime |
| CPU MoE semantics exact | M10 | CUDA MoE |
| CUDA MoE reference exact | M11 | native MoE optimization |
| Attention/KV exact | M12 | full model |
| Full slow model deterministic | M13 | converter A/B and preliminary quality diagnosis |
| Decode kernel exact and faster | M14 | optimized decode |
| Prefill kernel exact and bounded | M15 | optimized prefill |
| Quantized head exact enough | M16 | production full model |
| Optimized full model deterministic | M17 | final quality and performance qualification |
| Quantizer comparison and preliminary quality kill gate pass | M18 | native performance kernels and provenance claims |
| Quality accepted | M19 | performance release wording |
| Performance accepted | M20 | long-context release |
| 32K/64K accepted | M21 | product integration |
| Product surface accepted | M22 | release |
| Release evidence complete | M23 | optional extensions |

## Agent scheduling rule

A coding agent may prepare notes for a blocked milestone, but it may not merge implementation for that milestone. The exit report of each milestone must list exactly which downstream milestones are unblocked.

## Auxiliary imp reference lane

```text
M00
 └─ R-IMP-00 reference/license/source audit
      ├─ feeds M01 source locks
      ├─ feeds M03 producer-aware inventory
      ├─ feeds M06 scale fixtures
      └─ feeds M10 semantic goldens

M13 correctness complete
 └─ R-IMP-15 grouped-kernel study
      ├─ may feed M14 decode ideas
      └─ may feed M15 prefill implementation

M17 optimized runtime
 └─ imp-derived observability/lifecycle patterns
      ├─ actual dispatch record
      ├─ graph demotion reason
      └─ engine relaunch tests

M18/M19/M20
 └─ ModelOpt negative control + imp external context
```

No imp production code may be introduced before M13. The reference audit may proceed in parallel with M01/M02, but its scale and semantic findings must close before M06/M10 exit.
