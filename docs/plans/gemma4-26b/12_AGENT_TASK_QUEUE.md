# Agent task queue

## Queue policy

Only tasks marked `READY` may be implemented. The repository owner changes status after reviewing the preceding milestone.

| Order | Milestone | Initial status | Primary output |
|---:|---|---|---|
| 1 | M00 | PASSED | accepted policy and track skeleton |
| 2 | M01 | PASSED | immutable locks and golden suite |
| 3 | M02 | PASSED | static model variant contract |
| 4 | M03 | READY | exact 26B tensor inventory |
| 5 | M04 | BLOCKED | deterministic compiler scaffold |
| 6 | M05 | BLOCKED | FP8 compiler path |
| 7 | M06 | BLOCKED | NVFP4 compiler path |
| 8 | M07 | BLOCKED | head-format characterization |
| 9 | M08 | BLOCKED | compiled artifact and loader |
| 10 | M09 | BLOCKED | text-only memory plan |
| 11 | M10 | BLOCKED | CPU MoE oracle |
| 12 | M11 | BLOCKED | CUDA MoE reference |
| 13 | M12 | BLOCKED | 26B attention/KV |
| 14 | M13 | BLOCKED | slow full-model inference and preliminary quality screen |
| 15 | M18 | BLOCKED | quantizer A/B and quality kill-gate report |
| 16 | M14 | BLOCKED | native NVFP4 decode |
| 17 | M15 | BLOCKED | grouped NVFP4 prefill |
| 18 | M16 | BLOCKED | selected quantized tied head |
| 19 | M17 | BLOCKED | optimized full model |
| 20 | M19 | BLOCKED | quality acceptance report |
| 21 | M20 | BLOCKED | performance acceptance report |
| 22 | M21 | BLOCKED | 32K/64K report |
| 23 | M22 | BLOCKED | product integration |
| 24 | M23 | BLOCKED | release candidate |
| 25 | M24 | OPTIONAL | full Q4_0 backend |
| 26 | M25 | OPTIONAL | MTP/vision expansion |

## Handoff packet

Every task handoff must include:

- repository HEAD and dirty status;
- current milestone file;
- completed prerequisite evidence;
- exact source/checkpoint paths available locally;
- reference GPU and toolchain;
- known failures;
- confirmation that work remains on `feat/gemma4-26b`;
- whether the agent has permission to run long quality/benchmark suites.

## Agent stop response

When blocked, the agent should produce a concise blocker report with:

```text
blocker
evidence
why guessing is unsafe
smallest decision or input required
work completed without crossing the gate
```

It must not bypass a blocked source, memory or correctness gate with a fallback implementation.

## Auxiliary imp tasks

| Task | Attached gate | Initial status | Output |
|---|---|---|---|
| R-IMP-00 | M01 | READY after M00 | immutable source/license audit |
| R-IMP-10 | M10 | BLOCKED | semantic and scale goldens |
| R-IMP-15 | M14/M15 | BLOCKED | 5080 kernel adoption decision |
| R-IMP-23 | M23 | BLOCKED | provenance, lifecycle and baseline freeze |

Use [`references/imp/IMP_AGENT_TASK.md`](references/imp/IMP_AGENT_TASK.md) for the first task.
