# Start here: current Codex task

Do not ask Codex to implement all milestones at once. All remaining work stays on `feat/gemma4-26b`, but each task
must stop at its milestone gate.

## Current task: M03

Use this prompt after switching to Linux with the locked model sources available:

```text
Work on M03 only on the existing feat/gemma4-26b branch. Do not create another branch.

Repository: Danmoreng/gem16
Plan anchor: 1c4287965d318ba32a68e597f9d7b6678b883376

Read:
- repository AGENTS.md
- repository docs/DECISIONS.md
- repository docs/CORRECTNESS.md
- repository docs/BENCHMARKING.md
- this package 02_AGENT_OPERATING_CONTRACT.md
- milestones/M03_MANIFEST_AND_TENSOR_INVENTORY.md
- specs/TENSOR_NAMING_DISCOVERY.md
- specs/CHECKPOINT_PROVENANCE_SPEC.md
- appendices/REPOSITORY_TOUCHPOINTS.md
- checklists/BEFORE_EACH_MILESTONE.md
- accepted M01 and M02 evidence under docs/evidence/gemma4_26b/

Before editing, inspect the actual repository and write a drift report using
templates/DRIFT_REPORT.md. Then implement only the exact 26B tensor inventory,
role/validation contract and preliminary 32K admission evidence required by M03.
Do not begin compiler quantization or M04 work.

At completion, report exact files changed, tests, evidence, measured byte totals,
the direct-free-memory result and every M03 exit criterion as PASS/FAIL. Stop after M03.
```

M00, M01 and M02 are accepted. The imp source/license audit required before M01 closure is also complete. M03 is
the only current implementation milestone; M04 and later remain blocked.

## Platform boundary for this handoff

The Windows preparation step updates policy, runs host/Python checks and publishes the feature branch only. It does
not download or execute the 26B payload because the machine has only about 40 GB free. Perform source inventory,
synthetic CUDA admission and real-model validation on Linux with sufficient storage.

## Daily agent loop

```text
verify feat/gemma4-26b and current HEAD
read current milestone
verify prerequisites
write drift note
add tests/goldens
implement narrow slice
run required gates
store evidence/checksums
update decision/docs/ledger
mark exit criteria
stop
```

## Do not do this

```text
“Implement Gemma 4 26B according to this entire package.”
```

That prompt encourages assumptions before tensor discovery, compiler/runtime coupling, unreviewable CUDA changes,
quality leakage and benchmark claims without evidence.

## Recommended context packet per task

Give Codex only:

- repository access and the existing `feat/gemma4-26b` branch;
- root agent instructions;
- current milestone;
- directly relevant specs;
- one or two checklists/templates;
- prior milestone handoff.

The complete package remains available for lookup.
