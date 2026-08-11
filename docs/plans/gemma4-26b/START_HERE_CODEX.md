# Start here: current Codex task

Do not ask Codex to implement all milestones at once. All remaining work stays on `feat/gemma4-26b`, but each task
must stop at its milestone gate.

## Current task: M04

Use this prompt on the existing Linux 26B development environment:

```text
Work on M04 only on the existing feat/gemma4-26b branch. Do not create another branch.

Repository: Danmoreng/gem16
Plan anchor: 1c4287965d318ba32a68e597f9d7b6678b883376

Read:
- repository AGENTS.md
- repository docs/DECISIONS.md
- repository docs/CORRECTNESS.md
- repository docs/BENCHMARKING.md
- repository docs/CHECKPOINT_FORMAT.md
- this package 02_AGENT_OPERATING_CONTRACT.md
- milestones/M04_CHECKPOINT_COMPILER_SCAFFOLD.md
- specs/CHECKPOINT_PROVENANCE_SPEC.md
- specs/FILE_CHANGE_MAP.md
- specs/TEST_MATRIX.md
- appendices/REPOSITORY_TOUCHPOINTS.md
- checklists/BEFORE_EACH_MILESTONE.md
- accepted M03 handoff under docs/evidence/gemma4_26b/

Before editing, inspect the actual repository and write a drift report using
templates/DRIFT_REPORT.md. Then implement only the deterministic, bounded-memory
offline compiler scaffold, synthetic copy encoder, atomic publication, strict
verification and provenance required by M04. Do not implement production FP8,
NVFP4 or head quantization and do not begin M05 work.

At completion, report exact files changed, tests, evidence, deterministic fixture
hashes, peak host memory and every M04 exit criterion as PASS/FAIL. Stop after M04.
```

M00-M03 are accepted. M04 is the only current implementation milestone; M05 and later remain dependency-gated.

## Platform boundary for this handoff

Linux is the candidate canonical compiler environment and owns M04 atomic-publication and bounded-RSS evidence.
Synthetic compiler tests remain payload-free and cross-platform where filesystem semantics permit. M04 must not
produce or advertise a production 26B quantized artifact; M05-M07 own those encoders and format decisions.

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
