# Start here: first Codex task

Do not ask Codex to implement all milestones at once.

## First task

Use this prompt:

```text
Work on M00 only.

Repository: Danmoreng/gem16
Plan anchor: 1c4287965d318ba32a68e597f9d7b6678b883376

Read:
- repository AGENTS.md
- repository docs/DECISIONS.md
- this package 02_AGENT_OPERATING_CONTRACT.md
- milestones/M00_POLICY_AND_TRACK_BOOTSTRAP.md
- specs/CHECKPOINT_PROVENANCE_SPEC.md
- appendices/REPOSITORY_TOUCHPOINTS.md
- checklists/BEFORE_EACH_MILESTONE.md

Before editing, inspect the actual repository and write a drift report using
templates/DRIFT_REPORT.md. Then implement only the governance/track bootstrap
required by M00. Do not begin source downloads, compiler implementation or CUDA work.

At completion, report exact files changed, tests, decision text, unresolved owner
decisions and every M00 exit criterion as PASS/FAIL. Stop after M00.
```

## Owner action before M01

Review the M00 decision that freezes the deterministic derived-26B artifact contract and its provenance,
benchmark-labeling and runtime boundaries.

Only after that contract is accepted, start M01 with the milestone template.

## Daily agent loop

```text
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
“Implement Gemma 4 26B according to this entire ZIP.”
```

That prompt encourages:

- assumptions before tensor discovery;
- compiler/runtime coupling;
- missing reference path;
- unreviewable CUDA changes;
- quality leakage;
- benchmark claims without evidence.

## Recommended context packet per task

Give Codex only:

- repository access;
- root agent instructions;
- current milestone;
- directly relevant specs;
- one or two checklists/templates;
- prior milestone handoff.

The complete package remains available for lookup.

## v2 prerequisite — imp audit

After M00 policy acceptance, run [`references/imp/IMP_AGENT_TASK.md`](references/imp/IMP_AGENT_TASK.md) before closing M01. This task is source/evidence work only and must not start a general imp port.
