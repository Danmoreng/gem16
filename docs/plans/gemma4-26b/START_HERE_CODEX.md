# Start here: current Codex task

Do not ask Codex to implement all milestones at once. All remaining work stays on `feat/gemma4-26b`, but each task
must stop at its milestone gate.

## Current task: M04 owner review gate

M04 implementation is complete at `edd80cb6adae6d441924098870ceca9b4b1248d5` and has stopped at its owner gate.
Review:

- [M04 handoff](../../evidence/gemma4_26b/m04-checkpoint-compiler-scaffold-2026-08-11.md)
- [compiler contract](../../GEMMA4_26B_CHECKPOINT_COMPILER.md)
- [M04 reproducibility report](../../evidence/gemma4_26b/m04-reproducibility.json)
- [M04 bounded-memory report](../../evidence/gemma4_26b/m04-bounded-memory-report.json)

Confirm that the deterministic copy scaffold, strict source/plan coverage, bounded mmap/RSS policy, atomic
publication, restart-only recovery, provenance schemas, mutation tests and explicit non-runtime status satisfy M04.
Do not begin FP8 work or rewrite this page for M05 until owner acceptance is recorded on the milestone board.

M00-M03 are accepted. M04 passes implementation gates but remains the current approved milestone pending owner
review; M05 and later remain dependency-gated.

## Platform boundary for this handoff

Linux x86-64, little-endian, `C.UTF-8`, CPython 3.14.6 and one thread are the canonical M04 compiler environment and
own the retained atomic-publication, reproducibility and bounded-RSS evidence.
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
