# Before each milestone

Use this checklist before editing.

## Repository state

- [ ] Read root `AGENTS.md`.
- [ ] Read current `docs/DECISIONS.md`.
- [ ] Read `02_AGENT_OPERATING_CONTRACT.md`.
- [ ] Read the exact milestone file.
- [ ] Record current branch, commit and dirty state.
- [ ] Compare current tree with anchor `1c4287965d318ba32a68e597f9d7b6678b883376`.
- [ ] Write a drift note for changed files/contracts relevant to the milestone.

## Prerequisites

- [ ] Every prerequisite milestone exit gate is closed.
- [ ] Required source/model/toolchain locks verify.
- [ ] Required goldens/evidence are available.
- [ ] No prior unresolved decision blocks the work.
- [ ] Hardware/software needed for tests is available.

## Scope

- [ ] Define the narrow implementation slice.
- [ ] List files expected to change.
- [ ] List files explicitly out of scope.
- [ ] State arithmetic/precision changes, if any.
- [ ] State expected memory/performance effect.
- [ ] Choose the parent baseline and exact comparison command.

## Tests

- [ ] Identify host tests.
- [ ] Identify CUDA/operator tests.
- [ ] Identify real-model probes.
- [ ] Identify 12B regression gates.
- [ ] Identify sanitizer/allocation evidence.
- [ ] Identify quality/performance tests, if applicable.

## Evidence

- [ ] Create `artifacts/mXX/`.
- [ ] Record commands and environment.
- [ ] Decide machine-readable output schema.
- [ ] Ensure no secrets/private data will be captured.
- [ ] Define pass/fail before seeing results.

## Stop conditions

- [ ] Source/config mismatch stop condition understood.
- [ ] Memory hard stop understood.
- [ ] Correctness/quality stop condition understood.
- [ ] No silent fallback/offload.
- [ ] Do not begin the next milestone automatically.
