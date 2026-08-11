# Codex workflow for this program

## Start of every task

Codex must:

1. read repository `AGENTS.md`;
2. read accepted `docs/DECISIONS.md`;
3. read package `02_AGENT_OPERATING_CONTRACT.md`;
4. read the current milestone;
5. inspect the current tree and compare it with anchor commit `1c4287965d318ba32a68e597f9d7b6678b883376`;
6. write a short drift note;
7. confirm prerequisites from previous milestone evidence.

Do not begin coding from this package alone.

## Scope discipline

All M03-M25 work stays on the existing `feat/gemma4-26b` branch. One task/change set addresses one milestone or a
deliberately smaller slice. A task may stop before all exit criteria, but it must not silently start the next
milestone or create another development branch.

Allowed small refactors:

- directly necessary for the milestone;
- tested;
- named in the milestone review record;
- no arithmetic change unless milestone covers it.

## Test-first behavior

Where practical:

1. add parser/schema fixture;
2. add host oracle;
3. add failing operator test;
4. implement;
5. add real checkpoint probe;
6. add integration test;
7. collect performance only after correctness.

For performance work, keep a correct parent and one isolated candidate at a time.

## Evidence-first optimization

Every optimization candidate gets:

```text
hypothesis
expected bottleneck
changed arithmetic/layout/schedule
correctness result
microbenchmark
end-to-end result
memory delta
profile trace
decision: retain/reject
```

Record rejected candidates in `docs/PERFORMANCE_LEDGER.md`. Do not repeatedly rediscover them.

## Commands

Codex should retain exact commands in milestone artifacts. Suggested sequence:

```text
cmake configure/build host
host CTest
CUDA debug build
targeted CUDA tests
compute-sanitizer
real model probe
release build
microbenchmark
end-to-end benchmark
documentation update
```

## Changes to model arithmetic

Any change to:

- quantizer;
- scale interpretation;
- rounding;
- norm order;
- activation;
- router;
- expert reduction;
- attention;
- head/softcap;
- KV precision;

requires a decision or experiment record and appropriate quality rerun.

## Git behavior

- do not amend unrelated history;
- do not commit large model payloads;
- keep generated evidence according to repository policy;
- use descriptive commits;
- use `feat/gemma4-26b` for M03-M25 and do not create milestone branches;
- do not force-push reviewed milestone evidence;
- do not push/open PR unless explicitly requested by the project owner.

## Completion report

Codex reports:

```text
milestone/slice
files changed
tests and exact results
model/artifact hashes
memory change
performance change
evidence paths
known risks
exit criteria pass/fail
next unblocked work
```

No vague “tests pass” without commands/counts.

## Stop conditions

Stop and escalate when:

- source model/config differs from plan materially;
- artifact policy is not accepted;
- tensor inventory is ambiguous;
- weight arena exceeds 14,300 MiB;
- 32K cannot preserve required margin;
- QAT-derived quality misses frozen thresholds;
- native path is slower than Q4_0 and no bounded next hypothesis exists;
- fallback/offload is required;
- 12B regression cannot be isolated.

Stopping is a valid result when supported by evidence.

## Prompt usage

Use [`../templates/CODEX_MILESTONE_PROMPT.md`](../templates/CODEX_MILESTONE_PROMPT.md) and append the exact milestone file. Do not feed all 26 milestones into a single implementation prompt; it encourages scope collapse.

## Review handoff

Before review:

- run checklist;
- produce evidence;
- update decision/ledger/docs;
- mark each exit criterion;
- identify any test requiring owner hardware.

Reviewer verifies claims from raw artifacts, not prose alone.

## Settled-evidence and external-source rule

Before generating new optimization/audit hypotheses, Codex reads the repository's settled-evidence ledger. Every confirmed or refuted investigation adds an anchored entry using [`../templates/SETTLED_EVIDENCE_ENTRY.md`](../templates/SETTLED_EVIDENCE_ENTRY.md).

External implementation claims require an immutable source lock. Code copying follows [`../checklists/THIRD_PARTY_CODE_IMPORT_CHECKLIST.md`](../checklists/THIRD_PARTY_CODE_IMPORT_CHECKLIST.md). A source reference never authorizes a wholesale architecture port.
