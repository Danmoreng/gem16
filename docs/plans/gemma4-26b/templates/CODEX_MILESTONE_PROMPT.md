# Codex milestone prompt template

Copy this prompt and append the current milestone file.

```text
You are implementing <MILESTONE_ID>: <TITLE> in Danmoreng/gem16.

Authority and reading order:
1. repository AGENTS.md
2. accepted docs/DECISIONS.md
3. gem16-26b plan 02_AGENT_OPERATING_CONTRACT.md
4. <MILESTONE_FILE>
5. relevant specs/checklists

Repository anchor:
1c4287965d318ba32a68e597f9d7b6678b883376

Before editing:
- inspect the actual current branch/tree;
- record commit and dirty state;
- write a concise drift note against the anchor;
- verify all milestone prerequisites and model/artifact locks;
- identify exact files, tests, evidence and stop conditions.

Scope:
- implement only this milestone or the explicitly stated sub-slice;
- preserve all existing Gemma 4 12B behavior and gates;
- do not introduce silent precision fallback, CPU weight offload,
  token-loop allocations, unverified model inputs or unrelated refactors;
- do not begin a downstream milestone automatically.

Correctness:
- add or update an independent oracle/fixture;
- test shapes, dtypes, tensor roles, quantization and arithmetic boundaries;
- keep routing/reductions deterministic;
- fail visibly on unsupported states.

Evidence:
- store commands/environment/results under artifacts/<milestone>/;
- include machine-readable outputs and checksums;
- for CUDA performance, include native dispatch, memory and profile evidence;
- record rejected optimization candidates.

Completion response:
- summary of implemented slice;
- files changed;
- tests and exact results;
- model/source/artifact hashes;
- memory/performance/quality deltas;
- evidence paths;
- unresolved risks;
- checklist of every milestone exit criterion as PASS/FAIL/NOT RUN;
- stop without starting the next milestone.
```

## Optional sub-slice

Append:

```text
This task is limited to:
<EXACT SLICE>

Do not implement:
<OUT OF SCOPE>
```
