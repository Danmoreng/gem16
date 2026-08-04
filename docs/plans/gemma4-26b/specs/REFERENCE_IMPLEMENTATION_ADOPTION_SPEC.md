# External reference implementation adoption specification

## Purpose

Define how external runtimes such as imp may influence gem16 without becoming an untracked source of code, semantics or benchmark claims.

## Reference states

Every external item is classified as:

```text
OBSERVED
VERIFIED_AT_PIN
REIMPLEMENTED
COPIED_WITH_LICENSE
REJECTED
SUPERSEDED
```

The state, immutable source, destination and evidence must be machine-readable.

## Semantic adoption

- Official model code/checkpoint metadata is normative.
- A second runtime is a differential source and failure-mode catalogue.
- Conflicts require a recorded experiment; never choose the faster answer silently.
- Extract small deterministic fixtures instead of depending permanently on a foreign runtime.

## Code adoption

Copied code requires:

- license compatibility decision;
- original notice retained;
- exact source commit/path/hash;
- isolated destination boundary;
- modification record;
- tests independent of the donor repository;
- no hidden transitive vendoring.

## Benchmark adoption

External numbers are not local baselines. Store them as `external_context` with:

- hardware;
- software commit;
- model/checkpoint;
- quantization;
- command;
- context/output size;
- cache/speculation state;
- measurement date;
- source link.

## Update discipline

A new upstream commit is a new reference version. Never auto-follow `main`. Re-audit changed files and rerun affected fixtures before updating the lock.
