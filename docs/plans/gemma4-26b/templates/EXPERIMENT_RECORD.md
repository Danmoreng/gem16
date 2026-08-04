# Experiment: <title>

**Date:**
**Milestone:**
**Hypothesis:**
**Parent commit/artifact:**
**Candidate commit/artifact:**
**Machine:**
**Status:** planned | running | retained | rejected

## Expected mechanism

What bottleneck or error should change? Include expected direction and approximate magnitude.

## Change

- files:
- arithmetic change:
- layout change:
- scheduling change:
- memory change:
- graph change:

## Unchanged semantics

List prompt, tokens, precision, output, cache, sampling and timing boundaries that remain fixed.

## Correctness gates

| Gate | Command | Result |
|---|---|---|
| Host tests | | |
| CUDA tests | | |
| Real model | | |
| Determinism | | |
| Quality | | |

## Memory

| Metric | Parent | Candidate | Delta |
|---|---:|---:|---:|
| Weights | | | |
| KV | | | |
| Workspace | | | |
| Graph | | | |
| Process peak | | | |

## Performance

| Workload | Parent | Candidate | Delta | Confidence |
|---|---:|---:|---:|---|
| | | | | |

## Profile findings

- kernel:
- registers/local/shared:
- DRAM:
- Tensor Core:
- stalls:
- launch count:

## Decision

Retain/reject and why.

## Evidence

- commands:
- raw runs:
- telemetry:
- Nsight:
- SASS:
- checksums:

## Follow-up

Do not imply that follow-up is approved; name the next bounded hypothesis.
