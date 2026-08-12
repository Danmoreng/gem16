# Benchmark matrix — Fast Track R4

## Base candidate

Benchmark one frozen M17 artifact with:

- short decode prompt;
- representative 2K/8K prefill;
- 32K qualification prompt;
- 64K and max-fit prompts in M21;
- deterministic greedy and one production sampling control.

Report TTFT, prompt throughput, ITL and decode throughput separately. Use three warm-ups and ten retained runs for promotion.

## References

Official Q4_0 and direct Unsloth NVFP4 remain external references. Record their exact format, runtime, timing boundary, cache state and hardware. They do not require an internal Q4 backend.

## MTP matrix

M25 compares ordinary Target and each retained proposal mode at identical target tokens/settings. Report proposals, accepted/rejected tokens, Target batches, effective verified throughput, latency, memory and context. Proposed but unverified tokens never count as output.

## Context

Base and MTP have separate maximum-context rows. Allocation-only tests are excluded from supported-context claims.
