# Benchmark matrix — Fast Track R4

## Base candidate

Benchmark one frozen M17 artifact with:

- short decode prompt;
- representative 2K/8K prefill;
- the exact 16,384-prompt-token plus 64-output-forward M20 promotion row;
- 32K qualification prompt;
- 64K and max-fit prompts in M21;
- deterministic greedy and one production sampling control.

Report TTFT, prompt throughput, ITL and decode throughput separately. Use three warm-ups and ten retained runs for promotion.

For the current owner-approved bounded M20 wave, `wikipedia-real-16k64-greedy` is the sole numeric promotion row.
Short, 2K/8K, 32K and sampled controls remain diagnostic or context evidence and do not substitute for its fixed
targets. The promotion row uses batch one, FP8 KV, native CUDA Graph execution, no MTP/speculative decode, no prompt
cache/offload/fallback/recurring allocation and the unchanged runner timing boundaries. Its retained medians must
reach at least 6,000 prompt token/s and 150 ordinary decode token/s; 6,500 prompt token/s is a non-blocking stretch
target. This bounded-row decision supersedes the former generic multi-scenario promotion requirement for this wave.

## References

Official Q4_0 and direct Unsloth NVFP4 remain external references. Record their exact format, runtime, timing boundary, cache state and hardware. They do not require an internal Q4 backend.

## MTP matrix

M25 compares ordinary Target and each retained proposal mode at identical target tokens/settings. Report proposals, accepted/rejected tokens, Target batches, effective verified throughput, latency, memory and context. Proposed but unverified tokens never count as output.

## Context

Base and MTP have separate maximum-context rows. Allocation-only tests are excluded from supported-context claims.
