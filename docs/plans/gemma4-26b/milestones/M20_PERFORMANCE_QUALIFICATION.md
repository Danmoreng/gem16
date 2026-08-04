# M20 — Controlled performance qualification

## Objective

Measure and qualify the final 26B candidate against official Q4_0 and direct NVFP4 baselines on the same RTX 5080-class Blackwell machine with complete telemetry and honest timing boundaries.

## Why this milestone exists

Native NVFP4 is chosen for performance only if it wins end-to-end under equivalent semantics. Kernel throughput alone is insufficient.

## Prerequisites

- M19 final quality decision
- M17 optimized path
- M09 final memory accounting

## Repository areas to inspect first

- `docs/BENCHMARKING.md`
- `benchmarks/baselines/cross_engine_mtp/`
- `src/cli/bench_main.cpp`
- `include/gem16/engine.h`
- `tools/benchmark_server.py`
- `docs/PERFORMANCE_LEDGER.md`

## Suggested additions or boundaries

- `tools/benchmark_gemma4_26b_cross_engine.py`
- `benchmarks/baselines/gemma4_26b_16gb/`
- `docs/GEMMA4_26B_BENCHMARKING.md`

## Implementation sequence

1. Pin exact gem16, llama.cpp, vLLM/Unsloth-runtime, driver, CUDA, CUTLASS, model and benchmark revisions.
2. Define common token workloads for prefill 128/512/2K/8K/16K and decode at 128/2K/8K/32K context.
3. Use batch one, no prefix-cache hit, no CPU offload, identical prompt token IDs and equivalent sampling/greedy settings.
4. Record timing boundaries separately: model load, prompt processing, TTFT, first-token exclusion, steady decode, ITL and HTTP overhead where applicable.
5. Run three warm-ups and ten retained runs, serializing engines and controlling idle GPU, power profile and start temperature.
6. Collect 50–200 ms process VRAM/power/clock/thermal telemetry and Nsight traces for representative runs.
7. Verify output determinism and quality-qualified artifact hashes before every run.
8. Capture native instruction disassembly and runtime kernel dispatch evidence for FP8 and NVFP4 paths.
9. Report medians, means, standard deviations, confidence intervals, p95/p99 ITL and raw samples.
10. Promote the result only if both prefill and decode beat the accepted Q4_0 baseline. Treat vLLM/Unsloth matching as a stretch comparison with disclosed format/boundary differences.

## Required tests

- Harness refuses dirty worktrees, wrong model hashes, unsupported power state, busy GPU and missing telemetry.
- Output token counts and hashes are checked before accepting timing.
- No fallback count, CPU offload or token-loop allocation.
- Every engine receives the same prompt token file.
- Raw runs are never overwritten.
- Disassembly and Nsight prove selected native kernels execute.
- Memory peak stays within the product margin.

## Evidence and documentation outputs

- `docs/GEMMA4_26B_BENCHMARKING.md`
- `benchmarks/baselines/gemma4_26b_16gb/characterization.json`
- Raw per-run JSON/log/telemetry directories
- `artifacts/m20/native-dispatch.txt`
- `artifacts/m20/nsight/`
- Performance ledger entry with retained and rejected results.

## Suggested commands

```text
python tools/benchmark_gemma4_26b_cross_engine.py --profile full --warmups 3 --repetitions 10 --output benchmarks/results/$(date +%F)/$GIT_SHA/$MACHINE_ID
```
```text
nvidia-smi --query-gpu=name,uuid,memory.total,power.limit,clocks.max.sm --format=csv
```
```text
cuobjdump --dump-sass build/blackwell-release/bin/gem16-server > artifacts/m20/native-dispatch.txt
```

## Risks to watch in this milestone

- Different runtimes expose different timing boundaries.
- vLLM may need memory utilization tuning or unsupported Windows paths.
- Thermal drift on a laptop 5080 can overwhelm small improvements.
- Q4_0 and NVFP4 may produce different token sequences, invalidating equal-length assumptions unless fixed teacher forcing is used.

## Forbidden shortcuts

- Comparing cached prompts against uncached prompts.
- Counting MTP proposals as output tokens.
- Using different context, KV precision or output lengths without disclosure.
- Reporting a one-run best case.
- Ignoring a slower prefill result because decode wins.

## Exit criteria

- [ ] Complete controlled 3/10 results exist for all required workloads.
- [ ] Final gem16 candidate beats the accepted Q4_0 baseline in both prefill and decode medians.
- [ ] Native dispatch, no-offload, no-fallback and memory claims are evidenced.
- [ ] All timing-boundary and format differences are disclosed.
- [ ] Raw data and telemetry are retained and reproducible.
- [ ] Any vLLM/Unsloth claim is worded according to actual parity.

## Downstream milestones unblocked

- M21 long-context qualification
- M22 product integration
- M23 release

## Codex execution prompt

```text
You are implementing M20: Controlled performance qualification in Danmoreng/gem16.

Read repository AGENTS.md, docs/DECISIONS.md, docs/CORRECTNESS.md, docs/BENCHMARKING.md, the package master plan, and this milestone. Work only on M20. Inspect the actual current tree before editing and write a drift note if it differs from the anchored commit.

Implement the narrowest change that satisfies every exit criterion. Add tests before or with implementation. Do not add silent fallbacks, token-loop allocations, CPU weight offload, or unrelated refactors. Preserve all existing Gemma 4 12B exact gates.

At completion, report files changed, tests and exact results, generated evidence paths, memory/performance deltas where relevant, unresolved risks, and whether each M20 exit criterion passed. Stop before starting the next milestone.
```

## imp reference amendment

Publish ordinary greedy decode with MTP/speculation off at contexts 128/2K/8K/16K/32K/64K, including median and p95 ITL, actual dispatch path, output checksum and VRAM margin. Store a machine-readable perf/VRAM baseline. Imp 5090 figures may appear only as external context with full caveats, never as a 5080 acceptance threshold.
