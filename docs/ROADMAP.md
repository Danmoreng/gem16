# Roadmap

Current stage: run the bounded Linux short-context ordinary-decode investigation, close the direct all-regions
memory-reserve record, then begin Gemma 4 26B A4B bring-up while preserving the mature 12B product path. The 16K D2
performance sprint and its Windows regression are complete and remain regression evidence rather than active
optimization scope.

## Current product baseline

The pinned Gemma 4 12B Unified path is implemented and remains the regression baseline:

- direct loading of the mixed FP8/NVFP4 Safetensors checkpoint;
- deterministic manifest validation and fixed-address GPU arenas;
- batch-one chunked prefill and CUDA Graph decode;
- a hybrid cache with checkpoint-FP8 production mode qualified through the 262,144-position contract and BF16
  correctness mode for bounded contexts;
- greedy and GPU-resident temperature/top-k/top-p/min-p/repetition sampling;
- optional official BF16 MTP assistant with exact Target verification for D1, D2, and D4;
- resident text, image, and audio chat with exact media-prefix identity;
- native tool declarations, calls, and tool-result continuation;
- OpenAI-compatible Chat Completions and Responses APIs with streaming and bounded resident sessions;
- the cross-platform gem16 Studio desktop application.

Video input, continuous batching, response branching, and persistent prompt-cache files are not implemented.
Required/named tool forcing and `parallel_tool_calls=false` remain visibly unsupported until constrained generation
is qualified.

## Current performance reference

The retained Linux max-power comparison uses one RTX 5080 Laptop GPU, batch one, a fixed 16,384-token Wikipedia
prompt, 1,135 output positions, fixed D2, three warm-ups, and ten measurements:

| Engine | Prefill tok/s | Effective D2 tok/s | ITL |
|---|---:|---:|---:|
| vLLM 0.26.0 | **6,257.37** | 82.25 | 12.158 ms |
| **gem16 `a819d14c`** | 5,866.86 | **87.66** | **11.408 ms** |
| llama.cpp b10240 | 3,941.23 | 83.89 | 11.921 ms |

Gem16 preserves its own ordinary Target sequence under MTP. The three engines do not share output hashes or exact
formats, so this is a controlled performance comparison rather than semantic or format parity. Full commands,
telemetry, caveats, and raw-result locations are in [BENCHMARKING.md](BENCHMARKING.md),
[PERFORMANCE_LEDGER.md](PERFORMANCE_LEDGER.md), and `benchmarks/baselines/cross_engine_mtp/`.

The remaining measured 12B performance gap is prompt processing against direct-load vLLM: gem16 is 6.24% slower
on the retained 16K workload. Corrected D2 remains 6.57% faster than vLLM and 4.49% faster than llama.cpp. The
prior 89.58 tok/s row is historical because it omitted two required verifier BF16 boundaries.

## Active track: Linux short-context ordinary decode

[SHORT_CONTEXT_DECODE_OPTIMIZATION_PLAN.md](SHORT_CONTEXT_DECODE_OPTIMIZATION_PLAN.md) is the binding next plan. It
starts with a fresh Linux parent and ordinary-decode matrix, audits the `llama-bench tg128` timing-boundary
difference, captures Nsight Systems/Compute evidence, and admits at most two measured candidates. The current 16K
ordinary, fixed-D2, prefill, correctness, memory, and Windows paths are mandatory regression gates.

The standard-shape Windows trigger is 52.43 tok/s for gem16's context-1/128-token ordinary decode and 62.08 tok/s
for llama.cpp b10240 `tg128`. This is directional rather than exact parity because token feeds, starting position,
formats, K/V precision, and timing boundaries differ. Linux establishes its own parent before any implementation.

[PERFORMANCE_IMPROVEMENT_PLAN.md](PERFORMANCE_IMPROVEMENT_PLAN.md) owns the completed 16K-centered final sprint and
remains regression and closure evidence rather than the execution order for new candidates.

[PREFILL_OPTIMIZATION_PLAN.md](PREFILL_OPTIMIZATION_PLAN.md) is retained as historical evidence for the earlier
prefill program; it no longer owns execution order.

## Next track: Gemma 4 26B A4B

After the bounded short-context plan and direct memory-reserve record close, the next implementation program is the
experimental Gemma 4 26B A4B track. Its binding entry points are:

1. [plans/gemma4-26b/START_HERE_CODEX.md](plans/gemma4-26b/START_HERE_CODEX.md)
2. [plans/gemma4-26b/00_MASTER_IMPLEMENTATION_PLAN.md](plans/gemma4-26b/00_MASTER_IMPLEMENTATION_PLAN.md)
3. [plans/gemma4-26b/MILESTONE_STATUS_BOARD.md](plans/gemma4-26b/MILESTONE_STATUS_BOARD.md)

Begin M00 only after the 12B sprint closure record, then follow the plan's dependency order. In particular:

- lock source, compiler, tokenizer, and quality references before kernel work;
- run the early synthetic 32K residency gate against directly measured CUDA-visible memory;
- retain at least 700 MiB unused VRAM on the reference 16 GB GPU;
- reject CPU weight offload and silent precision or kernel fallback in primary results;
- run the converter quality kill gate before the native MoE optimization milestones;
- preserve the working 12B runtime instead of generalizing its hot path prematurely.

The 26B milestone board, not this high-level roadmap, owns detailed task status.

## Open correctness and product work

These items remain valid but do not enter the bounded 12B sprint or precede the subsequent 26B bootstrap unless
they block a regression or receive an explicit priority change:

- broaden task-quality and perplexity-style evaluation beyond the retained prompt/logit fixtures;
- complete publication-grade resource telemetry for sampled MTP;
- implement and qualify video as sampled image frames;
- implement constrained generation for required/named tool selection and disabled parallel calls;
- add continuous batching only after batch-one behavior remains stable;
- add another CUDA architecture backend only with native kernels and fresh qualification.

## Documentation ownership

- [SHORT_CONTEXT_DECODE_OPTIMIZATION_PLAN.md](SHORT_CONTEXT_DECODE_OPTIMIZATION_PLAN.md) owns the active bounded
  short-context ordinary-decode investigation.
- [PERFORMANCE_IMPROVEMENT_PLAN.md](PERFORMANCE_IMPROVEMENT_PLAN.md) owns the completed 16K-centered 12B sprint.
- [ARCHITECTURE.md](ARCHITECTURE.md), [MEMORY.md](MEMORY.md), and feature documents describe current behavior.
- [DECISIONS.md](DECISIONS.md) records accepted decisions and superseded alternatives.
- [PERFORMANCE_LEDGER.md](PERFORMANCE_LEDGER.md) retains optimization history and negative results.
- `docs/plans/gemma4-26b/` owns the detailed 26B implementation sequence.

Completed hand-off and optimization-plan documents are removed from the active tree after their durable evidence is
recorded in Decisions and the Performance Ledger; Git history remains the archive.
