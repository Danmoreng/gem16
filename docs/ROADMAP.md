# Roadmap

Current stage: preserve the mature Gemma 4 12B product path while bringing up Gemma 4 26B A4B as the next
model-specific track.

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
| vLLM 0.26.0 | **6,247.55** | 81.95 | 12.202 ms |
| **gem16 `8e86cb38`** | 5,863.59 | **89.58** | **11.163 ms** |
| llama.cpp b10240 | 3,922.61 | 82.88 | 12.065 ms |

Gem16 preserves its own ordinary Target sequence under MTP. The three engines do not share output hashes or exact
formats, so this is a controlled performance comparison rather than semantic or format parity. Full commands,
telemetry, caveats, and raw-result locations are in [BENCHMARKING.md](BENCHMARKING.md),
[PERFORMANCE_LEDGER.md](PERFORMANCE_LEDGER.md), and `benchmarks/baselines/cross_engine_mtp/`.

The remaining measured 12B performance gap is prompt processing against direct-load vLLM: gem16 is 6.15% slower
on the retained 16K workload. [PREFILL_OPTIMIZATION_PLAN.md](PREFILL_OPTIMIZATION_PLAN.md) remains the bounded plan
for work on that gap. No 12B optimization may weaken the existing correctness, memory, or cross-platform gates.

## Active track: Gemma 4 26B A4B

The next implementation program is the experimental Gemma 4 26B A4B track. Its binding entry points are:

1. [plans/gemma4-26b/START_HERE_CODEX.md](plans/gemma4-26b/START_HERE_CODEX.md)
2. [plans/gemma4-26b/00_MASTER_IMPLEMENTATION_PLAN.md](plans/gemma4-26b/00_MASTER_IMPLEMENTATION_PLAN.md)
3. [plans/gemma4-26b/MILESTONE_STATUS_BOARD.md](plans/gemma4-26b/MILESTONE_STATUS_BOARD.md)

Begin with M00 and follow the plan's dependency order. In particular:

- lock source, compiler, tokenizer, and quality references before kernel work;
- run the early synthetic 32K residency gate against directly measured CUDA-visible memory;
- retain at least 700 MiB unused VRAM on the reference 16 GB GPU;
- reject CPU weight offload and silent precision or kernel fallback in primary results;
- run the converter quality kill gate before the native MoE optimization milestones;
- preserve the working 12B runtime instead of generalizing its hot path prematurely.

The 26B milestone board, not this high-level roadmap, owns detailed task status.

## Open correctness and product work

These items remain valid but do not precede the 26B bootstrap unless they block a regression or receive an explicit
priority change:

- broaden task-quality and perplexity-style evaluation beyond the retained prompt/logit fixtures;
- complete publication-grade resource telemetry for sampled MTP;
- implement and qualify video as sampled image frames;
- implement constrained generation for required/named tool selection and disabled parallel calls;
- add continuous batching only after batch-one behavior remains stable;
- add another CUDA architecture backend only with native kernels and fresh qualification.

## Documentation ownership

- [ARCHITECTURE.md](ARCHITECTURE.md), [MEMORY.md](MEMORY.md), and feature documents describe current behavior.
- [DECISIONS.md](DECISIONS.md) records accepted decisions and superseded alternatives.
- [PERFORMANCE_LEDGER.md](PERFORMANCE_LEDGER.md) retains optimization history and negative results.
- `docs/plans/gemma4-26b/` owns the detailed 26B implementation sequence.

Completed hand-off and optimization-plan documents are removed from the active tree after their durable evidence is
recorded in Decisions and the Performance Ledger; Git history remains the archive.
