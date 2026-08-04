# AGENTS.md

## Purpose

This file contains the stable operating rules for coding agents working in `gem16`. Detailed model contracts,
implementation state, benchmarks, and milestone plans belong in the linked documentation rather than here.

## Project

`gem16` is a specialized C++20/CUDA inference engine for Gemma models on NVIDIA GPUs with approximately 16 GB of
VRAM. Blackwell compute capability 12.0 is the first optimized backend.

The mature Gemma 4 12B Unified path is the current production baseline. Gemma 4 26B A4B is an experimental product
track. Preserve the existing 12B behavior while adding model-specific 26B plans; do not turn the hot path into a
generic graph framework.

## Authoritative documentation

Read only the documents relevant to the task, but always inspect the current code before relying on prose.

- `docs/ARCHITECTURE.md`: current runtime architecture and ownership boundaries.
- `docs/ROADMAP.md`: active priorities and current implementation status.
- `docs/DECISIONS.md`: accepted project decisions and rejected alternatives.
- `docs/DEVELOPMENT.md`: C++, CUDA, dependency, testing, and security rules.
- `docs/BENCHMARKING.md`: benchmark boundaries, repetitions, baselines, and reporting.
- `docs/PERFORMANCE_IMPROVEMENT_PLAN.md`: active bounded Gemma 4 12B performance sprint.
- `docs/CORRECTNESS.md`: numerical references, fixtures, and quality evidence.
- `docs/MEMORY.md`: current allocator, residency, KV, and slot accounting.
- `docs/CHECKPOINT_FORMAT.md` and `docs/WEIGHT_LAYOUT.md`: checkpoint and device-layout contracts.
- `docs/MTP.md`, `docs/AUDIO.md`, `docs/VISION.md`, and `docs/SERVER.md`: feature-specific contracts.
- `docs/plans/gemma4-26b/00_MASTER_IMPLEMENTATION_PLAN.md`: binding 26B implementation program.
- `docs/PERFORMANCE_LEDGER.md`: measured optimization history; evidence, not current policy.

When documents disagree, use this order:

1. this file's permanent correctness and integrity rules;
2. the newest accepted entry in `docs/DECISIONS.md`;
3. explicit project-owner instructions for the current task when consistent with those rules, or when the task
   explicitly updates project policy;
4. the relevant feature specification or active plan;
5. historical roadmap and performance-ledger text.

Record a decision instead of silently resolving a material conflict.

## Priorities

Use this order when tradeoffs conflict:

1. correctness and model quality;
2. benchmark integrity and reproducibility;
3. stable execution within the declared VRAM budget;
4. batch-one decode latency and throughput;
5. prompt-processing throughput and TTFT;
6. long-context capability;
7. startup latency, portability, and broader product scope.

Do not trade a higher priority for a lower one without an explicit decision and evidence.

## Permanent runtime and benchmark rules

- Never silently change precision, tensor format, kernel family, context, or model semantics. Unsupported native
  paths fail visibly unless an explicitly requested diagnostic fallback records every affected tensor/operator.
- After plan creation and arena reservation, recurring inference loops perform no device or pageable-host
  allocation, filesystem access, dynamic compilation, growing-container operation, or hidden framework allocation.
- Promoted performance results keep all per-token model weights resident on the GPU. CPU weight offload or expert
  streaming is diagnostic only and must be labeled; it cannot support a primary speed claim.
- Keep one persistent device representation per weight. Do not duplicate tied weights or retain source-order and
  runtime-order device copies. Temporary experiments must report their extra bytes and are not production results.
- Do not obtain speedups by changing prompt/output token IDs, output length, sampling, chat template, context,
  attention semantics, cache reuse, batch size, quality, or timing boundaries without explicit disclosure.
- Every claimed optimization must retain correctness evidence, raw benchmark samples, peak VRAM, actual runtime
  dispatch, and the exact code/model/toolchain revisions.
- Treat model files, tokenizers, media, JSON, and shard indexes as untrusted input. Validate sizes, offsets, integer
  arithmetic, paths, dtypes, schemas, and allocation requests. Never execute repository model code or require
  `trust_remote_code` in the runtime.
- Native-kernel claims require runtime dispatch evidence and representative instruction inspection; filenames,
  compile flags, or theoretical throughput are not proof.

## Checkpoint and compilation policy

Supported model profiles may load immutable upstream checkpoints directly or consume project-compiled artifacts.
A project compiler is allowed when its model plan requires one.

Compiled artifacts must be produced offline from immutable source revisions by versioned, reproducible code. Record
source files, compiler/toolchain, tensor transformations, output hashes, omitted tensor families, and quality
results. The inference runtime must not silently quantize or requantize weights, create an undisclosed persistent
converted copy, or describe a project-built artifact as an official upstream quantization.

## Agent workflow

For every task:

1. read this file, `docs/ROADMAP.md`, `docs/DECISIONS.md`, and the narrow relevant specification;
2. inspect `git status`, current source, tests, and existing evidence before editing;
3. identify the smallest coherent change and state uncertain assumptions;
4. preserve unrelated user changes and avoid broad rewrites;
5. add or update tests with the implementation;
6. run the narrow tests first, then the required host/CUDA/integration gates;
7. benchmark and profile when changing a hot path or performance claim;
8. update architecture, decisions, memory, correctness, or the performance ledger when their facts change;
9. report files changed, exact commands/results, limitations, and unresolved uncertainty.

Do not begin blocked downstream milestones, weaken tolerances to pass a test, or retain an optimization because it
“should” be faster. Do not commit unless the user explicitly asks.

## Hot-path changes

Before changing a hot CUDA path, establish the current benchmark/profile, bottleneck hypothesis, expected limiting
resource, numerical boundary, and reference behavior. Afterward, report compiler target, registers, shared memory,
stack/local memory, spills, actual instruction/dispatch path, benchmark distribution, correctness result, and VRAM
delta. Keep complex reference implementations available to tests or diagnostic tools without creating a silent
production fallback.

## Repository hygiene

- Follow `docs/DEVELOPMENT.md` and the existing local style.
- Use one logical change per patch; do not combine formatting sweeps with arithmetic or kernel changes.
- Pin third-party source and preserve license/provenance as documented in `third_party/README.md`.
- Never overwrite prior benchmark results or checked evidence.
- Do not edit generated files by hand; update their checked-in generator and verify generated output.
- Do not modify or delete unrelated dirty-worktree files.

A feature is complete only when implementation, tests, actionable errors, documentation, memory accounting, and
relevant correctness/performance evidence agree. A microbenchmark win alone is not completion.
