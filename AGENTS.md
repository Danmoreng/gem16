# AGENTS.md

## Project

`gem16` is a C++20/CUDA inference engine and local product for Gemma models on approximately 16 GB NVIDIA GPUs;
Blackwell SM120 is its first optimized backend. Its public profiles are Gemma 4 12B Unified and Gemma 4 26B A4B
Compact Vision (Trellis35 text plus FP8 Vision), with explicitly different capabilities. The qualified text-only
26B NVFP4 path remains an internal regression and rollback profile. Preserve all three specialized paths and do not
turn any model-specific hot path into a generic framework.

## Default task loop

For every task:

1. inspect `git status`, branch, current source and relevant tests;
2. read this file, `docs/ACTIVE_DECISIONS.md`, and one narrow feature/task entry;
3. state the smallest change and any uncertainty;
4. implement only the owned slice, preserving unrelated dirty files;
5. run the smallest proportional checks, then widen them when shared runtime/CUDA behavior is affected;
6. report files, exact commands/results, limitations and evidence.

For product work, the default task entry is `docs/PRODUCT_CONTRACT.md`; API work additionally reads
`docs/OPENAI_AGENT_CORE_V1.md`. Use `docs/plans/gemma4-26b/START_HERE_CODEX.md` only for 26B model-specific work.
Read `ROADMAP.md`, the historical `DECISIONS.md`, `CORRECTNESS.md`, `BENCHMARKING.md`, ledgers and detailed
specifications only when the specific task, changed boundary or requested claim makes them relevant. Do not make a
documentation-only or isolated host change wait for the complete repository test matrix. Do not begin a blocked
milestone without an explicit active decision or owner instruction.

## Decision precedence

Use this order when documents disagree:

1. this file's permanent runtime, security and integrity rules;
2. `docs/ACTIVE_DECISIONS.md`;
3. current source, tests and accepted evidence for facts about what exists;
4. `docs/PRODUCT_CONTRACT.md`, then the active feature contract, milestone status and narrow specification;
5. historical `docs/DECISIONS.md`, ledgers and other planning records.

Report a material conflict instead of silently guessing. A new owner decision must state which future requirement it
supersedes. Historical evidence is not rewritten.

## Permanent runtime and integrity rules

- Never silently change precision, tensor format, kernel family, context, cache semantics or model behavior. An
  unsupported native path fails visibly; an explicitly requested diagnostic fallback must report every affected
  tensor/operator and cannot support a primary performance claim.
- After plan creation and arena reservation, recurring inference performs no device/pageable-host allocation,
  filesystem access, JIT/dynamic compilation, growing-container operation, hidden framework allocation or weight
  repack.
- Primary performance paths keep all per-token model weights resident on the GPU. CPU weight offload and expert
  streaming are diagnostic only. Keep one persistent device representation per weight and physically alias tied
  weights; do not retain source-order plus runtime-order copies.
- Treat model files, tokenizers, media, JSON, Safetensors, GGUF and shard indexes as untrusted. Validate paths,
  symlinks, sizes, offsets, overlaps, integer arithmetic, shapes, dtypes, schemas and allocation requests. Never
  execute model repository code or require `trust_remote_code` in the runtime.
- Compiled artifacts are offline, project-built derivatives from immutable sources. Record source/compiler/toolchain,
  transformations, omitted families, output hashes and quality status. Runtime startup never silently quantizes,
  requantizes or writes a compiled checkpoint.
- Native-kernel or speed claims require actual dispatch/instruction evidence, correctness evidence, raw samples,
  peak VRAM and exact code/model/toolchain revisions. Disclose changed timing boundaries and semantics.
- Do not obtain speedups by changing token IDs, output length, sampling, template, context, batch size, cache reuse,
  attention semantics or timing boundaries without explicit disclosure.

## Qualified model-profile policy

The two equal public product choices are 12B Unified and 26B Compact Vision, not a default and an experimental
alternative. Equal status does not imply identical capabilities: 12B supports qualified text, image and audio,
while Compact Vision supports text plus images bounded by context and request/storage capacity, is single-slot and may use its separately pinned fixed-D2
Assistant. The qualified text-only 26B NVFP4 profile is retained internally for regression and rollback and is not
offered in the normal Studio selection. Product surfaces must disclose capability differences and never substitute
profiles. New unqualified work remains visibly experimental/reference or diagnostic until its owning gates pass. It
must not weaken another profile, advertise unsupported capabilities, use CPU weight offload as a production path,
or silently fall back to another precision or kernel.

## Testing and evidence

Use tests proportional to risk: parser/host tests for host changes, operator and sanitizer tests for CUDA changes,
model/reference tests for numerical changes, and 12B regressions whenever shared runtime/loader/CUDA code changes.
Do not weaken tolerances to make a test pass. Hot-path changes additionally need a measured parent, numerical and
memory boundary, actual dispatch facts and adjacent end-to-end evidence. Record benchmark results as new evidence;
never overwrite prior results.

## Collaboration and hygiene

Disjoint temporary worktrees are allowed when file ownership and interfaces are explicit; integration remains
serialized at shared contracts and engine orchestration. Do not commit, push, rebase, reset, stash or change branches
unless explicitly authorized. Do not edit generated files by hand; update their generator. Do not modify unrelated
dirty files. Keep commits and patches narrow.

## Temporary experiment storage (owner decision2026-11-17)

Details and examples: `docs/EXPERIMENT_STORAGE.md`.

- Place bulky temporary profiler traces, SQLite exports, tensor dumps and offline
  diagnostic intermediates in RAM-backed `/tmp/gem16-experiments/`, not in SSD
  result directories. `/tmp` is already tmpfs on the owner's64GB-RAM Linux PC.
- Prefer `tools/experiment_scratch.py` or its context manager: verify tmpfs,
  expected capacity and available RAM before starting; remove the owned run
  directory after analysis. Keep GPU/pinned-host memory needs in the budget.
  The helper's capacity checks are preflight checks, not a hard memory quota.
- Keep compact primary samples, telemetry, correctness/dispatch summaries and
  exact source/model/toolchain identities on SSD. Retain large raw traces only
  when deliberately needed as non-regenerable evidence; prefer compact raw
  reports or verified lossless compression over duplicate SQLite exports.
- RAM scratch is ephemeral across reboot. Persist required evidence before
  removing scratch or the daily shutdown deadline. Check SSD free space too;
  do not start a run that cannot safely retain its small final results.
