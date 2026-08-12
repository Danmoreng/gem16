# AGENTS.md

## Project

`gem16` is a C++20/CUDA inference engine for Gemma models on approximately 16 GB NVIDIA GPUs; Blackwell SM120 is
its first optimized backend. The qualified Gemma 4 12B Unified path is the production baseline. Gemma 4 26B A4B is
an experimental, model-specific track. Preserve 12B behavior and do not turn its hot path into a generic framework.

## Default task loop

For every task:

1. inspect `git status`, branch, current source and relevant tests;
2. read this file, `docs/ACTIVE_DECISIONS.md`, and one narrow feature/task entry;
3. state the smallest change and any uncertainty;
4. implement only the owned slice, preserving unrelated dirty files;
5. run the smallest proportional checks, then widen them when shared runtime/CUDA behavior is affected;
6. report files, exact commands/results, limitations and evidence.

The default task entry is normally `docs/plans/gemma4-26b/START_HERE_CODEX.md` for 26B work. Read `ROADMAP.md`, the
historical `DECISIONS.md`, `CORRECTNESS.md`, `BENCHMARKING.md`, ledgers and detailed specifications only when the
specific task, changed boundary or requested claim makes them relevant. Do not make a documentation-only or isolated
host change wait for the complete repository test matrix. Do not begin a blocked milestone without an explicit active
decision or owner instruction.

## Decision precedence

Use this order when documents disagree:

1. this file's permanent runtime, security and integrity rules;
2. `docs/ACTIVE_DECISIONS.md`;
3. current source, tests and accepted evidence for facts about what exists;
4. the active feature contract, milestone status and narrow specification;
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

## Experimental 26B policy

A first 26B text-only execution may be implemented and exercised before final quality, performance or long-context
qualification. It must be visibly labeled experimental/reference or diagnostic until its owning gates pass. It must
not weaken the 12B product path, advertise unsupported capabilities, use CPU weight offload as a production path, or
silently fall back to another precision or kernel.

## Testing and evidence

Use tests proportional to risk: parser/host tests for host changes, operator and sanitizer tests for CUDA changes,
model/reference tests for numerical changes, and 12B regressions whenever shared runtime/loader/CUDA code changes.
Do not weaken tolerances to make a test pass. Hot-path changes additionally need a measured parent, numerical and
memory boundary, actual dispatch facts and adjacent end-to-end evidence. Record benchmark results as new evidence;
never overwrite prior results.

## Collaboration and hygiene

26B work uses the long-lived `feat/gemma4-26b` integration branch. Disjoint temporary worktrees are allowed when
file ownership and interfaces are explicit; integration remains serialized at shared contracts and engine
orchestration. Do not commit, push, rebase, reset, stash or change branches unless explicitly authorized. Do not edit
generated files by hand; update their generator. Do not modify unrelated dirty files. Keep commits and patches narrow.
