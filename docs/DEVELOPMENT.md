# Development guidelines

This document contains the implementation rules shared across gem16 features. Model arithmetic and execution
architecture belong in their feature specifications and `ARCHITECTURE.md`.

## Languages and runtime dependencies

- Runtime code uses C++20 and CUDA C++. Use C only for narrow system interfaces.
- Python is for tooling, reference generation, validation, and benchmark orchestration; it is not embedded in the
  inference runtime.
- Keep the runtime dependency surface small. CUDA Runtime/Driver APIs, pinned CUTLASS/CuTe, audited parsers,
  profiling annotations, and test-only frameworks are appropriate.
- Do not introduce PyTorch, Transformers, vLLM, TensorRT-LLM, dynamic Python, a JIT compiler, or a generic graph
  framework into the production runtime. Reference tools may use them.
- Every vendored dependency requires an immutable source revision, license review, retained notices, checksums when
  practical, update instructions, and a documented reason. See `../third_party/README.md`.

## C++

- Use RAII for CUDA handles, mapped files, allocations, streams, events, and other owned resources.
- Mark fallible results `[[nodiscard]]`. Prefer explicit `Status`/`Result<T>` in runtime code over exceptions.
- Use fixed-width integers for file formats and allocation arithmetic.
- Check overflow before shape products, byte calculations, alignment, offsets, and narrowing conversions.
- Packed nibble and byte access must not rely on undefined behavior.
- Do not reinterpret pointers without a proved size, lifetime, and alignment contract.
- Prefer spans and typed views over unrelated raw pointer/length pairs.
- Avoid virtual dispatch and shared ownership in hot paths. Use shared ownership only where the lifetime contract
  genuinely requires it, such as the immutable model runtime.
- Do not add global mutable state or hidden singletons. CUDA module-static state must be reset safely across engine
  destruction, reconstruction, and failed initialization.
- Keep public headers small and preserve existing ownership boundaries.
- Build with warnings enabled and do not hide new warnings globally.

## CUDA

- Check every CUDA API result and return an actionable error containing the failed operation.
- Check launch errors in debug/test builds and at existing asynchronous error boundaries.
- Do not add `cudaDeviceSynchronize()` to a hot path. Use stream ordering and events for dependencies.
- No recurring `cudaMalloc`, `cudaFree`, pageable allocation, graph capture, module load, or workspace discovery is
  allowed after execution-plan initialization.
- Use asynchronous copies only with explicit source lifetime and ordering. Avoid unnecessary host round trips.
- Add NVTX ranges around model phases and hot operators when profiling visibility would otherwise be ambiguous.
- Document hot-kernel block dimensions, tile shape, dynamic shared memory, expected occupancy, and numerical
  reduction order.
- Record register count, stack frame, local memory, spills, and representative instruction path for promoted hot
  kernels. Local-memory spills require measured end-to-end justification.
- Use `__restrict__` only when non-aliasing is part of the binding contract.
- Complex fused kernels require a CPU or CUDA reference path available to tests or diagnostics. A reference path
  must not become an automatic production fallback.
- Add stream or persistent-kernel complexity only after a timeline shows a relevant gap and an end-to-end
  benchmark proves the benefit.
- CUDA Graph addresses, workspace offsets, and plan shapes remain fixed after capture. Graph demotion or capture
  failure must be observable.

## Model and file handling

- Treat checkpoint, tokenizer, media, JSON, GGUF, Safetensors, and index data as hostile input.
- Validate JSON/header lengths, shape products, offset bounds, overlap, dtype, alignment, shard paths, tensor byte
  lengths, and requested allocations before use.
- Reject duplicate tensors and path traversal. Do not normalize tensor names except through explicit versioned
  maps.
- Memory-map large source files and use bounded staging; do not copy an entire checkpoint into ordinary host RAM.
- Upload into final device allocations. A load-time layout transform may preserve codes/scales exactly but must not
  leave a second persistent GPU copy.
- Tied embeddings/output heads use one physical resident allocation.
- Never execute code from a model repository or use `trust_remote_code` in the runtime.

## Correctness and tests

Run the smallest applicable gate first and widen only after it passes:

1. formatting/static host checks;
2. parser and host unit tests;
3. synthetic CUDA operator tests;
4. real-shape or checkpoint-backed operator tests;
5. layer/state/logit comparisons;
6. deterministic generation and cache tests;
7. long-context, lifecycle, and soak tests;
8. quality and benchmark suites.

Tests for packed and quantized operators should cover exact tile multiples, legal tails, minimum and representative
shapes, zero scales/blocks, extrema, rounding ties, malformed input, and real checkpoint bytes. Attention and cache
tests should cover local-window wrap, global growth, chunk boundaries, context limits, and separate K/V semantics.

Do not relax tolerances solely for performance. A tolerance change needs a failing example, numerical explanation,
quality impact, and updated golden evidence. Performance changes require fresh correctness evidence at the changed
arithmetic boundary.

Use host sanitizers and Compute Sanitizer where applicable. Lifecycle-sensitive CUDA work must test repeated
create/destroy and failed-initialization cleanup.

## Performance work

Before implementation, retain:

- exact parent commit and dirty state;
- benchmark command and raw parent result;
- profile identifying the bottleneck;
- expected limiting resource;
- numerical and memory behavior;
- reference/test route.

After implementation, retain:

- exact code/model/toolchain revisions;
- compiler and disassembly facts;
- actual runtime dispatch and graph status;
- repeated adjacent benchmark distributions;
- correctness and quality deltas;
- persistent, workspace, graph, and peak-VRAM deltas;
- relevant Nsight artifacts.

Follow `BENCHMARKING.md`. Do not compare across different prompts, token counts, contexts, sampling, cache state,
batch size, power state, or timing boundaries without labeling the difference. Proposed speculative tokens are not
accepted output tokens.

## Generated code

Generated tables or source files are allowed only when:

- the generator is checked in;
- generation is deterministic;
- output contains provenance;
- CI or a test verifies generated files are current;
- contributors do not edit generated output manually.

## Changes and commits

- Keep changes narrow and preserve unrelated dirty-worktree content.
- Avoid opportunistic refactors while changing arithmetic, memory layout, or performance-sensitive code.
- Update documentation in the same logical change when behavior or evidence changes.
- Suggested commit prefixes include `loader:`, `runtime:`, `cuda:`, `attention:`, `moe:`, `bench:`, and `docs:`.
- Coding agents do not create commits unless the user explicitly requests one.
