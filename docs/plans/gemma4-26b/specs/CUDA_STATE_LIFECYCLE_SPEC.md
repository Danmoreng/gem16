# CUDA state and arena lifecycle specification

## Goal

Prevent process-lifetime CUDA state from retaining pointers into model/runtime arenas after those arenas are destroyed.

## Ownership rule

Every device or pinned-host pointer has exactly one owner and a declared lifetime. Module-static caches may contain only:

- immutable process-wide objects whose allocation outlives every engine; or
- nullable borrowed pointers with a registered reset hook tied to the owning arena teardown.

A static pointer into an `ExecutionSlot`, model arena, graph-private arena or reusable workspace is forbidden without a tested reset path.

## Teardown order

The owner must document and test:

1. stop request/streaming threads;
2. synchronize or cancel outstanding GPU work;
3. destroy graphs/events/streams that reference arena memory;
4. release the arena;
5. reset every module-static borrowed pointer and cached size/descriptor;
6. query CUDA sticky error state;
7. allow a new engine to initialize in the same process.

Reset-before-free is not automatically correct: the exact order follows pointer validity and asynchronous work. The implementation must state why its order is safe.

## Required tests

- create/load/infer/destroy repeated at least 10 times in one process;
- two engines sequentially using the same model handle where supported;
- failure during initialization followed by a clean second initialization;
- failure during graph capture followed by teardown/relaunch;
- context creation failure must not be converted into a skip while CUDA is in an error state;
- `cudaDeviceSynchronize`/`cudaGetLastError` clean after teardown;
- VRAM returns to the declared process baseline within tolerance;
- no stale descriptor/workspace pointer under ASAN/compute-sanitizer-compatible lanes.

## Evidence

Lifecycle reports include:

- allocation IDs and owners;
- reset hooks invoked;
- CUDA error state per cycle;
- device free-memory before/after;
- process peak;
- cycle count and exact failure injection.
