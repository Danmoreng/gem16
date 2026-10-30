# Trellis35 D2 graph compatibility fix, 2026-10-11

Native Trellis35 D2 runs again on the installed Linux CUDA/NVIDIA stack. The
initial failure was a host SIGSEGV during CUDA graph instantiation, before any
inference or streaming-ring access. A separate mapped-host atomic issue was
found by Compute Sanitizer after fixing the graph. These are different findings;
the atomic issue does not explain the initial native startup crash.

This follow-up completes the missing D2 row from the
[2026-10-10 campaign](llama-cpp-trellis35-refresh-2026-10-10.md). Historical
measurements and failure evidence are preserved. The new raw-sample record is
[here](../../../benchmarks/baselines/llama_cpp/gemma4-26b-trellis35-q4_0-d2-followup-2026-10-11.json).
It is bounded Linux text-engine regression/performance evidence, not a new
Windows, media or broad quality qualification.

## What changed in the machine

The local `/var/log/pacman.log` confirms these updates on 2026-10-04:

| Component | Previous | Installed after update |
|---|---|---|
| NVIDIA userspace driver | 610.57.04-1 | 615.71.09-1 |
| NVIDIA open kernel module | 610.57.04-11 | 615.71.09-6 |
| CUDA package | 13.3.1-1 | 13.4.2-1 |
| Linux kernel | 7.2.2.arch1-1 | 7.2.8.arch1-2 |
| Nsight Compute | 2026.2.1.5-1 | 2026.3.1.2-1 |
| Nsight Systems | 2026.1.3.425-1 | 2026.3.2.476-1 |

The October 9 update changed the kernel to 7.2.9 and rebuilt the 615.71.09
kernel module; it did not change the userspace driver version. Relevant update
lines are retained in `system-update-history.log` in the raw evidence directory.
Thus an unchanged repository can run differently: both the driver and CUDA
toolkit used to rebuild and execute it changed. Earlier successful native D2
execution is consistent with these findings.

## Graph-instantiation failure and isolation

The parent binary crashes inside `libcuda.so.1`, through
`cuGraphInstantiateWithFlags` / `cudaGraphInstantiate`, called by
`PrepareFixedMtpChainGraph`. It fails with both default and EAGER module loading.
This is the startup failure recorded on October 10, not a token-loop failure.

A model-free reproducer builds a WHILE node with two IF branches, a trivial
kernel and optionally one four-byte device-to-device copy. It allocates no
mapped host memory and requires no model files. With the current driver:

| Graph construction | CUDA 13.4 runtime | Cached CUDA 13.3 runtime |
|---|---|---|
| Captured kernel-only child under IF | passes, result 1 | passes, result 1 |
| Captured child containing DtoD copy under IF | SIGSEGV at instantiate | SIGSEGV at instantiate |
| Same kernel/copy captured directly into IF body | passes, result 1 | passes, result 1 |

This runtime control uses the **same CUDA 13.4-compiled binary** and installed
615.71.09 driver, changing only `libcudart` via `LD_LIBRARY_PATH`. CUDA reports
runtime versions 13040 and 13030 respectively. The old library was extracted
from the cached package into the ignored evidence directory; no package was
installed or downgraded. A runtime-only rollback therefore does not resolve the
failure. This strengthens the driver/graph interaction hypothesis but does not
establish an exact first bad driver version or recreate the complete old stack.

A second control also rebuilds the reproducer with the cached **CUDA 13.3.73
compiler, headers and 13.3 runtime**. It produces the same three outcomes under
the installed driver: kernel-only/direct-copy pass, nested-child-copy SIGSEGV.
This diagnostic uses current GCC with explicit `--allow-unsupported-compiler`;
it is not a production build or an exact recreation of the historical machine.
Together these controls argue against CUDA 13.4 alone being the cause. The
driver and updated system remain the strongest suspects. No old-driver A/B,
system downgrade, reboot or kernel-module replacement was performed.

Additional diagnostics removed node families from a captured graph **only for
instantiation testing**, then deliberately stopped without inference. Removing
all 22 DtoD copies made instantiation pass; removing memset or kernel nodes did
not. Normalizing copy pitch metadata in the minimal reproducer also did not
help. No such node removal is part of the engine fix or benchmark.

NVIDIA's current [conditional-body requirements](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#conditional-node-body-graph-requirements)
permit child graphs and device copies. The tiny failing graph is compatible
with those documented node/memory restrictions. These observations support a
compatibility defect on the installed stack; they do not justify claiming an
undocumented new prohibition or a specifically acknowledged NVIDIA bug.

The engine now captures the D2 and ordinary-tail work directly into the
conditional-owned body with `cudaStreamBeginCaptureToGraph`, instead of inserting
an additional captured child graph. Kernels, copies, fork/join dependencies,
conditional WHILE/two-IF decisions, cache transactions and token semantics are
preserved. Body ownership remains with the root graph. No copy was omitted,
precision changed or fallback added. The same failure also reproduced in the
protected 12B D2 path, which received the corresponding construction fix.

## Separate streaming-ring sanitizer finding

After the graph-only fix, native Trellis35 D2 already completed an eight-token
16K-prompt probe and exact 64-token regressions with the original streaming-ring
code. Compute Sanitizer then reported an invalid eight-byte system-scope atomic
on an in-bounds mapped-host ring counter, followed by sticky errors. Its first
access is an `atomicAdd_system(counter, 0)` used as a load. This was not the
earlier graph-instantiation SIGSEGV.

The GPU reports `cudaDevAttrHostNativeAtomicSupported=0`. The ring formerly used
RMW atomics for loads, producer publication and a single-writer backpressure
counter. It now uses aligned 64-bit system-scope acquire loads and release
stores; the CPU side already uses acquire/release `std::atomic_ref`. Payload
writes precede the producer release store. Single-writer backpressure increments
use load/store; no multi-writer RMW requirement is introduced.

The [CUDA memory model](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cuda-cpp-memory-model.html#atomicity)
distinguishes mapped-memory native atomics from aligned system-scope loads and
stores. The restriction was already documented in
[CUDA 12.6 mapped-memory guidance](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html#mapped-memory).
Earlier native success does not establish the host/device atomicity guarantee,
and this investigation does not prove when sanitizer detection changed. It
would be incorrect to describe this as a newly introduced 615 driver rule.

The focused conditional-copy test, mapped-ring control test and native
Trellis35 D2 probe now each report **Compute Sanitizer: 0 errors**. The native
sanitizer probe has a short prompt, eight outputs and 17,519 context capacity;
the full 16K/1,135 benchmark runs are uninstrumented.

## Completed comparison

Three warmups and ten retained samples; medians:

| Engine / mode | Prefill tok/s | TTFT ms | Decode tok/s | Aggregate ITL ms | Sampled peak VRAM MiB |
|---|---:|---:|---:|---:|---:|
| gem16 Trellis35 ordinary, fixed source | 5,665.11 | 2,892.09 | 129.087 | 7.747 | 12,455 |
| gem16 Trellis35 native fixed D2 | 5,665.47 | 2,891.905 | 172.314 | 5.803 | 12,725 |
| llama.cpp b11552 QAT Q4_0 ordinary, October 10 | 4,607.79 | 3,555.716 | 138.461 | 7.222 | 15,201 |
| llama.cpp b11552 QAT Q4_0 D2, October 10 | 4,445.20 | 3,685.770 | 196.850 | 5.080 | 15,561 |

Both gem16 rows were freshly measured after the fix. llama.cpp rows reuse the
completed October 10 measurements of the same current b11552 binary on the same
installed stack. No old llama.cpp binary was executed and no historical number
was relabeled as a fresh run.

Trellis35 D2 improves decode by **33.49%** over its new ordinary row. It accepts
632/1,004 proposals (62.95%), rejects 372, and executes 502 D2 verification
groups per measurement, with no ordinary fallback tokens. All twenty retained
gem16 runs produce the identical 1,135-token vector, SHA-256
`9bc749fe3419fb50806e2b7124bacb7ad5d089d44890ba588842be708a3ab247`.
The new ordinary output also exactly matches the parent ordinary campaign.
Its median decode is +0.10% versus the measured parent; this is not a tuning
speedup claim. Native D2 was unavailable in that parent environment.

Current llama.cpp is 7.26% faster in ordinary decode and 14.24% faster in D2 for
these rows. Trellis35 prefill is 22.95% faster in ordinary mode and its D2 uses
2,836 MiB less sampled peak VRAM. llama D2 differs from llama ordinary at
zero-based output index 101; cross-engine outputs also differ. These are
descriptive throughput ratios, not equal-quality or equal-output comparisons.

The common protocol retains exactly 16,384 prompt token IDs, 1,135 forced greedy
outputs, 1,134 measured post-first-token intervals, batch one, cold prompt and
suppressed IDs `[1, 50, 106, 258882, 258883]`. Requested context is 17,519;
llama allocates 17,664. Native Trellis35 W4A8/FP8 KV and mixed native assistant
remain different from official llama QAT Q4_0/Q8_0 KV and the separate GGUF
assistant. All weights remain GPU-resident. Loading/initial plan creation are
excluded; gem16 uses fresh processes, llama a resident server with full prompt
recomputation. Detailed commands, hashes, model locks and timing boundaries are
retained in the records, together with device-wide 200 ms telemetry.

New gem16 decode sample means and 95% Student-t intervals of the mean are
129.101 [129.048, 129.154] ordinary and 172.342 [172.212, 172.472] D2 tok/s.
Firmware clocks remain dynamic: ordinary/D2 observed temperatures reach 74 C,
with peaks of 179.11/177.66 W. These intervals characterize this repeated
fixture, not a workload distribution.

## Regression, dispatch and reproducibility

Base source is `c01789a2b46520df8a699ee3338cb19aaccebae2` plus the local fix;
`followup-provenance.json` records binary, runtime-source, CMake, test, harness
and patch hashes before measurement. No commit or branch change was made.
Hardware remains the RTX 5080 Laptop GPU, 16,303 MiB, driver 615.71.09, CUDA
13.4.92, GCC 16.2.1 and SM120a Release build.

The owned source slice is the 26B/12B conditional graph construction, mapped
SPSC ring access, a new model-free conditional-copy regression, and an added
ring cancellation-during-backpressure case. The copy test alternates both
branches across twenty replays with three iterations each, bit-exact arbitrary
patterns including signed zero/NaN payloads and cross-stream fork/join copies.
The ring test checks cancellation with a full 256-token ring as well as the
existing replay/backpressure behavior.

Checks completed:

- Focused conditional-graph, Trellis35, attention-reference and NVFP4-consumption
  CUDA tests: 4/4 passed, 4.96 s.
- Complete `gem16-cuda` test executable: passed, 31.04 s.
- Host unit/allocation checks: 2/2 passed. Python benchmark harness: 18 passed.
- Final protected `gem16-12b-m22-product`: passed, 3.61 s.
- Parent-ordinary versus fixed-D2 64-token checks for 12B, Trellis35 and internal
  26B NVFP4: exact token equality, zero fallbacks, no token-loop allocations.
- Three focused memcheck runs described above: zero errors, no tolerance changes.
- Full fixed-fixture Trellis35 ordinary/D2 benchmark: 3+10 each, exact output
  parity, zero fallbacks and no token-loop allocations in every retained run.

The internal NVFP4 cache root contains additional profile directories that the
strict old loader rejects. Its regression used fully verified hardlinked views
of the unchanged locked runtime files under the evidence directory. No loader
validation was weakened, model repacked or second weight payload downloaded.

An eight-token Nsight decode-only capture with the exact 16K prompt, context and
common suppression IDs confirms actual grouped-T3/selected W4A8 projection,
fixed-row FP8 attention, GPU accept/advance and conditional continuation kernels.
Its output matches the full D2 prefix. Diagnostic profiling timings are excluded
from the benchmark and no new instruction-level efficiency claim is made.

The ignored raw directory is
`benchmarks/results/2026-10-10/c01789a2/d2-graph-fix/` (the investigation crossed
midnight). It retains parent binary, minimal reproducer, graph dumps and
diagnostic failures, runtime controls, exact commands, all new sample JSON,
telemetry, sanitizer/test logs, model regressions, Nsight capture and generators.
The tracked JSON identifies raw paths and hashes. Earlier failed setup attempts
remain separate from valid measurements.

Relevant check invocations, from the repository root:

```bash
cmake --build build/Linux/blackwell-release --parallel 4 --target gem16-run
ctest --test-dir build/Linux/blackwell-release --output-on-failure \
  -R '^gem16-cuda-(conditional-graph|trellis35|attention-reference|nvfp4-consumption)$'
ctest --test-dir build/Linux/blackwell-release --output-on-failure -R '^gem16-cuda$'
ctest --test-dir build/Linux/blackwell-release --output-on-failure \
  -R '^gem16-(unit|responses-allocation)$'
ctest --test-dir build/Linux/blackwell-release --output-on-failure \
  -R '^gem16-12b-m22-product$'
compute-sanitizer --tool memcheck --error-exitcode 99 \
  build/Linux/blackwell-release/bin/gem16-cuda-conditional-graph-tests
compute-sanitizer --tool memcheck --error-exitcode 99 \
  build/Linux/blackwell-release/bin/gem16-cuda-tests mtp-control
python -m unittest tests.python.test_benchmark_wikipedia_workload \
  tests.python.test_benchmark_gemma4_26b_wp8_text
git diff --check
```

The full model probe/benchmark commands are saved as JSON argument arrays.
`followup-run.py` records build provenance and refuses to overwrite existing
measurements. A rerun must use a new output directory; `summarize-followup.py`
derives the machine-readable record from raw evidence, including the reused
current llama rows and their original identities.
