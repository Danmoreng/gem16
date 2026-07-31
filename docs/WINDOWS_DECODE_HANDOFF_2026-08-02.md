# Windows decode optimization handoff — 2026-08-02

Purpose: continue the decode optimization program on Windows after the qualified Linux global-attention work.
Windows measurements must establish a new Windows parent; do not compare Windows timings directly with Linux.

## Read first

1. `AGENTS.md`
2. this handoff
3. `docs/DECODE_OPTIMIZATION_PLAN.md`
4. the newest entries in `docs/PERFORMANCE_LEDGER.md`
5. `docs/DECISIONS.md`

## Required semantics

- Direct `unsloth/gemma-4-12b-it-NVFP4` loading; no persistent converted gem16 checkpoint.
- Checkpoint FP8 K/V for gem16.
- llama.cpp comparisons use Q8_0 K/V, Flash Attention, and full GPU residency.
- Batch one; Ordinary output is the Target truth for MTP.
- Fixed-D2 proposed tokens are never reported as output throughput.
- No token-loop allocation, precision fallback, CPU weight offload, or dynamic compilation.
- Keep Prefill paused until the Ordinary/D2 decode gates are complete.

## Linux handoff commits

The following logical performance commits precede or contain this handoff:

```text
da65217  attention: vectorize 64K FP8 decode loads
d5dcbea  attention: stage global GQA cache tiles
<this handoff commit>  scalar-order all-head GQA at 16K/32K
```

Resolve and record the full Windows parent SHA after pulling `origin/main`. Build that exact commit before editing.

## What changed

### 64K FP8x4 global GQA

For every 512-token global split, one 256-thread CTA now covers all 16 D512 query heads. Each of eight warps owns
two heads. A 16-token physical K/V tile is copied once with aligned `cp.async`, decoded as E4M3x4, and reused by
all warps. The prior kernel loaded the shared single K/V head once for each of four query groups.

The 512-token split boundaries, online-softmax state, FP32 LSE merge, K/V bytes/scales, and workspace topology are
unchanged. Fixed-D2 invokes the same primitive once per row, retaining exact Ordinary Target arithmetic.

### 16K/32K scalar-order global GQA

The same all-head CTA is used from cache capacity 16,384 through 65,535, but dimensions remain assigned as
`lane + 32*i` with scalar E4M3 conversion. This preserves the established shorter-tier FP32 QK reduction order.
At capacity 65,536 and above, the FP8x4 specialization remains selected. Below 16K, the former four-query-head
grouped path remains selected pending measurement.

Dispatch is fixed when the immutable context plan/CUDA Graph is prepared; no per-token kernel selector was added.

Relevant files:

```text
src/cuda/attention/decode_sm120.cu
src/cuda/attention/sm120.h
src/runtime/result_json.cpp
tests/cuda/nvfp4_reference_test.cu
docs/PERFORMANCE_LEDGER.md
```

## Qualified Linux results

Hardware: RTX 5080 Laptop, SM120, CUDA 13.3. These numbers prove Linux code gains only; they are not Windows
performance claims.

### Synthetic decode matrix, 3 warm-ups / 10 measured / 256 generated

| Existing context | Parent tok/s | Final tok/s | Change | Output |
|---:|---:|---:|---:|---|
| 16,384 | 31.780 | 32.861 | +3.40% | exact parent checksum |
| 32,768 | 29.245 | 31.461 | +7.58% | exact parent checksum |
| 65,536 | 27.261 | 30.263 | +11.01% | exact parent checksum |

At 64K, p50/p95/p99 ITL changed from 36.642/38.041/39.484 ms to
32.978/34.440/36.184 ms. Workspace remained 800,530,176 bytes.

At 16K, p50/p95 changed from 31.393/32.476 ms to 30.361/31.405 ms. At 32K, p50/p95 changed from
34.062/35.374 ms to 31.737/32.720 ms.

### Exact Wikipedia 16K, 3 warm-ups / 10 measured / 1,135 output

| Mode | Retained parent | Final Linux | Change |
|---|---:|---:|---:|
| Ordinary | 31.472 tok/s | 32.919 tok/s | +4.60% |
| Fixed-D2 | 46.248 tok/s | 50.806 tok/s | +9.86% |

Ordinary and D2 both retain all 1,135 token IDs and SHA-256:

```text
43bc3380fc1cce5182a679fa3a340c04bcc79c52e73d5102ec1f737f57d0a1e1
```

Every D2 run retains 1,004 proposed, 632 accepted, 372 rejected, 502 groups, and mean accepted length 1.25896.
The hard fixed-D2 target remains 64.82 tok/s; it has not been reached.

Current Linux external references on the same workload remain:

```text
llama.cpp Ordinary/D2: 33.386 / 54.703 tok/s (qualified 3/10)
vLLM Ordinary/D2:      35.100 / 56.355 tok/s (one-run orientation only)
```

Gem16 now nearly closes Linux Ordinary at 16K but remains behind in D2.

## Profile evidence

The 64K child-node profile reduced the eight global split kernels from 8.112 to 3.749 ms/token (-53.8%). The
ordinary FP8x4 GQA kernel reports 64 registers/thread, 41,056 bytes static shared memory in Nsight Systems, and
zero local memory/thread. `cuobjdump` reports 42,080 bytes including compiler accounting. Scalar ordinary/D2 use
68/70 registers respectively, the same compiler-accounted shared memory, and no stack/local storage.

A corrected llama.cpp trace used combined `llama-bench -pg 16384,4`. The older `-p N -n 4` trace had run prompt
processing and generation as separate tests (`n_prompt: 0` for generation), so its large `mul_mat_q<...,128>`
kernels were Prefill, not context-preserving Decode. Actual M=1 llama.cpp uses `mul_mat_vec_q`; measured projection
families were approximately 11.15 ms/token NVFP4 and 6.62 ms/token Q8_0, versus gem16's prior 8.59/5.17 ms
NVFP4/FP8. Therefore copying llama.cpp's Prefill MMQ schedule into gem16 T=1 was rejected.

Artifacts:

```text
benchmarks/results/2026-08-02/da65217/blackwell-linux-kernel-comparison/
benchmarks/results/2026-08-02/da65217/blackwell-linux-global-gqa-tile16-chunk512/
benchmarks/results/2026-08-02/d5dcbea/blackwell-linux-global-gqa-scalar-tier16k/
```

Result directories are normally ignored and may not be present on Windows. The ledger and this handoff contain the
promotion evidence.

## Correctness and build status

Final Linux release build and CTest pass:

```text
cmake --build --preset blackwell-release --parallel 4
ctest --preset blackwell-release --output-on-failure
# 2/2 tests passed
```

The CUDA suite includes:

- the 64K global FP8 operator/reference fixture;
- four-run bit determinism;
- a dedicated three-row 64K D2 fixture requiring bitwise equality with three independent Ordinary GQA calls.

No precision, K/V representation, persistent weights, token-loop allocation, or fallback changed.

## Windows restart procedure

From the Windows repository root:

```powershell
git fetch origin
git switch main
git pull --ff-only origin main
git status --short
git rev-parse HEAD

cmake --preset blackwell-release
cmake --build --preset blackwell-release --parallel
ctest --preset blackwell-release --output-on-failure
.\build\Windows\blackwell-release\bin\gem16-run.exe --print-kernel-capabilities
```

Require SM120/SM120a native NVFP4 and FP8 capability. Record driver, CUDA toolkit/runtime, power mode, clocks,
temperature, display ownership, and free VRAM. Do not reuse Linux throughput as the Windows parent.

First Windows measurements:

1. Ordinary 16K/32K/64K, 256 output tokens, 1/3 screen.
2. If stable, 3/10 qualification at all three contexts.
3. Exact Wikipedia 16K Ordinary and Fixed-D2, fixed 1,135 output tokens, 3/10.
4. Verify the exact Ordinary/D2 hash above before any new MTP optimization.
5. Capture child CUDA Graph nodes with `tools/profile_windows.ps1`; if child attribution is unavailable, do not
   infer a kernel win from the outer graph duration alone.

## Next optimization target

Do not return to small global-attention geometry probes. The all-head path removed the largest global K/V traffic.
The next Windows profile should rank Fixed-D2 components, especially:

1. Target T=3 FP8 and NVFP4 projections;
2. Target batched output head;
3. assistant BF16 GEMVs and assistant output head;
4. local D256 fixed-window attention;
5. only then launch/control overhead.

Also screen scalar-order all-head GQA at 8K before lowering its 16K threshold. Promote it only with an adjacent
end-to-end win and exact output.

Every new winner must be committed before proceeding to the next optimization, with correctness, workspace,
resource usage, and qualified benchmark evidence recorded in `docs/PERFORMANCE_LEDGER.md`.
