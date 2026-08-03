# Decode optimization plan

Status: active execution goal

Target machine: NVIDIA GeForce RTX 5080 Laptop GPU, compute capability 12.0, approximately 16 GB VRAM

Execution environment: optimization moved from the reference Arch installation back to Windows on 2026-08-02 at
the user's request. The qualified Linux promotions remain evidence for their commits, but Windows must establish
an OS-local parent before further performance changes; cross-OS A/B claims remain invalid. Continue from
`WINDOWS_DECODE_HANDOFF_2026-08-02.md` and retain the machine facts in `toolchains/blackwell16gb.lock`.

Target workload: direct `unsloth/gemma-4-12b-it-NVFP4` loading, text-only, batch one, checkpoint FP8 KV cache;
the comparison llama.cpp run uses Q8_0 KV cache as requested

Primary competitor: pinned current upstream llama.cpp with CUDA SM120a, native NVFP4 MLP execution, Q8_0-mapped
attention weights, Q8_0 KV cache, full GPU residency, and no tensor split

Secondary Linux reference: pinned vLLM loading the exact Hugging Face checkpoint directly with FP8 KV cache. vLLM
is used to expose kernel and hardware headroom and to compare direct-checkpoint behavior; it does not replace the
llama.cpp competitive gate, and non-identical MTP output remains a disclosed characterization rather than an exact
speculative speedup.

## Objective and hard gates

Decode is the first performance priority. Prefill optimization is paused except for correctness fixes and
regression prevention until the decode gates in this document pass.

The mandatory product target is the fixed Wikipedia 16K, fixed-D2 MTP workload:

- at least **64.82 effective verified output tokens/s** as a 3-warm-up/10-measured-run median;
- exact identity with gem16 ordinary Target output in every measured run;
- no prompt-cache hit, CPU weight offload, hidden precision fallback, token-loop allocation, or reduced output;
- no worse p95 group latency, peak VRAM, or thermal behavior hidden behind the throughput number.

The absolute 64.82 token/s target does not become easier when llama.cpp improves. Final competitive promotion also
requires a fresh adjacent run against the pinned llama.cpp candidate on the same machine and conditions:

```text
gem16_mtp_effective_tps / llama_cpp_mtp_effective_tps >= 1.00
```

The stretch target is at least 70.0 effective tokens/s and at least 1.05x the fresh llama.cpp result, whichever is
stricter.

The current Linux max-power adjacent D2 comparison at `8e86cb38` exceeds both fixed-workload thresholds: gem16
reaches 89.58 tok/s versus llama.cpp b10240 at 82.88 tok/s (1.081x) and vLLM 0.26.0 at 81.95 tok/s (1.093x), with
its fixed ordinary-Target hash preserved. This is a controlled performance result, not external output parity.
The ordinary multi-context gate below remains open.

MTP optimization must not conceal a slow Target model. Before final MTP promotion, ordinary gem16 decode must equal
or exceed the freshly qualified llama.cpp candidate at 128, 2,048, 8,192, 16,384, 32,768, and 65,536 existing
tokens. The primary ordinary metrics are median output tokens/s and median/p95 inter-token latency over 256 output
tokens. No shorter-context point may regress by more than 1% to buy a long-context win unless a separate context-
tier dispatch makes both paths winners.

## Current adjacent fixed-D2 comparison

The 2026-08-03 Linux 3-warm-up/10-measurement run uses the exact 16,384-token prompt and 1,135-position output
budget, batch one, max-power plus `nvidia-powerd`, and no CPU offload or prompt-cache reuse:

| Engine | Effective tok/s | ITL | Target batches | Accepted/rejected drafts |
|---|---:|---:|---:|---:|
| **gem16 `8e86cb38`** | **89.58** | **11.163 ms** | 509 | 625/391 |
| vLLM 0.26.0 | 81.95 | 12.202 ms | 542 | 590/493 |
| llama.cpp b10240 | 82.88 | 12.065 ms | 519 | 616/419 |

Gem16 is 9.31% faster than vLLM and 8.08% faster than llama.cpp on effective target-verified output. Complete
configuration, distributions, telemetry, and format/semantic limitations are in
`benchmarks/baselines/cross_engine_mtp/characterization.json`.

## Historical screening baseline

These are Windows one-warm-up/one-measured-run screening values from 2026-07-31 at gem16 commit `5501b52`. They
are retained as history, not as the parent for the resumed Windows work. Linux subsequently established qualified
baselines and promoted vector/global-GQA attention through the 2026-08-02 handoff. Build and qualify the pulled
handoff commit on Windows before collecting any new A/B result.

The llama.cpp candidate is upstream commit `000547513f1530346ecd163db8b3e13962949961`, built for SM120a with
`GGML_CUDA_FA_ALL_QUANTS=ON`. Its same-source closest-parity GGUF is 9,366,658,112 bytes and contains 144 NVFP4,
184 Q8_0, 626 F32, and one BF16 tensor. The source FP8 attention tensors are mapped to Q8_0; therefore this is
similar in size and purpose, not exact tensor-format parity with gem16's direct mixed FP8/NVFP4 checkpoint.

### Ordinary decode

| Existing context | llama.cpp tok/s | gem16 tok/s | gem16/llama.cpp | gem16 latency gap |
|---:|---:|---:|---:|---:|
| 128 | 40.54 | 38.748 | 0.956x | +1.14 ms/token |
| 512 | 46.60 | 39.625 | 0.850x | +3.78 ms/token |
| 2,048 | 46.68 | 39.218 | 0.840x | +4.08 ms/token |
| 8,192 | 46.14 | 38.259 | 0.829x | +4.46 ms/token |
| 16,384 | 44.32 | 36.504 | 0.824x | +4.83 ms/token |
| 32,768 | 42.04 | 34.200 | 0.814x | +5.45 ms/token |
| 65,536 | 37.25 | 29.862 | 0.802x | +6.64 ms/token |

From context 512 to 65,536, gem16 falls by 24.6%, versus 20.1% for llama.cpp. The incremental gem16 latency is
8.25 ms/token over that interval; llama.cpp adds 5.39 ms/token. Because Gemma has forty 1,024-token sliding layers
and only eight context-growing global layers, the additional 2.86 ms/token slope gap is strong evidence that global
attention is the first long-context target. It is not proof that projections are already fast enough.

### Fixed-D2 MTP at 16K

Both rows process the same 16K Wikipedia token-ID prompt and 1,135-token output budget, but their generated outputs
differ. The result is a speed bound, not a quality-parity claim.

| Engine | Effective tok/s | Target batches | Accepted/rejected drafts | Mean accepted |
|---|---:|---:|---:|---:|
| llama.cpp | 64.822 | 519 | 616/419 | 1.187 |
| gem16 | 53.643 | 502 | 632/372 | 1.259 |

Gem16 accepts more assistant drafts and uses fewer Target batches, yet is 17.2% slower in throughput. Approximate
Target-group time is 42.11 ms for gem16 versus 33.71 ms for llama.cpp. Holding gem16's current acceptance and batch
count constant, 64.82 token/s requires at most 34.85 ms/group: a 17.25% group-latency reduction. Improving only
acceptance is therefore the wrong first lever.

Raw local artifacts are retained under
`benchmarks/results/2026-07-31/5501b52/blackwell16gb-windows-llama-q8-screen/`.

## What llama.cpp is likely doing better

These are source- and measurement-backed hypotheses. Phase 0 must confirm the exact selected kernels and costs
before an implementation is promoted.

1. **Context-shaped quantized attention.** llama.cpp's CUDA selector can use its vector Flash-Attention kernel for
   quantized KV with the local D256 shape. D512 is outside that vector kernel and is eligible for the GQA-aware MMA
   family. Gem16 currently uses the same scalar-FMA split/merge family for both geometries.
2. **Less intermediate global-attention traffic.** At 16K, every gem16 global layer launches 32 split groups for
   each active query group and then a merge. It materializes FP32 partial output and log-sum-exp data. At 64K this
   becomes 128 splits. The bytes and merge work scale with context in addition to the required FP8 K/V reads.
3. **More mature batch-one matrix kernels.** llama.cpp has Blackwell-specific NVFP4 MMQ configurations for M=1 and
   a range of output geometries. Gem16 also executes native block-scaled MMA, but the current direct T=1 FP8 and
   NVFP4 schedules have not been compared kernel-for-kernel against the new llama.cpp build.
4. **Better specialization across T=1 and T=3.** llama.cpp does not require its MTP verifier to reuse one generic
   prompt path. Gem16 already has decode-sized verifier kernels, but historical 16K profiles still attribute most
   group time to attention and projection families rather than host control.
5. **Format differences can favor llama.cpp.** The llama.cpp candidate maps source FP8 attention weights to Q8_0;
   gem16 preserves source FP8 attention semantics. Q8_0 may happen to feed a faster mature MMQ path. Gem16 must
   improve its native FP8 path; converting the primary checkpoint or silently changing precision is not an allowed
   shortcut.

Historical fixed-D2 profiles assigned approximately 47.7% of GPU time to local plus global attention and 33.6% to
FP8 plus NVFP4 projections. Graph/control variants moved total time only about 1.5%. These profiles predate the
current split/merge implementation and are directional evidence only. A current Windows Nsight Systems capture
shows the outer decode graph launches but does not expose its internal kernel attribution, so it cannot authorize
a kernel rewrite by itself. Linux profiling replaces this incomplete trace.

## Ordered implementation program

Work proceeds in this order. Each phase has a stop/go gate; a later phase does not excuse failure to measure the
earlier one.

### 0. Produce a trustworthy current-head decode profile

Add a profiling-only execution mode that launches the exact production graph nodes without graph encapsulation,
or enable tool-supported child-node tracing. It must be bit-identical to the production CUDA Graph and must never
be used for throughput claims. Keep the production benchmark on its normal full graph.

Instrument these boundaries with NVTX and machine-readable timing:

- complete token/Target group;
- each of the 48 layers and local versus global layer totals;
- Q/K/V prefix, KV append, attention split, attention merge, O projection;
- MLP normalization/quantization, Gate, Up, activation, and Down;
- tied output head and selection;
- assistant proposal, Target verification, acceptance, and commit for MTP.

Capture adjacent Linux Nsight Systems and Nsight Compute evidence for gem16 and llama.cpp at 512, 16K, and 64K
ordinary decode plus the fixed 16K D2 workload. Capture vLLM at the common context points that fit without offload.
Record kernel names, calls/token, GPU duration, DRAM bytes and bandwidth, L2 hit rate, issue stalls, achieved
occupancy, register/shared/local memory, and clocks. Disassemble the hot kernels and prove the intended FP8/NVFP4
instructions rather than inferring them from a function name.

Phase 0 is complete when at least 90% of ordinary token GPU time and 90% of fixed-D2 Target-group GPU time are
assigned to named families, and the exact llama.cpp attention/MMQ dispatches are captured rather than guessed.

### 1. Fix global decode-attention scaling

This is the first implementation phase because it is the only work that directly explains the widening 512-to-64K
gap.

Start from the retained scalar FP8 implementation and test bounded changes in increasing risk order:

1. vectorize physical FP8 K/V loads and E4M3 conversion, use contiguous absolute addressing, and reuse the single
   global KV head across all 16 query heads;
2. tune chunk size, CTA count, and query-head grouping by the explicit 8K/16K/32K/64K context tiers, launching only
   active splits rather than capacity-only work where graph-safe control permits it;
3. keep online maximum/sum and value accumulation in registers or shared memory and reduce FP32 partial-output/LSE
   traffic; use a hierarchical or cluster-local reduction only when multiple CTAs are needed for occupancy;
4. prototype a D512 FP8 GQA MMA path only after profiling proves the scalar path is arithmetic-bound or llama.cpp
   actually selects its MMA family. Treat the 16 query heads sharing one KV head as the useful M dimension, avoiding
   an M=1 tensor-core tile dominated by padding;
5. specialize a three-row D2 variant from the winning ordinary primitive, sharing K/V staging across rows and
   query heads without changing causal masks or the exact verification order.

The phase target is not merely a faster attention microbenchmark. At 16K, ordinary decode must reach at least the
qualified llama.cpp median, and the 512-to-64K incremental latency slope must be no worse than llama.cpp's adjacent
run. The phase must reduce the combined eight global-layer time enough to explain its end-to-end gain.

### 2. Reduce the fixed cost of forty local-attention layers

Local attention stops growing after the 1,024-token window, but it runs in five sixths of the layers and therefore
sets a large fixed floor.

- Build a D256, eight-KV-head, two-query-head-per-KV vector path over the circular FP8 cache.
- Process the full 1,024-token ring without the current four-way FP32 partial-output plus merge traffic when one
  CTA per query group is faster; otherwise keep a small fixed split selected by evidence.
- Stage each K/V row once for its two query heads and use vector FP8 loads/conversion.
- Fuse only proven memory boundaries: controlled Q/K normalization and RoPE into cache preparation, or KV append
  into attention, when exact cache bytes and outputs remain unchanged.
- Specialize the T=3 D2 local path from the same primitive so tentative rows and ring wrap do not re-read or
  materialize unnecessary data.

Promotion requires an end-to-end ordinary win at both 512 and 16K and no local-ring regression at positions around
1,023/1,024/1,025 and repeated wrap. A local-only microbenchmark win is insufficient.

### 3. Close the T=1 FP8 and NVFP4 projection gap

Use the phase-0 profile to rank, separately, grouped Q/K/V, O, Gate, Up, Down, and the tied output head. Compare
equivalent shapes and bytes with llama.cpp rather than comparing aggregate graph labels.

- Retune native SM120 T=1 CTA/warp geometry, weight-fragment reuse, and output-row scheduling for the real Gemma
  dimensions.
- Preserve the direct checkpoint FP8 attention format and exact NVFP4 values/scales.
- Reduce redundant activation reads by sharing the already quantized activation between projections when the
  existing graph does not already do so.
- Fuse the global raw K=V projection handoff into the distinct K and V post-processing kernels to remove the
  device-to-device raw copy, while retaining the model-required distinct normalized K/V cache values.
- Consider pointwise epilogues only when they reduce measured memory traffic without increasing the hot MMA
  critical path or changing mandatory BF16 boundaries.
- Keep separate ordinary and fixed-T3 schedules; one geometry is not required to win both.

This phase explicitly does not authorize persistent duplicate weights or an on-disk repack. An exact load-time
layout transform is allowed only under the existing direct-loading and memory rules and only when the final GPU
allocation remains the sole copy.

### 4. Remove only profile-proven layer and graph overhead

The production ordinary path already uses one complete CUDA Graph per forward, so launch optimization is not the
default explanation for a 4.8--6.6 ms/token gap.

- Remove remaining standalone round, quantize, residual, or normalization nodes only when their bytes and latency
  are material in the current trace.
- Preserve the already-qualified controlled Q/K/RoPE fusion and full-graph replay.
- Keep token selection, control updates, and cache commits device-resident.
- For MTP, compare the GPU-chained D2 graph with the exact same kernels outside capture to distinguish GPU work from
  host wait time; synchronization time is not assumed recoverable.

The phase ends when residual unassigned/pointwise/graph overhead is below 5% of token or group GPU time, or when
adjacent end-to-end runs show that further fusion is neutral. Do not keep code solely because it reduces node count.

### 5. Optimize MTP after the Target model is faster

MTP inherits the winning ordinary attention and projection primitives first. Then optimize the additional D2 work:

- profile the four-layer BF16 assistant separately from the 48-layer three-row Target verifier;
- share activation and K/V staging across the three verifier rows where exact arithmetic permits it;
- retune the existing T=3 FP8 Q/K/V/O and NVFP4 Gate/Up/Down schedules against their actual byte and occupancy
  limits;
- retain device-side propose, verify, accept, commit, stop, and tail routing;
- compare fixed D1, D2, D4, and adaptive policy only after per-group latency improves;
- tune policy for effective verified output tokens/s, never proposed tokens/s;
- preserve ordinary Target output exactly for greedy and same-seed sampled generation.

At the current 502 Target batches, the first concrete D2 milestone is at most 34.85 ms/group on the fixed workload.
The final milestone is the hard 64.82 token/s 3/10 gate, followed by adjacent parity and stretch comparisons.

### 6. Resume prefill work only after decode promotion

Once ordinary context-matrix parity and the MTP hard gate both pass, re-profile prefill. Decode changes must retain
the current prompt-throughput advantage within normal run variance. Any shared-kernel change that helps decode but
regresses prefill requires shape-specific dispatch, not a global replacement.

## Experiments that are not authorized without a new hypothesis

The following have already been measured and rejected. Repeating them unchanged wastes the benchmark budget:

- BF16 or TF32 Tensor-Core T=1 decode attention: up to 13.65% slower than retained scalar split-GQA;
- the prior ordinary fused Gate/Up path: slower than separate Gate/Up/GELU end to end;
- mandatory BF16 rounding moved into projection/GELU stores: 0.61% slower despite fewer graph bytes;
- a combined residual plus following MLP-quantization boundary: 1.7% slower;
- an exact verifier-suffix CUDA Graph: more memory with no speedup;
- a persistent duplicate decode weight layout;
- accepting numerically different MTP output to gain speed.

A retry must state what materially changed: data format, GQA mapping, tile geometry, Blackwell instruction family,
memory traffic, context-tier dispatch, or a new profile showing a different bottleneck.

## Correctness and quality gates

Every candidate must pass the smallest relevant operator tests before whole-model timing, then all applicable final
gates before promotion:

- FP8 attention against CPU/BF16 references for local/global shapes, partial chunks, 16K/64K positions, masks,
  local-ring wrap, and distinct K/V post-processing;
- exact NVFP4 activation bytes and required BF16 projection boundaries;
- complete Layer-0 and representative global-layer hidden/cache comparisons;
- first-token and 32-step logit metrics, top-k overlap, cosine, KL, maximum absolute and RMS error;
- deterministic greedy generation and the existing 12-prompt teacher-forced suite;
- gem16 MTP output IDs exactly equal gem16 ordinary Target IDs on the fixed 1,135-token workload;
- same-seed sampled ordinary/MTP identity and correct transactional cache/RNG/repetition state;
- Compute Sanitizer memcheck/racecheck on changed attention, cache, or device-control code;
- no silent precision fallback and no allocation, file access, or dynamic compilation in the token loop.

Cross-engine token identity is not required because tensor formats and arithmetic differ. Cross-engine speed claims
remain blocked until the paired quality suite shows the candidate has not gained speed by losing quality.

## Benchmark protocol

Screening may use one warm-up and one measured run to reject obvious losers. Nothing is promoted or described as a
final competitor result from a screening run.

Final qualification uses:

- Linux on the locked reference machine and toolchain; Windows/Linux ratios are never attributed to a kernel
  change;
- identical prompt token IDs and disclosed chat/generation settings;
- Q8_0 KV for llama.cpp and checkpoint FP8 KV for gem16, with the difference in every table heading;
- 3 warm-up and 10 measured runs, raw JSONL retained, median primary, mean/standard deviation/95% confidence
  interval also reported;
- 256 generated tokens for the ordinary matrix and the complete fixed 1,135-token output for MTP;
- alternating engine order, no prompt-cache reuse, full GPU residency, stable power mode, and continuous clocks,
  power, temperature, peak VRAM, and steady-state VRAM;
- core GPU and end-to-end boundaries reported separately;
- output hashes, accepted/rejected/proposed counts, Target batches, mean accepted length, TTFT, effective decode
  throughput, and median/p95/p99 latency retained per run.

Run the direct-checkpoint vLLM row beside the two competitive rows wherever it fits under the disclosed memory
policy. Report it as a reference ceiling with its own timing and output limitations, not as a substitute for
same-source llama.cpp parity.

The qualified ordinary matrix is 128, 512, 2,048, 8,192, 16,384, 32,768, and 65,536 existing tokens. Add 131,072
only after both engines fit under the same disclosed residency rules.

## Promotion, rollback, and documentation

For each candidate:

1. record the hypothesis and predicted removable milliseconds or bytes;
2. run operator correctness and a short end-to-end screen;
3. delete an obvious loser completely;
4. for a plausible winner, run adjacent parent/candidate 3/10 plus Nsight and resource checks;
5. promote only a statistically credible end-to-end win with unchanged correctness and quality;
6. make the winner the sole production path, retaining only a necessary correctness oracle;
7. add the measurement, checksum, kernel resources, VRAM, and decision to `docs/PERFORMANCE_LEDGER.md` and
   `docs/DECISIONS.md`.

Rollback is mandatory for unexplained quality drift, a hidden fallback, local-memory spills in a hot kernel without
evidence, token-loop allocation, peak VRAM above 15.3 GB, a p95 latency regression that throughput conceals, or a
context-tier regression outside the stated allowance.

## Completion criteria

This plan is complete only when all of the following are true:

- ordinary gem16 decode meets or beats the adjacent qualified llama.cpp candidate through 64K;
- gem16's long-context latency slope is no worse than the competitor's;
- fixed 16K D2 reaches at least 64.82 effective verified tokens/s as a 3/10 median;
- fixed 16K D2 also meets or beats the fresh adjacent llama.cpp result;
- all 1,135 MTP output IDs equal gem16 ordinary Target output in every run;
- quality, memory, telemetry, native-instruction, and no-allocation gates pass;
- prefill retains its current advantage within normal variance;
- raw artifacts and the final performance ledger make every format and timing difference explicit.
