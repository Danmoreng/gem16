# Prefill optimization plan

Status: active execution goal

Target machine: Linux, NVIDIA GeForce RTX 5080 Laptop GPU, compute capability 12.0

Target workload: direct `unsloth/gemma-4-12b-it-NVFP4` loading, text-only, batch one

## Objective

Bring correct end-to-end prompt processing to the performance of the retained direct-load vLLM reference, or
exceed it, without weakening checkpoint semantics, output quality, benchmark boundaries, the 16 GB memory budget,
or the no-allocation token-loop contract. Performance changes are developed directly on `main`. A candidate is
promoted only when it wins the prescribed repeated benchmark and passes all applicable correctness gates. The
winner becomes the sole production implementation; rejected and superseded implementations do not remain as
user-selectable optimization modes.

The initial Linux reference point is commit `1bc942b`; online attention, the 1,024-token plan, and the first
large-M NVFP4 winner advance the same 512-token characterization as follows:

| Workload | Initial gem16gb | Current gem16gb | vLLM | Current/vLLM |
|---|---:|---:|---:|---:|
| Prefill, 512 prompt tokens, batch 1 | 698.25 tok/s | 1,095.56 tok/s | 6,146.50 tok/s | 0.178x |

These numbers are diagnostic rather than a parity claim because the retained vLLM run and gem16gb do not yet have
identical timing boundaries and cache precision. The optimization goal does not depend on presenting the ratio as
a headline result; accepted comparisons must satisfy `docs/BENCHMARKING.md` and `AGENTS.md`.

## Profile-derived diagnosis

A direct Linux Nsight Systems characterization of both engines at 512 tokens gives the following approximate GPU
cost per prefill execution. The current column is the online-attention path; the initial column is the profile that
established this program:

| Phase | Initial gem16gb | Current gem16gb | vLLM | Current gap |
|---|---:|---:|---:|---:|
| NVFP4 MLP projections | 289.78 ms | 233.33 ms | 24.23 ms | 9.63x |
| Attention | 199.77 ms | 25.04 ms | 13.11 ms | 1.91x |
| FP8 attention projections | 131.13 ms | 127.23 ms | 27.15 ms | 4.69x |
| Other GPU work | 115.98 ms | 107.93 ms | 9.98 ms | 10.81x |
| Total GPU time | 736.66 ms | 493.53 ms | 74.47 ms | 6.63x |

The initial gem16gb path launched approximately 9,235 GPU operations per execution, versus 747 for vLLM. Online
attention and the 1,024-token plan reduce this to approximately 2,311. Attention is no longer the dominant gap;
NVFP4 projections now consume about 53% of current GPU time and remain almost 10x slower than vLLM. Large
projection tiles and substantially more weight reuse are therefore the next critical work.

The neighboring Apache-2.0 NInfer implementation supplies useful implementation concepts, not a compatible
runtime path: shape-specific plans, BF16 Tensor-Core QK/PV, FP32 online softmax, swizzled shared-memory staging,
and pipelined K/V tiles. gem16gb must adapt those concepts to direct mixed FP8/NVFP4 checkpoint storage, the hybrid
local/global Gemma attention geometry, circular local cache addressing, and its existing correctness contract.

## Ordered implementation program

### 1. Online Tensor-Core prefill attention

Status: implemented and model-qualified at `c0f42de`; final artifact bookkeeping remains part of the milestone.

Implement Gemma-specific attention without a global score matrix:

- use Tensor Cores for both QK and probability-times-V;
- retain FP32 row maxima, normalization sums, and output accumulation;
- apply exact causal and 1,024-token local-window masks from absolute positions;
- read keys and values belonging to the current chunk directly and older local positions from the circular cache,
  preventing overwrite hazards;
- support the 40 local layers (`16` query heads, `8` KV heads, dimension `256`) and eight global layers (`16`
  query heads, one KV head, dimension `512`) with shape-specific plans;
- eliminate the prompt-length-times-context score arena from the production plan;
- preserve the unfused scalar implementation as a test oracle, not a selectable production mode.

Qualification begins with the local D256 path, because it represents five sixths of the layers. It may be committed
as an internal verified milestone only when production behavior is not regressed. The phase is complete only when
both attention geometries are qualified and the online implementation is the sole production path.

### 2. Promote the largest deterministic winning prompt chunk

Status: 1,024 selected as the sole checkpoint-FP8 standard. At 2,048 prompt tokens it reaches 915.24 tok/s versus
893.60 tok/s for 512-token chunks (+2.42%), with non-overlapping 95% confidence intervals. At 512 tokens the
difference is statistically indistinguishable; 1,024 uses about 217 MB more reusable workspace and remains within
the 16 GB budget.

After the score arena is removed, measure complete-prompt chunks of `512` and `1,024` tokens against the current
context-budgeted plan. Select the fastest size that is deterministic, fits the measured arena budget across context
profiles, and passes generation/logit gates. Encode that selection as the standard plan; do not add a public chunk
or legacy-path switch. Long-context tiers may select a smaller compile-time plan only when their documented memory
geometry requires it.

### 3. Rebuild NVFP4 prefill projections around large SM120 CTA tiles

Status: in progress. The first promoted reuse step retains each source weight/scale fragment across eight
independent `m16n8k64` token tiles (M128), reducing NVFP4 profile time by 18.7% and context-512/2,048 end-to-end
time by about 11%. It uses 128 registers with zero stack/local memory. A tested M256 extension is rejected because
it reaches 255 registers and a 248-byte stack frame. Shared-memory staging, asynchronous pipelining, larger-N CTA
cooperation, and the measured swizzle decision below remain open.

Replace the current warp-level token tiling with a shape-specific block pipeline that:

- reuses packed E2M1 weights and E4M3 block scales across a substantially larger token tile;
- overlaps global-memory loads, source-layout preparation, and `m16n8k64` MMA work;
- consumes the source checkpoint values exactly, including local and global scaling;
- uses an exact load-time scale/layout swizzle into the final GPU allocation only if profiling proves direct source
  consumption is slower;
- never creates a persistent second weight copy and records load-time and peak-memory impact of any swizzle;
- has no local-memory spills in the selected hot kernel.

Gate, Up, and Down are measured individually and end to end. A layout transformation is accepted only when its
end-to-end benefit, exact value preservation, and memory cost are recorded in `docs/WEIGHT_LAYOUT.md`.

#### Phase-3 handoff after the M128 promotion

The current production geometry is one warp per N8 output slice and eight M16 token tiles per retained packed
weight fragment. Four warps therefore cover M128xN32 while preserving the original K64 accumulation order. This
is the stable resume point at commit `2366c03`; increasing the per-warp accumulator footprint to M256 is not a
viable continuation because it spills.

Inspection of NInfer's Apache-2.0 Q4 row-split GEMM identifies the next transferable mechanisms: shape-specific
M64xN64/M64xN128 CTA schedules, K64 iteration, two- or three-stage `cp.async` pipelines, XOR-swizzled shared
memory, CTA-wide reuse, and optional accumulator-fragment ping-pong. Its Q4 values are decoded to BF16 shared
memory before BF16 MMA, so that numerical/storage path is not reusable for this engine's native block-scaled
E2M1 MMA and must not be copied.

The next experiment is an exact direct-layout M128xN32xK64 CTA pipeline. It should stage and double-buffer packed
weight bytes and their E4M3 scale words while the current K64 fragment executes, share activation data across the
four N8 warps only when that reduces measured traffic, and load each repeated weight-scale word once per lane
quad before broadcasting it. Start without a load-time swizzle so the existing direct-checkpoint layout remains
the control. Preserve FP32 accumulator order, confirm zero stack/local memory, and qualify Gate, Up, and Down
separately before running the required 128/512/2,048 end-to-end matrix. Only if this CTA pipeline remains limited
by source-layout loads should an exact streamed load-time swizzle be measured.

### 4. Rebuild and group the FP8 attention projections

Apply the same large-token-tile and pipeline discipline to Q, K/V, and O while preserving per-token dynamic FP8
activation quantization and per-channel weight scaling. Evaluate shape-specific combined Q/K/V scheduling and the
full-attention K-projection reuse already required by model semantics. Promote grouping only when it reduces
end-to-end time; do not retain separate grouped/ungrouped user modes.

### 5. Fuse only profile-proven bandwidth and launch boundaries

After attention and projection kernels are no longer the old bottlenecks, profile again and consider, in order:

1. RMSNorm plus activation quantization;
2. Q/K normalization plus RoPE plus K/V write;
3. Gate/Up epilogue plus GELU-tanh product;
4. residual/norm boundaries whose intermediate values have no other consumer.

Each fusion must retain a test oracle, report numerical reordering, and improve repeated end-to-end prefill. An
isolated kernel win is insufficient, as shown by the previously rejected Gate/Up fusion.

## Mandatory correctness gates

Every promoted milestone must pass:

- host unit tests and CUDA operator tests;
- exact checkpoint-format, scale, packing, cache-addressing, causal-mask, and local-window fixtures relevant to the
  changed code;
- comparison with the retained unfused/reference operator, reporting maximum absolute error, RMS error, cosine
  similarity, and row-sum/finite checks where applicable;
- exact-blue generation and vLLM Top-1/full-logit comparison at the fixed 129- and 257-token prefill boundaries;
- the committed teacher-forced suite and full-logit comparison when the changed arithmetic can affect logits;
- checks at chunk boundaries, a wrapped local cache, and at least one global-attention layer;
- unchanged tokenizer, chat template, sampling configuration, checkpoint revision, and prompt token IDs.

Tensor-Core online softmax deliberately changes floating-point reduction order, so bit identity with the serial
attention oracle is not a valid universal requirement. Any tolerance used for promotion must be derived from the
observed distribution, recorded in `tests/tolerances.yaml`, and supported by model-logit and generation evidence.
No tolerance may be relaxed only to accept a speedup.

The former eight-token boundary sequences were generated by the old gem16gb path itself and are not a trustworthy
reference: direct vLLM generation differs from them. The committed boundary fixture is produced offline by vLLM
0.25.1 from the pinned checkpoint and checks the only position computed by batch prefill. Later positions belong
to the separate teacher-forced decode suite.

## Mandatory performance and resource gates

For every production promotion, collect on a thermally stable machine:

- prefill at `128`, `512`, and `2,048` prompt tokens, batch one;
- three warm-up and ten measured runs, retaining raw samples and reporting median, mean, standard deviation, and a
  95% confidence interval;
- prompt tokens/s and TTFT with the same timing boundaries before and after;
- Nsight Systems path confirmation, phase/kernel time, and launch count;
- register count, stack frame, spill stores/loads, and local-memory use for changed hot kernels;
- peak and steady-state VRAM plus named arena changes;
- confirmation of no recurring allocation or filesystem access in the execution loop;
- deterministic output checksums across repeated runs.

The main comparison is against a separately built parent commit, executed immediately adjacent to the candidate
under the same clocks and thermal conditions. The current direct vLLM profile is retained as an optimization
target. A parity or superiority claim requires a freshly controlled cross-engine run with identical prompt IDs,
cache precision where supported, warm-up policy, and explicitly reconciled timing boundaries.

## Promotion, commits, and rollback

Work proceeds directly on `main`. Stable intermediate commits are expected after a self-contained milestone has
passed its applicable correctness gates and either:

- improves the complete prefill benchmark with statistical support; or
- adds a necessary internal implementation/test foundation without changing or slowing the production path.

Before each performance promotion, retain benchmark and profile evidence under
`benchmarks/results/<date>/<git-sha>/<machine-id>/`, update `docs/PERFORMANCE_LEDGER.md`, build and run the full
available test suite, then commit and push to `origin/main`. A losing candidate is removed rather than hidden behind
an option. If a newly promoted path later fails a broader correctness or performance gate, fix it immediately or
revert the complete promotion with an explicit ledger entry.

## Completion criteria

This goal is complete only when all of the following hold:

1. Online Tensor-Core attention is standard for local and global prefill and no global score matrix is allocated.
2. The best qualified prompt chunk is the deterministic standard.
3. NVFP4 and FP8 prefill projections use the qualified large-tile plans.
4. Remaining fusions are exhausted based on current profiles, not assumed benefit.
5. The mandatory correctness, quality, memory, Nsight, and 3/10 benchmark evidence is retained.
6. Controlled batch-one prefill reaches or exceeds the current direct-load vLLM reference across the required
   `128`, `512`, and `2,048` points, or a documented hardware/resource lower bound demonstrates the remaining gap
   and the project records an explicit decision before changing the objective.
