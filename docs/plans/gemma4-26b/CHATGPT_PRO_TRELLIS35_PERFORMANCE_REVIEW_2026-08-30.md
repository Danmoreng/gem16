# ChatGPT Pro review request — Trellis35/W4A8 performance

## Requested outcome

Perform a source-based performance design review of the experimental Gem16
Trellis35/W4A8 path and propose the next implementation work packets in strict
priority order.  The answer should be concrete enough for Codex to implement:
name the source locations and kernel boundaries, describe the data movement and
thread/block organization, identify correctness risks, and define the smallest
benchmark and numerical gate for accepting or rejecting each proposal.

Do not run or recommend the remaining long quality suite yet.  The owner wants
performance made materially better before WP8B long-context and task quality
evaluation consumes more GPU time.

## Current state

- Branch: `codex/gemma4-26b-trellis35-w4a8`.
- Persistent routed-expert artifact: exact 3.5 payload bpw, 11.3666 GiB Target
  weight arena, 2.3208 GiB below the NVFP4 parent.
- Gate+Up is fused `2816 x 1408` with no padding. Down remains physically
  padded from 704 to 768 for the 128-wide transform/tiling contract.
- Ordinary M=1, one true batched Fixed-D2 T=3 family, and grouped prefill all
  decode the same persistent mixed-K3/K4 Trellis payload to E4M3 and compute
  W4A8. There is no persistent NVFP4 expert duplicate, CPU offload, expert
  streaming, fallback, or recurring token-loop allocation.
- WP8A numerical/sampled/three-placement 16K retrieval sanity passed. The full
  WP8B suite is frozen and deferred; this branch is not quality-qualified.

WP9 replaced repeated branch-history decoding with a two-U32 tail-biting state
window, paired state decode, warp-shared payload loads, and four-deep software
prefetch. Exact operator parity and unchanged bounded generation output were
retained.

Measured on an RTX 5080 Laptop GPU, CUDA 13.3, driver 610.43.03:

| Path | Before WP9 | Current | Existing NVFP4 reference |
|---|---:|---:|---:|
| 16K prefill | ~211.7 tok/s | ~951.1 tok/s | ~6966 tok/s |
| Ordinary decode | 38–40 tok/s | 119.4 tok/s | ~148 tok/s |
| Fixed-D2 end-to-end | 32.6 tok/s | 85.2 tok/s | ~203.8 tok/s |
| Fixed-D2 batched T3 verifier | 33.5 tok/s | 168.5 tok/s | inspect reference evidence |

The current bounded Fixed-D2 run used `drafts=2`, fixed chain, and the exact
shared-batched-MoE verifier. It passed ordinary Target identity, accepted 38 of
50 proposed drafts (76%), and had no non-finite steps. It is a single diagnostic
run, not a publication median.

## Current profile diagnosis

The final 512-token Nsight Systems capture attributes GPU time approximately as
follows:

- grouped Trellis prefill projection: 63.3%;
- activation transform scale and quantize: 16.5%;
- inverse/output transform: 6.9%;
- BF16-side transform scale and quantize: 4.5%.

The M=1 expert projection microkernel improved from 734.0 us to 73.15 us. Its
reported L1 traffic fell from about 1.88 GB to 35.95 MB and it uses 48
registers/thread. Increasing prefill rows per warp from four to eight regressed
prompt time by 23.1%, so that candidate was reverted.

The main implementation is in `src/cuda/trellis35/reference.cu`. Format and
runtime contracts are in
`docs/plans/gemma4-26b/GEM16_TRELLIS35_W4A8_DISCOVERY_PLAN_v2.md`, compact WP9
facts are in `artifacts/trellis35/wp9-runtime-decoder-optimization.json`, and
upstream-derived code provenance is in
`third_party/exllamav3_quant/PROVENANCE.md`.

## Questions to answer

1. What is the highest-probability route from ~951 toward several thousand
   prefill tok/s while keeping the same single persistent Trellis35 payload?
2. Should the next large-M backend reconstruct inline into the existing SM120
   block-scaled W4A4 MMA path, remain W4A8 with a different tile/persistent
   schedule, or stage a bounded transient tile? Compare traffic, transform,
   occupancy, and workspace consequences using the actual shapes.
3. How can Trellis reconstruction be amortized across more routed rows without
   repeating the failed eight-row register-heavy design? Consider CTA/warp
   specialization, shared-memory staging, TMA, persistent expert scheduling,
   and separation of decode from MMA, but tie every proposal to concrete code.
4. Which activation and inverse-transform kernels should be fused or reordered,
   and what numerical boundary must remain physical BF16 versus register-only?
5. Ordinary decode is within about 20% of its parent reference, but Fixed-D2
   end-to-end remains much slower even though its batched T3 verifier reaches
   168.5 tok/s. Identify the surrounding pipeline costs and propose isolated
   experiments that preserve exact Target identity and acceptance semantics.
6. Identify any correctness, undefined-behavior, alignment, warp-uniformity,
   register-spill, or tail-biting extraction risks in the WP9 implementation.

## Required response format

Provide:

1. a short verdict on the present architecture;
2. a ranked bottleneck table grounded in included evidence;
3. three to six narrowly scoped candidate work packets, each with expected
   payoff, exact source touchpoints, implementation sketch, rejection gate, and
   required Nsight/SASS/numerical evidence;
4. the recommended first candidate and why it has the best risk-adjusted
   payoff;
5. any prerequisite instrumentation missing from this archive;
6. an explicit list of tempting ideas that should not be pursued yet.

Separate demonstrated facts from estimates. Do not invent speedups, relax
precision, change prompts/timing boundaries, introduce a second persistent
expert representation, or treat this experimental path as product-qualified.
