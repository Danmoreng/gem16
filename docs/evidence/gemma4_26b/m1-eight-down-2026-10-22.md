# Eight-warp Trellis35 M1 Down — 2026-10-22

Accepted bounded Linux candidate: native/vector-store **M1 Down only** now uses
eight warps with two N8 fragments per warp. GateUp retains sixteen warps; T3 and
diagnostic nonnative/nonvector variants retain their existing dispatches.
This adds a reproducible **2.01–2.14% ordinary end-to-end gain** on top of the
allowed-top1 selection change. No precision, format, context, cache or output change.

Record: `artifacts/trellis35/m1-eight-down-2026-10-22.json`.
Raw root: `benchmarks/results/2026-10-22/c0a807af/m1-eight-down/`.
Real-input diagnostics: sibling `operator-replay/`.
The exact parent/candidate executables, all samples, telemetry and command records
remain local. The record includes canonical output-vector hashes per run, one full
common output vector, actual module SASS/resources, dispatch facts and source identities.

## Why Down only

The standalone candidate shares each payload and activation load between two N8
accumulators. Each accumulator preserves K iteration order, native E4M3 conversion,
FP32 accumulation, scaling, inverse H128 and aligned vector stores.

All 24 real-input projection cases (layers 0/5/15/29, three full-context positions,
GateUp/Down) are bit exact, both directly and under graph replay. Parent and candidate
timings use CUDA graphs with sixteen operator calls per sample, three warmups and
ten samples in alternating order. This avoids attributing CPU launch gaps to GPU work.
These are warm-resident operator diagnostics, not whole-model traffic measurements.

Down is faster in all twelve cases: 11.94–18.36%, median 15.23%. GateUp spans
-2.25% to +11.92%, so it is excluded from this production change. The candidate is
not automatically transferred to T3.

The actual production executable contains native `F2FP...E4M3.F16` conversion and
`QMMA.16832.F32.E4M3.E4M3` instructions for the new kernel. Resources and SASS are
retained in the record; no local-memory stack/spill storage is reported statically.
Launch evidence proves 256-thread Down CTAs while the parent GateUp stays unchanged.

## End-to-end qualification

Same RTX 5080 Laptop, driver 615.71.09 / CUDA 13.4.92, immutable model/assistant,
16,384 prompt IDs, 1,135 greedy outputs / 1,134 decode intervals, context 17,519,
native FP8 KV, batch one and suppression `[1,50,106,258882,258883]`.
Three warmups, ten retained runs per cohort; AB/BA ordinary plus an adjacent full D2
candidate cohort. Max-power/Dynamic Boost remain active; clocks are firmware-managed.
Each cohort starts at <=50 C. Lifecycle and timing boundaries are unchanged.

| Cohort | Median tok/s | Mean 95% CI tok/s | Sampled peak VRAM MiB |
|---|---:|---:|---:|
| Parent A | 134.976 | 134.869–135.122 | 12455 |
| Candidate B | 137.695 | 137.685–137.947 | 12455 |
| Candidate B2 | 137.642 | 137.586–137.830 | 12455 |
| Parent A2 | 134.761 | 134.690–134.909 | 12455 |
| Candidate D2 | 172.407 | 172.324–172.514 | 12725 |

Ordinary gains are 2.0142% and 2.1376%. All fifty retained runs have the same
1,135 output IDs, no fallbacks and no reported token-loop allocations. D2 is +0.044%
against the earlier same-session accepted parent cohort (172.331 tok/s), an adjacent
non-regression check rather than a fresh paired D2 speed claim.

High-utilization SM clock medians in ordinary AB/BA order are 2632/2610/2610/2625 MHz.
Telemetry spans startup, prefill and warmup; it is not isolated decode clock attribution.
The first and second ordinary confirmations are both retained. Current approximately
137.6–137.7 tok/s is still below the pinned llama audit's 138.31 tok/s reference;
no new cross-engine quality or performance equivalence is asserted.

## Correctness and memory boundary

- Six CUDA/host CTests pass, including complete CUDA, Trellis, MoE, attention,
  NVFP4 consumption and host unit checks. Existing tolerances are unchanged.
- The existing synthetic M1 test compares old nonvector and new vector Down
  outputs and final slot-ordered reductions with zero tolerance.
- Targeted Trellis Compute Sanitizer memcheck reports zero errors.
- Native ordinary/D2 8/64 outputs match the frozen fixtures exactly.
- Protected 12B and internal NVFP4 ordinary/D2 64-token vectors remain identical,
  with no fallbacks/loop allocations. Explicit-fixture 12B product CTest passes.
- Eight-output node traces show 210 new Down calls in ordinary and 60 in the
  D2 ordinary tail, proving the captured tail uses the qualified new dispatch.
  T3 remains its prior specialization.

The earlier full-model D2 sanitizer result is preserved separately; this leaf-kernel
change uses targeted operator memcheck, numerical parity and full adjacent D2 evidence.
No new Windows, media, broad quality or everyday-context release claim is made.
