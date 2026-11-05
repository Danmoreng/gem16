# Trellis35 Fixed-D2 RoPE qualification

The Fixed-D2 Target prepares one RoPE table per attention profile per group
and reuses the four existing persistent prefill buffers. The layer epilogues
consume those tables in stream order. The change applies to the three-row
Trellis35 verifier, with identical rotary arithmetic and KV precision.

## Throughput and scope

The original native Parent/Candidate run used the same 16,384 prompt IDs,
1,135 output positions and 1,134 timed decode intervals for both binaries.
Context capacity was 17,519 with batch 1, greedy decoding, seed 0, cold
prompt cache, FP8 KV and GPU-resident weights. Each unprofiled ABBA cohort
had three warmups and ten retained samples. Throughput excludes the first
token, which belongs to prefill.

| Mode | Order | Change | 95% bootstrap interval |
|---|---|---:|---:|
| Fixed-D2 | AB | +1.2251% | [+1.0045%, +1.4817%] |
| Fixed-D2 | BA | +1.6452% | [+1.4669%, +1.8701%] |
| Fixed-D2 | Pooled | +1.4602% | [+1.1968%, +1.7152%] |
| Ordinary | AB | −0.0731% | [−0.1779%, +0.0401%] |
| Ordinary | BA | −0.0518% | [−0.1254%, +0.0254%] |
| Ordinary | Pooled | −0.0483% | [−0.1358%, +0.0253%] |

Pooled Fixed-D2 throughput rose from **197.73 to 200.62 tok/s** on an RTX
5080 Laptop GPU. Both modes passed the separate prefill and time-to-first-token
checks. Sampled peak VRAM matched between binaries at 12,725 MiB for Fixed-D2
and 12,455 MiB for Ordinary. The trace shows 15,060 to 1,004 table launches
over 502 groups, with all 15,060 KV epilogues retained. Other kernel counts
and resource shapes matched. The measured and rebuilt CUDA code match after
normalizing generated symbol identifiers.

The original trial rejected the patch because both Ordinary point estimates
were negative. The [follow-up owner decision](../../ACTIVE_DECISIONS.md)
uses an Ordinary non-inferiority margin of −0.20% at the lower 95% bound in
each order. Both original orders satisfy that margin. The full original
sample set was retained. This is a native comparison against the prior
gem16 code and uses no new llama.cpp timings.

## Qualification

- The 90 layer captures matched bitwise at early, middle and late positions.
  All four table buffers were aligned and disjoint, and all four sentinel
  regions survived the 30 intervening attention and MoE layers in each capture.
- Parent and candidate produced identical short-run tokens at local-ring and
  context-capacity boundaries with the 1,024- and 2,048-token prefill settings.
  Forced acceptance and EOS in each output slot matched the visible KV and
  final hidden bytes. The product session exercised cache reuse, cancellation
  at the fifth and final callback, poisoned-state rejection and reset.
  Initial and continuation tokens matched between builds, including with
  diagnostics disabled.
- Nine host/CUDA CTest targets and the 12B product CTest passed. The 12B Unified
  and internal 26B NVFP4 runs matched across both binaries and both decode
  modes for 64 tokens. A scoped Compact Vision test matched the Ordinary and
  D2 text for one real image at a 70-soft-token budget.
- The isolated RoPE/epilogue replay passed Compute Sanitizer memcheck and
  synccheck. Integrated memcheck reported zero errors over a 16,384-token
  prefill and eight outputs. A separate diagnostic smoke of the rebuilt
  executable reproduced all 1,135 outputs and all 1,004 proposals with
  502 groups, 632 accepted proposals, zero fallbacks and zero token-loop
  allocations. Diagnostic timings were excluded from throughput.

With diagnostics disabled, cancellation at the fifth callback during an
active 32-output request returned the requested cancellation status in four
of six trials on each build. In the other two trials on each build, the chain
reported more outputs than the streaming ring delivered and returned an
internal inconsistency status. In all six trials on each build, the session
rejected further generation until reset and reproduced its initial output
after reset. This pre-existing streaming status issue remains open.

The [machine-readable record](../../../artifacts/trellis35/n1-rope-qualification-2026-11-05.json)
contains the retained samples, model and binary identities, measured patch
hash, sanitized commands and hashes of the raw evidence. Qualification ran
on Linux SM120a. The Vision check covers one scene.
