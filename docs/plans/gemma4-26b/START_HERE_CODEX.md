# Start here — current coding-agent task

Status: M00–M17, M20–M23 and M25 accepted. The qualified text-only 26B fixed-D2 checkpoint is selectable alongside 12B, supports 73,728 MTP context tokens with a 200 MiB reserve, and keeps a 98,304-token Target-only maximum. The broad historical M19 suite is waived for this checkpoint; claims remain bounded to the recorded GSM8K/AIME and product evidence. M18 remains conditional.
Plan revision: Fast Track R4.

## Read now

1. repository `AGENTS.md`;
2. [`../../ACTIVE_DECISIONS.md`](../../ACTIVE_DECISIONS.md);
3. [`ACTIVE_CONTRACT.md`](ACTIVE_CONTRACT.md);
4. [`FAST_TRACK_STATUS.json`](FAST_TRACK_STATUS.json) and the current milestone;
5. only the specs linked by that milestone.

Do not preload the full decision, correctness, benchmark or performance ledgers. Read historical records only for a concrete question or evidence check.

## Lead-agent orchestration

The M17 profile and M20–M23 slices are accepted. M23 freezes the exact native ordinary-decode Target and rollback;
do not rewrite its M20/M21 evidence when adding MTP.
Use [`MAX_PERFORMANCE_EXECUTION_PLAN.md`](MAX_PERFORMANCE_EXECUTION_PLAN.md) as the consolidated execution order for
the current optimization campaign. It reconciles the two 2026-08-25 owner-supplied German plans with active policy,
current source and accepted evidence; higher-precedence rules in `AGENTS.md` and `ACTIVE_DECISIONS.md` still win.
The lead agent may assign disjoint sub-agents for:

- bounded M19/Q4 numerical reconciliation only; no multi-hour task/prose suite in the current wave;
- profile-driven prefill and ordinary-decode optimization toward the fixed 6,000/150 token/s M20 gates and 6,500
  prompt-token/s stretch target;
- M20 controlled benchmark execution and telemetry;
- M21 real 32K/64K long-context qualification;
- M22 CLI/server product integration (accepted; evidence reconciliation only);
- M25 phase A assistant compatibility and memory modeling;
- independent evidence reconciliation.

The 2026-08-26 owner decision froze the performance checkpoint before product work. Commit `c4ead1d` subsequently
connected the M22 server runtime and M25 engine to the selectable Studio model catalog and completed bounded real
sampled-D2 chat and continuation. Preserve the 12B default and its audio/vision behavior; 26B remains text-only,
single-slot and fixed-depth, but is now a qualified selectable product checkpoint. Immutable Hugging Face publication
and Studio download integration are complete; retained sampled timing is optional additional
performance evidence, not an acceptance gate.

Use [`PARALLEL_WORKSTREAMS.md`](PARALLEL_WORKSTREAMS.md). M18 runs only after a recorded trigger. The owner-approved
Fast Track and Main promotion are complete; remaining work is optional bounded follow-up. The current external reference is llama.cpp build
10623 at 118.627 ordinary / 151.919 D2 decode token/s; vLLM 0.27.1 remains the ordinary-only comparison because a
fully GPU-resident 26B MTP configuration does not fit on the 16 GB reference device.

## M08 success

M08 is accepted at implementation commit `f433358b8e2c1250b95801fc898faee4fcedcbe5`. Two clean complete hybrid
builds are byte-identical, the external lock and direct C++ loader validate, the 12B inspect regression passes and
the exact single-arena reference-GPU admission succeeds. See `artifacts/m08/acceptance.json`.

## Full-run rule

For expensive conversions or publication claims, use reviewed, targeted-tested code, a clean worktree and source/output preflight. Small fixtures and bounded diagnostic probes do not need the full release workflow, but their diagnostic status must be recorded.

## M09 success and current M10 boundary

Reconcile the real artifact with named CUDA regions and prove one fully resident 32K slot with at least 700 MiB
directly measured free-device margin after initialization. M08's admission probe is exact-arena synthetic evidence,
not model execution. The 2026-08-14 diagnostic uploaded all 1,285 tensors into one 14,696,668,160-byte device arena,
left 818,741,248 bytes free at 32K, admitted 64K with 483,196,928 bytes free against the owner-approved 400 MiB base
gate and rejected a second slot without a CUDA allocation delta. This is residency evidence, not model execution; graph
capture and warm execution begin in M11/M12. See `artifacts/m09/diagnostic-summary.json`. Q4_0 backend work and MTP
verifier-head optimization remain outside M09.

M09 is accepted at implementation commit `6c3b9e456bc7fed68e2e90a51ba20c1c895fd085`; the clean record is
`artifacts/m09/acceptance.json`. M10 now owns the independent CPU MoE oracle for router, deterministic top-8, shared
MLP, selected expert contributions, reduction, norms and residual order. It must not call production CUDA code.

The M10 implementation now independently replays the locked BF16 shared branch, reconstructed FP32 router, 24 norm
boundaries and all 64 selected expert contributions across layers 0/5/6/29 and positions 0/17. It also decodes bounded real NVFP4 rows from
the accepted M08 artifact without importing production CUDA. M10 is accepted at implementation commit
`eac6b443b239d5e04c5be5daef3dd659d57d5de9`; the clean record is `artifacts/m10/acceptance.json`. M11 now owns the
fixed-address CUDA MoE reference path and must match these named boundaries before any performance fusion.

M11 is accepted at implementation commit `91ee47586cc426c051dee247ddfcf4a6b765ecfd`. The CUDA path keeps router
selection on device, runs all eight fixed expert slots without host routing, matches the locked quantized-versus-BF16
boundary gates, repeats bitwise identically without an allocation delta, and passes memcheck/racecheck/initcheck. See
`artifacts/m11/acceptance.json`.

M12 is accepted at implementation commit `bbee9cd930133dd49cb3acc79b4867658a0968cc`. Its immutable 30-layer trait
table drives both execution and exact cache sizing; real local/global layers pass the fixed quantized-versus-BF16
boundary gates, keep K and V physically distinct, repeat bitwise identically without an allocation delta and pass
memcheck/racecheck/initcheck. See `artifacts/m12/acceptance.json`. M13 now owns the first full slow-model integration
and the single early quality go/no-go screen.

M13 is accepted with decision `proceed` at implementation commit
`3cd697501585868d5ef41e60d212bb0e502c365c`. The full 30-layer reference path encodes the locked 18-token prompt,
generates deterministic `OK.`, preserves a resident second-turn prefix, retains the expected first token at rank 1
with BF16-to-M13 KL 0.0000121, passes selected layer/router drift gates and leaves 1,280,507,904 warm bytes free at
32K without a repeat-run allocation delta. See `artifacts/m13/acceptance.json`. The path is experimental and
reference-only; M14–M17 subsequently promoted and integrated the frozen native SM120 profile.

## M14–M17 success and current qualification boundary

M14–M16 are accepted at implementation commit `9a374c3dda10b7ae870c712cd70a60aa0a9e2c52`. M17 is accepted at original
implementation commit `57fdeb309aacfce2e4eba65745fba86f14ebd113` and closure-hardening commit
`348683e167c6c4d3b0be7580e404c097b199a3d8`. Router failures now fail safely, complete engine replay/relaunch is
automated, native slot-batched expert execution remains bitwise exact and hardware capability reporting is
profile-aware. The latest retained development series adds an engine-local BF16 Tensor-Core router, final-chunk-only
prefill output, value-cache ping-pong staging and exact fused expert epilogs to the physical-BF16/T1024, grouped
expert, asynchronous K64, prepared global K/V and exact dual-token parent. The current three-run row reaches
6,568.395 mean prompt tok/s and 150.413 mean ordinary-decode tok/s on 16K+64, passing all development thresholds.
A bounded four-prompt/14-row QAT-BF16 Golden Gate, real-shape CUDA and sanitizers, deterministic engine
relaunch, the real 26B product test and protected 12B regression pass. Tensor-Core is the integrated prefill default;
serial exact remains an explicit rollback. M20 subsequently accepted the exact candidate at retained medians of
6,572.809 prompt and 150.615 ordinary-decode tok/s after three warm-ups and ten retained runs. M21 accepted real
32K/64K/96K execution and measured `base_max_context=98,304`; 100K is reproducibly capacity-rejected. See
`artifacts/m20/router-tensor-core-diagnostic.json` and
`artifacts/m20/optimization-decode-moe-split-fused-quant.json`, with the latest host-tail evidence in
`artifacts/m20/optimization-compact-prediction-self-feed.json` and the latest exact decode-router evidence in
`artifacts/m20/optimization-decode-router-top8.json`; D09d's eliminated native contribution writes are recorded in
`artifacts/m20/optimization-decode-no-contribution-write.json`, and D12's exact native post-norm fusion in
`artifacts/m20/optimization-decode-fused-postnorm.json`; D13's shared-memory native intermediates are in
`artifacts/m20/optimization-decode-shared-postnorm.json`; the target-reaching fused W2 reduction, eight-warp W13
geometry and production diagnostic-write cleanup are in
`artifacts/m20/optimization-decode-fused-expert-epilog.json`.

The exact 16K+64 ordinary path passed the 6,000 prompt, 150 decode and 6,500 prompt-stretch gates without
MTP/speculative decode, cache/precision/semantic substitutions or timing-boundary changes. Compact acceptance is in
`artifacts/m20/acceptance.json`; matching long-context acceptance is in `artifacts/m21/acceptance.json`.

The later owner-paused development checkpoint retains exact staged weight reuse plus full-tile global-attention and
physical-BF16 Shared-Down Prefill boundaries. Its unchanged `c750d0...` output reaches 7,046.454, 7,073.667 and
7,068.125 prompt token/s (7,068.125 median). The exact fixed-D2 production chain now uses a parallel/shared Router,
fused final norm/Head quantization, concurrent local T=3 softmax reductions and compact status-only inter-group
commits. Its two completed final runs reach 204.415 and 204.246 token/s with all 1,135 outputs and 642/980 acceptance
unchanged; the owner stopped the planned third run and paused further optimization. Compact records are
`artifacts/m20/optimization-prefill-full-tile-bf16-down.json` and
`artifacts/m25/optimization-production-chain-cleanup.json`; the prior rollbacks remain
`artifacts/m20/optimization-prefill-token-reuse-rope.json` and
`artifacts/m25/optimization-weight-stationary-d2.json`. These bounded results do not replace formal M20 evidence;
`artifacts/m25/performance-freeze.json` remains the product-integration rollback checkpoint consumed by the later
owner acceptance of M25.

M22 product behavior and protected 12B regressions are accepted at
`f0aa302aa0246d44e1c8477dbbbb67fbbe2d2037`; see `artifacts/m22/acceptance.json`. M23 revalidated the updated
capability/product binaries at implementation revision `c8e09e4e337d58ac0cfe402585ef818135845faa` and froze the
accepted technical Target in `artifacts/m23/acceptance.json`. The owner subsequently accepted full GSM8K and AIME
2026 plus the bounded sampled/product evidence and waived the remaining multi-hour M19 suite for this local
qualified checkpoint. Claims remain limited to that evidence and the explicit SM120 product contract.
