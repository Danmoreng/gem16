# Start here — current coding-agent task

Status: M00–M13 accepted; M13 decided `proceed`, so M14–M17 are unblocked.
Plan revision: Fast Track R4.

## Read now

1. repository `AGENTS.md`;
2. [`../../ACTIVE_DECISIONS.md`](../../ACTIVE_DECISIONS.md);
3. [`ACTIVE_CONTRACT.md`](ACTIVE_CONTRACT.md);
4. [`FAST_TRACK_STATUS.json`](FAST_TRACK_STATUS.json) and the current milestone;
5. only the specs linked by that milestone.

Do not preload the full decision, correctness, benchmark or performance ledgers. Read historical records only for a concrete question or evidence check.

## Lead-agent orchestration

M08 was accepted on 2026-08-14. The lead agent may assign disjoint sub-agents for:

- M10 phase A BF16 MoE oracle;
- M12 phase A traits and attention/KV fixtures;
- M10 BF16 and quantized CPU MoE semantic oracle;
- M25 phase A assistant compatibility and memory modeling;
- evaluation/benchmark harness scaffolding.

Use [`PARALLEL_WORKSTREAMS.md`](PARALLEL_WORKSTREAMS.md). M13 is accepted; M14–M16 may proceed alongside rolling M17 integration.

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
reference-only; M14–M16 own native operator promotion and M17 owns rolling integration.
