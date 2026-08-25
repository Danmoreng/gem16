# M25 — 26B MTP integration and final qualification

Status: in progress; technical M23 Target accepted, exact greedy D2 CUDA-Graph development path implemented
Class: required final target

Normative inputs: [MTP platform contract](../../../MTP.md), [Tied embedding/head](../specs/EMBEDDING_HEAD_SPEC.md), [Memory arena](../specs/MEMORY_ARENA_SPEC.md), [Session ownership](../specs/SESSION_OWNERSHIP_AND_CONCURRENCY.md), [API/CLI changes](../specs/API_CLI_CHANGES.md), [Test matrix](../specs/TEST_MATRIX.md).

## Outcome

Add exact Target-verified MTP to the frozen 26B base target and qualify the highest safe MTP context on the 16 GB GPU.

## Phase A — early feasibility, parallel from M06

- identify and immutably lock a target-compatible 26B assistant source;
- validate tokenizer, vocabulary, hidden interfaces, layer/cache dependencies and maximum positions;
- inventory exact tensors and calculate BF16/FP8/NVFP4 resident sizes;
- model 32K/64K base+assistant+verifier memory;
- determine whether the assistant can read Target KV without an independent long-context cache;
- record any asset incompatibility early.

This phase may add docs, locks, inventory tools and fixtures only. It does not edit the base runtime.

## Phase B — assistant artifact and loader

The locked source is `google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant` at revision
`9537141506fe8875b3ed45b264af13580cb29166`. Compile one separately locked hybrid artifact: NVFP4 tied
embedding/output head and MLP matrices, FP8 attention Q/O and pre/post interface projections, and BF16 norms/scalar
controls. Use the official BF16 payload as the numerical oracle. Keep Target and Assistant provenance separate.
The BF16 oracle governs component-level numerical comparisons only; Target verification governs proposal
correctness and acceptance for both BF16 and hybrid Assistant outputs.

## Current development checkpoint (2026-08-25)

The separately resident hybrid Assistant and transactional greedy Target verifier are implemented but M25 is not
accepted yet. Before graph capture, the 32K development profile leaves 843,907,072 bytes free after loading. On the retained real
16K Wikipedia prompt with 64 outputs and D2, the Target accepted 37/50 drafts (74%), and the resulting 64-token
sequence exactly matched an independent ordinary Target engine. A short three-way fixture also demonstrates why
BF16 is not the acceptance authority: BF16 and hybrid both proposed token 8,269 while the Target verified token 607.

The current exact layer-major verifier batches Q/K/V, the frozen Split-K2 attention output projection, shared MLP,
and routed experts. The initial 64-output diagnostic boundary reported 113.49 token/s versus 103.54 token/s for the
same diagnostic ordinary boundary. Its groups produced 63 post-first outputs in 0.37118 seconds (169.73 token/s), but
that short projection benefited from 74% draft acceptance and included no sustained-context characterization.

The subsequent forced-output 16K+1,135 development run separates initial selection from 1,134 post-first tokens.
It sustains 155.05 MTP token/s versus 145.27 ordinary token/s on the identical long boundary (+6.73%), accepts
643/981 Target-verified drafts (65.55%), and exactly matches all 1,135 ordinary Target output IDs. It is one bounded
characterization rather than a formal retained distribution; its context advances to roughly 17.5K, so neither
long-run number is directly comparable to M20's accepted 63-interval median. The fixed 1,135-token profile ignores
EOS by design and therefore measures sustained decode rather than natural response completion. Compact evidence is
`artifacts/m25/development-long-d2-characterization.json`.

The fixed-depth implementation now prepares complete device-controlled D1, D2 and D4 chains. Each specialization
captures Assistant proposal, exact T=2/3/5 Target verification, greedy acceptance, transactional circular K/V and
hidden-state commit, loop continuation and the final ordinary tail in one CUDA Graph replay. Local and global
attention, controlled input construction, K/V backup/restore and chain transitions are specialized by fixed depth;
none of the three measured profiles uses the diagnostic verifier fallback. D3 is not a selected profile. A
batch-controlled RoPE-table kernel preserves the prior table bit-for-bit.

On the final bounded 16K+1,135 D2 graph run, one chain-graph launch executed 490 verification groups with no host
group round-trips. All 1,135 output IDs exactly match ordinary Target decode; 642/980 drafts were accepted (65.51%).
Post-first throughput is 154.89 token/s versus 145.97 ordinary (+6.11%). Capture/instantiation observes 4 MiB and
8 MiB free-device-memory deltas for the separately instantiated group and chain executables; these are conservative
residency observations rather than exact CUDA graph-pool allocation sizes. The profile leaves 831,324,160 bytes free
after Target, Assistant and both graphs, which
passes the 700 MiB 32K gate. This remains one development characterization, not the formal warm-up/retained
distribution. Compact evidence is `artifacts/m25/development-fixed-d2-graph-characterization.json`; raw reports stay
under the ignored `artifacts/raw/` tree.

The subsequent apples-to-apples fixed-depth comparison uses the same binary and bounded 16K+1,135 workload for all
three depths. Every D1/D2/D4 run matches all 1,135 ordinary Target tokens and reports zero non-finite steps. D1
accepts 484/650 drafts (74.46%) but reaches 149.64 token/s; D2 accepts 642/980 (65.51%) and wins at 154.81 token/s;
D4 accepts 758/1,488 (50.94%) and falls to 125.80 token/s because rejected T=5 rows overpay verification. All fixed
graphs are resident together and leave 808,255,488 bytes free, still above the 700 MiB 32K gate. D2 therefore remains
the fixed-depth product candidate. Compact evidence is
`artifacts/m25/development-fixed-depth-graph-comparison.json`.

The 145.97 ordinary number is local to this forced-output diagnostic: it suppresses both stop-token IDs so the model
cannot end before 1,135 outputs, which makes ordinary `SelectToken()` run an additional full-vocabulary suppressed
argmax every token. It is not an M20 product-path regression. A bounded current-worktree control using the unchanged
M20 16K+64 job, no forced-output suppression and the M20 timing boundary reports 149.994 token/s with the accepted
`c750d0...` output versus M20's 150.615 ten-run median. The MTP/ordinary forced-output comparison remains internally
fair because both sides use the same suppression and timing semantics.

Greedy qualification deliberately precedes sampled MTP so fixed-depth cost and deterministic Target identity remain
separable from stochastic acceptance. Ordinary 26B token selection already supports GPU sampling. Sampled MTP is a
follow-up M25 slice that will reuse the protected 12B lossless sampling and speculative repetition-mask primitives:
the Target samples every verifier row with identical seed/step semantics, drafts are accepted only when they equal
that Target sample, and seeded replay plus transactional RNG/repetition-state commit are tested inside the selected
fixed-depth graph. Standard p/q residual rejection sampling remains a later research path.

## Phase C — exact verification runtime

- proposal lengths selected from the existing verified design (for example D1/D2/D4), based on measured benefit;
- multi-row Target verification and any T>1 head kernels are implemented here;
- proposals never directly determine emitted output;
- tentative Target K/V, hidden, RNG and repetition state commit transactionally;
- ordinary and MTP output IDs match under the same deterministic controls;
- no token-loop allocation or host-driven per-token decision.

## Phase D — memory, context and performance

- qualify ordinary versus MTP speed and acceptance;
- pass 32K with at least 700 MiB free-device margin; if this fails, M25 fails, the M23 base profile remains supported and the blocker is recorded;
- attempt 64K first with at least 200 MiB free-device margin; a 100–199 MiB diagnostic profile may be promoted only
  after repeated fresh-process lifecycle, deterministic-output, zero-allocation-delta and capacity-rejection gates;
- determine `mtp_max_context` independently from `base_max_context`;
- ensure enabling MTP never silently reduces the requested context.

## Exit gate

- [ ] Compatible assistant source/artifact is locked and validated.
- [ ] Assistant plus verifier memory is fully named and admitted at 32K with at least 700 MiB free-device margin.
- [ ] Exact ordinary/MTP output identity and transactional state tests pass.
- [ ] MTP provides a measured benefit for at least one supported mode; otherwise M25 fails and the M23 base profile remains the supported result.
- [ ] 64K has an explicit pass/fail result with its measured reserve; `mtp_max_context`, acceptance and speed are
  published honestly.
- [ ] Base 26B and 12B paths remain unchanged when MTP is disabled.
- [ ] Deferred M19 is accepted before program-complete, shipping or production-quality status is claimed.

Vision is explicitly excluded and belongs to a separate future program.
