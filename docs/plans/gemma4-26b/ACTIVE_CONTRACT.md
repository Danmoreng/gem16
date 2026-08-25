# Active contract — Gemma 4 26B Fast Track R4

Status: M00–M17 and M20–M23 accepted; M25 baseline characterization and integration next; full M19 held-out qualification deferred by owner
Plan revision: `fast-track-r4`
Accepted baseline: M00–M17 plus M20 performance, M21 long context, M22 product integration and the M23 technical Target freeze
Current milestone: bounded external 26B MTP characterization followed by M25 (M19 full suite deferred; M18 conditional)
Integration branch: `feat/gemma4-26b`

## Purpose

This is the compact, normative contract for future Gemma 4 26B work. It deliberately replaces the former full-plan reading loop with a short operational path. Read historical ledgers only when the current task needs a specific record; do not turn every small task into a repository-wide analysis.

## Conflict precedence

When documents disagree, use this order:

1. repository root `AGENTS.md` and applicable permanent security/licensing rules;
2. `docs/ACTIVE_DECISIONS.md` for accepted fast-track policy;
3. current source, executable tests, immutable source locks and accepted evidence for factual state;
4. this active contract, `FAST_TRACK_STATUS.json` and the current milestone;
5. only the specifications linked by that milestone;
6. historical `docs/DECISIONS.md`, correctness, benchmark and evidence records for targeted deep analysis.

Report a material conflict instead of guessing. A stale status page does not override current source or accepted evidence.

## Accepted baseline

M00–M17 and M20–M23 are accepted and are not reopened by current work.

- Sources, manifests and source locks are immutable.
- The 12B path remains separately specialized and regression-protected.
- The checkpoint compiler is a Python control plane plus one promoted native C++20 numerical data plane.
- M05 FP8 attention conversion is accepted at implementation commit `d91388113d68974f9ab7cec1a90ef768285c0645`.
- M06 expert/shared NVFP4 conversion is accepted at implementation commit `81055eb48e05321481a8b63dd0dc5e7e017a7c00`.
- M07 provisional NVFP4 tied-head compiler/reference stage is accepted at implementation commit `60f500b7be567fafd483ebd6f5f9b07988197ca1`.
- M08 complete hybrid artifact and direct-loader stage is accepted at implementation commit `f433358b8e2c1250b95801fc898faee4fcedcbe5`.
- M09 real one-slot residency and 32K/64K admission are accepted at implementation commit
  `6c3b9e456bc7fed68e2e90a51ba20c1c895fd085`.
- M10 independent BF16/NVFP4 CPU MoE semantics are accepted at implementation commit
  `eac6b443b239d5e04c5be5daef3dd659d57d5de9`.
- M11 fixed-address CUDA correctness-first MoE is accepted at implementation commit
  `91ee47586cc426c051dee247ddfcf4a6b765ecfd`; it is explicitly not performance-qualified.
- M12 attention, RoPE and separate FP8 K/V integration is accepted at implementation commit
  `bbee9cd930133dd49cb3acc79b4867658a0968cc`; it is explicitly not performance-qualified.
- M13 full-model reference execution and early quality screen are accepted with decision `proceed` at implementation
  commit `3cd697501585868d5ef41e60d212bb0e502c365c`; the path remains experimental and reference-only.
- M14 native batch-one MoE decode, M15 bounded grouped MoE prefill and M16 native tied T=1 head are accepted at
  implementation commit `9a374c3dda10b7ae870c712cd70a60aa0a9e2c52`.
- M17 fixed-address whole-model integration is accepted at implementation commit
  `57fdeb309aacfce2e4eba65745fba86f14ebd113` and closure-hardening commit
  `348683e167c6c4d3b0be7580e404c097b199a3d8`; its artifact/profile contract remains the basis for current work, and
  the post-M17 implementation candidate is frozen separately by M20–M23 evidence.
- M22 CLI/server product integration is accepted at implementation commit
  `f0aa302aa0246d44e1c8477dbbbb67fbbe2d2037`; its real 26B product gate, exact M08 provenance, one-slot/text-only
  errors, observability and protected 12B regression record are in `artifacts/m22/acceptance.json`.
- M21 real long-context execution is accepted on the frozen native candidate: 32K, 64K and 96K each pass twice,
  100K is reproducibly capacity-rejected, and `base_max_context` is 98,304. See `artifacts/m21/acceptance.json`.
- M20 controlled performance is accepted after three warm-ups and ten retained real Wikipedia 16K+64 runs: retained
  medians are 6,572.809 prompt and 150.615 ordinary-decode token/s, with the unchanged `c750d0…` output hash. See
  `artifacts/m20/acceptance.json`.
- M23 freezes that exact ordinary-decode Target at capability implementation revision
  `c8e09e4e337d58ac0cfe402585ef818135845faa`. CLI/server reporting, the real 32K product lifecycle and the protected
  12B regression were revalidated; the hash reconciliation, capabilities, limitations and rollback are in
  `artifacts/m23/acceptance.json`.
- Runtime conversion, CPU expert offload, expert streaming and duplicate persistent GPU weight layouts remain forbidden.

## Product target

The program target is a single-user Gemma 4 26B A4B profile on one approximately 16 GB NVIDIA Blackwell GPU.

Required base target:

- text inference, batch one and resident multi-turn continuation;
- one resident 26B execution slot;
- fully resident target weights with no CPU weight offload;
- 32K context as the first production gate;
- 64K as the next qualification target;
- a measured `max-single-user` context profile equal to the highest context that passes the same safety and correctness rules;
- deterministic greedy execution and the existing sampling semantics;
- CLI and server support before optional Studio work.

Required base performance target for M20:

- exact `wikipedia-real-16k64-greedy` batch-one row: 16,384 prompt tokens, 64 output forwards and 63 timed decode
  intervals at context 16,448;
- native ordinary Target execution with FP8 KV and CUDA Graph replay; MTP/speculative decode, prompt cache, CPU
  offload, fallback and recurring allocation disabled;
- unchanged prompt/output/sampling/cache/KV semantics and unchanged recorded timing boundaries;
- median of the ten retained runs after three warm-ups at **at least 6,000 prompt token/s and 150 ordinary decode
  token/s**;
- a separately reported, non-blocking **6,500 prompt token/s** competitive stretch result.

M20 passed both hard targets and the prompt stretch on 2026-08-25. External vLLM
characterizations motivate the fixed values but do not replace the native correctness, memory or timing contract.

Required final target:

- a 26B-compatible MTP assistant or assistant artifact;
- exact Target verification, transactional state commit and ordinary/MTP output identity under matched deterministic controls;
- the highest safe MTP context measured independently from the base-model maximum;
- no reuse of the 12B assistant unless exact architecture and training compatibility are independently proven, which is not currently assumed.

Vision is not part of this program. It remains a separate post-program track and must not share the MTP milestone.

## Active precision and artifact contract

The first vertical candidate is:

| Family | Active candidate |
|---|---|
| Routed experts | NVFP4, group 16, native Blackwell path |
| Shared/dense MLP | same NVFP4 contract |
| Attention Q/K/V/O | accepted FP8 contract |
| Router, norms and scalar controls | BF16/F32 from the locked source |
| Tied embedding/output head | provisional NVFP4 for the first complete artifact |
| KV cache | separate FP8 K and V |
| Source model | one exact Google QAT BF16 revision |

The tied head may be revised only after measured quality or performance evidence. An internal Q4_0 encoder/backend is optional M24 work; official Q4_0 remains an external reference and is not a blocker for M08.

The final artifact is text-only Safetensors plus explicit versioned metadata and an external lock. It contains one physical tied head and one canonical expert payload. Runtime-specific tiling may be produced during bounded load into the single final device layout, but no second persistent layout may remain.

## Gate policy

Only these owner-level gates block the critical path:

1. **M06:** native NVFP4 contract and one complete QAT expert conversion are valid.
2. **M07:** one QAT tied head compiles, matches independent CPU references, and uses one physical aliased payload.
3. **M08:** one complete reproducible artifact loads directly and validates.
4. **M09:** the real artifact passes one-slot 32K admission with at least 700 MiB directly measured free-device margin.
5. **M13:** the full slow model is deterministic and passes the single early quality go/no-go screen.
6. **M17:** one fixed-address optimized artifact/profile is frozen with deterministic engine replay and no recurring allocation.
7. **M20–M22:** with M22 accepted, after bounded correctness-preserving prefill/decode optimization one clean
   candidate passes real long-context in M21 and the fixed 6,000 prompt/150 ordinary-decode token/s M20 gates using
   the matching M21 evidence; 6,500 prompt token/s is the reported stretch target.
8. **M23 engineering freeze:** accepted; Target hashes, available evidence and rollback are frozen with M19 explicitly pending.
9. **Deferred M19:** the multi-hour held-out task/prose suite eventually passes before any shipping or production-quality claim.
10. **M25:** MTP compatibility, exactness, performance and the 32K MTP memory gate pass; 64K and the maximum safe MTP context are measured.

M18 is conditional diagnosis, not a sequential prerequisite after the accepted M17 path. M24 is optional and never blocks the production path.

## Memory contract

- Use CUDA-visible capacity and direct `cudaMemGetInfo`, not nominal board memory.
- Keep the preliminary immutable-weight target at or below 14,100 MiB; above 14,300 MiB is a format/review stop before kernel work.
- 32K must leave at least 700 MiB after initialization, graph capture and warm execution.
- M09's earlier residency gate uses the real uploaded artifact plus touched named reserves before execution exists;
  M11/M12/M15 must revalidate the same 32K margin after graph capture, warm execution and measured prefill planning.
- Base-model 64K and larger advertised profiles must leave at least 400 MiB after the same process. MTP retains its
  separate 500 MiB rule until M25 qualifies assistant overhead.
- The accepted base-model maximum is 98,304 tokens, leaving 434,962,432 bytes after real execution against the
  419,430,400-byte gate. 102,400 is reproducibly rejected with a 26,411,008-byte shortfall.
- Base and MTP maxima are separate because assistant weights and verification workspace consume memory. M25 cannot pass with an MTP profile below 32K; a lower measured result remains diagnostic while M23 stays supported.
- One 26B slot is the only positive admission case on a 16 GB device. A second slot must be rejected clearly.
- No token-loop allocation, filesystem access, JIT, repack or host routing is permitted.

## Quality contract

M13 is the only early quality gate. It uses a development corpus and selected teacher-forcing captures to reject catastrophic or unexplained drift. M19 owns held-out qualification. M18 runs only when M13/M19 fails, when a head-format decision requires attribution, or when a causal claim about QAT versus quantizer behavior will be published.

Do not use the held-out set for quantizer tuning. Do not claim “QAT quality” from provenance alone.
For M19, the executable paired reference is Google's pinned official QAT Q4_0
GGUF through the pinned llama.cpp runtime. The owner removed a broad or long
49 GB QAT-BF16 run from M19's mandatory suite, but explicitly permits bounded
CPU or CPU/GPU/disk-offload QAT-BF16 runs for Golden Gates and targeted
numerical diagnosis. Such runs use an explicit memory budget, remain
performance-ineligible, and do not enter the production runtime. Accepted
M10/M13 BF16 evidence remains historical support rather than an M19 runtime
prerequisite.

The owner deferred M19's remaining multi-hour task and prose suite until after the current product, performance,
context and engineering-freeze work. The already completed bounded Q4 numerical comparison remains a correctness
signal only. Until M19 passes, the 26B profile remains experimental and M23 cannot be described as a shipping,
quality-qualified or production release.

## Parallel execution contract

One integration branch remains authoritative, but independent work may proceed in ephemeral worktrees or sub-agent branches. Each task packet must declare:

- owned files/directories;
- read-only shared interfaces;
- prerequisite commit/evidence;
- required tests and evidence;
- merge dependency;
- forbidden overlap.

Sub-agents do not edit `ACTIVE_CONTRACT.md`, `FAST_TRACK_STATUS.json`, shared manifest schemas or integration-owned orchestration files unless assigned by the integration agent. See `PARALLEL_WORKSTREAMS.md`.

## Full-run discipline

Dirty worktrees may run unit tests, fixtures, single-layer probes and bounded throughput probes. A full model conversion or publication benchmark should start from reviewed, targeted-tested code with a clean worktree and source/output preflight. If the run is intentionally diagnostic, record that fact instead of blocking unrelated implementation. The current owner-approved wave excludes multi-hour broad quality benchmarks; the bounded M20 3/10 run and M21 32K/64K execution are explicit exceptions.

Do not repeat an expensive run solely to produce a second prose record. M08 complete-artifact reproducibility still requires two clean builds with identical hashes.

## Current unblocked work

- M17 and M20–M23 are accepted; M23 is the frozen ordinary-decode Target and rollback for M25.
- Bounded, freshly pinned llama.cpp and vLLM 26B ordinary/MTP characterizations may run before native M25 integration.
- M19 numerical Q4 evidence is available, but its multi-hour task/prose suite is deferred and remains a release limitation.
- M18 remains inactive unless quality failure, head uncertainty or explicit causal-attribution work triggers it.
- M25 assistant feasibility and target integration are unblocked; M23 remains supported if MTP fails exactness,
  memory or measured-benefit gates.
