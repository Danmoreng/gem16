# Active contract — Gemma 4 26B Fast Track R4

Status: accepted through M11; M12 attention and FP8 K/V integration is next
Plan revision: `fast-track-r4`
Accepted baseline: M00–M11
Current milestone: M12 (ready)
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

M00–M11 are accepted and are not reopened by current work.

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
6. **M19–M21:** a frozen optimized artifact passes quality, performance and long-context qualification.
7. **M23:** base target hashes, evidence and rollback are frozen.
8. **M25:** MTP compatibility, exactness, performance and the 32K MTP memory gate pass; 64K and the maximum safe MTP context are measured.

M18 is conditional diagnosis, not a normal prerequisite for M14–M17. M24 is optional and never blocks the production path.

## Memory contract

- Use CUDA-visible capacity and direct `cudaMemGetInfo`, not nominal board memory.
- Keep the preliminary immutable-weight target at or below 14,100 MiB; above 14,300 MiB is a format/review stop before kernel work.
- 32K must leave at least 700 MiB after initialization, graph capture and warm execution.
- M09's earlier residency gate uses the real uploaded artifact plus touched named reserves before execution exists;
  M11/M12/M15 must revalidate the same 32K margin after graph capture, warm execution and measured prefill planning.
- Base-model 64K and larger advertised profiles must leave at least 400 MiB after the same process. MTP retains its
  separate 500 MiB rule until M25 qualifies assistant overhead.
- The experimental base-model maximum is the largest measured context that still leaves at least 400 MiB; do not
  advertise a larger allocation-only result.
- Base and MTP maxima are separate because assistant weights and verification workspace consume memory. M25 cannot pass with an MTP profile below 32K; a lower measured result remains diagnostic while M23 stays supported.
- One 26B slot is the only positive admission case on a 16 GB device. A second slot must be rejected clearly.
- No token-loop allocation, filesystem access, JIT, repack or host routing is permitted.

## Quality contract

M13 is the only early quality gate. It uses a development corpus and selected teacher-forcing captures to reject catastrophic or unexplained drift. M19 owns held-out qualification. M18 runs only when M13/M19 fails, when a head-format decision requires attribution, or when a causal claim about QAT versus quantizer behavior will be published.

Do not use the held-out set for quantizer tuning. Do not claim “QAT quality” from provenance alone.

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

Dirty worktrees may run unit tests, fixtures, single-layer probes and bounded throughput probes. A full model conversion or publication benchmark should start from reviewed, targeted-tested code with a clean worktree and source/output preflight. If the run is intentionally diagnostic, record that fact instead of blocking unrelated implementation.

Do not repeat an expensive run solely to produce a second prose record. M08 complete-artifact reproducibility still requires two clean builds with identical hashes.

## Current unblocked work

M09 was accepted on 2026-08-14.

- M08's complete hybrid artifact, external lock and direct loader are accepted.
- M09 real payload upload, named CUDA-region reconciliation and 32K/64K base residency are accepted.
- M11's device-routed, allocation-free CUDA MoE reference is accepted; M12 attention and FP8 K/V integration is the next vertical slice.
- Evaluation-harness work and M25 feasibility work remain independent of the vertical path.
