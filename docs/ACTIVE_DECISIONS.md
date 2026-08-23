# Active decisions

**Accepted:** 2026-08-23 · **Track:** Gemma 4 26B A4B Fast Track · **Status:** M00–M17 and M22 accepted; bounded prefill, M21 and M20 next; full M19 deferred

This is the short operational policy for current work. It is not a replacement for historical evidence. Permanent
rules in `AGENTS.md` remain binding. For facts about the implementation, current source, tests and accepted evidence
win; this file defines the active execution choices; the detailed active contract and milestone status provide the
next task entry. Read historical `docs/DECISIONS.md` only for a specific historical question or disputed detail.

## Fast-track target

Reach an experimental, text-only Gemma 4 26B A4B execution on one approximately 16 GB Blackwell GPU as quickly as
possible, first proving a directly usable QAT-derived FP8/NVFP4 path and a real 32K one-slot fit. This is not yet a
quality-, performance- or release-qualified product. The existing 12B path remains the protected production baseline.

The first vertical path is:

```text
M06 NVFP4 experts → M07 provisional tied head → M08 artifact/loader
→ M09 real 32K residency → M13 slow correct model/early quality screen
→ M17 optimized runtime → base qualification → MTP feasibility/integration
```

M08 was accepted on 2026-08-14 at implementation commit `f433358b8e2c1250b95801fc898faee4fcedcbe5` after
byte-identical clean complete-artifact builds, direct-loader validation, protected 12B inspection and exact-arena
reference-GPU admission. M09 was accepted on 2026-08-14 at implementation commit
`6c3b9e456bc7fed68e2e90a51ba20c1c895fd085`: the clean real-artifact probe passes 32K and owner-required 64K
residency, exact single-arena upload, second-slot rejection and protected 12B regressions. See
`artifacts/m09/acceptance.json`.

M10 now has a passing independent CPU/NumPy replay for the locked BF16 router, shared MLP, 24 post-norm boundaries
and all 64 selected expert contributions at eight real capture points, plus a bounded independent decoder for real
M08 NVFP4 rows. It deliberately
uses lower expert ID for exact top-k ties and FP32 top-k-slot accumulation; the pinned PyTorch capture has one
tie-equivalent ordering difference and uses expert-ID/BF16 accumulation. The measured drift passes the explicit M10
boundary gates. M10 was accepted on 2026-08-14 at clean implementation commit
`eac6b443b239d5e04c5be5daef3dd659d57d5de9`; see `artifacts/m10/acceptance.json`.

M11 was accepted on 2026-08-22 at implementation commit `91ee47586cc426c051dee247ddfcf4a6b765ecfd` and M12 at
implementation commit `bbee9cd930133dd49cb3acc79b4867658a0968cc`. The isolated CUDA references keep routing on device,
use fixed initialization-time bindings, preserve distinct FP8 K/V ownership and repeat without allocation deltas.
Their clean records are `artifacts/m11/acceptance.json` and `artifacts/m12/acceptance.json`. M13 is now the first
full-model integration and the only early quality go/no-go screen.

M13 was accepted on 2026-08-22 with decision `proceed` at implementation commit
`3cd697501585868d5ef41e60d212bb0e502c365c`. The complete experimental reference path passes deterministic
generation, BF16 teacher-forced KL/rank, selected layer/router drift, resident continuation, finite-numerics and
warm 32K memory gates. See `artifacts/m13/acceptance.json`.

M14–M16 were accepted on 2026-08-22 at implementation commit
`9a374c3dda10b7ae870c712cd70a60aa0a9e2c52`; M17 was accepted at implementation commit
`57fdeb309aacfce2e4eba65745fba86f14ebd113` and closed after safety, lifecycle and decode-performance hardening at
`348683e167c6c4d3b0be7580e404c097b199a3d8`. The frozen text-only profile keeps the 14,696,668,160-byte target arena,
separate FP8 K/V, fixed-address prefill/decode workspaces and whole-model decode graph. It rejects non-finite routing
and logits, reports SM120 capabilities only on compatible hardware and passes the real two-process engine lifecycle
smoke. The final short diagnostic remains logit-bitwise identical to the accepted profile and estimates 13.0903 ms
per decode token (76.3924 token/s); this is an M20 candidate measurement, not the controlled M20 promotion result.
See `artifacts/m17/closure-hardening.json`.

The M17 artifact/profile freeze unblocks M19 held-out quality, M20 controlled performance, M21 real long-context and
M22 CLI/server integration. M18 remains conditional and was not triggered; its number does not make it a sequential
prerequisite.

On 2026-08-23 the owner removed the 49 GB QAT-BF16 runtime comparison from M19 because that reference cannot be
executed safely on the target laptop. M19 instead uses Google's immutable official QAT Q4_0 GGUF at revision
`d1c082be9cf3c8a514acf63b8761f4b41935842e` through pinned llama.cpp `0b14b87d7c20cb753b94b96854dd7b45306fc696`
as its executable paired quality reference. This supersedes the former requirement that M19 task, prose and
teacher-forced evidence run the 49 GB QAT-BF16 checkpoint. Historical BF16 evidence from M10/M13 remains valid, but
no new 49 GB model execution is part of M19.

On 2026-08-23 the owner also deferred the remaining multi-hour M19 task and prose benchmark suite until the end of
the current implementation/performance program. Bounded numerical checks against the pinned official QAT Q4_0
reference remain valid correctness evidence, but M19 stays pending and no production-quality claim is allowed.
This explicitly supersedes the former requirement that M19 pass before an engineering M23 freeze. M22 product
qualification was accepted at implementation commit `f0aa302aa0246d44e1c8477dbbbb67fbbe2d2037`; its compact
record is `artifacts/m22/acceptance.json`. One bounded profile-driven prefill decision now closes before the candidate
is frozen. M21 then performs
real 32K/64K and maximum-context execution; M20's approved 3-warm-up/10-retained performance qualification consumes
that matching M21 evidence. M23 may then freeze a **technical base Target** while carrying
the deferred-M19 limitation; it is not a shipping or quality-qualified release until M19 is eventually accepted.
No other multi-hour or broad quality benchmark is authorized in this wave. Targeted operator/model correctness,
12B regressions, the explicitly approved M20 run and real M21 context runs remain in scope.

The owner set the base-model 64K residency and later execution reserve to 400 MiB on 2026-08-14 so that 64K remains
a required supported target. This supersedes the prior 500 MiB base-model rule for 64K and larger contexts. The 32K
gate remains 700 MiB, and MTP keeps its separate 500 MiB 64K gate until M25 measures assistant overhead.

The owner clarified on 2026-08-14 that M09 should load the real artifact and prove pre-execution residency so the
critical path reaches inference quickly. This supersedes the old M09-card wording that required captured execution
graphs, warm model execution and an executable bounded-prefill selector before M09 could pass. M09 must still reserve
and touch every named graph/workspace region. M11/M12 own the first executable/captured paths, M15 owns measured
prefill chunk selection, and the final 700 MiB post-warm 32K requirement remains binding and must be revalidated there.

M10/M12 semantic and attention fixture work may proceed in disjoint slices when it does not delay the vertical path.
The integration branch remains `feat/gemma4-26b`; temporary worktrees are allowed only with explicit file ownership.

## Active implementation choices

- **M06:** perform one clean full QAT-BF16-to-NVFP4 expert conversion. Require exhaustive small codec, shape, byte,
  determinism and representative real-shape operator-consumption tests, bounded-memory evidence and sampled
  Ordinary/Unsloth diagnostics sufficient to catch convention errors. A complete Ordinary conversion and exhaustive
  Ordinary-versus-Unsloth attribution are conditional work, not M06/M07 blockers. This explicitly supersedes the
  former future-stage requirement for a full Ordinary conversion at every M05–M07 partial stage.
- **M07:** use one provisional NVFP4 tied embedding/output head for the first complete artifact. The accepted
  implementation compiles one QAT tied source into four aliased components and validates CPU lookup/T=1 reference
  semantics. An internal Q4_0 encoder/backend and broad head A/B study are optional and do not block the first
  executable path. MTP verifier head batches belong to M25, not M07.
- **M13:** is the only early quality go/no-go screen. It must still check deterministic generation, teacher-forced
  drift and catastrophic numerical behavior. **M18** is conditional diagnosis/attribution when M13 or later quality
  fails, a head decision needs attribution, or the owner requests it; it is not a normal prerequisite for native work.
- **Memory:** qualify one fully resident 26B slot at 32K with at least 700 MiB directly measured free CUDA memory.
  Treat 64K and the measured maximum safe context as later qualification work. Keep the existing 14,100 MiB target
  and 14,300 MiB review stop for the immutable weight arena.
- **MTP and vision:** MTP follows a frozen base target and requires its own assistant, exactness and memory work.
  Vision is a separate later track and is outside this fast path.

## Scope boundary

The accepted M00–M17 source locks, evidence and historical records remain valid. The active choices above simplify
future execution; they do not authorize silent precision changes, CPU weight offload, duplicate device layouts,
runtime quantization, unreported fallback, weakened 12B behavior or unsupported capability claims. Experimental
results must say which gates have not yet been run.
