# Active decisions

**Accepted:** 2026-08-14 · **Track:** Gemma 4 26B A4B Fast Track · **Status:** M00–M08 accepted; M09 implemented, clean acceptance pending

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
reference-GPU admission. M09 now has a passing real-artifact diagnostic upload and 32K residency reconciliation;
formal acceptance still requires the same bounded probe on the clean implementation commit.

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

The accepted M00–M08 source locks, evidence and historical records remain valid. The active choices above simplify
future execution; they do not authorize silent precision changes, CPU weight offload, duplicate device layouts,
runtime quantization, unreported fallback, weakened 12B behavior or unsupported capability claims. Experimental
results must say which gates have not yet been run.
