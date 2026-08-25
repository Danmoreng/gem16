# Active decisions

**Accepted:** 2026-08-25 · **Track:** Gemma 4 26B A4B Fast Track · **Status:** M00–M17 and M20–M23 accepted; M25 next; full M19 deferred

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

On 2026-08-23 the owner removed the 49 GB QAT-BF16 runtime comparison from the mandatory M19 suite. M19 instead uses
Google's immutable official QAT Q4_0 GGUF at revision `d1c082be9cf3c8a514acf63b8761f4b41935842e` through pinned
llama.cpp `0b14b87d7c20cb753b94b96854dd7b45306fc696` as its executable paired quality reference. This supersedes the
former requirement that M19 task, prose and teacher-forced evidence run the 49 GB QAT-BF16 checkpoint.

On 2026-08-24 the owner clarified that the 49 GB checkpoint restriction applies to a fully GPU-resident load and to
long benchmark or quality-suite execution, not to short correctness work. Bounded CPU or explicit CPU/GPU/disk
offload runs of the pinned QAT-BF16 checkpoint are allowed for Golden Gates and targeted numerical diagnosis. They
remain diagnostic and performance-ineligible, must use an explicit host/GPU memory budget, and must not silently
enter the production path. The accepted M01 capture (approximately 49 GiB peak process RSS and 11.25 GB peak CUDA
allocation on this 62 GiB/no-swap laptop) is the feasibility boundary; a planned run that materially exceeds it
requires a new capacity preflight. Historical BF16 evidence from M10/M13 remains valid, while the broad M19 suite
stays on the official QAT Q4_0 reference and remains deferred.

On 2026-08-23 the owner also deferred the remaining multi-hour M19 task and prose benchmark suite until the end of
the current implementation/performance program. Bounded numerical checks against the pinned official QAT Q4_0
reference remain valid correctness evidence, but M19 stays pending and no production-quality claim is allowed.
This explicitly supersedes the former requirement that M19 pass before an engineering M23 freeze. M22 product
qualification was accepted at implementation commit `f0aa302aa0246d44e1c8477dbbbb67fbbe2d2037`; its compact
record is `artifacts/m22/acceptance.json`. Bounded profile-driven prefill/decode optimization toward the fixed M20
targets now closes before the candidate is frozen. M21 then performs
real 32K/64K and maximum-context execution; M20's approved 3-warm-up/10-retained performance qualification consumes
that matching M21 evidence. M23 may then freeze a **technical base Target** while carrying
the deferred-M19 limitation; it is not a shipping or quality-qualified release until M19 is eventually accepted.
No other multi-hour or broad quality benchmark is authorized in this wave. Targeted operator/model correctness,
12B regressions, the explicitly approved M20 run and real M21 context runs remain in scope.

On 2026-08-23 the owner replaced llama.cpp parity and relative-only performance wording with fixed vLLM-class M20
targets. The bounded promotion row is `wikipedia-real-16k64-greedy`: batch one, the exact 16,384-token prompt manifest
SHA-256 `9a5859b979d91fccf71bcbb61aade6372cf2cc3c708e6c47b8b6cfd99f7abd2d`, 64 output forwards, 63 timed
post-first-token decode intervals and a 16,448-token context. It uses the native base Target, FP8 KV, deterministic
greedy execution and CUDA Graph replay with MTP/speculative decode, prompt cache, CPU offload, fallback and recurring
allocation disabled. The existing timing boundaries remain fixed: prompt time is first prefill launch through prefill
synchronization, TTFT is request-ready through first-token-ready, decode time is the sum of the 63 synchronized
post-first-token intervals, and model load is excluded.

M20 now requires the median of the ten retained runs after three warm-ups to reach at least **6,000 prompt token/s**
and at least **150 ordinary decode token/s**. **6,500 prompt token/s** is the non-blocking competitive stretch target.
No prompt, output length, cache state, KV precision, sampling, speculative mode or timing-boundary substitution may
satisfy these gates. A result below either hard target keeps M20 open unless a later explicit owner decision changes
the gate. The vLLM 0.27.1 community-W4A16 observation of 6,475.795 prompt token/s and 149.348 decode token/s only
motivates the fixed targets; its checkpoint and prefill timing boundary differ, so it is not numerical parity or a
moving acceptance dependency. This decision supersedes the prior generic multi-scenario/relative-only M20 promotion
wording for the current bounded wave. The M20 qualifier must be aligned with this contract before formal execution;
its current mandatory multi-scenario and non-deferred-M19 gates are stale under the active owner decisions.

On 2026-08-24 the integrated SM120 26B prefill backend promoted the BF16 Tensor-Core router after a bounded
four-prompt, 14-row QAT-BF16 Golden Gate, real-shape CUDA and sanitizer checks, deterministic engine relaunch, real
26B product execution and protected 12B regression. Two canonical default-path runs average 6,574.16 prompt tok/s,
passing both the 6,000 hard target and 6,500 stretch target with unchanged timing boundaries, memory semantics and
zero fallback/allocation. Tensor-Core changes the exact quantized output hash but improves the bounded aggregate
QAT-BF16 Top-1/KL/NLL tail metrics; the serial exact router remains an explicit per-engine pre-execution rollback.
Formal M20 remains open because ordinary decode averages 139.05 rather than 150 tok/s and the 3-warm-up/10-retained
protocol has not run. The next bounded work is last-chunk-only output-head/argmax, safe non-aliasing prefill staging,
and reducing greedy Prediction host synchronization/D2H transfers.

On 2026-08-25 the retained exact development series advanced the same canonical row to 6,586.40 mean prompt tok/s
and 140.459 mean ordinary-decode tok/s. Final-chunk-only prefill output, global value-cache ping-pong staging, split
routed Gate/Up plus fused BF16-product/NVFP4 quantization, and a compact pinned status/device-self-feed host tail
preserve the `c750d0…` output hash, fixed arena and zero fallback/allocation contract. The full-logit D10 head fusion
was rejected. M20 remains open by 9.541 token/s (6.79% throughput); exact parallel Router Top-8 is the next isolated
candidate. Compact evidence is under `artifacts/m20/optimization-final-chunk-only.json`,
`artifacts/m20/optimization-global-value-pingpong.json` and
`artifacts/m20/optimization-decode-moe-split-fused-quant.json`, plus
`artifacts/m20/optimization-compact-prediction-self-feed.json`.

Later on 2026-08-25 the exact development series reached 6,568.395 mean prompt tok/s and 150.413 mean ordinary
decode tok/s over three adjacent canonical runs. The retained slice fuses selected W2 with the original BF16,
slot-ordered reduction epilogue, uses eight useful warps per CTA for routed and shared W13 without changing logical
work or arithmetic, and omits an unconsumed native router-normalization diagnostic write. The output remains
`c750d0…`; real-shape randomized differentials, the five focused test groups, memcheck/racecheck/initcheck, real
two-process engine replay/relaunch and protected 12B product regression pass. This satisfies the fixed throughput
targets for a **development candidate**, not formal M20: M21 matching 32K/64K execution and M20's three-warm-up/ten-
retained median qualification are still required. See
`artifacts/m20/optimization-decode-fused-expert-epilog.json`.

The frozen candidate then passed M21 and formal M20 on 2026-08-25. M21 executes 32K, 64K and 98,304 tokens twice
in fresh processes with deterministic finite logits, real ring/global-boundary coverage, zero fallback/allocation
and unchanged margins; 102,400 is reproducibly capacity-rejected, so `base_max_context=98,304`. Formal M20 uses
the real Wikipedia 16K+64 row, three warm-ups and ten retained runs with continuous telemetry. Its retained medians
are 6,572.809 prompt tok/s and 150.615 ordinary-decode tok/s; the 6,000/150 gates and 6,500 prompt stretch all pass
with unchanged `c750d0…` output. M20 and M21 are accepted and consumed by the technical M23 freeze, while M19
remains explicitly deferred and blocks shipping/production-quality claims. See `artifacts/m20/acceptance.json` and
`artifacts/m21/acceptance.json`.

Technical M23 was accepted on 2026-08-25 with implementation/evidence revision
`c8e09e4e337d58ac0cfe402585ef818135845faa`. The CLI and server keep 32K as the default, report 64K as qualified
and publish `base_max_context=98,304`; the real 32K product lifecycle and protected 12B product regression pass on
the updated binaries. `artifacts/m23/acceptance.json` reconciles the exact M20/M21 artifact, source, toolchain,
benchmark-binary and output hashes and freezes this ordinary-decode profile as the M25 rollback Target. M19 remains
visibly pending, so M23 is an experimental engineering checkpoint rather than a shipping or production-quality
release. Before native M25 integration, update and pin the available llama.cpp and vLLM reference runtimes and run
bounded 26B ordinary/MTP baseline characterizations to select proposal lengths and a realistic performance target.

That prerequisite completed on 2026-08-25. vLLM 0.27.1 remains the latest published wheel and its retained ordinary
CUDA-Graph comparison stays near 6,400 prompt / 150 decode token/s; native 26B MTP construction with both the
official BF16 and a supported ModelOpt NVFP4 assistant exceeds the 16 GB device, so the owner directed that vLLM not
be rerun and remain the ordinary-only comparison. llama.cpp was updated to build 10623
(`f1357e49980f5462af9783164f3fdec407d90137`). On the fixed 16K+1,135 workload its ordinary/D2 medians are
118.627/151.919 decode token/s, a 28.06% D2 gain with 62.85% draft acceptance. D2 leaves only 343 MiB free and its
output first differs from ordinary at zero-based token 44, so it is a performance target for M25 feasibility—not
quality, exactness, memory-gate or default-mode acceptance. See
`benchmarks/baselines/llama_cpp/gemma4-26b-a4b-qat-q4_0-mtp-b10623.json`.

The owner set the base-model 64K residency and later execution reserve to 400 MiB on 2026-08-14 so that 64K remains
a required supported target. This supersedes the prior 500 MiB base-model rule for 64K and larger contexts. The 32K
gate remains 700 MiB, and MTP keeps its separate 500 MiB 64K gate until M25 measures assistant overhead.

On 2026-08-25 the owner selected Google's official QAT-Q4_0-matched 26B Assistant at immutable revision
`9537141506fe8875b3ed45b264af13580cb29166` as the sole native M25 source. The production candidate uses NVFP4 for
the tied embedding/output head and MLP matrices, FP8 for attention Q/O and the 2,816↔1,024 interface projections,
and BF16 for norms and scalar controls. The official BF16 payload is the numerical oracle; all-FP8 and all-NVFP4
production candidates are out of scope unless a measured failure triggers a later owner decision. The Assistant
must read the Target KV cache and may not allocate a second long-context cache.

"Numerical oracle" applies only to component-level Assistant differential tests. It does not make the BF16
Assistant's proposed token authoritative. M25 draft quality and acceptance are decided by the actual frozen Target
engine: both BF16 and compiled-hybrid proposals must be verified independently against the Target continuation. A
hybrid proposal that differs from BF16 is not a failure when the Target accepts it, and a BF16 proposal that the
Target rejects is not counted as correct.

The owner also superseded M25's fixed 500 MiB rule for 64K and larger with a measured, profile-specific safety
reserve. M25 still qualifies 32K first against the conservative 700 MiB gate. After exact Assistant and verifier
residency is known, 64K is tested first with a 200 MiB reserve and may additionally be tested down to 100 MiB only
with repeated fresh-process lifecycle, deterministic output, allocation-delta and capacity-rejection evidence.
The accepted reserve, `mtp_max_context` and all excluded larger contexts must be published explicitly. A smaller
MTP reserve never changes the base-model 400 MiB rule or permits an allocation failure during inference.

The first native 26B MTP CUDA-Graph checkpoint was implemented on 2026-08-25 as a draft-length-indexed fixed-Graph
registry with exact greedy D1, D2 and D4 specializations. One graph replay owns proposal, T=2/3/5 Target verification,
accept/commit, device-side continuation and the final ordinary tail. On identical bounded 16K+1,135 runs, all three
depths exactly match ordinary Target tokens and report no non-finite step. D1 reaches 149.64 token/s at 74.46% draft
acceptance, D2 wins at 154.81 token/s and 65.51%, and D4 reaches 125.80 token/s at 50.94%. All fixed graphs remain
resident together with 808,255,488 bytes free at 32K, passing the 700 MiB gate; D2 remains the fixed-depth candidate.
This is development evidence, not formal M25 acceptance; graph-byte observations are not exact CUDA graph-pool sizes.
The 145.97 comparator is deliberately forced-output/suppressed-token diagnostic timing, not a new ordinary product
baseline; the current worktree produces 149.994 token/s and the unchanged `c750d0...` output on one exact M20
16K+64 control, consistent with the accepted 150.615 retained median.
Greedy depth qualification remains first; sampled MTP follows as a separate correctness slice even though ordinary
26B GPU sampling and shared speculative-sampling primitives already exist.

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
