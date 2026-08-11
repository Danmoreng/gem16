# imp reference integration amendment

## Decision

The discovery of `kekzl/imp` does **not** reverse the primary architecture or checkpoint strategy in this package. The production hypothesis remains:

```text
Google Gemma 4 26B QAT-BF16 master
  → FP8 attention projections
  → NVFP4 routed experts and shared MLP
  → BF16 router/norm/scalars
  → separately qualified Q4_0 or NVFP4 tied embedding/head
  → FP8 KV
  → no resident vision in the first release
```

`imp` changes the **evidence plan, implementation order and reference set**. It is a useful implementation reference for Gemma 4 MoE semantics, consumer-Blackwell NVFP4 kernels, dispatch observability and quality/performance discipline. It is not the architecture to transplant wholesale into `gem16`.

## Pinned reference

```text
repository: https://github.com/kekzl/imp
commit: a392904d4216388828d0d56317de046f4ca49627
license: MIT, Copyright (c) 2026 kekzl
primary target: RTX 5090 / sm_120a
RTX 5080 status: compute_120f PTX fallback; not the published primary test target
```

M01 must create an immutable local source lock and verify that every cited file still matches this commit before extracting goldens or porting code.

## Measured RTX 5080 Q4_0 evidence

A same-machine exploratory run now exists for this exact imp commit and Google's official Gemma 4 26B A4B QAT
Q4_0 GGUF. On the 16 GB RTX 5080 Laptop, imp keeps only 21 of 30 MoE expert layers on the GPU and leaves 3.59 GiB
of expert weights on the host. CUDA Graphs are disabled, the expert-cache hit rate is 64.2%, and repeated
`cudaHostRegister` failures make the offload route still less representative of a resident engine.

At pp512/tg256 with ten measured repetitions, imp reaches 1,533.75 prefill tok/s and 51.64 decode tok/s. The
directly adjacent, fully GPU-resident llama.cpp b10240 result reaches 5,087.77 and 169.762 tok/s: 3.32x and 3.29x
faster. Imp's default 3,900 MiB planner reserve cannot provision a 512-token prompt; the valid run uses its
documented 256 MiB override and FP8 K/V. INT8 K/V rejects global head dimension 512, and the experimental
graph-under-offload path fails capture.

This evidence strengthens the decision not to port imp's broad runtime or host-offload architecture. It does **not**
evaluate imp's intended all-resident NVFP4 path, because the tested GGUF uses Q4_0 and resolves MoE prefill through
`legacy_fallback`. Native grouped-kernel adoption still requires the M13/M18 correctness and quality gates plus an
isolated RTX 5080 kernel study. Full evidence is in
`benchmarks/baselines/imp/gemma4-26b-a4b-qat-q4_0-characterization.json`.

## What changes in the program

1. Add an **imp reference-audit lane before runtime implementation**. This lane is documentation, fixture discovery and provenance work; it is not permission to copy a general executor.
2. Add **NVIDIA/ModelOpt Gemma 4 NVFP4 as a negative/control checkpoint arm**. The quality loss documented by imp must be reproduced or refuted independently before any claim that “NVFP4 is quality-neutral.”
3. Freeze **ModelOpt versus llm-compressor scale semantics** in host fixtures. A multiplier/divisor mix-up is a silent model-destruction bug.
4. Promote **FP32 router input/logits, exact scale-free router RMSNorm, `1/sqrt(d)`, per-expert scaling, separate shared/expert norms and FP32 residual diagnostics** into early goldens.
5. Require **plain-prose quality corpora** as the primary Gemma 4 PPL signal. Technical Markdown remains a useful out-of-domain diagnostic but may not be the sole promotion corpus.
6. Record the **actual winning dispatch path**, not merely the predicted path. Also retain the first CUDA-Graph demotion reason.
7. Add a machine-readable **decode/prefill/VRAM regression baseline** with thresholds and raw-run provenance.
8. Add a **settled-evidence ledger** so rejected hypotheses and verified absences are not repeatedly rediscovered by coding agents.
9. Add **engine destroy/recreate and CUDA static-state lifetime tests**. Any module-static pointer into an arena must be reset after that arena is destroyed.
10. Create an explicit **MIT code-adoption policy**. Directly copied files remain MIT and retain attribution; clean-room reimplementations cite the design source in provenance records.

## What does not change

Do not import these imp product choices into the 16 GB batch-one target without a separate approved product requirement:

- Paged KV or prefix-cache infrastructure;
- continuous batching;
- a general `GraphExecutor` or broad weight-handle dispatcher;
- GGUF, Mamba, GDN, LoRA or unrelated model support;
- multiple permanent weight caches or source-order plus decode-order expert copies;
- host expert offload as the production path;
- runtime fallback ladders that silently change precision.

The final `gem16` 26B design remains a small set of explicit, model-specific plans with one resident weight representation and no token-loop allocation.

## Milestone amendments

| Existing milestone | Required imp-derived amendment |
|---|---|
| M00 | Accept third-party code/provenance policy and the reference-only role of imp. |
| M01 | Pin imp commit, license and selected source files; capture reference commands and source hashes. |
| M03 | Add quantization producer, scale direction and final-layout fields to the canonical tensor inventory. |
| M06 | Add ModelOpt multiplier and llm-compressor divisor fixtures; compile a ModelOpt-style negative/control arm. |
| M10 | Freeze router/branch/residual goldens at multiple layers, including near-tie top-8 cases. |
| M11 | Prove FP32 router path and expert-scale ordering independently on CUDA. |
| M14 | Study imp's native per-expert/small-M code, but retain a Gemma-specific T=1 path and one final layout. |
| M15 | Evaluate the persistent grouped small-M work queue and M-tile policy; port only after layout/memory gates. |
| M17 | Add actual-dispatch and graph-demotion telemetry plus engine-relaunch tests. |
| M18 | Add the NVIDIA/ModelOpt arm to isolate source weights, quantizer recipe and runtime effects. |
| M19 | Use prose-first PPL, exact token/window alignment and explicit LM-head attribution experiments. |
| M20 | Treat imp's 5090 numbers as external context, never as an RTX 5080 release baseline. Publish ordinary decode with MTP off. |
| M23 | Freeze third-party notices, settled-evidence ledger, machine-readable perf baseline and lifecycle tests. |

## Auxiliary review slices

These do not add new milestones or branches; they are narrow support slices committed on `feat/gemma4-26b` and
attached to existing gates:

### R-IMP-00 — reference audit

Timing: after M00, before M01 closes.

Outputs:

- immutable imp lock;
- source map;
- MIT provenance decision;
- list of semantics and kernels to study;
- no production code copied.

### R-IMP-10 — semantic golden extraction

Timing: during M10, before CUDA MoE promotion.

Outputs:

- official-HF versus imp versus local-oracle comparison;
- router and residual fixtures;
- ModelOpt/llm-compressor scale fixtures.

### R-IMP-15 — optional grouped-kernel adoption

Timing: after M18 passes the preliminary quality gate, before M15 promotion.

Outputs:

- clean-room versus MIT-port decision;
- shape/layout compatibility report;
- microbenchmarks on the RTX 5080;
- proof of zero additional permanent expert copy.

## Immediate coding-agent instruction

Before changing model arithmetic, execute the task in [`references/imp/IMP_AGENT_TASK.md`](references/imp/IMP_AGENT_TASK.md). The result should update M01/M03/M10 evidence and may reject source assumptions. It must not start a general imp port.
