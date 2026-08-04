# gem16 Gemma 4 26B A4B implementation plan — repository revision 3

This package is a milestone-by-milestone execution plan for extending `gem16` from its current Gemma 4 12B Unified path to a text-only Gemma 4 26B A4B MoE path that fits and runs efficiently on one approximately 16 GB Blackwell GPU.

The intended primary artifact is derived from Google's unquantized QAT BF16 checkpoint. The performance path keeps the large expert matrices in native NVFP4, attention projections in FP8, routing and norms in BF16, KV in FP8, and selects the tied embedding/output-head format only after an isolated Q4_0-versus-NVFP4 experiment. Vision and MTP are explicitly deferred.

## Start here

For an immediate first Codex task, open [`START_HERE_CODEX.md`](START_HERE_CODEX.md).

Then read, in order:

1. [`INDEX.md`](INDEX.md)
2. [`00_MASTER_IMPLEMENTATION_PLAN.md`](00_MASTER_IMPLEMENTATION_PLAN.md)
3. [`02_AGENT_OPERATING_CONTRACT.md`](02_AGENT_OPERATING_CONTRACT.md)
4. [`04_TARGET_PRODUCT_PROFILE.md`](04_TARGET_PRODUCT_PROFILE.md)
5. [`07_MEMORY_BUDGET_AND_RESIDENCY.md`](07_MEMORY_BUDGET_AND_RESIDENCY.md)
6. [`06_DEPENDENCY_GRAPH.md`](06_DEPENDENCY_GRAPH.md)

Then execute exactly one approved file from [`milestones/`](milestones/).

## Repository baseline

This plan is anchored to `Danmoreng/gem16` commit:

```text
1c4287965d318ba32a68e597f9d7b6678b883376
```

The agent must produce a drift report before implementation if the working tree is based on a later revision.

## Non-negotiable result

A release candidate must:

- preserve the existing 12B path and its exact gates;
- keep every per-token weight resident on the GPU;
- allocate nothing in the token loop;
- expose every precision path and fallback in machine-readable output;
- fit a 32K text-only session on the reference 16 GB GPU with a measured safety margin;
- demonstrate deterministic output for deterministic settings;
- publish raw quality, memory and performance evidence;
- beat the selected Q4_0 baseline in both prefill and decode before being called a performance release.

The master plan intentionally separates compiler correctness, model quality, memory feasibility and speed. Passing one does not imply passing the others.


## Package structure

At generation time this package contains:

- 26 milestone plans (`M00`–`M25`);
- 28 normative technical specifications;
- 12 operational checklists;
- 13 reusable templates;
- 13 technical appendices;
- 8 focused imp-reference documents plus an index;

Directory-level indexes are available in each subdirectory. The regenerated manifest and checksums verify the repository-integrated plan contents.

## imp reference amendment

This plan includes a pinned `kekzl/imp` reference lane. It does not replace the QAT-BF16→FP8/NVFP4 strategy. It adds producer-specific scale tests, a ModelOpt quality-control arm, earlier router/residual goldens, actual-path telemetry, machine-readable performance gates, CUDA lifecycle tests and an optional MIT-compliant grouped-kernel study. Start with [`13_IMP_REFERENCE_INTEGRATION.md`](13_IMP_REFERENCE_INTEGRATION.md).

## Repository revision 3

After integration at the exact repository anchor, this revision:

- aligns M00 with the repository-wide allowance for reproducible project-compiled model artifacts;
- budgets against the approximately 15,881 MiB CUDA-visible capacity rather than 16,303 MiB nominal board memory;
- adds a preliminary M03 synthetic 32K/700 MiB admission gate before compiler implementation;
- adds source-storage and BF16-reference-memory preflight to M01;
- moves converter attribution and a preliminary quality kill gate ahead of native M14–M17 optimization;
- records the local RTX 5080 Q4_0 imp characterization without treating it as native-NVFP4 evidence.
