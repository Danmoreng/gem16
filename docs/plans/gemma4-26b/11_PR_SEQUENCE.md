# Suggested pull-request sequence

## Branch naming

Use:

```text
feat/26b-m00-policy
feat/26b-m01-source-locks
feat/26b-m02-model-traits
...
perf/26b-m14-moe-decode
perf/26b-m15-moe-prefill
```

Do not keep one long-lived mega-branch for the full program.

## PR boundaries

| PR | Core change | Must not include |
|---|---|---|
| M00 | Policy, decision, roadmap, feature flags | Compiler or CUDA code |
| M01 | Locks, fetch support, golden-capture harness | Runtime arithmetic |
| M02 | Model config/traits and tests | Tensor upload |
| M03 | Manifest/inventory and inspect JSON | Quantization |
| M04 | Compiler CLI/schema/scaffolding | Production quantizer tuning |
| M05 | FP8 encoder/oracle | NVFP4 kernels |
| M06 | NVFP4 encoder/oracle | Model runtime |
| M07 | Head-format experiment harness | Final product selection without evidence |
| M08 | Derived artifact schema and loader | Native MoE optimization |
| M09 | Residency and memory planner | Performance claims |
| M10 | CPU MoE semantics | CUDA optimization |
| M11 | CUDA reference MoE | Native Tensor Core promotion |
| M12 | Attention/KV 26B | MoE performance |
| M13 | Slow full-model integration and preliminary quality screen | Headline benchmarks |
| M18 | Quantizer comparison and quality kill-gate reports | Kernel modifications |
| M14 | Native decode expert path | Prefill changes |
| M15 | Grouped prefill expert path | Decode arithmetic changes |
| M16 | Embedding/head production path | Router/expert changes |
| M17 | Whole-model optimized scheduling | New quantization recipe |
| M19 | Quality qualification | Performance tuning |
| M20 | Performance qualification | Arithmetic changes |
| M21 | 32K/64K qualification | UI redesign |
| M22 | CLI/server/Studio integration | Core model arithmetic |
| M23 | Release docs, locks, evidence | New features |
| M24/M25 | Optional later work | Backport into already qualified base without rerun |

## Review order within each PR

1. contract and tests;
2. host implementation;
3. CUDA implementation if applicable;
4. memory accounting;
5. documentation;
6. evidence.

A reviewer should be able to reject a performance candidate without losing unrelated correctness work. Keep experimental kernels isolated until promoted.

## Commit guidance

Recommended commits for a kernel milestone:

```text
test: add 26b expert decode fixtures
cuda: add reference expert decode path
cuda: add native nvfp4 expert candidate
bench: add isolated expert benchmark
docs: record decode experiment and decision
```

Do not squash away raw evidence references in the final PR description.

## Auxiliary reference PRs

| PR | Timing | Core change | Must not include |
|---|---|---|---|
| R-IMP-00 | after M00, before M01 exit | imp lock, source map, license/provenance decision, semantic/kernel audit | Production code copy or architecture refactor |
| R-IMP-10 | within M10 | official-HF/imp/local-oracle semantic goldens and producer-scale fixtures | Optimized CUDA kernels |
| R-IMP-15 | after M18 passes, before M15 promotion | optional small-M kernel study or isolated MIT port | Paged KV, general executor, second permanent weight layout |
| R-IMP-23 | within M23 | third-party notice, settled ledger, lifecycle and perf-baseline release gates | New arithmetic |
