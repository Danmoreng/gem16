# M23 — Release qualification, evidence freeze and rollback

## Objective

Freeze a reproducible release candidate, run the full regression/quality/performance matrix, publish evidence and prepare a tested rollback.

## Why this milestone exists

The 26B path touches loader, compiler, memory planning, CUDA kernels, server and UI. Release requires a single coherent evidence set and a way to disable the profile without destabilizing 12B.

## Prerequisites

- M22 complete
- All required M00–M22 exit gates closed

## Repository areas to inspect first

- `AGENTS.md`
- `docs/DECISIONS.md`
- `docs/ROADMAP.md`
- `docs/CORRECTNESS.md`
- `docs/BENCHMARKING.md`
- `docs/PERFORMANCE_LEDGER.md`
- `.github/workflows/`
- `scripts/package-studio.*`

## Suggested additions or boundaries

- `docs/releases/GEMMA4_26B_RC1.md`
- `benchmarks/releases/gemma4_26b_rc1/`
- `scripts/validate-gemma4-26b-release.*`

## Implementation sequence

1. Select the exact code commit, compiler commit, toolchain lock, source locks and compiled artifact lock.
2. Require a clean tree and rebuild host, CUDA, compiler and Studio artifacts from scratch.
3. Run complete host/CUDA tests, sanitizers, 12B regressions, 26B operator/full-model/product tests and platform-specific packaging tests.
4. Rerun the held-out quality suite and controlled performance suite on the exact release artifact.
5. Rerun 32K and any advertised 64K context gates with continuous telemetry.
6. Verify documentation, capability output, lock files and package manifests match the artifact.
7. Create release notes with supported hardware, context, features, exact quality/performance wording and known limitations.
8. Test rollback by disabling/removing the 26B profile and confirming 12B packages still build and run.
9. Freeze raw evidence read-only and publish checksums.
10. Do not merge or tag until every required gate has an owner sign-off.

## Required tests

- Clean clone bootstrap and artifact verification.
- Linux and Windows reference builds as supported by the project.
- Full CTest and compute-sanitizer matrix.
- 12B performance smoke to detect broad regression.
- 26B held-out quality, controlled performance, memory and long-context reruns.
- CLI/server/Studio packaging and upgrade/rollback.
- SHA-256 verification for every published report and model artifact.

## Evidence and documentation outputs

- `docs/releases/GEMMA4_26B_RC1.md`
- `benchmarks/releases/gemma4_26b_rc1/`
- `SHA256SUMS` for evidence and artifacts
- Release checklist with named sign-offs
- Rollback test log

## Suggested commands

```text
scripts/validate-gemma4-26b-release.sh --model-lock models/gemma4-26b-gem16-hybrid.lock.json --output benchmarks/releases/gemma4_26b_rc1
```
```text
git status --porcelain=v1
```
```text
python tools/verify_release_evidence.py --root benchmarks/releases/gemma4_26b_rc1 --checksums benchmarks/releases/gemma4_26b_rc1/SHA256SUMS
```

## Risks to watch in this milestone

- A compiler rebuild can produce a different artifact if any dependency or source lock is incomplete.
- Packaging can omit CUDA objects or model-profile metadata.
- Performance can drift with driver/toolchain updates between qualification and release.
- Rollback may leave persistent Studio settings pointing at an unavailable model.

## Forbidden shortcuts

- Tagging from a dirty tree.
- Reusing evidence from a different artifact hash.
- Publishing only best-case performance.
- Skipping 12B regressions.
- Calling 64K supported if only 32K is release-qualified.

## Exit criteria

- [ ] All release checklist items pass on exact frozen hashes.
- [ ] Quality, performance, memory and context evidence correspond to the release artifact.
- [ ] Packages and product surfaces are verified.
- [ ] 12B regressions and rollback pass.
- [ ] Known limitations and claims are accurate.
- [ ] Evidence checksums and sign-offs are frozen.

## Downstream milestones unblocked

- Release/tag decision
- Optional M24/M25 tracks

## Codex execution prompt

```text
You are implementing M23: Release qualification, evidence freeze and rollback in Danmoreng/gem16.

Read repository AGENTS.md, docs/DECISIONS.md, docs/CORRECTNESS.md, docs/BENCHMARKING.md, the package master plan, and this milestone. Work only on M23. Inspect the actual current tree before editing and write a drift note if it differs from the anchored commit.

Implement the narrowest change that satisfies every exit criterion. Add tests before or with implementation. Do not add silent fallbacks, token-loop allocations, CPU weight offload, or unrelated refactors. Preserve all existing Gemma 4 12B exact gates.

At completion, report files changed, tests and exact results, generated evidence paths, memory/performance deltas where relevant, unresolved risks, and whether each M23 exit criterion passed. Stop before starting the next milestone.
```

## imp reference amendment

Release evidence must include third-party notices for any imported imp code, the pinned source map, a settled-evidence ledger, machine-readable performance thresholds, actual-path/graph-demotion reports and repeated engine lifecycle results. A release cannot ship with an untracked copied kernel or a test that hides a poisoned CUDA context as a skip.
