# M00 — Policy and 26B track bootstrap

## Objective

Create the repository-level governance, feature-track skeleton and explicit artifact contract for a reproducible
project-built QAT-derived 26B checkpoint without weakening checkpoint provenance or benchmark integrity.

## Why this milestone exists

The repository now permits profile-specific compiled artifacts, but the concrete QAT-BF16→FP8/NVFP4 contract must
still be recorded before compiler work begins. Source identity, transformation provenance, output hashes, runtime
behavior and benchmark wording cannot be left to implementation convention.

## Prerequisites

- Working tree is clean or all unrelated changes are documented.
- Repository baseline `1c4287965d318ba32a68e597f9d7b6678b883376` is available for comparison.
- Project owner has reviewed the intended text-only 26B scope.

## Repository areas to inspect first

- `AGENTS.md`
- `docs/DECISIONS.md`
- `docs/ROADMAP.md`
- `docs/CHECKPOINT_FORMAT.md`
- `docs/BENCHMARKING.md`
- `docs/CORRECTNESS.md`
- `README.md`
- `CMakeLists.txt`

## Suggested additions or boundaries

- `docs/GEMMA4_26B.md`
- `docs/evidence/gemma4_26b/baseline-drift.md`

## Implementation sequence

1. Generate a drift report from the anchored commit to current HEAD, focusing on policy, model loading, quantization, memory, CUDA toolchain and benchmark changes.
2. Add a dated decision record defining the derived-checkpoint contract for Gemma 4 26B A4B QAT. State that compilation is offline, deterministic, source-locked, output-locked, auditable, and never performed by the inference server.
3. Confirm that `AGENTS.md` permits reproducible project-compiled profiles and that this plan supplies the stricter 26B-specific provenance and quality gates; direct Unsloth NVFP4 and official Q4_0 remain mandatory baselines.
4. Define terminology: source checkpoint, compiled checkpoint, runtime layout, production profile, diagnostic profile and baseline.
5. Add an initial `docs/GEMMA4_26B.md` or equivalent track page that links the master goals and explicitly excludes vision and MTP.
6. Add build-time feature boundaries or placeholders only if needed to prevent incomplete 26B code from becoming default. Prefer a compile option such as `GEM16_ENABLE_GEMMA4_26B_EXPERIMENTAL` until M23.
7. Create issue/task labels or a tracking checklist matching M00–M23 if the repository process uses issues.
8. Record the reference hardware class and the initial immutable-weight target of ≤14,100 MiB.
9. Confirm that existing 12B releases and model locks remain the default and unchanged.

## Required tests

- Documentation-link check for every new relative link.
- Host build and existing unit tests, even though this milestone should contain no arithmetic code.
- A repository grep proving no model compiler or 26B runtime path was accidentally added.

## Evidence and documentation outputs

- `docs/DECISIONS.md` entry with date, context, alternatives, consequences and evidence.
- Repository drift report under a dated 26B planning/evidence path.
- Updated roadmap showing milestones without claiming implementation.
- A short policy review checklist signed off in the PR discussion.

## Suggested commands

```text
git diff --stat 1c4287965d318ba32a68e597f9d7b6678b883376..HEAD
```
```text
./scripts/build.sh --cuda --test
```
```text
.\scripts\build.ps1 -Cuda -Test
```

## Risks to watch in this milestone

- A vague artifact contract could permit irreproducible conversions or misleading provenance claims.
- A feature flag that defaults on could expose incomplete model support.

## Forbidden shortcuts

- Do not sneak compiler code into this PR.
- Do not weaken source locking for the existing 12B model.
- Do not state that QAT→NVFP4 quality is known.
- Do not call the eventual artifact official Google NVFP4.

## Exit criteria

- [ ] Accepted decision explicitly defines the QAT-derived artifact contract.
- [ ] AGENTS rules still prohibit untracked conversions and silent fallback.
- [ ] Vision, MTP, CPU offload and continuous batching are explicit non-goals.
- [ ] No runtime/compiler implementation is included.
- [ ] All existing tests pass.
- [ ] M01 is marked ready and every later milestone remains blocked.

## Downstream milestones unblocked

- M01

## Codex execution prompt

```text
You are implementing M00: Policy and 26B track bootstrap in Danmoreng/gem16.

Read repository AGENTS.md, docs/DECISIONS.md, docs/CORRECTNESS.md, docs/BENCHMARKING.md, the package master plan, and this milestone. Work only on M00. Inspect the actual current tree before editing and write a drift note if it differs from the anchored commit.

Implement the narrowest change that satisfies every exit criterion. Add tests before or with implementation. Do not add silent fallbacks, token-loop allocations, CPU weight offload, or unrelated refactors. Preserve all existing Gemma 4 12B exact gates.

At completion, report files changed, tests and exact results, generated evidence paths, memory/performance deltas where relevant, unresolved risks, and whether each M00 exit criterion passed. Stop before starting the next milestone.
```

## imp reference amendment

The policy decision must also classify external implementation use: reference-only, clean-room reimplementation or copied MIT code. Approve the structure in `references/imp/IMP_LICENSE_AND_PROVENANCE.md` before any source is copied.
