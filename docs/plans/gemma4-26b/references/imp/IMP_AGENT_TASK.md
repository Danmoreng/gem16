# Codex task — pinned imp reference audit

## Objective

Audit the pinned `kekzl/imp` implementation as a reference for the existing Gemma 4 26B plan. Produce source-anchored findings and fixtures; do not port the general engine.

## Inputs

```text
gem16 anchor: 1c4287965d318ba32a68e597f9d7b6678b883376
imp anchor: a392904d4216388828d0d56317de046f4ca49627
```

Read:

- repository `AGENTS.md` and accepted decisions;
- [`../../13_IMP_REFERENCE_INTEGRATION.md`](../../13_IMP_REFERENCE_INTEGRATION.md);
- every document in this directory;
- M01, M03, M06, M10, M14, M15, M18, M19 and M20.

## Work

1. Clone/check out the exact imp commit outside production source.
2. Record SHA-256 for every file in `IMP_SOURCE_MAP.md` that exists.
3. Verify MIT license text and identify any selected file with another license/header.
4. Produce a table of Gemma 4 tensor names and quantization metadata found in imp.
5. Extract exact ModelOpt and llm-compressor scale formulas and create local test vectors.
6. Trace the Gemma 4 router and FFN branch order; map each operation to the official reference.
7. Identify every point where imp uses FP32 specifically to avoid expert-selection/residual drift.
8. Inspect actual-path dispatch recording and graph-demotion reporting.
9. Inspect the grouped small-M kernel interfaces and layout requirements.
10. Reproduce the quality-audit commands only when the exact checkpoint/corpus is available; otherwise record the evidence as external and do not fabricate results.
11. Update the local source-lock, golden and risk documents.
12. Record confirmed, refuted and unresolved findings in a settled-evidence ledger.

## Forbidden work

- no Paged KV port;
- no continuous batching;
- no wholesale executor or weight-registry import;
- no copied source without owner-approved license decision;
- no performance claim on RTX 5080 from 5090 data;
- no change to gem16 arithmetic in this task.

## Deliverables

```text
docs/gemma4_26b/reference/imp_source_lock.json
docs/gemma4_26b/reference/imp_source_map.md
docs/gemma4_26b/reference/imp_semantic_diff.md
docs/gemma4_26b/reference/imp_quant_scale_fixtures.json
docs/gemma4_26b/reference/imp_kernel_layout_report.md
docs/gemma4_26b/SETTLED.md
```

Report exact commands, file hashes, findings and unresolved assumptions.
