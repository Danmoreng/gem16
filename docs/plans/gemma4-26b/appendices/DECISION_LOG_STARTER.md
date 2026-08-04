# Proposed decision-log entries

These are drafts to adapt into repository `docs/DECISIONS.md`.

## M00 — Freeze the reproducible derived-26B artifact contract

Decision: apply the repository's project-compiled-artifact policy to the 26B QAT profile through one locked,
repository-owned offline compiler and an explicitly project-built checkpoint.

Required conditions:

- immutable BF16 source lock;
- deterministic open compiler;
- Safetensors plus provenance;
- no runtime conversion;
- no driver-tied artifact;
- one final GPU layout;
- direct Unsloth baseline;
- complete quality/performance qualification.

## M07/M19 — Select tied head format

Decision options:

- Q4_0 head for QAT-target fidelity;
- NVFP4 head for native speed;
- reject candidate if neither passes.

Record isolated and full-model evidence.

## M19 — Select master source

Decision options:

- QAT BF16-derived hybrid;
- ordinary BF16-derived hybrid;
- direct Unsloth checkpoint;
- reject 26B release.

Do not pre-commit to QAT superiority.

## M21 — Default context

Decision:

- 32K default after measured margin;
- 64K qualified optional or experimental;
- reserve/admission values.

## M23 — Release claims

Freeze exact wording:

- hardware;
- weight profile;
- text-only;
- context;
- Q4_0 comparison;
- Unsloth/vLLM caveats;
- unsupported MTP/vision.
