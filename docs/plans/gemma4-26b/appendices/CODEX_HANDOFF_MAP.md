# Codex handoff map

## Program owner decisions

Owner approval required at:

- M00 derived-artifact policy;
- M01 source/license/distribution locks;
- M07 provisional head choice;
- M09 memory hard-stop exception;
- M18 threshold freeze;
- M19 final source/head choice;
- M21 64K support wording;
- M23 release.

## Agent-specialist handoffs

### Repository/model agent

M00–M04:

- contracts;
- locks;
- manifests;
- compiler framework.

Hands off:

- exact model traits;
- tensor map;
- goldens;
- accepted artifact schema.

### Quantization agent

M05–M08:

- codecs;
- compiler stages;
- reproducible artifact.

Hands off:

- final tensors;
- quantizer reports;
- runtime bindings.

### Correctness/runtime agent

M09–M13:

- arenas;
- CPU/CUDA reference;
- attention/KV;
- full model.

Hands off:

- accepted intermediate captures;
- reference execution path.

### Evaluation agent — preliminary gate

M18 after M13:

- causal converter/source/head A/B;
- development-corpus quality kill gate;
- frozen candidate profiles and held-out thresholds.

Hands off only passing candidates to native optimization.

### CUDA performance agent

M14–M17 after M18 passes:

- native MoE decode/prefill;
- head;
- graph integration.

Hands off:

- optimized deterministic runtime;
- profiles/disassembly/memory.

### Evaluation agent — final qualification

M19–M21:

- held-out quality;
- benchmarks;
- long context.

Hands off:

- final locked profile;
- release evidence.

### Product/release agent

M22–M23:

- CLI/server/Studio;
- packaging;
- release/rollback.

## Required handoff packet

Every handoff includes:

```text
code commit
model/artifact locks
completed exit criteria
test commands/results
evidence hashes/paths
known risks
assumptions not yet proven
next milestone prerequisites
```

A conversation summary is not a sufficient handoff.
