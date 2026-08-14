# Memory gate checklist

## Manifest

- [ ] All resident tensors classified.
- [ ] Modality/MTP tensors omitted.
- [ ] Tied head one allocation.
- [ ] Scales counted.
- [ ] Alignment/padding counted.
- [ ] No unknown tensor.
- [ ] No persistent duplicate runtime layout.

## Plan

- [ ] Immutable weights by family.
- [ ] Local K+V formula.
- [ ] Global K+V formula.
- [ ] Shared-KV ownership if present.
- [ ] Prefill workspace by named region.
- [ ] Decode workspace.
- [ ] Sampling.
- [ ] Graph-private.
- [ ] CUDA context/allocator reserve.
- [ ] Checked overflow.

## Measurement

- [ ] `cudaMemGetInfo` before/after.
- [ ] Named allocator total.
- [ ] Process VRAM continuous telemetry.
- [ ] Peak host RSS.
- [ ] Model load transient.
- [ ] 32K full execution.
- [ ] 64K if claimed.
- [ ] Multiple slots if claimed.
- [ ] Margin at peak.

## Policy

- [ ] Weight target ≤14,100 MiB or decision.
- [ ] Hard stop >14,300 MiB.
- [ ] 32K margin ≥700 MiB.
- [ ] Base-model 64K margin ≥400 MiB if qualified; MTP 64K margin remains ≥500 MiB.
- [ ] No CPU offload.
- [ ] No silent context reduction.
- [ ] Admission fails before unsafe operation.
