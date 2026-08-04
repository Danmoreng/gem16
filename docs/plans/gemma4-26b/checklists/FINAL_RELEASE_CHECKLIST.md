# Final release checklist

## Frozen identity

- [ ] Code commit.
- [ ] Clean tree.
- [ ] Toolchain lock.
- [ ] Source locks.
- [ ] Compiler lock/commit.
- [ ] Final artifact lock.
- [ ] Quality suite hash.
- [ ] Benchmark prompt hash.
- [ ] Machine IDs.

## Builds

- [ ] Clean host build.
- [ ] Clean Linux CUDA build.
- [ ] Clean Windows CUDA build where release-supported.
- [ ] Studio/package builds.
- [ ] Native SASS verification.

## Correctness

- [ ] Full host tests.
- [ ] Full CUDA tests.
- [ ] Sanitizers.
- [ ] 12B regressions.
- [ ] 26B operator/full-model tests.
- [ ] Determinism.
- [ ] No fallback/allocation/offload.

## Quality

- [ ] Held-out suite rerun on exact artifact.
- [ ] Thresholds pass.
- [ ] Worst regressions reviewed.
- [ ] Head/source decision matches artifact.
- [ ] Long-context quality.

## Performance/memory

- [ ] Controlled 3/10 suite.
- [ ] Q4_0 baseline win in required prefill/decode.
- [ ] Memory margins.
- [ ] 32K qualification.
- [ ] 64K wording matches qualification.
- [ ] Telemetry/raw evidence/checksums.

## Product

- [ ] Download/resume/verify.
- [ ] CLI.
- [ ] Server/SDK/streaming/tools.
- [ ] Studio.
- [ ] Unsupported media/MTP errors.
- [ ] Context admission.
- [ ] User docs.

## Release governance

- [ ] Decision/roadmap/correctness/memory/benchmark docs updated.
- [ ] Release notes.
- [ ] License/attribution.
- [ ] Rollback tested.
- [ ] Evidence read-only/frozen.
- [ ] Named owner sign-offs.
