# Program definition of done — Fast Track R4

## Technical base Target complete at M23

- [ ] One QAT-derived text-only artifact builds twice from clean inputs with identical hashes.
- [ ] One final device weight layout is resident; no runtime conversion/offload/streaming exists.
- [ ] One 26B slot passes 32K with at least 700 MiB free-device margin.
- [ ] Base-model 64K is measured and documented; larger supported base contexts leave at least 400 MiB. MTP keeps its
  separate 500 MiB gate.
- [ ] CPU and CUDA MoE references, attention/KV tests and the complete model are accepted.
- [ ] The optimized path is deterministic and has no token-loop allocations.
- [ ] Performance, long-context and product reports reference one frozen artifact hash.
- [ ] CLI/server expose accurate profile, context and unsupported-feature metadata.
- [ ] 12B regressions remain green.
- [ ] Engineering evidence and rollback are hash-bound rather than rerun by default.
- [ ] Deferred M19 and the prohibition on shipping/production-quality claims are explicit.

## Final technical target complete at M25

- [ ] A compatible 26B MTP assistant source/artifact is immutably locked.
- [ ] Assistant residency and verifier workspace pass direct-memory admission at 32K with at least 700 MiB free-device margin.
- [ ] Ordinary and MTP target outputs are identical under matched deterministic controls.
- [ ] Tentative KV/hidden/RNG state commits transactionally.
- [ ] MTP speed and acceptance are measured against ordinary Target execution.
- [ ] 64K is attempted and `mtp_max_context` is measured separately and advertised honestly; M25 does not pass below 32K.
- [ ] No 12B or base-26B regression is introduced.

## Release quality complete after deferred M19

- [ ] The owner-deferred task/prose held-out suite passes on the frozen Target.
- [ ] Only then may the project claim shipping, production-quality or program-complete status.

Vision and an internal Q4_0 backend are not required for program completion.
