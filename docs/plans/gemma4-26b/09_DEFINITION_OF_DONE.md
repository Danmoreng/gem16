# Program definition of done — Fast Track R4

## Base target complete at M23

- [ ] One QAT-derived text-only artifact builds twice from clean inputs with identical hashes.
- [ ] One final device weight layout is resident; no runtime conversion/offload/streaming exists.
- [ ] One 26B slot passes 32K with at least 700 MiB free-device margin.
- [ ] 64K or the highest lower safe profile is measured and documented; larger supported contexts leave at least 500 MiB.
- [ ] CPU and CUDA MoE references, attention/KV tests and the complete model are accepted.
- [ ] The optimized path is deterministic and has no token-loop allocations.
- [ ] Held-out quality, performance and long-context reports reference one frozen artifact hash.
- [ ] CLI/server expose accurate profile, context and unsupported-feature metadata.
- [ ] 12B regressions remain green.
- [ ] Release evidence and rollback are hash-bound rather than rerun by default.

## Final program complete at M25

- [ ] A compatible 26B MTP assistant source/artifact is immutably locked.
- [ ] Assistant residency and verifier workspace pass direct-memory admission at 32K with at least 700 MiB free-device margin.
- [ ] Ordinary and MTP target outputs are identical under matched deterministic controls.
- [ ] Tentative KV/hidden/RNG state commits transactionally.
- [ ] MTP speed and acceptance are measured against ordinary Target execution.
- [ ] 64K is attempted and `mtp_max_context` is measured separately and advertised honestly; M25 does not pass below 32K.
- [ ] No 12B or base-26B regression is introduced.

Vision and an internal Q4_0 backend are not required for program completion.
