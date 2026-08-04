# Security and supply-chain checklist

## Source acquisition

- [ ] Full immutable revisions.
- [ ] Size and SHA-256.
- [ ] LFS/Xet identity where available.
- [ ] HTTPS and authenticated gated access.
- [ ] No `trust_remote_code`.
- [ ] No executable model repository code.
- [ ] Safe cache paths.

## Compiler

- [ ] Safetensors/JSON only.
- [ ] No pickle.
- [ ] Header/config size limits.
- [ ] Path traversal rejected.
- [ ] Symlink escape rejected.
- [ ] Atomic output.
- [ ] Incomplete files distinct.
- [ ] Dependency lock.
- [ ] Clean compiler commit.
- [ ] Secrets redacted from logs.

## Artifact

- [ ] Provenance metadata.
- [ ] Per-file and per-tensor hashes.
- [ ] License/terms.
- [ ] Omitted tensor disclosure.
- [ ] No executable files required.
- [ ] Verify before load.
- [ ] Corruption fails closed.

## Runtime/API

- [ ] Model ID cannot select arbitrary path through remote API.
- [ ] Unsupported media rejected before decode/allocation.
- [ ] Health/metrics do not expose tokens/private paths.
- [ ] Bounded request/config sizes.
- [ ] Resource exhaustion handled.
- [ ] No automatic unverified fallback download.

## Evidence

- [ ] No API/HF tokens.
- [ ] No private prompts unless approved.
- [ ] Environment redacted.
- [ ] Checksums after redaction.
- [ ] External storage access controlled.
