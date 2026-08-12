# Tied embedding/output-head specification — Fast Track R4

## Invariant

One logical `[262144, 2816]` matrix serves lookup and output projection. Exactly one physical payload may be resident.

## Base profile

The first complete artifact uses NVFP4. M07 supplies a reference lookup and T=1 projection; M16 optimizes or records that the reference is sufficient.

Required semantics:

- token bounds and model embedding scale;
- exact final norm/softcap boundary;
- suppression and existing sampling handoff;
- deterministic lowest-token tie rule;
- diagnostic full logits only when explicitly requested;
- no persistent row cache or second head representation.

## Deferred formats

Q4_0 is an external reference and optional M24 implementation. BF16 is diagnostic and not resident in the 16 GB production profile.

## MTP extension

T>1/multi-row Target verification is not required by M07 or M16. M25 adds the exact row count needed by selected proposal modes and validates ordinary/MTP identity. The MTP implementation must reuse the same physical tied matrix.
