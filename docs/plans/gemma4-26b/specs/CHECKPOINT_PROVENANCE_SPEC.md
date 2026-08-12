# Checkpoint provenance specification — Fast Track R4

## Source lock

Record repository, full immutable revision, every required file, byte size, SHA-256, tokenizer/config identity and applicable license/distribution notes. Mutable revisions are rejected.

## Compilation record

The final metadata binds:

- source-lock hash;
- repository/compiler commit;
- native protocol and executable hash;
- host compiler/toolchain and explicit thread count;
- profile/format identifiers;
- exact input and output tensor mappings;
- quantizer parameters and dequantization equations;
- omitted families and reasons;
- file/tensor hashes;
- complete command/config identity.

The base profile records `embedding_head=nvfp4` unless a later accepted decision changes it. Vision/audio/video/MTP tensors are omitted from the base target; M25 uses a separate assistant artifact and lock.

## Milestone reproducibility

- M05 accepted policy remains historical.
- M06: one clean full QAT expert conversion plus exhaustive/native deterministic fixtures.
- M07: one clean QAT tied-head conversion plus fixtures.
- M08: two clean complete builds with identical hashes and one external derived-artifact lock.
- M18: complete ordinary/alternative conversions only when diagnosis/attribution is triggered.
- M25: target and assistant locks remain separate and are jointly referenced by the MTP profile.

Do not create a duplicate full partial artifact solely to claim reproducibility.

## Evidence binding

Every quality, performance, context and release report carries the final artifact-lock SHA-256, binary hash and configuration identity. MTP reports additionally carry the assistant-lock SHA-256.

## Rerun rules

| Changed item | Minimum invalidated evidence |
|---|---|
| tokenizer/template | tokenization/chat/generation quality |
| source or quantizer | compiler, operator, model, quality, memory and performance |
| CUDA arithmetic/schedule | affected operator, full model, allocation and performance |
| driver/toolkit/CUTLASS | dispatch, sanitizer, memory and performance |
| hosting metadata only | lock/download/product tests |

M23/M25 verify already valid exact-hash evidence and rerun only invalidated work.
