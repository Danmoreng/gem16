# Executive architecture — Fast Track R4

## Runtime shape

The 26B target is a separate statically selected model variant. Hot 12B and 26B kernels remain independently specialized. Model dispatch occurs at initialization, not per token.

## Weight profile

- QAT BF16 is the sole mathematical source for the production target.
- Attention uses the accepted FP8 representation.
- Shared and routed MLP weights use NVFP4.
- Router, norms and scalar controls retain source BF16/F32 semantics.
- The first complete artifact uses one physically tied NVFP4 embedding/output matrix.
- Official Q4_0 and Unsloth remain external references; an internal Q4_0 backend is optional.

## Compiler

Python owns source verification, plans, publication and small independent fixtures. One native C++20 data plane owns promoted tensor conversion and large comparisons. The compiler streams source shards into final Safetensors output with bounded host memory.

## Loader and residency

The loader validates the complete artifact, allocates one immutable device arena and retains one runtime layout. It does not quantize at startup, offload experts or keep source-order and runtime-order copies. The first server profile has exactly one 26B execution slot.

## Correctness layers

1. CPU MoE semantic oracle.
2. Transparent CUDA MoE reference.
3. Attention/RoPE/KV reference tests.
4. Complete slow model.
5. Native decode, prefill and head operators.
6. Whole-model integration and graph capture.

## Context

32K is the first hard gate. 64K and the largest safe base context are measured after optimization. MTP has a separate memory plan and may have a lower maximum context than the base path.

## MTP

MTP is a final program goal and is isolated from base bring-up. It requires a compatible assistant artifact, exact Target verification and no independent long-context cache unless the assistant contract proves one necessary. Multi-row verifier head work belongs to M25.

## Product boundary

CLI and server are required. Studio is nonblocking. Vision is outside the 26B Fast Track.
