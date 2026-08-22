# M12 — 26B attention, RoPE and KV

Status: accepted 2026-08-22 at implementation commit `bbee9cd930133dd49cb3acc79b4867658a0968cc`
Class: parallel-critical
Unblocks: M13

Normative inputs: [Attention/KV](../specs/ATTENTION_KV_SPEC.md), [Model variant traits](../specs/MODEL_VARIANT_TRAITS_SPEC.md).

## Phase A — start now

- validated 30-layer local/global trait table;
- Q/K/V/O shape fixtures and missing-V semantics;
- RoPE fixtures including 1023/1024 and long positions;
- local ring and global extent ownership tests;
- exact cache-byte formulas.

## Phase B — after M08/M09

- bind real FP8 weights and separate K/V caches;
- implement local/global reference execution;
- validate producer/consumer ownership and any cross-layer sharing;
- reconcile actual bytes with M09.

## Exit gate

- [x] All layer traits and tensor bindings are validated.
- [x] Local/global reference comparisons pass.
- [x] Final K and V are physically distinct where required.
- [x] Ring wrap and global append/read pass.
- [x] Cache bytes match the memory planner at 8K, 32K and 64K.
- [x] 12B attention/long-context tests remain green.

## Implementation evidence (2026-08-22)

The implementation constructs an immutable 30-entry trait table only after the
exact 26B config contract passes. The table is copied from validated
`layer_types`, not inferred by layer-number modulo, and names cache ownership,
V-projection ownership, head geometry, RoPE parameters and capacity. The same
table now owns the M09 FP8 cache arithmetic: 188,743,680 bytes at 8K,
440,401,920 at 32K and 775,946,240 at 64K. The former duplicate residency
formula was removed.

`cuda/attention/gemma4_26b_reference` is an isolated correctness-only one-token
path over the immutable M08 arena. It applies dynamic per-token FP8 activation
quantization and per-output-channel FP8 Q/K/V/O projections with FP32
accumulation and BF16 boundaries. Local layers use separate V projections,
D256 full RoPE and a 1,024-row circular cache. Global layers omit V, copy raw K
into a distinct V workspace, then apply learned K norm plus proportional D512
RoPE to K and scale-free norm without RoPE to V. Attention reads prior cache
plus staged current K/V and commits current state afterward. Recurring forward
does no allocation, filesystem access, host routing or repacking.

The ignored real fixture exercises locked M08 layers 0 and 5 at position 0
against the accepted BF16 capture. Across all named projection, norm, attention
and post-attention boundaries, worst relative-L2 is 0.056467 and worst cosine is
0.998529 under fixed 0.07/0.998 gates. Both real cases repeat bitwise identically
with no CUDA-visible allocation delta. The focused CUDA test additionally
covers the 1023-to-1024 local ring wrap, global append/read, missing-V binding,
physically distinct K/V, exact cache addresses and invalid alias rejection.
Compute Sanitizer memcheck, racecheck and initcheck pass. Compact hashes and
metrics are in `artifacts/m12/diagnostic-summary.json`; no performance claim is
made. The clean commit-bound record is `artifacts/m12/acceptance.json`.
