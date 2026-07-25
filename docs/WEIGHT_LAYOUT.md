# Weight layout

Packed NVFP4 values are consumed directly from the checkpoint representation. The inspector records source shape,
logical shape, byte offset, alignment, storage dtype, and scale relationships. The production device view changes
only local-scale byte order at load time.

Source matrices are packed row-major as two E2M1 values per byte and have one positive E4M3FN scale per 16
contracting elements. SM120 block-scaled MMA consumes K in 64-element steps, so each step pairs four source scale
bytes with its 64 E2M1 values. Gate/Up `[15360,3840]` and Down `[3840,15360]` require no logical padding for the
native geometry.

For an eight-output-row by 64-K tile, lane `l` owns source row `l / 4` and K quarter `l % 4`. Its two FP4 operand
registers are direct little-endian 32-bit loads for eight nibbles at K offsets `(l % 4) * 8` and
`32 + (l % 4) * 8`. Packed weights need no persistent copy and avoid the fourfold materialization of a naive
lane-fragment layout.

The qualified M128xN64 prefill CTA double-buffers current and next K64 activation slices for reuse by eight output
warps. Local weight-scale bytes use the sole runtime order:

```text
[row tile of 8][K block of 64][row within tile][4 source E4M3 scale bytes]
```

Thus the eight scale-vector words required by one output warp occupy one contiguous 32-byte region. The loader
recognizes tensors through the authoritative `NVFP4_LOCAL_SCALE_E4M3` manifest class, preserves every byte, and
copies one bounded transformed tensor directly into its final device-arena address. Maximum transient host staging
is 3,686,400 bytes; persistent device growth is zero and the 9,200,135,680-byte weight arena is unchanged.

Against `e17049b`, the final layout reduces context-512 NVFP4 Nsight time by 4.61% and total GPU-operation time by
2.22%. Prefill medians improve by 1.10%/1.43%/3.19% at 128/512/2,048 tokens. Removing strided scale addressing is
larger for decode: the complete Layer-0 MLP falls from 0.480 to 0.260 ms and a context-128 short decode rises from
25.54 to 31.63 tok/s with identical checksum.

The scale tiling is a load-time implementation detail rather than checkpoint conversion. It:

- preserves every source nibble and local-scale byte exactly;
- streams one bounded source region at a time into the final device allocation;
- retains neither a raw device copy nor parallel cuBLASLt/CUTLASS and custom-kernel copies;
- exposes deterministic byte counts, alignment, padding, and provenance in runtime metadata;
- passes source-to-layout logical mapping tests plus real checkpoint layer/model gates;
- includes transformation and bounded staging time in model load.

The source-scale ordering survives only in correctness and SIMT probes. Production has no selector or parallel scale
layout; all native decode and prefill kernels require the tiled final allocation.
