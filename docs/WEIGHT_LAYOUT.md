# Weight layout

The default policy is direct consumption of the packed source representation. The inspector currently records
source shape, logical shape, byte offset, alignment, storage dtype, and scale relationships.

The first architecture-specific candidate consumes the source layout directly as SM120a register fragments. The source
matrices are packed row-major as two E2M1 values per byte and have one positive E4M3FN scale per 16 contracting
elements. SM120 block-scaled MMA consumes K in 64-element steps, so each step pairs four source scale bytes with its
64 E2M1 values. Gate/Up `[15360,3840]` and Down `[3840,15360]` require no logical padding for the intended native
geometry.

For an eight-output-row by 64-K tile, lane `l` owns source row `l / 4` and K quarter `l % 4`. Its two FP4 operand
registers are direct little-endian 32-bit loads for eight nibbles at K offsets `(l % 4) * 8` and
`32 + (l % 4) * 8`. The four source E4M3FN bytes for that row and K block form the required scale-vector register.
This mapping needs no persistent weight copy and avoids the fourfold scale duplication present in a naive
lane-fragment materialization.

The qualified M128xN64 prefill CTA leaves that weight mapping unchanged. It stages only the current K64 slice of
the quantized activation and its activation scales for reuse by eight output warps; no weight or weight-scale
payload is transformed. Nsight measures a 51.2% NVFP4 projection-time reduction from this direct-layout CTA, so a
load-time weight/scale swizzle remains an unproven later experiment rather than part of the selected allocation.

If a measured kernel ultimately requires transformation, it remains a load-time implementation detail rather than
a checkpoint conversion. It must:

- preserve every source nibble and local-scale byte exactly;
- stream one bounded source region at a time into the final device allocation;
- retain neither a raw device copy nor parallel cuBLASLt/CUTLASS and custom-kernel copies;
- expose deterministic byte counts, alignment, padding, and provenance in the memory report;
- pass source-to-layout-to-logical round-trip tests using real checkpoint rows;
- include transformation time and peak staging memory in model-load results.

Direct source-layout SM120 MMA and direct packed SIMT/GEMV remain the required batch-one candidates. A transformed
layout is promoted only if its end-to-end projection result repays its memory and startup costs at the real shapes.
