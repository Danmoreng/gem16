# Weight layout

The engine accepts the checkpoint's row-major NVFP4 tensors directly, but does not retain that byte order on the
GPU. At model load it validates each `NVFP4_PACKED` tensor against its manifest `logical_shape`, preserves every
low-nibble-first E2M1 value exactly, and writes the bytes into the sole final device allocation in this order:

```text
[row tile of 8][K block of 64][row within tile][32 packed E2M1 bytes]
```

Its matching `NVFP4_LOCAL_SCALE_E4M3` tensor uses:

```text
[row tile of 8][K block of 64][row within tile][4 source E4M3 scale bytes]
```

Source matrices contain two E2M1 values per byte and one positive E4M3FN scale per 16 contracting elements.
Gate/Up `[15360,3840]` and Down `[3840,15360]` need no padding for the Row8/K64 geometry.

For an eight-output-row by K64 tile, lane `l` owns row `l / 4` and K quarter `l % 4`. Its two FP4 operand
registers are little-endian 32-bit loads at packed offsets `(l % 4) * 4` and `16 + (l % 4) * 4` within that row.
The eight row fragments required by a warp now occupy one contiguous 256-byte region instead of eight locations
separated by a full source row. The corresponding eight scale-vector words occupy one contiguous 32-byte region.
This is the sole persistent runtime layout used directly by decode and native Down prefill. Gate and Up prefill
derive one temporary CUTLASS operand view at a time in the preallocated prompt arena, reuse that scratch
immediately, and retain no second persistent layout.

The loader transforms bounded groups of row tiles into a reusable host staging vector of at most 4 MiB and copies
each group directly to its final weight-arena address. It never uploads a raw NVFP4 tensor and therefore never
holds both GPU layouts. Model-load timing includes the transformation and transfers. The persistent weight arena
remains 9,200,135,680 bytes, `persistent_repack_bytes` remains zero, and deleting any runtime state still leaves
the original Hugging Face checkpoint as the only persistent weight copy.

The layout contract is tested at three levels:

- host tests prove byte-exact mapping for multiple K blocks, multiple row tiles, and tail rows;
- CUDA tests compare tiled SM120 projections with source-layout reference/SIMT projections;
- checkpoint probes and full inference require the tiled binding and retain the fixed exact-blue output
  `[9503, 106]` with checkpoint FP8 KV.

On the Linux RTX 5080 Laptop characterization machine, the persistent Row8/K64 promotion improved the
8K-context/64-token decode median from 31.604 to 33.143 tok/s (+4.87%) under the same 1-warm-up/3-run policy.
The later temporary CUTLASS Gate/Up view improved the 8K prefill median from 2,135.93 to 2,584.77 tok/s (+21.0%)
and median TTFT from 3,835.33 to 3,169.34 ms (-17.4%). These are development characterizations, not qualified
cross-engine benchmark claims.

The source ordering survives only in CPU/CUDA reference and SIMT probes. Runtime JSON reports
`weight_layout=sm120_row8_k64`, `weight_scale_layout=sm120_row8_k64`,
`load_time_weight_swizzle=true`, `load_time_scale_swizzle=true`, and
`packed_weight_source_layout_direct=false`, and
`nvfp4_gate_up_prefill_weight_scratch=true`.
