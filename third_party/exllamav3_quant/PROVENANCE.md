# ExLlamaV3 quantizer provenance

Gem16's Trellis35 discovery ports selected format and quantizer concepts from:

- repository: `https://github.com/turboderp-org/exllamav3`
- revision: `0c49587a7c235e6303a6bbedc8b665272ad3a2ea`
- upstream license: MIT, copyright 2025 Turboderp
- pinned revision date: `2026-08-26T17:45:10+02:00`

The upstream project is not a Gem16 runtime or compiler dependency. Gem16-owned
ports are reviewed and tested independently, and the license in this directory
applies to portions derived from the upstream implementation.

The file `quant/codebook.cuh` is a byte-identical copy from the pinned
revision. `quant/quantize_tiles_kernel.cuh` began as the byte-identical source
identified below and carries one reviewed Gem16 patch: only lane 0 publishes
each warp's completed argmin reduction to shared memory. The upstream source
has every lane write the same four locations; Compute Sanitizer racecheck
reports those stores as write/write hazards. The guard preserves the intended
strict-comparison reduction order while removing undefined shared-memory
publication. The adjacent minimal `util.h`/`util.cuh` compatibility surface
and the standalone Gem16 producer CLI are Gem16-owned code; they intentionally
omit the upstream Torch-extension interface.

## Ported in WP2a/WP2b

`tools/gem16_compile/trellis35_quant.py` derives the following bounded
primitives from the pinned sources:

| Gem16 primitive | Pinned upstream source | SHA-256 |
|---|---|---|
| 16x16 tensor-core permutation | `exllamav3/modules/quant/exl3_lib/quantize.py` | `531eba086a296b29ae4c10aa6da455285f4ae43623eaf8ca1a02da1c8c59ca80` |
| K-bit trellis packing | `exllamav3/exllamav3_ext/quant/pack.cu` | `27606eed6650acc31c6b6484aad1e89195da88823a5bd62ffb3e9911a9b47e60` |
| procedural codebooks | `exllamav3/exllamav3_ext/quant/codebook.cuh` | `0e3c63b323f8d3cc15c6a8f2e2b3816efafd71e149a99997b30b2f806375138a` |
| tail-biting Viterbi recurrence | `exllamav3/exllamav3_ext/quant/quantize_tiles_kernel.cuh` | `85a9ab6295362212f3c6edc990cb6edb57c77a7b5473fe89b5109fdf57c28bfa` |
| 128-wide Hadamard reconstruction contract | `exllamav3/exllamav3_ext/quant/hadamard_inner.cuh` | `63ca981a4a706924b4ab87047a903bc51ed86d6abe713b8d8034d8812d3467f3` |
| two-word runtime trellis state extraction | `exllamav3/exllamav3_ext/quant/exl3_dq.cuh` | `7e3009b0f70635e8a98edf8b929832ca0fe16b6a5ad20f13b77ebfa72723feac` |

The CPU implementation additionally validates tail-biting state consistency
before packing. Upstream masks the low K bits because its encoder already
produces valid states; Gem16 treats compiler intermediates as untrusted and
fails closed instead.

The CPU Viterbi oracle intentionally preserves the pinned CUDA kernel's
observable first-pass ping-pong-buffer reduction behavior. The upstream
kernel writes step 255 to one buffer but performs its initial argmin over the
other (step-254) buffer. This affects the K3 golden fixture and is therefore
part of the reproducible v1 producer contract, irrespective of whether a
later producer revision elects to correct it together with an explicit
format/compiler version change.

WP9 ports only the bounded two-word bit-window calculation from upstream
`exl3_dq.cuh::dq` into Gem16's mixed-K3/K4 cb2 runtime decoder.  Gem16 retains
its own tensor-core coordinate mapping, E4M3 conversion, SM120 MMA, expert
dispatch, fused Gate+Up shape, and caller-owned workspace.  No upstream runtime
dependency or full EXL3 operator is introduced.

## Reviewed for later work, not yet ported

| Purpose | Pinned upstream source | SHA-256 |
|---|---|---|
| LDLQ, Hessian finalization, regularization | `exllamav3/modules/quant/exl3_lib/quantize.py` | `531eba086a296b29ae4c10aa6da455285f4ae43623eaf8ca1a02da1c8c59ca80` |
| fused reconstruction | `exllamav3/exllamav3_ext/quant/reconstruct.cu` | `f47f121cdb452cf881a0299f878f812519ab994b345eba67db83b751a2651f37` |

The exact upstream `LICENSE` SHA-256 at the pinned revision is
`27a32b6263fcd96c79d3beeecf221c4366780bdf15ad51986f48650bd7369bff`.
