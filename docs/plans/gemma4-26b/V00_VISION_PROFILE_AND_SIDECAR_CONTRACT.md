# V00 — Gemma 4 26B Trellis35 Vision profile and sidecar contract

Status: owner-approved and implemented as the fail-closed input contract for V01.

## Ownership and profile boundary

Vision v1 is the explicit experimental profile
`gemma4_26b_trellis35_vision_fp8`. It requires the frozen compact text artifact
profile `gem16-trellis35-w4a8-v1` based on PFX31 commit `7649a84`.

It does not extend the qualified NVFP4 profile, `main`, or every Gemma 4 26B
checkpoint. The presence of `vision.gem16` never changes a runtime capability.
The future loader must require explicit profile selection and validate all four
files before it reports image support:

- `vision.gem16` — immutable Safetensors-compatible FP8/BF16 payload;
- `gem16_vision.json` — explicit runtime descriptor and text-profile binding;
- `vision_compilation.json` — complete source/compiler/tensor provenance;
- `vision.lock.json` — immutable hashes for the three files above and source.

Target, optional Assistant, and Vision remain independently pinned artifacts.
Text-only Trellis35 and qualified NVFP4 behavior remain unchanged.

## Immutable source and oracle

The only V01 weight source is Google's unquantized BF16 QAT repository
`google/gemma-4-26B-A4B-it-qat-q4_0-unquantized` at revision
`f1e06dc520982d9b9edd76859fdb7ab209449949`. Repository lock SHA-256 is
`3d9441fdebef33785e33181c335338208b8bf868cb8c7da692fd9c765cca8230`.
The config, processor config, and Safetensors index are separately checked
against the hashes in
`tools/gem16_compile/specs/gemma4-26b-vision-fp8-v1.json`.

The semantic oracle is Transformers 5.14.1 at commit
`a08ace4bbd97e721c98751deec37d87b026acadc`. The exact Gemma 4 modeling,
image-processing, processing, and configuration source hashes are pinned in
the same machine-readable specification.

## Reference execution order

The implementation must preserve this order from the pinned oracle:

1. Convert to RGB. Resize with aspect ratio preserved, bicubic interpolation
   and antialiasing. Both dimensions are multiples of `3 × 16`; the maximum is
   2,520 raw patches (`280 × 3²`). Rescale by `1/255`; do not image-normalize.
2. Patchify to row-major 16×16×3 vectors, generate `(x,y)` positions, then pad
   raw patches to 2,520 and positions with `(-1,-1)`.
3. In the model, transform patch values with `2 × (value - 0.5)`, apply the
   768→1,152 patch projection, and add the independent x and y BF16 position
   embeddings. Padding embeddings are zero.
4. Run 27 bidirectional Vision encoder layers. Each layer is
   input-RMSNorm → self-attention → post-attention-RMSNorm → residual, then
   pre-FFN-RMSNorm → GELU-tanh gate×up MLP → post-FFN-RMSNorm → residual.
5. Attention uses 16 Q heads and 16 KV heads of logical width 72. Q and K have
   learned RMSNorm; V uses scale-free RMSNorm. Two-dimensional RoPE treats x
   and y independently, splitting the head channels equally. Attention is
   non-causal and uses scaling 1.0 as in the pinned reference.
6. Zero padded states; spatially average each 3×3 group; scale in FP32 by
   `sqrt(1152)`; remove padded pooled tokens. The result has at most 280 image
   soft tokens.
7. Standardize in FP32 as `(hidden - std_bias) × std_scale`, then cast back to
   the working dtype.
8. Apply scale-free RMSNorm and the 1,152→2,816 Vision embedding projection
   before inserting the image states at image placeholder positions.

## V1 scope

- one image, batch size one;
- at most 280 resulting image soft tokens;
- text output only;
- no audio and no video;
- Ordinary decoding first;
- fixed-D2 is disabled until Ordinary multimodal correctness is accepted.

## Physical shape decisions

No loader-created padding or runtime weight repack is allowed.

- Attention head width 72 remains logical and physical width 72. V04 owns an
  explicit kernel tail rather than storing eight synthetic lanes.
- MLP Down contracting width remains logical and physical width 4,304. V05
  owns an explicit tail rather than silently changing it to 4,320 or 4,352.

The choice minimizes persistent bytes and makes any later padding experiment a
new, measured artifact-format revision instead of an implicit loader behavior.

## V01 payload contract

The source contains exactly 356 BF16 Vision tensors and 1,145,588,832 payload
bytes. V01 classifies them as:

- 191 two-dimensional linear weights / 549,070,848 parameters: E4M3FN weight
  bytes plus one BF16 scale per output row;
- 165 position, norm, and standardization tensors: byte-identical BF16 copy.

This produces 547 stored tensors and exactly 597,301,792 tensor bytes without
shape padding. Each tensor starts at a 256-byte device-arena-relative offset;
the resulting 11,232 explicit zero padding bytes bring the payload extent to
exactly 597,313,024 bytes, reproducing the review target. Container header
bytes are reported separately by each compilation and are not uploaded.

The file layout is final row-major NK FP8 plus row-major BF16 scale, or source
BF16 for copied tensors. Future CUDA kernels consume these layouts directly.
