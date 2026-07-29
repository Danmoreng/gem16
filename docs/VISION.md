# Vision input

Status: implemented for cross-platform one-shot chat; Windows GPU path qualified

Gemma 4 Unified does not contain a separate transformer-style vision tower.
Its complete encoder-free vision embedder is:

```text
RGB image
-> aspect-ratio-preserving bicubic resize
-> 48 x 48 x 3 merged patches
-> BF16 LayerNorm(6912)
-> BF16 Dense(6912 -> 3840) + bias
-> BF16 LayerNorm(3840)
-> factorized X/Y position embeddings
-> BF16 LayerNorm(3840)
-> scale-free RMSNorm(epsilon=1e-6)
-> BF16 Dense(3840 -> 3840, no bias)
-> image soft-token rows
```

All ten vision tensors are bound by manifest name and validated for BF16 dtype
and exact checkpoint geometry. Together they occupy 99,844,608 source bytes and
are always resident with the text and audio tensors.

## CLI

```powershell
.\build\Windows\blackwell-release\bin\gem16-chat.exe `
  --model .\models\checkpoints\unsloth-gemma-4-12b-it-NVFP4-b1f6497 `
  --message "Beschreibe das Bild." `
  --image C:\path\image.png `
  --no-thinking --max-context 1024 --max-tokens 128
```

PNG, JPEG, and BMP are decoded by the pinned stb_image single-header library
into 24-bit RGB on Windows and Linux. The checkpoint permits at most 280 soft
tokens per image; 1120 is the position-table extent, not a larger qualified
per-image tier. The automatic policy reserves output, text/boundary, and audio
capacity, divides the remaining context across all images, caps each at 280,
and never upscales a small source merely to fill the budget. The processor applies the selected
soft-token budget, 16-pixel teacher
patch, 3x3 merge, bicubic antialias resize, `1/255` rescale, no mean/std
normalization, row-major patch order, and `(x,y)` position IDs. Empty, oversized,
malformed, and unsupported images fail before GPU execution.

`--stats` reports source size, processed size, actual soft-token count, and the
automatic per-image budget. In the context-512 qualification, two large diagrams
were processed as 192/206 and 190/206 tokens and completed without truncation.

The prompt renderer emits `<|image>`, one `<|image|>` placeholder per valid
patch, and `<image|>`. Projected rows replace only those validated placeholders.
Each image block receives its own unsplit prefill chunk, so repeated images cannot
share or confuse the qualified bidirectional image-mask range.

## Attention semantics

Only queries inside the image placeholder span may see later keys in that same
span, and only in the 40 sliding-attention layers. The left 1,024-token window
still applies. Text, audio, delimiters, and all eight global-attention layers
remain causal. Generated-token decode is unchanged and causal.

Vision currently requires checkpoint-FP8 KV mode because the optimized local
prefill kernel owns this qualified mask. BF16 correctness-mode KV rejects image
input visibly instead of falling back to a causal approximation.

## Verification

The qualification includes:

- full SM120a release build;
- cross-platform host image fixture checking 256 patches, RGB scaling, `(x,y)`
  order, and boundary positions;
- CUDA mask fixture proving pre-image and post-image text remain causal while an
  early image query changes when future same-image keys become visible;
- complete host and CUDA regression suites;
- real-checkpoint greedy chat on `natural_scene.png`, correctly describing the
  house, mountains, sun, visible heading, and reading the blue sign as `24`;
- the same sign-reading request under MTP D2, also returning `24`.

The first projector is deliberately correctness-oriented. Its two BF16 dense
operators use direct source weights and FP32 accumulation but are not yet
Tensor-Core tiled; profiling and replacement with shape-specific BF16 MMA are
the next vision performance step.

## Current limits

- repeated images and audio are supported in one-shot and resident messages;
- resident `/image`, `/audio`, `/media`, and `/clear-media` commands manage the
  next message's ordered media queue without reloading model weights;
- PNG, JPEG, and BMP input through the portable stb_image decoder;
- maximum 280 valid image soft tokens;
- automatic context-aware budgets from 1 through 280, with no default upscaling;
- input-only vision and text output;
- no video-frame adapter yet.
