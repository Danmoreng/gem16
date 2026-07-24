# Memory

The deterministic base-arena planner is implemented. The verified checkpoint contains 9,304,786,336 tensor payload bytes. The explicit
text-only selection retains 9,200,026,528 bytes and skips 104,759,808 bytes of audio/vision projection and
embedding tensors. The planner separates 8,668,020,512 bytes of weights/model state from 532,006,016 bytes of
scales and aligns every named region to 256 bytes. These remain planned payload bytes, not measured CUDA allocations.

For the parsed 48-layer architecture and one-byte FP8 cache, the formula after the local window is full is:

```text
local one-state lower bound = 40 * min(tokens, 1024) * 8 * 256
global one-state lower bound = 8 * tokens * 1 * 512
required separate K and V   = 2 * one-state lower bound
```

At 64K, the one-state lower bound is 336 MiB and the required separate K/V payload is 672 MiB. Although
`attention_k_eq_v=true` reuses the raw full-attention K projection for V, learned K normalization plus RoPE and
scale-free V normalization produce distinct final cache states. Shared physical storage is therefore rejected.
These are formulas, not allocator measurements. Metadata, scale storage, alignment, CUDA context, workspaces, and
graph pools must be added to future measured reports.

The one-byte FP8 payload plans are:

| Profile | Context | One-state lower bound | Required separate K/V | Invalid shared arena | Selected separate arena |
|---|---:|---:|---:|---:|---:|
| `interactive` | 8,192 | 112 MiB | 224 MiB | 8,885.83 MiB | 8,997.83 MiB |
| `standard` | 32,768 | 208 MiB | 416 MiB | 8,981.83 MiB | 9,189.83 MiB |
| `long` | 65,536 | 336 MiB | 672 MiB | 9,109.83 MiB | 9,445.83 MiB |
| `xlong` | 131,072 | 592 MiB | 1,184 MiB | 9,365.83 MiB | 9,957.83 MiB |
| `max` | 262,144 | 1,104 MiB | 2,208 MiB | 9,877.83 MiB | 10,981.83 MiB |

Every plan reports both byte formulas for auditability, requires an explicit layout, and accepts only `separate`.
Checked multiplication, addition, and alignment reject integer overflow. Activation A/B, logits, sampling, CUDA
Graph, kernel, and prefill workspaces remain explicitly unplanned until their execution shapes are defined;
`total_arena_bytes` is therefore the known base arena, not a peak-VRAM claim.

The current full-model characterization separately measures a 9,200,135,680-byte aligned device weight arena and
a roughly 1.47 MB reusable workspace. The runtime now applies the planned hybrid layout: 40 local layers allocate
at most 1,024 physical slots and reuse them as chronological rings, while eight global layers allocate the requested
context extent. Separate K/V storage is retained for both. Optional
full-logit diagnostics use host memory (`steps * 262144 * 4` bytes) allocated before generation and do not change
persistent device storage.

A real FP8 allocation at a 1,026-position execution plan measured 176,177,152 cache bytes: 167,772,160 bytes for the
40 local K/V rings plus 8,404,992 bytes for eight global K/V extents. This exactly matches the formula above and
crosses the first local-ring wrap in full-model execution.

The native prefill arena is allocated once during engine initialization. It holds 32-token hidden/projection/MLP
tiles and a causal score matrix sized as `32 * 16 * planned_context * sizeof(float)`. Thus the dominant score region
is 128 MiB at 65,536 positions and 512 MiB at the 262,144-position maximum; it remains below the initial 1 GiB
activation-arena target and performs no growth inside prompt processing.
