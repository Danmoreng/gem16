# Q4_0, NVFP4 and FP8 comparison

| Property | Q4_0 | NVFP4 | FP8 per-channel |
|---|---|---|---|
| Value bits | 4 | 4 | 8 |
| Value type | integer-like nibble around block scale | E2M1 float | E4M3 float |
| Scale group | 32 weights | 16 weights | output row/channel |
| Scale storage | FP16 | E4M3 plus global F32 | BF16 |
| Approx. bits/weight | 4.5 | 4.5 plus tiny global | ~8 plus tiny scale |
| Activations | usually BF16/FP16 or another runtime quant | dynamic NVFP4 W4A4 | dynamic FP8 W8A8 |
| Native Blackwell block-scaled NVFP4 MMA | no | yes | separate FP8 path |
| Google QAT target | yes | no direct guarantee | no direct guarantee |
| Best planned role | quality reference/head | experts/shared/head speed candidate | attention |

## Why Q4_0 can be smaller than an “NVFP4 model”

The format itself is not necessarily smaller. Both Q4_0 and group-16 NVFP4 use roughly 4.5 bits per quantized weight.

Checkpoint size differs because of which tensor families are quantized and whether vision/embedding/router/attention remain BF16/FP8.

## Why NVFP4 is still preferred for experts

The 22.8B routed-expert weights dominate memory and compute. Native block-scaled Tensor Core execution can accelerate both prefill and selected-expert decode while keeping them fully resident.

## Why Q4_0 remains valuable

- format Google trained toward;
- exact external quality reference;
- tied head is quality-sensitive;
- T=1 W4A16 can be bandwidth competitive;
- isolates whether native W4A4 activation quantization hurts logits.

## Why FP8 remains in attention

- existing gem16 native path;
- only about 1.06 GiB weight payload;
- Q/K/V quantization sensitivity;
- new 4-bit attention path would save less than expert/head changes while adding substantial risk.

## Invalid statement

Do not say:

```text
QAT BF16 converted to NVFP4 is an NVFP4-QAT model.
```

Correct:

```text
The master weights were obtained from Google's Q4_0-targeted QAT release
and were subsequently converted by gem16 to an NVFP4/FP8 hybrid. Quality
was evaluated empirically.
```
