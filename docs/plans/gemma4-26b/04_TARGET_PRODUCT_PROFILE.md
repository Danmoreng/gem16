# Target product profile — Fast Track R4

## Profile

Working ID:

```text
gem16-gemma4-26b-a4b-qat-hybrid-text
```

A later MTP-qualified profile may add `-mtp`; it must reference the same frozen target artifact plus a separately locked assistant artifact.

## Hardware and residency

- one NVIDIA Blackwell SM120-class GPU with approximately 16 GB VRAM;
- batch one and one resident 26B slot;
- no CPU weight offload, expert streaming or multi-GPU requirement;
- direct load of a checksum-locked compiled artifact.

## Base behavior

- text input/output and the pinned instruction template;
- resident multi-turn continuation;
- deterministic greedy and existing sampling controls;
- 32K default qualification gate;
- 64K target;
- an advertised maximum only after measured execution with at least 500 MiB free-device margin.

## Final behavior

- exact Target-verified MTP using a compatible assistant;
- ordinary/MTP output identity under matched deterministic controls;
- MTP at 32K as a hard minimum, 64K as the next target, and separate base/MTP maximum-context values;
- no proposal counted as output until Target verification commits it.

## Precision

| Family | Format |
|---|---|
| Experts/shared MLP | NVFP4 |
| Attention | FP8 |
| Router/norm/scalars | source BF16/F32 |
| Tied head | provisional NVFP4; one resident copy |
| KV | separate FP8 K and V |

An internal Q4_0 head/backend is not required for the first artifact.

## Release checkpoints

- M23 is a usable base text release checkpoint.
- M25 is the program-complete MTP profile.

## Unsupported on this program path

- vision, audio and video for the 26B profile;
- multiple 26B slots on a 16 GB card;
- continuous batching;
- CPU expert offload;
- on-the-fly inference-process quantization;
- unverified model or compiler artifacts.

Unsupported requests fail before execution with precise capability metadata.
