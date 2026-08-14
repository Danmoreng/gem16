# Memory calculations

These are planning calculations. M03/M08/M09 replace estimates with exact manifest and allocator measurements.

All binary units use:

```text
1 MiB = 1,048,576 bytes
1 GiB = 1,073,741,824 bytes
```

## 1. Routed experts

Expected dimensions:

```text
layers = 30
experts per layer = 128
hidden = 2816
expert intermediate = 704
matrices = gate + up + down
```

Logical weights:

```text
per expert:
  gate = 704 × 2816
  up   = 704 × 2816
  down = 2816 × 704

  total = 3 × 2816 × 704
        = 5,947,392

all:
  30 × 128 × 5,947,392
  = 22,837,985,280 weights
```

NVFP4 storage with one E4M3 scale per 16 values:

```text
8 bytes packed E2M1 + 1 byte scale per 16 weights
= 9 / 16 byte per weight
= 0.5625 byte
```

```text
22,837,985,280 × 0.5625
= 12,846,366,720 bytes
= 12,251.25 MiB
= 11.964111328125 GiB
```

Tensor-global scalar overhead is negligible but must be counted exactly.

## 2. Always-active shared dense MLP

```text
intermediate = 2112
weights = 30 × 3 × 2816 × 2112
        = 535,265,280
storage = × 0.5625
        = 301,086,720 bytes
        = 287.138671875 MiB
```

## 3. Attention FP8 weights

Per sliding layer:

```text
Q = (16 × 256) × 2816 = 4096 × 2816
K = (8 × 256) × 2816 = 2048 × 2816
V = 2048 × 2816
O = 2816 × 4096

total = 34,603,008 bytes at 1 byte/weight
```

For 25 sliding layers:

```text
865,075,200 bytes
```

Per full layer:

```text
Q = (16 × 512) × 2816 = 8192 × 2816
K = (2 × 512) × 2816 = 1024 × 2816
V is omitted when raw K projection is reused
O = 2816 × 8192

total = 49,020,928 bytes
```

For 5 full layers:

```text
245,104,640 bytes
```

Total FP8 payload:

```text
1,110,179,840 bytes
= 1,058.75 MiB
```

Per-row BF16 scales must be added.

Expected number of scale rows:

```text
25 × (4096 + 2048 + 2048 + 2816)
+ 5 × (8192 + 1024 + 2816)
= 335,360 rows
```

At two bytes:

```text
670,720 bytes
```

This depends on confirmed tensor inventory.

## 4. Tied embedding/output

Logical weights:

```text
262,144 × 2,816
= 738,197,504
```

BF16:

```text
1,476,395,008 bytes
= 1,407.9990234375 MiB
```

Q4_0 or ideal group-16 NVFP4:

```text
738,197,504 × 0.5625
= 415,236,096 bytes
= 396 MiB exactly
```

Saving versus BF16:

```text
1,061,158,912 bytes
= 1,011.9990234375 MiB
```

## 5. Router projection

Assuming one BF16 `[128, 2816]` projection per layer:

```text
30 × 128 × 2816 × 2
= 21,626,880 bytes
= 20.625 MiB
```

Router hidden scale and per-expert scale add small BF16 tensors.

## 6. Core subtotal

Before attention channel scales, norms, layer scalars, router scales, cache-scale metadata, headers and alignment:

```text
routed experts     12,846,366,720
shared MLP            301,086,720
attention FP8        1,110,179,840
tied head              415,236,096
router projection       21,626,880
--------------------------------
core                 14,694,496,256 bytes
                     14,013.763671875 MiB
                     13.6853158474 GiB
```

With expected attention scales:

```text
14,695,166,976 bytes
= 14,014.4033203125 MiB
```

Small tensors and alignment determine whether the final arena remains at or below the 14,100 MiB target.

## 7. FP8 KV cache

Assuming separate final K and V.

### Local fixed ring

```text
25 layers
× min(context, 1024)
× 8 KV heads
× 256 dimensions
× 1 byte
× 2 (K and V)
```

At context ≥1024:

```text
104,857,600 bytes = 100 MiB
```

### Global

```text
5 layers
× context
× 2 KV heads
× 512 dimensions
× 1 byte
× 2
```

This is:

```text
10,240 × context bytes
```

| Context | Global | Local | Total |
|---:|---:|---:|---:|
| 8,192 | 80 MiB | 100 MiB | 180 MiB |
| 16,384 | 160 MiB | 100 MiB | 260 MiB |
| 32,768 | 320 MiB | 100 MiB | 420 MiB |
| 65,536 | 640 MiB | 100 MiB | 740 MiB |
| 131,072 | 1,280 MiB | 100 MiB | 1,380 MiB |
| 262,144 | 2,560 MiB | 100 MiB | 2,660 MiB |

Cross-layer KV sharing, if present, changes physical producer counts and must be incorporated after M03/M12.

## 8. Runtime-visible 16 GB card headroom

The board reports approximately 16,303 MiB nominally, while the current CUDA runtime exposes approximately
15,881 MiB for allocation/free-memory accounting. Use the lower measured value:

```text
15,881 - 14,100 = 1,781 MiB
```

At 32K:

```text
1,781 - 420 = 1,361 MiB
```

remaining before:

- CUDA context (currently approximately 216 MiB);
- prefill/decode workspace;
- graphs;
- output/sampling;
- allocator overhead;
- the required 700 MiB directly measured free margin.

At the 14,100 MiB weight target, roughly 445 MiB remains for those execution allocations after a 216 MiB context
and 700 MiB margin. At the 14,014.40 MiB planning estimate, roughly 531 MiB remains.

At 64K:

```text
1,781 - 740 = 1,041 MiB
```

Therefore 32K is extremely tight and needs an early synthetic admission probe; 64K is high risk even with strict
workspace control. Payload arithmetic is not qualification, and nominal board memory must not be used to infer the
safety margin.

## 9. Program thresholds

```text
immutable weights target     ≤ 14,100 MiB
review range                 14,101–14,300 MiB
hard stop                    > 14,300 MiB
32K measured peak margin     ≥ 700 MiB
base 64K measured peak margin ≥ 400 MiB if advertised; MTP remains ≥ 500 MiB
```
