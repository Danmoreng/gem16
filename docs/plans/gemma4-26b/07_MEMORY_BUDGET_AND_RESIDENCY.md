# Memory budget and residency — Fast Track R4

## Fixed model facts

```text
layers 30; local/global 25/5
hidden 2816; shared intermediate 2112
128 routed experts; top 8; expert intermediate 704
vocabulary 262144
```

The accepted planning estimate before small tensors/alignment is approximately 14,014.40 MiB. Existing M03 evidence records the exact candidate arenas and the synthetic 32K result.

## Base FP8 KV payload

Separate K and V are mandatory.

| Context | Local K/V | Global K/V | Total payload |
|---:|---:|---:|---:|
| 8K | 100 MiB | 80 MiB | 180 MiB |
| 16K | 100 MiB | 160 MiB | 260 MiB |
| 32K | 100 MiB | 320 MiB | 420 MiB |
| 64K | 100 MiB | 640 MiB | 740 MiB |
| 128K | 100 MiB | 1,280 MiB | 1,380 MiB |

These are payloads, not process peaks.

## Admission gates

Use direct CUDA-visible memory accounting.

- 32K: at least 700 MiB free after weights, slot, graphs and warm generation.
- Base-model 64K and larger advertised profiles: at least 400 MiB free after the same process. MTP keeps a separate
  500 MiB requirement.
- `max-single-user`: highest measured base-model context satisfying the 400 MiB rule and correctness tests.
- one positive 26B slot; a second slot is an expected rejection case.

M09 does not implement `/health`, `/metrics` or multi-slot scaling. It exposes internal named counters and a CLI/report form sufficient for M22 to publish later.

## Parallel M09 phases

- Phase A may begin from M03: checked formulas, alignment, overflow, one-slot and second-slot-rejection tests.
- Phase B begins after M08: real artifact load, allocation reconciliation and final 32K gate.

## Workspace policy

Allowed workspace is fixed, named and bounded by the selected prefill chunk. Forbidden: all-expert outputs, prompt×expert activations, dequantized expert copies, persistent source/runtime duplicates, per-layer or token-loop allocation.

Measure 256/512/1024 prompt chunks. 2048 is considered only after real 64K headroom exists.

## MTP memory planning

MTP has a separate plan:

```text
base target weights
+ assistant weights
+ target KV
+ assistant/verifier workspace
+ graph-private memory
+ safety margin
```

Do not assume a BF16 assistant fits. M25 phase A must inventory the compatible assistant and calculate BF16, FP8 and NVFP4 candidate sizes. Prefer an assistant contract that reads Target KV and does not allocate another long-context cache. MTP must pass 32K with the 700 MiB margin, then attempt 64K; publish separate `base_max_context` and `mtp_max_context`.

## Regression thresholds

Explain any change above:

- 1 MiB immutable weights;
- 8 MiB reusable workspace;
- 1 MiB graph-private bytes;
- 32 MiB at the 32K process peak.
