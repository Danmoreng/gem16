# Gemma 4 26B benchmark matrix

## Engines and candidates

| ID | Engine/artifact | Purpose |
|---|---|---|
| G16-QAT | gem16 final QAT-derived hybrid | production candidate |
| G16-BASE | gem16 ordinary-BF16-derived hybrid | quantizer control |
| UNSLOTH | verified direct Unsloth mixed FP8/NVFP4 runtime | practical NVFP4 baseline |
| GOOGLE-Q4 | official Google Q4_0 in pinned llama.cpp | official QAT-format baseline |
| VLLM | pinned direct runtime where 16 GB plan fits | stretch competitor |
| G16-REF | gem16 correctness path | internal differential, not headline |

## Core matrix

### Prefill

| Tokens | Warmups | Retained | Required |
|---:|---:|---:|---|
| 128 | 3 | 10 | yes |
| 512 | 3 | 10 | yes |
| 2,048 | 3 | 10 | yes |
| 8,192 | 3 | 10 | yes |
| 16,384 | 3 | 10 | yes |
| 32,768 | 1–3 | 3–10 | M21 |
| 65,536 | 1–3 | 3–10 | qualified target |

Report prompt ms, tokens/s and TTFT separately.

### Decode

| Cached context | Output forwards | Warmups | Retained |
|---:|---:|---:|---:|
| 128 | 256 | 3 | 10 |
| 2,048 | 256 | 3 | 10 |
| 8,192 | 256 | 3 | 10 |
| 32,768 | 256 or 1,024 soak | 3 | 10 or M21 |
| 65,536 | 256 | 1–3 | 3–10 if qualified |

Report effective selected/accepted output tokens/s, ITL p50/p95/p99 and checksum.

### Output head microbenchmark

- formats Q4_0/NVFP4/BF16 reference;
- rows 1, 3, 5;
- greedy candidate;
- full logits;
- sampling;
- lookup;
- 3 warmups/10 retained.

### MoE microbenchmarks

- router T=1 and T=chunk;
- W13/W2 selected experts;
- shared MLP;
- grouped prefill by routing skew;
- end-to-end layer;
- real and synthetic activations.

## Common semantics

Required:

- identical token ID files;
- batch one;
- no prompt cache hit;
- no CPU offload;
- same output count;
- same stop policy or forced teacher-forced positions;
- same sampling parameters;
- disclosed KV precision;
- one model resident;
- output validity check;
- only target-verified tokens counted.

## Environment control

Record and validate:

- GPU name/UUID/VBIOS;
- device total memory;
- power limit/profile;
- clocks;
- driver/CUDA/CUTLASS;
- OS/kernel;
- ambient/start temperature;
- other GPU processes;
- host RAM/swap;
- compiler JIT concurrency;
- git/artifact hashes.

Run engines serially. Establish a thermal reset criterion instead of waiting an arbitrary fixed time.

## Telemetry

Continuous sampling:

- process VRAM;
- GPU utilization;
- memory utilization;
- power;
- SM/memory clocks;
- temperature;
- throttling reasons.

Representative Nsight Systems/Compute runs are outside retained timing unless explicitly stated.

## Statistics

For every distribution:

- raw values;
- mean;
- median;
- standard deviation;
- min/max;
- p95/p99 where applicable;
- 95% confidence interval for mean;
- paired ratio when runs are naturally paired.

Headline wording:

- use “faster” for throughput;
- use “lower” for latency;
- disclose overlapping intervals;
- never report only the best run.

## Qualification conditions

Production performance passes only when final gem16:

- is quality-qualified;
- beats GOOGLE-Q4 median prefill and decode over the required matrix;
- has no fallback/offload;
- stays inside memory margin;
- dispatches native FP8/NVFP4 paths;
- is deterministic in deterministic workloads.

VLLM/UNSLOTH comparisons can be reported even when timing boundaries/formats differ, but limitations must accompany every table.

## Raw artifact path

```text
benchmarks/results/<date>/<code-sha>/<machine-id>/<engine>/<scenario>/
```

Never overwrite.

## imp reference amendment

Add `IMP-5090-REF` only as external context and candidate `MODELopt-NVFP4` as a quality-control checkpoint. Neither is a 5080 release baseline. The mandatory ordinary-decode matrix is MTP/speculation off at 128/2K/8K/16K/32K/64K cached context with 256 output forwards, median/p95 ITL, actual dispatch and VRAM margin.
