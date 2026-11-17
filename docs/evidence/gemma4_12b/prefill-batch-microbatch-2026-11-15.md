# 12B llama.cpp logical batch / microbatch follow-up — 2026-11-15

The owner requested separating the two controls after the matched-chunk curve. Logical batch is held at 8,192 while physical microbatch is 512, 1,024 or 2,048. An adjacent 1,024/1,024 control checks the effect of logical batch at fixed physical microbatch. All four points use three warmups and ten retained samples, the same 16,384 prompt IDs, 17,519 requested context, one greedy output token, no prefix reuse, Q8/NVFP4 GGUF and Q8_0 KV. No engine runtime changes or Decode measurements are involved.

| Logical batch | Physical microbatch | Median tok/s | Bootstrap 95% | CUDA compute buffer MiB | Sampled device peak MiB |
|---:|---:|---:|---:|---:|---:|
|8,192|512|4,728.51|4,710.84–4,752.88|198.78|9,825|
|8,192|1,024|4,798.45|4,782.03–4,822.04|398.55|10,111|
|1,024|1,024|4,812.04|4,802.05–4,823.39|398.55|10,111|
|8,192|2,048|4,598.25|4,589.62–4,611.16|801.11|10,693|

At physical microbatch 1,024, changing logical batch from 1,024 to 8,192 changes the measured median by -0.282%. These sequential cohorts do not establish a general advantage for such a small difference.

Earlier same-revision diagonal controls (three warmups, ten retained, separate cohorts):

- Batch = microbatch = 1,024: 4,839.77 tok/s.
- Batch = microbatch = 8,192: 3,536.49 tok/s.

All four new server logs confirm the requested logical/physical sizes, 49/49 GPU layers and explicit CUDA0 token-embedding placement. All retained first tokens are 61684 and reported cross-request cached tokens are zero. The central resolver verifies official upstream before each cohort; all use `18b74fa41dcac1f55ae1371fe58153f7225941f6` and the same executable/library set as the earlier curve. Source, model and helper identities, raw samples, commands and external telemetry are preserved.

The sweep isolates the batch configuration; it does not profile which CUDA operator causes large-microbatch performance to decline. Reserved compute buffer size is an allocation fact, not proof of the causal performance bottleneck. The original matched-chunk curve and historical measurements remain unchanged. Decode optimization remains paused.

[JSON](../../../artifacts/gemma4_12b/prefill-batch-microbatch-2026-11-15/results.json) · [CSV](../../../artifacts/gemma4_12b/prefill-batch-microbatch-2026-11-15/results.csv) · [Original matched-chunk curve](matched-prefill-2026-11-15.md)

Raw evidence: `benchmarks/results/2026-11-15/58febf4/12b-matched-prefill/batch-microbatch-followup/`. No commit or push is part of this task.
