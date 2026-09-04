# NVFP4 Wikipedia recheck — 2026-09-04

Current Linux characterization, not a new optimization or global maximum claim.
Engine commit: `61622046ffc5f4a33392862e127cdbf4c3982f97`; clean source at start.
RTX 5080 Laptop GPU, driver 610.57.04, CUDA 13.3.73, GCC 16.2.1,
Release/SM120a, `max-power`, nvidia-powerd active. No tuning environment overrides.

## Results

Three warmup pairs, ten retained alternating pairs, batch one, 16,384 fixed
Wikipedia input tokens, checkpoint FP8 KV, sampled temperature 1, top-k 64,
top-p 0.95, repetition penalty 1, seed 0. All runs generated 942 tokens.
Throughput counts 941 post-first-token intervals, not speculative proposals.

| Decode token/s | README median | Current median | Current maximum | Mean | SD | 95% CI of mean |
|---|---:|---:|---:|---:|---:|---|
| Ordinary | 148.293 | 148.336 | 148.384 | 148.271 | 0.186 | 148.138–148.404 |
| Fixed D2 | 203.842 | 203.552 | 203.979 | 203.632 | 0.175 | 203.506–203.757 |

D2 median differs by -0.1426% from the historical baseline; ordinary by +0.0292%.
D2 improves current ordinary throughput by 37.2233%.
D2 median TTFT 2365.5 ms; prefill 6926.231 token/s;
inference end-to-end 6988.395 ms, excluding process startup/model loading.
Sampled peak GPU memory: D2 15026 MiB, ordinary 14734 MiB (whole-GPU telemetry).

All current output token vectors equal the historical baseline exactly:
`ee97710da349b4bcb5feedfaa81e10ff003c5a54642c9886d494db5c76ccc112`.
Every D2 run reports 418 D2 verification groups/target batches,
523 accepted of 836 proposed tokens (62.5598%), no D1/D4 groups or ordinary tail.
Runner checks zero reported fallbacks and no token-loop allocations on every run.
No fresh instruction profiler was run; this is a runtime remeasurement of the
qualified path, not a new kernel/instruction claim.

## Artifact and reproduction

Target uses the pure-root snapshot `63508b5826527484e707b4b46e2eacf077cf2b35`.
Its 14,696,668,160-byte `model.gem16` and the current consolidated revision
`6de2a057f11332420819f8e6efd08e42d7a03bc7` are the same inode (33788273).
Metadata artifact identity:
`471805f7dad8abb84300be78b2822a63dcb1d35bff5aa98426a162cc8532ee17`.
No full model payload hash was recomputed. Assistant uses the current consolidated
revision's independent runtime view. Historical README evidence used the old
sharded target layout; this measurement uses the published device image.

An initial pre-measurement attempt used the older `build/blackwell-release` binary
and consolidated root and failed closed on a non-regular entry (component
subdirectories). It produced no measurements. The successful run uses the fresh
`build/Linux/blackwell-release` binary and pure Target view below.

```sh
python3 tools/qualify_mtp.py \
  --workload benchmarks/prompts/wikipedia-summary-16k.json \
  --output benchmarks/results/2026-09-04/61622046/rtx5080-laptop/nvfp4-wikipedia-sampled-d2-current-3w10.json \
  --model /home/sebastian/.cache/huggingface/hub/models--danmoreng--gemma-4-26B-A4B-it-GEM16/snapshots/63508b5826527484e707b4b46e2eacf077cf2b35 \
  --assistant-model /home/sebastian/.cache/huggingface/hub/.gem16/snapshots/danmoreng--gemma-4-26B-A4B-it-GEM16--6de2a057f11332420819f8e6efd08e42d7a03bc7--assistant \
  --executable build/Linux/blackwell-release/bin/gem16-run \
  --warmup-pairs 3 --measured-pairs 10 --sampled --seed 0 \
  --minimum-mtp-decode-tps 203.84246326483017 --characterization
```

Exit 0, status `characterized`. The supplied threshold is the historical median,
not a newly opened qualification target. Raw JSON retains all measured samples,
200 ms telemetry, binary/compilation metadata hashes and output identity.
No source or README changes; historical evidence remains unchanged.
