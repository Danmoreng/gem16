> Historical exploratory record. Current qualification and final dispositions
> are indexed in [the closed archive](../benchmark-archive-2026-11-17.md).
> This is not an active benchmark or optimization task.

# 12B Prefill chunk audit — 2026-11-15

The current three-engine12B results compare different Prefill configurations.
They do not establish an advantage at matched chunk sizes or each engine's
best attainable Prefill performance. No new GPU measurements were made for
this audit.

| Engine | Configured Prefill batching | Ordinary Prefill, median tok/s | D2 Prefill, median tok/s |
|---|---|---:|---:|
| gem16 |8,192 tokens per actual outer Prefill chunk |5,984.74 |5,951.39 |
| llama.cpp |logical batch2,048; physical microbatch512 |4,753.98 |4,749.90 |
| vLLM0.33.0 |Chunked Prefill, max_num_batched_tokens1,024; max_num_seqs1 |6,936.53 |6,494.41 |

All use the same16,384 prompt token IDs,1,135 requested output tokens, batch-one
request workload,3 warmups and10 retained runs, without prefix/prompt reuse.
The token budget above is not the internal CUDA matrix/attention tile size.
Different scheduler/kernel implementations do not become identical merely by
setting the same outer token budget. Model-format/KV/Assistant and timing-boundary
qualifications in the underlying comparison still apply.

The llama.cpp helper defaults are `--llama-batch-size2048` and
`--llama-ubatch-size512`, forwarded explicitly to the server. These are
confirmed by both retained Ordinary/D2 configuration JSONs. In the pinned
llama.cpp source, `llama_context_params.n_batch` is the logical submission
maximum and `n_ubatch` is the physical maximum. gem16's reported
`prefill_chunk_tokens=8192` is used by the actual chunk loop in
`src/cuda/engine/inference_engine_prefill_api.cuh`. vLLM's startup log explicitly
confirms `Chunked prefill is enabled with max_num_batched_tokens=1024`.

No chunk sweep was performed. In particular, the roughly25.9% gem16 Prefill
lead over this llama.cpp configuration must not be described as an intrinsic
engine advantage independent of batching. A useful follow-up is a fixed-model,
fixed-prompt sweep of512/1024/2048/4096/8192 where each engine fits in GPU memory,
including a common1024-token control and separately identified best valid
settings. For llama.cpp, logical batch must be at least the physical microbatch.
Record actual sizes, peak VRAM, output IDs and timing boundaries; preserve the
existing results as historical evidence.

Evidence: `benchmarks/results/2026-11-15/58febf4/history-and-12b/`, files
`gem16-{ordinary,d2}.json`, `llama-cpp-{ordinary,d2}.json`,
`vllm-{ordinary,d2}.json` and `vllm-ordinary.stdout.log`.
Public comparison: [current12B evidence](../../../artifacts/benchmark_history/2026-11-15/current-12b.json).
This audit adds no new llama.cpp speed claim or qualification.
