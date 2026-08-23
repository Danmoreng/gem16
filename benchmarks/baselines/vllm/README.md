# vLLM direct-checkpoint characterization

The current environment pin is vLLM 0.27.1 from its official wheel with Torch 2.13.0 / CUDA 13.0, Transformers
5.14.1, compressed-tensors 0.17.0, vllm-gguf-plugin 0.0.5, gguf 0.19.0, and Setuptools 80.10.2. Existing 12B
cross-engine results below were produced by vLLM 0.26.0 and remain historical; updating the reproducible environment
does not relabel them.

## Build the pinned environment

```bash
./benchmarks/baselines/vllm/build.sh
```

The helper requires the recorded CPython 3.13.14 (`VLLM_BASE_PYTHON` may point to it), creates
`third_party/cache/vllm-0.27.1-env`, installs exact package versions, checks CUDA access, and
applies the audited [`gemma4-mtp-suppress-graph.patch`](patches/gemma4-mtp-suppress-graph.patch). vLLM versions
0.25.1, 0.26.0, and 0.27.1 ship the original `gemma4_mtp.py` with SHA-256
`4eee061c81430be28f029ed66360887a57f8711a75c863067d30e3840a488918`. Python-list suppression indexing constructs
a CPU index tensor during CUDA Graph capture; the patch replaces it with two graph-safe scalar assignments without
changing the suppressed IDs. The patched file SHA-256 is
`2436a940cc7f525880588392a08f5f2b509b51f91394d6666dba181302cf92f7`.

Transformers 5.15.0 is intentionally not selected: its configuration normalization rejects Gemma 4's heterogeneous
per-layer `head_dim` before runtime construction.
Transformers 5.14.1 is within vLLM 0.27.1's declared range and passes the same local configuration probe.

## Gemma 4 26B official Q4_0 loader result

The latest environment was tested with Google's immutable official text-only QAT Q4_0 GGUF. vLLM correctly selects
`Gemma4ForConditionalGeneration`, the external GGUF loader and GGUF quantization when the local Hugging Face
configuration/tokenizer is supplied separately from `model_weights`. Actual engine construction then fails before
GPU weight upload with `Failed to map GGUF parameters (416)`. The unmapped set includes the vision tower expected by
the multimodal model class although the GGUF is text-only, along with Gemma 4 router scale names. Consequently this
exact upstream vLLM/plugin combination cannot currently run the selected Google Q4_0 checkpoint as-is, and there is
no valid vLLM throughput row to report.

We do not patch the plugin for this competitor measurement: doing so would stop being a latest-upstream reference.
The exact versions, model identity, attempted semantics, failure boundary and limitations are retained in
[`gemma4-26b-q4_0-load-characterization.json`](gemma4-26b-q4_0-load-characterization.json).

## Gemma 4 26B community W4A16 16K+64 result

A pinned community compressed-tensors checkpoint does load in vLLM 0.27.1 after the harness validates the declared
static symmetric weight-only channel-wise INT4 metadata for configured Linear targets and its declared BF16
exceptions. It explicitly maps the checkpoint metadata's `group_size=null` to vLLM's equivalent `-1` convention in
memory; checkpoint files are not modified. The exact 16,384-token Wikipedia workload plus 64 forced greedy output
tokens, one warm-up and three measurements produced:

| Mode | Prefill | TTFT | Decode | ITL | Output hashes | Peak device-wide `memory.used` |
|---|---:|---:|---:|---:|---:|---:|
| CUDA Graph | 6,475.80 tok/s | 2,530.04 ms | 149.35 tok/s | 6.696 ms | 3 different / 3 | 15,818 MiB |
| Eager control | 6,367.42 tok/s | 2,573.10 ms | 58.50 tok/s | 17.095 ms | 3 different / 3 | 15,546 MiB |

The CUDA-Graph row is an external performance-headroom observation only. Both modes diverged across repeated greedy
runs after the same first output token, so CUDA Graph replay is not a necessary cause and neither row is a
correctness, determinism or quality baseline. The checkpoint also differs from gem16's compiled mixed FP8/NVFP4
artifact and Google's official QAT Q4_0 GGUF; the numbers do not establish model-quality parity.

Positive runtime logs select Marlin W4A16 linear and MoE kernels, Marlin experts, and Triton attention. The graph run
uses aggressive `gpu_memory_utilization=0.985` and external 200 ms telemetry observes only 63 MiB free at peak. Both
modes emit two non-fatal allocator OOM warnings during initialization before continuing. The graph service
force-kills a remaining EngineCore after writing its result; the Eager service merely reports a teardown timeout
after its result is complete, without identifying the remaining process. All samples, exact commands, checkpoint
and raw-report hashes, dispatch lines, telemetry, failed-attempt audit and limitations are retained in
[`gemma4-26b-w4a16-wikipedia-16k64-characterization.json`](gemma4-26b-w4a16-wikipedia-16k64-characterization.json).
The run did not record a preflight snapshot of the laptop power profile, power limit, starting temperature or other
GPU processes.

Against the current gem16 development row for the same prompt token IDs and output count, the reported graph metrics
give a 2.04x prefill ratio (6,475.80 versus 3,169.46 tok/s) and a 1.24x ordinary-decode ratio (149.35 versus 120.40
tok/s). The checkpoints differ, and vLLM's `first_token_latency` includes request/first-token work that gem16's
prefill-only interval excludes. These directional ratios only prioritize engineering work: the larger reported gap
is prefill.

The following startup note belongs to the historical 0.26.0 run. A cold start JIT-builds several memory-heavy
FlashInfer NVFP4 CUTLASS variants. Unbounded startup on the
32-core/64-GiB reference host launched enough concurrent `cicc` processes to exhaust RAM. The cross-engine harness
therefore uses `MAX_JOBS=4`, `TORCHINDUCTOR_COMPILE_THREADS=4`, and one internal NVCC thread per job. A controlled
cold start reached at most four compilers; the one-job diagnosis retained at least 49.0 GiB available RAM. Startup
compilation is outside measured inference timing.

## Historical 0.26.0 16K fixed-D2 result

At `gpu_memory_utilization=0.92`, vLLM provisions 19,069 FP8-KV tokens for the exact 17,519-position workload. On
the 16,384-token Wikipedia prompt followed by 1,135 fixed output positions, three warm-ups and ten measurements
produce:

| Metric | Median | Mean 95% CI |
|---|---:|---:|
| Prefill | 6,247.55 tok/s | [6,242.76, 6,262.95] tok/s |
| TTFT | 2,622.47 ms | [2,616.03, 2,624.47] ms |
| Effective D2 decode | 81.95 tok/s | [81.90, 81.98] tok/s |
| ITL | 12.202 ms | [12.198, 12.211] ms |

Every run proposes 1,083 drafts, accepts 590, rejects 493, and executes 542 Target batches. The output is
deterministic within vLLM. The complete data and telemetry are in
[`../cross_engine_mtp/characterization.json`](../cross_engine_mtp/characterization.json).

FlashInfer autotuning reports GPU-OOM fallbacks for some tactics, and the 8,192-token NVFP4 prompt shape remains
outside its tuned bucket range. Those fallbacks are retained in the raw console log and must accompany the result.
vLLM MTP also differs from its own ordinary greedy output, so this remains a hardware/performance characterization,
not exact speculative-decoding correctness evidence.

## Historical characterizations

`direct-bf16-kv-characterization.json` and `mtp-characterization.json` retain the earlier vLLM 0.25.1 BF16-KV and
MTP investigations. They are historical evidence and are not the runtime behind the current README table. The
older comparison used different memory policy and timing boundaries; do not combine its rows with the current
0.26.0 result.

## Reproduce

Use the complete three-engine harness after preparing all artifacts:

```bash
systemd-run --user --scope \
  -p MemoryMax=48G -p MemorySwapMax=0 \
  ./scripts/benchmark-cross-engine-mtp.sh
```

The harness fixes batch one, token-ID input, no detokenization, no prefix cache, no CPU offload, text-only loading,
chunked prefill, greedy decoding, CUDA Graphs, FP8 KV, D2, three warm-ups, ten measurements, and
`gpu_memory_utilization=0.92`.
