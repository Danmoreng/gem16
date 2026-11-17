# Gemma 4 12B cold prefill, cached append and decode — 2026-11-17

vLLM has the lowest median first-ID latency in all four cold cells and the 16K warm append. GemSixteen is faster than llama.cpp in the three short cells. Both competitors lead GemSixteen on the two longer warm appends. GemSixteen leads decode at 4,096 IDs and near the context limit. vLLM leads decode at 16,384, 65,536 and 130,560 IDs in the measured D3 runs.

| Request, cached + fresh IDs where warm | GemSixteen first ID / decode | llama.cpp first ID / decode | vLLM first ID / decode |
|---|---:|---:|---:|
| Cold 4,096 | 0.506 s / 109.35 | 0.717 s / 102.54 | 0.474 s / 85.87 |
| Cold 16,384 | 2.774 s / 104.10 | 3.401 s / 100.13 | 2.573 s / 115.84 |
| Warm 12,288 + 4,096 | 0.897 s / 104.44 | 0.979 s / 100.02 | 0.838 s / 117.24 |
| Cold 65,536 | 22.849 s / 132.35 | 22.234 s / 111.15 | 19.188 s / 134.95 |
| Cold 130,560 | 76.331 s / 89.66 | 66.768 s / 62.57 | 62.181 s / 94.34 |
| Warm 122,880 + 7,680 | 7.645 s / 89.86 | 6.453 s / 62.87 | 6.471 s / 94.17 |
| Warm 245,760 + 12,288 | 23.296 s / 81.31 | 18.355 s / 49.80 | 19.506 s / 78.84 |

Decode is in IDs/s over 127 post-first intervals. Warm labels give the requested prefix and append. GemSixteen and llama.cpp reuse the full stated prefix. vLLM observes 12,224, 122,816 and 245,696 hits, respectively, so its timed requests evaluate 64 additional prompt IDs.

At cold 130,560, vLLM and llama.cpp have 18.5% and 12.5% lower TTFT than GemSixteen. The native warm append gap is about 1.18 s at 130,560 total IDs and 3.79–4.94 s at 258,048. The matching warm 130,560 native trace spends 6.845 s in global D512 attention, which accounts for most of the optimization opportunity.

At cold 4,096, vLLM records 20.83 ms of elapsed decode per verification group and GemSixteen records 23.70 ms. GemSixteen accepts 51.0% of proposed IDs over 49 groups. vLLM accepts 26.3% over 71 groups. The fewer groups raise native output throughput in this sampled stream. At cold 130,560, the group durations are closer at 28.04 ms for vLLM and 28.33 ms for GemSixteen. vLLM accepts 56.3% of proposals and GemSixteen accepts 50.0%. These group-normalized elapsed values include scheduling and any ordinary tail work.

Sampled near-limit whole-device peaks are 14,421 MiB for GemSixteen, 14,787 MiB for llama.cpp and 14,651 MiB for vLLM. Native's excluded one-token primer takes 239.486 s. Each engine contributes one retained near-limit measurement.

The adjacent native Ordinary control at cold 16K measures 2.777 s to first ID and 52.85 post-first IDs/s, with three retained requests after one warmup. D3 measures 2.774 s and 104.10 IDs/s on the same input. The decode rate is 1.97× Ordinary for this fixture, and the two sampled streams differ. Sampled peak whole-device memory rises from 10,927 to 11,779 MiB. The same llama.cpp Ordinary control measures 3.436 s and 56.67 IDs/s, ahead of native's Ordinary decode rate. The measured D3 rates put native ahead of llama.cpp on that cell.

The later cold 16K D3 control after the reference builds gives 2.744 s and 104.38 IDs/s at a median 2,430 MHz during prefill. The corresponding vLLM control at a 1,024-ID batch limit gives 2.519 s and 115.87 IDs/s. Both controls retain three samples. vLLM's separate ten-sample 4,096-ID batch tuning point gives 2.407 s and 111.81 IDs/s. The primary table retains the 1,024-ID vLLM configuration for every cell.

## Workload and timing

Each request supplies raw prompt IDs, including the existing BOS. The 16,384-ID fixture is the previously retained Wikipedia-derived prompt. Longer requests repeat its body IDs after the first BOS. All measured requests ask for 128 output IDs at temperature 1, top-k 64, top-p 0.95 and seed 42. The shared hashes cover the exact input arrays. The native and vLLM requests disable EOS termination. llama.cpp suppresses its end-of-generation logits to reach the fixed output length. Samplers, speculation algorithms, numeric formats and KV layouts differ. The emitted output IDs and acceptance rates are properties of each engine run. [Workload manifest and cell data](../../../artifacts/gemma4_12b/three-engine-prefill-decode-2026-11-17/comparison.json)

On each cold request the engines compute the full prompt. On each warm request a one-output primer starts from an empty cache and populates the specified prefix. The measured request supplies the exact extended prompt, with the primer output discarded. The results report measured cache hits and computed suffix IDs even when an implementation admits fewer cached IDs than requested. The primer's latency is recorded separately.

Time to first ID ends at the first observed model ID. Post-first decode throughput counts 127 intervals from the first to the last observed ID. Native uses the synchronous resident-library call and an in-process token callback. vLLM measures streaming token deltas from its Python AsyncLLM API. llama.cpp measures its loopback HTTP event stream when that stream contains a validated first ID and all 128 IDs. Engine-side timing is preserved separately. Batching during speculative decode can deliver multiple IDs in one event, so these throughput figures do not measure per-ID cadence. Model loading, graph initialization and primer time are outside the measured request.

## Execution settings

This is a single-slot, sequential comparison on an RTX 5080 Laptop GPU with 16,303 MiB and driver 620.70.06. Fixed-D3 requests use three draft tokens where supported. GemSixteen and vLLM use the pinned FP8-attention/NVFP4 target at `b1f649734b34aa5575b03d186abd1b9be3d0d5c4` and the BF16 assistant at `364bd03c9952e5b7da73665ee30c9eccfc408345`. llama.cpp loads the derived compatibility GGUF with Q8_0 attention weights and NVFP4 MLP weights and a BF16 GGUF of the same assistant. Native uses checkpoint FP8 KV, vLLM uses its own FP8 KV layout and llama.cpp uses Q8_0 KV. This is an engine-configuration comparison with different weight and KV formats and no separate model-quality evaluation.

Native and llama.cpp use greedy assistant proposals against sampled target decisions. vLLM 0.33.0 reports standard MTP rejection sampling. It runs with TRITON_ATTN, FlashInfer CUTLASS NVFP4 GEMMs, full-and-piecewise CUDA Graphs, one sequence and 1,024 maximum batched prompt IDs. The active startup log reports a 64-ID group among the hybrid KV block sizes. Each primary warm vLLM request records 64 fewer prefix hits than requested. [Sample and startup receipts](../../../artifacts/gemma4_12b/three-engine-prefill-decode-2026-11-17/samples-and-receipts.json.gz)

The llama.cpp primary campaign pins official master `9048bad8d6e3a0f9f1726e238e451181c7e02613`, verified at its 21:18:13 launch. It uses one slot, Flash Attention, logical and physical batches of 1,024, Q8_0 KV and three greedy Draft-MTP proposals with the sampled target. Target and assistant load all layers on CUDA0 with explicit GPU embedding placement. The earlier current-at-build revision `a9d43ec` has a complete short validation suite recorded in the raw archive. The upstream branch advanced between cohorts, so the primary measurements rerun the short cells alongside every longer cell on the pinned newer binary and library set.

vLLM runs through the 131,072-slot cohort at a 0.90 GPU-memory budget. The first 262,144-slot attempt at 0.90 fails admission with 2.47 GiB of KV required and 1.64 GiB available. Two attempts with warm compilation caches admit the full slot. The 0.97 attempt reports 4.56 GiB available, and the 0.90 repeat reports 3.47 GiB available. The 0.90 warm-compilation run is the near-limit comparison point. All measured direct-checkpoint runs keep CPU weight offload at zero.

GemSixteen uses the ff0a152d9a79c89436b8e0eed8ac8f0aa7f61a98 base with the current dirty D3 sampling/context changes. The native driver links the release library built with CUDA 13.4.92 and GCC 16.2.1. Automatic D3 selects the optimized graph route. Every measured native point reports an 8,192-token prefill plan, checkpoint FP8 KV, zero fallback and zero token-loop allocations. Native telemetry samples whole-device memory every 250 ms. NVIDIA reports an active 175 W GPU ceiling and an 87 °C target. Software thermal slowdown was active during the long native cohort, so the wall-clock ratios include observed thermal and scheduling behavior. [Source, commands, samples and telemetry](../../../artifacts/gemma4_12b/three-engine-prefill-decode-2026-11-17/samples-and-receipts.json.gz)

The GPU runs are sequential. Six-worker central llama.cpp CPU builds ran from 20:23:55–20:31:33 and 20:44:44–20:52:27 Berlin time. The first overlaps all native short and 65,536-ID cells and most of the two cold 130,560-ID requests. Native's warm 130,560-ID and near-limit append times follow that build. The second overlaps the vLLM short run. Its 65,536-ID and longer cold or warm cases run after the second build. Concurrent CPU load can affect host latency and laptop GPU Dynamic Boost. The telemetry records operating conditions for each phase.

The short cells use 16,640 requested context slots with three excluded warmups and ten retained requests. The 65,536-ID cell uses 65,792 slots and one warmup plus three retained requests. The 130,560-ID cells use 131,072 slots and two retained requests per cell. The near-limit append uses 262,144 slots and a single retained request. Each reported capacity refers to the requested usable slot, with engine-specific padding and physical reservations recorded separately. Long points are descriptive with no statistical extrapolation.

vLLM's first cold 130,560-ID request triggers a sampler JIT warning before the first output ID. Its 61.737 s TTFT remains in the raw archive. The primary median uses the 62.355 s second request and a 62.006 s clean repeat with the sampler initialized before timing. The short and 65,536-ID runs report the warning in excluded warmups. The near-limit run reports it during the excluded primer.

## Prefill kernel breakdown

Two bounded Nsight Systems CUDA/NVTX captures ran separately from primary timing. Capture starts after initialization and, for the warm request, after the 122,880-ID primer. Capture stops on the first output ID. The diagnostic captures correlate all 2,020 cold and 1,019 warm kernel calls to the native prefill range.

| Prefill kernel family | Cold 16,384 IDs | Warm 122,880 + 7,680 IDs |
|---|---:|---:|
| D512 global attention | 994.63 ms / 35.44% | 6,844.91 ms / 87.72% |
| NVFP4 GEMM | 618.23 ms / 22.03% | 317.37 ms / 4.07% |
| FP8 GEMM | 559.06 ms / 19.92% | 297.33 ms / 3.81% |
| D256 local attention | 332.50 ms / 11.85% | 186.03 ms / 2.38% |
| Normalization, quantization and formatting | 276.09 ms / 9.84% | 142.35 ms / 1.82% |
| Other, including NVFP4 weight preparation and output head | 26.26 ms / 0.94% | 15.39 ms / 0.20% |
| Sum of kernel durations | 2,806.77 ms | 7,803.37 ms |

Both captures dispatch the inline `OnlineGlobalAttentionFp8Kernel<1,false>` with 256 threads per block, 254 registers per thread, 98,304 static shared bytes and no recorded local memory. No prepared-global K/V conversion kernel appears. The exact SM120a symbol's static disassembly contains 96 `HMMA.16816.F32.BF16` instruction sites. The inline kernel consumes checkpoint FP8 K/V, converts operands to BF16 and runs BF16 Tensor Core QK/PV. [12B prefill dispatch](../../../src/cuda/engine/inference_engine_prefill_layers.cuh#L364)

The second cold chunk contains 12.89 billion logical causal query-head/key-position pairs across the eight global layers. Its D512 cost is 0.05676 ns per logical pair. The appended chunk contains 124.57 billion pairs and costs 0.05495 ns per pair, about 3% lower in these captures. The growth in global compute time tracks the growth in causal work. The denominator excludes masked or padded tile work, and the normalized interval is whole-device time per logical pair. The kernel interval unions cover 2,806.75 of 2,807.85 ms cold and 7,803.35 of 7,803.89 ms warm. These traces attribute cost within GemSixteen.

The separate vLLM append trace computes 7,744 fresh IDs in seven 1,024-ID steps plus one 576-ID step. The target D512 kernels take 4,925.83 ms and assistant D512 kernels take 611.83 ms, within 6,396.37 ms summed kernel time. Target D512 uses the Triton unified-attention kernel with 128 threads, 168 registers and 41,728 shared bytes per block. Its 0.03923 ns per logical pair is 28.6% lower than the native 0.05495 ns from the separate capture. Kineto records E4M3 queries and BF16 outputs. The matching installed Triton source uses FP8 QK and FP8 probability/value dot inputs, with FP32 softmax and accumulation. Its CTA covers one query position and 16 heads. The native CTA covers 16 query positions and four heads with BF16 Tensor Core math. The time-per-pair comparison therefore includes different numerical, tiling and cache choices. [CUDA/NVTX launch and resource receipts](../../../artifacts/gemma4_12b/three-engine-prefill-decode-2026-11-17/samples-and-receipts.json.gz)

The llama.cpp warm trace reads the full 122,880-ID prefix and evaluates 7,680 new IDs. It records 64 launches of the D512 `flash_attn_ext_f16` specialization taking 5,185.46 ms, or 0.04163 ns per logical pair at the same effective prefix and suffix lengths as native. Total kernel interval union is 6,499.12 ms through the first streamed ID. The GGUF model uses Q8_0 KV. The exact D512 symbol contains static `HMMA.16816.F32` and `HMMA.16816.F16` sites. The observed NVFP4 MLP symbols contain SM120 `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X` sites. [llama.cpp phase trace and exact disassembly receipts](../../../artifacts/gemma4_12b/three-engine-prefill-decode-2026-11-17/samples-and-receipts.json.gz)

## Bit-exact D512 warp-sharing diagnostic

The temporary candidate computes QK and softmax once for each pair of warps that share a query head. One warp publishes the exact BF16 probability fragments and FP32 softmax state to the existing inactive raw K/V buffer. The peer consumes them before the next K copy. The candidate uses the existing synchronization points, 251 registers per thread versus 254 for the parent, and the same 96 KiB static shared allocation. Both variants report zero stack or spill bytes.

| One-layer geometry, cached + fresh IDs | Parent CUDA event | Candidate CUDA event | Paired latency reduction |
|---|---:|---:|---:|
| First chunk of cold 16K, 0 + 8,192 | 28.260 ms | 23.413 ms | 17.17% |
| Second chunk of cold 16K, 8,192 + 8,192 | 83.983 ms | 69.028 ms | 17.59% |
| Long append, 122,880 + 7,680 | 821.008 ms | 662.110 ms | 19.34% |

Each row retains ten alternating AB/BA event pairs after four untimed launches. The long append's paired reductions range from 19.08% to 19.56%. Its active samples have median 2,715 MHz and 72.78 °C. The exactness check compares 592,994,304 BF16 elements across 18 geometries and seeds with nonuniform K/V, with no mismatches or nonfinite values. Memcheck, racecheck and synccheck all pass the three small ragged cases. These are isolated operator measurements. [Candidate generator, raw pairs and validator receipts](../../../artifacts/gemma4_12b/three-engine-prefill-decode-2026-11-17/samples-and-receipts.json.gz)

The resident follow-up links both kernels into one temporary executable and uses one loaded target, assistant and slot per context. The control entry points use the exact release SASS, and the candidate contains the exact operator-test SASS. Both sides use the same D3 graph and a reset before each request. Warm requests get an independently generated, parent-only prefix primer on each side. This controls the starting cache and excludes both primers from append timing.

The first untimed preflight identified lazy D3-graph allocation. A reported a 16 MiB graph reservation before the first generation expanded it to 30 MiB for B, with identical output and proposal IDs. The timed run initializes that graph outside the paired window, then checks exact output, proposals, cache, acceptance and plan data on both sides. The initial preflight remains in the raw archive.

| Resident request | Retained pairs | Parent TTFT | Candidate TTFT | Median paired TTFT reduction | Median paired full request reduction |
|---|---:|---:|---:|---:|---:|
| Cold 16,384 | 8 | 2.757 s | 2.586 s | 169.56 ms / 6.15% | 4.32% |
| Cold 130,560 | 2 | 73.945 s | 63.012 s | 10,933.27 ms / 14.78% | 14.51% |
| Warm 122,880 + 7,680 | 4 | 7.613 s | 6.310 s | 1,301.91 ms / 17.10% | 14.44% |
| Warm 245,760 + 12,288 | 1 | 23.207 s | 19.103 s | 4,103.63 ms / 17.68% | 16.58% |

The cold 130,560-ID candidate result of 63.012 s sits between the 62.181 s vLLM and 66.768 s llama.cpp reference observations. The warmed 130,560-ID result of 6.310 s is below both reference rows on this fixture. At the near limit, its 19.103 s sits between 18.355 s for llama.cpp and 19.506 s for vLLM. The near-limit case is one pair.

The cold 16K candidate reports 104.44 decode IDs/s and the parent reports 104.27. The long warm pair medians are 90.27 and 89.99. Sampled TTFT-interval device peaks match between A and B at 11,779 MiB for 16K, 13,011 MiB for both 130,560-ID cells, and 14,421 MiB near the limit.

The first native release build runs during the near-limit B primer, outside both TTFT windows. A four-second incremental compile overlaps the first cold 130,560-ID parent request. A single-worker host link for the short sweep overlaps the first candidate request. The later reversed pair measures a 15.18% reduction, and the first pair measures 14.38%. The measured candidate uses the frozen diagnostic executable throughout. [Paired requests, exact IDs and hardware samples](../../../artifacts/gemma4_12b/three-engine-prefill-decode-2026-11-17/samples-and-receipts.json.gz)

The lower-bound sweep retains three pairs per length, with the same 128 output IDs and draft, cache and plan data in each pair. Candidate first-ID time is earlier in all nine pairs. The gains at 32 and 512 IDs are under a millisecond.

| Cold input IDs | Parent TTFT | Candidate TTFT | Median paired reduction |
|---|---:|---:|---:|
| 32 | 36.071 ms | 35.816 ms | 0.332 ms / 0.92% |
| 512 | 73.578 ms | 73.345 ms | 0.216 ms / 0.29% |
| 4,096 | 506.795 ms | 495.530 ms | 11.153 ms / 2.20% |

## Release integration

The normal CUDA Release build selects warp sharing for the inline one-KV-head D512 route. Its 3,968 instruction encoding words match the exact B kernel used in both diagnostic harnesses. The two-KV-head inline, prepared and prepared-input instruction encodings match the frozen parent. The recorded A/B medians come from the frozen diagnostic executables built from `ff0a152` plus the archived source patch. The separately recorded D3 sampling and full-context change is commit `8d70ca1432af5f175cde2ac1c32e232f003d5997`. [Release instruction receipts](../../../artifacts/gemma4_12b/three-engine-prefill-decode-2026-11-17/samples-and-receipts.json.gz)

The integrated Release CLI reproduces the historical 262,080-input plus 64-output sampled D3 golden, including all 64 output IDs and 78 proposal IDs. The output hashes to `8ad172a95d295dc73e410a60a815a855ba76be19065e32c7d69e81b2dcc7ddf3` as little-endian 32-bit IDs. The run reports 26 D3 verification groups and 34 accepted proposals. Whole-device memory peaks at a sampled 14,421 MiB. It reports zero fallbacks and zero token-loop allocations. [Release raw result and telemetry](../../../artifacts/gemma4_12b/three-engine-prefill-decode-2026-11-17/samples-and-receipts.json.gz)

All 19 configured Release CTests pass with zero skips. They comprise 2 host tests, 7 CUDA and product tests, and 10 remaining configured tests. The added cold and cached partial-query-tile fixtures pass their FP32-reference comparisons. The cold fixture also checks the exact first-row causal result. The initial cold fixture exceeded the oracle tolerance with identical BF16 outputs from the parent and integrated builds. The final fixture uses scaled input values and the existing operator tolerance. Release Compute Sanitizer passes memcheck, synccheck and racecheck on the filtered `OnlineGlobalAttentionFp8Kernel` family, with zero errors or reported hazards. [CTest, fixture and sanitizer receipts](../../../artifacts/gemma4_12b/three-engine-prefill-decode-2026-11-17/samples-and-receipts.json.gz)

The frozen-parent and integrated Release CLI checks match output and proposal IDs and counters at capacities 3 and 4, at 4,096 greedy input IDs, and at 16,384 and 130,560 sampled input IDs. Release smoke checks also pass exact-D3 and short BF16-KV cases, 12B image and audio requests, and a 26B Compact Vision image request. The 12B audio smoke output is byte-identical to the frozen-parent result and serves as a functional check. [Release validation summary and commands](../../../artifacts/gemma4_12b/three-engine-prefill-decode-2026-11-17/samples-and-receipts.json.gz)

## Follow-up work

Cold 16K spends 42% of kernel time in the combined FP8 and NVFP4 GEMMs. The warmed append gives greater leverage to D512 attention because the new queries attend to the full cached prefix.

Profile the actual D512 causal kernel on both the second 8,192-ID cold chunk and the 7,680-ID append at prefix 122,880. Measure Tensor Core utilization, shared-memory load and stall behavior, register/shared occupancy and L2/DRAM traffic. Normalize kernel time by visible query-head/key-position pairs and preserve absolute latency. The historical two-CTA DSM experiment removed 46% of L2 sectors but increased operator latency 93%. The direct CUTLASS FMHA substitution failed its D512 output geometry. A larger 32-key exactness screen failed numerical limits, and approximate exponentials changed boundary Top-1. These candidates should stay closed. [S07B and earlier D512 experiments](../../PERFORMANCE_LEDGER.md#2026-08-04-reject-two-cta-cluster-sharing-for-global-d512-prefill)

Additional workloads can measure how the released scheduling change behaves across prompts, ordinary and sampled generation, cold chunk boundaries and resident ring wrap. Changes to tiling or precision need their own numerical and memory gates. [Inline kernel and operand schedule](../../../src/cuda/attention/prefill_global_sm120.cu#L399)

If the counters instead show significant conversion cost, isolate per-layer transient BF16 K/V staging versus inline FP8 conversion at the actual append lengths. Staging both K and V at 130,560 positions costs approximately 255 MiB per live layer. The candidate needs a verified workspace alias and measured peak headroom, with the physical FP8 cache retained. The prepared path's small positive result for 26B at short context is prior evidence for a different model and geometry. [Staging implementation](../../../src/cuda/attention/prefill_global_sm120.cu#L900) · [26B prepared K/V evidence](../../PERFORMANCE_LEDGER.md)

For decode, compare milliseconds per actual verification group and accepted IDs per group at the same prefix band before changing a kernel. A high-acceptance 65,536-ID stream can decode faster than a shorter stream. At 16K the previous optimized D3 diagnostic had competitive Target global/local attention costs against vLLM, with remaining losses in Target Down, norm/quant/elementwise, and intragroup gaps. Confirm those costs on this sampled fixture before reusing the diagnosis at longer prefixes. Report ordinary tail steps and assistant acceptance so the per-group speed and output-stream effects stay distinguishable. [Bounded D3 diagnostic](d3-fast-followup-2026-11-17.md)

## Pi session rebuilds

Pi request 38 rebuilds a 136,272-ID tool prompt and spends 80.064 s on prefill with zero cache hits. It contributes 23.9% of the original conversation's 335.065 s measured prefill. Requests 11 and 38 both record `history_or_tools_changed`. The server rebuilds when continuation validation fails. The available counters do not record an exact shared-prefix length for either reset. [Anonymized request and prefill counters](../../../artifacts/gemma4_12b/three-engine-prefill-decode-2026-11-17/pi-cache-rebuild-summary.json) · [Server continuation check](../../../src/cli/server_main.cpp#L597)

Measure the longest identical prefix of the fully rendered token IDs at each rebuild, including the tool definitions and message history. A trusted same-session checkpoint could restore that exact prefix if the current weights, tokenizer, template, KV geometry, sliding-ring positions and contents, assistant state, and sampling, reasoning and tool state all match. Verify restored and rebuilt output IDs and proposals. Measure restore and capture costs, fresh suffix time and the frequency of eligible rebuilds before reporting a gain. The benchmark's verified resident 122,880-ID cache avoids 68.686 s on the matched 130,560-ID native request. This demonstrates the scale of an eligible hit, while the observed Pi reset remains an unmeasured candidate for reuse.

## Evidence

[Comparison data](../../../artifacts/gemma4_12b/three-engine-prefill-decode-2026-11-17/comparison.json) · [CSV](../../../artifacts/gemma4_12b/three-engine-prefill-decode-2026-11-17/comparison.csv) · [Chart](../../../artifacts/gemma4_12b/three-engine-prefill-decode-2026-11-17/comparison.svg) · [Raw requests, command, telemetry and profile receipts](../../../artifacts/gemma4_12b/three-engine-prefill-decode-2026-11-17/samples-and-receipts.json.gz) · [Native build source patch](../../../artifacts/gemma4_12b/three-engine-prefill-decode-2026-11-17/native-source.diff.gz) · [Exact diagnostic kernel patch](../../../artifacts/gemma4_12b/three-engine-prefill-decode-2026-11-17/d512-tested.patch.gz) · [Diagnostic reproduction recipe](../../../artifacts/gemma4_12b/three-engine-prefill-decode-2026-11-17/reproduce-d512.md)

Full local outputs are under `benchmarks/results/2026-11-17/three-engine-12b-prefill-decode/`. Large diagnostic intermediates use checked tmpfs under `/tmp/gem16-experiments/`.
