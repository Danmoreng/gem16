# IMP exploratory characterization

IMP is an external, MIT-licensed, consumer-Blackwell-specific CUDA inference engine. It is not an accepted gem16
baseline. The first characterization pins upstream commit `a392904d4216388828d0d56317de046f4ca49627` and tests
the exact official Google Gemma 4 26B A4B QAT Q4_0 GGUF used by the adjacent llama.cpp characterization.

Docker was unavailable on the reference host, so the canonical IMP container build could not be used. A clean host
Release build succeeded with GCC 15.3, CUDA 13.3.73, and CUTLASS 4.6.1. GCC 16.1 exposed a missing direct
`<cstdint>` include in upstream `weight_cache_file.h`; no source patch was retained.

The official Q4_0 file loads and produces coherent text, but it does not fit IMP's optimized all-resident execution
plan on the 16 GB RTX 5080 Laptop GPU. IMP uploads 21 of 30 MoE expert layers and leaves 3.59 GiB of expert weights
on the host. CUDA Graphs are consequently disabled, and the 10-repetition run records a 64.2% expert-cache hit
rate. This violates gem16's no-CPU-weight-offload baseline contract.

IMPs default conservative 3,900 MiB library reserve leaves only 256 K/V tokens and cannot execute the 512-token
prompt benchmark. The characterized run uses the documented planner override `vram.library_reserve_mb=256`, above
the approximately 100 MiB observed first-forward device increase, plus explicit FP8 K/V and a 1,024-token K/V
minimum. INT8 K/V is invalid for this model because IMP's kernel rejects the global-attention head dimension 512.
The experimental `moe.allow_graphs_under_offload=true` route fails graph capture and falls back without a speedup.

At synthetic pp512/tg256, batch one, Max Power, and ten repetitions after IMP's built-in discarded warm-up, IMP
reaches 1,533.75 prefill tok/s and 51.64 decode tok/s. The directly adjacent llama.cpp b10240 run uses the same GGUF,
full GPU residency, Q8_0 K/V, Flash Attention, and ten repetitions: mean throughput is 5,087.77 prefill tok/s and
169.762 decode tok/s at existing context 512. llama.cpp is therefore 3.32x and 3.29x faster respectively. IMP peak
sampled VRAM is 14,804 MiB versus 15,316 MiB for llama.cpp because IMP offloads experts.

These numbers are an implementation-fit result, not a general judgment on IMP: its published Gemma results use
Q4_K_M or NVFP4 on a 32 GB RTX 5090, where all experts fit and its optimized CUDA-Graph path remains active. No
quality comparison or profiler-level kernel audit was performed. Full commands, telemetry, caveats, and statistics
are in [`gemma4-26b-a4b-qat-q4_0-characterization.json`](gemma4-26b-a4b-qat-q4_0-characterization.json).
