# 12B D2/D3/D4 optimization parity —2026-11-17

All three fixed-depth modes now use the same exact M16 FP8 row-packing optimization. On the same final binary, fixed D3 is the fastest measured mode at **100.31758 output tok/s**. D4 improves substantially; D2 shows no demonstrated throughput gain. No default-depth, precision, layout, model/profile, timing-boundary or release change.

| Mode | Previous median | Final median | Change | Three final samples |
|---|---:|---:|---:|---|
| D2 |99.00644|98.72631|-0.283%|[98.59926441818608, 98.72630873301237, 98.73834339001645]|
| D3 |100.33001|100.31758|-0.012%|[100.3797434739889, 100.31758388549287, 100.27057138309725]|
| D4 |86.03488|97.17973|+12.954%|[97.3073160685785, 97.17973108465948, 97.12895710565985]|

Each final cohort is one warmup plus three retained requests, as requested. All three are measured anew from one final executable, with cooling to≤50°C before each cohort and200ms NVML telemetry. Previous medians are immutable earlier cohorts, not newly paired AB/BA controls. The small D2 difference does not establish a causal regression; it also supports no speedup claim. D3 remains effectively unchanged.

Final gem16-run SHA256: `b979998e11d4ee9016ea35a693de5dac991863bcc3072c2c747e4c54524e2e53`.

## Scope and parity

Raw root (`P` below): `benchmarks/results/2026-11-17/12b-mtp-parity/`. Parent: `.../d3-fp8-packed/production-candidate/`; that build contained packing only for D3. Owner scope/authorization: `docs/plans/gemma4-12b/MTP_PARITY_2026_11_17.md`. The current instruction supersedes the prior T4-only scope and requests fresh measurements of all three. No commit, push, branch switch, new tactic search or other-model benchmark.

| Feature | D2 | D3 | D4 |
|---|---|---|---|
| Verifier rows |3|4|5|
| Device-controlled conditional graph and mapped-ring streaming |yes|yes|yes|
| Fixed-row attention/head, norm fusion, fixed Down and Gate/Up policy |yes|yes|yes|
| Shared M16/K32 FP8 projection of all verifier rows |yes|yes|yes|
| First-window-crossing correction on12B dispatch |yes|yes|yes|

`src/cuda/fp8/sm120.cu` specializes `TwelveBPackedFp8ProjectionKernel<3/4/5>` for the known12B QKV/O shapes only: K3840 local Q/K/V4096/2048/2048 or global Q/K8192/512; O N3840/K4096 or8192. All real rows share one native M16/K32 MMA;32 ephemeral shared-memory padding bytes per row avoid bank conflicts. K ordering, FP32 scale order, BF16 boundaries, resident weight bytes and tied aliases remain exact. No weight repack, second representation, recurring allocation or framework fallback. The earlier disjoint shape branches remain preserved.

`src/cuda/attention/sm120.h` and `decode_sm120.cu` add a12B-specific local verifier entry. `inference_engine_prefill_layers.cuh` selects it for12B. Its T3 instantiation applies the same row-wise window shift/token cap already present at T4/T5 when a group first crosses position1023. The existing shared three-row entry retains its previous instantiation; this correction is localized to12B. The timed16K workload is already beyond this first crossing, and its sequence/state is unchanged.

Tests extend `twelve_b_t5_ports.cu` to the T3 fused boundary and select the12B entry in `twelve_b_fixed_attention_ports.cu`. The diagnostic analyzer recognizes all packed row specializations, configurable diagnostic length and explicitly expected D2 dispatch changes. All unrelated dirty patches are checked unchanged. Source snapshots, generator-derived existing headers, matched core/runtime binaries and exact per-command records are retained.

## Actual dispatch and cost

| Family, diagnostic ms/group | D2 | D3 | D4 |
|---|---:|---:|---:|
|FP8 QKV/O|3.53571|3.53560|3.60232|
|Assistant, all proposal steps|3.14178|4.57253|6.06187|
|Target attention|3.43750|5.01027|5.56493|
|Target norm/quantization/RoPE|2.22289|2.20704|2.16392|

The same sixteen complete warmed groups (indices16–31) are selected from each128-output Nsight capture. Each has1536 packed projection calls (96/group), exactly one request-level cudaGraphLaunch, no synchronization/allocation APIs in the captured range and zero token-loop allocations/fallbacks. The diagnostic output prefix matches the first128 IDs of the1135-output fixture. Shorter traces are diagnostic only; they do not change the primary workload or timing.

Actual SM120a SASS confirms static MMA counts15→5 for D2,5→5 for D3 and25→5 for D4 in the corresponding loop body. The arithmetic operation per K32 block packs three/four/five rows into one atom. Resource reports, SASS, ordered timelines, grids, blocks, registers and shared-memory data are retained. D2’s planned invocation differences are restricted to FP8 projection and its localized attention specialization; all family counts remain unchanged.

D4’s earlier FP8 family cost was7.79196ms/group; it now falls to approximately3.60ms. D2 was already around3.52ms and remains around3.54ms, so fewer MMA instructions do not translate into a measured D2 throughput gain here. No hardware stall-counter/memory-bound diagnosis is claimed. D4 still performs four Assistant proposal steps and five attention rows, with only a modest extra accepted prefix over D3. These costs explain why complete optimization parity need not make D4 fastest. The prior vLLM117tok/s result is not reached or remeasured by this task.

## Workload, correctness and checks

RTX5080 Laptop SM120,16,303MiB, driver620.70.06; CUDA13.4.92/GCC16.2.1; Release C++20, compute_120a/sm_120a and RDC=true. Full prior toolchain identity: `benchmarks/results/2026-11-17/vllm-alignment/toolchain.json`. Target revision `b1f649734b34aa5575b03d186abd1b9be3d0d5c4`; Assistant revision `364bd03c9952e5b7da73665ee30c9eccfc408345`. Base commit `81118043fb526a8f86f07f030d53d9ddbd5b7104`, dirty branch `main`; source and binary hashes accompany the result.

Unchanged fixture:16,384 exact prompt IDs,1,135 outputs,1,134 timed decode intervals, context17,519, greedy, suppression1/50/106/258882/258883, physical FP8 KV, batch1, no prefix reuse or CPU weight offload. Primary throughput is1134/decode_seconds, excluding model load/prefill/first token. No profiler/interposer is present in primary cohorts.

- For each of T3/T4/T5, all130 pinned-weight operator outputs/guards are byte-exact to the parent:390 comparisons, including full logits/candidates and layers0/15/31/47 with zero/finite/large cases.
- All1135 output IDs and every proposal are exact at all depths: D2 has1014 proposals/626 accepted/507 groups; D3 has1311/694/437; D4 has1640/721/410.
- Whole311287808-byte KV hash for every depth remains `6b03676412022f6be775dfec7ec4e9e68ef3d997a28e6f5f3f100ff36c577231`. Exact full-state payloads share an immutable hard link instead of duplicate disk blocks.
- 42 direct ordinary-attention cases per row count pass byte-exactly. Five full-model prompt lengths1019..1023 at each depth (15 cases) match all32 outputs and the whole KV state of pinned ordinary references. Matching transient payloads are hashed and removed; ordinary references remain retained.
- 12B text, image, audio, BF16-KV, small-context/direct behavior, full CUDA tests, host unit tests and29 benchmark-helper Python tests pass.
- Greedy/sampled17-case resident chat passes for all depths, including cached continuation/penalties, streaming, reset, stop/tails, reasoning and cancellation. Sampled CLI seeds0/1/42 and two-turn interactive chat pass at D2/D3/D4.
- New row-packed operators at all three row counts, MTP control and12B T3 boundary attention pass memcheck/synccheck/racecheck, with instrumented outputs rechecked. Fixed-chain full-model memcheck passes at all depths, including first-window-crossing inputs. Unchanged T4/T5 attention’s expanded scoped checks are retained from the preceding qualification.
- All three local server modes pass two sampled HTTP turns with frequency/presence penalties, correct fixed-depth health metadata and positive corresponding depth counters. HTTP prefix-KV reuse is not claimed by these smokes; SDK cached behavior is checked separately.

The known unfiltered full conditional-model Assistant PreProjection synccheck report remains unresolved and is reproduced by the pinned old D2 parent. It is not waived or labeled a false positive. Full conditional-model racecheck is not claimed; an earlier direct-model run hit600s. The new operators pass their scoped tools and all three model memchecks pass. This remains local experimental work rather than formal release promotion.

## Reproduction and resource record

Exact commands/streams: `P/*command.json`, `P/qualification/`, `P/sanitizers/` and `P/server-smoke-d{2,3,4}/`. The staged scripts are `prepare.py`, `qualify_numerics.py`, `run_primary.py DEPTH`, `capture_dispatch.py`, `qualify_protected.py`, `qualify_sanitizers.py`, `server_smoke.py DEPTH` and `check_source.py`; use a new output root to preserve existing evidence. Each primary command explicitly specifies one warmup/three requests. All run on the same pinned executable in `P/production-candidate/`.

Complete Nsight reports remain `.nsys-rep`. SQLite exports are losslessly compressed as `.sqlite.gz`; `compressed-sqlite.json` records uncompressed lengths/SHA256 and compressed lengths. Decompress to a new diagnostic directory before rerunning the analyzer. Primary JSONs and NVML CSVs are uncompressed and unchanged.

| Mode |200ms sampled peak VRAM, MiB |
|---|---:|
|D2|11763|
|D3|11767|
|D4|11771|

These are sampled peaks, not proof of the absolute instantaneous maximum. All per-token weights remain resident; zero fallback, recurring allocation and persistent repack are reported at all depths. No formal speedup claim is made for D2. D3 is the best depth measured on this fixture; other workloads/acceptance patterns may differ.

Machine-readable result: `artifacts/gemma4_12b/mtp-parity-2026-11-17.json`.
