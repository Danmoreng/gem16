# Linux short-context ordinary-decode optimization plan

Status: ready to start after the current Windows benchmark documentation is committed and pushed

Primary platform: Linux x86-64 on the reference NVIDIA GeForce RTX 5080 Laptop GPU, Blackwell SM120, firmware
profile `max-power`, active `nvidia-powerd`

Primary model: `unsloth/gemma-4-12b-it-NVFP4` at
`b1f649734b34aa5575b03d186abd1b9be3d0d5c4`

External engine: llama.cpp b10240 at `0b14b87d7c20cb753b94b96854dd7b45306fc696`

Plan revision: 1, 2026-08-05

## 1. Purpose and authority

Measure, explain, and conditionally reduce gem16's ordinary batch-one decode latency at short context. The trigger
is the current Windows standard-shape characterization: gem16 reaches 52.43 tok/s for its closest `tg128`
counterpart while llama.cpp reaches 62.08 tok/s in standard `llama-bench tg128`.

This is a new bounded workload-specific investigation. It does not reopen the completed 16K fixed-D2 sprint, change
the production precision contract, or authorize an unbounded sequence of speculative kernels. The retained 16K D2
result remains the regression reference. Gemma 4 26B A4B M00 resumes after this plan and the outstanding direct
all-regions memory-reserve record are closed.

Permanent rules in [AGENTS.md](../AGENTS.md), [DEVELOPMENT.md](DEVELOPMENT.md),
[BENCHMARKING.md](BENCHMARKING.md), [CORRECTNESS.md](CORRECTNESS.md), and [MEMORY.md](MEMORY.md) remain binding.
Rejected approaches in [DECISIONS.md](DECISIONS.md) may not be repeated without a materially new measured
hypothesis.

## 2. Why the investigation runs under Linux

Linux is the qualification and profiling environment for this work because it provides:

- the locked reference toolchain and the established `max-power` plus `nvidia-powerd` Dynamic Boost contract;
- command-line Nsight Systems capture with CUDA Graph child-node visibility;
- more reliable Nsight Compute kernel replay and counter collection than the observed Windows WDDM path;
- lower profiler and scheduler ambiguity for short sub-20-millisecond kernel groups;
- the existing pinned llama.cpp and direct-checkpoint reference environment on the same physical GPU.

Windows and Linux measurements are separate characterizations. The Windows result motivates this plan but is not a
Linux parent or promotion baseline. Every candidate is compared only with an adjacent Linux parent built and run
under the same machine state.

## 3. Questions this plan must answer

1. Does the short ordinary-decode gap reproduce under Linux?
2. How much of the observed gap comes from different benchmark boundaries rather than model execution?
3. Which gem16 phase dominates at context depths 1, 128, and 512 after attention cost becomes small?
4. Is the limiting resource GPU kernel work, graph/API synchronization, output selection, or a mixture?
5. Can a bounded production change improve short ordinary decode without regressing 16K ordinary, fixed-D2,
   prefill, correctness, memory, or Windows buildability?

No implementation begins before questions 1–4 have retained evidence.

## 4. Benchmark classification and timing boundaries

The external result is a standardized-shape characterization, not exact token or format parity:

- `llama-bench pp512` uses synthetic random prompt IDs; gem16 uses its deterministic benchmark prompt formula.
- `llama-bench tg128` begins at position zero and feeds a random token at every step.
- `gem16-bench decode --context 1 --tokens 128` begins after the smallest legal one-token prompt and follows the
  actual greedy Target sequence.
- llama.cpp uses the patched NVFP4/Q8_0 GGUF. Its standard row uses F16 K/V; gem16 directly loads FP8 attention,
  NVFP4 MLP, and checkpoint-FP8 K/V.
- llama.cpp reports aggregate generation time. Gem16 includes its documented full forward and greedy-selection
  boundary and also retains every inter-token interval.

These differences must remain visible. Do not remove production token selection, change precision, or patch the
competitor's timed semantics merely to manufacture parity. A diagnostic phase timer may isolate gem16 costs, but it
must not become a silent production shortcut.

## 5. Required workload matrix

### 5.1 Canonical standard-shape row

Retain the user-facing `llama-bench` shape:

| Engine | Prefill | Decode |
|---|---|---|
| llama.cpp | `pp512` | `tg128`, depth 0 |
| gem16 | context 512 prefill | context 1, 128 generated Target tokens |

Run llama.cpp with its built-in warm-up and 13 repetitions, discard repetitions 1–3, and report 4–13. Run gem16
with three warm-ups and ten measured repetitions. Preserve all 13 llama.cpp samples in raw evidence.

### 5.2 Product-relevant ordinary-decode matrix

Use 256 timed decode positions at existing contexts:

```text
128, 512, 2,048, 8,192, 16,384
```

Gem16 uses checkpoint-FP8 K/V. Llama.cpp runs the primary F16-K/V standard row and a separate Q8_0-K/V sensitivity
row. Do not merge the two llama.cpp configurations. Model load and depth construction remain outside decode timing;
cache reset remains outside timing for every measured repetition.

### 5.3 Regression workloads

Every production candidate that survives screening must retain:

- context-512 prefill;
- ordinary decode at context 128 and 16K;
- the exact fixed 16K Wikipedia D2 workload with 1,135 output positions;
- deterministic output hashes and ordinary/MTP Target identity inside gem16;
- the direct all-regions memory-reserve measurement with Target, assistant, graphs, media, sampling, and the final
  prompt plan resident.

## 6. Phase sequence

### L00 — synchronize, build, and lock the Linux parent

1. Fast-forward from `origin/main` and require a clean worktree.
2. Record the full gem16 SHA, submodule SHAs, model locks, llama.cpp commit, GGUF hash, compiler, CUDA, driver,
   firmware, VBIOS, Nsight versions, power profile, service state, display use, and idle GPU state.
3. Rebuild host-debug and Blackwell release from the committed tree.
4. Run host and CUDA CTest.
5. Rebuild pinned llama.cpp and verify the selected GGUF inventory and 49/49 GPU layer residency.
6. Record the outstanding direct all-regions CUDA-visible memory reserve on the untouched parent.

Exit gate: no benchmark or profile is accepted until the parent is reproducible, the GPU is idle, `max-power` is
active, `nvidia-powerd` is active, and the required model bytes remain resident.

### L01 — establish the Linux performance parent

1. Run the canonical standard-shape row.
2. Run the product-relevant context matrix for gem16 and both disclosed llama.cpp K/V configurations.
3. Use three conditioning repetitions and ten reported samples, adjacent engine order, matched start temperature,
   and continuous 200 ms telemetry.
4. Report median, mean, sample standard deviation, 95% Student-t mean interval, range, aggregate milliseconds per
   token, gem16 p50/p95/p99 ITL, peak VRAM, power, clocks, and thermals.
5. Retain output hashes, actual graph status, fallbacks, allocation counters, and layer residency.

Exit gate: if the short-context gap does not reproduce or its intervals are inconclusive, increase to 30 reported
samples. Do not begin kernel work from the Windows ratio.

### L02 — audit the boundary difference

Add profiling-only measurement visibility where existing NVTX/JSON boundaries are insufficient. Attribute gem16
ordinary decode into:

- graph launch and host/API synchronization;
- embeddings and input preparation;
- Q/K/V/O projections and their norm/quantization boundaries;
- local and global attention;
- NVFP4 Gate/Up and Down;
- residual/norm operations;
- output head, softcap, candidate reduction, and final greedy selection;
- device-to-host publication required by the benchmark boundary.

Measure rather than infer the cost of actual greedy selection relative to llama.cpp's random-token feed. Any
diagnostic forced-token or phase-timer path must be explicit, test-only or benchmark-only, and excluded from primary
production claims.

Exit gate: the phase totals must reconcile with end-to-end time closely enough to rank the recoverable gap. If they
do not, improve attribution before selecting a candidate.

### L03 — Nsight Systems timeline

Capture one uninstrumented timing parent plus scoped Nsight Systems traces for context 1, 128, 512, and 16K. Use
CUDA/NVTX tracing, CUDA Graph child-node visibility, no CPU sampling, and the minimum output length that still shows
steady behavior.

For each profile retain:

- CUDA Graph launch count and child-kernel order;
- CPU gaps, CUDA API duration, stream waits, and synchronizations;
- kernel launch count and summed/mean duration by family;
- overlap or serialization between phases;
- the first token separately from steady tokens;
- clocks and thermals beside the trace.

Exit gate: produce a ranked table covering at least 95% of steady-token wall/GPU time. A source-visible operation
without measured cost is not a candidate.

### L04 — Nsight Compute counters

Collect targeted counters only for kernel families admitted by L03. Record:

- executed kernel and representative SASS path;
- launch geometry, registers, shared memory, stack/local memory, and spills;
- achieved occupancy, eligible/active warps, issue utilization, and scheduler stalls;
- DRAM/L2 throughput and hit rates;
- Tensor Core utilization where applicable;
- launch and tail underfill at batch-one M=1 geometry.

Use kernel filters and bounded launch selection rather than profiling a complete 256-token run. Profiler results are
diagnostic and never substituted for uninstrumented end-to-end timing.

### L05 — admit at most two implementation candidates

A candidate is admitted only when it accounts for at least 5% of steady short-context time or has a profile-backed
recoverable bound of at least 2% end to end. L03/L04 decide the order. Plausible families include:

1. a smaller M=1 FP8 projection or Q/K/V boundary path;
2. a smaller M=1 NVFP4 Gate/Up or Down path;
3. output-head or greedy-selection work, only if its measured share is material;
4. graph/API synchronization removal, only if the timeline exposes a removable wait;
5. short-context attention, only if it remains material after projection and MLP attribution.

This list is not pre-approval. Do not repeat rejected T=1 Tensor-Core attention, ordinary Gate/Up fusion, rounded
projection epilogues, prefill CUDA Graphs, or verifier-suffix graph experiments without a new profile-backed
hypothesis that changes their limiting resource.

Deliverable: append a candidate admission table to this document or a phase summary with measured share,
hypothesis, expected resource change, numerical boundary, memory delta, tests, and promotion threshold.

### L06 — implement and screen one candidate at a time

For each admitted candidate:

1. retain the exact clean parent and its adjacent benchmark;
2. implement one coherent change with focused reference tests;
3. inspect compiler resources and SASS;
4. run focused correctness, CUDA memcheck/racecheck where applicable, then full host/CUDA CTest;
5. run a one-warm-up/three-measurement rejection screen at contexts 1, 128, and 512;
6. remove the candidate immediately if it loses or is numerically incorrect;
7. if positive, run adjacent parent/candidate 3/10 qualification and the 16K regression workloads;
8. retain or remove the implementation before beginning the next candidate.

No selector zoo, second persistent weight layout, diagnostic fallback, or losing implementation may remain.

### L07 — qualify, document, and close

Run the complete short-context matrix, the 16K ordinary/D2 regressions, full correctness gates, Windows compilation
regression, direct memory-reserve record, and final profiling attribution. Update the README only with the current
qualified state and without historical narrative. Record all promoted and rejected work in the Performance Ledger
and material execution decisions in Decisions.

The closure statement must distinguish:

- gem16 product-boundary improvement;
- standardized llama-bench shape comparison;
- remaining format, token, and timing-boundary differences;
- any gap that cannot be recovered without changing a higher-priority correctness or semantic contract.

## 7. Candidate promotion gates

A production candidate is promotable only when all applicable conditions hold:

- exact generated IDs/checksums remain unchanged for ordinary gem16;
- fixed-D2 remains exactly identical to gem16 ordinary Target output;
- no precision, checkpoint, K/V, sampling, context, graph, or timing boundary changes silently;
- no token-loop allocation, filesystem access, JIT, graph capture, or hidden fallback is introduced;
- complete host/CUDA CTest and focused operator/reference tests pass;
- representative sanitizer and instruction/resource evidence pass;
- short-context end-to-end median improves with statistical support;
- context-16K ordinary, fixed-D2, and context-512 prefill do not regress materially;
- direct CUDA-visible free memory remains at least 700 MiB on the declared reference configuration;
- Linux wins are followed by a Windows build/test regression before closure.

If a small apparent win has overlapping intervals, run 30 reported samples. Reject it if the larger result remains
inconclusive. A microbenchmark, reduced kernel duration, or theoretical operation count alone cannot promote code.

## 8. Stop conditions and target

Stop the plan on the first applicable condition:

1. gem16 comes within 5% of the current adjacent llama.cpp `tg128` throughput while preserving its fuller product
   boundary and every regression gate;
2. gem16 improves at least 10% on the short product boundary and no further candidate meets admission thresholds;
3. the two-candidate budget is exhausted;
4. two consecutive admitted candidates fail their end-to-end promotion gate;
5. profiling attributes the residual gap to disclosed semantic/format boundaries that cannot be removed without
   violating project priorities;
6. correctness, quality, memory reserve, or cross-platform stability blocks further work.

The target is a bounded engineering decision, not parity at any cost. Close with the measured residual gap and move
to Gemma 4 26B A4B M00 rather than extending the plan with unprofiled ideas.

## 9. Result layout

Never overwrite evidence. Use:

```text
benchmarks/results/<date>/<git-sha>/blackwell16gb-linux-maxpower-short-decode/
  L00-parent/
  L01-matrix/
  L02-boundary/
  L03-nsys/
  L04-ncu/
  L05-admission/
  L06-candidate-1/
  L06-candidate-2/
  L07-final/
```

Every directory contains exact commands, system/model/toolchain identity, dirty state, raw JSON, telemetry, output
hashes, memory accounting, and profiler reports relevant to that phase.

## 10. Linux start commands

Initialize and verify the environment:

```bash
git pull --ff-only origin main
git status --short
git rev-parse HEAD

test "$(cat /sys/firmware/acpi/platform_profile)" = max-power
systemctl is-active --quiet nvidia-powerd.service
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw \
  --format=csv,noheader,nounits

MODEL="$(python -c 'from tools.hf_cache import default_target_model; print(default_target_model())')"
BENCH=build/Linux/blackwell-release/bin/gem16-bench
LLAMA=build/Linux/llama_cpp/release/bin/llama-bench
GGUF=build/Linux/llama_cpp/gemma4-12b-mixed-q8-nvfp4.gguf
```

Build and test:

```bash
cmake --preset host-debug
cmake --build --preset host-debug -j
ctest --preset host-debug --output-on-failure

cmake --preset blackwell-release
cmake --build --preset blackwell-release -j
ctest --preset blackwell-release --output-on-failure

./benchmarks/baselines/llama_cpp/build.sh
```

Canonical shapes:

```bash
"$LLAMA" -m "$GGUF" -ngl 99 -fa on -ctk f16 -ctv f16 \
  -b 2048 -ub 512 -p 512 -n 128 -r 13 -o json --progress \
  > <unique-result>/llama-standard-raw.json

"$BENCH" prefill --model "$MODEL" --context 512 --kv-cache fp8 \
  --warmups 3 --repetitions 10 > <unique-result>/gem16-pp512.json

"$BENCH" decode --model "$MODEL" --context 1 --tokens 128 --kv-cache fp8 \
  --warmups 3 --repetitions 10 > <unique-result>/gem16-tg128.json
```

Product-relevant matrix:

```bash
for context in 128 512 2048 8192 16384; do
  "$BENCH" decode --model "$MODEL" --context "$context" --tokens 256 \
    --kv-cache fp8 --warmups 3 --repetitions 10 \
    > "<unique-result>/gem16-decode-${context}.json"
done

"$LLAMA" -m "$GGUF" -ngl 99 -fa on -ctk f16 -ctv f16 \
  -b 2048 -ub 512 -p 0 -n 256 -d 128,512,2048,8192,16384 \
  -r 13 -o json --progress > <unique-result>/llama-decode-f16-raw.json

"$LLAMA" -m "$GGUF" -ngl 99 -fa on -ctk q8_0 -ctv q8_0 \
  -b 2048 -ub 512 -p 0 -n 256 -d 128,512,2048,8192,16384 \
  -r 13 -o json --progress > <unique-result>/llama-decode-q8-raw.json
```

The first Linux session may add a checked harness for alternating order, telemetry, sample trimming, and summary
generation. That harness must refuse overwrites, retain raw 13-sample llama.cpp output, and have focused parser and
statistics tests before it owns promoted evidence.
