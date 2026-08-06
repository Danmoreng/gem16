# M01 Unsloth vLLM cold-start OOM incident

Date: 2026-08-06

Scope: diagnostic reference-runtime startup only; no gem16 runtime or CUDA production path was involved

Result: initial unbounded attempt failed; bounded follow-up completed with token-deterministic but logprob-nondeterministic diagnostic evidence

## Attempted reference configuration

The local Unsloth NVFP4 checkpoint at revision
`20df0542b1a86ce19f495ac2eca2c7c12bce82f9` was opened with pinned vLLM 0.26.0, batch one, a 1,024-token maximum,
6 GiB diagnostic CPU offload, `gpu_memory_utilization=0.78`, BF16 KV, eager execution, no prefix cache and all
multimodal limits set to zero. This configuration was diagnostic and ineligible for performance claims.
`trust_remote_code` was false.

vLLM loaded the 15.75 GiB checkpoint in 4.81 seconds, reported 6.05 GiB of CPU-offloaded parameters and placed
9.16 GiB on the GPU. Although eager mode disabled Torch Inductor and CUDA Graphs, it did not disable FlashInfer's
custom fused-MoE JIT. FlashInfer 0.6.14, pinned to source commit `19f1a41e6b21f0c422d775e377b6fdf9a1fc9d23`, selected its CUTLASS NVFP4
backend and generated a large Ninja build for `fused_moe_120`.

## Confirmed cause

The reference host has approximately 62 GiB physical RAM and no swap. The generated Ninja build inherited
unbounded default parallelism and launched many `nvcc`/`cicc` processes:

- at the first global OOM event, the vLLM Python process held 8.6 GiB RSS;
- 28 concurrent `cicc` processes held 44.0 GiB RSS in aggregate;
- later OOM snapshots recorded individual `cicc` processes growing to 7.2 GiB RSS;
- the systemd session scope reached 60.8 GiB;
- the kernel first killed unrelated desktop-session processes because of their OOM scores, then Python and
  compiler processes;
- compiler descendants remained after Python was killed, prolonging the OOM cascade until a manual REISUB reboot.

The root cause was therefore host-memory exhaustion from parallel CUDA JIT compilation, not the 14,300 MiB VRAM
boundary. VRAM polling, eager mode and a process timeout did not bound host RAM or guarantee descendant cleanup.
The REISUB journal recorded emergency sync and read-only remount before reboot.

## Remediation

The incomplete 13 MiB `fused_moe_120` FlashInfer cache was removed. Commit `f901044` adds
`tools/run_isolated_vllm_reference.sh`, which runs the complete future reference process tree in a transient
systemd user service with:

- `MemoryHigh=40G` and `MemoryMax=45G`;
- `MemorySwapMax=0`;
- `OOMPolicy=kill`, which was directly verified to set cgroup-v2 `memory.oom.group=1`;
- `KillMode=control-group`, `TasksMax=128`, and a three-hour runtime limit;
- `MAX_JOBS=4` and `FLASHINFER_NVCC_THREADS=1`.

The observed worst-case accounting is approximately 37.4 GiB for four 7.2 GiB `cicc` processes plus the 8.6 GiB
Python process, leaving cgroup headroom and approximately 17 GiB outside the service. Exceeding the cgroup limit
may still fail the reference build, but it must not be converted into a global host OOM by orphaned compiler
children.

The wrapper passed unit tests, a dry run and a real `/usr/bin/true` transient-service smoke test.

## Controlled follow-up

The owner subsequently authorized one bounded reference investigation. The cold `gpu_memory_utilization=0.78` run
completed the full four-job FlashInfer build and saved 66 autotune configurations without a host OOM. Its cgroup
peaked at 35,722,186,752 bytes, at most four `cicc` processes ran concurrently and no cgroup OOM event occurred.
The outer VRAM monitor stopped it when sampled allocation reached 14,560 MiB, above the 14,300 MiB project stop;
no output from that run was retained. The completed `fused_moe_120.so` and autotune cache were valid and had no
remaining Ninja lock.

Warm-cache runs used `gpu_memory_utilization=0.70`, reducing KV allocation from 2.64 GiB to 1.39 GiB. The final
supported configuration kept chunked prefill enabled, used one explicit unretained warmup and captured two retained
runs. It stayed within all boundaries:

- sampled peak VRAM: 11,874 MiB;
- cgroup memory peak: 11,222,122,496 bytes;
- minimum globally available host memory: 50,131,788 KiB;
- no OOM, hard stop, compiler residue, worker residue or retained GPU allocation.

The final capture process wrote the complete JSON and returned status 0. Its vLLM EngineCore child did not finish
implicit interpreter-shutdown cleanup within systemd's 30-second stop deadline, so the transient service itself
reported a post-output cleanup timeout and removed the remaining control group. No process or GPU allocation
survived. The checked capture tool now calls the pinned V1 `EngineCore.shutdown()` explicitly for future runs; the
model was not rerun merely to re-demonstrate teardown.

Runtime logs prove `CutlassFP8ScaledMMLinearKernel` for FP8 attention projections,
`FlashInferCutlassNvFp4LinearKernel` for NVFP4 GEMM, `FLASHINFER_CUTLASS` for NVFP4 MoE and Triton attention. All
three FlashInfer autotune families were cache hits. The first request in each new engine still JIT-compiled three
Triton support kernels.

The supported warmup and both retained runs generated `[7676, 236761]` (`"OK."`). The two retained token IDs and
text match exactly, but their Top-20 sets and logprobs do not; the evidence therefore has status
`diagnostic_reference_token_deterministic`, not a deterministic-logit golden. This differs from the QAT-BF16 and
official-Q4_0 result `[7676, 106]`. One earlier unsupported run with manually disabled chunked prefill is retained
only under ignored `build/m01/` diagnostics and is not accepted as evidence.

## M01 consequence

The locked Unsloth checkpoint, full tensor inventory, quantization metadata, tokenizer parity and direct token-level
runtime output are now available. [`unsloth-nvfp4-reference.json`](../../../benchmarks/goldens/gemma4_26b/unsloth-nvfp4-reference.json)
retains the warmup, both runs and their full Top-20 distributions. It is diagnostic, CPU-offloaded and explicitly
ineligible for performance claims. Later quality comparisons may use the repeated token sequence but must not treat
its varying logprobs as an exact numerical oracle.
