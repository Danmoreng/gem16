# MTP in Gem16

## Gemma 4 26B Fast Track

The 26B program now treats MTP as the required final target in M25, after the base target is frozen at M23. Early feasibility work may begin during M06: lock a compatible assistant, inventory its tensors and model BF16/FP8/NVFP4 residency. The 12B assistant is not assumed compatible. Multi-row Target verification and T>1 head kernels belong to M25, not the base head milestones.

The 26B MTP profile must preserve ordinary Target output exactly under matched deterministic controls, commit tentative state transactionally, allocate nothing in the token loop and pass 32K with the normal 700 MiB margin. It must then attempt 64K and publish a separate measured `mtp_max_context`. Vision is not part of M25.

Status: fixed D1/D2/D4 CUDA-Graph verification and sampled Target selection are implemented; D2 is the selected
product profile. The 258,313,728-byte hybrid Assistant is loaded beside the frozen Target and reads its K/V cache
without an independent long-context cache. Sampled verification applies the ordinary suppression, repetition,
temperature, top-k, min-p, top-p and seed/step rules to every Target row on the GPU, then transactionally commits
only emitted state. CLI, server and Studio support sampled D2 text chat at 32K. Adaptive 26B MTP, formal retained
sampled timing and the separate 64K MTP context gate remain open, so M25 is not accepted yet. See
`artifacts/m25/sampled-mtp-product.json` and the M25 milestone card.

---

# Gemma 4 12B MTP

Status: implemented with exact Target verification. Greedy MTP is qualified; sampled MTP has completed correctness
and repeated-timing gates, with publication-grade resource telemetry still open.

## Checkpoints

The pinned target checkpoint does not contain draft weights. The compatible official assistant is loaded from a
separate direct Safetensors snapshot:

- target: `unsloth/gemma-4-12b-it-NVFP4` at `b1f649734b34aa5575b03d186abd1b9be3d0d5c4`;
- assistant: `google/gemma-4-12B-it-assistant` at `364bd03c9952e5b7da73665ee30c9eccfc408345`;
- assistant lock: `models/gemma4-12b-mtp-assistant.lock.json`.

The assistant has 48 BF16 tensors and 845,713,928 payload bytes. Its contract is:

| Property | Value |
|---|---:|
| Architecture | `Gemma4UnifiedAssistantForCausalLM` |
| Target/assistant hidden size | 3,840 / 1,024 |
| Intermediate size | 8,192 |
| Layers | 4 Q-only layers |
| Layer pattern | sliding, sliding, sliding, full |
| Query heads | 16 |
| Local/global head dimension | 256 / 512 |
| Vocabulary | 262,144 |
| Maximum positions | 262,144 |
| Storage | BF16 |

The three sliding layers read the Target Layer-46 local cache; the full layer reads the Target Layer-47 global
cache. The assistant has no K/V projections or independent long-context cache. Its tied 1,024-wide embedding/head,
pre-projection, post-projection, final norm, and Q/O layer tensors remain separate from the Target weights. Runtime
tokenization and generation controls always come from the pinned Target metadata.

## Execution contract

MTP is exact by Target verification. Assistant proposals never directly determine emitted output:

1. the assistant proposes one to four token IDs;
2. one causal Target batch evaluates the current input and proposals;
3. each proposal is compared with the Target-selected token at that position;
4. only the matching prefix is committed;
5. the first mismatch emits the Target token and discards later tentative state;
6. a fully accepted group emits the Target batch's bonus prediction.

Tentative Target K/V and hidden rows live in fixed workspace until a GPU transaction kernel commits the verified
prefix. Proposed tokens are never counted as output tokens. Every result reports proposal, acceptance, rejection,
Target-batch, fallback, and effective verified-token counters.

`--mtp-draft-tokens 1|2|4` selects a direct fixed draft length. `--mtp-adaptive` selects D4/D2/D1 or ordinary
fallback from context and recent acceptance. All modes must preserve the corresponding ordinary Target sequence.

Fixed D2 uses one device-routed conditional CUDA Graph. The device owns current token, position, remaining length,
stop state, sampling step, response-channel state, reasoning budget, and branch selection. D2 handles complete
safe groups; an ordinary Target child handles partial response markers, exact reasoning-budget boundaries, stops,
and short tails. A fixed mapped SPSC ring streams only verified IDs to the host without making generation depend on
a per-group host decision. D1, D4, and the current adaptive route retain the direct transaction boundary and one
host synchronization per verification group.

## Sampled MTP

Sampled MTP reproduces ordinary same-seed Target sampling. For every verifier row, the Target applies the normal
suppression, repetition penalty, temperature, top-k, min-p, top-p, and SplitMix64 output-step mapping to the exact
committed history plus proposal prefix. A proposal is accepted only when it equals that Target sample. RNG,
repetition history, K/V, and hidden state advance by exactly the emitted count.

This is deterministic same-seed verification, not probability-ratio speculative sampling. The assistant does not
materialize proposal probabilities, and no `min(1,p/q)` acceptance claim is made.

## Memory

The assistant source payload occupies a 845,714,944-byte aligned device arena, including 1,016 alignment bytes.
Direct `cudaMemGetInfo` measurement observed an approximately 808 MiB load delta. It adds no assistant K/V cache.
At context 128, assistant-plus-verifier workspace is 2,213,376 bytes with FP8 Target K/V and 7,374,336 bytes with
BF16 Target K/V. Fixed-D2 graph control, tentative rows, transaction records, and streaming storage are all
preallocated. No MTP path may allocate in the token loop or retain a second weight layout.

Current allocator formulas and slot accounting are in [MEMORY.md](MEMORY.md). Admission decisions use directly
measured CUDA-visible memory and the configured safety margin, not nominal board capacity.

## Correctness gates

Required properties include:

- ordinary and MTP token-ID identity within gem16 for the selected greedy or sampled controls;
- exact transactional K/V and hidden-row commit;
- local-ring wrap and global-cache growth;
- D1, D2, D4, adaptive fallback, stop, tail, and reasoning-boundary coverage;
- resident multi-turn continuation without Target-prefix reconstruction;
- zero token-loop allocation and zero silent fallback;
- Compute Sanitizer coverage for active proposal, verification, and transaction kernels.

The independent assistant fixture reconstructs four drafts from captured Target Layer-46/47 state and matches the
official implementation at `[1884, 5745, 993, 236771]`. The fixed 16K gate preserves all 1,135 ordinary Target IDs
under D2. Sampled validators cover multiple seeds, repetition penalties, FP8/BF16 K/V, local-ring wrap, and resident
chat. Detailed numerical evidence is retained in [CORRECTNESS.md](CORRECTNESS.md).

Run the bounded reference gate with:

```bash
python tools/validate_mtp.py \
  --binary build/Linux/blackwell-release/bin/gem16-run \
  --target models/checkpoints/unsloth-gemma-4-12b-it-NVFP4-b1f6497 \
  --assistant models/checkpoints/google-gemma-4-12B-it-assistant-364bd03 \
  --draft-tokens 4 --output /tmp/mtp-reference.json
```

Sampled identity is checked by `tools/validate_sampled_mtp.py` and resident sampled chat by
`tools/validate_sampled_mtp_chat.py`.

## Current performance evidence

The retained 2026-08-03 Linux max-power comparison uses the exact 16,384-token Wikipedia prompt, 1,135 fixed
output positions, batch one, fixed D2, three warm-ups, and ten measurements:

| Engine | Effective D2 tok/s | ITL | Sampled peak VRAM |
|---|---:|---:|---:|
| **gem16 `8e86cb38`** | **89.58** | **11.163 ms** | 11,867 MiB |
| vLLM 0.26.0 | 81.95 | 12.202 ms | 15,465 MiB |
| llama.cpp b10240 | 82.88 | 12.065 ms | 10,631 MiB |

Gem16 is 9.31% faster than vLLM and 8.08% faster than llama.cpp for these recorded configurations. This is not
external semantic parity: all three outputs are internally deterministic but have different hashes, and checkpoint
formats and K/V precision differ. Only gem16 proves identity with its own ordinary Target output. Full commands,
raw-result locations, acceptance counters, telemetry, and caveats are in [BENCHMARKING.md](BENCHMARKING.md),
[PERFORMANCE_LEDGER.md](PERFORMANCE_LEDGER.md), and `benchmarks/baselines/cross_engine_mtp/`.

The older 64.82 tok/s floor remains encoded in `tools/qualify_mtp.py` for historical regression reporting and is
surpassed by the retained result; it is no longer a pending roadmap gate.

## Qualification commands

Use the short screen only to reject candidates:

```bash
python tools/screen_mtp.py \
  --workload benchmarks/results/<workload>/workload.json \
  --output benchmarks/results/<date>/<git-sha>/<machine>/mtp-short-screen.json \
  --executable build/Linux/blackwell-release/bin/gem16-run
```

Promising changes require the full alternating qualification:

```bash
python tools/qualify_mtp.py \
  --workload benchmarks/results/<workload>/workload.json \
  --output benchmarks/results/<date>/<git-sha>/<machine>/mtp-qualification.json \
  --executable build/Linux/blackwell-release/bin/gem16-run \
  --warmup-pairs 3 --measured-pairs 10
```

A promoted MTP change must retain exact ordinary/MTP identity, report effective verified output rather than
proposals, include memory and resource telemetry, and win a representative end-to-end workload. Microbenchmark or
graph-capture improvement alone is insufficient.

## Deferred work

- publication-grade continuous power/clock/thermal telemetry for sampled MTP;
- adaptive D1/D2/ordinary routing inside the chained graph if it wins end to end;
- probability-ratio speculative sampling only if assistant distributions become available and are independently
  qualified;
- an optional bounded N-gram proposal source only after measured hit-rate, acceptance, memory, and latency gates.
