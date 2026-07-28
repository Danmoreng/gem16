# Gemma 4 12B MTP

## Feasibility conclusion

The MTP weights are **not embedded** in the pinned
`unsloth/gemma-4-12b-it-NVFP4` target checkpoint. Google publishes the compatible drafter separately as
`google/gemma-4-12B-it-assistant`. The first feasibility gate is positive: the official BF16 assistant is directly
loadable from Safetensors, its architecture matches the target, and its 806.54 MiB tensor payload leaves useful
headroom on the 16 GB reference GPU. Performance remains acceptance-dependent and must be proven in gem16 rather
than inferred from Google's “up to 3x” model-card statement.

The assistant is pinned in `models/gemma4-12b-mtp-assistant.lock.json` at commit
`364bd03c9952e5b7da73665ee30c9eccfc408345`.

## Evidence that the target contains no MTP weights

The target `config.json` declares only `Gemma4UnifiedForConditionalGeneration` and its 48-layer
`Gemma4UnifiedTextConfig`. It contains no assistant sub-config, MTP layer count, draft projection, centroid, or
speculative-decoding field. The authoritative target manifest contains 1,389 tensors:

- 1 tied `[262144, 3840]` target embedding/output matrix;
- exactly 48 normal decoder-layer families;
- 1 final target norm;
- 7 modality-only tensors.

No tensor name contains an MTP/draft/assistant family. In particular, the target has no `pre_projection`,
`post_projection`, assistant `[262144, 1024]` output matrix, or extra decoder layers. Tokenizer vocabulary strings
such as “draft” and “assistant” are ordinary vocabulary entries and are not model components.

## Official assistant contract

The pinned assistant config declares:

| Property | Value |
|---|---:|
| Architecture | `Gemma4UnifiedAssistantForCausalLM` |
| Model type | `gemma4_unified_assistant` |
| Target/backbone hidden size | 3,840 |
| Assistant hidden size | 1,024 |
| Assistant intermediate size | 8,192 |
| Assistant layers | 4 |
| Layer pattern | sliding, sliding, sliding, full |
| Query heads | 16 |
| Local/global head dimension | 256 / 512 |
| Shared-KV layers | 4 |
| Vocabulary | 262,144 |
| Maximum positions | 262,144 |
| Storage dtype | BF16 |

The Safetensors file has 48 tensors and 845,713,928 payload bytes:

| Tensor family | Bytes |
|---|---:|
| Assistant embedding/tied LM head `[262144,1024]` | 536,870,912 |
| Four Q-only decoder layers | 285,248,008 |
| Pre-projection `[1024,7680]` | 15,728,640 |
| Post-projection `[3840,1024]` | 7,864,320 |
| Final norm | 2,048 |

Each assistant attention layer has Q, Q norm, and O weights but no K/V weights. The three sliding layers read the
target cache produced by target Layer 46; the full layer reads target Layer 47. The assistant therefore needs no
independent long-context KV payload. Its pre-projection combines the current target token embedding and target
hidden state; the post-projection feeds a 3,840-dimensional state into the next proposal step. The assistant must
retain its own 1,024-dimensional tied LM head because it cannot share the target's 3,840-dimensional output matrix.

The target and assistant tokenizer JSON files are not byte-identical, but their complete 262,144-entry
`token -> id` vocabularies are equal. Runtime tokenization remains owned by the pinned target. The assistant's older
post-processor and missing added `<|video|>` marker are irrelevant to text-only drafting and must not replace target
metadata.

## 16 GB memory estimate

The assistant adds 845,713,928 persistent payload bytes (806.54 MiB), plus alignment, graph, and small activation
workspaces. It does not add a separate KV cache. Adding only the exact payload to measured target peaks gives:

| Target profile | Target peak | Target + assistant payload | Remaining from 16,303 MiB |
|---|---:|---:|---:|
| 8K decode characterization | 9,852 MiB | about 10,659 MiB | about 5,644 MiB |
| 128K QA | 11,022 MiB | about 11,829 MiB | about 4,474 MiB |
| 262,144-position QA | 12,244 MiB | about 13,051 MiB | about 3,252 MiB |

The first allocator proof now loads all 48 tensors into one independent 256-byte-aligned device arena. The exact
845,713,928-byte source payload occupies 845,714,944 arena bytes (806.54 MiB, including 1,016 alignment bytes).
`cudaMemGetInfo` measures an 847,249,408-byte device-memory delta (808 MiB) around that load. A sequential 50 ms
`nvidia-smi` probe at context 128 measures 9,660 MiB total GPU usage for the target-only process and 10,468 MiB
for target plus assistant, also an 808 MiB difference. The original assistant proposal workspace is 289,024 bytes
at context 128. Active batched MTP adds fixed tentative per-layer K/V and five-row output-selection storage: total
assistant-plus-verifier workspace measures 2,213,376 bytes with FP8 KV and 7,374,336 bytes with BF16 KV at context
128. The FP8 assistant workspace reserves the larger of its reference-score and split-online requirements. The
GPU acceptance result, stop-token table, and committed-hidden row are fixed workspace regions. No MTP graph pool
is retained: an exact 48-layer suffix-graph candidate added 6–8 MiB without improving 16K D2 throughput and was
removed.

## External runtime characterization

vLLM 0.25.1 recognizes the official assistant directly as `Gemma4MTPModel`, loads it beside the pinned Unsloth
target, and maps its sliding/full layers to target Layers 46/47. It reports 10.07 GiB model loading for the pair.
This confirms checkpoint-level compatibility without conversion. The unmodified graph path fails because
suppression-token list indexing creates a CPU index tensor during CUDA capture. The bounded patch in
`benchmarks/baselines/vllm/patches/gemma4-mtp-suppress-graph.patch` uses two scalar indices and enables capture.

On the exact 16K Wikipedia prompt with FP8 KV, graph D1/D2/D4 screens reach 49.59/58.69/56.06 effective tok/s. D2
then reaches 57.390 median tok/s over 3 warm-ups and 10 measured runs. Its 513 verifier groups accept 556 drafts
(mean accepted drafts 1.084), have 36.24 ms median group latency, and peak at a sampled 14,166 MiB. A fixed-1,135-token
screen reaches 57.363 tok/s and 35.75 ms/group. vLLM ordinary and MTP remain separately deterministic but are not
internally token-identical: they first differ at index 33 with stop semantics and index 2 at fixed length. These
results are therefore performance-headroom evidence, not an exact MTP baseline.

llama.cpp's dedicated Gemma 4 assistant path is also functional. The pinned converter produces an 861,520,160-byte
BF16 GGUF from the official assistant; runtime logs prove target-Layer-46/47 K/V sharing and full target/assistant
GPU residency. Current upstream `da5b4486` reaches 48.38 D2 tok/s in the fixed-1,135-token screen and 50.21/49.75
D2/D4 tok/s under stop semantics. Its target uses the patched closest-parity GGUF with BF16-mapped attention and
Q8_0 KV, and ordinary first differs from D2 at fixed output index 133, so it is likewise not an exact or
format-parity baseline.

The hardware conclusion is bounded. gem16's exact D2 groups take about 52.98 ms at mean accepted drafts 1.259;
60 tok/s requires at most 37.65 ms/group at that acceptance. vLLM demonstrates 35.75 ms/group with a numerically
different target batch route, while current llama.cpp remains around 48–50 tok/s. Thus 60 tok/s is physically
plausible on this GPU, but not yet demonstrated by any internally exact comparator on this workload. Further gem16
work should be one bounded exact-verifier sprint; non-exact causal/batched routes remain inadmissible.

Earlier eager probes remain useful acceptance evidence: a random-token prompt reached mean acceptance 1.22 and
about 37.1 tok/s, while a natural CUDA essay reached 2.29 and about 65 tok/s. Together with the Wikipedia matrix,
this confirms that ordinary fallback remains mandatory.

## Correctness command

```bash
gem16-run \
  --model models/checkpoints/unsloth-gemma-4-12b-it-NVFP4-b1f6497 \
  --assistant-model models/checkpoints/google-gemma-4-12B-it-assistant-364bd03 \
  --mtp-draft-tokens 4 \
  --input-token-ids 2,9259,107 \
  --max-context 128 --max-tokens 16 --kv-cache fp8 --greedy
```

The independent bounded reference gate is:

```bash
third_party/cache/unsloth-nvfp4-env/bin/python tools/validate_mtp.py \
  --binary build/Linux/blackwell-release/bin/gem16-run \
  --target models/checkpoints/unsloth-gemma-4-12b-it-NVFP4-b1f6497 \
  --assistant models/checkpoints/google-gemma-4-12B-it-assistant-364bd03 \
  --draft-tokens 4 --output /tmp/mtp-reference.json
```

Active MTP currently supports greedy `gem16-run` generation only. It performs fixed-shape batched exact target
verification for draft lengths 1, 2, and 4, retains tentative K/V rows in a fixed workspace, and accepts and commits
the exact prefix on GPU. Drafts remain device-resident through verification; one small pinned result and one host
synchronization remain per group for output callbacks and variable-length scheduling. `--mtp-adaptive` explicitly
enables context/acceptance-based D4→D2→D1 selection and bounded ordinary decode fallback. Sampling, chat sessions,
diagnostic dumps, and full position-controlled MTP CUDA Graphs remain deliberately rejected or deferred rather
than silently using incorrect semantics.

## Implementation status and order

1. **Complete:** extend the inspector/config parser for `gemma4_unified_assistant` and its exact 48-tensor BF16
   manifest. Primary inference rejects an assistant passed as the target instead of entering target-only code.
2. **Complete:** verify the pinned assistant with `tools/fetch_model.py --lock
   models/gemma4-12b-mtp-assistant.lock.json`.
3. **Complete:** add a separate fixed-address BF16 assistant arena without quantization or conversion. Every
   tensor is bound at its exact shape, and device prefix/suffix probes cover all 48 uploads. `gem16-run
   --assistant-model` reports source, arena, and measured device-delta bytes. A 16-step paired residency run retains
   identical ordinary target IDs, memcheck reports zero errors, and Nsight places all five `cudaMalloc` calls before
   the prefill range with none in prefill or decode.
4. **Complete:** bind the three sliding assistant layers to the target Layer-46 cache and the full assistant layer
   to target Layer 47, for both checkpoint-FP8 and BF16 cache modes. Draft iterations keep the target position and
   shared cache constant, matching the official proposer contract.
5. **Complete:** implement BF16 target-embedding/pre-projection, four Q-only attention/MLP layers, final norm,
   post-projection feedback, the tied 1,024-dimensional LM head, and the assistant's two suppressed token IDs.
6. **Complete for correctness:** implement draft lengths 1, 2, and 4 with fixed-shape batched exact target
   verification. The target evaluates `[input token, draft_1, ...]` in one causal batch; target predictions, not
   assistant agreement, choose every emitted token. Tentative per-layer K/V rows are retained in a fixed workspace,
   local-ring writes are restored before host acceptance, and only the accepted prefix plus first mismatch is
   committed to the target cache.
7. **Complete for correctness:** report proposed/accepted/rejected token counts, proposed IDs, verification groups,
   evaluated target positions and batches, mean accepted length, effective output tok/s, and incremental memory.
   FP8 and BF16 runs retain exact ordinary greedy output, including a local-ring wrap test. `tools/validate_mtp.py`
   proves the first four recurrent BF16 drafts exactly equal Transformers (`1884,5745,993,236771`) on a complete
   one-token shared cache. Active full-process memcheck passes. Full-process initcheck is not accepted as MTP
   evidence because it reports pre-existing uninitialized padded target-prefill reads in CUTLASS
   `ApplyScalesKernel` before proposal execution; full-process racecheck exits before application output with no
   displayed hazards. Targeted assistant sanitizer coverage remains required before performance promotion.
8. **Complete first verifier optimization:** recursive assistant token selection remains device-resident through a
   draft group, and exact fused native Gate/Up/GELU reuse replaces separate MTP Gate and Up launches. The initial
   FP8 CUTLASS batch promotion passed the bounded context-512 gate but was later narrowed to O only after Q/K/V
   failed the full 16K exactness gate. On the natural 53-token/256-output context-512 workload, 3 warm-ups plus 10
   alternating runs measure 42.90 tok/s MTP versus 35.27 ordinary (+21.6%), with mean accepted length 1.89. This is a workload-specific effective-throughput win, not a general 60 tok/s claim.
9. **Complete bounded scheduler work:** drafts now remain device-resident, GPU acceptance applies stop IDs and
   commits tentative K/V plus the selected hidden row before one compact D2H result, and `--mtp-adaptive` exposes
   deterministic context/acceptance thresholds with ordinary fallback. An exact fixed-shape graph over each
   layer's position-independent verifier suffix added 6–8 MiB but measured 35.291 versus 35.340 tok/s and was
   removed. Full position-controlled graph capture is deferred because the refreshed profile is GPU-kernel-bound.
   The exact direct decode-attention verifier remains mandatory: a faster causal-prefill attention candidate
   reached 55.06 tok/s but changed output at step 15 and was removed. NVFP4 CUTLASS verifier projections also
   changed the natural sequence and were removed.
10. **16K correctness and short-batch kernels complete:** target verification keeps decode-order direct grouped
   Q/K/V and exact CUTLASS O. For T≤5, FP8 Q/K/V now uses the latency-oriented decode MMA over 2/3/5 rows instead
   of staging a 128-token tile. NVFP4 Down uses one unstaged 16-token tile and four warps instead of the prefill
   128-token/8-warp plan. Both preserve MMA K accumulation and all 1,135 ordinary IDs. D2 screening moves from
   35.340 to 39.150 after FP8 Q/K/V and to 43.200 tok/s after NVFP4 Down. T1 Gate/Up, 8/16-head global attention,
   and suffix graphs were exact but did not win and were removed. Active FP8 ring-wrap memcheck reports zero errors.
11. **Qualified 16K D2 win:** three alternating warm-up pairs and ten alternating measured pairs produce 31.798
   ordinary versus 42.639 MTP D2 median tok/s, a 1.341x throughput speedup (+34.1%). All 20 measured outputs contain
   the same 1,135 IDs; mean accepted length is exactly 1.259 in every MTP run. The 95% mean CIs are
   `[31.783,31.806]` and `[42.623,42.658]`. Peak sampled GPU memory is 10,838 MiB. D2 remains workload-dependent;
   explicit ordinary decode and adaptive fallback are retained. The 60 tok/s stretch target is not yet reached.
12. **External feasibility matrix complete:** patched graph-vLLM reaches 57.390 D2 tok/s over 3/10 runs and current
   llama.cpp reaches 48.38 tok/s in a fixed-length D2 screen. Both execute the official assistant with shared target
   KV, but neither preserves its own ordinary greedy sequence, and llama.cpp also changes target/KV formats.
   vLLM's 35.75 ms verifier-group latency proves sufficient hardware headroom for 60 tok/s at gem16's
   acceptance, not an admissible numerical implementation. Permit one final bounded exact-verifier sprint; if it
   cannot materially close the 52.98-to-37.65 ms/group gap, retain 42.639 as the exact result and proceed to
   multimodal.
