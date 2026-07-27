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

This is sufficient for implementation to proceed, but it is not an allocator proof. The production MTP plan must
measure assistant alignment, verification buffers, proposal state, and CUDA Graph pools at each context tier.

## External runtime probe

vLLM 0.25.1 recognizes the official assistant directly as `Gemma4MTPModel`, loads it beside the pinned Unsloth
target, and maps its sliding/full layers to target Layers 46/47. It reports 10.07 GiB model loading for the pair.
This confirms checkpoint-level compatibility without conversion.

The current vLLM graph path fails during initialization because its assistant suppression-token indexing attempts
a CPU-to-CUDA copy during graph capture. Eager mode avoids that unrelated runtime bug. With four proposal steps:

- a synthetic random-token prompt achieved only 1.22 mean acceptance length and about 37.1 tok/s;
- a natural long-form CUDA essay prompt achieved 2.29 mean acceptance length, deterministic output across runs,
  and 65.14/65.35 tok/s in two measured eager runs;
- the retained ordinary direct-vLLM graph characterization is 39.20 tok/s at short context, but this is not a fair
  same-mode speedup comparison.

The probe demonstrates that acceptance is highly workload-dependent. Independent community llama.cpp evidence
also reports that MTP can be slower for single-stream workloads despite 36–71% draft-token acceptance. gem16 must
therefore retain ordinary decode and promote MTP only where effective accepted output throughput improves.

## Implementation order

1. Extend the inspector/config parser for `gemma4_unified_assistant` and its 48-tensor manifest.
2. Verify the pinned assistant with `tools/fetch_model.py --lock models/gemma4-12b-mtp-assistant.lock.json`.
3. Add a separate BF16 assistant arena; do not quantize or convert the first official result.
4. Bind assistant sliding/full attention to target cache states from Layers 46 and 47.
5. Implement pre-projection, four Q-only layers, post-projection, and the exact assistant LM head.
6. Build a correctness-only proposal/target-verification path for draft lengths 1, 2, and 4.
7. Require output equivalence with ordinary greedy decode and report proposed, accepted, rejected, mean acceptance
   length, effective output tokens/s, and incremental VRAM.
8. Add fixed-address MTP workspaces and CUDA Graphs only after direct execution passes.
9. Promote MTP only for workloads where end-to-end effective throughput wins; otherwise adaptively use ordinary
   decode.
