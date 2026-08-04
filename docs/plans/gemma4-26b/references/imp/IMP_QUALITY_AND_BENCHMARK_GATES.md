# imp-derived quality and benchmark amendments

## Additional candidate/control arms

Add to M18/M19:

| ID | Source/checkpoint | Runtime | Role |
|---|---|---|---|
| G | NVIDIA/ModelOpt Gemma 4 26B NVFP4 | pinned imp or another validated loader | negative/control recipe; reproduce or refute checkpoint-intrinsic quality loss |
| H | UD-Q4_K_M Gemma 4 26B | pinned imp and/or llama.cpp | external quality/speed context; not the official Google QAT reference |
| I | Google QAT BF16 → gem16 NVFP4 with expert-only W4A16 diagnostic | gem16 diagnostic | isolate activation quantization from weight quantization |

G/H are not production candidates for the 16 GB release unless they satisfy the same source, memory and quality gates.

## Corpus strata

Use at least four disjoint corpora:

1. **Plain prose:** primary Gemma 4 PPL and NLL signal.
2. **Code and technical prose:** important product domain, but report separately because distribution shift can dominate absolute PPL.
3. **Instruction/chat/tool traces:** deterministic response and contract checks.
4. **Multilingual subset:** only when intended product claims include those languages.

Never average the strata into one number that hides a severe failure.

## Exact parity requirements

Before comparing PPL or NLL:

- compare tokenizer ID streams byte-for-byte;
- align conditioning prefix and scored logit rows;
- use the same BOS/EOS/template policy;
- disable prompt/prefix caches for one-shot prefill measurements;
- record KV dtype and output-head format;
- retain raw per-token NLL, not only aggregate PPL.

## Required attribution experiments

For every material quality gap, run controlled toggles where supported:

```text
head Q4_0 ↔ head NVFP4 ↔ BF16 diagnostic
FP8 KV ↔ BF16 KV diagnostic
NVFP4 W4A4 ↔ W4A16 diagnostic
own QAT-derived experts ↔ own ordinary-BF16-derived experts
own ordinary-BF16 conversion ↔ Unsloth conversion
own conversion ↔ NVIDIA/ModelOpt control
```

Do not claim “the runtime is wrong” or “the checkpoint is bad” without isolating the same bytes through at least two execution paths or the same execution path with two checkpoints.

## Performance treatment of imp

Imp's published 5090 measurements are useful for:

- identifying viable algorithm families;
- setting a rough 5090 lab expectation;
- designing microbenchmarks;
- finding likely bottlenecks.

They are not a release comparison for the RTX 5080. A headline table may include imp only when:

- exact hardware, commit, model, quant, context and command are shown;
- ordinary decode is compared with ordinary decode;
- MTP/speculation is separated;
- timing boundaries and caches are disclosed;
- wording states that the result is cross-machine or cross-format when applicable.

## Ordinary-decode public matrix

M20 must publish MTP-off greedy decode for cached contexts:

```text
128 / 2K / 8K / 16K / 32K / 64K
256 output forwards
median and p95 ITL
VRAM peak and margin
actual dispatch path
output checksum
```

MTP may be reported only in a separate later table.
