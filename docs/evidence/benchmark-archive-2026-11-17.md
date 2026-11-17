# Closed benchmark archive — 17 November 2026

The owner authorized curating and publishing the previously uncommitted records,
removing obsolete helpers/duplicates and leaving a clean working tree. This is
an archive/documentation/tooling change; production runtime code is unchanged.
No historical GPU campaign was restarted and no release is published here.

## Current native 12B state

- [Ordinary O/global qualification](gemma4_12b/ordinary-o-global-2026-11-17.md):
  53.17228 tok/s, measured separately with one warmup and three retained requests.
- [MTP with only the optimized Ordinary tail](gemma4_12b/mtp-ordinary-only-2026-11-17.md):
  D2/D3/D4 98.67648 / 100.17579 / 97.05996 tok/s, one warmup plus three per depth.
  Verifier/Assistant are unchanged; small deltas do not establish an MTP gain.
- Runtime is secured by `23da107`; the RAM experiment-storage policy/helper is
  secured by `d789159`. This cleanup adds no new kernel or model change.

## Retained useful qualification and references

| Record | Final interpretation |
|---|---|
| [12B retained fusion/Assistant](gemma4_12b/retained-decode-2026-11-16.md) | Separate qualified patches in `8111804`; no combined package gain claim |
| [12B matched Prefill](gemma4_12b/matched-prefill-2026-11-15.md) | Three-engine chunk curve; original formats/protocol disclosed |
| [12B batch/microbatch](gemma4_12b/prefill-batch-microbatch-2026-11-15.md) | Independent follow-up, retained raw requests |
| [External MTP D3/D4 and rejected DFlash](gemma4_12b/dflash-2026-11-16.md) | MTP reference remains useful; DFlash is closed and weights deleted |
| [vLLM decode-path research](gemma4_12b/vllm-decode-path-2026-11-16.md) | Source-specific diagnostic findings, not a new throughput measurement |
| [26B central llama repeat](gemma4_26b/central-llama-repeat-2026-11-15.md) | Historical cross-engine reference, distinct formats and outputs |
| [26B vLLM characterization](gemma4_26b/vllm-q4-repeat-2026-11-15.md) | INT4 checkpoint/memory/compiler differences remain explicit |
| [26B AP1 measurement](gemma4_26b/decode-ap1-measurement-2026-11-04.md) | Attribution, counters, source/module provenance, scoped sanitizer limits |
| [26B matched replay](gemma4_26b/matched-ordinary-replay-2026-11-05.md) | Diagnostic replay; not primary engine comparison |
| [26B Ordinary follow-up](gemma4_26b/ordinary-decode-followup-2026-11-05.md) | Closed diagnosis, not an uncommitted production patch |
| [Benchmark history](benchmark-history-2026-11-15.md) | Final JPEGs and exact source datasets/renderer; duplicate graphics removed |

## Closed negative trials

| Trial | Disposition and evidence |
|---|---|
| Original five-step 12B sequence | [Closed summary](gemma4_12b/decode-five-steps-2026-11-15.md); later E1/E2 have separate qualification |
| 12B E3 QKV / E4 Down / E5 attention | [Final campaign table](gemma4_12b/pro-campaign-2026-11-15.md); rejected, no runtime candidate retained |
| DFlash 15/8 drafts | Slower than each engine's prior MTP D2; [final raw results](../../artifacts/dflash/2026-11-16/results.json) |
| 26B AP2 flat-QKV | [End-to-end gate failed](gemma4_26b/qkv-flat-grid-2026-11-04.md) |
| 26B AP3 R1 warp norm | [End-to-end gate failed](gemma4_26b/r1-global-warp-normalization-2026-11-04.md) |
| 26B AP4 final feedback | [Primary gate failed](gemma4_26b/final-d2-feedback-2026-11-04.md) |
| 26B AP5 postnorm | [Operator gate failed](gemma4_26b/attention-postnorm-warp-2026-11-04.md) |
| Original N1–N5 campaign | [Original gate results](gemma4_26b/n1-n5-decode-2026-11-05.md); distinguish N1's later owner exception below |

N1's original rejection is historical. The subsequent owner changed its
Ordinary acceptance gate and the [RoPE qualification](gemma4_26b/n1-rope-qualification-2026-11-05.md)
retained that patch. These are successive decisions, not contradictory current
instructions. Likewise, the provisional E2 JSON status predates the later owner
publication of the retained 12B patches; the final publication report governs.
Immutable JSON samples/statuses are not silently rewritten.

## What was removed and preserved

Obsolete candidate builders/integrators/replays, old review-package generators,
DFlash execution helpers/logs and duplicate renderings were removed. Shared
capture utilities and tests needed by accepted norm/Assistant paths remain.
Two rejected-variant provenance files moved out of `src/` into the cleanup
record. Final primary JSON/JSONL/CSV, model/code/tool identities and accepted
qualification records are retained byte-identically.

Four obsolete intermediate measurement files are removed from their former
active locations, but their exact bytes remain losslessly archived in
`artifacts/maintenance/worktree-cleanup-2026-11-17/historical-intermediate-measurements.json.gz`.
The original filenames and hashes are recorded. They are historical screens,
not pending qualification. Progress narratives now point to final results.

The [cleanup manifest](../../artifacts/maintenance/worktree-cleanup-2026-11-17/removed-files.json)
records each removed file/hash. Adjacent receipts record the two deleted DFlash
weight files (2.926 GB total), reference checks, retained-evidence hashes,
validation and the compile-only replay check. No other model was deleted.

## Closed scope and remaining technical questions

The stopped Pro campaign's interrupted combined comparison and unfinished
Prefill diagnosis are not a resumed task or a claimed success. The inherited
recurring Prefill weight-repack issue remains separately documented in the
[retained-patch limitations](gemma4_12b/retained-decode-2026-11-16.md).
Historical task contracts and technical hypothesis lists are reference material;
they do not schedule new experiments. This cleanup changes neither issue.

## Cleanup validation

All 22 related host tests pass. The diagnostic Linux replay driver compiles
and links against the recorded installed llama.cpp revision; compilation
without its source identity fails as intended. No inference was run. New/changed
Python, JSON and JSONL files pass syntax/parse checks; production runtime source
and all 70 retained original evidence files remain unchanged. The three final
JPEGs match their original hashes. Optional rerendering was not run because
Matplotlib is absent; duplicate-format removal changes no plotted values.
