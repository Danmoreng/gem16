# imp code adoption matrix

## Decision classes

- **Reference:** read and cite; no code copied.
- **Clean-room reimplementation:** implement the documented contract in gem16 style after recording the source and tests used for comparison.
- **Isolated MIT port:** copy a narrowly bounded file or algorithm with its original notice and explicit modification record.
- **Reject:** incompatible with the product architecture or memory target.

| Area | Default decision | Reason | Promotion gate |
|---|---|---|---|
| Gemma 4 router semantics | Clean-room reimplementation | Small, model-specific and already covered by official reference semantics | Official HF + imp + CPU oracle agree |
| ModelOpt/llm-compressor scale recognition | Clean-room or narrow adapter | High correctness value, modest scope | Byte fixtures and malformed-metadata tests |
| Native T=1 MoE decode | Reference first | gem16 should retain its model-specific row8/K64 plan | Faster on 5080 and exact against M11 |
| Grouped small-M NVFP4 kernel | Optional isolated MIT port | Substantial low-level implementation may be worth reusing | License accepted; layout compatible; no second persistent copy; 5080 win |
| Work-queue and M-tile policy | Clean-room reimplementation | Simple policy can be tailored to actual Gemma routing distribution | Real-prompt and synthetic-skew benchmark |
| Actual-path dispatch recording | Clean-room reimplementation | Small pattern, major observability benefit | Every runtime branch records one canonical path |
| Graph-demotion reason | Clean-room reimplementation | Small pattern, avoids silent graph loss | First reason retained and surfaced in reports |
| Machine-readable perf baseline | Reimplement schema | Process feature, not runtime code | CI lane stable enough for thresholds |
| Settled-evidence ledger | Adopt process | Prevents agent churn and stale priors | Anchors checked in CI |
| Paged KV / prefix cache | Reject for initial product | Extra metadata and architecture not required for batch one | Separate future product decision only |
| Continuous batching | Reject for initial product | Conflicts with first single-session memory target | Separate future product decision only |
| General GraphExecutor/WeightRegistry | Reject | Would erase gem16's strongest specialization boundaries | Not applicable |
| GGUF/general quant dispatch | Reject from critical path | Q4_0 remains a reference or optional backend | M24 only |
| Host expert offload | Reject as production path | Violates all-resident speed/memory goal | Diagnostic only if explicitly approved |

## Porting rule

No imp source enters production code during M00–M13. Correctness must exist first. A later kernel port requires:

1. an owner-approved decision record;
2. exact imp commit and file hashes;
3. copied MIT license and copyright notice;
4. file-level provenance headers;
5. a patch showing gem16 modifications;
6. independent operator tests and compute-sanitizer runs;
7. proof that the port does not introduce a second permanent weight layout;
8. a retained clean-room alternative or reference path for differential testing.
