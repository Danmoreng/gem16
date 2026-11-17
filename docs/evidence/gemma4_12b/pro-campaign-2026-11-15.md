# 12B Pro campaign — closed after owner stop, 16 November 2026

The owner stopped the campaign and subsequently authorized publishing the two
retained patches. Runtime commit `8111804` contains E1 post-attention fusion
and E2 Assistant exact-tree GEMV. This replaces the unpublished running log.
The [publication report](retained-decode-2026-11-16.md) is the final qualification
and limitation record; no combined package speedup is claimed.

| Experiment | Final disposition | Evidence |
|---|---|---|
| E1 norm/residual fusion | Retained; separate Ordinary +0.708245% | [Qualified record](../../../artifacts/gemma4_12b/norm-fusion-qualified-2026-11-16.json) |
| E2 Assistant exact tree | Retained by later owner publication decision; separate D2 +0.39% | [Qualification](../../../artifacts/gemma4_12b/assistant-exact-tree-2026-11-16.json) |
| E3 QKV load64 | Rejected; full D2 −3.00401% | [Final samples](../../../artifacts/gemma4_12b/qkv-load64-full-2026-11-16.json) |
| E4 Down weight pipeline | Rejected; full D2 −1.68796% | [Final samples](../../../artifacts/gemma4_12b/down-weight-pipeline-2026-11-16.json) |
| E5 global KV preload | Rejected; full D2 −0.01668% | [Final samples](../../../artifacts/gemma4_12b/global-attention-preload-2026-11-16.json) |

These are independent trial results, not additive gains. The original-parent/
final-package comparison was interrupted; bounded Prefill diagnosis was not
completed. The later 17 November Ordinary/MTP work has its own
[separate evidence](mtp-ordinary-only-2026-11-17.md). No old campaign resumes.

The [historical contract](../../plans/gemma4-12b/PRO_CAMPAIGN_2026_11_15.md) and
[owner threshold amendment](../../../artifacts/gemma4_12b/pro-campaign-package-amendment-2026-11-16.json)
remain available. Earlier operator screens are historical measurements in the
lossless cleanup archive, not active qualification status. Candidate integration
helpers and orphaned runtime provenance were removed; final raw values and
identities in the linked JSON records remain byte-identical.

The inherited recurring Prefill weight-repack issue is documented in the
[publication limitations](retained-decode-2026-11-16.md); it is not repaired or
newly qualified by this archival cleanup.
