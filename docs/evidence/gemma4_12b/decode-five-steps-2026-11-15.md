# Five bounded 12B Decode experiments — closed

The original 15 November sequence is closed and superseded by the final
[retained-patch report](retained-decode-2026-11-16.md) and
[Pro-campaign closure](pro-campaign-2026-11-15.md). No pending experiment is
authorized by this historical task. The earlier in-progress narrative is removed.

| Original step | Final disposition |
|---|---|
| M16 D2 Gate/Up | Rejected: full D2 +0.188335%, below the 0.5% gate |
| Packed FP8 T3 | Rejected at cache-evicted operator screen |
| Ordinary FP8 unroll | No retained runtime patch from this original trial |
| Norm/residual boundary | Subsequently qualified in the separate E1 campaign |
| Assistant GEMV | Subsequently qualified in the separate E2 campaign |

The initial task contract is [retained](../../plans/gemma4-12b/DECODE_FIVE_STEPS_2026_11_15.md).
Final E1/E2 records, with their own timing/numerical boundaries, take precedence
over preliminary screens. Historical intermediate measurement bytes are
losslessly retained in
`artifacts/maintenance/worktree-cleanup-2026-11-17/historical-intermediate-measurements.json.gz`.
No old measurement is relabeled as a new result.
