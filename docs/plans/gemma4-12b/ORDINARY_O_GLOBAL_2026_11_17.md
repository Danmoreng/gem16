# 12B Ordinary FP8 O and global attention —2026-11-17

The owner explicitly requests committing the qualified Ordinary gains first,
then optimizing12B FP8 O projection and global attention and comparing with
one warmup plus three retained requests. Checkpoint941e69e plus log supplement
c9b9527 is complete; no push. This task entry supersedes the earlier
norm/overlap-only tuning scope for these two12B Ordinary T1 operators only.
Storage cleanup/RAM scratch was prioritized and completed first. No26B tuning.

Parent:51.69772785294868tok/s, same16K/1135 fixture, MTP0, batch1, FP8 KV,
resident weights, same timing/sampling/cache policy. Preserve parent binary
and raw samples. New large diagnostic derivatives use checked tmpfs through
`tools/experiment_scratch.py`, expected4GiB plus host-memory reserve. Keep
compact numeric/timing/dispatch/raw primary evidence on SSD; no duplicate
large SQLite exports. Verify Berlin time before long operations.

Try one narrow exact-arithmetic scheduling family per operator first:
FP8 O finer output-row tiles with the existing FP8 MMA K chain/scales, and
12B T1 global GQA groups of four query heads with the same per-head dot,
256-thread softmax tree and V accumulation order. Reuse existing actual12B
FP8/attention tensor fixtures; they were explicitly preserved during cleanup.
Only positive, correct offline variants enter full-model qualification.
Require sanitizer/resource/actual-dispatch evidence, same outputs and no hot
allocations/fallback. Compare with one warmup+three retained; no13-run campaign.
Failed candidates remain diagnostic and the committed parent remains enabled.
No new commit/push is authorized for this subsequent slice yet.

The first row-tile screen is closed without promotion: both variants reproduce
all actual/finite boundaries and pass scoped sanitizers, but local O does not
improve and global O's isolated2.3% is small/noisy. Global GQA4 is exact and
wins both sampled positions by6.6%. For FP8 O, admit one bounded ordered-load
bundle4/8 screen: only load scheduling changes, existing K32 MMA accumulation
order and scales remain exact, no shared/persistent weight repack. Reject spills
or nonpositive roles before full-model comparison. This is12B Ordinary only.

Owner update 2026-11-17: commit and push this completed, qualified state are
now explicitly authorized; this supersedes the prior no-commit/push restriction.
