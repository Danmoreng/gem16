# Fast Track R4 revision notes

## Scope

This revision changes future execution from M06 onward. The owner accepted the streamlined policy on 2026-08-12. M00–M05 plans, accepted evidence and historical records remain intact; superseded future gates are listed below and in `docs/ACTIVE_DECISIONS.md`.

## Main corrections

- Replaced the many-document read requirement with `ACTIVE_CONTRACT.md` plus the current milestone and linked specs.
- Added a machine-readable status source and made human status pages derived views.
- Allowed isolated sub-agent worktrees with explicit file ownership while retaining one integration branch.
- Removed the rule that only one implementation slice may be active.
- Made the Q4_0 in-engine encoder/backend optional instead of blocking M08.
- Selected a provisional NVFP4 tied head for the first complete artifact.
- Removed T=3/T=5 verifier-head work from M07/M16; it now belongs to M25.
- Made M13 the single early quality go/no-go gate.
- Made M18 conditional diagnosis rather than a prerequisite for M14–M17.
- Corrected the M17/M18 dependency contradiction.
- Reduced M09 to one positive 26B slot and explicit rejection of a second slot.
- Moved `/health`, `/metrics` and Studio concerns out of the residency critical path.
- Made M17 rolling integration instead of a final assembly step after all native kernels.
- Allowed M19, M20 and M21 evidence collection in parallel on one frozen artifact.
- Changed M23 into a base-target evidence freeze rather than a full duplicate rerun.
- Separated MTP from vision. MTP is now the final required program target in M25; vision is outside the program.
- Replaced the former combined MTP/vision milestone with one unambiguous MTP-only M25 file.
- Added separate base and MTP maximum-context qualification, with 32K as the MTP minimum and a required 64K attempt.
- Added a clean-worktree preference before expensive conversions and publication runs; bounded diagnostics may proceed earlier when clearly labeled.
- Replaced the former mandatory full Ordinary conversion in M06/M07 with one full QAT path plus small exhaustive tests and sampled diagnostics; full Ordinary/causal attribution is conditional M18 work.
- Added `docs/ACTIVE_DECISIONS.md` as the short owner-accepted policy source; the historical decision ledger remains available for targeted analysis.

## Files that remain historical

- M00–M05 milestone files;
- M00–M05 evidence records;
- the append-only content of `docs/DECISIONS.md` before the R4 decision;
- source snapshots and accepted artifact hashes.

Historical files are not automatically part of the coding-agent read set.
