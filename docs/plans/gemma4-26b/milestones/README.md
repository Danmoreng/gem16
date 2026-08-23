# Milestone cards — Fast Track R4

M00–M07 are accepted milestone plans/cards. M08 onward uses concise active cards with explicit dependencies, parallelism, exit gates and evidence.

Normal path:

```text
M06 → M07 → M08 → M09 → M11/M12 → M13
→ M14/M15/M16 + rolling M17
→ M22 → bounded fixed-target prefill/decode optimization → M21/M20 → technical M23 → M25 → deferred M19 release gate
```

M13 passed with `proceed`, M14–M17 are accepted and M22 product behavior is accepted. Bounded fixed-target
prefill/decode optimization precedes the clean candidate freeze; M21 long-context and M20 performance then qualify
that exact candidate. The owner deferred M19's multi-hour task/prose suite until the end; it remains required
before shipping or production-quality claims.
M18 is a conditional diagnosis lane, not the next mandatory step. M24 is
optional. M25 is [`M25_MTP_INTEGRATION_AND_FINAL_QUALIFICATION.md`](M25_MTP_INTEGRATION_AND_FINAL_QUALIFICATION.md).
The obsolete vision-named M25 file is removed by this revision.
