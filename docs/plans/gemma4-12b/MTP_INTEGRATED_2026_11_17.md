# Rejected 12B MTP row/Assistant port experiment — 2026-11-17

This entry reflected the agent's overbroad interpretation of the Ordinary
request. The owner clarified that existing optimized MTP verification and
Assistant must remain unchanged. The active scope is exclusively
[MTP_ORDINARY_ONLY_2026_11_17.md](MTP_ORDINARY_ONLY_2026_11_17.md).

The extra row/Assistant ports were slower and were removed completely from
runtime source. Their immutable measurements, bounded diagnostic traces,
pre-port source hashes and rejected patch remain under
`benchmarks/results/2026-11-17/mtp-integrated/`. They are unqualified diagnostic
history, not the final runtime or a promoted optimization. A proposed operator
check failed to link and was never run; no sanitizer claim applies to those
rejected ports. No further investigation of them is in scope.
