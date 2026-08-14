# Artifact retention policy

`artifacts/mNN/` retains only compact, reviewable milestone evidence: acceptance
records, immutable contracts/configuration, hash summaries, short verification
or diagnostic reports, and command transcripts.

High-cardinality compiler reports, full compilation manifests, expanded plans,
model payloads, and other reproducible raw outputs belong under
`artifacts/raw/mNN/`. That tree is intentionally ignored by Git. Accepted raw
outputs remain bound by SHA-256, byte size, source/compiler identity, and their
original path in [`raw-evidence-index.json`](raw-evidence-index.json) and the
milestone acceptance record.

Do not remove an accepted hash record when its raw output is pruned. Do not use
an ignored raw report as a CI fixture; tests must use compact checked evidence or
small purpose-built fixtures.
