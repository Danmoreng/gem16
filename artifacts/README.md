# Artifact retention policy

New work follows the [testing and evidence rules](../AGENTS.md#testing-and-evidence).
Existing `artifacts/mNN/` directories contain historical milestone acceptance
records, contracts, hash summaries, diagnostic reports and command transcripts.

High-cardinality compiler reports, full compilation manifests, expanded plans,
model payloads, and other reproducible raw outputs belong under
`artifacts/raw/mNN/`. That tree is intentionally ignored by Git. Accepted raw
outputs remain bound by SHA-256, byte size, source/compiler identity, and their
original path in [`raw-evidence-index.json`](raw-evidence-index.json) and the
milestone acceptance record.

Do not remove an accepted hash record when its raw output is pruned. Do not use
an ignored raw report as a CI fixture; tests must use compact checked evidence or
small purpose-built fixtures.

## Historical captures

The [artifact tree at the archive revision](https://github.com/Danmoreng/gem16/tree/9f7f7a6555ac169a9e49580f11824eef0e55fd85/artifacts)
contains the complete September server and Pi runs, November maintenance receipts, closed DFlash trials
and detailed 12B Ordinary and MTP captures. Paths in dated reports refer to that tree. For local replay
with the original paths, use a separate checkout at `9f7f7a6555ac`.
