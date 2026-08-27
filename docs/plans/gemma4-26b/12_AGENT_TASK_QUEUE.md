# Agent task queue — derived view

The machine-readable source is [`FAST_TRACK_STATUS.json`](FAST_TRACK_STATUS.json). Do not edit this page as an independent status system.

## Active queue

M00–M17, M20–M23 and M25 are accepted. The qualified 72K MTP product checkpoint now needs immutable publication and
Studio download integration; full M19 is waived for this bounded checkpoint.

| Priority | Slice | State | Notes |
|---:|---|---|---|
| — | M22 product acceptance | accepted | exact M08 product identity, CLI/server behavior, observability and protected 12B regressions |
| — | M20 controlled performance | accepted | 6,572.809 prompt / 150.615 ordinary-decode retained medians; prompt stretch passes |
| — | M21 real context | accepted | 32K/64K/96K pass twice; 100K rejected; `base_max_context=98,304` |
| — | M23 technical Target freeze | accepted | exact hashes, capability report, M25 rollback and protected 12B regression |
| — | M25 sampled product integration | accepted | fixed D2 CLI/server/Studio chat at `c4ead1d`; same-seed Target identity passes |
| — | M25 context | accepted | 64K and 72K pass at 200 MiB reserve; `mtp_max_context=73,728` |
| 1 | Target/Assistant HF publication | active next | separate immutable repos with source provenance and locked revisions |
| 2 | Studio 26B download | next | authenticated resumable download and qualified profile selection |

Sub-agent packets follow [`PARALLEL_WORKSTREAMS.md`](PARALLEL_WORKSTREAMS.md).
