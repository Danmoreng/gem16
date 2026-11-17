# Experiment storage cleanup —2026-11-17

The owner requested SSD recovery, RAM-backed temporary SQL/profile exports,
and then explicitly approved deleting the selected old gem16 profiler raw data
before2026-11-17, including associated compressed SQLite originals.

Our own audited actions reclaimed28.528GB (26.569GiB). Another agent's separate
cleanup is not included in that number. SSD free space after these actions is
about80.8GB (75.3GiB). The original177 old-profile selection from the other
agent is fully absent; the last approved purge removed157 remaining files and
newly compressed old SQLite views,10.838GB. Prior verified regeneration/lossless
compaction accounts for the rest of our reduction. Original export metadata and
action/source hashes are retained, but the explicitly purged old detailed traces
are no longer available for independent reanalysis without a new measurement.

The purge metadata-checks21,826 protected primary/model/current-day evidence
files: none changed. Models, source, numerical `.bin` fixtures, JSON/CSV primary
samples and all today's retained raw/compact evidence stay in place. No model
precision or runtime behavior was changed by this storage work.

`/tmp` already uses32GiB tmpfs on the62GiB-RAM host; no administrator mount or
swap change was needed. `tools/experiment_scratch.py` and six host tests provide
RAM/filesystem startup checks and automatic owned-directory cleanup. Actual CLI
child-failure cleanup and SSD-root refusal also pass. The12B vLLM diagnostic
now rejects non-tmpfs trace directories. Details:`docs/EXPERIMENT_STORAGE.md`.

Audit:[`artifacts/maintenance/disk-cleanup-2026-11-17/`](https://github.com/Danmoreng/gem16/tree/9f7f7a6555ac169a9e49580f11824eef0e55fd85/artifacts/maintenance/disk-cleanup-2026-11-17); fuller working records:
`benchmarks/results/2026-11-17/disk-cleanup/`. The compaction process was stopped
before the authorized old-raw purge; no exporter/cleanup job remains active.

The prior Ordinary checkpoint remains941e69e plus raw-log supplementc9b9527.
FP8-O/global-attention work resumes after storage cleanup, with bounded RAM
scratch and fresh one-warmup/three-retained comparison to51.6977tok/s.
