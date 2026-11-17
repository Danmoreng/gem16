# RAM scratch for GPU experiments

The owner's2026-11-17 decision makes RAM-backed temporary storage the default
for large profiler traces, SQL exports and diagnostic tensor dumps. On this
Linux PC, `/tmp` is already tmpfs, with32GiB capacity and no swap configured.
Capacity is a limit, not a32GiB reservation. Production model/checkpoint files
and compact permanent evidence stay on SSD. GPU VRAM behavior is unchanged.

Use the helper for a complete capture/export/analysis workflow:

```python
from experiment_scratch import experiment_scratch

with experiment_scratch('ordinary-fp8-o', expected_gib=4) as scratch:
    # Capture to scratch/'trace.nsys-rep'.
    # Export to scratch/'trace.sqlite' and analyze it here.
    # Save primary JSON/CSV, source/model/toolchain hashes and selected compact
    # diagnostic evidence to a new durable result directory before leaving.
    pass
```

From the repository root, a single child process can use the CLI:

```sh
python tools/experiment_scratch.py --label inspect --expected-gib 4 -- nsys export --type sqlite --output '{scratch}/trace.sqlite' /path/to/retained.nsys-rep
```

The CLI passes `TMPDIR` and `GEM16_EXPERIMENT_DIR` to its child and expands the
literal`{scratch}` placeholder without a shell. Its owned run directory is
removed after the child exits, including unsuccessful exits. To use an export
for subsequent analysis, perform both steps inside the same child workflow or
Python context; the single-export example intentionally retains no SQL dump.
`GEM16_EXPERIMENT_TMPDIR` can select another tmpfs root. SSD roots are refused;
there is no silent SSD fallback.

The preflight checks accept planned scratch sizes up to16GiB and require the
planned size plus1GiB free on tmpfs and plus8GiB available host RAM. These are
**startup checks, not a hard per-job quota or a RAM reservation**. Keep GPU
startup/pinned-host memory needs in the budget, serialize large experiments,
and check SSD space for durable results too. The32GiB tmpfs mount limit is
shared by all `/tmp` users. This task does not change system mounts or RAM/swap
configuration.

Do not routinely retain both a large`.nsys-rep` and its uncompressed SQLite
export on SSD. Keep the compact raw report when it is needed for dispatch
provenance, and regenerate the SQL view in RAM. Prefer bounded warmed diagnostic
windows; keep primary throughput measurements unprofiled. Old SQLite files with
rows/schema that current Nsight cannot reproduce exactly are preserved with
verified lossless gzip rather than discarded.

The2026-11-17 cleanup audit is
`benchmarks/results/2026-11-17/disk-cleanup/actions.jsonl`. Each completed action
records original/export/raw-report hashes and original export metadata.
Only untracked, single-link SQLite views with a matching original Nsight report
are eligible. All actual trace tables and rows are checked on re-export; export
time/output-path metadata is retained separately. Primary measurements, model
files, source and raw reports remain intact. For a compressed original, restore
from its adjacent`.sqlite.gz` to RAM instead of exporting a changed view.

RAM scratch disappears on reboot. Persist the deliberately required evidence
before ending the workflow or the daily22:55 save deadline. Leave no scratch
job running past23:00 Berlin time.
