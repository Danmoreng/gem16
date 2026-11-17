# Disk recovery and RAM scratch —2026-11-17

The owner pauses the FP8-O/global-attention work until the SSD usage has been
explained and made safe. The owner additionally requests RAM-backed temporary
SQL dumps, exports and experiment intermediates. This supersedes the earlier
practice of retaining full uncompressed SQLite exports in result directories.
Keep original Nsight reports and primary samples. Verify re-exported trace
rows before removing regenerable SQLite; retain original export metadata.
If data differ, preserve the original losslessly instead of deleting it.

`/tmp` is already tmpfs with32GiB capacity on this62GiB-RAM Linux host; SwapTotal0.
No administrative mount or global machine configuration is required. New
heavy diagnostic intermediates go into a checked `/tmp/gem16-experiments/`
run directory. Retain only compact primary JSON/CSV, source/model/toolchain
identities, qualification summaries and deliberately selected compact trace
proofs on SSD. Never move model checkpoints or compiled production weights
into ephemeral scratch. GPU buffers and pinned-host allocations remain separate.
Use the helper's preflight RAM/filesystem checks and clean its run directory.
These are startup checks, not a kernel memory quota or inference-loop work.
