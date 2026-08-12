# Documentation package integrity

Package manifests and SHA-256 files describe a documentation snapshot. They are not an operational status source and are not part of the normal task reading loop.

For day-to-day work use:

- `../../ACTIVE_DECISIONS.md`;
- `ACTIVE_CONTRACT.md`;
- `FAST_TRACK_STATUS.json`;
- Git commit identity;
- model/compiler/evidence hashes.

After normative plan changes, run the repository-maintained `tools/update_gemma4_26b_plan_integrity.py` generator and verify its output. Do not edit generated metadata manually. Model artifacts, compiler binaries and release evidence retain their independent locks.
