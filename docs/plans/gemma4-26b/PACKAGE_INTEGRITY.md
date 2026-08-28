# Documentation package integrity

Package manifests and SHA-256 files describe a documentation snapshot. They are not an operational status source and are not part of the normal task reading loop.

For day-to-day work use:

- `../../ACTIVE_DECISIONS.md`;
- `ACTIVE_CONTRACT.md`;
- `FAST_TRACK_STATUS.json`;
- Git commit identity;
- model/compiler/evidence hashes.

Normal host CI runs `tools/update_gemma4_26b_plan_integrity.py --validate-source`. That stable check validates required
files, milestone uniqueness, UTF-8 Markdown, headings and relative links; it deliberately does not compare the
release-snapshot hash indexes. Before publishing or archiving a new documentation package, run the generator and
then `tools/update_gemma4_26b_plan_integrity.py --check`. Do not edit generated metadata manually. Model artifacts,
compiler binaries and release evidence retain their independent locks.
