# Working-tree closure — 2026-11-17

Owner-authorized removal of obsolete untracked candidate/DFlash helpers and
duplicate graphics; publication of useful final records and diagnostic tools.
No production runtime changes, new GPU campaigns or release publication.

- removed-files.json: exact paths, sizes and hashes of every removed file.
- retained-original-evidence.json: original retained artifact identities.
- historical-intermediate-measurements.json.gz: a JSON mapping of original
  filenames to exact UTF-8 bytes of four superseded measurement records.
- rejected-provenance.json: metadata formerly orphaned in src, not runtime code.
- deleted-dflash-*.json: separately approved draft-weight deletion receipts.
- host-tests.txt/json: all 22 related host tests pass.
- replay-build.json/txt: compile/link against the recorded locally installed
  central llama revision and required-identity negative compile check. No
  inference or assertion of official-upstream freshness is made by this check.
- graphics-check.json: retained final exports are byte-identical. Matplotlib
  was unavailable for optional regeneration; the new format loop is JPEG only.
- parse-checks.json and repository-checks.json: proportional syntax, source,
  retained-data and staged documentation checks.

Absolute measurement/build paths are historical identities, not dependencies
that a fresh checkout silently creates. See the archive index for final status.
