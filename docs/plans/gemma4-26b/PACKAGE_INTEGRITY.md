# Package integrity and validation

## Validation performed

The repository-integrated plan is checked for:

- non-empty authored Markdown files;
- UTF-8 readability;
- all relative Markdown links resolving to an existing file or directory;
- exactly one milestone file for every `M00` through `M25`;
- required directory indexes, including `references/imp/README.md`;
- file-level SHA-256 checksums;
- authored-file manifest size/hash consistency;
- a clean `sha256sum -c` verification after metadata regeneration.

## Anchors

```text
Danmoreng/gem16@1c4287965d318ba32a68e597f9d7b6678b883376
kekzl/imp@a392904d4216388828d0d56317de046f4ca49627
```

## Files

- [`PACKAGE_MANIFEST.md`](PACKAGE_MANIFEST.md) — human-readable authored-file inventory.
- `PACKAGE_MANIFEST.json` — machine-readable authored-file inventory.
- `SHA256SUMS.txt` — hashes of every final file except the checksum file itself.
- `VALIDATION_REPORT.json` — machine-readable validation results.

## Verify the integrated plan

```bash
python tools/update_gemma4_26b_plan_integrity.py --check
```

After an intentional normative edit, regenerate the metadata from the repository root:

```bash
python tools/update_gemma4_26b_plan_integrity.py
python tools/update_gemma4_26b_plan_integrity.py --check
```

The generated checksums prove package byte integrity. They do not prove that future repository code satisfies the
implementation plan. Each milestone retains separate correctness, quality, memory, provenance, lifecycle and
performance gates.
