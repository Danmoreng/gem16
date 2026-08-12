# Checkpoint lock schema template

The actual lock is JSON. This document describes required fields.

```json
{
  "schema_version": 2,
  "repository": "owner/repository",
  "revision": "40-character-commit",
  "resolved_at_utc": "YYYY-MM-DDTHH:MM:SSZ",
  "model_card_revision": "40-character-commit",
  "config_revision": "40-character-commit",
  "tokenizer_revision": "40-character-commit",
  "expected": {
    "variant": "gemma4-26b-a4b",
    "architecture": "...",
    "profile": "source-qat-bf16"
  },
  "license": {
    "identifier": "...",
    "accepted_distribution": true
  },
  "files": [
    {
      "path": "config.json",
      "size": 0,
      "sha256": "...",
      "git_oid": "...",
      "lfs_oid": null,
      "xet_hash": null,
      "source": {
        "repository": "optional/override",
        "revision": "optional-full-commit",
        "path": "optional/path"
      }
    }
  ]
}
```

## Validation

- immutable full revisions;
- safe relative paths;
- unique path;
- positive expected sizes;
- lowercase SHA-256;
- required source identities;
- no unexpected remote code;
- model config/tensor file/tokenizer assets complete.

## Derived lock extension

A derived artifact lock also records:

```json
{
  "derived_from_lock_sha256": "...",
  "compiler_commit": "...",
  "compiler_config_sha256": "...",
  "artifact_profile": "sm120-text-hybrid-v1",
  "head_format": "nvfp4-group16-divisor-v1"
}
```
