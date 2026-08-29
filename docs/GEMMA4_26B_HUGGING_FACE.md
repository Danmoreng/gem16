# Qualified Gemma 4 26B GEM16 repositories

Status: public and ungated; metadata refreshed and anonymously verified
2026-08-29.

The qualified 26B checkpoint is split because Target and MTP Assistant derive
from different immutable Google repositories:

- Target: `danmoreng/gemma-4-26B-A4B-it-GEM16`, revision
  `63508b5826527484e707b4b46e2eacf077cf2b35`;
- Assistant: `danmoreng/gemma-4-26B-A4B-it-assistant-GEM16`, revision
  `466cc26d157fad0cc946f094ae904445147c38b4`.

Both repositories are public and ungated. Anonymous API, model-card, Target
weight HEAD, and Assistant weight HEAD requests returned HTTP 200 after the
metadata commits. Studio pins the full revisions above, verifies every file,
and no longer requires or transmits a Hugging Face token for this public pair.

## Target package and freshness

The Target repository contains `model.gem16`, a 14,696,668,160-byte raw SM120
device-arena image with SHA-256
`1ed73cf105b68db937ac0992283d31fdb2225474204341440721f41fe871bb72`.
Its tensors already use the exact offsets and Row-8/K-64 layouts consumed in
VRAM. Runtime startup uploads this file directly and performs no weight
repacking. Source-order Safetensors shards are deliberately not duplicated;
`gem16_compilation.json` supplies the strict inventory and provenance.

The source is
`google/gemma-4-26B-A4B-it-qat-q4_0-unquantized` at revision
`f1e06dc520982d9b9edd76859fdb7ab209449949`. The checkpoint is text-only and
SM120-specific.

The published image is already the newest accepted direct-load layout. Commit
`5abba6e` introduced the direct device-image builder before commit `bd5b1af`
created the Hugging Face packages. The package, retained M08 image, and final
smoke-test image have the same byte count and SHA-256. Later context
qualification changed the MTP limit from 73,728 to 86,016 but did not change
weight bytes. A 14.7 GB re-upload is therefore not required.

## Assistant package

The Assistant repository contains the separately compiled 97-tensor hybrid
NVFP4/FP8 artifact. Its 258,317,280-byte weight file has SHA-256
`4d3ce2102ad0631d9e7e0586be0b108d5789cbc5b90d21b4c50613979228d927`.
Its source is
`google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant` at revision
`9537141506fe8875b3ed45b264af13580cb29166`. It is not a standalone chat model
and must be paired with the qualified Target.

## Qualified runtime contract

- batch one and one resident session;
- fixed D2 MTP with Target verification;
- FP8 KV cache;
- `mtp_max_context=86,016` with a 200 MiB long-context reserve;
- `base_max_context=98,304` with the separate 400 MiB base reserve;
- text only; no 26B image, audio or video input;
- no CPU weight offload, runtime repacking, silent precision fallback or
  recurring token-loop allocation.

The measured context record is `artifacts/m25/context-capacity-2026-08-28.json`.
Full GSM8K and AIME 2026, bounded sampled Target identity and the
product/runtime gates are the accepted quality evidence. The owner waived the
broader historical M19 suite; quality claims remain limited to the published
evidence.

## Model cards, relationship and license

`tools/package_gemma4_26b_hf.py` generates a GEM16-specific preamble and then
retains the body of the exact pinned Google model card. Generated Hub front
matter uses:

```yaml
license: apache-2.0
base_model: google/<exact-pinned-26B-source>
base_model_relation: quantized
```

This is Hugging Face's model-tree relationship for a quantized derivative. It
provides navigation from GEM16 back to Google and allows the GEM16 variant to
appear among derivatives on the upstream page. The preamble makes the narrower
text-only SM120 contract explicit, so inherited upstream multimodal statements
are not accidentally attributed to GEM16.

Each generated package also contains `LICENSE` and a source-specific `NOTICE`
with the Google repository, immutable revision, GEM16 modifications, upstream
terms link, and project link. The classifier is Apache 2.0, matching both
pinned Google QAT repositories; the obsolete `license: gemma` value is removed.

## Reproducing the packages

Fetch both upstream `README.md` files at the exact source revisions, then run:

```bash
python3 tools/package_gemma4_26b_hf.py \
  --target artifacts/raw/m08/qat-hybrid-clean-1 \
  --target-image artifacts/raw/m08/qat-hybrid-clean-1.gem16-sm120-device-image-v1.bin \
  --target-lock artifacts/raw/m08/qat-hybrid-clean-1.lock.json \
  --target-upstream-card /path/to/pinned-target-README.md \
  --assistant artifacts/raw/m25/qat-q4_0-assistant-hybrid-diagnostic-v2 \
  --assistant-lock artifacts/raw/m25/qat-q4_0-assistant-hybrid-diagnostic-v2.lock.json \
  --assistant-upstream-card /path/to/pinned-assistant-README.md \
  --output /path/to/empty/hf-staging
```

The tool verifies every source-lock record and the fixed Target image identity,
creates hardlinked staging directories where possible, refuses non-empty
outputs, and never mutates accepted artifacts.

The 2026-08-29 publication uploaded only `README.md`, `LICENSE`, `NOTICE`, and
`gem16_model.json` in one commit per repository. It did not replace
`model.gem16` or the Assistant Safetensors file. The resulting locks and native
Studio catalog pin the new revisions above.

Keep the 2026-08-27 private publication record immutable. The public release
audit is `artifacts/m25/hf-publication-2026-08-29.json`.

## Current Studio cache behavior

Native Studio resolves its Hub root in this order:

1. `HF_HUB_CACHE` exactly;
2. `$HF_HOME/hub`;
3. `$XDG_CACHE_HOME/huggingface/hub`;
4. `$HOME/.cache/huggingface/hub`;
5. a repository-local fallback.

The 26B snapshots already use canonical Hub directories such as
`models--danmoreng--.../snapshots/<revision>`. The 12B Target currently uses a
custom `.gem16/snapshots/...` directory because its lock assembles files from
more than one upstream repository. On Windows, a process without `HOME` also
falls through to the repository-local directory instead of the normal user
cache.

Both are known catalog-slice defects. The common model catalog must resolve the
actual Hugging Face user cache on Windows and Linux. The 12B Target and
Assistant remain in their existing upstream repositories; the catalog/runtime
must resolve any cross-repository files directly from those canonical
snapshots without publishing a GEM16 mirror or duplicating model blobs.
