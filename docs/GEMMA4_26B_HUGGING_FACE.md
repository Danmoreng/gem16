# Qualified Gemma 4 26B GEM16 repositories

Status: qualified product publication complete, 2026-08-27.

The qualified 26B checkpoint is split because Target and MTP Assistant derive
from different immutable Google repositories:

- Target: `danmoreng/gemma-4-26B-A4B-it-GEM16`, immutable revision
  `b5feb4d146c5ce943160514df0c70a31059885bd`
- Assistant: `danmoreng/gemma-4-26B-A4B-it-assistant-GEM16`, immutable
  revision `a741c642353ccdaefc6f987a3120f434dc9487c7`

Both repositories are initially private. Studio supports authenticated,
resumable downloads through the regular Hugging Face token sources and pins a
full immutable revision plus per-file size and SHA-256. Making the repositories
public later does not change the pinned revision or runtime identity.
Studio embeds the two repository locks and offers the qualified pair as a
first-class model download and selection alongside the protected 12B profile.
The native ImGui app uses `curl` from `PATH`, retains `.incomplete` files for
resume, verifies the final SHA-256, and hardlinks verified blobs into the
immutable Hub snapshots. Authentication comes from `HF_TOKEN`,
`HUGGING_FACE_HUB_TOKEN`, or the normal Hugging Face token file.

## Target package

The Target repository contains `model.gem16`, a 14,696,668,160-byte raw SM120
device-arena image with SHA-256
`1ed73cf105b68db937ac0992283d31fdb2225474204341440721f41fe871bb72`.
Its tensors already use the exact offsets and Row-8/K-64 layouts consumed in
VRAM. Runtime startup uploads this file directly and performs no weight
repacking. The historical source-order Safetensors shards are deliberately not
duplicated in the product repository; `gem16_compilation.json` supplies the
strict tensor inventory and provenance.

The source is
`google/gemma-4-26B-A4B-it-qat-q4_0-unquantized` at revision
`f1e06dc520982d9b9edd76859fdb7ab209449949`. The checkpoint is text-only and
SM120-specific.

## Assistant package

The Assistant repository contains the separately compiled 97-tensor hybrid
NVFP4/FP8 artifact. Its source is
`google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant` at revision
`9537141506fe8875b3ed45b264af13580cb29166`. It is not a standalone chat model
and must be paired with the qualified Target.

## Qualified runtime contract

- batch one and one resident session;
- fixed D2 MTP with Target verification;
- FP8 KV cache;
- `mtp_max_context=73,728` with a 200 MiB long-context reserve;
- `base_max_context=98,304` with the separate 400 MiB base reserve;
- text only; no 26B image, audio or video input;
- no CPU weight offload, runtime repacking, silent precision fallback or
  recurring token-loop allocation.

The measured context record is
`artifacts/m25/context-capacity-2026-08-27.json`. Full GSM8K and AIME 2026,
bounded sampled Target identity and the product/runtime gates are the accepted
quality evidence for this checkpoint. The owner waived the broader historical
M19 suite; quality claims remain limited to the published evidence.

## Reproducing the upload packages

Run `tools/package_gemma4_26b_hf.py` with the accepted M08 Target directory,
the accepted device image and the accepted M25 Assistant directory. The tool
verifies all source lock records and the fixed Target image identity before it
creates hardlinked staging directories. It refuses a non-empty output
directory and never mutates the accepted source artifacts.
