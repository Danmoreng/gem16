# Gemma 4 26B GEM16 Hugging Face layout

Status: normalized repository layout published and anonymously verified
2026-09-04; product locks intentionally remain on the preceding immutable
revision until acquisition and loader compatibility are verified.

Repository: `danmoreng/gemma-4-26B-A4B-it-GEM16`

- normalized revision: `5a7a0225b3f23067e082c21312a4c38676cc237f`;
- currently runtime-pinned revision: `6de2a057f11332420819f8e6efd08e42d7a03bc7`;
- source Target: `google/gemma-4-26B-A4B-it-qat-q4_0-unquantized` at
  `f1e06dc520982d9b9edd76859fdb7ab209449949`;
- source Assistant: `google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant`
  at `9537141506fe8875b3ed45b264af13580cb29166`.

## Normalized components

The new revision has one descriptive payload per component:

| Role | Payload | Exact format | Bytes |
|---|---|---|---:|
| Public text Target | `components/gemma-4-26b-a4b-it-trellis35-w4a8/gemma-4-26b-a4b-it-trellis35-w4a8.gem16` | GEM16 Trellis35 W4A8 direct-load image; mixed K3/K4 routed experts at approximately 3.5 bpw, FP8 activations | 12,204,692,480 |
| Public Vision | `components/gemma-4-26b-a4b-it-vision-fp8-e4m3fn/gemma-4-26b-a4b-it-vision-fp8-e4m3fn.gem16` | FP8 E4M3FN linear weights per output row, BF16 row scales and support tensors | 597,390,648 |
| Optional fixed-D2 Assistant | `components/gemma-4-26b-a4b-it-assistant-hybrid-nvfp4-fp8-bf16/gemma-4-26b-a4b-it-assistant-hybrid-nvfp4-fp8-bf16.safetensors` | NVFP4 group-16 embedding/head/experts, FP8 attention, remaining BF16 tensors | 258,317,280 |
| Internal Target | `internal/gemma-4-26b-a4b-it-target-hybrid-nvfp4-fp8-bf16/gemma-4-26b-a4b-it-target-hybrid-nvfp4-fp8-bf16.gem16` | NVFP4 group-16 embedding/head/experts, FP8 attention, BF16 state/control | 14,696,668,160 |

Vision is therefore FP8, not Q8. Trellis35 is EXL3-derived offline packing,
but the published artifact is the GEM16-native Trellis35 W4A8 device image,
not a generic EXL3 checkpoint. The Assistant is genuinely quantized and is
not a BF16-only model.

Every directory uses the same supporting names:

- `component.json` — human- and machine-readable model, role, quantization,
  payload size and immutable SHA-256 identity;
- `runtime.json` — the compiler-emitted runtime descriptor retained for
  compatibility/provenance;
- `compilation.json` — exact offline transformation and tensor inventory;
- `source.lock.json` — immutable compiler input/output record.

`profiles.json` composes the public Compact Vision profile from text and Vision
and names the Assistant as optional. The qualified NVFP4 Target is retained
under `internal/` solely for regression and rollback; it is not a third normal
Studio product choice.

## Publication and integrity

`tools/reorganize_gemma4_26b_hf.py` created the new commit with server-side Hub
copies from revision `6de2a057...`. It did not download, hash, re-encode or
re-upload the 27.8 GB of model payloads. Before publication it checked the
immutable source revision and the Hub-recorded LFS SHA-256/size identities.
After publication an anonymous Hub query verified the same identities at the
four descriptive paths and confirmed that the generic payload paths were gone.

This offline publication validation is separate from runtime loading. GEM16
does not hash the model payload during each load; acquisition/install performs
integrity verification, then the loader uploads the already compiled direct
image in large transfers without repacking.

The old revision remains permanently addressable and current product locks
still pin it. This prevents a repository-layout cleanup from changing a tested
runtime. A later lock-only migration may map descriptive remote paths back to
the stable local filenames expected by existing loaders, but it must first pass
the normal downloader and loader checks on Linux and Windows.

The audit record is
`artifacts/vision/hf-layout-normalization-2026-09-04.json`.
