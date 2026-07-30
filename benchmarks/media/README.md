# Multimodal benchmark media

This directory contains the repository-local media suite used by
`tools/benchmark_server_long_conversation.py`. The benchmark validates every
file against `suite.json` before starting the server. Do not replace a file
without updating its provenance, expected facts, checksum, and benchmark
evidence.

## Images

The three PNG files are original gem16 fixtures generated from deterministic SVG
source strings in `tools/generate_benchmark_images.py`:

| File | Intended visible facts |
|---|---|
| `images/orbit_station_47.png` | desert observatory, `ORBIT 47` |
| `images/cedar_greenhouse_82.png` | greenhouse, `CEDAR 82`, three purple planters |
| `images/harbor_lighthouse_19.png` | lighthouse, `HARBOR 19`, four red sailboats |

They are project-authored test data under the repository's license. The checked-in
PNGs were rasterized at 896x640 with `rsvg-convert`; regenerate them with:

```bash
python tools/generate_benchmark_images.py
```

## Audio

The WAV files are 14-second, mono, 16-kHz PCM excerpts from LibriVox recordings
published with a public-domain mark. The underlying books are also public domain.
The source MP3 is not committed. `tools/fetch_benchmark_audio.py` downloads each
exact source, verifies its SHA-256, and extracts the checked-in segment with
FFmpeg. The retained WAVs were produced with FFmpeg n8.1.2.

| File | Work and recording | Archive.org item | Segment | Recognition facts |
|---|---|---|---:|---|
| `audio/alice_sister_bank.wav` | Lewis Carroll, *Alice's Adventures in Wonderland*, version 8 | [`aliceinwonderland_2106_librivox`](https://archive.org/details/aliceinwonderland_2106_librivox) | 30-44 s | sister, bank |
| `audio/moby_lexicons_flags.wav` | Herman Melville, *Moby Dick; or, The Whale*, version 2 | [`mobydick2_2511_librivox`](https://archive.org/details/mobydick2_2511_librivox) | 35-49 s | lexicons, grammars, flags |
| `audio/pride_wife_neighborhood.wav` | Jane Austen, *Pride and Prejudice*, version 4 | [`prideandprejudice_1005_librivox`](https://archive.org/details/prideandprejudice_1005_librivox) | 30-44 s | wife, neighborhood |

Source rights declarations:

- Alice and Moby Dick recordings: [Creative Commons Public Domain Mark 1.0](https://creativecommons.org/publicdomain/mark/1.0/)
- Pride and Prejudice recording: Archive.org/LibriVox public-domain declaration (`http://creativecommons.org/licenses/publicdomain/`)
- LibriVox states that its recordings are dedicated to the public domain in the
  United States. Users redistributing outside the United States should verify
  local status.

Reproduce the WAV files explicitly with:

```bash
python tools/fetch_benchmark_audio.py --force
```

The fetch tool pins source URLs and source checksums. Generated WAV checksums may
also depend on the exact FFmpeg decoder/resampler version; `suite.json` remains
the authority for the committed bytes.
