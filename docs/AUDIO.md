# Audio input

## Supported test path

`gem16-chat --message <text> --audio <file.wav>` accepts one audio part after
the text of a one-shot user message. WAV decoding is bounded to 30 seconds and
supports PCM16 or IEEE float32, one or two channels, and sample rates from 8 to
48 kHz. Stereo is averaged and non-16-kHz input is deterministically linearly
resampled to mono 16 kHz. Unsupported formats fail visibly; there is no codec
or precision fallback.

The chat processor pads the final waveform frame with zeros, splits the signal
into 640-sample (40 ms) rows, and permits at most 750 rows. It expands the
single logical audio part to the checkpoint-qualified sequence:

```text
<|audio> <|audio|> repeated once per valid frame <audio|>
```

The repeated placeholder positions are checked against token IDs before
prefill. Each waveform row is rounded to BF16, scale-free RMS-normalized with
epsilon `1e-6`, projected by
`model.embed_audio.embedding_projection.weight` (`[3840, 640]`, BF16), and
written over the normal token embedding before decoder Layer 0. Boundary
tokens retain their normal text embeddings. Audio affects prompt prefill only;
ordinary and MTP-assisted decode use the existing unchanged paths.

## Residency and memory

The primary target loader now uploads all 1,389 checkpoint tensors into one
fixed-address arena. Audio and vision weights are always resident and add
104,759,808 source bytes compared with the former text-only selection. The
legacy `loaded_in_text_only_mode` manifest field remains a tensor provenance
classification and no longer controls runtime residency. Audio scratch buffers
are allocated with the prefill arena, so audio adds no allocation to the token
loop.

## Current boundary

This milestone qualifies audio in one-shot native chat. Multiple in-memory
audio parts are supported by the public `ChatSession` request boundary, but the
CLI exposes one `--audio` file. Interactive `/audio`, additional codecs,
band-limited production resampling, image preprocessing, video, and server
multipart/base64 adapters remain follow-up work.
