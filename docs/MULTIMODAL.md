# Multimodal expansion plan

Status: design and implementation specification; no multimodal runtime path is implemented yet

Target checkpoint: `unsloth/gemma-4-12b-it-NVFP4` at
`b1f649734b34aa5575b03d186abd1b9be3d0d5c4`

Initial product scope: batch-one text generation from interleaved text, image, and audio input on one approximately
16 GB Blackwell CUDA GPU

Follow-up scope: video input as sampled image frames after the image path is qualified

## Objective

Extend the existing text-only Gemma 4 12B Unified engine through the checkpoint's native encoder-free image and
audio paths without replacing or weakening the optimized text transformer. The multimodal implementation must:

- load the original pinned Hugging Face checkpoint directly;
- retain text-only loading as a first-class low-memory mode;
- load only the modality weights selected by the execution plan;
- preserve the checkpoint's BF16 modality weights exactly;
- preprocess media according to the pinned `processor_config.json`;
- replace image, video, and audio placeholder IDs with projected soft-token embeddings before transformer prefill;
- implement the checkpoint's vision-specific bidirectional local-attention mask exactly;
- reuse the existing FP8/NVFP4 transformer, hybrid KV cache, output head, sampling, and decode CUDA Graph;
- perform no media decoding, allocation, filesystem access, or modality dispatch in the generated-token loop;
- account for media tokens in the same context limit and position sequence as text;
- preserve exact media identity when a resident conversation session reuses cached K/V state;
- expose media preprocessing, projection, transformer prefill, memory, and quality results independently;
- fail visibly for unsupported formats, malformed media, missing tensors, or unavailable kernels.

This is an input-only expansion. The model continues to generate text. Image generation, audio generation, speech
synthesis, arbitrary any-to-any output, training, and fine-tuning are out of scope.

## Why the 12B Unified model is the selected path

Gemma 4 12B Unified has no separate vision or audio transformer. It projects raw merged image patches and framed
audio waveform samples directly into the 3,840-dimensional language-model embedding space. Once those embeddings
have been inserted, the existing 48-layer decoder processes them with the same attention and MLP weights used for
text.

This keeps the expansion suitable for 16 GB cards:

- the checkpoint already contains all required modality tensors;
- enabling both modalities adds only 104,759,808 persistent tensor bytes before allocator alignment;
- no second model or separately downloaded encoder is required;
- decode after prompt ingestion is the same text-token decode path;
- video reuses the image embedder and adds no video-specific model weights.

The model capability and format sources are:

- the locked checkpoint artifacts, especially `config.json`, `processor_config.json`, `tokenizer_config.json`,
  `chat_template.jinja`, and the Safetensors manifest;
- Google's Gemma 4 model card:
  <https://ai.google.dev/gemma/docs/core/model_card_4>;
- Google's image preparation documentation:
  <https://ai.google.dev/gemma/docs/capabilities/vision/image>;
- Google's audio preparation documentation:
  <https://ai.google.dev/gemma/docs/capabilities/audio>;
- the exact Transformers implementation pinned for reference-fixture generation.

The locked local artifacts remain authoritative for engine behavior. External documentation must be revalidated
when reference packages change.

## Current boundary

The repository currently:

- parses and classifies the modality tensors but marks them as skipped in text-only mode;
- plans and uploads only the 9,200,026,528-byte text tensor set;
- accepts token IDs or text-only `ChatMessage::content`;
- requires prefill to be causal;
- identifies a resident conversation prefix only by token IDs;
- runs reference generation with all vLLM multimodal limits set to zero;
- has no native image decoder, image processor, audio decoder, resampler, modality projector, or multimodal
  template branch.

These are deliberate current limitations, not evidence that the checkpoint lacks multimodal support.

## Pinned multimodal model contract

### Special tokens

The exact IDs must be parsed and validated rather than inferred from this document:

| Role | Config field | Pinned ID | Tokenizer spelling |
|---|---|---:|---|
| Image soft-token placeholder | `image_token_id` | 258880 | `<|image|>` |
| Audio soft-token placeholder | `audio_token_id` | 258881 | `<|audio|>` |
| Video soft-token placeholder | `video_token_id` | 258884 | `<|video|>` |
| Begin image | `boi_token_id` | 255999 | `<|image>` |
| End image | `eoi_token_id` | 258882 | `<image|>` |
| Begin audio | `boa_token_id` | 256000 | `<|audio>` |
| End audio | `eoa_token_index` | 258883 | `<audio|>` |

Startup must require that each spelling encodes to the declared single ID. Placeholder IDs are control slots, not
ordinary vocabulary embeddings: the runtime substitutes projected modality rows before layer 0.

### Vision configuration

The pinned `vision_config` declares:

| Field | Value |
|---|---:|
| Input patch size | 16 |
| Spatial pooling kernel | 3 |
| Effective merged patch edge | 48 |
| Merged RGB patch width | 6,912 |
| Vision hidden width | 3,840 |
| Text projection width | 3,840 |
| Factorized position table extent | 1,120 |
| Default soft-token budget | 280 |
| RMSNorm epsilon | `1e-6` |

The processor default is 280 soft tokens. The model supports the documented budgets 70, 140, 280, 560, and 1,120;
the actual row count may be lower after aspect-ratio-preserving resize. The engine must not assume every image
produces exactly the selected maximum.

### Audio configuration

The pinned processor declares:

| Field | Value |
|---|---:|
| Required sample rate | 16,000 Hz |
| Samples per soft token | 640 |
| Time per soft token | 40 ms |
| Feature width | 640 |
| Maximum sequence length | 750 |
| Maximum documented clip length | 30 seconds |
| Projection output width | 3,840 |
| RMSNorm epsilon | `1e-6` |

The number of audio rows is:

```text
audio_soft_tokens = ceil(mono_sample_count / 640)
```

The final partial frame is zero-padded. A 30-second clip produces 750 rows. There is no mel-spectrogram,
convolutional downsampling, Whisper model, or Conformer tower in the Unified path.

### Attention behavior

The text configuration declares:

```text
use_bidirectional_attention = "vision"
```

Audio tokens remain causal. Image and video soft tokens receive blockwise bidirectional visibility only in the 40
sliding-attention layers. The eight full-attention layers remain causal.

For a valid local key at absolute position `k`, query position `q`, sliding window `W`, and a non-negative vision
block identifier `block(position)`, the reference mask is equivalent to:

```text
within_left_window = k > q - W
causal             = k <= q
same_vision_block  = block(q) >= 0 && block(q) == block(k)

allow_local(q, k) = within_left_window && (causal || same_vision_block)
allow_full(q, k)  = causal
```

Padding and sequence-validity masks are applied in addition. The sliding overlay is a lower-bound window, not a
symmetric absolute-distance test. Different images, different video-frame blocks, audio spans, and text spans must
never gain bidirectional visibility merely because they are adjacent.

## Authoritative tensor inventory

The modality tensors are excluded from the NVFP4/FP8 quantization groups and stored in BF16. They must not be
silently requantized to reuse an existing low-precision projection kernel.

| Tensor or tensor group | Shape | Bytes | Use |
|---|---:|---:|---|
| `model.embed_audio.embedding_projection.weight` | `[3840,640]` | 4,915,200 | Audio RMSNorm output to text width |
| `model.embed_vision.embedding_projection.weight` | `[3840,3840]` | 29,491,200 | Vision RMSNorm output to text width |
| `model.vision_embedder.patch_dense.weight` | `[3840,6912]` | 53,084,160 | Merged RGB patch projection |
| `model.vision_embedder.patch_dense.bias` | `[3840]` | 7,680 | Patch projection bias |
| `model.vision_embedder.patch_ln1.{weight,bias}` | two `[6912]` tensors | 27,648 | Pre-projection LayerNorm |
| `model.vision_embedder.patch_ln2.{weight,bias}` | two `[3840]` tensors | 15,360 | Post-projection LayerNorm |
| `model.vision_embedder.pos_embedding` | `[1120,2,3840]` | 17,203,200 | Factorized two-axis positions |
| `model.vision_embedder.pos_norm.{weight,bias}` | two `[3840]` tensors | 15,360 | Post-position LayerNorm |

The totals are:

```text
audio-only additional tensors  =   4,915,200 bytes
vision additional tensors      =  99,844,608 bytes
vision + audio                 = 104,759,808 bytes
all checkpoint tensor payload  = 9,304,786,336 bytes
```

Video uses the vision tensor set. `gem16-inspect` must retain text-only classification for compatibility and add
explicit capability/residency totals for text, image/video, and audio. Execution code must bind these names through
the manifest and reject shape or dtype changes.

## Target request and execution model

### Gemma-specific content parts

The current string-only message representation must evolve into an ordered list of Gemma content parts. This is a
model-specific request boundary, not a generic tensor graph or arbitrary Transformers abstraction. A suitable
logical shape is:

```text
GemmaContentPart =
  TextPart{text}
  ImagePart{decoded RGB image or validated local source}
  AudioPart{decoded mono waveform or validated local source}
  VideoPart{sampled frames plus timestamps}       # follow-up

GemmaChatMessage{role, ordered content parts}
```

The exact C++ names may differ, but the interface must preserve interleaving and must not collapse media into
display strings. The core engine should accept validated decoded buffers. Local-file decoding can be a CLI or
processor facility above that boundary. Network URL fetching is not part of the native runtime.

### Compiled prompt plan

The processor should compile content parts once into an immutable prompt plan containing at least:

- rendered hard token IDs;
- one modality type per sequence position;
- contiguous vision block IDs or enough metadata to derive them;
- placeholder ranges and expected projected-row counts;
- image patch tensors and two-axis position IDs;
- audio frame tensors and valid-row masks;
- absolute sequence positions;
- media identity records;
- selected token budget, sample rate, preprocessing revision, and source ordering;
- total context positions, including delimiters and media soft tokens.

The plan must be completely validated before device execution starts. A placeholder count mismatch is a fatal
error; it must never truncate projected rows or leave a placeholder backed by the PAD embedding.

### Modality residency

Use explicit immutable model residency modes:

```text
text
text+vision
text+audio
text+vision+audio
```

Video selects vision residency. The selected mode is fixed when the engine and arena are created. Text-only must
remain available and must not upload or reserve the 104,759,808 modality bytes. A multimodal session keeps all
selected weights resident and performs no weight offload between turns.

An eventual product default may load both modality sets, but it may only replace text-only as the default after
measured startup, peak-VRAM, and text-only regression evidence. The first implementation should require an explicit
mode so benchmark provenance remains unambiguous.

## Media preprocessing

All media preprocessing happens before transformer prefill and outside decode graph replay. The first correctness
implementation may use CPU processing and bounded pinned staging. GPU preprocessing is a later optimization only
if end-to-end profiling identifies it as material.

### Image path

For each image:

1. Decode into exactly three RGB channels.
2. Reject invalid dimensions, resource-limit violations, unsupported sample depth, and non-finite values.
3. Resize while preserving aspect ratio to a height and width divisible by `patch_size * pooling_kernel_size`,
   which is 48 for the pinned model.
4. Select a maximum patch count of `soft_token_budget * 3 * 3`.
5. Rescale integer RGB values by `1/255`; the pinned image processor applies no additional mean/std normalization.
6. Convert the image into row-major 16x16 RGB teacher patches.
7. Merge spatial 3x3 teacher-patch groups in the exact reference element order into 6,912-value 48x48 RGB rows.
8. Produce an `(x,y)` position pair for every valid merged row.
9. Pad processor batches with zero patch rows and `(-1,-1)` position IDs only when required by the projection
   implementation; padding rows are removed before insertion into the language sequence.
10. Insert exactly one image placeholder ID per valid projected row between the begin/end image tokens.

Resize rounding, channel order, interpolation, patch flattening, 3x3 merge order, and position numbering are part
of the numerical model contract. They require byte- or value-level golden fixtures against the pinned reference
processor.

### Audio path

For each clip:

1. Decode a bounded local media buffer.
2. Convert multichannel input to one channel with a documented deterministic rule.
3. Resample to 16,000 Hz with one pinned implementation and parameters.
4. Convert to float32 samples normalized to `[-1,1]`.
5. Reject NaN, infinity, empty input, clips beyond the configured 30-second limit, and samples outside the accepted
   normalized range.
6. Split the waveform into consecutive 640-sample rows.
7. Zero-pad only the final partial row.
8. Produce a valid-row mask; padding used to batch multiple clips is not inserted into the language sequence.
9. Insert exactly one audio placeholder ID per valid row between the begin/end audio tokens.

The engine should accept already-normalized 16 kHz mono float32 buffers so tests and embedders do not depend on a
codec library. File-format support is a separate dependency decision. WAV/PCM is the narrowest useful first CLI
format; MP3, FLAC, and container audio require an explicitly pinned decoder.

### Video follow-up

Video is implemented only after image correctness and memory gates pass:

- decode and sample frames deterministically;
- preserve or derive timestamps;
- process each frame through the image path;
- use the configured low default budget of 70 soft tokens per frame;
- render timestamps and frame placeholders in the exact template order;
- use vision modality type IDs so frame blocks receive the intended local bidirectional mask;
- report frame count, sampling rate, per-frame token counts, and total media tokens.

The pinned processor default is 32 frames. Google's model card documents up to 60 seconds at one frame per second,
but support above the pinned default must be established by processor fixtures and memory/performance tests rather
than assumed.

## Modality projection operators

### Audio embedder

The audio path is:

```text
float32 waveform rows [T,640]
-> model-required input cast
-> scale-free RMSNorm(epsilon=1e-6)
-> BF16 weight projection [640 -> 3840]
-> language input embeddings [T,3840]
```

The projection should use a shape-qualified BF16 Tensor-Core GEMM with FP32 accumulation where that matches the
reference distribution. The correctness path must define the exact input cast, RMSNorm reduction, output cast, and
matrix orientation before optimizing it. At the 750-row maximum the operator is prompt-sized; no T=1 decode
variant is needed.

### Vision embedder

The vision path is:

```text
merged RGB rows [T,6912]
-> affine LayerNorm
-> BF16 patch Dense [6912 -> 3840] plus bias
-> affine LayerNorm
-> add factorized X and Y position embeddings
-> affine LayerNorm
-> scale-free RMSNorm(epsilon=1e-6)
-> BF16 embedding projection [3840 -> 3840]
-> language input embeddings [T,3840]
```

For each valid row, the two selected position vectors are summed. `(-1,-1)` rows contribute no position embedding
and are discarded before sequence insertion.

Use separate, shape-specific BF16 operators initially. Fusing norms, position addition, or projections requires
the same adjacent A/B, numerical, and memory evidence as text-path fusion. The BF16 source weights should be
consumed directly when their layout is compatible. If a load-time transformation becomes necessary, it must stream
into the sole final device allocation, preserve values exactly, and be documented in `WEIGHT_LAYOUT.md`.

### Embedding substitution

The existing input embedding gather remains responsible for hard text and delimiter tokens. Before layer 0:

1. replace modality placeholder IDs with PAD only for the embedding gather, preventing an out-of-range or
   unintended vocabulary lookup;
2. gather all hard-token embeddings;
3. project image/video/audio rows;
4. scatter projected rows into their validated placeholder ranges;
5. retain the full `[sequence,3840]` input buffer for normal transformer prefill.

Do not alter the visible token IDs used for positions, masks, diagnostics, or cache identity. The PAD substitution
is only an internal gather operation.

## Prefill integration and mask implementation

### Reuse boundary

After embedding substitution, reuse without modality-specific variants:

- FP8 attention projection kernels;
- Q/K normalization and RoPE;
- local and global KV formats;
- NVFP4 Gate/Up/Down paths;
- residual/norm fusions;
- tied output projection and softcap;
- sampling;
- generated-token decode graph.

The transformer must receive additional per-position vision block metadata during prompt prefill. Audio does not
need a special attention kernel or decode path.

### Vision-aware local attention

Add a qualified mask mode to the existing local online Tensor-Core prefill kernel. It must combine:

- absolute-position left-window eligibility;
- ordinary causal eligibility;
- same-nonnegative-vision-block eligibility;
- padding validity;
- current-chunk K/V and older local-ring addressing;
- grouped-query head sharing;
- FP8 cache scale semantics.

The global online-attention kernel remains causal. The token-at-a-time decode kernel remains causal because media
has already been ingested and all generated positions occur after the prompt.

Maintain a scalar score-materializing or CPU reference for the new mask. Operator fixtures must cover both allowed
future keys inside a vision block and forbidden future keys outside it.

### Prefill chunk boundaries

The current 2,048-token prefill plan can process current-chunk K/V before committing the local ring, but it cannot
let a query attend to future vision keys that have not yet been projected in a later chunk. Therefore the prompt
planner must not split one bidirectional vision block across prefill chunks.

For the supported image budgets, one block is at most 1,120 soft tokens and fits in a 2,048-token chunk. The planner
should:

1. identify every vision block before execution;
2. end a preceding text chunk early when the next complete vision block would cross the chunk boundary;
3. execute the complete vision block in one chunk;
4. resume ordinary chunking afterward;
5. include surrounding delimiter and timestamp tokens according to their actual causal modality type;
6. fail during plan construction if a future supported block exceeds the available qualified chunk geometry.

Silently splitting a block and treating the first half causally changes model semantics. Increasing the chunk at
runtime would violate execution-plan immutability.

### Position and KV semantics

Every media soft token consumes one ordinary absolute position and one KV position in every decoder layer. Position
IDs do not reset at modality boundaries. Media tokens count toward:

- `max_context_tokens`;
- local ring chronology;
- global cache offsets;
- proportional/global RoPE positions;
- the resident session's cached-prefix length.

For a fixed execution plan, the KV capacity is already reserved for the total context, so enabling media does not
create a separate KV allocation. It reduces the remaining text/generation capacity by the number of inserted media
tokens.

## Resident conversation sessions

### Why token-prefix identity is insufficient

Two different images with the same aspect ratio and budget produce the same hard placeholder token IDs. The same is
true for equal-length audio clips. Their layer-0 embeddings and cached K/V states differ even though the rendered
token sequence is identical.

The existing exact-token prefix check must therefore become an exact prompt-materialization check. Each cached
prefix needs an ordered identity containing:

- hard token IDs;
- modality type and soft-token range;
- selected preprocessing parameters;
- canonical media dimensions or sample count;
- a cryptographic digest of the canonical preprocessed rows, or an equivalently strong digest bound to the exact
  processor version and parameters.

A later prompt may reuse cached state only if all token and media identities in the materialized prefix match
exactly. A mismatch must fail visibly; it must not silently keep stale K/V or infer equality from file paths,
URLs, timestamps, or decoded text.

### Session lifecycle

- Initial media decoding and projection occur before its prompt suffix is prefilled.
- After K/V is committed, raw decoded media and temporary modality activations may be released or reused.
- The small identity record remains for the session lifetime.
- Later text turns reuse the media-derived K/V without re-decoding or re-projecting old media.
- A failed multimodal prefill poisons the session under the same partial-cache rule as text prefill.
- Cache reset, clone, and persistence are separate features and are not implied by multimodal support.

## Memory plan

### Persistent allocations

The exact additional tensor payload is approximately 99.91 MiB for both modality sets:

| Residency | Additional payload |
|---|---:|
| Audio | 4.69 MiB |
| Vision or video | 95.22 MiB |
| Vision/video and audio | 99.91 MiB |

These values are checkpoint payload, not complete process peak VRAM. The allocator must add alignment and report
actual device bytes. Existing text weights, scales, KV storage, graph allocations, and prefill workspace remain
separate named regions.

### Temporary input and projection storage

The planner must derive bounded regions from the selected maximum media request:

- canonical image patch rows, potentially float32 before the model-required cast;
- image position IDs and valid-row masks;
- vision 3,840-wide intermediate rows;
- audio float32 `[rows,640]` input and valid-row masks;
- audio 3,840-wide projected rows;
- placeholder/scatter metadata;
- vision block IDs for the complete prompt;
- bounded host decoder and resampler staging.

Representative payloads, excluding allocator alignment and reusable intermediates, are:

```text
30 s audio input float32       = 750 * 640  * 4 = 1,920,000 bytes
30 s audio BF16 embeddings     = 750 * 3840 * 2 = 5,760,000 bytes
1120 image rows float32        = 1120 * 6912 * 4 = 30,965,760 bytes
1120 image BF16 row storage    = 1120 * 6912 * 2 = 15,482,880 bytes
1120 vision BF16 embeddings    = 1120 * 3840 * 2 = 8,601,600 bytes
```

These buffers should reuse the existing prefill arena where lifetimes do not overlap, but overlap must be proven
from the execution schedule. No temporary maximum is a peak-memory claim until measured with CUDA context,
CUTLASS workspaces, graph pools, decoder buffers, and the selected context tier.

### Context accounting

At minimum, report:

- hard text and delimiter tokens;
- image tokens per image;
- audio tokens per clip;
- video tokens per frame and total;
- total prompt positions;
- reserved generation positions;
- remaining context positions;
- persistent modality bytes;
- modality workspace high-water mark;
- complete process peak VRAM.

A request that exceeds the immutable plan must be rejected before media projection.

## Loader and configuration work

### `config.json`

Extend `ModelConfig` and primary-contract validation to parse:

- image, audio, and video placeholder IDs;
- begin/end image and audio IDs;
- `use_bidirectional_attention`;
- all used `vision_config` fields;
- all used `audio_config` fields;
- modality presence independently of selected residency.

Validate field ranges and checked products before allocation. Do not hard-code 6,912, 3,840, 1,120, 640, or 750
without comparing them with parsed metadata and manifest shapes.

### `processor_config.json`

Add a strict bounded parser for:

- processor class and component type names;
- image rescale, RGB, resize, normalization, patch, pooling, and budget fields;
- audio sample rate, frame width, padding, mask, and sequence limits;
- video frame count, sampling, normalization, patch, pooling, and budget fields.

Reject unknown semantics that would change the canonical input tensor. A missing required processor file is an
actionable unsupported-checkpoint error in a selected multimodal mode, while text-only loading may continue.

### Tokenizer and template

Parse and validate the modality token spellings in `tokenizer_config.json`. Extend only the exact pinned
`chat_template.jinja` branches needed for ordered image, audio, and later video content. Unknown content-part types,
tool-plus-media structures not covered by fixtures, and different template revisions must continue to fail visibly.

### Manifest and upload

Replace the binary `loaded_in_text_only_mode` planning assumption with a residency decision derived from tensor
role and selected capabilities while preserving the existing text-only JSON field for compatibility. Export:

- modality role;
- supported residency modes;
- selected residency;
- bytes by modality;
- BF16 modality projection capability;
- whether any transformation occurred;
- persistent raw/repacked copy counts.

Upload selected BF16 tensors through bounded staging directly into their final aligned regions. Do not duplicate
the vision projection for video.

## Public interfaces and observability

The first implementation should expose:

- selected modality residency at engine creation;
- one-shot generation from a compiled multimodal prompt;
- resident-session generation with complete prompt-materialization identity;
- a render/plan-only mode that prints token IDs, modality ranges, token counts, shapes, and media digests without
  loading CUDA;
- explicit image token budget;
- explicit maximum clip duration and accepted audio sample format;
- machine-readable errors and result JSON.

Result JSON should include at least:

```text
modalities_requested
modalities_resident
image_count
image_soft_tokens[]
image_token_budget
audio_count
audio_soft_tokens[]
audio_seconds[]
video_frame_count
media_decode_ms
media_preprocess_ms
media_upload_ms
modality_projection_ms
transformer_prefill_ms
time_to_first_token_ms
modality_weight_bytes
modality_workspace_bytes
prompt_tokens_total
fallback_count
token_loop_allocations
```

Kernel tracing should add initialization/prefill NVTX ranges for `image_decode`, `image_patchify`,
`vision_projection`, `audio_decode`, `audio_resample`, `audio_projection`, `embedding_scatter`, and
`vision_block_attention`. Decode graph metadata should remain identical to a text-only run with the same context
plan.

## Correctness program

### Reference fixture lock

Before native implementation, pin a reference environment that supports the exact Unified checkpoint and record:

- Transformers, PyTorch, compressed-tensors, vLLM where applicable, and media-library versions;
- processor and model source revisions;
- all media input file hashes;
- preprocessing parameters;
- requested token budget and sample rate;
- reference dtype and cache mode;
- whether any compilation or fallback occurred.

Reference tooling may use Python. The runtime may not.

### Processor fixtures

Image fixtures must include:

- square, portrait, and landscape inputs;
- tiny and large source dimensions;
- RGB and accepted grayscale conversion;
- all supported token budgets;
- a case whose actual row count is below the maximum;
- exact resize dimensions, patch rows, merge order, positions, placeholder IDs, and modality type IDs;
- multiple and interleaved images;
- invalid dimensions and resource-limit rejection.

Audio fixtures must include:

- less than, exactly, and more than 640 samples;
- 1-second, 10-second, and 30-second clips;
- silence, impulses, deterministic speech, and non-16-kHz input;
- mono and accepted multichannel conversion;
- exact resampled samples, zero padding, frame rows, masks, placeholder IDs, and modality type IDs;
- empty, overlength, NaN, infinity, and out-of-range rejection.

### Operator fixtures

Validate independently:

- affine LayerNorm for widths 6,912 and 3,840;
- scale-free RMSNorm for widths 640 and 3,840;
- BF16 patch Dense including bias;
- BF16 audio and vision embedding projections;
- factorized two-axis position lookup and sum;
- padded-row removal;
- hard-token gather plus modality scatter;
- block-ID construction;
- local causal-or-same-block mask;
- global causal-only mask;
- local attention at ordinary, vision-block, 1,024-window, ring-wrap, and 2,048-chunk boundaries.

Report maximum absolute error, RMS error, cosine similarity, finiteness, and exact discrete metadata agreement.
Set tolerances from observed reference distributions rather than copying text-operator tolerances.

### Model fixtures

Capture from the trusted runtime:

- complete rendered IDs and modality metadata;
- projected image/audio embeddings before layer 0;
- selected layer-0 and layer-5 hidden states;
- local and global attention contexts;
- newly written K/V states;
- first-position full logits;
- teacher-forced logits for multiple positions;
- deterministic greedy sequences.

Qualify checkpoint-FP8 and BF16 K/V separately. A text-only result cannot validate the vision mask, and a matching
caption cannot replace embedding/logit evidence.

### Task-quality fixtures

Use fixed, licensed inputs for:

- image captioning and visual question answering;
- OCR/document reading at multiple token budgets;
- object localization or structured extraction;
- multi-image comparison;
- audio speech recognition in several supported languages;
- speech-to-text translation;
- long or noisy audio near the documented limit.

Record exact prompts, media hashes, token budgets, generation controls, and stable metrics such as WER/CER for
transcription. Broader sound, music, or event understanding must not be advertised solely from ASR evidence.

### Session fixtures

Test:

- a media prompt followed by text-only turns without media reprocessing;
- identical media reuse;
- same placeholder IDs with different media bytes;
- same file path with changed contents;
- changed image budget or audio resampler settings;
- a failed prefill and poisoned-session behavior;
- context exhaustion after media tokens;
- exact continuation after generated stop and length-limit endings.

## Benchmark program

Multimodal benchmarking must separate:

1. file decode;
2. resize/resample and canonical tensor construction;
3. host-to-device upload;
4. modality projection;
5. transformer prefill;
6. output-head first token;
7. steady-state text decode.

Report both core GPU and end-to-end timing. Do not hide CPU media preparation from TTFT, and do not include it in a
kernel-only projection number.

### Minimum image matrix

| Input | Budget | Repetitions |
|---|---:|---:|
| Square natural image | 70, 280, 1,120 | 3 warm-ups, 10 measured |
| Portrait document/OCR | 70, 280, 1,120 | 3 warm-ups, 10 measured |
| Wide image | 70, 280, 1,120 | 3 warm-ups, 10 measured |
| Two interleaved images | 280 each | 3 warm-ups, 10 measured |

### Minimum audio matrix

| Input | Duration | Repetitions |
|---|---:|---:|
| 16-kHz mono speech | 1 s, 10 s, 30 s | 3 warm-ups, 10 measured |
| Resampled speech | 10 s | 3 warm-ups, 10 measured |
| Silence/control | 30 s | 3 warm-ups, 10 measured |
| Two clips | 10 s each | 3 warm-ups, 10 measured |

### Required regressions

For every multimodal promotion, rerun:

- the existing text-only correctness suite;
- text-only prefill at 128, 512, 2,048, and 8,192 tokens;
- text-only decode at context 128 and 8K;
- model-load time and peak VRAM for text-only and selected multimodal residency;
- generated-token CUDA Graph metadata and allocation checks.

Text-only runs must not load modality weights, execute modality kernels, change prompt IDs, alter transformer
arithmetic, or regress decode because multimodal support exists in the binary.

## Security and resource limits

Treat media as untrusted input:

- bound encoded bytes before decode;
- bound decoded pixel count, width, height, frame count, sample count, duration, and channel count;
- protect all dimension and byte products with checked arithmetic;
- reject decompression bombs and malformed container metadata;
- never infer allocation size only after a decoder has already allocated an unbounded output;
- reject path traversal in request-controlled paths where a serving layer introduces a media root;
- do not fetch remote URLs in the core runtime;
- do not execute checkpoint repository code or use `trust_remote_code`;
- sanitize NaN and infinity according to an explicit reject policy;
- record the media decoder/resampler dependency, version, license, and update procedure.

Resource-limit errors must occur before GPU state is partially modified whenever possible.

## Ordered implementation milestones

### M0: Ground truth and fixture generation

Deliver:

- strict multimodal config/processor parsers;
- exact modality tensor inventory;
- pinned image and audio source fixtures;
- reference rendered IDs, processor outputs, embeddings, masks, logits, and generations;
- documented reference-package lock.

Gate:

- every discrete processor output and tensor binding is independently reproducible;
- no runtime implementation has begun from guessed semantics.

### M1: Residency-aware loader and memory planner

Deliver:

- explicit residency modes;
- manifest modality roles and byte totals;
- direct BF16 upload into final allocations;
- modality workspace planning;
- memory/result JSON;
- text-only plan unchanged.

Gate:

- byte-perfect device upload;
- no duplicate video weights;
- no persistent second layout;
- measured peak VRAM remains below the project limit with safety margin.

### M2: Common prompt materialization and session identity

Deliver:

- ordered Gemma content parts;
- exact multimodal template branches;
- compiled prompt plan;
- placeholder/feature count validation;
- media-bound resident-prefix identity;
- plan-only JSON diagnostics.

Gate:

- text rendering remains byte/token identical;
- different media with identical placeholder IDs cannot reuse cached state;
- malformed requests fail before CUDA execution.

### M3: Audio vertical slice

Deliver:

- bounded 16-kHz mono float32 input;
- 640-sample framing and masks;
- scale-free RMSNorm plus BF16 640-to-3,840 projection;
- embedding substitution;
- full-model audio prefill and text generation;
- audio processor/operator/model fixtures.

Gate:

- projected embeddings and selected logits pass measured reference tolerances;
- 30-second input fits the selected context and memory profile;
- existing text gates are unchanged;
- ASR/translation smoke quality is credible and recorded.

Audio comes first because it needs no new transformer-attention semantics and adds only one small BF16 weight
matrix. File codecs beyond a canonical waveform input do not block this model-path milestone.

### M4: Vision projection vertical slice

Deliver:

- bounded RGB input;
- exact aspect-ratio resize, patchify, 3x3 merge, and position IDs;
- complete BF16 vision embedder;
- embedding substitution;
- reference vision projection fixtures at every supported budget.

Gate:

- processor metadata is exact;
- projected embeddings pass measured tolerances;
- peak workspace and persistent memory are accounted;
- no claim is made from causal-only transformer execution.

### M5: Vision-aware transformer prefill

Deliver:

- vision block IDs;
- local causal-or-same-block online attention;
- causal global attention confirmation;
- chunk planner that keeps blocks intact;
- full-model image generation;
- mask, ring, chunk, layer, logit, and quality fixtures.

Gate:

- future vision keys are visible only where the reference allows them;
- separate vision blocks remain isolated;
- 1,120-token blocks and chunk-boundary cases pass;
- text-only attention output and performance remain unchanged.

### M6: Resident multimodal chat and product CLI

Deliver:

- one-shot and resident image/audio requests;
- exact continuation without old-media reprocessing;
- documented local file formats;
- complete observability and errors;
- end-to-end memory, TTFT, prefill, and decode benchmarks.

Gate:

- no decode-loop allocation or media work;
- cache identity is media-safe;
- stable multi-turn behavior;
- all text and multimodal correctness gates pass.

### M7: Optimization and video reuse

Deliver:

- profile-driven modality projection or preprocessing improvements;
- video frame sampling, timestamps, and vision reuse;
- video correctness, memory, and benchmark fixtures.

Gate:

- every optimization wins end to end and preserves the reference distribution;
- video adds no duplicate model weights;
- frame/token limits are explicit;
- text, image, and audio paths retain their qualified results.

## Definition of multimodal done

Image and audio support are complete only when:

- the pinned checkpoint loads directly with selected modality tensors;
- canonical media preprocessing matches the pinned reference;
- projected modality embeddings and transformer masks are validated;
- full-model logits and deterministic generations have independent reference evidence;
- task-quality fixtures pass documented thresholds;
- text-only and multimodal memory high-water marks are measured;
- media token usage and remaining context are reported;
- text-only prefill/decode do not regress;
- persistent sessions bind cached K/V to media identity;
- no token-loop allocation, media work, CPU weight offload, persistent second layout, or silent fallback exists;
- errors are actionable and all paths are visible in result JSON;
- documentation, dependency provenance, and benchmark artifacts are current.

Until those gates pass, README and benchmark output must describe multimodal execution as experimental or
unsupported rather than implying production support.
