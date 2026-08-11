# Q4_0 format and reference specification

## Intended use

Q4_0 is a planned native extension of the shared converter data plane. The pinned llama.cpp implementation is an
external reference/golden source for exact block semantics, not a runtime GGUF dependency and not a reason to create a
large BF16 intermediate artifact. See [`NATIVE_CONVERTER_ARCHITECTURE.md`](NATIVE_CONVERTER_ARCHITECTURE.md).

Q4_0 is:

1. the official Google QAT-target format used as an external quality reference;
2. a candidate format for the tied embedding/output head;
3. an optional later full-model reference backend.

It is not NVFP4 and cannot be passed directly to Blackwell's NVFP4 block-scaled MMA.

## Block format

One block represents 32 logical weights:

```text
FP16 d
16 bytes qs
```

Total:

```text
18 bytes / 32 weights = 0.5625 byte per weight = 4.5 bits
```

Reference quantization:

```text
amax = maximum absolute value in block
max  = signed value whose absolute value is amax
d    = max / -8
id   = d != 0 ? 1 / d : 0
q    = clamp(int(x * id + 8.5), 0, 15)
```

The pinned llama.cpp implementation is the normative external reference for exact block encoding. Tie/float-cast behavior must be captured in fixtures rather than paraphrased loosely. The inspected local research checkout is version-scoped to commit `0b14b87d7c20cb753b94b96854dd7b45306fc696`; the desired benchmark pin is separately recorded and must not be conflated with this fixture source. Any copied code requires the llama.cpp MIT notice and explicit code provenance.

Dequantization maps nibble codes back with the block scale according to the reference implementation.

## Nibble arrangement

The reference packs values from the first and second 16-value halves into low/high nibbles. Tests must pin the exact order. Do not assume adjacent values share a byte.

## Tied head storage

For 738,197,504 logical weights:

```text
738,197,504 / 32 * 18 = 415,236,096 bytes
```

The artifact stores exactly one matrix for input lookup and output projection.

## Execution candidates

### Lookup

- identify row;
- iterate Q4_0 blocks;
- decode 32 values;
- write BF16 or FP32 hidden row;
- apply embedding scale if the model requires it.

### Output T=1

Preferred reference path:

```text
Q4_0 weights × BF16/FP32 hidden
→ FP32 accumulation
→ final logit softcap
→ suppression
→ per-block candidate
→ deterministic global argmax
```

This is a W4A16-style path. It does not use NVFP4 W4A4 MMA directly.

### Batch T≤5

Reuse each decoded weight block across bounded hidden rows. Do not regress T=1 solely for a future MTP path.

## Comparison to official Google tensor

When extracting a tensor from GGUF for tests:

- pin GGUF hash and llama.cpp commit;
- verify tensor name, logical shape and byte offset;
- preserve bytes exactly;
- do not make GGUF extraction part of production startup;
- document any row/axis order conversion.

Project Q4_0 generated from QAT BF16 should be byte-compared only if the source checkpoint and exact quantizer are known to match. Otherwise compare dequantized values and outputs.

## Error metrics

- reconstruction relative L2/cosine/max/SQNR;
- block scale distribution;
- code histogram;
- zero blocks;
- output-head logit KL/top-k agreement;
- argmax margin sensitivity;
- task quality in M19.

## Performance interpretation

Q4_0 has similar storage to NVFP4, so T=1 can be bandwidth competitive. However:

- unpack/dequant work is explicit;
- activation stays higher precision;
- no direct NVFP4 block-scaled MMA;
- prefill may lag native NVFP4 substantially.

Only controlled end-to-end measurements decide.

## Tests

The production candidate encoder must run in the shared native C++ compiler. Python/NumPy may be used only for
small independent diagnostics. No silent F16 fallback, GGUF intermediate or second persistent head representation is
permitted.

- exact reference blocks;
- zero and signed-extreme blocks;
- nibble order;
- full row lookup;
- output logits and argmax;
- official tensor fixture;
- deterministic compiler output;
- no hidden BF16 copy.
