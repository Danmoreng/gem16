# Quantized tied embedding and output-head specification

## Invariant

One logical matrix:

```text
[vocabulary 262144, hidden 2816]
```

serves both input embedding and output projection. Exactly one resident payload is allowed.

## Binding

Suggested:

```cpp
enum class TiedWeightFormat { kBf16, kQ4_0, kNvfp4 };

struct TiedWeightBinding {
  TiedWeightFormat format;
  const void* payload;
  const void* local_scales;
  const float* global_scale;
  std::uint32_t rows;
  std::uint32_t columns;
  std::uint64_t resident_bytes;
};
```

The production 26B path does not support BF16 residency.

## Embedding lookup

Input:

```text
token ID
```

Output:

```text
one hidden row at the engine's accepted input boundary
```

Requirements:

- bounds check before launch in debug/validation;
- decode only selected row;
- apply model embedding scale exactly;
- deterministic conversion;
- support chunked prompt lookup;
- no row cache that changes benchmark semantics without disclosure.

## Output projection

Input:

```text
T hidden rows, T = 1 ordinary, T ≤ 5 bounded verifier/diagnostic
```

For every vocabulary row:

1. dot product;
2. final logit softcap;
3. optional diagnostic write;
4. suppression;
5. candidate reduction;
6. deterministic final argmax.

Sampling path materializes or processes all required logits according to the existing exact sampling contract.

## Q4_0 path

- W4A16-style decode;
- explicit block decode;
- reuse decoded weight across T rows;
- no BF16 weight cache;
- same stored matrix for lookup/head.

## NVFP4 path

- native W4A4 candidate;
- dynamically quantize hidden row;
- direct block-scaled MMA where advantageous;
- preserve QAT quality evaluation caveat;
- use accepted scale/divisor contract.

## Candidate blocks

Current constant `4096` may remain if geometry is valid, but vocabulary/hidden must not remain hard-coded. Derive grid and candidate allocation from traits and validate fixed upper bounds.

## Softcap

Use checkpoint value, expected 30.0:

```text
softcapped = tanh(logit / cap) * cap
```

Apply to all logits before selection and diagnostics according to existing semantics.

## Tie break

For equal softcapped values, lower token ID wins.

## Suppression

- fixed checkpoint suppression;
- dynamic control count;
- exact behavior in greedy and batch;
- full-logit diagnostics remain unmodified or explicitly documented relative to suppression.

## Full logits

Only allocate host full-logit storage before generation when requested. Device full-logit/sampling workspace is preallocated.

For T steps:

```text
T × vocabulary × 4 bytes
```

must be checked.

## Tests

- row lookup;
- tied pointer;
- Q4/NVFP4 CPU reference;
- full logits versus fused candidates;
- softcap;
- suppression;
- tie;
- T=1/3/5;
- sampling;
- graph capture;
- no duplicate BF16 matrix;
- quality and performance A/B.
