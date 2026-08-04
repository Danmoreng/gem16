# Reference and candidate variant matrix

| ID | Master weights | Weight formats | Runtime | Primary question |
|---|---|---|---|---|
| A | ordinary IT, published Unsloth conversion | FP8 attention, NVFP4 MLP/experts, BF16 ignored families | verified direct runtime | What is the practical published NVFP4 baseline? |
| B | ordinary Google IT BF16 | own FP8/NVFP4 + selected head | gem16 | Does our compiler reproduce a competitive ordinary conversion? |
| C-Q4H | Google QAT BF16 | own FP8/NVFP4 + Q4_0 head | gem16 | Does QAT master help while preserving Q4_0 head? |
| C-NVH | Google QAT BF16 | own FP8/NVFP4 + NVFP4 head | gem16 | Does native head speed justify added W4A4 error? |
| D | Google QAT official Q4_0 | Q4_0 | llama.cpp/reference | What quality does the QAT-target format achieve? |
| E | selected C profile | production hybrid | gem16 optimized | Final product candidate |
| F | QAT BF16 | BF16/high precision | trusted reference hardware | Architecture/quality oracle |
| G | ordinary BF16 | BF16/high precision | trusted reference hardware | Ordinary master control |

## Required comparisons

### Quantizer control

```text
A versus B
```

Interpret carefully: both originate from ordinary master weights, but converter/runtime can differ.

### QAT master effect

```text
B versus C
```

Same project compiler. This is the cleanest available isolation of master-weight change.

### Target-format comparison

```text
C versus D
```

Same QAT master family; formats/runtime differ.

### Production quality

```text
E versus D, A and F
```

### Production speed

```text
E versus D and A
```

with common token inputs and disclosed runtime differences.

## Invalid shortcuts

Do not:

- compare A and C only and call difference a QAT effect;
- mix A expert tensors with C router/norms and call it higher quality;
- use D head with C body without proving source/tensor identity and complete model quality;
- omit B because it requires additional compiler time;
- call C “NVFP4 QAT” as if trained for NVFP4.
