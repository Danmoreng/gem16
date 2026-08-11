# Reference and candidate variant matrix

| ID | Master weights | Weight formats | Conversion path | Runtime | Primary question |
|---|---|---|---|---|---|
| A | Unsloth published Gemma 4 26B NVFP4 | Published mixed FP8 attention/NVFP4 MLP-experts | external artifact; local llama.cpp mixed path is diagnostic only | verified direct runtime | What is the practical published NVFP4 baseline? |
| B | Ordinary Google IT BF16 | Own FP8/NVFP4 plus selected head | planned native gem16 data plane; M05 native FP8 first | gem16 | Does our compiler reproduce a competitive ordinary conversion? |
| C | Google QAT BF16 | Own FP8/NVFP4 plus selected head | planned native gem16 data plane | gem16 | Does the QAT master improve the project hybrid? |
| D | Official Google QAT Q4_0 | Q4_0 | official GGUF; llama.cpp native quantizer is reference only | llama.cpp/reference | What quality does the QAT-target format achieve? |
| E | Selected C profile | Production hybrid | unified native gem16 compiler | gem16 optimized | What is the final product candidate? |
| F | Google QAT BF16 | BF16/high precision | no production conversion; trusted reference load | trusted reference hardware | What is the numerical architecture/quality oracle? |
| G | NVIDIA/ModelOpt Gemma 4 26B NVFP4 | Producer-specific NVFP4 | pinned `imp` or another validated loader | diagnostic/reference | What is the ModelOpt recipe control and failure mode? |
| H | UD-Q4_K_M Gemma 4 26B | UD-Q4_K_M | pinned `imp`/llama.cpp | external reference | What quality and speed context exists outside the official QAT reference? |

C has two head-format subcandidates that must remain subordinate to the primary C ID:

| Subcandidate | C head format | Question |
|---|---|---|
| C-Q4H | Q4_0 | Does the QAT master retain quality with the QAT-targeted head? |
| C-NVH | NVFP4 | Does native W4A4 head speed justify its additional error? |

The binding conversion architecture is [`../specs/NATIVE_CONVERTER_ARCHITECTURE.md`](../specs/NATIVE_CONVERTER_ARCHITECTURE.md).
M04's Python `copy-v1` is byte copying, not numerical conversion. M05's current production data plane is the native
`gem16-fp8-compiler`; `gem16-checkpoint-compiler` is a planned unified name and is not yet runnable. The version-scoped
llama.cpp research is [`../../../evidence/gemma4_26b/m05-llama-cpp-converter-research-2026-08-11.md`](../../../evidence/gemma4_26b/m05-llama-cpp-converter-research-2026-08-11.md).

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
