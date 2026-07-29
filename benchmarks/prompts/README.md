# Benchmark and correctness prompts

`correctness-v1.json` is the source prompt suite used to generate pinned,
tokenized reference fixtures. It deliberately contains text because the
reference generation step must exercise the checkpoint's own tokenizer and
chat template. Actual engine comparisons consume the rendered
`prompt_token_ids` stored in the generated golden JSON, so tokenizer behavior
cannot hide model-execution differences.

`quality-longform-v1.json` contains semantic, long-form diagnostics rather
than exact-token goldens. Its Harry Potter cases pin the observed German
factual failure and correction follow-up with required and forbidden phrases.
Run them with both thinking disabled and the qualified thinking-budget modes;
a phrase check is only a regression screen and never a substitute for human
review or reference-logit comparison.

Generate an FP8-cache vLLM fixture with:

```bash
python tools/generate_golden.py \
  --model <checkpoint> \
  --prompts benchmarks/prompts/correctness-v1.json \
  --kv-cache-dtype fp8 \
  --max-tokens 24 \
  --output tests/golden/vllm-gemma4-12b-nvfp4-correctness-v1-fp8.json
```
