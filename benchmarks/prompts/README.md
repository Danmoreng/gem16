# Benchmark and correctness prompts

`correctness-v1.json` is the source prompt suite used to generate pinned,
tokenized reference fixtures. It deliberately contains text because the
reference generation step must exercise the checkpoint's own tokenizer and
chat template. Actual engine comparisons consume the rendered
`prompt_token_ids` stored in the generated golden JSON, so tokenizer behavior
cannot hide model-execution differences.

Generate an FP8-cache vLLM fixture with:

```bash
python tools/generate_golden.py \
  --model <checkpoint> \
  --prompts benchmarks/prompts/correctness-v1.json \
  --kv-cache-dtype fp8 \
  --max-tokens 24 \
  --output tests/golden/vllm-gemma4-12b-nvfp4-correctness-v1-fp8.json
```
