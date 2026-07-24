# Golden fixtures

Fixtures in this directory are tied to the exact checkpoint revision and reference-runtime versions embedded in
each file. Regenerate them only with the documented local/offline command, retain token IDs rather than relying on
decoded text, and review any numerical or token change before updating expected data.

The initial vLLM fixture contains greedy token IDs and top-20 log probabilities for 32 steps or until stop. It is a
generation and partial-logit fixture, not a full-vocabulary logit or hidden-state golden.

The `correctness-v1-fp8` and `correctness-v1-bf16` fixtures use the 12-chat
suite in `benchmarks/prompts/correctness-v1.json`, explicit vLLM K/V-cache
dtypes, and up to 24 generated tokens per prompt. They are intended for
position-aligned teacher-forced comparisons. They likewise contain top-20
distributions rather than full reference logits.
