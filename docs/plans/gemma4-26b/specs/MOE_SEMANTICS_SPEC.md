# Gemma 4 26B MoE semantic specification

## Layer-level sequence

The accepted reference order for a 26B text decoder layer is conceptually:

```text
residual0 = hidden

attention_input = input_layernorm(hidden)
attention_output = self_attention(attention_input)
attention_output = post_attention_layernorm(attention_output)
hidden = residual0 + attention_output

residual1 = hidden

shared_input = pre_feedforward_layernorm(hidden)
shared_output = shared_dense_mlp(shared_input)
shared_output = post_feedforward_layernorm_1(shared_output)

router_input = residual1
router_probabilities, top8_weights, top8_ids = router(router_input)

expert_input = pre_feedforward_layernorm_2(residual1)
expert_output = routed_experts(expert_input, top8_ids, top8_weights)
expert_output = post_feedforward_layernorm_2(expert_output)

ff_output = shared_output + expert_output
ff_output = post_feedforward_layernorm(ff_output)
hidden = residual1 + ff_output

hidden *= layer_scalar
```

The exact source config/reference implementation must confirm names and precision boundaries. This sequence is normative after M10 accepts it.

## Shared dense MLP

The shared branch runs for every token and every layer.

Typical logical operation:

```text
gate = W_gate x
up   = W_up x
product = gelu_tanh(gate) * up
down = W_down product
```

Its intermediate size is 2112.

Do not:

- route it through top-k;
- count it as one of 128 experts;
- skip it when no routed expert hits;
- merge its norm boundaries with routed experts without proof.

## Routed experts

There are 128 experts and 8 selected per token. Each expert has intermediate size 704.

Logical expert:

```text
gate, up = linear_gate_up(x)
product  = activation(gate) * up
output   = linear_down(product)
weighted = output * selected_router_weight
```

The implementation may store gate/up fused, but logical ordering remains explicit.

## Combination

The eight contributions are summed in a locked order. Recommended deterministic order:

1. top-k slot order returned by the accepted router;
2. within one slot, ordinary row/channel order;
3. FP32 accumulation;
4. model-required cast/norm.

If a faster grouped reduction uses expert-ID order or atomics, it must prove quality equivalence and determinism or remain experimental.

## Precision boundaries

Required questions to answer and freeze in M10:

- router input stored BF16 or FP32;
- norms compute FP32 and cast point;
- shared MLP gate/up output BF16 boundary;
- expert input quantization boundary;
- expert contribution accumulation precision;
- shared+expert addition precision;
- final norm and residual cast order.

The current 12B arithmetic is useful but not automatically the 26B contract.

## Router independence

Router operates on `residual1`, not the already normalized shared MLP input unless the trusted reference proves otherwise.

Expert input uses its own pre-feedforward norm.

## Layer scalar

Apply at the final layer boundary exactly once. Never fold it into one branch unless the equivalence and cast order are proven.

## Diagnostics

Every correctness path can optionally capture:

- shared branch;
- router logits/probabilities;
- top-8 IDs and selected weights;
- each expert product and down output;
- weighted contributions;
- routed sum;
- shared+routed sum;
- final layer result.

Production path does not need to persist these arrays.

## Tests

- shared-only and expert-only synthetic layers;
- all-equal router;
- one dominant expert;
- top-k boundary/tie;
- fused gate-up axis;
- deterministic reduction;
- layer scalar;
- real BF16 captures;
- dequantized ordinary/QAT artifact modes.

## imp-derived branch diagnostics

Diagnostics must expose shared and routed branches independently, including their distinct pre/post norms, FP32 weighted expert reduction, per-expert scales and final residual/layer-scalar boundary. A single combined hidden-state checksum cannot localize the failure modes observed in independent Gemma 4 implementations.
