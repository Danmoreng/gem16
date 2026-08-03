# Gemma 4 native constrained tool decoding plan

Status: planned and explicitly deferred; this document does not authorize implementation yet

Target model: Gemma 4 12B Unified instruction-tuned checkpoint

Target checkpoint: `unsloth/gemma-4-12b-it-NVFP4` at
`b1f649734b34aa5575b03d186abd1b9be3d0d5c4`

Primary use case: schema-valid native function calls for the OpenAI-compatible Chat Completions and Responses APIs

## Objective

Add constrained decoding only where it maps directly to the pinned Gemma 4 tool protocol. The future implementation
should provide:

- reliable `tool_choice: "required"`;
- reliable selection of one named function;
- schema-valid arguments for the deliberately supported function-schema subset;
- identical constraint semantics for greedy, sampled, ordinary, and MTP-assisted decode;
- no CPU round trip, dynamic allocation, or host scheduling decision per generated token;
- no measurable regression for requests that do not use constrained decoding;
- visible rejection of OpenAI features that the checkpoint does not natively represent.

This is not a plan for generic OpenAI API coverage or an arbitrary structured-output framework.

## Checkpoint capability boundary

The locked tokenizer metadata and chat template are authoritative. They explicitly define:

- tool declarations through `<|tool>declaration:...<tool|>`;
- calls through `<|tool_call>call:NAME{ARGUMENTS}<tool_call|>`;
- results through `<|tool_response>...<tool_response|>`;
- unquoted argument keys;
- `<|"|>` as the native string delimiter;
- repeatable tool calls through `response_template.fields.tool_calls.repeats = true`;
- a response parser for optional reasoning, repeated function calls, and visible text;
- function parameter declarations with object properties, required fields, scalar values, nested objects, arrays,
  nullability, and string enums.

The checkpoint does **not** define:

- an intrinsic `tool_choice` control;
- a native switch that disables repeated calls;
- generic OpenAI `response_format: json_schema` output;
- OpenAI built-in tool types such as web search, file search, computer use, code interpreter, or MCP;
- the complete JSON Schema specification.

The engine may enforce a narrower choice or schema by masking invalid next tokens, but it must use the native Gemma
markers and argument DSL. It must not claim that an OpenAI-only control is a checkpoint field.

## Initial scope

### In scope

- function tools only;
- `tool_choice: "auto"`, `"none"`, and `"required"`;
- named function choice;
- native Gemma function names containing 1 through 64 ASCII letters, digits, or underscores;
- object-root parameter schemas;
- strings, integers, numbers, booleans, null, objects, and arrays;
- declared properties and required properties;
- `additionalProperties: false`;
- string enums;
- nested object and array item schemas within explicit depth and size bounds;
- constraints during greedy and sampled target selection;
- transactional constraint state during MTP verification;
- validated Chat Completions and Responses output.

### Deferred or out of scope

- generic JSON response mode;
- arbitrary JSON Schema;
- external or recursive references;
- regex and format constraints;
- OpenAI built-in or server-executed tools;
- remote schema retrieval;
- unbounded grammar programs;
- CPU-side token filtering in the generation loop;
- silently approximating unsupported schema keywords;
- `parallel_tool_calls: false` in the first implementation.

`parallel_tool_calls: false` means that one assistant response may contain at most one call. Gemma explicitly supports
repeated calls but exposes no native disable switch. Continue rejecting `false` until a real client requirement
justifies implementing exact first-call termination or an equivalent constrained path. Do not generate multiple
calls and discard extras after the fact.

## Existing implementation to preserve

The current runtime already provides:

- bounded OpenAI request parsing;
- checkpoint-native declaration and result rendering;
- an incremental Gemma tool-call parser;
- conversion from the native argument DSL to JSON;
- declaration, identity, and strict-schema validation after generation;
- multiple calls and tool-result continuation;
- non-streaming and SSE output for Chat Completions and Responses;
- exact resident message and KV-prefix continuation;
- official OpenAI SDK qualification for an `auto` weather-tool loop.

The constrained path must reuse these public representations and validators. The CPU validator remains the final
independent safety check even after GPU constraints are added.

## Architecture

Use a hybrid design:

1. validate and compile the request-specific grammar on the CPU before generation;
2. upload a bounded immutable grammar program into a fixed device region;
3. retain the mutable grammar state on the GPU;
4. filter candidate tokens on the GPU before argmax or probability truncation;
5. commit only the transition associated with the selected or target-verified token.

A CPU decision for every token is prohibited. Copying the selected token to the CPU, advancing a host parser, and
uploading a new mask would serialize GPU-to-CPU-to-GPU work, break the GPU-controlled graph, and materially increase
inter-token latency.

## CPU grammar compiler

The CPU compiler runs before the generated-token loop and may use bounded temporary host storage. It should:

1. parse the already validated tool definitions;
2. reject schema constructs outside the model-native subset;
3. normalize property order and value constraints deterministically;
4. compile declared function names into a trie;
5. compile argument syntax into a compact bounded grammar program;
6. calculate maximum stack, counter, state, and table requirements with checked arithmetic;
7. reject programs that exceed the configured grammar arena;
8. serialize the program into its final device representation;
9. upload it before CUDA Graph replay begins;
10. export a human-readable compilation report for tests and diagnostics.

The compiler must have a CPU reference executor. The reference executor is the correctness oracle for all GPU
transition tests.

## Grammar representation

A pure dense DFA may become unnecessarily large for nested objects and arrays. Prefer a deterministic bounded grammar
VM with:

- a small phase/state identifier;
- a fixed-depth stack;
- bounded object/array counters;
- immutable byte-class and literal tables;
- a trie for valid function and property names;
- explicit accepting, incomplete, and invalid results.

The program remains deterministic for a given current state and candidate token byte sequence:

```text
Transition(current_state, token_bytes) -> invalid | next_state
```

Candidate evaluation uses a local copy of the state. It must not mutate the committed state. A separate commit step
applies the transition for the selected token.

Characterize the tokenizer's exact token-to-byte transition for every vocabulary entry. Store context-independent
pieces once in an immutable model-level device table. If detokenization depends on preceding-token state, represent
that bounded state explicitly in `ConstraintControl` and include it in candidate simulation and commit; do not assume
that every display string is context independent. Byte-fallback, whitespace, UTF-8, and special control tokens require
explicit fixtures. No kernel may infer token text from token IDs.

## Device control

Reserve a bounded fixed-address state block per execution slot, conceptually:

```cpp
struct ConstraintControl {
  uint32_t enabled;
  uint32_t mode;
  uint32_t phase;
  uint32_t grammar_state;
  uint32_t stack_size;
  uint32_t selected_tool;
  uint32_t call_count;
  uint32_t error;
  // Fixed-capacity stack and counters follow in the planned region.
};
```

The concrete layout must be versioned, aligned, included in allocator reports, and covered by host/device layout
assertions. State reset occurs before generation, never through allocation in the token loop.

## Token filtering

Constraints must be applied before token selection.

### Greedy path

The optimized greedy path currently reduces output-head values directly into candidates without requiring a full
FP32 vocabulary-logit buffer. Add a separate constrained output-head variant that excludes invalid tokens during the
existing candidate reduction:

```text
compute softcapped logit
-> test GrammarTransition(state, token_bytes)
-> retain candidate only when transition is valid
-> reduce to constrained argmax
```

Do not route constrained greedy generation through the diagnostic full-logits path merely for implementation
convenience.

### Sampled path

The sampled path already materializes vocabulary logits and then applies repetition penalty, suppression,
temperature, sorting, top-k, top-p, min-p, and seeded selection. The grammar mask belongs before sorting and
probability truncation:

```text
raw softcapped logits
-> repetition penalty and fixed suppression
-> grammar validity mask
-> temperature
-> sort
-> top-k / top-p / min-p
-> seeded selection
```

An invalid token receives negative infinity. Masking after top-k or top-p is incorrect because valid tokens may
already have been discarded.

### State commit

After selection, a small device operation advances the committed state with the selected token. Invalid selected
transitions are internal errors and poison the mutable session rather than silently disabling constraints.

## Tool-choice semantics

### `none`

Do not render tool declarations and explicitly prevent entry into the native tool-call marker if needed for a hard
guarantee. Existing ordinary text generation must otherwise remain unchanged.

### `auto`

Allow ordinary reasoning and visible text. If the model selects `<|tool_call>`, activate the native function-name and
argument grammar for the complete call. Every successful call must pass the independent final CPU validator.

### `required`

Allow the configured reasoning envelope when requested. At the first model-output decision that may begin visible
content or a tool call, permit only the native tool-call path. A successful response must contain at least one
complete validated call. Exhausting the output budget before a complete call produces an incomplete/error outcome,
not a successful text response.

### Named function

Use the same required path but restrict the function-name trie to the selected declared function. Do not implement
this by changing the user's system prompt or silently dropping the other definitions from response validation.

## MTP integration

MTP requires transactional grammar state just like transactional KV, repetition, and sampling state.

For a D2 group, derive tentative states entirely on the GPU:

```text
row_state[0] = committed_state
row_state[1] = Transition(row_state[0], draft[0])
row_state[2] = Transition(row_state[1], draft[1])
```

A row after an invalid draft prefix is marked unusable. Each Target verifier row uses its corresponding tentative
state when filtering logits. Acceptance still compares the assistant proposal with the constrained Target decision;
constraints must never accept a token that the Target did not select.

Commit only the state produced by the emitted verified prefix and mismatch token. States belonging to rejected rows
are discarded. D1 and D4 follow the same rule.

The assistant should eventually use the same tentative grammar when proposing tokens. Leaving the assistant
unconstrained is correctness-safe if the Target rejects invalid drafts, but it can materially reduce acceptance and
is not the final production design.

Ordinary and MTP generation must remain token-identical for the same target sampling seed and constraint program.
A temporary ordinary-only implementation must fail visibly or report an explicit optimization fallback; it must not
silently benchmark as constrained MTP.

## CUDA Graph and performance isolation

Create separate immutable execution plans and captured graphs:

```text
unconstrained ordinary decode
constrained ordinary decode
unconstrained fixed-D2 MTP
constrained fixed-D2 MTP
```

Do not add a generic `if (constraint_enabled)` branch to the existing hot output-head kernels. Even a uniform branch
can change register allocation, scheduling, or SASS. The unconstrained translation path should remain unchanged and
bit-identical.

The model-level token-byte table may be shared by all slots. Each slot receives a bounded grammar program/state
region with stable addresses. All additional bytes must appear in the device-arena and server-slot accounting. No
program growth, container growth, JIT compilation, or memory allocation is allowed after generation starts.

## Error handling and observability

Reject before inference when:

- a tool name cannot be represented by the Gemma protocol;
- a schema uses unsupported constructs;
- the compiled program exceeds depth, state, literal, counter, or arena limits;
- `required` or named choice has no declared eligible function;
- the selected named function does not exist;
- constraints are requested with an execution mode that has not been implemented and qualified.

Expose at least:

- constraint mode;
- compiled program version and byte size;
- maximum and used states/stack depth;
- selected function restriction;
- constrained tokens generated;
- invalid assistant drafts rejected by MTP;
- constraint fallback count, which must remain zero in qualified results;
- terminal grammar state and validation result.

Never return malformed arguments as a successful OpenAI tool call.

## Implementation phases

### Phase 0: Contract fixtures

- Freeze fixtures from the pinned `tokenizer_config.json` and `chat_template.jinja`.
- Enumerate the exact supported schema subset.
- Add native declaration, call, result, and repeated-call golden strings.
- Characterize token pieces that cross marker, delimiter, number, and UTF-8 boundaries.

Gate: the subset is documented and every unsupported construct fails visibly.

### Phase 1: CPU reference compiler and VM

- Implement deterministic normalization and compilation.
- Implement the bounded CPU transition VM.
- Test arbitrary byte and token chunking.
- Fuzz malformed programs, depth/count limits, and parser agreement.

Gate: every accepted stream converts to arguments accepted by the existing independent validator.

### Phase 2: Ordinary constrained greedy decode

- Add fixed arena regions and upload.
- Add the constrained greedy output-head variant.
- Keep grammar state device-resident.
- Implement required and named choice for one and multiple declared functions.

Gate: no host synchronization or allocation per token; required never returns a successful text-only response.

### Phase 3: Sampled decode

- Apply grammar validity before top-k/top-p/min-p.
- Preserve ordinary seeded RNG-step semantics.
- Cover repetition penalty and reasoning transitions.

Gate: deterministic same-seed constrained output and no change to unconstrained sampled output.

### Phase 4: Auto mode and strict argument guarantees

- Activate argument constraints after a model-selected tool marker.
- Support the complete documented model-native schema subset.
- Retain the independent final validator.
- Emit OpenAI structures only after successful completion and validation.

Gate: all generated calls across the qualification suite are declared, complete, parseable, and schema-valid.

### Phase 5: MTP

- Build tentative row states.
- Filter each Target verifier row with the correct state.
- Commit only verified grammar state.
- Constrain assistant proposals.
- Integrate D1/D2/D4 and stop/tail behavior.

Gate: ordinary/MTP identity across seeds, local-ring wrap, tool-result continuation, and rejection cases.

### Phase 6: Streaming and production qualification

- Stream only complete UTF-8 and protocol-safe argument prefixes.
- Add Responses argument-delta events only when their semantics are exact.
- Run official SDK Chat Completions and Responses agent suites.
- Collect full performance, memory, graph, and allocation evidence.

Gate: all correctness and performance requirements below pass.

## Testing requirements

### Host tests

- every supported scalar type;
- nested objects and arrays;
- required and optional properties;
- enums and nullability;
- additional-property rejection;
- duplicate/unknown functions and properties;
- incomplete markers, strings, numbers, objects, and arrays;
- depth, count, size, and recursion-limit failures;
- arbitrary input chunk boundaries;
- agreement between grammar acceptance, native parser, and final schema validator.

### CUDA tests

- CPU/GPU transition agreement for every vocabulary token over representative states;
- tokens spanning multiple grammar symbols;
- byte-fallback and UTF-8 boundaries;
- greedy and sampled masking before selection;
- no-valid-token detection;
- graph reset and replay;
- state commit after stop, length, and complete calls;
- D1/D2/D4 tentative-state commit and rollback;
- assistant invalid-draft rejection;
- Compute Sanitizer memcheck and synchronization checks.

### Full-model tests

- auto text response with tools present;
- auto single and repeated calls;
- required call;
- named call among multiple declarations;
- strict nested arguments;
- tool-result continuation to a grounded final answer;
- reasoning before a required call;
- greedy and multiple sampled seeds;
- ordinary/MTP exact identity;
- Chat Completions and Responses, streaming and non-streaming;
- resident continuation and context-boundary failure.

## Performance gates

For unconstrained requests:

- use the exact existing kernels and graph;
- retain output hashes and sampled same-seed output;
- retain graph node/launch counts;
- show no statistically meaningful decode regression;
- report any persistent VRAM increase.

For constrained requests:

- report grammar compilation and upload latency separately;
- report ordinary and MTP tokens/s and inter-token latency;
- report output-head and grammar-transition kernel time;
- report register count, shared memory, occupancy, and spills;
- report grammar program/state bytes;
- prove zero token-loop allocations and zero fallback;
- compare constrained ordinary and constrained MTP under identical output semantics.

Do not claim success from a microbenchmark alone. The final decision requires a complete SDK tool loop with valid
arguments and an end-to-end repeated benchmark.

## Likely implementation locations

The final file split should be decided only when implementation starts. Likely boundaries are:

```text
include/gem16/chat.h                         public constraint/tool-choice state
src/runtime/tool_constraint_compiler.*       CPU compiler and reference VM
src/runtime/tool_call_parser.*               independent final parser/validator
src/cuda/sampling/tool_constraints.*         device program and transitions
src/cuda/output_head.*                       constrained greedy variant
src/cuda/sampling/sampling.*                 constrained sampled variant
src/cuda/mtp/*                               tentative and committed row states
src/cuda/engine/*                            arena, dispatch, and graph plans
src/server/openai_chat.*                     protocol mapping and errors
tests/unit/*                                  compiler/parser fixtures
tests/cuda/*                                  CPU/GPU transition fixtures
tools/validate_openai_agent.py               SDK qualification
```

Do not create a generic grammar framework for hypothetical models. The implementation should remain Gemma 4 tool
protocol specific until a second real model demonstrates a compatible reusable boundary.

## Open questions before implementation

1. What exact schema keywords can the pinned template render without semantic loss?
2. Should constrained object keys use one canonical order or permit every valid order?
3. What grammar depth, property count, enum size, and literal-byte limits fit the server memory contract?
4. Does required choice permit visible text after one or more calls in the same turn?
5. Should reasoning remain optional for required choice, and at what exact state is the call requirement activated?
6. Can token validation be fused into the output head without unacceptable register growth, or should it use a
   precomputed per-step allowed-token bitset?
7. What is the measured acceptance impact of constraining the Target before constraining the assistant?
8. Which Responses argument-delta sequence is accepted by the pinned SDK while preserving post-validation safety?
9. Is exact first-call termination for `parallel_tool_calls: false` worth implementing after required/named choice?

Resolve these with checkpoint fixtures and measured prototypes, not API-shape assumptions.

## Definition of done

The feature is complete only when:

- the supported subset is derived from and documented against the pinned checkpoint;
- required and named function choice work without prompt-only enforcement;
- every successful strict call is valid by both the GPU grammar and independent CPU validator;
- ordinary and MTP constrained generation are target-identical for the same seed;
- the official SDK completes Chat Completions and Responses tool loops;
- unsupported schemas and modes fail visibly;
- no token-loop allocation or CPU control round trip exists;
- unconstrained decode has no statistically meaningful regression;
- constrained performance, VRAM, kernel resources, and fallback counts are recorded;
- documentation clearly distinguishes model-native support from engine-enforced constraints.
