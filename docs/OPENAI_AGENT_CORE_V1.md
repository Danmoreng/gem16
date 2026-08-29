# OpenAI Agent Core v1

**Status:** product contract; implementation substantially present, full
two-platform/two-model qualification pending

**Transport:** local HTTP/1.1 JSON and server-sent events

OpenAI Agent Core v1 is gem16's deliberately bounded compatibility surface for
local chat applications and coding agents. It follows the OpenAI Chat
Completions and Responses object shapes where listed below. It does not claim
full OpenAI platform compatibility.

The official Responses API is broader than this contract and includes hosted,
MCP, custom, background, stored-conversation, and other facilities. Gem16 v1
implements the client-executed function-calling subset required for local
agents and rejects unsupported behavior visibly.

Official reference:

- <https://developers.openai.com/api/reference/cli/resources/chat/subresources/completions>
- <https://developers.openai.com/api/reference/cli/resources/responses/methods/create>

## Required endpoints

| Endpoint | v1 purpose |
|---|---|
| `GET /v1/models` | List the configured model identifier |
| `POST /v1/chat/completions` | Stateless/full-history chat and function-tool turns |
| `POST /v1/responses` | Typed response items, streaming, and resident continuation |
| `POST /v1/responses/{response_id}/cancel` | Cancel an active response |
| `GET /health`, `/live`, `/ready`, `/metrics` | Local operation and diagnostics |

Every HTTP response carries `X-Request-Id`. Errors use an OpenAI-style `error`
envelope and an appropriate HTTP status. Unsupported options must never be
silently ignored or translated into different model behavior.

## Chat Completions request subset

Supported top-level fields:

- `model`;
- `messages`;
- `max_completion_tokens`, with `max_tokens` as an alias;
- `stream`;
- `stream_options.include_usage`;
- `reasoning_effort`: `none`, `low`, `medium`, or `high`;
- function `tools`;
- `tool_choice`: `none`, `auto`, `required`, or a named function;
- `parallel_tool_calls`;
- `n=1`.

Supported roles are `developer`, `system`, `user`, `assistant`, and `tool`.
Assistant `tool_calls` and tool results linked by `tool_call_id` are required.
Text input is required for both profiles. Inline image/audio content remains a
12B capability extension and is not necessary for Agent Core conformance.

Successful streaming emits assistant-role, text or reasoning, indexed
function-call, finish-reason, optional usage, and terminal `[DONE]` chunks in
that order.

## Responses request subset

Supported top-level fields:

- `model`;
- `input` as a string or ordered item array;
- `instructions`;
- `max_output_tokens`;
- `stream`;
- `store=true`;
- `reasoning.effort`;
- function `tools`;
- `tool_choice`;
- `parallel_tool_calls`;
- `previous_response_id`.

Required input item types are `message`, `function_call`, and
`function_call_output`. Required output item types are `message`, `reasoning`
when present, and `function_call`. Function calls preserve `id`, `call_id`,
name, arguments, ordering, and completion status.

Successful streaming uses typed Responses events, including the applicable:

- `response.created`;
- `response.output_item.added` and `.done`;
- `response.content_part.added` and `.done`;
- `response.output_text.delta` and `.done`;
- reasoning-text events when reasoning is exposed;
- `response.function_call_arguments.done`;
- `response.completed`.

Sequence numbers and identifiers must remain internally consistent and the
final SDK response must reproduce the streamed content and usage.

## Function tools

Agent Core v1 supports client-executed JSON-schema function tools. It requires:

- one or more declared functions;
- deterministic tool identity and opaque call IDs;
- `strict:true` validation for the schema subset documented by the server;
- multiple and parallel calls when the model emits them;
- complete function results returned by the client in the next turn;
- visible rejection of unsupported schema keywords or incompatible names.

Gem16 executes no arbitrary tool code in the server. The calling agent owns
filesystem, shell, network, approval, and sandbox policy. Tool arguments and
outputs are untrusted data and must be validated by both server and client at
their respective boundaries.

## Conversation state

Chat Completions accepts full history and may additionally reuse a resident
session through `X-Gem16-Session-Id`.

Responses supports a linear resident chain through `previous_response_id`.
The identifier must name the latest response in that chain. Stale, evicted,
unknown, or branched identifiers fail visibly. Multiple independent root
chains may coexist only within the configured execution-slot limit.

`store=false`, durable server-side conversation storage, response retrieval,
branching, and resumption after process restart are outside v1. Coding-agent
clients that require durable history must retain and resend their own canonical
history rather than assuming OpenAI-hosted storage semantics.

## Explicitly outside v1

- authentication, TLS, billing, quotas, organizations, projects, and remote
  multi-user operation;
- Batch, Files, Assistants, Realtime, fine-tuning, embeddings, image generation,
  moderation, and audio-output APIs;
- hosted web/file search, code interpreter, computer use, connectors, and MCP
  tools;
- free-form custom tools, grammar tools, structured response formats, logprobs,
  prompt caching controls, background responses, and stored conversations;
- response branching and persistent prompt-cache files;
- silent acceptance of OpenAI fields whose semantics gem16 cannot preserve.

These exclusions may be revised only through a new versioned compatibility
decision. They do not prevent an external coding agent from providing similar
capabilities as ordinary function tools.

## Qualification gate

Agent Core v1 becomes release-qualified only when all of the following pass on
Windows and Linux, for both 12B and 26B where capability-applicable:

1. official OpenAI Python SDK non-streaming and streaming Chat tool loop;
2. official OpenAI Python SDK non-streaming and streaming Responses tool loop;
3. an official OpenAI JavaScript/TypeScript SDK equivalent;
4. multiple sequential and parallel function calls with strict schemas;
5. long tool output, Unicode, malformed request, unsupported option, context
   overflow, disconnect, cancellation, queue saturation, and restart cases;
6. resident continuation and client-managed full-history behavior;
7. at least one unmodified external coding-agent workflow that reads files,
   calls multiple tools, applies a bounded edit, runs a check, and completes;
8. exact version, model, capability, sampling, context, fallback, allocation,
   and usage reporting.

The existing Python SDK validators are development evidence. They do not alone
complete this two-platform, two-model release gate.
