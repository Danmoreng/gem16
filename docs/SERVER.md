# OpenAI-compatible server

## Start

```powershell
.\build\Windows\blackwell-release\bin\gem16-server.exe `
  --model .\models\checkpoints\unsloth-gemma-4-12b-it-NVFP4-b1f6497 `
  --model-name gem16 `
  --host 127.0.0.1 --port 8080 `
  --max-context 8192 --max-sessions 2
```

The default uses the checkpoint-recommended sampling profile. `--greedy`
selects deterministic generation. `--assistant-model`,
`--mtp-draft-tokens 1|2|4`, and `--mtp-adaptive` expose the same qualified MTP
path as resident chat. The server has no authentication or TLS layer; bind to
loopback unless a trusted reverse proxy supplies those controls.

Startup creates one `ModelRuntime` and logs its target/assistant weight bytes
and load time. It constructs one temporary execution-slot probe, measures the
larger of allocator accounting and observed VRAM delta, and rejects a
`--max-sessions`/`--max-context` combination that cannot retain every configured
slot plus a 700 MiB safety margin. The probe is released before listening.
Sessions are then created on demand. Each receives an isolated `SessionState`
plus `ExecutionSlot` while sharing the immutable runtime.
`--max-sessions` bounds resident slots; inactive least-recently-used sessions
are evicted when the limit is reached, while active sessions are never evicted.
Admission reserves capacity under the pool mutex, constructs the CUDA execution
slot without holding that mutex, then publishes the completed session. Health,
metrics, cancellation, and unrelated resident-session acquisition therefore
remain responsive while a new slot is being prepared.

## Endpoints

- `GET /health` reports status, resident session count, and the configured limit.
- `GET /metrics` exports Prometheus text metrics.
- `GET /v1/models` lists the configured `--model-name`.
- `POST /v1/chat/completions` returns an OpenAI Chat Completion or chunked SSE.
- `POST /v1/responses` returns an OpenAI Response or typed Responses SSE
  events and supports a resident `previous_response_id` continuation.
- `POST /v1/responses/{response_id}/cancel` cancels an active response.

The 16 MiB request limit, JSON depth/value limits, codec limits, 30-second
audio limit, 100-megapixel image limit, maximum 280 image soft tokens, output
capacity, and session context are all checked before or at generation. Errors
use the OpenAI `error` envelope and an appropriate 4xx/5xx status before SSE
headers; generation-time stream failures are emitted as an SSE error record.

## Requests

Supported request fields are `model`, `messages`, `max_completion_tokens`
(`max_tokens` alias), `stream`, `stream_options.include_usage`,
`reasoning_effort` (`none`, `low`, `medium`, `high`), `tools`, `tool_choice`,
`parallel_tool_calls`, and `n=1`. Every other top-level or nested protocol field
is rejected rather than silently ignored; this includes per-request sampling,
stop, response-format, logprob, metadata, and unsupported media/tool options.

Message content accepts strings and ordered arrays containing:

- `{"type":"text","text":"..."}` (also `input_text`);
- `{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}`;
- `{"type":"input_audio","input_audio":{"format":"wav","data":"..."}}`.

## Responses API

`POST /v1/responses` accepts `model`, `input`, `instructions`,
`max_output_tokens`, `stream`, `store`, `reasoning.effort`, `tools`,
`tool_choice`, `parallel_tool_calls`, and `previous_response_id`; unknown fields
are rejected at every parsed protocol level. Function tools
use the Responses top-level shape (`type`, `name`, `description`, `parameters`,
`strict`). Input may be a string or ordered `message`, `function_call`, and
`function_call_output` items. Message content supports `input_text`, inline
data-URL `input_image`, and Base64 `input_audio`.

The response contains typed `message` and `function_call` output items, exact
input/output/reasoning usage, resident-prefix `cached_tokens`, newly prefetched
`cache_write_tokens`, and `completed` or `incomplete` status. Streaming
emits ordered `response.created`, output-item/content/function events, and a
final `response.completed` object consumable by the official OpenAI SDK.
Private reasoning is materialized as a completed `reasoning` output item and,
for streaming requests, as matching `response.reasoning_text.*` events before
the visible assistant message. Reasoning and visible-text deltas are written as
soon as complete UTF-8 code points become available during decode; they are not
buffered until generation completes. After a tool result, the checkpoint template
leaves generation directly at the model boundary. With reasoning disabled, Gemma
emits an empty thought envelope containing one newline before the visible answer;
the runtime tracks and bounds that token so resident accounting and streamed
continuation remain exact. Function-call events remain intentionally post-generation
because strict schemas and the complete native tool call must be validated before
the server exposes a successful call.

Each Responses session deliberately remains one exact linear chain:

- a continuation must name the latest returned response ID;
- stale, unknown, or branched IDs return 404;
- omitted tools and tool choice inherit from the previous response; explicitly
  supplied values must be identical;
- `store=false` and changed continuation instructions are rejected visibly.

Many independent roots may coexist up to `--max-sessions`. Creating another
root evicts the inactive least-recently-used chain; its response IDs then return
404. Branches remain unsupported because a single KV prefix cannot represent
two continuations. `client.responses.cancel(id)` sets a generation-loop
cancellation flag. Cancelled or disconnected generations discard their slot
because a partially advanced KV cache is unsafe to reuse.

Official SDK gate:

```powershell
py -m pip install -r tools\requirements-openai-sdk.txt
py tools\validate_openai_responses.py `
  --base-url http://127.0.0.1:8080/v1 --model gem16
```

JPEG/BMP and MP3/FLAC use the analogous MIME/format values. Remote image URLs
are deliberately unsupported in this milestone. Images and audio are decoded
in memory without temporary files. Multiple parts retain JSON order, and all
images share the context-aware automatic soft-token budget.

Assistant `tool_calls` and `tool` messages with `tool_call_id` are accepted.
Native Gemma tool DSL never leaks through the API: the resident identity is
canonical visible assistant text plus structured calls, which an OpenAI client
can reproduce exactly in the following request.

Tool names and parameter names must fit the checkpoint-native protocol grammar:
1 through 64 ASCII letters, digits, or underscores. Incompatible names such as
hyphenated identifiers are rejected before generation instead of being
silently changed. Parameter schemas require an object root. Schema constraints
that the native declaration renderer supports are retained in the model prompt.
For `strict:true`, generated argument JSON is additionally checked for declared
tool identity, object shape, required/additional properties, type, enum/const,
array and string bounds, numeric bounds, local references, and composition
constraints before a successful response is exposed. Unsupported strict-schema
keywords return a visible request error instead of weakening the contract.

## Streaming

With `"stream":true`, the response is `text/event-stream`. It begins with an
assistant-role delta, emits UTF-8 text or `reasoning_content` deltas, emits
indexed function tool-call/name/argument deltas, then a finish-reason chunk.
When `stream_options.include_usage` is true, an empty-choices usage chunk
follows. Every successful stream ends with `data: [DONE]`.

Both streaming endpoints preallocate their decoded-token, partial UTF-8, JSON,
and HTTP/SSE framing buffers before generation. The token callback writes fixed
buffers directly through cpp-httplib's raw content provider, including HTTP/1.1
chunk framing, so the library does not construct a fresh host string per token.
Completion objects, full reasoning items, and validated tool events may allocate
after decode has ended; no pageable host allocation or growing container is
introduced in the successful token loop.

## Session identity and admission

Responses sessions are addressed by `previous_response_id`. Chat Completions
uses `X-Gem16-Session-Id`: omit it to create a session and read the generated ID
from the response header; send it on later requests to reuse the same resident
conversation. Server-generated chat-completion, response, and session handles
contain 128 bits from the operating system cryptographic random generator; they
are opaque and must not be inferred from creation order. A session is single-flight and rejects a second concurrent
request with HTTP 503, while distinct sessions can run concurrently. Request-validation
or unsupported-option errors leave an unchanged resident cache available for a
corrected retry; only state-mutating inference, cancellation, or streaming failures
poison and discard the slot. If every configured slot is active, admission also
returns 503 instead of evicting live state or allocating beyond the configured bound.

The HTTP stream owns its session lease from admission until the provider is
released. Consequently a peer that disconnects after admission but before the
SSE provider starts cannot leave `active_requests` pinned. A generation-time
disconnect still discards the affected mutable KV slot.

When a later full-history Chat Completions request adds another image, the
adapter may recompute a smaller aggregate image budget for all supplied media.
Resident prefix validation identifies the unchanged encoded source and retains
the already-prefilled canonical patch representation for old turns; only new
media use the new request's budget. Changed source payloads remain a prefix
mismatch.

`/metrics` reports request/failure/active counts, resident/limit/created/evicted
sessions, requested/observed cancellations, client disconnects, total/cached/cache-write
input tokens, output tokens, generation time, immutable target/assistant bytes,
pending session creations, planned/configured/resident execution-slot bytes,
device capacity and safety margin, and the latest measured execution-slot byte
count. Responses `completed_at` is sampled only after successful generation and
KV-chain commit rather than copied from `created_at`.

## Official OpenAI SDK qualification

The reproducible agent gate uses the official Python SDK pinned in
`tools/requirements-openai-sdk.txt`. Install it in an isolated environment,
start a fresh gem16 server, then run:

```powershell
python -m venv .venv-openai
.\.venv-openai\Scripts\python.exe -m pip install `
  -r .\tools\requirements-openai-sdk.txt
.\.venv-openai\Scripts\python.exe .\tools\validate_openai_agent.py `
  --base-url http://127.0.0.1:8080/v1 --model gem16
```

The default gate uses `client.chat.completions.create(stream=True)`, requests
an indexed `get_weather` function call, accumulates the SDK's typed tool
deltas, validates its JSON arguments, executes the deterministic local fixture,
appends the assistant call and tool result, and checks the streamed grounded
answer plus usage chunks. `--no-stream` exercises the same loop with ordinary
Chat Completion objects.

The scheduler qualification additionally exercises two independent Responses
chains, exact continuation, LRU eviction, two concurrent generations,
cancellation, and metrics:

```powershell
.\.venv-openai\Scripts\python.exe .\tools\validate_server_scheduler.py `
  --base-url http://127.0.0.1:8080/v1 --model gem16
```
