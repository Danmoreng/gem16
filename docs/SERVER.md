# OpenAI-compatible server

## Start

```powershell
.\build\Windows\blackwell-release\bin\gem16-server.exe `
  --model .\models\checkpoints\unsloth-gemma-4-12b-it-NVFP4-b1f6497 `
  --model-name gem16 `
  --host 127.0.0.1 --port 8080 `
  --max-context 8192 --max-sessions 2 `
  --max-queued-requests 64 --log-level info --log-format text
```

The default uses the checkpoint-recommended sampling profile. `--greedy`
selects deterministic generation. `--assistant-model`,
`--mtp-draft-tokens 1|2|4`, and `--mtp-adaptive` expose the same qualified MTP
path as resident chat. The server has no authentication or TLS layer; bind to
loopback unless a trusted reverse proxy supplies those controls.

The qualified Gemma 4 26B A4B profile uses a compiled Target and separately compiled Assistant:

```bash
gem16-server \
  --model /models/gemma4-26b-a4b-compiled \
  --assistant-model /models/gemma4-26b-a4b-assistant-compiled \
  --mtp-draft-tokens 2 --model-name gemma4-26b-a4b \
  --host 127.0.0.1 --port 8080 \
  --max-context 32768 --max-sessions 1
```

For 26B, fixed D1/D2/D4 and the normal GPU sampling controls are supported; D2 is the selected profile and
`--mtp-adaptive` is rejected. Health reports `model_variant=gemma4_moe_26b_a4b`, `text_only=true`, actual MTP
capability and the configured `mtp_max_context`. Images and audio are rejected, and exactly one resident execution
slot is admitted. These restrictions do not change the 12B server path.

The measured 26B limits are separate: ordinary Target execution supports up to 98,304 tokens with a 400 MiB
long-context reserve, while fixed-D2 MTP supports up to 86,016 tokens with a 200 MiB reserve. The server applies the
smaller reserve only when MTP is active and rejects larger Assistant contexts during initialization.

Startup creates one `ModelRuntime` and logs its target/assistant weight bytes
and load time. It constructs one temporary execution-slot probe, measures the
larger of allocator accounting and observed VRAM delta, and rejects a
`--max-sessions`/`--max-context` combination that cannot retain every configured
slot plus the selected safety margin: 700 MiB for the primary profile, 400 MiB for long-context 26B Target-only,
or 200 MiB for qualified long-context 26B MTP. The probe is released before listening.
Sessions are then created on demand. Each receives an isolated `SessionState`
plus `ExecutionSlot` while sharing the immutable runtime.
`--max-sessions` bounds resident slots; inactive least-recently-used sessions
are evicted when the limit is reached, while active sessions are never evicted.
Admission reserves capacity under the pool mutex, constructs the CUDA execution
slot without holding that mutex, then publishes the completed session. Health,
metrics, cancellation, and unrelated resident-session acquisition therefore
remain responsive while a new slot is being prepared.

Generation admission uses a bounded FIFO in front of those execution slots.
Up to `--max-sessions` independent requests may execute concurrently; the next
`--max-queued-requests` wait in arrival order. A request beyond that bound gets
HTTP 503 with `resource_exhausted` instead of growing host memory without a
limit. Requests targeting the same resident session also wait for its current
turn to finish. Thus the one-slot 26B profile serializes all generation while
the default two-slot 12B profile retains two-request parallelism. Cancellation,
health, readiness, liveness and metrics bypass the generation queue.
The underlying HTTP worker-task queue is bounded separately with a small
control-plane reserve, so connections cannot accumulate unbounded host work
before reaching generation admission.

## Endpoints

- `GET /health` reports status, resident session count, and the configured limit.
- `GET /live` reports process liveness.
- `GET /ready` reports readiness and changes to HTTP 503 while draining.
- `GET /metrics` exports Prometheus text metrics.
- `GET /v1/models` lists the configured `--model-name`.
- `POST /v1/chat/completions` returns an OpenAI Chat Completion or chunked SSE.
- `POST /v1/responses` returns an OpenAI Response or typed Responses SSE
  events and supports a resident `previous_response_id` continuation.
- `POST /v1/responses/{response_id}/cancel` cancels an active response.

## Logging and lifecycle

Every HTTP response receives an `X-Request-Id`. The server writes one atomic
access-log record after the response (including after an SSE stream ends), with
request ID, method, matched route, status, total duration, queue wait and input
byte count. Lifecycle, model-load, memory-admission, unexpected-exception and
shutdown events use the same logger. Request bodies, generated text, tool
arguments and session IDs are not logged.

`--log-format text` is the local-readable default; `--log-format json` emits
one JSON object per line. `--log-level` accepts `debug`, `info`, `warning`,
`error`, or `off`. Health/readiness/liveness/metrics access records are debug
level unless they fail. Logs go to stderr so a process supervisor can perform
rotation without the inference process opening or managing log files.

SIGINT and SIGTERM switch readiness and health to draining, reject and wake
queued-but-not-admitted work, stop accepting connections and let admitted
handlers unwind before model teardown. HTTP reads time out after 30 seconds,
writes after 60 seconds, and idle keep-alive connections after 5 seconds.

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
are opaque and must not be inferred from creation order. A session remains
single-flight internally; a second concurrent request waits in FIFO admission
and then for that session, while distinct sessions can use separate slots.
Request-validation
or unsupported-option errors leave an unchanged resident cache available for a
corrected retry; only state-mutating inference, cancellation, or streaming failures
poison and discard the slot. If every configured slot is active, admission
waits up to the configured queue bound instead of evicting live state or
allocating beyond the configured limit.

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
input tokens, output tokens, generation/prompt/decode time, decode-timed tokens,
MTP proposed/accepted/rejected tokens, verifier and D1/D2/D4 group counts,
ordinary fallback tokens, immutable target/assistant bytes, pending session
creations, planned/configured/resident execution-slot bytes,
device capacity and safety margin, the latest measured execution-slot byte
count, current/maximum/high-water queue depth, queue admissions/waits/rejections,
model-load/server-startup time, and cumulative Prometheus histograms for request,
queue, generation, prompt and decode duration. Responses `completed_at` is
sampled only after successful generation and
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

For measured HTTP root, resident-cache, token-streaming, and two-slot contention
characterization, run the standard 3/10 harness and retain its JSON at a new
result path:

```powershell
python .\tools\benchmark_server.py `
  --base-url http://127.0.0.1:8080/v1 --model gem16 `
  --scenario all --warmup 3 --repetitions 10 `
  --output benchmarks\results\2026-07-30\<git-sha>\<machine-id>\server.json
```

The harness refuses to overwrite evidence and distinguishes complete HTTP wall
time, first streamed delta, resident cache usage, and concurrent aggregate
throughput from core-GPU benchmark boundaries.

A separate managed-server harness exercises the production-style long-session
case: one 262,144-position FP8 slot, recommended checkpoint sampling, fixed MTP
D2, a checksum-locked root containing three generated images and three
public-domain speech excerpts, incremental text turns, and measurements at the
roughly 2K empty-cache root and near 4K/8K/32K/64K/128K resident context. It
records engine prefill/decode time from
per-request Prometheus deltas, streamed first-delta and burst-aware delta
intervals, exact cache writes, MTP acceptance, continuous GPU telemetry, and a
final retrieval check against the original media:

```powershell
python .\tools\benchmark_server_long_conversation.py `
  --server-executable .\build\Windows\blackwell-release\bin\gem16-server.exe `
  --model .\models\checkpoints\unsloth-gemma-4-12b-it-NVFP4-b1f6497 `
  --assistant-model .\models\checkpoints\google-gemma-4-12B-it-assistant-364bd03 `
  --output benchmarks\results\<date>\<git-sha>\<machine-id>\server-long.json
```

The tool starts and stops the server itself, verifies the advertised slot,
context, sampling, and MTP configuration, and validates every default media file
against `benchmarks/media/suite.json` before sending traffic. Optional repeated
`--image` and `--audio` arguments append local stress media rather than replacing
the repository suite.
