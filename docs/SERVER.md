# OpenAI-compatible server

## Start

```powershell
.\build\Windows\blackwell-release\bin\gem16-server.exe `
  --model .\models\checkpoints\unsloth-gemma-4-12b-it-NVFP4-b1f6497 `
  --model-name gem16 `
  --host 127.0.0.1 --port 8080 `
  --max-context 8192
```

The default uses the checkpoint-recommended sampling profile. `--greedy`
selects deterministic generation. `--assistant-model`,
`--mtp-draft-tokens 1|2|4`, and `--mtp-adaptive` expose the same qualified MTP
path as resident chat. The server has no authentication or TLS layer; bind to
loopback unless a trusted reverse proxy supplies those controls.

Startup creates one `ModelRuntime` and logs its target/assistant weight bytes
and load time. The current single conversation is then created as an isolated
`SessionState` plus `ExecutionSlot` sharing that runtime. This ownership split
is the prerequisite for the bounded multi-session scheduler; the current HTTP
surface remains serialized until that scheduler lands.

## Endpoints

- `GET /health` returns `{"status":"ok"}`.
- `GET /v1/models` lists the configured `--model-name`.
- `POST /v1/chat/completions` returns an OpenAI Chat Completion or chunked SSE.

The 16 MiB request limit, JSON depth/value limits, codec limits, 30-second
audio limit, 100-megapixel image limit, maximum 280 image soft tokens, output
capacity, and session context are all checked before or at generation. Errors
use the OpenAI `error` envelope and an appropriate 4xx/5xx status before SSE
headers; generation-time stream failures are emitted as an SSE error record.

## Requests

Supported request fields are `model`, `messages`, `max_completion_tokens`
(`max_tokens` alias), `stream`, `stream_options.include_usage`,
`reasoning_effort` (`none`, `low`, `medium`, `high`), `tools`, `tool_choice`,
`parallel_tool_calls`, and `n=1`. `temperature`, `top_p`, penalties, and `seed`
are rejected rather than silently ignored because sampling state currently
belongs to the resident session.

Message content accepts strings and ordered arrays containing:

- `{"type":"text","text":"..."}` (also `input_text`);
- `{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}`;
- `{"type":"input_audio","input_audio":{"format":"wav","data":"..."}}`.

JPEG/BMP and MP3/FLAC use the analogous MIME/format values. Remote image URLs
are deliberately unsupported in this milestone. Images and audio are decoded
in memory without temporary files. Multiple parts retain JSON order, and all
images share the context-aware automatic soft-token budget.

Assistant `tool_calls` and `tool` messages with `tool_call_id` are accepted.
Native Gemma tool DSL never leaks through the API: the resident identity is
canonical visible assistant text plus structured calls, which an OpenAI client
can reproduce exactly in the following request.

## Streaming

With `"stream":true`, the response is `text/event-stream`. It begins with an
assistant-role delta, emits UTF-8 text or `reasoning_content` deltas, emits
indexed function tool-call/name/argument deltas, then a finish-reason chunk.
When `stream_options.include_usage` is true, an empty-choices usage chunk
follows. Every successful stream ends with `data: [DONE]`.

## Current ownership boundary

This is intentionally a single-conversation, single-execution-slot server.
After the first request, each request must contain the exact public messages
returned so far and append one user turn or consecutive tool results. Requests
are serialized. This preserves the qualified resident KV prefix without a
second weight copy. The next runtime milestones separate shared immutable
weights, per-conversation state, and execution workspaces before adding
multiple resident sessions, cancellation, LRU eviction, and metrics.

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
Chat Completion objects. Use a fresh server because A8 intentionally tests the
current single resident conversation from its first request.
