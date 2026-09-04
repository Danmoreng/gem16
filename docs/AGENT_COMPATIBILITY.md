# SDK and coding-agent compatibility

This is a bounded development matrix, not full Agent Core v1 release qualification.
The [contract](OPENAI_AGENT_CORE_V1.md) retains the two-platform release gate.
The [retained Linux run](../artifacts/agent-core/2026-09-04-linux-verified/result.json)
records commands, source and binary hashes, locks, hardware, health, sampling and raw outputs.

## Matrix

| Check | Linux 12B Unified | Linux 26B Compact Vision | Windows live GPU |
|---|---|---|---|
| Python 2.50.0: Chat, streamed and non-streamed tool loops | Passed | Passed | Pending |
| Python 2.50.0: Responses, streamed and non-streamed tool loops | Passed | Passed | Pending |
| TypeScript / openai-node 7.10.0: equivalent Chat and Responses cases | Passed | Passed | Pending |
| Sequential calls, strict schemas, Unicode and long tool results | Passed | Passed | Pending |
| Two-call full-history fixture, both APIs and transports | Passed | Passed | Pending |
| Output limits, stale Responses IDs, unsupported options | Passed | Passed | Pending |
| Malformed JSON, context overflow, cancellation/disconnect and recovery (Python) | Passed | Passed | Pending |
| Unmodified Pi 0.85.0: read, edit, run check, final answer | Passed | Passed | Pending |

The independent-lookup prompt produced grouped calls on 12B and sequential calls
on Compact Vision. `parallel_tool_calls=true` permits grouping; it does not force
a model to choose it. The separate full-history fixture verifies two pending call
IDs and both results. It is protocol evidence, not a claim that 26B generated
parallel calls in this run.

Tests use one slot, 16,384 context, the pinned fixed-D2 Assistant, checkpoint
sampling defaults and `reasoning_effort=none`. These are compatibility tests,
not throughput or broad coding-quality benchmarks. Image/audio quality, two-slot
concurrency, queue saturation, process-restart recovery and Windows live execution
are not qualified by this matrix. Existing evidence for those boundaries remains separate.

## Reproduce

Build the current CUDA server and acquire the [locked models](SERVER.md#start).
Use Python 3.11+ and Node 22.19+; Git Bash is required for this Pi fixture on Windows.
No OpenAI account or remote inference key is used.

```bash
python3 -m venv .venv-agent-core
. .venv-agent-core/bin/activate
python -m pip install -r tools/requirements-openai-sdk.txt
npm ci --prefix tools/openai-sdk --ignore-scripts
npm --prefix tools/openai-sdk run check
npm ci --prefix tools/pi-agent --ignore-scripts
python tools/run_agent_core_matrix.py \
  --server build/Linux/blackwell-release/bin/gem16-server \
  --pi-cli tools/pi-agent/node_modules/@earendil-works/pi-coding-agent/dist/cli.js \
  --output-dir artifacts/raw/agent-core-new-run
```

On Windows activate the venv with `.\.venv-agent-core\Scripts\Activate.ps1`,
use `build/Windows/blackwell-release/bin/gem16-server.exe`, and pass the same
runner arguments on one line or with PowerShell backtick continuations.
The output directory must be new; runs do not overwrite earlier evidence.
The runner owns and stops each server, then proceeds to the other profile.

For an already running server at 16,384 context, run individual probes:

```bash
python tools/validate_openai_sdk.py --base-url http://127.0.0.1:8080/v1
npm --prefix tools/openai-sdk run validate -- http://127.0.0.1:8080/v1
python tools/validate_external_agent.py \
  --base-url http://127.0.0.1:8080/v1 \
  --pi-cli tools/pi-agent/node_modules/@earendil-works/pi-coding-agent/dist/cli.js \
  --output-dir artifacts/raw/pi-new-run
```

The Pi harness creates an isolated temporary project and configuration. It proves
the original test fails, requires both files to be read and the source edited,
checks the agent actually ran a passing test, and independently reruns the
unchanged test. Its transcript and before/after files are retained. Pi's tools
execute locally; the harness is for the supplied disposable fixture, not a sandbox
for arbitrary repositories. No agent plugin, proxy, protocol rewrite or source patch is used.

[SDK dependencies](../tools/openai-sdk/package-lock.json) and
[Pi dependencies](../tools/pi-agent/package-lock.json) are pinned. Pi 0.85.0's
published CLI imports `pi-server` without declaring it, so the harness explicitly
installs the matching package as well. It does not launch a Pi server.

## Compatibility limits to account for

- Python 2.50.0's `responses.stream().get_final_response()` only accumulates
  `response.completed`. For an output limit, consume the typed `response.incomplete`
  event's `response` instead. The server must not mislabel an incomplete response
  to satisfy this helper. The TypeScript helper handles the terminal incomplete response.
- Native generation currently accepts `tool_choice=auto|none` and
  `parallel_tool_calls=true`. Required/named tool choice and `parallel_tool_calls=false`
  are parsed but fail visibly at execution; constrained decoding remains outstanding.
- Assistant `reasoning_content` is an output extension, not an accepted replay field.
  The tested Pi configuration maps thinking `off` to `reasoning_effort=none` and
  disables the unsupported Chat `store` field. Thinking-enabled Pi replay is not qualified.
- Responses state is a resident linear chain. Clients must serialize the documented
  input subset; arbitrary SDK output objects can contain unsupported fields such as
  `id` or `status`. Stale/evicted IDs, branching and durable storage remain outside v1.

## Fixes established by this matrix

Full-history agent conversations previously failed after the second tool round.
The native renderer now keeps assistant continuations after tool results inside
one model turn, matching both immutable Gemma templates. The host regression
covers two tool rounds, the final answer, a subsequent user turn and invalid role pairs.
The expected tool-chain text was also compared with Jinja rendering of all three
pinned profile templates.

Responses arrays previously materialized adjacent `function_call` items as
separate assistant messages. The adapter now groups those items into one assistant
turn while preserving result boundaries and call order. Host and live SDK fixtures
cover this independently of the model's scheduling choice.

Primary client references: [OpenAI Python](https://github.com/openai/openai-python),
[OpenAI TypeScript streaming](https://github.com/openai/openai-node/blob/main/docs/responses.md),
and [Pi model configuration](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/models.md).
