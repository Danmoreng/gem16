# gem16

gem16 is a Kotlin Compose Desktop application for local Gemma 4 chat and
`gem16-server` lifecycle management. It follows the desktop structure proven by
the neighboring Qwen-TTS Studio while remaining an HTTP client: the JVM does
not load CUDA or model weights directly.

## Features

- streamed Chat Completions with separate reasoning and answer presentation;
- CommonMark headings, emphasis, lists, quotes, links, and fenced code rendering;
- selection across complete multi-paragraph Markdown answers, one-click copying
  of full model responses, and per-block copy actions for fenced code and HTML;
- ordered PNG/JPEG/BMP image and WAV/FLAC/MP3 audio attachments;
- drag-and-drop attachment handling across the complete chat area, with a
  visible drop target and the same size/type validation as the file picker;
- direct image paste from the system clipboard with Ctrl+V or Cmd+V, encoded
  in memory as PNG without creating a temporary file;
- in-memory microphone recording with live level, timer, cancel, and automatic
  stop after the server's 30-second audio limit;
- Enter-to-send with Shift+Enter for multiline input;
- streaming auto-follow that pauses for explicit user scrolling and resumes at
  the bottom, with a desktop scrollbar and **Jump to latest** action;
- resident multi-turn sessions through `X-Gem16-Session-Id`;
- selectable off/low/medium/high thinking budgets;
- cancellation, new-chat, usage, and finish-reason state;
- automatic managed-server startup plus explicit start/stop on Linux and Windows;
- attachment to a compatible externally started server;
- first-run download of the pinned target model, MTP assistant, and official
  tokenizer configuration into the shared Hugging Face Hub cache;
- automatic reuse of existing content-addressed Hugging Face blobs, resumable
  downloads, SHA-256 verification, and cache-backed server configuration;
- MTP, context, session, host, port, and sampling settings;
- live `/health` state and bounded process logs;
- persistent settings under `~/.gem16-studio/settings.properties`;
- dark and light themes;
- a logo-driven light palette and a neutral charcoal dark palette with gem16
  green accents, plus compact sidebar navigation with inline server state;
- live client-observed stream throughput and time to first token while a reply
  is generated, followed by exact decode/prefill metrics from `/metrics` when
  the request completes.

Tool-call interaction, Responses history, sampled video frames, and remote
authenticated deployments remain follow-ups. Markdown parsing uses the pinned
BSD-2-Clause `org.commonmark:commonmark` dependency; raw HTML is displayed as
text rather than executed.

## Build and run

Requirements:

- JDK 21;
- network access for the first Gradle dependency resolution;
- a built SM120/SM120a `gem16-server`;
- enough disk space in the configured Hugging Face cache for the pinned target
  and assistant checkpoints.

Linux:

```bash
./scripts/build.sh --cuda --test
./scripts/run-studio.sh
```

Windows PowerShell:

```powershell
.\scripts\build.ps1 -Cuda -Test
.\scripts\run-studio.ps1
```

Direct Gradle commands:

```bash
./gradlew :studioApp:desktopTest
./gradlew :studioApp:run
./gradlew :studioApp:packageDistributionForCurrentOS
```

A local microphone smoke test is opt-in because CI machines may not expose a
capture device:

```bash
GEM16_TEST_MIC=1 ./gradlew :studioApp:desktopTest \
  --tests com.gem16.studio.AudioRecorderDeviceTest --rerun-tasks
```

Compose packaging can produce DMG, MSI, and DEB images. Packaging first stages
the current platform's release `gem16-server` into the application resources;
the installed application discovers that binary automatically. Checkpoints are never
embedded in the application. On first launch, the **Models** screen downloads
the exact locked snapshots and configures the managed server automatically.

On Windows, `scripts/package-studio.ps1` removes only previous gem16 MSI outputs,
forces the Compose packaging tasks to rerun, and fails unless exactly one fresh MSI
is produced. The result is written below
`studioApp/build/compose/binaries/main/msi/` and installs per machine to
`C:\Program Files\gem16`. The packaged CUDA server lives at
`app\resources\bin\gem16-server.exe` inside that installation.

`.github/workflows/windows-release.yml` provides the corresponding reproducible
GitHub path. A `v*` tag or manual dispatch builds with pinned CUDA 13.3 and WiX
toolchains, verifies the server contains the expected native SM120a NVFP4 and FP8
instructions, packages the MSI, and uploads both the installer and a SHA-256 file
to the workflow run and GitHub Release. The server executable has no dynamic CUDA
Toolkit DLL dependency, so only a compatible NVIDIA driver is required on the
target machine. The model payload is still downloaded separately on first use.

gem16 honors `HF_HUB_CACHE`, then `HF_HOME`, then `XDG_CACHE_HOME`, and otherwise
uses `~/.cache/huggingface/hub`, matching Hugging Face Hub conventions. The
target model view is composed inside that cache from content-addressed hardlinks:
the 9.3 GB payload is not duplicated. An access token can come from `HF_TOKEN`,
`HUGGING_FACE_HUB_TOKEN`, the normal Hugging Face token file, or the in-memory
field on the Models screen. Google repository licenses must be accepted by the
account before gated files can be downloaded.

## Managed server behavior

gem16 probes the configured endpoint at application startup only after the
configured model set exists. It attaches when
a compatible server is already healthy; otherwise it immediately executes the
configured binary with an argument vector and without invoking a shell. A
typical command is equivalent to:

```bash
gem16-server \
  --model /models/unsloth-gemma-4-12b-it-NVFP4 \
  --assistant-model /models/google-gemma-4-12B-it-assistant \
  --mtp-draft-tokens 2 \
  --model-name gem16 \
  --host 127.0.0.1 --port 8080 \
  --max-context 32768 --max-sessions 1
```

Output is captured into a 1,000-line in-memory ring. On application shutdown,
gem16 stops only the process it created. A server discovered through `/health`
is labeled external and is never terminated by the UI.

The server has no built-in authentication or TLS. gem16 defaults to loopback
and visibly warns for any other bind address.

## Chat protocol

The client sends the complete visible conversation to
`POST /v1/chat/completions`, requests SSE, and retains the opaque session ID
returned by gem16. Text and `reasoning_content` deltas are rendered separately.
User content is either plain text or an ordered OpenAI content array containing
text, inline data-URL images, and Base64 audio. Media is held in memory so a
resident continuation can reproduce the exact prior request; gem16 limits one
file to 10 MiB and the conversation to 14 MiB of Base64 payload below the
server's 16 MiB request limit. On Linux the microphone path prefers the default
PipeWire source through `pw-record`, temporarily lowers an overdriven source
volume and restores it afterward. The first click asks `wpctl` to resolve the default source immediately; if
PipeWire/WirePlumber is still activating, it polls every 100 ms for at most two seconds and only then starts one
`pw-record` process. An already-active source returns on the first check without polling. Recording is never opened
proactively at app startup. Systems without both PipeWire CLI tools use a ranked Java Sound input.
Both paths reject silent or clipped recordings, normalize valid PCM16 audio,
and wrap it as WAV without writing a temporary file. Changing or removing
history starts a new resident root because an existing KV cache cannot be
rolled back safely.

Sampling remains a server-level choice, matching gem16's current strict API:

- default server startup uses checkpoint sampling (`temperature=1`, `top_k=64`,
  `top_p=0.95`);
- selecting **Greedy** on the Server screen adds `--greedy`;
- the UI never sends unsupported per-request sampling fields.

## Security and privacy

All model requests target the configured endpoint. The default is
`http://127.0.0.1:8080/v1`. Model downloads contact only pinned immutable files
on `huggingface.co`; gem16 never executes repository code or transmits
telemetry. A token entered in the UI remains in memory and is not written to
gem16 settings. Settings contain cache-backed paths and server configuration.
