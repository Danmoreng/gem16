# gem16 Studio

gem16 Studio is a Kotlin Compose Desktop application for local Gemma 4 chat and
`gem16-server` lifecycle management. It follows the desktop structure proven by
the neighboring Qwen-TTS Studio while remaining an HTTP client: the JVM does
not load CUDA or model weights directly.

## Features

- streamed Chat Completions with separate reasoning and answer presentation;
- CommonMark headings, emphasis, lists, quotes, links, and fenced code rendering;
- ordered PNG/JPEG/BMP image and WAV/FLAC/MP3 audio attachments;
- in-memory microphone recording with live level, timer, cancel, and automatic
  stop after the server's 30-second audio limit;
- Enter-to-send with Shift+Enter for multiline input;
- resident multi-turn sessions through `X-Gem16-Session-Id`;
- selectable off/low/medium/high thinking budgets;
- cancellation, new-chat, usage, and finish-reason state;
- automatic managed-server startup plus explicit start/stop on Linux and Windows;
- attachment to a compatible externally started server;
- model, assistant, MTP, context, session, host, port, and sampling settings;
- live `/health` state and bounded process logs;
- persistent settings under `~/.gem16-studio/settings.properties`;
- dark and light themes.

Tool-call interaction, Responses history, sampled video frames, and remote
authenticated deployments remain follow-ups. Markdown parsing uses the pinned
BSD-2-Clause `org.commonmark:commonmark` dependency; raw HTML is displayed as
text rather than executed.

## Build and run

Requirements:

- JDK 21;
- network access for the first Gradle dependency resolution;
- a built SM120/SM120a `gem16-server`;
- the pinned target checkpoint and optional assistant checkpoint.

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

Compose packaging can produce DMG, MSI, and DEB images. The package contains the
JVM application, not the 9.3 GB checkpoint or CUDA engine binary. Configure an
existing engine/model installation on the Server screen.

## Managed server behavior

Studio probes the configured endpoint at application startup. It attaches when
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
Studio stops only the process it created. A server discovered through `/health`
is labeled external and is never terminated by the UI.

The server has no built-in authentication or TLS. Studio defaults to loopback
and visibly warns for any other bind address.

## Chat protocol

The client sends the complete visible conversation to
`POST /v1/chat/completions`, requests SSE, and retains the opaque session ID
returned by gem16. Text and `reasoning_content` deltas are rendered separately.
User content is either plain text or an ordered OpenAI content array containing
text, inline data-URL images, and Base64 audio. Media is held in memory so a
resident continuation can reproduce the exact prior request; Studio limits one
file to 10 MiB and the conversation to 14 MiB of Base64 payload below the
server's 16 MiB request limit. On Linux the microphone path prefers the default
PipeWire source through `pw-record`, temporarily lowers an overdriven source
volume and restores it afterward. Other systems use a ranked Java Sound input.
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
`http://127.0.0.1:8080/v1`. Studio does not download models, execute model
repository code, or transmit telemetry. Settings contain local paths and server
configuration only.
