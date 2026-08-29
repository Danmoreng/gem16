# gem16 Native Studio

gem16 Studio is a C++20 Dear ImGui desktop client for Linux and Windows. It keeps CUDA and model weights in the
separate `gem16-server` process and communicates through the bounded OpenAI Agent Core v1 HTTP/SSE interface. It is
the only active GUI and owns all new product and release work. The Kotlin/Compose application in `studioApp/` is
deprecated and retained temporarily as read-only migration evidence; see [`PRODUCT_CONTRACT.md`](PRODUCT_CONTRACT.md).

## Native architecture

| Layer | Linux | Windows |
|---|---|---|
| Window and input | GLFW 3.4 + GTK file dialogs | Win32 |
| Renderer | OpenGL 3.3 | Direct3D 11 |
| Immediate UI | Dear ImGui 1.92.9b | Dear ImGui 1.92.9b |
| Animated background | Native GLSL full-screen pass | Equivalent HLSL full-screen pass |
| Server process | `fork`/`exec`, pipe capture | `CreateProcessW`, pipe capture |
| Chat transport | `cpp-httplib` HTTP/1.1 and SSE | `cpp-httplib` HTTP/1.1 and SSE |

The visual foundation is adapted from `Free-Solace-ImGui-Interface` commit
`bb35bb3f11ef390fa94ca4aa57daa0a6ee379e67` under MIT. The app reuses the cross-platform ImGui backend approach and
shader aesthetic, not the example's authentication screens, image assets, brands, avatars, or glass cursor. The
normal operating-system cursor remains enabled. Attribution is kept in `nativeStudio/licenses/` and Dear ImGui's
license remains beside the vendored source.

## Current product slice

- Chat, Models, Server, and Settings screens with a dark/light glass palette, procedural gemstone branding, and the
  original animated GPU science-fiction wave.
- Neutral first-run onboarding and persisted selectors for qualified Gemma 4 12B Unified and qualified text-only
  Gemma 4 26B A4B. Either profile or both can be installed.
- One generated, lock-derived catalog for the independently pinned 12B Target, 12B Assistant, 26B Target, and 26B
  Assistant components.
- Resumable, SHA-256-verified anonymous downloads with a preflight storage check. Source payloads and immutable
  snapshots remain in the standard Hugging Face cache selected through `HF_HUB_CACHE`, `HF_HOME`, platform cache
  conventions, or the repository fallback.
- The cross-repository 12B Target receives a hardlink-only composed runtime view below the same Hub cache. It reuses
  canonical verified blobs and does not duplicate model payloads.
- Qualified 26B Target and Assistant paths, one resident session, 86,016-token MTP context, and fixed D2 selected
  as its selected MTP profile.
- Managed `gem16-server` start, stop, output capture, health polling, and non-owning attachment to an external server.
- Incremental SSE rendering for answer and `reasoning_content`, cancellation, resident session IDs, and new-chat
  reset. Starting a new chat omits the old session ID; the one-slot 26B server then evicts the inactive root.
- Selectable Markdown rendering for headings, emphasis, strong text, inline and fenced code, ordered/unordered lists,
  quotes, rules, links, and image labels. Fenced code blocks expose their own copy action.
- Reasoning is available per assistant response, collapsed by default, and expandable without affecting the answer.
- Desktop-style wrapped response selection with mouse drag, double-click word selection, Shift extension,
  `Ctrl+A`/`Ctrl+C`, a context menu, per-message copy feedback, and equivalent selectable/copyable server logs.
- A compact, dynamically growing composer uses `Enter` to send and `Shift+Enter` for a line break. Undo removes the
  most recent user/assistant exchange and invalidates the resident session; Delete clears the complete chat. The
  composer itself never exposes an outer scrollbar.
- Persistent server, generation, and theme settings under `$XDG_CONFIG_HOME/gem16/studio.conf`,
  `~/.config/gem16/studio.conf`, or `%APPDATA%\gem16\studio.conf`.
- Normal operating-system window chrome and cursor on both platforms.
- Platform-aware interface scaling with a 125% Linux minimum in `Auto` mode, explicit 100/125/150% choices, scaled
  procedural icons and responsive stacked layouts for narrower windows.
- Composer file selection and desktop drag-and-drop for bounded UTF-8 text/code documents, PNG/JPEG/BMP images, and
  WAV/FLAC/MP3 audio. Images and audio use OpenAI-compatible content parts on 12B Unified; the text-only 26B profile
  rejects them before a request is sent. PDF text extraction is enabled when the mature Poppler `pdftotext` utility
  is installed and otherwise fails visibly with an export-to-text instruction.
- Cross-platform microphone capture through the already pinned miniaudio boundary. Recordings are bounded to five
  minutes and attached as mono 16 kHz WAV input.
- Chat autoscroll that disengages when the user reads earlier output, a `Jump to latest` action, context usage,
  streamed progress, terminal usage statistics, and quick thinking-effort controls.
- Model cache open/reverify actions, automatic managed-server restart after a profile switch, server preflight
  status, path pickers, auto-sized errors, and responsive configuration/log panels.

This native slice covers text and 12B multimodal chat plus two-profile onboarding but does not yet satisfy the full product contract.
Complete real-model downloads and clean-machine installation still require equal Windows and Linux qualification.
Bundled PDF extraction, local time tools, exact server-side metrics cards, model-payload reclamation, and native
installers remain open work. The 26B profile is intentionally text-only. Deprecated Compose behavior is input to a
new native product decision, not a parity requirement and not a reason to continue modifying the old GUI.

## Build and run

Requirements are CMake 3.28+, a C++20 compiler, and a current `gem16-server`. Linux additionally needs OpenGL, GTK 3,
and the GLFW build dependencies for X11 or Wayland; GLFW 3.4 is pinned and fetched by CMake. Poppler `pdftotext` is
an optional runtime dependency for PDF attachments. Windows uses platform SDK Direct3D 11 libraries and requires no
GLFW dependency.

Linux:

```bash
./scripts/run-studio.sh
```

Windows PowerShell:

```powershell
.\scripts\run-studio.ps1
```

Both scripts rebuild the CUDA server and native Studio incrementally. Use `--skip-server-build` on Linux or
`-SkipServerBuild` on Windows when the workspace server is already current. Direct native build and test:

```bash
cmake -S nativeStudio -B build/native-studio -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build/native-studio --parallel
ctest --test-dir build/native-studio --output-on-failure
./build/native-studio/bin/gem16-studio
```

The native host test exercises the generated four-component model catalog, neutral first-run settings, empty-cache
storage accounting, exact 26B launch arguments, and an in-process SSE endpoint, including reasoning, answer text,
terminal marker, and resident session-ID propagation. It also drives the real selectable-text ImGui widget and
verifies that click plus `Ctrl+A`/`Ctrl+C` reaches the platform clipboard callback, including UTF-8 helper coverage.
It also covers Markdown parsing and streaming fences, undo-turn history behavior, the multimodal OpenAI request
shape and usage events, UI-scale policy/persistence, and the real ImGui `Enter`/`Shift+Enter` composer interaction.
It does not download a full model, open a microphone, or load a GPU.

## Packaging

The current packaging scripts produce development-preview portable archives containing `gem16-studio`, the
already-built CUDA `gem16-server`, the central `VERSION`, and available licenses/notices. Checkpoints remain external.
The archives are not yet described as self-contained: Linux runtime dependencies, complete notices/manifests, and
clean-machine installation still need qualification on both equal product platforms.

```bash
./scripts/package-studio.sh
```

```powershell
.\scripts\package-studio.ps1
```

Outputs are `build/packages/gem16-linux-x64.tar.gz` and `build/packages/gem16-windows-x64.zip`. The Windows release
workflow packages native Studio; an equal Linux release workflow, MSI/DEB installation, complete package manifests,
and two-platform clean-machine smoke evidence remain open. Historical Compose packaging details are preserved only
as deprecated evidence in [`legacy/KOTLIN_COMPOSE_STUDIO.md`](legacy/KOTLIN_COMPOSE_STUDIO.md).

## Security and lifecycle

- Model paths and server executable paths are passed as direct argument arrays, never through a shell.
- Attachment files are size-bounded before reading; text must be valid UTF-8. `pdftotext` receives paths as direct
  process arguments and writes only inside a newly created private temporary directory.
- Server logs are held in a bounded 1,000-line memory ring.
- Studio terminates only a process it started. A healthy external server is labeled `Attached` and is never stopped.
- The server defaults to loopback and still has no TLS or authentication; do not bind it to an untrusted network.
- Settings contain paths and UI preferences only. Tokens and credentials are neither requested nor persisted by this
  native slice.
- Closing or cancelling a stream closes the client connection; the server's existing cancellation and session
  lifecycle rules remain authoritative.
