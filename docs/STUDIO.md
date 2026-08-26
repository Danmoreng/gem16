# gem16 Native Studio

gem16 Studio is a C++20 Dear ImGui desktop client for Linux and Windows. It keeps CUDA and model weights in the
separate `gem16-server` process and communicates through the existing OpenAI-compatible HTTP/SSE interface. The
native client replaces the Kotlin/Compose application as the default development launcher while the previous source
remains in `studioApp/` as a parity reference during migration.

## Native architecture

| Layer | Linux | Windows |
|---|---|---|
| Window and input | GLFW 3.4 | Win32 |
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
- Persisted Gemma 4 12B Unified and experimental Gemma 4 26B A4B profiles.
- Correct 26B compiled Target and Assistant paths, one resident session, 32K default context, and fixed D1/D2/D4
  controls with D2 selected by default.
- Managed `gem16-server` start, stop, output capture, health polling, and non-owning attachment to an external server.
- Incremental SSE rendering for answer and `reasoning_content`, cancellation, resident session IDs, and new-chat
  reset. Starting a new chat omits the old session ID; the one-slot 26B server then evicts the inactive root.
- Desktop-style wrapped response selection with mouse drag, double-click word selection, Shift extension,
  `Ctrl+A`/`Ctrl+C`, a context menu, per-message copy feedback, and equivalent selectable/copyable server logs.
- Persistent server, generation, and theme settings under `$XDG_CONFIG_HOME/gem16/studio.conf`,
  `~/.config/gem16/studio.conf`, or `%APPDATA%\gem16\studio.conf`.
- Normal operating-system window chrome and cursor on both platforms.

This first native slice is text-chat complete but does not yet claim feature parity with Compose. The old app's model
downloader, Markdown/code-block renderer, document/PDF extraction, image/audio attachments, microphone capture,
local time tools, exact metrics cards, and MSI installer remain follow-up ports. The 12B server still supports its
multimodal capabilities; the current native UI sends text only. The 26B profile is intentionally text-only.

## Build and run

Requirements are CMake 3.28+, a C++20 compiler, and a current `gem16-server`. Linux additionally needs OpenGL and
the GLFW build dependencies for X11 or Wayland; GLFW 3.4 is pinned and fetched by CMake. Windows uses platform SDK
Direct3D 11 libraries and requires no GLFW dependency.

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

The native host test exercises the exact 26B launch arguments and an in-process SSE endpoint, including reasoning,
answer text, terminal marker, and resident session-ID propagation. It also drives the real selectable-text ImGui
widget and verifies that click plus `Ctrl+A`/`Ctrl+C` reaches the platform clipboard callback, including UTF-8 helper
coverage. It does not load a model or GPU.

## Packaging

The current packaging scripts produce self-contained portable archives containing `gem16-studio`, the already-built
CUDA `gem16-server`, and third-party notices. Checkpoints remain external.

```bash
./scripts/package-studio.sh
```

```powershell
.\scripts\package-studio.ps1
```

Outputs are `build/packages/gem16-linux-x64.tar.gz` and `build/packages/gem16-windows-x64.zip`. Native MSI/DEB
installation and the corresponding Windows release-workflow migration remain open; the historical Compose packaging
contract is preserved in [`legacy/KOTLIN_COMPOSE_STUDIO.md`](legacy/KOTLIN_COMPOSE_STUDIO.md).

## Security and lifecycle

- Model paths and server executable paths are passed as direct argument arrays, never through a shell.
- Server logs are held in a bounded 1,000-line memory ring.
- Studio terminates only a process it started. A healthy external server is labeled `Attached` and is never stopped.
- The server defaults to loopback and still has no TLS or authentication; do not bind it to an untrusted network.
- Settings contain paths and UI preferences only. Tokens and credentials are neither requested nor persisted by this
  native slice.
- Closing or cancelling a stream closes the client connection; the server's existing cancellation and session
  lifecycle rules remain authoritative.
