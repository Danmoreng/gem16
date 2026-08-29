# gem16 product contract

**Product baseline:** 2026-08-29 · **Version line:** 0.2.x development preview

This document is the owner-approved product contract. It defines what gem16 is
being productized into. Current implementation gaps remain visible in the
roadmap and must not be hidden by broader capability claims.

## Product

gem16 is a local Gemma inference product for one user on one Windows or Linux
machine with a supported approximately 16 GB NVIDIA Blackwell GPU. It consists
of two independently runnable programs:

- `gem16-server`, the resident inference server and OpenAI Agent Core v1 API;
- `gem16-studio`, the native Dear ImGui application for model installation,
  server management, diagnostics, and local chat.

The server remains a separate process. Studio may start and stop a server it
owns or attach non-destructively to an already running local server.

## Equal product platforms

Windows x64 and Linux x86-64 are equal product platforms. A product release is
not complete until both platforms satisfy the same applicable source, host,
GPU, API, Studio, packaging, and clean-machine smoke gates.

Platform-native implementation details may differ. Windows currently uses
Win32 and Direct3D 11; Linux uses GLFW and OpenGL. Installer formats and system
dependency handling may also differ, but neither platform is a secondary or
best-effort port.

## Equal model profiles

Gemma 4 12B Unified and Gemma 4 26B A4B are equal, user-selectable product
profiles. Studio must present both without labeling one as the preferred or
default product choice, and users may install either or both side by side.
Selection, download, verification, launch configuration, status, and removal
must follow one coherent model-management experience.

Equal product status does not imply identical model capabilities:

| Profile | Qualified product capability |
|---|---|
| Gemma 4 12B Unified | Text, image, and audio input; up to two resident execution slots on the qualified profile |
| Gemma 4 26B A4B | Text-only; one resident slot; optional separately pinned fixed-D2 Assistant |

Studio and the server must report these differences before execution. They may
not silently substitute a profile, Assistant, precision, context limit, or
fallback path. Model files remain external to application archives and must be
resolved from immutable locks, downloaded resumably, and verified before use.

The current native Studio can install the qualified 26B pair but still lacks
native 12B installation, and a fresh settings file still seeds the 12B profile
instead of asking for a neutral first-run choice. Closing both gaps is a product
gate, not optional polish.

## Local-machine boundary

The initial product is for one local user on one machine:

- server networking defaults to loopback;
- no cloud account or remote control plane is required by gem16;
- authentication and TLS are not part of the initial loopback contract;
- non-loopback exposure requires an explicit later security contract or a
  trusted reverse proxy and must never happen silently;
- model credentials are used only for model acquisition and are not embedded
  in checkpoints, logs, settings, or release archives.

The existing bounded request queue, privacy-safe logging, graceful shutdown,
health/readiness/liveness endpoints, and operational metrics remain required.

## API contract

The server product API is the explicitly bounded
[OpenAI Agent Core v1](OPENAI_AGENT_CORE_V1.md). It supports the OpenAI Chat
Completions and Responses shapes needed for streamed text generation and
client-executed function-tool loops. “OpenAI-compatible” in product material
means compatible with that published subset, not complete emulation of every
OpenAI platform endpoint, field, tool category, or storage behavior.

Both qualified model profiles must pass the same applicable Agent Core
protocol suite. Capability-specific media tests remain separate.

## One active GUI

`nativeStudio/` is the only active GUI implementation. It owns all new UI,
model-management, server-management, packaging, and release work.

The Kotlin/Compose implementation in `studioApp/` is deprecated as of this
baseline. It is retained temporarily as read-only migration evidence and must
not receive new product features, release integration, or parity fixes. After
the native Studio passes the first complete two-platform product release, the
Compose source and Gradle-only root infrastructure may be removed from the
default branch; its final state remains available through Git history and the
legacy documentation.

## Release boundary

A release candidate requires, on both Windows and Linux:

1. a build from the central repository `VERSION` and an immutable source
   revision;
2. successful host, native Studio, protected 12B, qualified 26B, server
   lifecycle, and Agent Core v1 gates;
3. a versioned package containing Studio, server, required runtime components
   or an exact declared system-dependency contract, licenses, notices, hashes,
   and a machine-readable manifest;
4. a clean-machine first-run smoke for installing either model profile,
   starting it, chatting, exercising a function tool, stopping, and restarting;
5. explicit hardware, driver, model-lock, capability, context, and known-limit
   reporting.

Code signing, native installers, update delivery, SBOM generation, and broader
network deployment are follow-up release-hardening work unless promoted by a
later owner decision. Portable archives may be development previews, but must
not be described as self-contained unless that is verified on a clean machine.

## Non-goals for this baseline

- a generic multi-model or multi-backend framework;
- cloud hosting or multi-tenant serving;
- continuous batching or a many-client production service;
- full OpenAI platform parity;
- changing qualified precision, cache, sampling, or model behavior merely to
  satisfy an adapter or packaging goal;
- resuming the frozen 26B decode optimization campaign without a new decision.

## Decision precedence

Permanent safety and integrity rules in `AGENTS.md` remain binding. For product
scope, this contract and `docs/ACTIVE_DECISIONS.md` supersede older roadmap,
milestone, GUI, branch, and “12B default / 26B experimental” wording. Historical
records remain evidence and are not rewritten.
