# gem16 Studio

Studio is the native Windows/Linux desktop application for model installation,
local server management and streamed chat. Inference runs in a separate
`gem16-server` process. Follow the [source installation](../README.md#run-the-desktop-app-from-source)
to launch it.

## First run

1. Open **Models** and choose 12B Unified or 26B Compact Vision. Neither is preselected;
   either or both may be installed.
2. Install the selected profile's components. Studio checks disk capacity, resumes
   downloads and verifies locked SHA-256 hashes in the shared Hugging Face cache.
3. Open **Server**, check the executable and model paths, then start the server.
   Studio can also attach to an already running compatible local server.
4. Open **Chat**. `Enter` sends; `Shift+Enter` inserts a newline.

On subsequent launches, Studio starts the server with the saved configuration.
Disable **Start server when Studio opens** on the Server screen to opt out.
Autostart never chooses a model before first-run selection. A failed start appears
in the sidebar status and Server logs; Studio does not substitute another profile.
Click the status at the bottom left for **Start**, **Stop**, **Restart**, or logs.
Stop/restart cancels the current answer and clears its GPU session; the saved chat
remains available. These controls only stop processes started by Studio. An
attached external server is controlled by its owner.

| Profile | Inputs | Execution |
|---|---|---|
| 12B Unified | Text, image, audio | Optional Assistant; up to two resident slots subject to VRAM |
| 26B Compact Vision | Text and one image; no audio | One slot; optional fixed-D2 on the validated composite |

Compact Vision offers 70/140/280 image budgets with preview, estimate and remove
controls. D2 requires the server's matching live capability. See the
[26B guide](GEMMA4_26B.md) for context limits.

## Chat and recovery

Answers support selectable Markdown, code copying, bounded SVG previews and math.
Reasoning is collapsed by default. Web links open only on explicit interaction;
HTML is inert and remote images are not automatically fetched.
The context meter adds the server's input and output usage, including reasoning.
It shows **Pending** until usage arrives at completion, also after reopening a
saved chat until its next request. Stream chunks are not treated as exact tokens.

**Stop** interrupts an in-flight request. Partial text remains visible and copyable;
failed or cancelled exchanges are excluded from future context until retried.
Retry resends the exchange using a fresh session. **Undo** removes the last exchange;
**Delete** confirms removal of the saved conversation. Starting a new chat resets the resident session.

Chats persist automatically in local SQLite. The chat window contains the conversation
and composer; titles appear only in the sidebar, with no title editor or maintenance
toolbar. Saving is silent unless it fails. The sidebar restores conversations across
restarts and supports search and opening previously archived chats. **Temporary** creates a
chat whose content is never written to the chat store. User messages are committed
before dispatch; streamed answers checkpoint about once a second and on completion.
Retry preserves the previous answer attempt. Reopening a chat rebuilds server
context on continuation and restores its generation settings. A different model
installation or missing attachment blocks continuation with an explanation.
The sidebar searches titles, messages, reasoning and attached documents and jumps
to matching turns. Export, backup and attachment cleanup remain internal storage
operations with host-test coverage; they are not exposed in the chat interface.
See [chat storage](STUDIO_STORAGE.md) for paths, durability, restore and limits.

## Models and verification

`HF_HUB_CACHE` and `HF_HOME` select the standard shared Hugging Face cache.
Cross-repository runtime views contain hardlinks to verified blobs, not duplicate weights.
Removing a profile's runtime view preserves shared Hub blobs used elsewhere.

**Verify again** performs a fresh SHA-256 scan of installed payloads and runtime
views, including same-size corruption. It runs in the background with progress
and cancellation. Ordinary refresh uses cached installation state. Components distinguish missing,
partial/resumable, unverified, damaged and verified files. **Install / resume / repair**
reuses valid shared blobs; a detected hash failure remains visible across refreshes.
Cards show unique profile payloads and bytes reusable across catalog profiles.

**Remove profile views** preserves shared blobs and components required by another
installed profile. **Review unused cache** previews removable known files; cleanup
rechecks references, hardlinks and download locks and keeps files changed since review.
It preserves Hub snapshots and files belonging to unknown cache clients.

Model activation saves the complete selection atomically and retains the previous
selection for rollback. A chat can install/restore its exact current-catalog model.
Unknown, custom or no-longer-qualified revisions require explicit configuration and
are never substituted automatically. Downloads stay pinned to the build's qualified
catalog; no migration to the normalized 26B Hub layout is implied. Runtime
startup does not repeat full multi-gigabyte payload hashing.

## Settings and diagnostics

Settings include server location, model paths, context, generation, system prompt,
reasoning and appearance. Configuration lives in `$XDG_CONFIG_HOME/gem16/studio.conf`,
`~/.config/gem16/studio.conf`, or `%APPDATA%\gem16\studio.conf`.
The Server screen exposes status and copyable logs. Loopback is the supported
network boundary; see [server operations](SERVER.md).

## Packaging and development

After building the CUDA server, `scripts/package-studio.sh` or
`scripts/package-studio.ps1` builds the platform archive. Models remain external.
These are development previews: equal two-platform dependency, large-download
and clean-machine qualification remains a release gate.

[Implementation details](development/STUDIO_INTERNALS.md) cover rendering,
transport, limits, build dependencies and tests. `nativeStudio/` is the only active
GUI. Deprecated `studioApp/` is retained as migration evidence until the first
complete two-platform release, as required by the [product contract](PRODUCT_CONTRACT.md).
