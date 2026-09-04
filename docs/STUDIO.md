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

**Stop** interrupts an in-flight request. Partial text remains visible and copyable;
failed or cancelled exchanges are excluded from future context until retried.
Retry resends the exchange using a fresh session. **Undo** removes the last exchange;
**Delete** confirms removal of the saved conversation. Starting a new chat resets the resident session.

Chats now persist locally in SQLite. The sidebar restores conversations across
restarts and supports rename, pin, archive and deletion. **Temporary** creates a
chat whose content is never written to the chat store. User messages are committed
before dispatch; streamed answers checkpoint about once a second and on completion.
Retry preserves the previous answer attempt. Reopening a chat rebuilds server
context on continuation and restores its generation settings. A different model
installation or missing attachment blocks continuation with an explanation.
The sidebar searches titles, messages, reasoning and attached documents and jumps
to matching turns. Export produces JSON/Markdown plus attachments; backup creates
a complete restorable chat bundle. Cleanup removes unreferenced chat attachments.
See [chat storage](STUDIO_STORAGE.md) for paths, durability, restore and limits.

## Models and verification

`HF_HUB_CACHE` and `HF_HOME` select the standard shared Hugging Face cache.
Cross-repository runtime views contain hardlinks to verified blobs, not duplicate weights.
Removing a profile's runtime view preserves shared Hub blobs used elsewhere.

**Verify again** performs a fresh SHA-256 scan of installed payloads and runtime
views, including same-size corruption. It runs in the background with progress
and cancellation. Ordinary refresh uses cached installation state. Runtime
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
