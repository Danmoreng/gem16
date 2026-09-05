# Windows Studio context and status-menu fixes — 2026-09-05

Parent: `56e83c3906d89a7e4941af37aa3a768e912ae088`, with the earlier local
WebView2 include-order fix. This is development evidence, not a release or
performance qualification.

## Reproduction and changes

The actual Studio-managed Compact Vision server failed at the saved 228,120
context: `model_load_failed`, `free=0`, `required_slot=15261390560`,
`probe_resident=12204692480`, `required_margin=419430400`. The earlier successful
large-capacity CLI smoke did not establish Studio-managed server admission.

Windows Compact Vision profile defaults now use 65,536 tokens, leaving room for
desktop activity. Linux defaults, supported maxima, model formats, MTP and
runtime admission margins are unchanged. Existing saved settings are not
automatically clamped. For this user's requested repair, `studio.conf` was backed
up to `build/studio-before-context-fix-20260905.conf`, then its context was set to
65,536. Larger contexts remain explicitly configurable; there is no runtime
retry or silent fallback.

Studio now preserves model-load errors instead of replacing them with only
`exited with code 2`. Memory failures recommend reducing Context tokens; the
original detailed message remains in the server log. Log delivery also updates
the error when the final line arrives after the process exit callback.

The Server controls popup is explicitly anchored above its footer trigger with
a bottom-left pivot and a bounded width. Its position is refreshed each frame,
including after viewport resizing, rather than depending on automatic popup
placement and the mouse position.

## Validation

```powershell
cmake --build build/Windows/native-studio --config Release --parallel 6
ctest --test-dir build/Windows/native-studio -C Release --output-on-failure
```

All three Studio groups pass (9.89 seconds). New coverage exercises the Windows
default and emitted context argument, actionable versus non-memory errors, and
the real rendered Studio popup at viewport heights 1,050 and 1,250 with the mouse
away from its anchor. The tests use the existing isolated lifecycle fixture.

Live Windows checks with Studio open:

- The rebuilt Studio automatically starts its managed server at 65,536,
  fixed-D2, with FP8 Vision. `/health` reports `ok` and 1,454,374,912 free device
  bytes after admission. Checkpoint sampling defaults are retained.
- The status popup appears directly above the Running button in the maximized
  window; visually inspected using the native Windows UI.
- A separate 131,072-context server with Studio open passed the existing scene
  image's four checks and the text arithmetic smoke. It left only 714,080,256
  bytes free at admission, motivating the more conservative default.

Local logs: `build/windows-studio-context-menu-{build,tests}-20260905.log`.
Live health and response: `build/windows-gui-64k-health-20260905.json` and
`build/windows-gui-managed-chat-20260905.json`. The 128K command, server log and
image/text responses are retained in `build/smoke_gui_128k_20260905.py` and
`build/windows-gui-128k-*`. No CUDA implementation changed; no new throughput,
long-context quality or peak-memory claim is made.
