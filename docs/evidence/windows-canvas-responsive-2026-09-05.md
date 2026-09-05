# Windows Canvas loading and responsive viewport

Based on `56e83c3906d89a7e4941af37aa3a768e912ae088` plus existing local fixes.
The user's address-form HTML reproduces a blank preview with navigation failure.
WebView2 reports NavigateToString's host-generated bootstrap as
`data:text/html;charset=utf-8;base64,...`; the previous about-only navigation
filter cancels it (WebErrorStatus 14). Windows now admits exactly the generated
bootstrap URI once, only for the top-level view. Subsequent navigation, frame
navigation, CSP network/file restrictions and opaque-origin sandbox remain in place.
Submission and navigation failures now include error codes.

Per owner direction, the viewport follows the panel's available width and height
in framebuffer pixels, rather than a fixed 1024x768 image scaled to 4:3. Resize
updates the existing browser without reloading the document. Asynchronous captures
from earlier sizes are discarded. Windows explicitly uses raw pixel coordinates;
the GUI and CLI smoke initialize process DPI awareness consistently. Mouse mapping
uses the displayed image dimensions. Screenshot attachment metadata, review prompt
and tool result report the actual captured viewport. Existing 16-megapixel decode
and 8 MiB PNG limits remain; no model or CUDA settings change.

Windows previously ignored the Linux-only `--canvas-smoke` argument. Its entry
point now supports it, and a Windows CTest invokes the real installed WebView2.
The test recognizes each engine's actual deliberate JavaScript-error diagnostic,
checks 1024x768 pixels, blocked file/network access, mouse interaction, then resizes
to 800x600 and 1440x900 while retaining the clicked state, and checks revision
invalidation, SVG diagnostics and close. All four native Studio CTest groups pass.

Commands:

```powershell
cmake --build build/Windows/native-studio --config Release --parallel 6
ctest --test-dir build/Windows/native-studio -C Release --output-on-failure
```

The retained user's saved address-form Canvas was also observed rendering in the
small Studio window and after maximization. The HTML source was not modified.
Linux code was updated for the same resize API but not built or live-tested on
this Windows host. This is bounded Studio validation, not full release qualification.

## Follow-up: remove outer preview scrolling

The inline iframe left a text baseline below its full-height box, causing an
additional outer scrollbar. The host now uses a block, absolutely positioned
iframe and clips overflow on its own html/body only. Generated page CSS is
unchanged. The real browser smoke also checks the right/bottom edge pixels of a
short page for outer overflow and scrolls a separate long page via mouse wheel,
requiring both its scroll event and changed screenshot pixels. The rebuilt
Windows Studio passes all four CTest groups with this regression included.

## Final Windows path: direct live embedding

Owner feedback rejected screenshot-based interactive display. Windows Studio now
creates a windowed WebView2 controller in a child HWND at the Canvas panel's client
coordinates. The browser handles native mouse/keyboard input and rendering; it is
hidden when the panel is not presented. The previous composition controller remains
only for the standalone diagnostic smoke, not the normal Windows Studio display.
The live path skips periodic PNG capture, decoding and ImGui texture upload.
`canvas_check` explicitly requests a screenshot when needed. Diagnostic observation
is throttled to 100 ms. This removes the prior screenshot-display pipeline; no
measured frame-rate or latency improvement is claimed.

The smoke additionally exercises the actual windowed controller: child placement,
no unsolicited PNG capture, on-demand screenshot, resize and hiding. All four
CTest groups passed in 13.21 seconds. The owner subsequently tried the new Studio
and accepted the interaction as substantially better. Automated UI inspection of
the final dropdown was interrupted by the owner's Escape key; it is not claimed
as a completed automated check.

### Linux handoff

Linux still uses the GTK snapshot path in `canvas_browser.cpp`, and the fallback
image display/input forwarding in `app_canvas.cpp`. Replace that display with real
WebKitGTK embedding in the GLFW/OpenGL application, accounting for the actual Linux
window system. Preserve the sandbox, dynamic viewport, document state on resize,
native dropdown/keyboard behavior and explicit screenshot-only review contract.
The Linux resize changes in this commit have not been compiled or exercised here.
Do not copy the Windows HWND implementation or claim Linux live rendering is done.
