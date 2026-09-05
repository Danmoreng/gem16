# Linux live Canvas — 2026-09-05

## Scope and source

Owner request: pull the Windows work from origin, build on Linux and embed a live
WebView in Canvas instead of displaying screenshots.

Clean `main` fast-forwarded from `56e83c3` to
`a67c37fb52883f1c09c878fbd307df54c4bf8659` using
`git pull --ff-only origin main`. The changes described here are an uncommitted
working-tree patch on that revision. No inference/runtime or model files were edited.

## Implementation

- `nativeStudio/src/main.cpp`: select GLFW X11, register its native window and
  return keyboard focus to ImGui when clicking outside the WebKit child.
- `nativeStudio/src/canvas_browser.cpp`: reparent a real WebKitGTK child surface
  into the foreign X11 host, route native input, preserve state when resizing,
  and map/unmap once at frame completion. A dedicated process-lifetime GDK display
  uses physical pixels without changing GTK file-dialog scaling. Normal live
  presentation requests no PNGs; `canvas_check` explicitly requests snapshots,
  including when the preview is hidden. Snapshot completion checks revision and
  actual dimensions. Diagnostic observations remain throttled to 100 ms.
- `nativeStudio/src/app.cpp` and `canvas_browser.h`: pass popup visibility at frame
  completion so Linux child windows cannot cover ImGui menus or dialogs. Windows
  presentation behavior is unchanged.
- `nativeStudio/CMakeLists.txt` and `cmake/webview-dependency.cmake`: require GLFW's
  X11 backend and link Xlib explicitly. WebKit remains the installed system runtime.
- `docs/STUDIO.md` and `docs/development/STUDIO_INTERNALS.md`: document live input,
  the X11/XWayland dependency and desktop smoke commands.

## Environment and checks

Linux X11 desktop, `DISPLAY=:0.0`; GTK 3.24.52, WebKitGTK 2.52.6,
CMake 4.4.3, GCC 16.2.1 (20260810), pinned GLFW 3.4.

Commands (from repository root):

```sh
git pull --ff-only origin main
cmake --preset blackwell-release
cmake --build --preset blackwell-release --target gem16-server --parallel 8
build/Linux/blackwell-release/bin/gem16-server --help
cmake --preset host-debug
cmake --build --preset host-debug --parallel 8
ctest --preset host-debug --output-on-failure
cmake -S nativeStudio -B build/native-studio -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build/native-studio --parallel 8
ctest --test-dir build/native-studio --output-on-failure
build/native-studio/bin/gem16-studio --canvas-smoke build/native-studio/linux-live-smoke.png
GDK_SCALE=2 build/native-studio/bin/gem16-studio --canvas-smoke build/native-studio/linux-live-hidpi-smoke.png
git diff --check
```

Results: server and Studio builds succeeded; server help executed; 5/5 host tests
and 3/3 Studio test groups passed. Both desktop smokes exited 0.

The extended smoke checks the actual X11 parent and child placement, JavaScript
animation without PNG capture, real XTEST mouse and keyboard input, 800×600 and
900×700 on-demand screenshots, state preservation, hidden capture, popup hiding,
show/hide and close with pending callbacks. It uses a GTK-created foreign X11 host
through the same native-handle boundary used by GLFW, rather than a full Studio
chat session. The earlier diagnostic path additionally checks 1024×768 PNG pixels,
800×600/1440×900 resize, network/file isolation, JS/SVG diagnostics and inner
scrolling. Its Linux scroll adapter now preserves wheel magnitude. The scrolling
pixel assertion samples inside the viewport to exclude WebKit's native focus
indicator at the top edge; exact RGB assertions and the separate no-overflow edge
assertions remain in place.

Local raw logs and PNGs are under `build/native-studio/linux-live-*`;
repository host logs are `build/linux-live-host-build.log` and
`build/linux-live-host-tests.log`. WebKit emitted D-Bus name-release warnings when
replacing diagnostic renderer processes; the completed smokes had no GTK critical
or X11 errors. Existing Studio compiler warnings were not part of this change.

## Limits

- Actual desktop execution was X11. `GDK_SCALE=2` checks scaling on that desktop;
  it does not claim a separate Wayland-compositor run. Wayland desktops require
  XWayland; native Wayland-only operation is not implemented.
- Windows was not rebuilt here; no Windows rendering implementation was changed.
- No real model was loaded and no GPU numerical/performance or release claim is
  made. The updated CUDA server was rebuilt, not performance-qualified.
- XTEST (`libXtst.so.6`) is needed only for the interactive Linux smoke.

API references: [GDK reparent](https://docs.gtk.org/gdk3/method.Window.reparent.html)
and [GLFW 3.4 initialization](https://www.glfw.org/docs/3.4/intro_guide.html).
