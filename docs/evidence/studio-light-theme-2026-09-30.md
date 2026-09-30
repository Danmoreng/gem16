# Native Studio Light theme checks (2026-09-30)

The native UI now derives custom drawing colors from the selected theme. Light
uses the former Compose GemLight palette as read-only design evidence: neutral
white/gray surfaces, forest-green accents, mint containers and dark readable text.
Conversation bubbles, code, reasoning, logs, status labels, form controls,
download progress and default selectable text follow the current theme.

The bright green/mint background shader is visible through the main UI glass
layers. Chat bubbles and code panels remain opaque. Light navigation uses its
own bright green animated flame palette; OpenGL and Direct3D sources match.
Both platforms also clear to a bright color in Light mode. Dark shader colors
and animation parameters retain their previous values.

## Linux checks

Environment: Linux x86-64, NVIDIA driver 610.57.04, RTX 5080 Laptop GPU,
OpenGL 3.3.0 NVIDIA. The following commands passed with root VERSION 0.2.1:

```sh
cmake -S nativeStudio -B build/native-studio -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build/native-studio --parallel 4
ctest --test-dir build/native-studio --output-on-failure
git diff --check
```

Studio: 3/3 tests passed. Theme checks enforce 4.5:1 contrast on the opaque
Light surfaces, verify every ImGui widget color is restored after Dark/Light/Dark
and check default selectable text inherits the current theme. Software captures
of actual ImGui draw lists were reviewed for Chat, Models, Server and Settings.

A local offscreen EGL harness compiled the production `shader_background.cpp`
and rendered both themes on the actual NVIDIA GPU at 1.25 and 7.0 seconds.
Background and navigation images changed between frames, with no OpenGL errors.
GPU captures were also used underneath the software UI screenshots via
`GEM16_STUDIO_SHADER_PREVIEW_DIR`; screenshot rendering samples the captured
navigation texture. Sampled Light contrast minima were 5.76:1 for muted text,
5.17:1 for accents over the composed background and 4.75:1 for navigation text.
These are bounded samples, not an exhaustive time/size contrast proof.
The temporary harness, raw RGBA files, screenshots and build/test logs remain
under ignored `build/light-waves-preview/` and `build/native-studio/`.

All three Windows HLSL entry points also passed offline syntax compilation:

```sh
glslangValidator -D -V -S vert -e vs build/light-waves-preview/shader.hlsl -o build/light-waves-preview/vs.spv
glslangValidator -D -V -S frag -e ps build/light-waves-preview/shader.hlsl -o build/light-waves-preview/ps.spv
glslangValidator -D -V -S frag -e flame_ps build/light-waves-preview/shader.hlsl -o build/light-waves-preview/flame.spv
```

The HLSL file is extracted verbatim from the production shader string. This
check is not Direct3D or live Windows GPU visual qualification. Windows CI's
Studio tests compile the production HLSL on a software D3D device. No inference,
model-lock, sampling or CUDA implementation changes are part of this UI slice.
