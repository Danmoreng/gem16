# Multiple images and resident Canvas checks — 2026-09-05

## Owner direction and implementation

The owner explicitly removed the conversation-wide one-image restriction and
rejected the separate Canvas visual-review request. This supersedes the earlier
single-image future requirement; historical qualification evidence is unchanged.
Base source: `a67c37fb52883f1c09c878fbd307df54c4bf8659`, plus the uncommitted Linux
live-WebView changes and this patch. No user chat/database was modified.

- The 26B session validates ordered, disjoint image spans before writing KV and
  handles all uncached spans sequentially through the existing single-image
  Vision workspace. Cached images are skipped. Each image's complete attention
  span is retained; no new CUDA kernel, precision, weight format or offload path
  was introduced. Per-image phase times are accumulated across newly encoded images.
- Chat Completions and Responses no longer reject multiple images. Context,
  request-size, geometry and budget checks remain. `maximum_images=UINT32_MAX`
  indicates no fixed product count cap.
- Chat Completions tool content additionally accepts text/image arrays. Text/image
  order and image indices survive parsing; both fresh prompt rendering and resident
  tool continuation contain actual image placeholders. Existing string outputs
  remain compatible. Responses tool outputs remain text-only.
- Studio attaches PNG bytes directly to the completed `canvas_check` tool message,
  persists them in the existing attachment store and resends them in later history.
  Expanded tool rows display their screenshots. Temporary chats stay in memory.
  The extra `canvas_vision_` client and session-clear path have been removed.
- Canvas instructions no longer contain a changing document/revision list.
  `canvas_list`, read/create/edit results provide that information without changing
  the system prefix. Failed edits report the requested/current revision and the
  instruction to reread; the model is told not to claim success after a failed edit.
- Studio's one-image policy/UI was removed. A 12B live regression also exposed an
  existing server bug: 26B's module-specific context/MTP admission ran on 12B images.
  That admission now applies only to actual 26B image parts; 12B retains its own
  integrated Vision and MTP path.

## Bounded verification

Machine: Linux x86-64 / NVIDIA Blackwell SM120. Server builds used the existing
`blackwell-release` preset and locked Hub snapshots. No GUI windows were opened.

```sh
cmake --build --preset host-debug --parallel 8
ctest --preset host-debug --output-on-failure
cmake --build build/native-studio --parallel 8
ctest --test-dir build/native-studio -R gem16-native-studio-host --output-on-failure
cmake --build --preset blackwell-release --target gem16-server gem16-cuda-vision26b-tests --parallel 8
python3 -m py_compile tools/check_multi_image_conversation.py
git diff --check
```

All builds succeeded; host tests passed 5/5 and the native Studio host group passed.
Tests cover two-image Chat Completions/Responses parsing, a 17-image request,
rejection on insufficient context, image tool content on both model adapters,
stable Canvas system instructions and screenshot persistence/wire replay.

The live checker is `tools/check_multi_image_conversation.py`; it targets an
explicitly started local test server and never starts/stops a user's server.
For each profile a separately owned test server was started at port 18169 with
8,192 context, one slot and greedy decoding, then stopped. Exact server arguments,
health, responses and usage are preserved in
[`artifacts/studio/multi-image-session-2026-09-05.json`](../../artifacts/studio/multi-image-session-2026-09-05.json).
Client invocation against each server:

```sh
python3 tools/check_multi_image_conversation.py --base-url http://127.0.0.1:18169 --compact --output <new-evidence-file.json>
# Omit --compact for the 12B server.
```

The five-step resident chain sends red, then blue, then a request for
`canvas_check`, then green/yellow as two images in the tool result, then a text
question recalling all four. The five calls preserve one session ID. A sixth,
explicitly fresh root separately verifies replay of saved image/tool history.
That new root is a test operation, not the Canvas execution path.

| Profile | Cached tokens on the four continuations | Final image recall | Fresh history replay |
|---|---|---|---|
| Compact Vision Ordinary | 147, 254, 287, 466 | Red, blue, green, yellow | Same colors |
| Compact Vision fixed-D2 | 147, 254, 287, 466 | Red, blue, green, yellow | Same colors |
| 12B Unified Ordinary | 87, 134, 167, 222 | Red, Blue, Green, Yellow | Same colors |

The 12B immediate tool response named only the two new colors; its next answer
correctly recalled all four. This is model output behavior, not missing image KV.
The first 12B attempt failed on the incorrect 26B admission guard described above;
the successful result follows its correction. Logs remain under
`build/multiimage-*-client.log` and `build/multiimage-*-server.log`.

## Limits

This is a four-image synthetic visual/session regression, not an extended
multi-image quality or capacity qualification. Windows was not executed here.
No token-speed claim or new memory-capacity claim is made. Native GUI interaction
is left to the owner as requested. The normal finite context, 16 MiB API request
body, bounded media decoder and chat attachment-store limits remain; “multiple
images” does not promise infinite RAM or a context beyond the admitted limit.

### Vision operator and sanitizer checks

The existing real-module Vision operator suite also passed, including finite
outputs, bitwise repeatability and unchanged free device memory across repeated
encodes:

```sh
timeout 30s build/Linux/blackwell-release/bin/gem16-cuda-vision26b-tests --fixtures tests/fixtures/gemma4_26b_vision --vision-module /home/sebastian/.cache/huggingface/hub/.gem16/snapshots/danmoreng--gemma-4-26B-A4B-it-GEM16--6de2a057f11332420819f8e6efd08e42d7a03bc7--vision
```

Exit 0; raw output: `build/multiimage-vision-operators.log`.
Two Compute Sanitizer attempts against this suite were stopped for excessive
instrumentation runtime: full `--tool memcheck --error-exitcode 99`, followed by
`--tool memcheck --kernel-name kns=PoolStandardize --error-exitcode 99`. Both used
the same executable/fixture/module arguments above. Neither produced a completed
sanitizer result; they are **not** sanitizer passes. Their logs are
`build/multiimage-vision-memcheck.log` and
`build/multiimage-vision-pool-memcheck.log`. All owned test processes were stopped
before returning the GPU to the owner.
