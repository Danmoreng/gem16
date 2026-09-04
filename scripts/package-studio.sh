#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
server="$repo_root/build/Linux/blackwell-release/bin/gem16-server"
if [[ ! -x "$server" ]]; then
  echo "Build the Linux CUDA release server before packaging: $server" >&2
  exit 1
fi
version="$(tr -d '\r\n' < "$repo_root/VERSION")"
server_version="$($server --version)"
if [[ "$server_version" != "gem16-server $version" ]]; then
  echo "Release server version mismatch: expected 'gem16-server $version', got '$server_version'" >&2
  exit 1
fi
build_dir="$repo_root/build/native-studio-package"
stage_dir="$repo_root/build/packages/gem16-linux-x64"
cmake -S "$repo_root/nativeStudio" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
cmake --build "$build_dir" --target gem16-studio --parallel
mkdir -p "$stage_dir/bin" "$stage_dir/licenses"
cp "$build_dir/bin/gem16-studio" "$stage_dir/bin/"
cp -R "$build_dir/bin/math-res" "$stage_dir/bin/"
cp "$build_dir/bin/licenses/"* "$stage_dir/licenses/"
cp "$server" "$stage_dir/bin/"
cp "$repo_root/VERSION" "$stage_dir/"
cp "$repo_root/docs/STUDIO.md" "$stage_dir/STUDIO.md"
printf 'System dependencies: WebKitGTK 4.1 >= 2.40, GTK 3, OpenGL, NVIDIA driver/CUDA runtime. No browser engine is bundled.\n' > "$stage_dir/SYSTEM-DEPENDENCIES.txt"
cp "$repo_root/LICENSE" "$stage_dir/"
cp "$repo_root/nativeStudio/third_party/imgui/LICENSE.txt" "$stage_dir/licenses/Dear-ImGui-MIT.txt"
cp "$repo_root/nativeStudio/licenses/Free-Solace-ImGui-Interface-MIT.txt" "$stage_dir/licenses/"
cp "$repo_root/third_party/miniaudio/LICENSE" "$stage_dir/licenses/miniaudio-MIT-0-or-Public-Domain.txt"
cp "$build_dir/_deps/glfw-src/LICENSE.md" "$stage_dir/licenses/GLFW-zlib.txt"
tar -C "$repo_root/build/packages" -czf "$repo_root/build/packages/gem16-linux-x64.tar.gz" gem16-linux-x64
sha256sum "$repo_root/build/packages/gem16-linux-x64.tar.gz"
