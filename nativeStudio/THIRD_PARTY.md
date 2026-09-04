# Native Studio third-party notices

- Dear ImGui 1.92.9b is vendored under `third_party/imgui/` and licensed under MIT. Its original license is
  `third_party/imgui/LICENSE.txt`.
- The cross-platform backend and shader direction was adapted from Free Solace ImGui Interface commit
  `bb35bb3f11ef390fa94ca4aa57daa0a6ee379e67`, Copyright 2026 Pondot, licensed under MIT. Its license is
  `licenses/Free-Solace-ImGui-Interface-MIT.txt`.
- GLFW 3.4 is fetched from its upstream repository at configure time on Linux and is licensed under zlib/libpng.

No Free Solace login/authentication implementation, slide/avatar/brand assets, or glass-cursor code is included.

## SQLite

SQLite 3.53.4 is fetched as the official amalgamation from sqlite.org and statically
linked with FTS5 enabled and loadable extensions disabled. SQLite is public domain.
Source: https://sqlite.org/2026/sqlite-amalgamation-3530400.zip
SHA-256: `1e71ddf93849c6a6ecf58b827c0692073d2dd7ee40196158068f7b29f422e87d`.
The archive is pinned in `cmake/sqlite-dependency.cmake`.
