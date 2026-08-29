# Deprecated Kotlin/Compose Studio

`studioApp/` is deprecated as of the 2026-08-29 product baseline.

The active desktop application is the C++20 Dear ImGui implementation in
`nativeStudio/`. Do not add product features, release integration, or parity
fixes to this Compose application. Its source remains temporarily available as
read-only migration evidence for capabilities that still need a native product
decision.

After the first complete Windows-and-Linux native Studio product release, this
module and Gradle-only root infrastructure may be removed from the default
branch. The final implementation remains available through Git history and
`docs/legacy/KOTLIN_COMPOSE_STUDIO.md`.
