#pragma once

struct ImFont;

namespace gem16::studio {
// System fonts only; no font files are copied into the application package.
// Emoji outlines are rendered monochromatically by the existing stb backend.
ImFont* InitializeStudioFonts();
}
