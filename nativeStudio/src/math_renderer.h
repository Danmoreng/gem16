#pragma once
#include "imgui.h"
#include <memory>
#include <string_view>
namespace gem16::studio {
struct MathData;
struct MathLayout {
  float width = 0, height = 0;
  std::shared_ptr<MathData> data;
};
void InitializeMathFonts();
MathLayout LayoutMath(std::string_view source, bool display, float size, float width);
void DrawMath(const MathLayout& layout, ImVec2 origin, ImU32 color);
}
