#include "markdown.h"
#include "fonts.h"
#include "math_renderer.h"
#include "platform_ui.h"
#include "selectable_text.h"
#include "imgui.h"
#include "imgui_internal.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>

namespace {
bool Check(bool value, const char* name) {
  if (!value) std::fprintf(stderr, "Markdown regression: %s\n", name);
  return value;
}
std::string opened;
void CaptureLink(std::string_view url) { opened = url; }
std::string clipboard;
void CaptureClipboard(ImGuiContext*, const char* text) { clipboard = text; }

// Small deterministic software screenshot of the real ImGui draw output.
// Test-only: samples the font atlas and blends the emitted triangles.
void Screenshot(const char* path) {
  constexpr int w=900,h=1050;
  std::vector<unsigned char> image(w*h*4,20);
  unsigned char* atlas=nullptr;int aw=0,ah=0;
  ImGui::GetIO().Fonts->GetTexDataAsRGBA32(&atlas,&aw,&ah);
  auto cross=[](ImVec2 a,ImVec2 b,ImVec2 p) { return (b.x-a.x)*(p.y-a.y)-(b.y-a.y)*(p.x-a.x); };
  for (const auto* list : ImGui::GetDrawData()->CmdLists) for (const auto& cmd : list->CmdBuffer) {
    if(cmd.UserCallback)continue;
    for(unsigned i=0;i+2<cmd.ElemCount;i+=3) {
      const auto& a=list->VtxBuffer[cmd.VtxOffset+list->IdxBuffer[cmd.IdxOffset+i]];
      const auto& b=list->VtxBuffer[cmd.VtxOffset+list->IdxBuffer[cmd.IdxOffset+i+1]];
      const auto& c=list->VtxBuffer[cmd.VtxOffset+list->IdxBuffer[cmd.IdxOffset+i+2]];
      const float area=cross(a.pos,b.pos,c.pos);if(std::abs(area)<0.00001f)continue;
      const int x0=std::max({0,static_cast<int>(cmd.ClipRect.x),static_cast<int>(std::floor(std::min({a.pos.x,b.pos.x,c.pos.x})))});
      const int y0=std::max({0,static_cast<int>(cmd.ClipRect.y),static_cast<int>(std::floor(std::min({a.pos.y,b.pos.y,c.pos.y})))});
      const int x1=std::min({w,static_cast<int>(cmd.ClipRect.z),static_cast<int>(std::ceil(std::max({a.pos.x,b.pos.x,c.pos.x})))});
      const int y1=std::min({h,static_cast<int>(cmd.ClipRect.w),static_cast<int>(std::ceil(std::max({a.pos.y,b.pos.y,c.pos.y})))});
      const auto ca=ImGui::ColorConvertU32ToFloat4(a.col),cb=ImGui::ColorConvertU32ToFloat4(b.col),cc=ImGui::ColorConvertU32ToFloat4(c.col);
      for(int y=y0;y<y1;++y)for(int x=x0;x<x1;++x) {
        ImVec2 p(x+0.5f,y+0.5f);const float u=cross(b.pos,c.pos,p)/area,v=cross(c.pos,a.pos,p)/area,t=1-u-v;
        if(u<0||v<0||t<0)continue;
        const int ax=std::clamp(static_cast<int>((a.uv.x*u+b.uv.x*v+c.uv.x*t)*aw),0,aw-1);
        const int ay=std::clamp(static_cast<int>((a.uv.y*u+b.uv.y*v+c.uv.y*t)*ah),0,ah-1);
        const auto* tex=atlas+(ay*aw+ax)*4;
        const float alpha=(ca.w*u+cb.w*v+cc.w*t)*tex[3]/255;
        const float colors[]={ca.z*u+cb.z*v+cc.z*t,ca.y*u+cb.y*v+cc.y*t,ca.x*u+cb.x*v+cc.x*t};
        for(int k=0;k<3;++k)image[(y*w+x)*4+k]=static_cast<unsigned char>(colors[k]*tex[2-k]*alpha+image[(y*w+x)*4+k]*(1-alpha));
      }
    }
  }
  std::ofstream out(path,std::ios::binary);
  auto u16=[&](unsigned n){out.put(static_cast<char>(n));out.put(static_cast<char>(n>>8));};
  auto u32=[&](unsigned n){u16(n);u16(n>>16);};
  out.write("BM",2);u32(54+w*h*4);u32(0);u32(54);u32(40);u32(w);u32(static_cast<unsigned>(-h));u16(1);u16(32);
  u32(0);u32(w*h*4);u32(0);u32(0);u32(0);u32(0);
  out.write(reinterpret_cast<const char*>(image.data()),image.size());
}
}

bool TestExtendedMarkdown() {
  using namespace gem16::studio;
  using namespace gem16::studio::markdown;
  bool ok=true;
  const auto nested=Parse("- Parent\n    - Child\n        - Deep\n- Sibling\n\n    real code\n");
  ok &= Check(nested.size()>=4&&nested[0].kind==BlockKind::kBulletItem&&nested[1].kind==BlockKind::kBulletItem&&nested[1].indent==1&&nested[2].indent==2,"nested lists");
  const auto styles=Parse("***both*** ~~***all***~~ and &amp; &#x1f60a; `**literal**`");
  bool both=false,all=false;
  for(const auto& s:styles[0].spans) { both |= s.strong&&s.emphasis;all |= s.strong&&s.emphasis&&s.strike; }
  ok &= Check(both&&all&&styles[0].text.find("& ")!=std::string::npos,"nested styles/entities");
  const auto table=Parse("| A | B |\n| :--- | ---: |\n| **one** | two |\n");
  ok &= Check(table.size()==1&&table[0].kind==BlockKind::kTable&&table[0].rows.size()==2&&table[0].alignments[1]==3,"GFM table");
  const auto tasks=Parse("- [x] done\n- [ ] todo");
  ok &= Check(tasks.size()==2&&tasks[0].task&&tasks[0].checked&&!tasks[1].checked,"task list");
  const auto math=Parse("Inline $E=mc^2$\n\n$$\\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}$$");
  bool latex=false;for(const auto& b:math)for(const auto& s:b.spans)latex |= s.math&&b.text.find("\\frac")!=std::string::npos;
  ok &= Check(latex,"math retains backslashes");
  const std::string matrix = "\\begin{pmatrix} a & b \\\\\n c & d \\end{pmatrix}\n"
      "\\begin{pmatrix} x \\\\ y \\end{pmatrix}\n=\n"
      "\\begin{pmatrix} ax + by \\\\ cx + dy \\end{pmatrix}";
  for (const auto& source : {"$$\n" + matrix + "\n$$", "$$ " + matrix + " $$"}) {
    const auto parsed = Parse(source);
    ok &= Check(parsed.size() == 1 && parsed[0].kind == BlockKind::kParagraph &&
        parsed[0].text == matrix && parsed[0].spans.size() == 1 &&
        parsed[0].spans[0].display_math, "multiline matrix is one intact formula, not a heading");
  }
  for (const auto& fence : {std::string("```"), std::string("~~~~")}) {
    const auto code = Parse(fence + "latex\n$$\n" + matrix + "\n$$\n" + fence);
    ok &= Check(code.size() == 1 && code[0].kind == BlockKind::kCode &&
        code[0].text == "$$\n" + matrix + "\n$$", "math inside code remains literal");
  }
  const auto indented = Parse("    $$\n    x = y\n    $$");
  ok &= Check(indented.size() == 1 && indented[0].kind == BlockKind::kCode, "indented code stays code");
  const auto partial = Parse("Before\n\n$$\n\\begin{pmatrix}a & b");
  ok &= Check(!partial.empty() && partial.back().text.find("$$") != std::string::npos, "unfinished display math remains visible");
  for(const auto& input:{"***", "- parent\n    -", "```cpp\nint x", "$\\frac{a}{", "| a | b |\n| ---"})
    ok &= Check(!Parse(input).empty(),"streaming partial input");
  ok &= Check(IsSafeWebLink("https://example.com/?x=1&y=2")&&!IsSafeWebLink("file:///C:/test.exe")&&!IsSafeWebLink("javascript:alert(1)")&&!IsSafeWebLink("https://x\n.exe")&&!IsSafeWebLink("https://user@host"),"safe URL schemes");
  ImGui::CreateContext();auto& io=ImGui::GetIO();io.IniFilename=nullptr;io.DisplaySize={900,1050};io.DeltaTime=1.0f/60;
  InitializeStudioFonts();unsigned char* pixels;int w,h;io.Fonts->GetTexDataAsRGBA32(&pixels,&w,&h);
  ImFont* mono = StudioCodeFont();
  ok &= Check(mono != io.Fonts->Fonts[0] &&
      std::abs(mono->CalcTextSizeA(17,1000,0,"iiii").x - mono->CalcTextSizeA(17,1000,0,"WWWW").x) < 0.1f,
      "code font uses equal-width glyphs");
  const auto matrix_layout = LayoutMath(matrix, true, 22, 700);
  ok &= Check(matrix_layout.data && matrix_layout.height > 30 && matrix_layout.width > 100,
      "multiline matrix multiplication layout");
  for(const auto& formula:{"E=mc^2", "\\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}", "\\sum_{i=1}^{n} \\sqrt{i}", "\\begin{pmatrix}a&b\\\\c&d\\end{pmatrix}"}) {
    auto layout=LayoutMath(formula,true,22,700);
    ok &= Check(layout.data&&layout.width>0&&layout.height>0,"MicroTeX layout");
  }
  ok &= Check(!LayoutMath("\\input{secret}",false,17,700).data&&!LayoutMath("\\def\\x{\\x}\\x",false,17,700).data&&!LayoutMath("\\frac{",false,17,700).data,"unsafe/incomplete TeX");
  const std::string sample=
    "# Markdown rendering\n\nNormal, **bold**, *italic*, ***both*** and ~~***all together***~~.\n\n"
    "Inline `iiii WWWW int answer = 42;` followed by proportional text.\n\n"
    "- Parent\n    - Indented child\n        - Deeper child\n- Sibling\n\n"
    "1. First\n2. Second\n\n- [x] Completed\n- [ ] Pending\n\n"
    "| Product | Amount | Price |\n| :--- | :---: | ---: |\n| **Apple** | 5 | 2.50 |\n| Banana | 12 | 1.20 |\n\n"
    "Inline $E=mc^2$ followed by text.\n\n$$\\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}$$\n\n"
    "$$\\sum_{i=1}^{n} \\sqrt{i}$$\n\n[Clickable web link](https://example.com)\n\n"
    "$$\n" + matrix + "\n$$\n\n"
    "> A quote with **bold** text.\n\n```cpp\nint answer = 42;\n// iiii WWWW\n```";
  for(int frame=0;frame<2;++frame) {
    ImGui::NewFrame();ImGui::SetNextWindowPos({0,0});ImGui::SetNextWindowSize({900,1050});
    ImGui::Begin("Preview",nullptr,ImGuiWindowFlags_NoDecoration);Render("sample",sample,860);ImGui::End();ImGui::Render();
  }
  Screenshot("markdown-preview.bmp");
  // Run after the legacy-atlas preview: additional baked sizes require a
  // texture-updating backend, which the software screenshot does not provide.
  for (float scale : {1.0f, 1.25f, 1.5f}) {
    const float size = 17 * scale;
    auto* body = io.Fonts->Fonts[0]->GetFontBaked(size);
    auto* code = mono->GetFontBaked(size);
    for (ImWchar ch : {ImWchar('M'), ImWchar('x')}) {
      const auto* normal = body->FindGlyph(ch);
      const auto* fixed = code->FindGlyph(ch);
      const float normal_height = (normal->Y1-normal->Y0) * size/body->Size;
      const float code_height = (fixed->Y1-fixed->Y0) * size/code->Size;
      ok &= Check(code_height <= normal_height + 1.5f && code_height >= normal_height * 0.8f,
          "monospace visual height matches body text at each DPI");
    }
  }
  // Real hit testing: a click opens, dragging selects without opening.
  selectable_text::StyleSpan link;link.end=9;link.link="https://example.com";link.underline=true;
  std::vector<selectable_text::StyleSpan> spans{link};
  auto frame=[&] {
    ImGui::NewFrame();ImGui::SetNextWindowPos({0,0});ImGui::SetNextWindowSize({900,1050});ImGui::Begin("link-test",nullptr,ImGuiWindowFlags_NoDecoration);
    ImGui::SetCursorPos({30,30});selectable_text::Wrapped("link","Click me!",{.width=300,.spans=&spans,.open_link=CaptureLink});ImGui::End();ImGui::Render();
  };
  frame();frame();io.AddMousePosEvent(45,38);io.AddMouseButtonEvent(0,true);frame();io.AddMouseButtonEvent(0,false);frame();
  ok &= Check(opened=="https://example.com","link click");opened.clear();
  io.AddMouseButtonEvent(0,true);frame();io.AddMousePosEvent(90,38);frame();io.AddMouseButtonEvent(0,false);frame();
  ok &= Check(opened.empty(),"link drag does not launch");
  // Header placement must respect real frame padding, child padding and DPI.
  ImGui::GetPlatformIO().Platform_SetClipboardTextFn = CaptureClipboard;
  for (float scale : {1.0f, 1.25f, 1.5f}) for (float width : {240.0f, 650.0f}) {
    ImGuiWindow* child = nullptr;
    auto code_frame = [&] {
      ImGui::NewFrame(); ImGui::SetNextWindowPos({0,0}); ImGui::SetNextWindowSize({900,1050});
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {24*scale,24*scale});
      ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {17*scale,8*scale});
      ImGui::Begin("code-bounds", nullptr, ImGuiWindowFlags_NoDecoration);
      ImGui::SetWindowFontScale(scale);
      Render("code", "```a-long-language-label-that-must-not-overlap-copy\niiii WWWW\n```", width);
      for (auto* window : GImGui->Windows)
        if (window->ParentWindow == ImGui::GetCurrentWindow() && window->Active) child = window;
      ImGui::End(); ImGui::PopStyleVar(2); ImGui::Render();
    };
    code_frame(); code_frame();
    ok &= Check(child && child->ContentSize.x <= child->InnerRect.GetWidth() - 2*child->WindowPadding.x + 1,
        "copy button and code remain inside child padding at each DPI/width");
    if (child) {
      auto* font = io.Fonts->Fonts[0];
      const float button_width = font->CalcTextSizeA(17*scale,1000,0,"Copy").x + 34*scale;
      io.AddMousePosEvent(child->InnerRect.Max.x-child->WindowPadding.x-button_width/2,
          child->InnerRect.Min.y+child->WindowPadding.y+8*scale);
      clipboard.clear(); io.AddMouseButtonEvent(0,true);code_frame();io.AddMouseButtonEvent(0,false);code_frame();
      ok &= Check(clipboard == "iiii WWWW", "code copy button remains clickable");
    }
  }
  ImGui::DestroyContext();return ok;
}
