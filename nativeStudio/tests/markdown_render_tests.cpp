#include "markdown.h"
#include "fonts.h"
#include "math_renderer.h"
#include "platform_ui.h"
#include "selectable_text.h"
#include "svg_preview.h"
#include "imgui.h"
#include "imgui_internal.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>
#ifdef _WIN32
#include <d3d11.h>
#include <wrl/client.h>
#endif

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
void Screenshot(const char* path, const gem16::studio::DecodedImage* svg = nullptr,
                ImTextureID svg_id = ImTextureID_Invalid) {
  constexpr int w=900,h=1050;
  std::vector<unsigned char> image(w*h*4,20);
  unsigned char* atlas=nullptr;int aw=0,ah=0;
  ImGui::GetIO().Fonts->GetTexDataAsRGBA32(&atlas,&aw,&ah);
  auto cross=[](ImVec2 a,ImVec2 b,ImVec2 p) { return (b.x-a.x)*(p.y-a.y)-(b.y-a.y)*(p.x-a.x); };
  for (const auto* list : ImGui::GetDrawData()->CmdLists) for (const auto& cmd : list->CmdBuffer) {
    if(cmd.UserCallback)continue;
    const bool is_svg = svg && cmd.GetTexID() == svg_id;
    const auto* texture = is_svg ? svg->rgba.data() : atlas;
    const int tw = is_svg ? svg->width : aw, th = is_svg ? svg->height : ah;
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
        const int ax=std::clamp(static_cast<int>((a.uv.x*u+b.uv.x*v+c.uv.x*t)*tw),0,tw-1);
        const int ay=std::clamp(static_cast<int>((a.uv.y*u+b.uv.y*v+c.uv.y*t)*th),0,th-1);
        const auto* tex=texture+(ay*tw+ax)*4;
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
  const std::string svg = R"SVG(<svg xmlns="http://www.w3.org/2000/svg" width="800" height="400" viewBox="0 0 800 400">
    <rect width="800" height="400" rx="15" fill="#f8f9fa"/>
    <text x="400" y="45" text-anchor="middle" font-family="Arial, sans-serif" font-size="24" font-weight="bold" fill="#333">SVG diagrams in chat</text>
    <defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#15805d"/></marker></defs>
    <rect x="50" y="100" width="280" height="230" rx="10" fill="white" stroke="#d33" stroke-dasharray="5,5"/>
    <rect x="470" y="100" width="280" height="230" rx="10" fill="#ddf5e9" stroke="#15805d"/>
    <text x="190" y="145" text-anchor="middle" font-family="sans-serif" font-size="20">Separate items</text>
    <text x="610" y="145" text-anchor="middle" font-family="sans-serif" font-size="20">Connected diagram</text>
    <circle cx="150" cy="230" r="25" fill="#e88"/><circle cx="230" cy="260" r="25" fill="#e88"/>
    <path d="M560 230 L650 270 L650 205 Z" fill="none" stroke="#15805d" stroke-width="3"/>
    <circle cx="560" cy="230" r="15" fill="#15805d"/><circle cx="650" cy="270" r="15" fill="#15805d"/>
    <path d="M350 210 L450 210" stroke="#15805d" stroke-width="3" marker-end="url(#arrow)"/>
    </svg>)SVG";
  ok &= Check(IsSvgCode("svg", "<svg") && IsSvgCode("xml", svg) && IsSvgCode("", svg) &&
      !IsSvgCode("python", svg) && !IsSvgCode("xml", "<document/>"), "SVG fence detection");
  const auto raster = RasterizeSvg(svg);
  ok &= Check(raster.error.empty() && raster.image.width == 800 && raster.image.height == 400,
      "SVG diagram with text and arrows renders");
  const auto text_only = RasterizeSvg("<svg width='200' height='80'><text x='10' y='45' font-size='30'>Hello</text></svg>");
  unsigned ink = 0;
  for (std::size_t i = 3; i < text_only.image.rgba.size(); i += 4) ink += text_only.image.rgba[i] > 0;
  ok &= Check(ink > 100, "SVG text emits visible pixels");
  for (const auto& unsafe : {"<svg><script>alert(1)</script></svg>", "<svg onload='alert(1)'/>",
      "<svg><image href='file:///secret'/></svg>", "<svg><use href='#x'/></svg>",
      "<svg><style>@import 'https://example.com';</style></svg>",
      "<svg><rect fill='url(https://example.com)'/></svg>",
      "<!DOCTYPE svg SYSTEM 'file:///secret'><svg/>", "<svg><foreignObject/></svg>",
      "<svg><defs><marker id='x'><path marker-end='url(#x)'/></marker></defs></svg>", "<svg><rect"})
    ok &= Check(!RasterizeSvg(unsafe).error.empty(), "unsafe/incomplete SVG fails visibly");
  ok &= Check(!RasterizeSvg(std::string(256*1024+1, 'x')).error.empty(), "SVG source size limit");
  const auto bounded = RasterizeSvg("<svg width='8000' height='4000'><rect width='8000' height='4000'/></svg>");
  ok &= Check(bounded.image.width == 1024 && bounded.image.height == 512, "SVG raster size is bounded");
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
#ifdef _WIN32
  Microsoft::WRL::ComPtr<ID3D11Device> device;
  ok &= Check(SUCCEEDED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0, nullptr, 0,
      D3D11_SDK_VERSION, &device, nullptr, nullptr)), "SVG software D3D device");
  if (device) {
    ImageTexture::InitializeRenderer(device.Get());
    SvgPreviewCache cache;
    ImGuiWindow* child = nullptr;
    auto svg_frame = [&] {
      ImGui::NewFrame(); ImGui::SetNextWindowPos({0,0}); ImGui::SetNextWindowSize({900,1050});
      ImGui::Begin("SVG preview", nullptr, ImGuiWindowFlags_NoDecoration);
      Render("svg", "```xml\n" + svg + "\n```", 850, &cache);
      for (auto* window : GImGui->Windows)
        if (window->ParentWindow == ImGui::GetCurrentWindow() && window->Active) child = window;
      ImGui::End(); ImGui::Render();
    };
    svg_frame(); svg_frame();
    auto* entry = cache.Get(svg);
    ok &= Check(entry && entry->texture.Valid(), "SVG uploads through real preview texture path");
    if (entry && entry->texture.Valid()) Screenshot("svg-preview.bmp", &raster.image, entry->texture.Id());
    const auto mode = child ? ImHashStr("##svg-code-view", 0, child->IDStack.back()) : 0;
    ok &= Check(child && !child->StateStorage.GetBool(mode), "SVG defaults to preview");
    if (child) {
      // The toggle is the first control under the header separator.
      io.AddMousePosEvent(child->Pos.x + child->WindowPadding.x + 15,
          child->Pos.y + child->WindowPadding.y + 29);
      io.AddMouseButtonEvent(0,true);svg_frame();io.AddMouseButtonEvent(0,false);svg_frame();
      ok &= Check(child->StateStorage.GetBool(mode), "SVG toggle opens code view");
      io.AddMouseButtonEvent(0,true);svg_frame();io.AddMouseButtonEvent(0,false);svg_frame();
      ok &= Check(!child->StateStorage.GetBool(mode), "SVG toggle restores preview");
      clipboard.clear();
      io.AddMousePosEvent(child->InnerRect.Max.x-child->WindowPadding.x-15,
          child->InnerRect.Min.y+child->WindowPadding.y+7);
      io.AddMouseButtonEvent(0,true);svg_frame();io.AddMouseButtonEvent(0,false);svg_frame();
      ok &= Check(clipboard == svg, "copy in SVG preview retains exact source");
    }
    auto* reused = cache.Get(svg);
    ok &= Check(reused == entry, "unchanged SVG reuses texture");
    for (int i=0; i<7; ++i)
      ok &= Check(cache.Get(svg + "<!--" + std::to_string(i) + "-->") != nullptr, "bounded SVG cache fill");
    ok &= Check(cache.Get(svg + "<!--overflow-->") == nullptr, "SVG cache protects current-frame textures");
    cache.Clear(); ImageTexture::InitializeRenderer(nullptr);
  }
#endif
  ImGui::DestroyContext();return ok;
}
