#include "math_renderer.h"
#include "graphic/graphic.h"
#include "core/formula.h"
#include "latex.h"
#include "utils/utf.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <map>
#include <set>
#include <stdexcept>
#ifdef _WIN32
#undef TRANSPARENT
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace {
std::map<std::string, ImFont*> fonts;
std::filesystem::path resource_root;
bool initialized = false;
ImFont* default_font = nullptr;
std::filesystem::path Resources() {
#ifdef _WIN32
  wchar_t path[32768]{};
  GetModuleFileNameW(nullptr, path, 32768);
  auto root = std::filesystem::path(path).parent_path() / "math-res";
#else
  char path[4096]{};
  const auto n = readlink("/proc/self/exe", path, sizeof(path) - 1);
  auto root = n > 0 ? std::filesystem::path(path).parent_path() / "math-res" : std::filesystem::path{};
#endif
  if (std::filesystem::is_regular_file(root / ".clatexmath-res_root")) return root;
  return GEM16_MATH_RESOURCE_DIR;
}
class TexFont final : public tex::Font {
 public:
  ImFont* font;
  float size;
  float em_scale = 1;
  TexFont(ImFont* f, float s) : font(f), size(s) {}
  float getSize() const override { return size; }
  tex::sptr<tex::Font> deriveFont(int) const override { return std::make_shared<TexFont>(*this); }
  bool operator==(const tex::Font& other) const override {
    const auto* rhs = dynamic_cast<const TexFont*>(&other);
    return rhs && rhs->font == font && rhs->size == size;
  }
  bool operator!=(const tex::Font& other) const override { return !(*this == other); }
};
class Graphics final : public tex::Graphics2D {
  ImDrawList* draw = ImGui::GetWindowDrawList();
  const TexFont* font = nullptr;
  tex::Stroke stroke;
  tex::color color = tex::white;
  float a=1,b=0,c=0,d=1,tx=0,ty=0;
  ImVec2 point(float x,float y) const { return {a*x+c*y+tx,b*x+d*y+ty}; }
  ImU32 ink() const { return IM_COL32(tex::color_r(color),tex::color_g(color),tex::color_b(color),tex::color_a(color)); }
 public:
  void setColor(tex::color value) override { color=value; }
  tex::color getColor() const override { return color; }
  void setStroke(const tex::Stroke& value) override { stroke=value; }
  const tex::Stroke& getStroke() const override { return stroke; }
  void setStrokeWidth(float width) override { stroke.lineWidth=width; }
  const tex::Font* getFont() const override { return font; }
  void setFont(const tex::Font* f) override { font=dynamic_cast<const TexFont*>(f); }
  void translate(float x,float y) override { const auto p=point(x,y);tx=p.x;ty=p.y; }
  void scale(float x,float y) override { a*=x;b*=x;c*=y;d*=y; }
  void rotate(float r) override {
    const float co=std::cos(r),si=std::sin(r),aa=a,bb=b;
    a=aa*co+c*si;b=bb*co+d*si;c=c*co-aa*si;d=d*co-bb*si;
  }
  void rotate(float r,float x,float y) override { translate(x,y);rotate(r);translate(-x,-y); }
  void reset() override { a=d=1;b=c=tx=ty=0; }
  float sx() const override { return std::hypot(a,b); }
  float sy() const override { return std::hypot(c,d); }
  void drawChar(wchar_t ch,float x,float y) override {
    if (!font || !font->font) return;
    const float factor=std::max(sy(),0.001f),pixels=font->size*factor*font->em_scale;
    if (!std::isfinite(pixels) || pixels < 0.1f || pixels > 512) return;
    auto* baked=font->font->GetFontBaked(pixels);
    const int first=draw->VtxBuffer.Size;
    font->font->RenderChar(draw,pixels,{0,-baked->Ascent * pixels / baked->Size},ink(),static_cast<ImWchar>(ch));
    for(int i=first;i<draw->VtxBuffer.Size;++i) {
      auto& p=draw->VtxBuffer[i].pos;p=point(x+p.x/factor,y+p.y/factor);
    }
  }
  void drawText(const std::wstring& text,float x,float y) override {
    if (!font) return;
    for (wchar_t ch:text) {
      drawChar(ch,x,y);
      x+=font->font->GetFontBaked(17)->FindGlyph(static_cast<ImWchar>(ch))->AdvanceX * font->size/17;
    }
  }
  void drawLine(float x,float y,float xx,float yy) override { draw->AddLine(point(x,y),point(xx,yy),ink(),std::max(1.0f,stroke.lineWidth*sy())); }
  void drawRect(float x,float y,float w,float h) override { drawLine(x,y,x+w,y);drawLine(x+w,y,x+w,y+h);drawLine(x+w,y+h,x,y+h);drawLine(x,y+h,x,y); }
  void fillRect(float x,float y,float w,float h) override { draw->AddQuadFilled(point(x,y),point(x+w,y),point(x+w,y+h),point(x,y+h),ink()); }
  void drawRoundRect(float x,float y,float w,float h,float,float) override { drawRect(x,y,w,h); }
  void fillRoundRect(float x,float y,float w,float h,float,float) override { fillRect(x,y,w,h); }
};
class TextLayout final : public tex::TextLayout {
  std::wstring text;
  tex::sptr<tex::Font> font;
 public:
  TextLayout(std::wstring t,tex::sptr<tex::Font> f):text(std::move(t)),font(std::move(f)){}
  void getBounds(tex::Rect& out) override {
    const auto* f=dynamic_cast<const TexFont*>(font.get());
    const auto utf8=tex::wide2utf8(text);
    const auto s=f->font->CalcTextSizeA(17,1e6f,0,utf8.c_str());
    out={0,-font->getSize()*0.8f,s.x*font->getSize()/17,font->getSize()};
  }
  void draw(tex::Graphics2D& g,float x,float y) override { g.setFont(font.get());g.drawText(text,x,y); }
};
bool SafeMath(std::string_view source) {
  if (source.empty() || source.size()>4096 || source.find('\0')!=source.npos) return false;
  // Bound matrix/array expansion as well as ordinary brace nesting.
  if (source.find("*{") != source.npos || std::count(source.begin(),source.end(),'&') > 256) return false;
  unsigned environments = 0;
  for (std::size_t pos = 0; (pos = source.find("\\begin", pos)) != source.npos; pos += 6)
    if (++environments > 8) return false;
  static const std::set<std::string_view> commands={
    "frac","dfrac","tfrac","sqrt","sum","prod","int","iint","oint","lim","sin","cos","tan","log","ln","exp",
    "alpha","beta","gamma","delta","epsilon","theta","lambda","mu","pi","rho","sigma","tau","phi","psi","omega",
    "Gamma","Delta","Theta","Lambda","Pi","Sigma","Phi","Psi","Omega","infty","pm","mp","times","cdot","div",
    "le","leq","ge","geq","neq","approx","equiv","to","rightarrow","leftarrow","in","notin","subset","cup","cap",
    "partial","nabla","forall","exists","emptyset","ldots","cdots","vdots","ddots","left","right","big","Big",
    "text","mathrm","mathbf","mathit","mathbb","mathcal","operatorname","overline","underline","hat","bar","vec",
    "begin","end","displaystyle","textstyle","quad","qquad"};
  int depth=0;
  for(std::size_t i=0;i<source.size();++i) {
    if(source[i]=='{') { if(++depth>32)return false; }
    else if(source[i]=='}') { if(--depth<0)return false; }
    else if(source[i]=='\\') {
      const auto start=++i;
      while(i<source.size()&&((source[i]>='a'&&source[i]<='z')||(source[i]>='A'&&source[i]<='Z')))++i;
      if(i>start) { if(!commands.contains(source.substr(start,i-start)))return false;--i; }
    }
  }
  return depth==0;
}
}
namespace tex {
Font* Font::create(const std::string& file,float size) {
  auto found=fonts.find(std::filesystem::path(file).lexically_normal().generic_string());
  if(found==fonts.end())throw std::runtime_error("Math font unavailable");
  auto result = std::make_unique<TexFont>(found->second,size);
  // MicroTeX measures in em units; stb's pixel height uses ascent-descent.
  // Extension fonts (cmex10, sums/integrals) have very tall global metrics.
  // Convert explicitly instead of shrinking these glyphs to the line height.
  const auto* config = found->second->Sources[0];
  const auto* bytes = static_cast<const unsigned char*>(config->FontData);
  const std::size_t length = static_cast<std::size_t>(config->FontDataSize);
  auto u16 = [&](std::size_t offset) -> unsigned { return offset+2<=length ? (bytes[offset]<<8)|bytes[offset+1] : 0; };
  auto u32 = [&](std::size_t offset) -> std::size_t { return offset+4<=length ? (static_cast<std::size_t>(u16(offset))<<16)|u16(offset+2) : length; };
  unsigned units = 0; int ascent = 0, descent = 0;
  for (std::size_t i = 0, count = std::min<std::size_t>(u16(4), 256); i < count; ++i) {
    const std::size_t entry = 12 + i*16;
    if (entry+16>length) break;
    const auto offset = u32(entry+8);
    if (offset>length || length-offset<20) continue;
    if (u32(entry)==0x68656164) units=u16(offset+18); // head
    if (u32(entry)==0x68686561) { // hhea
      ascent=static_cast<short>(u16(offset+4));descent=static_cast<short>(u16(offset+6));
    }
  }
  if (units && ascent>descent) result->em_scale=static_cast<float>(ascent-descent)/units;
  return result.release();
}
sptr<Font> Font::_create(const std::string&,int,float size) { return std::make_shared<TexFont>(default_font,size); }
sptr<TextLayout> TextLayout::create(const std::wstring& text,const sptr<Font>& font) { return std::make_shared<::TextLayout>(text,font); }
}
namespace gem16::studio {
struct MathData { std::unique_ptr<tex::TeXRender> render; };
static std::map<std::string,MathLayout> cache;
void InitializeMathFonts() {
  cache.clear();
  if(initialized) { tex::LaTeX::release(); initialized=false; }
  fonts.clear();default_font=ImGui::GetIO().Fonts->Fonts[0];resource_root=Resources();
  if(!std::filesystem::is_regular_file(resource_root/".clatexmath-res_root"))return;
  static const ImWchar ranges[]={1,0x3ff,0};
  for(const auto& entry:std::filesystem::recursive_directory_iterator(resource_root)) {
    if(entry.path().extension()!=".ttf")continue;
    const auto key=entry.path().lexically_normal().generic_string();
    if(auto* font=ImGui::GetIO().Fonts->AddFontFromFileTTF(key.c_str(),17,nullptr,ranges)) fonts[key]=font;
  }
  try { tex::LaTeX::init(resource_root.generic_string());initialized=true; } catch (...) {}
}
MathLayout LayoutMath(std::string_view source,bool display,float size,float width) {
  if(!initialized||!SafeMath(source)||!std::isfinite(size)||size<=0||size>256||!std::isfinite(width)||width<=0)return {};
  // Cache bounded layouts; streamed formulas and untrusted input never create an unbounded cache.
  const std::string key=std::string(source)+"\n"+std::to_string(display)+":"+std::to_string(size)+":"+std::to_string(width);
  if(auto it=cache.find(key);it!=cache.end())return it->second;
  try {
    tex::Formula formula(tex::utf82wide(std::string(source)));
    tex::TeXRenderBuilder builder;
    auto data=std::make_shared<MathData>();
    data->render.reset(builder.setStyle(display?tex::TexStyle::display:tex::TexStyle::text).setTextSize(size).setForeground(tex::white).build(formula));
    if(!data->render)return {};
    if(data->render->getWidth()>width)data->render->setTextSize(size*width/data->render->getWidth());
    MathLayout layout{static_cast<float>(data->render->getWidth()),static_cast<float>(data->render->getHeight()),data};
    if(!std::isfinite(layout.width)||!std::isfinite(layout.height)||layout.height>4096)return {};
    if(cache.size()>=128)cache.clear();cache.emplace(key,layout);return layout;
  } catch (...) { return {}; }
}
void DrawMath(const MathLayout& layout,ImVec2 origin,ImU32 color) {
  if(!layout.data)return;
  Graphics g;
  const auto rgba=ImGui::ColorConvertU32ToFloat4(color);
  layout.data->render->setForeground(tex::argb(rgba.w,rgba.x,rgba.y,rgba.z));
  layout.data->render->draw(g,static_cast<int>(origin.x),static_cast<int>(origin.y));
}
}
