#include "image_texture.h"

#include <climits>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO
#include <stb_image.h>

#ifdef _WIN32
#include <d3d11.h>
#else
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glext.h>
#endif

namespace gem16::studio {
namespace {

constexpr int kMaximumPreviewDimension = 8192;
constexpr std::uint64_t kMaximumPreviewPixels = 16ULL * 1024ULL * 1024ULL;

#ifdef _WIN32
ID3D11Device* g_device = nullptr;
#endif

}  // namespace

ImageDimensions ProbePreviewImage(const std::uint8_t* encoded,
                                  std::size_t size) {
  ImageDimensions result;
  if (encoded == nullptr || size == 0 ||
      size > static_cast<std::size_t>(INT_MAX))
    return result;
  int channels = 0;
  if (stbi_info_from_memory(encoded, static_cast<int>(size), &result.width,
                            &result.height, &channels) == 0 ||
      result.width <= 0 || result.height <= 0 ||
      result.width > kMaximumPreviewDimension ||
      result.height > kMaximumPreviewDimension ||
      static_cast<std::uint64_t>(result.width) *
              static_cast<std::uint64_t>(result.height) >
          kMaximumPreviewPixels) {
    return {};
  }
  return result;
}

DecodedImage DecodePreviewImage(const std::uint8_t* encoded, std::size_t size) {
  DecodedImage result;
  const ImageDimensions dimensions = ProbePreviewImage(encoded, size);
  if (!dimensions.Valid()) return result;
  int width = dimensions.width;
  int height = dimensions.height;
  int channels = 0;
  stbi_uc* pixels = stbi_load_from_memory(
      encoded, static_cast<int>(size), &width, &height, &channels, STBI_rgb_alpha);
  if (pixels == nullptr) return result;
  const std::size_t byte_count = static_cast<std::size_t>(width) *
                                 static_cast<std::size_t>(height) * 4U;
  result.rgba.assign(pixels, pixels + byte_count);
  result.width = width;
  result.height = height;
  stbi_image_free(pixels);
  return result;
}

ImageTexture::~ImageTexture() { Reset(); }

void ImageTexture::InitializeRenderer(void* device) {
#ifdef _WIN32
  g_device = static_cast<ID3D11Device*>(device);
#else
  (void)device;
#endif
}

bool ImageTexture::Load(const std::uint8_t* encoded, std::size_t size) {
  Reset();
  DecodedImage decoded = DecodePreviewImage(encoded, size);
  if (decoded.rgba.empty()) return false;
#ifdef _WIN32
  if (g_device == nullptr) return false;
  D3D11_TEXTURE2D_DESC description{};
  description.Width = static_cast<UINT>(decoded.width);
  description.Height = static_cast<UINT>(decoded.height);
  description.MipLevels = 1;
  description.ArraySize = 1;
  description.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
  description.SampleDesc.Count = 1;
  description.Usage = D3D11_USAGE_DEFAULT;
  description.BindFlags = D3D11_BIND_SHADER_RESOURCE;
  D3D11_SUBRESOURCE_DATA data{};
  data.pSysMem = decoded.rgba.data();
  data.SysMemPitch = static_cast<UINT>(decoded.width * 4);
  ID3D11Texture2D* texture = nullptr;
  if (FAILED(g_device->CreateTexture2D(&description, &data, &texture)))
    return false;
  D3D11_SHADER_RESOURCE_VIEW_DESC view_description{};
  view_description.Format = description.Format;
  view_description.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
  view_description.Texture2D.MipLevels = 1;
  ID3D11ShaderResourceView* view = nullptr;
  const HRESULT view_result = g_device->CreateShaderResourceView(
      texture, &view_description, &view);
  texture->Release();
  if (FAILED(view_result)) return false;
  texture_id_ = static_cast<ImTextureID>(reinterpret_cast<std::uintptr_t>(view));
#else
  GLuint texture = 0;
  GLint previous = 0;
  glGetIntegerv(GL_TEXTURE_BINDING_2D, &previous);
  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, decoded.width, decoded.height, 0,
               GL_RGBA, GL_UNSIGNED_BYTE, decoded.rgba.data());
  glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(previous));
  if (texture == 0) return false;
  texture_id_ = static_cast<ImTextureID>(texture);
#endif
  width_ = decoded.width;
  height_ = decoded.height;
  return true;
}

void ImageTexture::Reset() {
  if (texture_id_ == ImTextureID_Invalid) return;
#ifdef _WIN32
  auto* view = reinterpret_cast<ID3D11ShaderResourceView*>(
      static_cast<std::uintptr_t>(texture_id_));
  view->Release();
#else
  const GLuint texture = static_cast<GLuint>(texture_id_);
  glDeleteTextures(1, &texture);
#endif
  texture_id_ = ImTextureID_Invalid;
  width_ = 0;
  height_ = 0;
}

}  // namespace gem16::studio
