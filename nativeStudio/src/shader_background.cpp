#include "shader_background.h"

#include <array>
#include <cstdint>

#ifdef _WIN32
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#else
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glext.h>
#endif

namespace gem16::studio {

namespace {
constexpr int kFlameWidth = 384;
constexpr int kFlameHeight = 96;
}

struct ShaderBackground::Impl {
#ifdef _WIN32
  Microsoft::WRL::ComPtr<ID3D11DeviceContext> context;
  Microsoft::WRL::ComPtr<ID3D11VertexShader> vertex_shader;
  Microsoft::WRL::ComPtr<ID3D11PixelShader> pixel_shader;
  Microsoft::WRL::ComPtr<ID3D11PixelShader> flame_pixel_shader;
  Microsoft::WRL::ComPtr<ID3D11Buffer> constants;
  Microsoft::WRL::ComPtr<ID3D11Texture2D> flame_texture;
  Microsoft::WRL::ComPtr<ID3D11RenderTargetView> flame_target;
  Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> flame_view;
#else
  GLuint program = 0;
  GLuint flame_program = 0;
  GLuint flame_texture = 0;
  GLuint flame_framebuffer = 0;
  GLuint vertex_array = 0;
  GLint time_location = -1;
  GLint resolution_location = -1;
  GLint theme_location = -1;
  GLint flame_time_location = -1;
  GLint flame_theme_location = -1;
#endif
};

#ifdef _WIN32
namespace {
constexpr char kShader[] = R"HLSL(
cbuffer Params : register(b0) { float4 u_params; };

float hash21(float2 p) {
  p = frac(p * float2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return frac(p.x * p.y);
}

float noise(float2 p) {
  float2 i = floor(p), f = frac(p);
  f = f * f * (3.0 - 2.0 * f);
  return lerp(lerp(hash21(i), hash21(i + float2(1, 0)), f.x),
              lerp(hash21(i + float2(0, 1)), hash21(i + 1), f.x), f.y);
}

float3 palette(float t) {
  float3 a = float3(0.035, 0.055, 0.060);
  float3 b = float3(0.020, 0.310, 0.225);
  float3 c = float3(0.055, 0.720, 0.520);
  return a + b * (0.5 + 0.5 * cos(6.28318 * (c * t + float3(0.08, 0.20, 0.34))));
}

float4 vs(uint id : SV_VertexID) : SV_Position {
  float2 p = float2((id << 1) & 2, id & 2);
  return float4(p * float2(2, -2) + float2(-1, 1), 0, 1);
}

float4 ps(float4 position : SV_Position) : SV_Target {
  float2 resolution = max(u_params.yz, 1.0);
  float2 uv = position.xy / resolution;
  float2 p = (position.xy - 0.5 * resolution) / resolution.y;
  float t = u_params.x * 0.09;
  float wave = sin(p.x * 3.8 + t * 1.7) * 0.10 + sin(p.x * 8.0 - t) * 0.035;
  float ribbon = exp(-18.0 * abs(p.y + 0.15 - wave));
  float glow = exp(-4.5 * length(p - float2(0.35 * sin(t), -0.26)));
  float grain = noise(uv * 7.0 + t) * 0.035;
  float3 dark = float3(0.018, 0.024, 0.024);
  float3 light = float3(0.90, 0.93, 0.91);
  float3 base = lerp(dark, light, u_params.w);
  float energy = ribbon * 0.95 + glow * 0.42;
  float3 color = base + palette(uv.x + t) * energy;
  color += float3(0.015, 0.34, 0.22) * energy;
  color += grain * float3(0.05, 0.22, 0.16) * (1.0 - u_params.w * 0.7);
  return float4(color, 1.0);
}

float4 flame_ps(float4 position : SV_Position) : SV_Target {
  float2 uv = position.xy / float2(384.0, 96.0);
  float t = u_params.x;
  float n = noise(float2(uv.x * 5.2 - t * 0.72, uv.y * 6.0 + t * 0.24));
  n = 0.68 * n + 0.32 * noise(float2(uv.x * 10.0 + t * 0.31, uv.y * 11.0 - t * 0.38));
  float center = 0.50 + 0.055 * sin(t * 2.1 + uv.x * 8.5) + (n - 0.5) * 0.23;
  float width = 0.17 + uv.x * 0.075;
  float band = exp(-pow(abs(uv.y - center) / width, 2.0));
  float front = saturate((1.03 - uv.x + (n - 0.5) * 0.38) * 2.8);
  float source = exp(-uv.x * 2.45);
  float flame = band * front * source;
  float core = exp(-pow(abs(uv.y - center) / (width * 0.40), 2.0)) * exp(-uv.x * 4.0);

  float2 spark_space = float2(uv.x * 18.0 - t * 1.65, uv.y * 12.0 + t * 0.11);
  float2 spark_cell = floor(spark_space);
  float2 spark_local = frac(spark_space) - 0.5;
  float spark_seed = hash21(spark_cell);
  float2 spark_offset = float2(hash21(spark_cell + 7.1), hash21(spark_cell + 19.7)) - 0.5;
  float spark = step(0.925, spark_seed) * smoothstep(0.105, 0.012, length(spark_local - spark_offset * 0.58));
  spark *= smoothstep(0.50, 0.07, abs(uv.y - 0.5)) * exp(-uv.x * 1.25);

  float alpha = saturate(flame * 0.82 + core * 0.50 + spark * 0.92);
  alpha *= lerp(0.70, 0.94, 1.0 - u_params.w);
  float3 deep = float3(0.015, 0.32, 0.20);
  float3 bright = float3(0.32, 1.00, 0.68);
  float3 hot = float3(0.73, 1.00, 0.87);
  float3 color = lerp(deep, bright, saturate(flame + n * 0.28));
  color = lerp(color, hot, saturate(core * 1.35 + spark));
  return float4(color, alpha);
}
)HLSL";
}  // namespace
#else
namespace {
constexpr char kVertexShader[] = R"GLSL(#version 330 core
void main() {
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
)GLSL";

constexpr char kFragmentShader[] = R"GLSL(#version 330 core
uniform float u_time;
uniform vec2 u_resolution;
uniform float u_light;
out vec4 o_color;

float hash21(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}
float noise(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash21(i), hash21(i + vec2(1, 0)), f.x),
             mix(hash21(i + vec2(0, 1)), hash21(i + 1.0), f.x), f.y);
}
vec3 palette(float t) {
  vec3 a = vec3(0.035, 0.055, 0.060);
  vec3 b = vec3(0.020, 0.310, 0.225);
  vec3 c = vec3(0.055, 0.720, 0.520);
  return a + b * (0.5 + 0.5 * cos(6.28318 * (c * t + vec3(0.08, 0.20, 0.34))));
}
void main() {
  vec2 resolution = max(u_resolution, vec2(1.0));
  vec2 uv = gl_FragCoord.xy / resolution;
  uv.y = 1.0 - uv.y;
  vec2 p = (vec2(gl_FragCoord.x, resolution.y - gl_FragCoord.y) - 0.5 * resolution) / resolution.y;
  float t = u_time * 0.09;
  float wave = sin(p.x * 3.8 + t * 1.7) * 0.10 + sin(p.x * 8.0 - t) * 0.035;
  float ribbon = exp(-18.0 * abs(p.y + 0.15 - wave));
  float glow = exp(-4.5 * length(p - vec2(0.35 * sin(t), -0.26)));
  float grain = noise(uv * 7.0 + t) * 0.035;
  vec3 base = mix(vec3(0.018, 0.024, 0.024), vec3(0.90, 0.93, 0.91), u_light);
  float energy = ribbon * 0.95 + glow * 0.42;
  vec3 color = base + palette(uv.x + t) * energy;
  color += vec3(0.015, 0.34, 0.22) * energy;
  color += grain * vec3(0.05, 0.22, 0.16) * (1.0 - u_light * 0.7);
  o_color = vec4(color, 1.0);
}
)GLSL";

constexpr char kFlameFragmentShader[] = R"GLSL(#version 330 core
uniform float u_time;
uniform float u_light;
out vec4 o_color;

float hash21(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}
float noise(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash21(i), hash21(i + vec2(1, 0)), f.x),
             mix(hash21(i + vec2(0, 1)), hash21(i + 1.0), f.x), f.y);
}
void main() {
  vec2 uv = gl_FragCoord.xy / vec2(384.0, 96.0);
  float t = u_time;
  float n = noise(vec2(uv.x * 5.2 - t * 0.72, uv.y * 6.0 + t * 0.24));
  n = 0.68 * n + 0.32 * noise(vec2(uv.x * 10.0 + t * 0.31, uv.y * 11.0 - t * 0.38));
  float center = 0.50 + 0.055 * sin(t * 2.1 + uv.x * 8.5) + (n - 0.5) * 0.23;
  float width = 0.17 + uv.x * 0.075;
  float band = exp(-pow(abs(uv.y - center) / width, 2.0));
  float front = clamp((1.03 - uv.x + (n - 0.5) * 0.38) * 2.8, 0.0, 1.0);
  float source = exp(-uv.x * 2.45);
  float flame = band * front * source;
  float core = exp(-pow(abs(uv.y - center) / (width * 0.40), 2.0)) * exp(-uv.x * 4.0);

  vec2 spark_space = vec2(uv.x * 18.0 - t * 1.65, uv.y * 12.0 + t * 0.11);
  vec2 spark_cell = floor(spark_space);
  vec2 spark_local = fract(spark_space) - 0.5;
  float spark_seed = hash21(spark_cell);
  vec2 spark_offset = vec2(hash21(spark_cell + 7.1), hash21(spark_cell + 19.7)) - 0.5;
  float spark = step(0.925, spark_seed) * smoothstep(0.105, 0.012, length(spark_local - spark_offset * 0.58));
  spark *= smoothstep(0.50, 0.07, abs(uv.y - 0.5)) * exp(-uv.x * 1.25);

  float alpha = clamp(flame * 0.82 + core * 0.50 + spark * 0.92, 0.0, 1.0);
  alpha *= mix(0.70, 0.94, 1.0 - u_light);
  vec3 deep = vec3(0.015, 0.32, 0.20);
  vec3 bright = vec3(0.32, 1.00, 0.68);
  vec3 hot = vec3(0.73, 1.00, 0.87);
  vec3 color = mix(deep, bright, clamp(flame + n * 0.28, 0.0, 1.0));
  color = mix(color, hot, clamp(core * 1.35 + spark, 0.0, 1.0));
  o_color = vec4(color, alpha);
}
)GLSL";

GLuint Compile(GLenum type, const char* source) {
  const GLuint shader = glCreateShader(type);
  glShaderSource(shader, 1, &source, nullptr);
  glCompileShader(shader);
  GLint compiled = GL_FALSE;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
  if (compiled == GL_FALSE) {
    glDeleteShader(shader);
    return 0;
  }
  return shader;
}
}  // namespace
#endif

ShaderBackground::~ShaderBackground() { Shutdown(); }

bool ShaderBackground::Initialize(void* device, void* context) {
  Shutdown();
  impl_ = new Impl();
#ifdef _WIN32
  auto* native_device = static_cast<ID3D11Device*>(device);
  auto* native_context = static_cast<ID3D11DeviceContext*>(context);
  if (!native_device || !native_context) return false;
  Microsoft::WRL::ComPtr<ID3DBlob> vertex_blob;
  Microsoft::WRL::ComPtr<ID3DBlob> pixel_blob;
  Microsoft::WRL::ComPtr<ID3DBlob> flame_pixel_blob;
  if (FAILED(D3DCompile(kShader, sizeof(kShader) - 1, "gem16-aurora", nullptr, nullptr,
                        "vs", "vs_5_0", 0, 0, &vertex_blob, nullptr)) ||
      FAILED(D3DCompile(kShader, sizeof(kShader) - 1, "gem16-aurora", nullptr, nullptr,
                        "ps", "ps_5_0", 0, 0, &pixel_blob, nullptr)) ||
      FAILED(D3DCompile(kShader, sizeof(kShader) - 1, "gem16-navigation-flame", nullptr, nullptr,
                        "flame_ps", "ps_5_0", 0, 0, &flame_pixel_blob, nullptr))) return false;
  if (FAILED(native_device->CreateVertexShader(vertex_blob->GetBufferPointer(), vertex_blob->GetBufferSize(),
                                                nullptr, &impl_->vertex_shader)) ||
      FAILED(native_device->CreatePixelShader(pixel_blob->GetBufferPointer(), pixel_blob->GetBufferSize(),
                                               nullptr, &impl_->pixel_shader)) ||
      FAILED(native_device->CreatePixelShader(flame_pixel_blob->GetBufferPointer(),
                                               flame_pixel_blob->GetBufferSize(), nullptr,
                                               &impl_->flame_pixel_shader))) return false;
  D3D11_BUFFER_DESC descriptor{};
  descriptor.ByteWidth = 16;
  descriptor.Usage = D3D11_USAGE_DYNAMIC;
  descriptor.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
  descriptor.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
  if (FAILED(native_device->CreateBuffer(&descriptor, nullptr, &impl_->constants))) return false;
  D3D11_TEXTURE2D_DESC flame_description{};
  flame_description.Width = kFlameWidth;
  flame_description.Height = kFlameHeight;
  flame_description.MipLevels = 1;
  flame_description.ArraySize = 1;
  flame_description.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
  flame_description.SampleDesc.Count = 1;
  flame_description.Usage = D3D11_USAGE_DEFAULT;
  flame_description.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
  if (FAILED(native_device->CreateTexture2D(&flame_description, nullptr,
                                            &impl_->flame_texture)) ||
      FAILED(native_device->CreateRenderTargetView(impl_->flame_texture.Get(), nullptr,
                                                    &impl_->flame_target)) ||
      FAILED(native_device->CreateShaderResourceView(impl_->flame_texture.Get(), nullptr,
                                                      &impl_->flame_view))) return false;
  impl_->context = native_context;
  return true;
#else
  (void)device;
  (void)context;
  const GLuint vertex = Compile(GL_VERTEX_SHADER, kVertexShader);
  const GLuint fragment = Compile(GL_FRAGMENT_SHADER, kFragmentShader);
  const GLuint flame_fragment = Compile(GL_FRAGMENT_SHADER, kFlameFragmentShader);
  if (!vertex || !fragment || !flame_fragment) return false;
  impl_->program = glCreateProgram();
  glAttachShader(impl_->program, vertex);
  glAttachShader(impl_->program, fragment);
  glLinkProgram(impl_->program);
  GLint linked = GL_FALSE;
  glGetProgramiv(impl_->program, GL_LINK_STATUS, &linked);
  if (linked == GL_FALSE) return false;
  impl_->flame_program = glCreateProgram();
  glAttachShader(impl_->flame_program, vertex);
  glAttachShader(impl_->flame_program, flame_fragment);
  glLinkProgram(impl_->flame_program);
  glDeleteShader(vertex);
  glDeleteShader(fragment);
  glDeleteShader(flame_fragment);
  glGetProgramiv(impl_->flame_program, GL_LINK_STATUS, &linked);
  if (linked == GL_FALSE) return false;
  glGenVertexArrays(1, &impl_->vertex_array);
  impl_->time_location = glGetUniformLocation(impl_->program, "u_time");
  impl_->resolution_location = glGetUniformLocation(impl_->program, "u_resolution");
  impl_->theme_location = glGetUniformLocation(impl_->program, "u_light");
  impl_->flame_time_location = glGetUniformLocation(impl_->flame_program, "u_time");
  impl_->flame_theme_location = glGetUniformLocation(impl_->flame_program, "u_light");

  GLint previous_texture = 0;
  GLint previous_framebuffer = 0;
  glGetIntegerv(GL_TEXTURE_BINDING_2D, &previous_texture);
  glGetIntegerv(GL_FRAMEBUFFER_BINDING, &previous_framebuffer);
  glGenTextures(1, &impl_->flame_texture);
  glBindTexture(GL_TEXTURE_2D, impl_->flame_texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, kFlameWidth, kFlameHeight, 0,
               GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
  glGenFramebuffers(1, &impl_->flame_framebuffer);
  glBindFramebuffer(GL_FRAMEBUFFER, impl_->flame_framebuffer);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                         impl_->flame_texture, 0);
  const bool framebuffer_ready =
      glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
  glBindFramebuffer(GL_FRAMEBUFFER, static_cast<GLuint>(previous_framebuffer));
  glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(previous_texture));
  if (!framebuffer_ready) return false;
  return true;
#endif
}

void ShaderBackground::RenderNavigationFlame(float seconds, bool dark_theme) {
  if (!Ready()) return;
#ifdef _WIN32
  ID3D11ShaderResourceView* no_view = nullptr;
  impl_->context->PSSetShaderResources(0, 1, &no_view);
  ID3D11RenderTargetView* target = impl_->flame_target.Get();
  impl_->context->OMSetRenderTargets(1, &target, nullptr);
  impl_->context->OMSetBlendState(nullptr, nullptr, 0xffffffffU);
  const float clear[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  impl_->context->ClearRenderTargetView(target, clear);
  const D3D11_VIEWPORT viewport{0.0f, 0.0f, static_cast<float>(kFlameWidth),
                                static_cast<float>(kFlameHeight), 0.0f, 1.0f};
  impl_->context->RSSetViewports(1, &viewport);
  const D3D11_RECT scissor{0, 0, kFlameWidth, kFlameHeight};
  impl_->context->RSSetScissorRects(1, &scissor);
  D3D11_MAPPED_SUBRESOURCE mapped{};
  if (FAILED(impl_->context->Map(impl_->constants.Get(), 0,
                                 D3D11_MAP_WRITE_DISCARD, 0, &mapped))) return;
  const std::array<float, 4> values{seconds, static_cast<float>(kFlameWidth),
                                    static_cast<float>(kFlameHeight),
                                    dark_theme ? 0.0f : 1.0f};
  *static_cast<std::array<float, 4>*>(mapped.pData) = values;
  impl_->context->Unmap(impl_->constants.Get(), 0);
  ID3D11Buffer* constants = impl_->constants.Get();
  impl_->context->VSSetShader(impl_->vertex_shader.Get(), nullptr, 0);
  impl_->context->PSSetShader(impl_->flame_pixel_shader.Get(), nullptr, 0);
  impl_->context->PSSetConstantBuffers(0, 1, &constants);
  impl_->context->IASetInputLayout(nullptr);
  impl_->context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
  impl_->context->Draw(3, 0);
#else
  GLint previous_framebuffer = 0;
  GLint previous_program = 0;
  GLint previous_vertex_array = 0;
  GLint previous_viewport[4]{};
  GLfloat previous_clear[4]{};
  glGetIntegerv(GL_FRAMEBUFFER_BINDING, &previous_framebuffer);
  glGetIntegerv(GL_CURRENT_PROGRAM, &previous_program);
  glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &previous_vertex_array);
  glGetIntegerv(GL_VIEWPORT, previous_viewport);
  glGetFloatv(GL_COLOR_CLEAR_VALUE, previous_clear);
  const GLboolean blend_enabled = glIsEnabled(GL_BLEND);
  const GLboolean scissor_enabled = glIsEnabled(GL_SCISSOR_TEST);
  glBindFramebuffer(GL_FRAMEBUFFER, impl_->flame_framebuffer);
  glViewport(0, 0, kFlameWidth, kFlameHeight);
  glDisable(GL_BLEND);
  glDisable(GL_SCISSOR_TEST);
  glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
  glClear(GL_COLOR_BUFFER_BIT);
  glUseProgram(impl_->flame_program);
  glUniform1f(impl_->flame_time_location, seconds);
  glUniform1f(impl_->flame_theme_location, dark_theme ? 0.0f : 1.0f);
  glBindVertexArray(impl_->vertex_array);
  glDrawArrays(GL_TRIANGLES, 0, 3);
  glBindFramebuffer(GL_FRAMEBUFFER, static_cast<GLuint>(previous_framebuffer));
  glViewport(previous_viewport[0], previous_viewport[1], previous_viewport[2],
             previous_viewport[3]);
  glClearColor(previous_clear[0], previous_clear[1], previous_clear[2],
               previous_clear[3]);
  glUseProgram(static_cast<GLuint>(previous_program));
  glBindVertexArray(static_cast<GLuint>(previous_vertex_array));
  if (blend_enabled) glEnable(GL_BLEND); else glDisable(GL_BLEND);
  if (scissor_enabled) glEnable(GL_SCISSOR_TEST); else glDisable(GL_SCISSOR_TEST);
#endif
}

std::uint64_t ShaderBackground::NavigationFlameTexture() const {
  if (!Ready()) return 0;
#ifdef _WIN32
  return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(
      impl_->flame_view.Get()));
#else
  return static_cast<std::uint64_t>(impl_->flame_texture);
#endif
}

void ShaderBackground::Render(float seconds, float width, float height, bool dark_theme) {
  if (!Ready()) return;
#ifdef _WIN32
  D3D11_MAPPED_SUBRESOURCE mapped{};
  if (FAILED(impl_->context->Map(impl_->constants.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) return;
  const std::array<float, 4> values{seconds, width, height, dark_theme ? 0.0f : 1.0f};
  *static_cast<std::array<float, 4>*>(mapped.pData) = values;
  impl_->context->Unmap(impl_->constants.Get(), 0);
  ID3D11Buffer* constants = impl_->constants.Get();
  impl_->context->VSSetShader(impl_->vertex_shader.Get(), nullptr, 0);
  impl_->context->PSSetShader(impl_->pixel_shader.Get(), nullptr, 0);
  impl_->context->PSSetConstantBuffers(0, 1, &constants);
  impl_->context->IASetInputLayout(nullptr);
  impl_->context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
  impl_->context->Draw(3, 0);
#else
  glDisable(GL_SCISSOR_TEST);
  glDisable(GL_BLEND);
  glUseProgram(impl_->program);
  glUniform1f(impl_->time_location, seconds);
  glUniform2f(impl_->resolution_location, width, height);
  glUniform1f(impl_->theme_location, dark_theme ? 0.0f : 1.0f);
  glBindVertexArray(impl_->vertex_array);
  glDrawArrays(GL_TRIANGLES, 0, 3);
  glBindVertexArray(0);
  glUseProgram(0);
#endif
}

void ShaderBackground::Shutdown() {
  if (!impl_) return;
#ifndef _WIN32
  if (impl_->flame_framebuffer)
    glDeleteFramebuffers(1, &impl_->flame_framebuffer);
  if (impl_->flame_texture) glDeleteTextures(1, &impl_->flame_texture);
  if (impl_->flame_program) glDeleteProgram(impl_->flame_program);
  if (impl_->vertex_array) glDeleteVertexArrays(1, &impl_->vertex_array);
  if (impl_->program) glDeleteProgram(impl_->program);
#endif
  delete impl_;
  impl_ = nullptr;
}

bool ShaderBackground::Ready() const {
#ifdef _WIN32
  return impl_ && impl_->vertex_shader && impl_->pixel_shader &&
      impl_->flame_pixel_shader && impl_->flame_target && impl_->flame_view &&
      impl_->context;
#else
  return impl_ && impl_->program != 0 && impl_->flame_program != 0 &&
      impl_->flame_texture != 0 && impl_->flame_framebuffer != 0;
#endif
}

}  // namespace gem16::studio
