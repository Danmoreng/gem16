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

struct ShaderBackground::Impl {
#ifdef _WIN32
  Microsoft::WRL::ComPtr<ID3D11DeviceContext> context;
  Microsoft::WRL::ComPtr<ID3D11VertexShader> vertex_shader;
  Microsoft::WRL::ComPtr<ID3D11PixelShader> pixel_shader;
  Microsoft::WRL::ComPtr<ID3D11Buffer> constants;
  Microsoft::WRL::ComPtr<ID3D11BlendState> blend;
#else
  GLuint program = 0;
  GLuint vertex_array = 0;
  GLint time_location = -1;
  GLint resolution_location = -1;
  GLint theme_location = -1;
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
  if (FAILED(D3DCompile(kShader, sizeof(kShader) - 1, "gem16-aurora", nullptr, nullptr,
                        "vs", "vs_5_0", 0, 0, &vertex_blob, nullptr)) ||
      FAILED(D3DCompile(kShader, sizeof(kShader) - 1, "gem16-aurora", nullptr, nullptr,
                        "ps", "ps_5_0", 0, 0, &pixel_blob, nullptr))) return false;
  if (FAILED(native_device->CreateVertexShader(vertex_blob->GetBufferPointer(), vertex_blob->GetBufferSize(),
                                                nullptr, &impl_->vertex_shader)) ||
      FAILED(native_device->CreatePixelShader(pixel_blob->GetBufferPointer(), pixel_blob->GetBufferSize(),
                                               nullptr, &impl_->pixel_shader))) return false;
  D3D11_BUFFER_DESC descriptor{};
  descriptor.ByteWidth = 16;
  descriptor.Usage = D3D11_USAGE_DYNAMIC;
  descriptor.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
  descriptor.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
  if (FAILED(native_device->CreateBuffer(&descriptor, nullptr, &impl_->constants))) return false;
  impl_->context = native_context;
  return true;
#else
  (void)device;
  (void)context;
  const GLuint vertex = Compile(GL_VERTEX_SHADER, kVertexShader);
  const GLuint fragment = Compile(GL_FRAGMENT_SHADER, kFragmentShader);
  if (!vertex || !fragment) return false;
  impl_->program = glCreateProgram();
  glAttachShader(impl_->program, vertex);
  glAttachShader(impl_->program, fragment);
  glLinkProgram(impl_->program);
  glDeleteShader(vertex);
  glDeleteShader(fragment);
  GLint linked = GL_FALSE;
  glGetProgramiv(impl_->program, GL_LINK_STATUS, &linked);
  if (linked == GL_FALSE) return false;
  glGenVertexArrays(1, &impl_->vertex_array);
  impl_->time_location = glGetUniformLocation(impl_->program, "u_time");
  impl_->resolution_location = glGetUniformLocation(impl_->program, "u_resolution");
  impl_->theme_location = glGetUniformLocation(impl_->program, "u_light");
  return true;
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
  if (impl_->vertex_array) glDeleteVertexArrays(1, &impl_->vertex_array);
  if (impl_->program) glDeleteProgram(impl_->program);
#endif
  delete impl_;
  impl_ = nullptr;
}

bool ShaderBackground::Ready() const {
#ifdef _WIN32
  return impl_ && impl_->vertex_shader && impl_->pixel_shader && impl_->context;
#else
  return impl_ && impl_->program != 0;
#endif
}

}  // namespace gem16::studio
