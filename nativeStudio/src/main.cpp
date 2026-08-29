#include "app.h"
#include "platform_ui.h"
#include "settings.h"
#include "shader_background.h"

#include "imgui.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <vector>

#ifdef _WIN32
#include "imgui_impl_dx11.h"
#include "imgui_impl_win32.h"
#include <d3d11.h>
#include <dwmapi.h>
#include <tchar.h>
#include <wrl/client.h>

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(
    HWND window, UINT message, WPARAM w_param, LPARAM l_param);
#else
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#endif

namespace {

void InitializeImGuiStyle(float ui_scale) {
  ImGuiIO& io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
  io.IniFilename = nullptr;
#ifdef _WIN32
  const char* font_path = "C:\\Windows\\Fonts\\segoeui.ttf";
#else
  const char* noto = "/usr/share/fonts/noto/NotoSans-Regular.ttf";
  const char* dejavu = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf";
  const char* font_path = std::filesystem::is_regular_file(noto) ? noto : dejavu;
#endif
  ImFont* font = std::filesystem::is_regular_file(font_path)
                     ? io.Fonts->AddFontFromFileTTF(font_path, 17.0f * ui_scale)
                     : nullptr;
  if (!font) io.Fonts->AddFontDefault();
  ImGui::GetStyle().CircleTessellationMaxError = 0.12f;
}

#ifdef _WIN32
struct D3dState {
  Microsoft::WRL::ComPtr<ID3D11Device> device;
  Microsoft::WRL::ComPtr<ID3D11DeviceContext> context;
  Microsoft::WRL::ComPtr<IDXGISwapChain> swap_chain;
  Microsoft::WRL::ComPtr<ID3D11RenderTargetView> render_target;
};

D3dState* g_d3d = nullptr;

bool CreateRenderTarget(D3dState& state) {
  Microsoft::WRL::ComPtr<ID3D11Texture2D> back_buffer;
  if (FAILED(state.swap_chain->GetBuffer(0, IID_PPV_ARGS(&back_buffer)))) return false;
  return SUCCEEDED(state.device->CreateRenderTargetView(back_buffer.Get(), nullptr,
                                                         &state.render_target));
}

LRESULT WINAPI WindowProc(HWND window, UINT message, WPARAM w_param, LPARAM l_param) {
  if (ImGui_ImplWin32_WndProcHandler(window, message, w_param, l_param)) return true;
  switch (message) {
    case WM_SIZE:
      if (g_d3d && g_d3d->swap_chain && w_param != SIZE_MINIMIZED) {
        g_d3d->render_target.Reset();
        g_d3d->swap_chain->ResizeBuffers(0, static_cast<UINT>(LOWORD(l_param)),
                                         static_cast<UINT>(HIWORD(l_param)), DXGI_FORMAT_UNKNOWN, 0);
        CreateRenderTarget(*g_d3d);
      }
      return 0;
    case WM_SYSCOMMAND:
      if ((w_param & 0xfff0) == SC_KEYMENU) return 0;
      break;
    case WM_DESTROY:
      PostQuitMessage(0);
      return 0;
    case WM_DROPFILES: {
      const HDROP drop = reinterpret_cast<HDROP>(w_param);
      const UINT count = DragQueryFileW(drop, 0xFFFFFFFF, nullptr, 0);
      std::vector<std::filesystem::path> paths;
      for (UINT index = 0; index < count; ++index) {
        const UINT length = DragQueryFileW(drop, index, nullptr, 0);
        std::wstring path(length + 1, L'\0');
        DragQueryFileW(drop, index, path.data(), length + 1);
        path.resize(length);
        paths.emplace_back(std::move(path));
      }
      DragFinish(drop);
      gem16::studio::QueueDroppedFiles(paths);
      return 0;
    }
  }
  return DefWindowProcW(window, message, w_param, l_param);
}

int RunWindows() {
  ImGui_ImplWin32_EnableDpiAwareness();
  WNDCLASSEXW window_class{sizeof(window_class), CS_CLASSDC, WindowProc, 0, 0,
                           GetModuleHandleW(nullptr), nullptr, nullptr, nullptr, nullptr,
                           L"gem16NativeStudio", nullptr};
  RegisterClassExW(&window_class);
  HWND window = CreateWindowW(window_class.lpszClassName, L"gem16 Studio",
                              WS_OVERLAPPEDWINDOW, 100, 100, 1320, 840, nullptr,
                              nullptr, window_class.hInstance, nullptr);
  if (!window) return 2;

  D3dState d3d;
  DXGI_SWAP_CHAIN_DESC descriptor{};
  descriptor.BufferCount = 2;
  descriptor.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
  descriptor.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
  descriptor.OutputWindow = window;
  descriptor.SampleDesc.Count = 1;
  descriptor.Windowed = TRUE;
  descriptor.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
  UINT flags = 0;
#ifndef NDEBUG
  flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
  D3D_FEATURE_LEVEL level{};
  const D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_0};
  HRESULT result = D3D11CreateDeviceAndSwapChain(
      nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, flags, levels, 2,
      D3D11_SDK_VERSION, &descriptor, &d3d.swap_chain, &d3d.device, &level,
      &d3d.context);
#ifndef NDEBUG
  if (FAILED(result)) {
    flags &= ~D3D11_CREATE_DEVICE_DEBUG;
    result = D3D11CreateDeviceAndSwapChain(
        nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, flags, levels, 2,
        D3D11_SDK_VERSION, &descriptor, &d3d.swap_chain, &d3d.device, &level,
        &d3d.context);
  }
#endif
  if (FAILED(result) || !CreateRenderTarget(d3d)) return 3;
  g_d3d = &d3d;

  BOOL dark = TRUE;
  DwmSetWindowAttribute(window, 20, &dark, sizeof(dark));
  ShowWindow(window, SW_SHOWDEFAULT);
  DragAcceptFiles(window, TRUE);
  UpdateWindow(window);

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  const auto settings = gem16::studio::LoadSettings();
  const float platform_scale = static_cast<float>(GetDpiForWindow(window)) / 96.0f;
  const float ui_scale = gem16::studio::ResolveUiScale(
      settings.ui_scale, platform_scale, false);
  InitializeImGuiStyle(ui_scale);
  ImGui_ImplWin32_Init(window);
  ImGui_ImplDX11_Init(d3d.device.Get(), d3d.context.Get());

  gem16::studio::ShaderBackground background;
  if (!background.Initialize(d3d.device.Get(), d3d.context.Get())) {
    std::fprintf(stderr, "gem16 Studio: Direct3D aurora shader unavailable\n");
  }
  const auto started = std::chrono::steady_clock::now();
  {
    gem16::studio::StudioApp app(settings, ui_scale);
    bool running = true;
    while (running) {
      MSG message;
      while (PeekMessage(&message, nullptr, 0U, 0U, PM_REMOVE)) {
        TranslateMessage(&message);
        DispatchMessage(&message);
        if (message.message == WM_QUIT) running = false;
      }
      if (!running) break;
      ImGui_ImplDX11_NewFrame();
      ImGui_ImplWin32_NewFrame();
      ImGui::NewFrame();
      app.Render();
      ImGui::Render();

      RECT area{};
      GetClientRect(window, &area);
      ID3D11RenderTargetView* target = d3d.render_target.Get();
      d3d.context->OMSetRenderTargets(1, &target, nullptr);
      const float clear[4] = {0.01f, 0.015f, 0.015f, 1.0f};
      d3d.context->ClearRenderTargetView(target, clear);
      D3D11_VIEWPORT viewport{0.0f, 0.0f, static_cast<float>(area.right),
                              static_cast<float>(area.bottom), 0.0f, 1.0f};
      d3d.context->RSSetViewports(1, &viewport);
      D3D11_RECT scissor{0, 0, area.right, area.bottom};
      d3d.context->RSSetScissorRects(1, &scissor);
      const float seconds = std::chrono::duration<float>(std::chrono::steady_clock::now() - started).count();
      background.Render(seconds, static_cast<float>(area.right), static_cast<float>(area.bottom), app.DarkTheme());
      ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
      d3d.swap_chain->Present(1, 0);
    }
  }
  background.Shutdown();
  ImGui_ImplDX11_Shutdown();
  ImGui_ImplWin32_Shutdown();
  ImGui::DestroyContext();
  g_d3d = nullptr;
  DestroyWindow(window);
  UnregisterClassW(window_class.lpszClassName, window_class.hInstance);
  return 0;
}
#else
void GlfwError(int code, const char* description) {
  std::fprintf(stderr, "GLFW error %d: %s\n", code, description ? description : "unknown");
}

int RunLinux() {
  glfwSetErrorCallback(GlfwError);
  if (!glfwInit()) return 2;
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
  GLFWwindow* window = glfwCreateWindow(1320, 840, "gem16 Studio", nullptr, nullptr);
  if (!window) {
    glfwTerminate();
    return 3;
  }
  glfwSetWindowPos(window, 100, 100);
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);
  glfwSetDropCallback(window, [](GLFWwindow*, int count, const char** paths) {
    std::vector<std::filesystem::path> dropped;
    dropped.reserve(static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index) dropped.emplace_back(paths[index]);
    gem16::studio::QueueDroppedFiles(dropped);
  });

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  float x_scale = 1.0f;
  float y_scale = 1.0f;
  glfwGetWindowContentScale(window, &x_scale, &y_scale);
  const auto settings = gem16::studio::LoadSettings();
  const float ui_scale = gem16::studio::ResolveUiScale(
      settings.ui_scale, std::max(x_scale, y_scale), true);
  InitializeImGuiStyle(ui_scale);
  if (!ImGui_ImplGlfw_InitForOpenGL(window, true) || !ImGui_ImplOpenGL3_Init("#version 330")) {
    glfwDestroyWindow(window);
    glfwTerminate();
    return 4;
  }

  gem16::studio::ShaderBackground background;
  if (!background.Initialize(nullptr, nullptr)) {
    std::fprintf(stderr, "gem16 Studio: OpenGL aurora shader unavailable\n");
  }
  const auto started = std::chrono::steady_clock::now();
  {
    gem16::studio::StudioApp app(settings, ui_scale);
    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();
      if (glfwGetWindowAttrib(window, GLFW_ICONIFIED)) {
        glfwWaitEventsTimeout(0.05);
        continue;
      }
      ImGui_ImplOpenGL3_NewFrame();
      ImGui_ImplGlfw_NewFrame();
      ImGui::NewFrame();
      app.Render();
      ImGui::Render();
      int width = 0;
      int height = 0;
      glfwGetFramebufferSize(window, &width, &height);
      glViewport(0, 0, width, height);
      glClearColor(0.01f, 0.015f, 0.015f, 1.0f);
      glClear(GL_COLOR_BUFFER_BIT);
      const float seconds = std::chrono::duration<float>(std::chrono::steady_clock::now() - started).count();
      background.Render(seconds, static_cast<float>(width), static_cast<float>(height), app.DarkTheme());
      ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
      glfwSwapBuffers(window);
    }
  }
  background.Shutdown();
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwDestroyWindow(window);
  glfwTerminate();
  return 0;
}
#endif

}  // namespace

#ifdef _WIN32
int WINAPI wWinMain(HINSTANCE, HINSTANCE, PWSTR, int) {
  return RunWindows();
#else
int main() {
  return RunLinux();
#endif
}
