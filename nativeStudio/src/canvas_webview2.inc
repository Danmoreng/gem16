// Windows implementation of CanvasBrowser, included inside gem16::studio.
using Microsoft::WRL::Callback;
using Microsoft::WRL::ComPtr;
namespace {
std::wstring Wide(const std::string& text) {
  int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(),
                              static_cast<int>(text.size()), nullptr, 0);
  std::wstring out(n, L'\0');
  if (n)
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(),
                        static_cast<int>(text.size()), out.data(), n);
  return out;
}
std::string Utf8(const wchar_t* text) {
  if (!text) return {};
  int n =
      WideCharToMultiByte(CP_UTF8, 0, text, -1, nullptr, 0, nullptr, nullptr);
  std::string out(n, '\0');
  if (n)
    WideCharToMultiByte(CP_UTF8, 0, text, -1, out.data(), n, nullptr, nullptr);
  if (n) out.pop_back();
  return out;
}
struct WinPreview : Preview {
  HWND window = nullptr;
  ComPtr<ICoreWebView2Environment> environment;
  ComPtr<ICoreWebView2Controller> controller;
  ComPtr<ICoreWebView2CompositionController> composition;
  ComPtr<ICoreWebView2> view;
  ComPtr<IDCompositionDevice> device;
  ComPtr<IDCompositionVisual> visual;
  ComPtr<IDCompositionTarget> target;
  unsigned buttons = 0;
  std::string page;
};
}  // namespace
struct CanvasBrowser::Impl {
  std::shared_ptr<WinPreview> state = std::make_shared<WinPreview>();
  void Capture() {
    auto s = state;
    if (!s->view || s->closed || !s->loaded ||
        Clock::now() - s->loaded_at < std::chrono::milliseconds(500))
      return;
    if (!s->observing) {
      s->observing = true;
      std::weak_ptr<WinPreview> weak = s;
      auto hr = s->view->ExecuteScript(
          Wide(kCanvasObservationScript).c_str(),
          Callback<ICoreWebView2ExecuteScriptCompletedHandler>(
              [weak](HRESULT hr, LPCWSTR json) -> HRESULT {
                if (auto s = weak.lock()) {
                  s->observing = false;
                  if (!s->closed && SUCCEEDED(hr)) {
                    auto p = gem16::json::Parse(Utf8(json));
                    if (p.ok() && p.value().is_string())
                      s->Observe(p.value().as_string());
                  }
                }
                return S_OK;
              })
              .Get());
      if (FAILED(hr)) s->observing = false;
    }
    if (s->capturing ||
        Clock::now() - s->captured_at < std::chrono::milliseconds(100))
      return;
    ComPtr<IStream> stream;
    if (FAILED(CreateStreamOnHGlobal(nullptr, TRUE, &stream))) return;
    s->capturing = true;
    s->captured_at = Clock::now();
    std::weak_ptr<WinPreview> weak = s;
    auto hr = s->view->CapturePreview(
        COREWEBVIEW2_CAPTURE_PREVIEW_IMAGE_FORMAT_PNG, stream.Get(),
        Callback<ICoreWebView2CapturePreviewCompletedHandler>([weak, stream](
                                                                  HRESULT hr)
                                                                  -> HRESULT {
          if (auto s = weak.lock()) {
            s->capturing = false;
            if (!s->closed && SUCCEEDED(hr)) {
              STATSTG stat{};
              if (SUCCEEDED(stream->Stat(&stat, STATFLAG_NONAME)) &&
                  stat.cbSize.QuadPart <= 8U * 1024U * 1024U) {
                std::vector<std::uint8_t> png(
                    static_cast<size_t>(stat.cbSize.QuadPart));
                LARGE_INTEGER zero{};
                ULONG read = 0;
                stream->Seek(zero, STREAM_SEEK_SET, nullptr);
                if (SUCCEEDED(stream->Read(
                        png.data(), static_cast<ULONG>(png.size()), &read)) &&
                    read == png.size()) {
                  s->pixels = DecodePreviewImage(png.data(), png.size());
                  s->png = std::move(png);
                }
              }
            }
          }
          return S_OK;
        }).Get());
    if (FAILED(hr)) s->capturing = false;
  }
};
int InitializeCanvasBrowser(int, char**) {
  initialized = SUCCEEDED(CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED));
  if (initialized) {
    LPWSTR version = nullptr;
    bool available = SUCCEEDED(
        GetAvailableCoreWebView2BrowserVersionString(nullptr, &version));
    CoTaskMemFree(version);
    if (!available) {
      CoUninitialize();
      initialized = false;
    }
  }
  return -1;
}
void PumpCanvasBrowser() {
  MSG message;
  for (int i = 0; i < 64 && PeekMessage(&message, nullptr, 0, 0, PM_REMOVE);
       ++i) {
    if (message.message == WM_QUIT) {
      PostQuitMessage(static_cast<int>(message.wParam));
      break;
    }
    TranslateMessage(&message);
    DispatchMessage(&message);
  }
}
void ShutdownCanvasBrowser() {
  if (initialized) CoUninitialize();
  initialized = false;
}
bool CanvasBrowserAvailable() { return initialized; }
CanvasBrowser::CanvasBrowser() : impl_(std::make_unique<Impl>()) {}
CanvasBrowser::~CanvasBrowser() { Close(); }
void CanvasBrowser::Close() {
  auto s = impl_->state;
  s->closed = true;
  if (s->controller) s->controller->Close();
  s->view.Reset();
  s->composition.Reset();
  s->controller.Reset();
  s->visual.Reset();
  s->target.Reset();
  s->device.Reset();
  s->environment.Reset();
  if (s->window) {
    DestroyWindow(s->window);
    s->window = nullptr;
  }
  impl_->state = std::make_shared<WinPreview>();
}
void CanvasBrowser::Load(const CanvasDocument& d) {
  auto key = d.id + ":" + std::to_string(d.revisions.back().number);
  if (Key() == key) {
    impl_->Capture();
    return;
  }
  Close();
  if (!initialized) return;
  auto s = impl_->state;
  s->key = key;
  s->page = CanvasPage(d);
  s->window = CreateWindowExW(
      WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE, L"STATIC", L"gem16 Canvas", WS_POPUP,
      0, 0, 1024, 768, nullptr, nullptr, GetModuleHandle(nullptr), nullptr);
  if (!s->window) {
    s->diagnostics = "Cannot create WebView host.";
    return;
  }
  // InPrivate controller: document storage is not persisted. The installed
  // Evergreen runtime owns its standard profile infrastructure outside our
  // package.
  wchar_t local[MAX_PATH]{};
  GetEnvironmentVariableW(L"LOCALAPPDATA", local, MAX_PATH);
  auto data = std::filesystem::path(local) / L"gem16" / L"webview";
  std::weak_ptr<WinPreview> weak = s;
  auto hr = CreateCoreWebView2EnvironmentWithOptions(
      nullptr, data.c_str(), nullptr,
      Callback<ICoreWebView2CreateCoreWebView2EnvironmentCompletedHandler>(
          [weak](HRESULT hr, ICoreWebView2Environment* env) -> HRESULT {
            auto s = weak.lock();
            if (!s || s->closed) return S_OK;
            if (FAILED(hr) || !env) {
              s->diagnostics = "System WebView2 initialization failed.";
              return S_OK;
            }
            s->environment = env;
            ComPtr<ICoreWebView2Environment10> env10;
            ComPtr<ICoreWebView2ControllerOptions> options;
            if (FAILED(env->QueryInterface(IID_PPV_ARGS(&env10))) ||
                FAILED(env10->CreateCoreWebView2ControllerOptions(&options))) {
              s->diagnostics = "Update the installed WebView2 runtime.";
              return S_OK;
            }
            options->put_IsInPrivateModeEnabled(TRUE);
            auto result = env10->CreateCoreWebView2CompositionControllerWithOptions(
                s->window, options.Get(),
                Callback<
                    ICoreWebView2CreateCoreWebView2CompositionControllerCompletedHandler>(
                    [weak](HRESULT hr,
                           ICoreWebView2CompositionController* c) -> HRESULT {
                      auto s = weak.lock();
                      if (!s || s->closed) return S_OK;
                      if (FAILED(hr) || !c) {
                        s->diagnostics = "System WebView2 controller failed.";
                        return S_OK;
                      }
                      s->composition = c;
                      c->QueryInterface(IID_PPV_ARGS(&s->controller));
                      if (!s->controller) {
                        s->diagnostics =
                            "System WebView2 controller unavailable.";
                        return S_OK;
                      }
                      if (FAILED(DCompositionCreateDevice(
                              nullptr, IID_PPV_ARGS(&s->device))) ||
                          FAILED(s->device->CreateVisual(&s->visual)) ||
                          FAILED(s->device->CreateTargetForHwnd(s->window, TRUE,
                                                                &s->target))) {
                        s->diagnostics =
                            "WebView composition initialization failed.";
                        return S_OK;
                      }
                      s->target->SetRoot(s->visual.Get());
                      c->put_RootVisualTarget(s->visual.Get());
                      s->device->Commit();
                      s->controller->put_Bounds({0, 0, 1024, 768});
                      s->controller->put_IsVisible(TRUE);
                      s->controller->get_CoreWebView2(&s->view);
                      ComPtr<ICoreWebView2Settings> settings;
                      s->view->get_Settings(&settings);
                      settings->put_AreDefaultContextMenusEnabled(FALSE);
                      settings->put_AreDevToolsEnabled(FALSE);
                      settings->put_AreDefaultScriptDialogsEnabled(FALSE);
                      settings->put_IsWebMessageEnabled(FALSE);
                      settings->put_AreHostObjectsAllowed(FALSE);
                      EventRegistrationToken token{};
                      auto nav =
                          Callback<ICoreWebView2NavigationStartingEventHandler>(
                              [](ICoreWebView2*,
                                 ICoreWebView2NavigationStartingEventArgs* args)
                                  -> HRESULT {
                                LPWSTR uri = nullptr;
                                args->get_Uri(&uri);
                                bool allowed =
                                    uri && (!wcscmp(uri, L"about:blank") ||
                                            !wcscmp(uri, L"about:srcdoc"));
                                CoTaskMemFree(uri);
                                args->put_Cancel(!allowed);
                                return S_OK;
                              });
                      s->view->add_NavigationStarting(nav.Get(), &token);
                      s->view->add_FrameNavigationStarting(nav.Get(), &token);
                      s->view->add_NewWindowRequested(
                          Callback<ICoreWebView2NewWindowRequestedEventHandler>(
                              [](ICoreWebView2*,
                                 ICoreWebView2NewWindowRequestedEventArgs* args)
                                  -> HRESULT {
                                args->put_Handled(TRUE);
                                return S_OK;
                              })
                              .Get(),
                          &token);
                      s->view->add_PermissionRequested(
                          Callback<
                              ICoreWebView2PermissionRequestedEventHandler>(
                              [](ICoreWebView2*,
                                 ICoreWebView2PermissionRequestedEventArgs*
                                     args) -> HRESULT {
                                args->put_State(
                                    COREWEBVIEW2_PERMISSION_STATE_DENY);
                                return S_OK;
                              })
                              .Get(),
                          &token);
                      s->view->add_NavigationCompleted(
                          Callback<
                              ICoreWebView2NavigationCompletedEventHandler>(
                              [weak](ICoreWebView2*,
                                     ICoreWebView2NavigationCompletedEventArgs*
                                         args) -> HRESULT {
                                if (auto s = weak.lock()) {
                                  BOOL success = FALSE;
                                  args->get_IsSuccess(&success);
                                  s->loaded = success;
                                  s->loaded_at = Clock::now();
                                  if (!success)
                                    s->diagnostics =
                                        "WebView navigation failed.";
                                }
                                return S_OK;
                              })
                              .Get(),
                          &token);
                      s->view->add_ProcessFailed(
                          Callback<ICoreWebView2ProcessFailedEventHandler>(
                              [weak](ICoreWebView2*,
                                     ICoreWebView2ProcessFailedEventArgs*)
                                  -> HRESULT {
                                if (auto s = weak.lock()) {
                                  s->loaded = s->ready = false;
                                  s->diagnostics =
                                      "System WebView2 renderer terminated.";
                                }
                                return S_OK;
                              })
                              .Get(),
                          &token);
                      // CSP also applies to subresources in the opaque
                      // generated frame.
                      s->view->NavigateToString(Wide(s->page).c_str());
                      s->page.clear();
                      return S_OK;
                    })
                    .Get());
            if (FAILED(result))
              s->diagnostics = "System WebView2 creation failed.";
            return S_OK;
          })
          .Get());
  if (FAILED(hr))
    s->diagnostics =
        "System WebView2 is unavailable. Install the Evergreen runtime.";
}
void CanvasBrowser::Mouse(int x, int y, int button, bool up, bool move,
                          int wheel) {
  auto s = impl_->state;
  if (!s->composition) return;
  auto kind = COREWEBVIEW2_MOUSE_EVENT_KIND_MOVE;
  unsigned flag = button == 0   ? MK_LBUTTON
                  : button == 1 ? MK_RBUTTON
                                : MK_MBUTTON;
  if (wheel)
    kind = COREWEBVIEW2_MOUSE_EVENT_KIND_WHEEL;
  else if (!move) {
    if (button == 0)
      kind = up ? COREWEBVIEW2_MOUSE_EVENT_KIND_LEFT_BUTTON_UP
                : COREWEBVIEW2_MOUSE_EVENT_KIND_LEFT_BUTTON_DOWN;
    else if (button == 1)
      kind = up ? COREWEBVIEW2_MOUSE_EVENT_KIND_RIGHT_BUTTON_UP
                : COREWEBVIEW2_MOUSE_EVENT_KIND_RIGHT_BUTTON_DOWN;
    else
      kind = up ? COREWEBVIEW2_MOUSE_EVENT_KIND_MIDDLE_BUTTON_UP
                : COREWEBVIEW2_MOUSE_EVENT_KIND_MIDDLE_BUTTON_DOWN;
    if (up)
      s->buttons &= ~flag;
    else
      s->buttons |= flag;
  }
  s->composition->SendMouseInput(
      kind, static_cast<COREWEBVIEW2_MOUSE_EVENT_VIRTUAL_KEYS>(s->buttons),
      wheel, {x, y});
}
