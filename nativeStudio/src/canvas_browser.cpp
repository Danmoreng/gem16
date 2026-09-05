#include "canvas_browser.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <thread>

#include "canvas_page.h"
#include "util/json.h"
#if defined(GEM16_WITH_WEBKIT)
#include <webkit2/webkit2.h>
#include <gdk/gdkx.h>
#include <dlfcn.h>
#elif defined(GEM16_WITH_WEBVIEW2)
#include <windows.h>
#include <objbase.h>
#include <WebView2.h>
#include <dcomp.h>
#include <wrl.h>

#include <filesystem>
#endif

namespace gem16::studio {
namespace {
using Clock = std::chrono::steady_clock;
bool initialized = false;
struct Preview {
  std::string key, diagnostics;
  DecodedImage pixels;
  std::vector<std::uint8_t> png;
  bool closed = false, loaded = false, capturing = false, observing = false,
       ready = false;
  Clock::time_point loaded_at{}, captured_at{};
  unsigned viewport_revision = 0;
  void Observe(const std::string& text) {
    auto p = json::Parse(text, {8, 100, 32768});
    if (!p.ok() || !p.value().is_object()) return;
    auto d = p.value().find("diagnostics"), r = p.value().find("ready");
    if (d && d->is_string()) diagnostics = d->as_string().substr(0, 24000);
    ready = r && r->is_bool() && r->as_bool();
  }
};
}  // namespace
#if defined(GEM16_WITH_WEBKIT)
namespace {
GdkWindow* canvas_host = nullptr;
GdkDisplay* canvas_display = nullptr;
struct GtkPreview : Preview {
  bool direct = false, presented = false, visible = false, want_capture = false;
  Clock::time_point observed_at{};
  int width = 1024, height = 768;
};
}  // namespace
struct CanvasBrowser::Impl {
  std::shared_ptr<GtkPreview> state = std::make_shared<GtkPreview>();
  GtkWidget* window = nullptr;
  WebKitWebView* view = nullptr;
  GCancellable* cancel = nullptr;
  void Capture() {
    auto s = state;
    if (!view || !s->loaded || s->closed ||
        Clock::now() - s->loaded_at < std::chrono::milliseconds(500))
      return;
    if (!s->observing && Clock::now() - s->observed_at > std::chrono::milliseconds(100)) {
      s->observed_at = Clock::now();
      s->observing = true;
      webkit_web_view_evaluate_javascript(
          view, kCanvasObservationScript, -1, nullptr, nullptr, cancel,
          [](GObject* object, GAsyncResult* result, gpointer data) {
            std::unique_ptr<std::shared_ptr<GtkPreview>> hold(
                static_cast<std::shared_ptr<GtkPreview>*>(data));
            auto s = *hold;
            GError* error = nullptr;
            auto value = webkit_web_view_evaluate_javascript_finish(
                WEBKIT_WEB_VIEW(object), result, &error);
            if (value) {
              auto text = jsc_value_to_string(value);
              if (!s->closed && text) s->Observe(text);
              g_free(text);
              g_object_unref(value);
            }
            if (error) g_error_free(error);
            s->observing = false;
          },
          new std::shared_ptr<GtkPreview>(s));
    }
    if (s->direct && !s->want_capture) return;
    if (s->capturing ||
        Clock::now() - s->captured_at < std::chrono::milliseconds(100))
      return;
    s->capturing = true;
    s->captured_at = Clock::now();
    const auto viewport_revision = s->viewport_revision;
    struct Snapshot { std::shared_ptr<GtkPreview> state; unsigned revision; };
    webkit_web_view_get_snapshot(
        view, WEBKIT_SNAPSHOT_REGION_VISIBLE, WEBKIT_SNAPSHOT_OPTIONS_NONE,
        cancel,
        [](GObject* object, GAsyncResult* result, gpointer data) {
          std::unique_ptr<Snapshot> hold(static_cast<Snapshot*>(data));
          auto s = hold->state;
          GError* error = nullptr;
          auto surface = webkit_web_view_get_snapshot_finish(
              WEBKIT_WEB_VIEW(object), result, &error);
          if (surface && !s->closed && hold->revision == s->viewport_revision) {
            std::vector<std::uint8_t> png;
            auto status = cairo_surface_write_to_png_stream(
                surface,
                [](void* p, const unsigned char* bytes, unsigned count) {
                  auto& out = *static_cast<std::vector<std::uint8_t>*>(p);
                  if (out.size() + count > 8U * 1024U * 1024U)
                    return CAIRO_STATUS_WRITE_ERROR;
                  out.insert(out.end(), bytes, bytes + count);
                  return CAIRO_STATUS_SUCCESS;
                },
                &png);
            if (status == CAIRO_STATUS_SUCCESS) {
              auto pixels = DecodePreviewImage(png.data(), png.size());
              if (pixels.width == s->width && pixels.height == s->height) {
                s->pixels = std::move(pixels);
                s->png = std::move(png);
                s->want_capture = false;
              }
            }
          }
          if (surface) cairo_surface_destroy(surface);
          if (error) {
            if (!s->closed) s->diagnostics = error->message;
            g_error_free(error);
          }
          s->capturing = false;
        },
        new Snapshot{s, viewport_revision});
  }
};
int InitializeCanvasBrowser(int, char**) {
  // Both toolkits must use the same X11 display. Wayland sessions use
  // XWayland; native Wayland cannot reparent a foreign toolkit surface.
  gdk_set_allowed_backends("x11");
  initialized = gtk_init_check(nullptr, nullptr);
  if (initialized) {
    // Dedicated connection: Canvas uses physical pixels, while GTK file
    // dialogs retain the desktop's scale on the default connection.
    if (!canvas_display)
      canvas_display = gdk_display_open(gdk_display_get_name(gdk_display_get_default()));
    initialized = canvas_display && GDK_IS_X11_DISPLAY(canvas_display);
    if (initialized) gdk_x11_display_set_window_scale(canvas_display, 1);
  }
  return -1;
}
void PumpCanvasBrowser() {
  if (initialized)
    for (int i = 0; i < 32 && g_main_context_pending(nullptr); ++i)
      g_main_context_iteration(nullptr, false);
}
void ShutdownCanvasBrowser() {
  PumpCanvasBrowser();
  SetCanvasBrowserHost(nullptr);
  // GDK owns the display until process exit, just like its default display.
  // WebKit can still release asynchronous renderer resources after Close().
  initialized = false;
}
bool CanvasBrowserAvailable() { return initialized; }
CanvasBrowser::CanvasBrowser() : impl_(std::make_unique<Impl>()) {}
CanvasBrowser::~CanvasBrowser() { Close(); }
void CanvasBrowser::Close() {
  impl_->state->closed = true;
  if (impl_->cancel) {
    g_cancellable_cancel(impl_->cancel);
    g_object_unref(impl_->cancel);
    impl_->cancel = nullptr;
  }
  if (impl_->window) {
    gtk_widget_destroy(impl_->window);
    impl_->window = nullptr;
    impl_->view = nullptr;
  }
  impl_->state = std::make_shared<GtkPreview>();
}
void CanvasBrowser::Load(const CanvasDocument& d) {
  auto key = d.id + ":" + std::to_string(d.revisions.back().number);
  if (Key() == key) {
    impl_->Capture();
    return;
  }
  Close();
  if (!initialized) return;
  auto& i = *impl_;
  auto s = i.state;
  s->key = key;
  s->width = width_;
  s->height = height_;
  i.cancel = g_cancellable_new();
  auto context = webkit_web_context_new_ephemeral();
  webkit_web_context_set_sandbox_enabled(context, true);
  g_signal_connect(
      context, "download-started",
      G_CALLBACK(+[](WebKitWebContext*, WebKitDownload* download, gpointer) {
        webkit_download_cancel(download);
      }),
      nullptr);
  i.view = WEBKIT_WEB_VIEW(webkit_web_view_new_with_context(context));
  g_object_unref(context);
  auto settings = webkit_web_view_get_settings(i.view);
  webkit_settings_set_hardware_acceleration_policy(
      settings, WEBKIT_HARDWARE_ACCELERATION_POLICY_NEVER);
  webkit_settings_set_enable_webgl(settings, false);
  webkit_settings_set_enable_html5_local_storage(settings, false);
  webkit_settings_set_enable_html5_database(settings, false);
  webkit_settings_set_enable_page_cache(settings, false);
  webkit_settings_set_media_playback_requires_user_gesture(settings, true);
  webkit_web_view_set_is_muted(i.view, true);
  // A real X11 child surface receives native WebKit mouse, keyboard and
  // scrolling events. The no-host window exists only for diagnostic smokes.
  s->direct = canvas_host != nullptr;
  i.window = gtk_window_new(GTK_WINDOW_POPUP);
  gtk_window_set_screen(GTK_WINDOW(i.window), gdk_display_get_default_screen(canvas_display));
  gtk_window_set_accept_focus(GTK_WINDOW(i.window), s->direct);
  gtk_window_set_focus_on_map(GTK_WINDOW(i.window), false);
  if (!s->direct) {
    gtk_widget_set_opacity(i.window, 0);
    gtk_window_move(GTK_WINDOW(i.window), -12000, -12000);
  }
  // GLFW X11 coordinates and viewport dimensions are physical pixels.
  // Keep the GTK surface in the same coordinate space even with GDK_SCALE=2.
  gtk_widget_realize(i.window);
  gtk_widget_set_size_request(GTK_WIDGET(i.view), width_, height_);
  gtk_container_add(GTK_CONTAINER(i.window), GTK_WIDGET(i.view));
  if (s->direct) {
    gdk_window_reparent(gtk_widget_get_window(i.window), canvas_host, 0, 0);
    g_signal_connect(i.view, "button-press-event",
        G_CALLBACK(+[](GtkWidget* widget, GdkEventButton* event, gpointer window) -> gboolean {
          auto surface = gtk_widget_get_window(GTK_WIDGET(window));
          XSetInputFocus(gdk_x11_display_get_xdisplay(canvas_display),
                         gdk_x11_window_get_xid(surface), RevertToParent, event->time);
          gtk_widget_grab_focus(widget);
          return false;
        }), i.window);
  }
  auto raw = s.get();
  g_signal_connect(
      i.view, "load-changed",
      G_CALLBACK(+[](WebKitWebView*, WebKitLoadEvent event, gpointer p) {
        auto s = static_cast<GtkPreview*>(p);
        if (event == WEBKIT_LOAD_FINISHED) {
          s->loaded = true;
          s->loaded_at = Clock::now();
        }
      }),
      raw);
  g_signal_connect(
      i.view, "decide-policy",
      G_CALLBACK(+[](WebKitWebView*, WebKitPolicyDecision* decision,
                     WebKitPolicyDecisionType type, gpointer p) -> gboolean {
        if (type == WEBKIT_POLICY_DECISION_TYPE_NAVIGATION_ACTION ||
            type == WEBKIT_POLICY_DECISION_TYPE_NEW_WINDOW_ACTION) {
          auto nav = WEBKIT_NAVIGATION_POLICY_DECISION(decision);
          auto action =
              webkit_navigation_policy_decision_get_navigation_action(nav);
          const char* uri = webkit_uri_request_get_uri(
              webkit_navigation_action_get_request(action));
          if (type == WEBKIT_POLICY_DECISION_TYPE_NEW_WINDOW_ACTION || !uri ||
              (std::strcmp(uri, "about:blank") &&
               std::strcmp(uri, "about:srcdoc"))) {
            auto s = static_cast<GtkPreview*>(p);
            if (s->diagnostics.size() < 22000)
              s->diagnostics += "Blocked navigation\n";
            webkit_policy_decision_ignore(decision);
            return true;
          }
        }
        return false;
      }),
      raw);
  g_signal_connect(i.view, "permission-request",
                   G_CALLBACK(+[](WebKitWebView*, WebKitPermissionRequest* r,
                                  gpointer) -> gboolean {
                     webkit_permission_request_deny(r);
                     return true;
                   }),
                   nullptr);
  g_signal_connect(i.view, "script-dialog",
                   G_CALLBACK(+[](WebKitWebView*, WebKitScriptDialog*,
                                  gpointer) -> gboolean { return true; }),
                   nullptr);
  g_signal_connect(
      i.view, "web-process-terminated",
      G_CALLBACK(
          +[](WebKitWebView*, WebKitWebProcessTerminationReason, gpointer p) {
            auto s = static_cast<GtkPreview*>(p);
            s->ready = s->loaded = false;
            s->diagnostics = "System WebView renderer terminated.";
          }),
      raw);
  gtk_widget_show_all(i.window);
  if (s->direct) {
    gtk_widget_hide(i.window);
  } else {
    auto empty_region = cairo_region_create();
    gdk_window_input_shape_combine_region(gtk_widget_get_window(i.window),
                                          empty_region, 0, 0);
    cairo_region_destroy(empty_region);
  }
  auto page = CanvasPage(d);
  webkit_web_view_load_html(i.view, page.c_str(), "about:blank");
}
void CanvasBrowser::Mouse(int x, int y, int button, bool up, bool move,
                          int wheel) {
  if (!impl_->view) return;
  auto widget = GTK_WIDGET(impl_->view);
  auto window = gtk_widget_get_window(widget);
  if (!window) return;
  auto type = wheel  ? GDK_SCROLL
              : move ? GDK_MOTION_NOTIFY
              : up   ? GDK_BUTTON_RELEASE
                     : GDK_BUTTON_PRESS;
  auto event = gdk_event_new(type);
  event->any.window = GDK_WINDOW(g_object_ref(window));
  event->any.send_event = true;
  auto seat = gdk_display_get_default_seat(gdk_window_get_display(window));
  gdk_event_set_device(event, gdk_seat_get_pointer(seat));
  gdk_event_set_source_device(event, gdk_seat_get_pointer(seat));
  if (wheel) {
    event->scroll.x = x;
    event->scroll.y = y;
    event->scroll.direction = GDK_SCROLL_SMOOTH;
    event->scroll.delta_y = -wheel / 100.0;
  } else if (move) {
    event->motion.x = x;
    event->motion.y = y;
  } else {
    event->button.x = x;
    event->button.y = y;
    event->button.button = button == 0 ? 1 : button == 1 ? 3 : 2;
    event->button.time = static_cast<guint32>(g_get_monotonic_time() / 1000);
    event->button.state = up ? GDK_BUTTON1_MASK : 0;
  }
  if (type == GDK_BUTTON_PRESS) gtk_widget_grab_focus(widget);
  gboolean handled = false;
  g_signal_emit_by_name(widget,
                        wheel  ? "scroll-event"
                        : move ? "motion-notify-event"
                        : up   ? "button-release-event"
                               : "button-press-event",
                        event, &handled);
  gdk_event_free(event);
}
#elif defined(GEM16_WITH_WEBVIEW2)
#include "canvas_webview2.inc"
#else
struct CanvasBrowser::Impl {
  std::shared_ptr<Preview> state = std::make_shared<Preview>();
  void Capture() {}
};
CanvasBrowser::CanvasBrowser() : impl_(std::make_unique<Impl>()) {}
CanvasBrowser::~CanvasBrowser() = default;
int InitializeCanvasBrowser(int, char**) { return -1; }
void PumpCanvasBrowser() {}
void ShutdownCanvasBrowser() {}
bool CanvasBrowserAvailable() { return false; }
void CanvasBrowser::Load(const CanvasDocument&) {}
void CanvasBrowser::Close() {}
void CanvasBrowser::Mouse(int, int, int, bool, bool, int) {}
#endif
void CanvasBrowser::SetViewport(int width, int height) {
  // Match the image decoder's bounded preview allocation.
  width = std::clamp(width, 1, 7680);
  height = std::clamp(height, 1, 4320);
  const double scale = std::min(1.0, std::sqrt(16.0 * 1024 * 1024 / (double(width) * height)));
  width = std::max(1, static_cast<int>(width * scale));
  height = std::max(1, static_cast<int>(height * scale));
  if (width == width_ && height == height_) return;
  width_ = width;
  height_ = height;
  auto s = impl_->state;
  ++s->viewport_revision;
  s->png.clear();
  s->pixels = {};
  s->captured_at = {};
#if defined(GEM16_WITH_WEBVIEW2)
  if (s->window)
    SetWindowPos(s->window, nullptr, 0, 0, width, height,
                 SWP_NOMOVE | SWP_NOACTIVATE | SWP_NOZORDER);
  s->width = width;
  s->height = height;
  if (s->controller) s->controller->put_Bounds({0, 0, width, height});
#elif defined(GEM16_WITH_WEBKIT)
  s->width = width;
  s->height = height;
  if (impl_->view) gtk_widget_set_size_request(GTK_WIDGET(impl_->view), width, height);
  if (impl_->window) gtk_window_resize(GTK_WINDOW(impl_->window), width, height);
#endif
}
void SetCanvasBrowserHost(void* window) {
#if defined(GEM16_WITH_WEBVIEW2)
  canvas_host = static_cast<HWND>(window);
#elif defined(GEM16_WITH_WEBKIT)
  if (canvas_host) g_object_unref(canvas_host);
  canvas_host = window && initialized
      ? gdk_x11_window_foreign_new_for_display(canvas_display,
            static_cast<Window>(reinterpret_cast<std::uintptr_t>(window)))
      : nullptr;
#else
  (void)window;
#endif
}
void CanvasBrowser::BeginFrame() {
#if defined(GEM16_WITH_WEBVIEW2) || defined(GEM16_WITH_WEBKIT)
  impl_->state->presented = false;
#endif
}
void CanvasBrowser::EndFrame(bool covered) {
#if defined(GEM16_WITH_WEBKIT)
  if (covered) impl_->state->presented = false;
#else
  (void)covered;
#endif
#if defined(GEM16_WITH_WEBVIEW2)
  auto s = impl_->state;
  if (s->direct && s->visible && !s->presented) {
    ShowWindow(s->window, SW_HIDE);
    s->visible = false;
    if (s->controller) s->controller->put_IsVisible(FALSE);
  }
#elif defined(GEM16_WITH_WEBKIT)
  auto s = impl_->state;
  if (s->direct && s->visible != s->presented) {
    // Decide mapping once, after ImGui has finished opening menus/modals.
    // Showing in Present and hiding here would flash beneath an open popup.
    if (s->presented) gtk_widget_show(impl_->window);
    else gtk_widget_hide(impl_->window);
    s->visible = s->presented;
  }
#endif
}
bool CanvasBrowser::Present(int client_x, int client_y) {
#if defined(GEM16_WITH_WEBVIEW2)
  auto s = impl_->state;
  if (!s->direct) return false;
  s->presented = true;
  POINT point{client_x, client_y};
  RECT bounds{point.x, point.y, point.x + width_, point.y + height_};
  if (!EqualRect(&bounds, &s->placement)) {
    SetWindowPos(s->window, nullptr, point.x, point.y, width_, height_,
                 SWP_NOACTIVATE | SWP_NOZORDER);
    s->placement = bounds;
    if (s->controller) s->controller->NotifyParentWindowPositionChanged();
  }
  if (!s->visible) {
    ShowWindow(s->window, SW_SHOWNOACTIVATE);
    s->visible = true;
    if (s->controller) s->controller->put_IsVisible(TRUE);
  }
  return true;
#elif defined(GEM16_WITH_WEBKIT)
  auto s = impl_->state;
  if (!s->direct || !impl_->window) return false;
  s->presented = true;
  auto window = gtk_widget_get_window(impl_->window);
  gdk_window_move_resize(window, client_x, client_y, width_, height_);
  return true;
#else
  (void)client_x;
  (void)client_y;
  return false;
#endif
}
void CanvasBrowser::RequestScreenshot() {
#if defined(GEM16_WITH_WEBVIEW2) || defined(GEM16_WITH_WEBKIT)
  auto s = impl_->state;
  s->png.clear();
  s->want_capture = true;
#endif
}
bool CanvasBrowser::Ready() const {
  impl_->Capture();
  auto s = impl_->state;
#if defined(GEM16_WITH_WEBVIEW2) || defined(GEM16_WITH_WEBKIT)
  if (s->direct) return s->ready && !s->want_capture;
#endif
  return s->ready && !s->png.empty();
}
std::string CanvasBrowser::Key() const { return impl_->state->key; }
std::string CanvasBrowser::Diagnostics() const {
  return initialized ? impl_->state->diagnostics
                     : "System WebView unavailable.";
}
DecodedImage CanvasBrowser::Pixels() const {
  impl_->Capture();
  return impl_->state->pixels;
}
std::vector<std::uint8_t> CanvasBrowser::Screenshot() const {
  return Ready() ? impl_->state->png : std::vector<std::uint8_t>{};
}
int RunCanvasBrowserSmoke(const std::string& output) {
  const auto wait = [](CanvasBrowser& browser, auto predicate) {
    for (int i = 0; i < 1000; ++i) {
      PumpCanvasBrowser();
      browser.Ready();
      if (predicate()) return true;
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return false;
  };
  CanvasDocument d{
      std::string(64, 'a'),
      "System WebView smoke",
      "html",
      {{1,
        R"HTML(<body style="margin:0;background:#123456;color:white;font:32px sans-serif"><h1>Canvas system WebView</h1><button style="position:absolute;left:20px;top:220px;width:180px;height:48px;background:#eeeecc" onclick="this.textContent='Clicked';document.body.style.background='#118844';console.error('interaction-passed')">Interactive button</button><div id="state"></div><script>document.getElementById('state').textContent='JavaScript runs in the isolated preview';fetch('http://127.0.0.1:9/blocked').catch(()=>console.error('Network blocked'));fetch('file:///etc/passwd').catch(()=>console.error('File blocked'));throw new Error('deliberate-js-error');</script></body>)HTML"}}};
  CanvasBrowser browser;
  browser.Load(d);
  if (!wait(browser, [&] {
        const auto diagnostics = browser.Diagnostics();
        return browser.Ready() &&
               (diagnostics.find("Script error") != std::string::npos ||
                diagnostics.find("deliberate-js-error") != std::string::npos);
      })) {
    std::fprintf(stderr, "Canvas render/diagnostics failed: %s\n",
                 browser.Diagnostics().c_str());
    return 1;
  }
  auto image = browser.Screenshot();
  auto pixels = browser.Pixels();
  if (pixels.width != 1024 || pixels.height != 768 || pixels.rgba.empty() ||
      pixels.rgba[0] != 0x12) {
    std::fprintf(stderr, "Canvas screenshot dimensions/color failed: %dx%d, first red=%u\n",
                 pixels.width, pixels.height,
                 pixels.rgba.empty() ? 0 : pixels.rgba[0]);
    return 1;
  }
  if (browser.Diagnostics().find("Network blocked") == std::string::npos ||
      browser.Diagnostics().find("File blocked") == std::string::npos) {
    std::fprintf(stderr, "Canvas isolation diagnostics missing\n");
    return 1;
  }
  // The host document must not add a scrollbar or an inline-iframe baseline
  // below a page that fits. Both far edges must show the child background.
  for (const auto at : {((768 * 1024) - 1) * 4, (400 * 1024 + 1023) * 4}) {
    if (pixels.rgba[at] != 0x12 || pixels.rgba[at + 1] != 0x34 ||
        pixels.rgba[at + 2] != 0x56) {
      std::fprintf(stderr, "Canvas outer frame adds overflow or a border\n");
      return 1;
    }
  }
  std::ofstream file(output, std::ios::binary);
  file.write(reinterpret_cast<const char*>(image.data()), image.size());
  file.close();
  if (!file) return 1;
  int button_y = 0;
  for (int y = 200; y < 600; ++y) {
    const auto at = (y * 1024 + 50) * 4;
    if (pixels.rgba[at] == 0xee && pixels.rgba[at + 1] == 0xee &&
        pixels.rgba[at + 2] == 0xcc) {
      button_y = y + 10;
      break;
    }
  }

  browser.Mouse(50, button_y, 0, false, true);
  browser.Mouse(50, button_y, 0, false, false);
  browser.Mouse(50, button_y, 0, true, false);
  if (!wait(browser, [&] {
        auto p = browser.Pixels();
        return browser.Diagnostics().find("interaction-passed") !=
                   std::string::npos &&
               !p.rgba.empty() && p.rgba[0] == 0x11;
      })) {
    std::fprintf(stderr, "Canvas mouse interaction failed: %s\n",
                 browser.Diagnostics().c_str());
    return 1;
  }
  for (const auto size : {ImageDimensions{800, 600}, ImageDimensions{1440, 900}}) {
    browser.SetViewport(size.width, size.height);
    if (browser.Ready()) {
      std::fprintf(stderr, "Canvas reused a screenshot from before resize\n");
      return 1;
    }
    if (!wait(browser, [&] {
          const auto p = browser.Pixels();
          return browser.Ready() && p.width == size.width && p.height == size.height &&
                 !p.rgba.empty() && p.rgba[0] == 0x11 &&
                 browser.Diagnostics().find("interaction-passed") != std::string::npos;
        })) {
      std::fprintf(stderr, "Canvas resize/state preservation failed: %s\n",
                   browser.Diagnostics().c_str());
      return 1;
    }
  }
  d.type = "svg";
  d.revisions.push_back(
      {2, "<svg xmlns=\"http://www.w3.org/2000/svg\"><path></svg>"});
  browser.Load(d);
  if (browser.Ready()) {
    std::fprintf(stderr, "Canvas reused stale screenshot\n");
    return 1;
  }
  if (!wait(browser, [&] {
        return browser.Ready() && browser.Diagnostics().find(
                                      "SVG parse error") != std::string::npos;
      })) {
    std::fprintf(stderr, "Canvas SVG diagnostics failed: %s\n",
                 browser.Diagnostics().c_str());
    return 1;
  }
  d.type = "html";
  d.revisions.push_back({3, R"HTML(<body style="margin:0;background:#118844"><div style="height:200px;background:#123456"></div><div style="height:3000px"></div><script>addEventListener('scroll',()=>{if(scrollY>200)console.error('inner-scroll-passed')})</script></body>)HTML"});
  browser.Load(d);
  if (!wait(browser, [&] { return browser.Ready(); })) return 1;
  browser.Mouse(100, 100, 0, false, true);
  browser.Mouse(100, 100, 0, false, false, -1200);
  if (!wait(browser, [&] {
        auto p = browser.Pixels();
        // Inspect inside the page: WebKit may draw a focus indicator at
        // the viewport edge after the native click preceding the scroll.
        const auto at = (50 * p.width + 50) * 4;
        return browser.Diagnostics().find("inner-scroll-passed") != std::string::npos &&
               p.width > 50 && p.height > 50 && !p.rgba.empty() &&
               p.rgba[at] == 0x11 && p.rgba[at + 1] == 0x88 && p.rgba[at + 2] == 0x44;
      })) {
    std::fprintf(stderr, "Canvas inner page scrolling failed: %s\n",
                 browser.Diagnostics().c_str());
    return 1;
  }
  // Closing while a snapshot/observation is pending must not retain callbacks
  // into the destroyed browser. The same lifecycle is used when switching
  // chats.
  browser.Close();
  PumpCanvasBrowser();
#if defined(GEM16_WITH_WEBKIT)
  // Exercise a native child on the same foreign-window boundary as GLFW.
  auto host = gtk_window_new(GTK_WINDOW_TOPLEVEL);
  gtk_window_set_default_size(GTK_WINDOW(host), 1200, 900);
  gtk_widget_show(host);
  SetCanvasBrowserHost(reinterpret_cast<void*>(static_cast<std::uintptr_t>(
      gdk_x11_window_get_xid(gtk_widget_get_window(host)))));
  const bool direct_passed = [&] {
    CanvasBrowser live;
    d.revisions.push_back({4, R"HTML(<body style="margin:0;background:#123456"><input style="position:absolute;left:20px;top:20px" onfocus="console.error('native-focus')" oninput="console.error('native-key-'+this.value)"><button style="position:absolute;left:20px;top:80px;width:180px;height:50px" onclick="document.body.style.background='#118844';console.error('native-click')">Click</button><script>let frames=0;function tick(){if(++frames===30)console.error('animation-passed');requestAnimationFrame(tick)}tick()</script></body>)HTML"});
    live.SetViewport(800, 600);
    live.Load(d);
    live.BeginFrame();
    if (!live.Present(200, 100)) return false;
    live.EndFrame();
    if (!wait(live, [&] {
          return live.Ready() && live.Diagnostics().find("animation-passed") != std::string::npos;
        })) { std::fprintf(stderr, "Live load/animation: %s\n", live.Diagnostics().c_str()); return false; }
    auto state = live.impl_->state;
    auto window = gtk_widget_get_window(live.impl_->window);
    int x = 0, y = 0;
    gdk_window_get_position(window, &x, &y);
    Window root = 0, parent = 0, *children = nullptr;
    unsigned count = 0;
    XQueryTree(gdk_x11_display_get_xdisplay(canvas_display),
               gdk_x11_window_get_xid(window), &root, &parent, &children, &count);
    if (children) XFree(children);
    if (!state->direct || !state->png.empty() || state->capturing ||
        x != 200 || y != 100 || parent != gdk_x11_window_get_xid(canvas_host))
      { std::fprintf(stderr, "Live placement: %d,%d\n", x, y); return false; }
    // XTEST drives the real server input route (not CanvasBrowser::Mouse or
    // synthetic GTK signals). This dependency is only needed by the smoke.
    auto xtst = dlopen("libXtst.so.6", RTLD_NOW | RTLD_LOCAL);
    if (!xtst) return false;
    auto motion = reinterpret_cast<int (*)(Display*, int, int, int, unsigned long)>(
        dlsym(xtst, "XTestFakeMotionEvent"));
    auto button = reinterpret_cast<int (*)(Display*, unsigned, Bool, unsigned long)>(
        dlsym(xtst, "XTestFakeButtonEvent"));
    auto key = reinterpret_cast<int (*)(Display*, unsigned, Bool, unsigned long)>(
        dlsym(xtst, "XTestFakeKeyEvent"));
    if (!motion || !button || !key) { dlclose(xtst); return false; }
    auto display = gdk_x11_display_get_xdisplay(canvas_display);
    int root_x = 0, root_y = 0;
    gdk_window_get_origin(window, &root_x, &root_y);
    motion(display, -1, root_x + 50, root_y + 100, 0);
    button(display, 1, True, 0);
    button(display, 1, False, 0);
    XFlush(display);
    if (!wait(live, [&] { return live.Diagnostics().find("native-click") != std::string::npos; })) {
      dlclose(xtst);
      std::fprintf(stderr, "Live click: %s\n", live.Diagnostics().c_str()); return false;
    }
    motion(display, -1, root_x + 50, root_y + 30, 0);
    button(display, 1, True, 0);
    button(display, 1, False, 0);
    XSync(display, False);
    if (!wait(live, [&] { return live.Diagnostics().find("native-focus") != std::string::npos; })) {
      dlclose(xtst); return false;
    }
    key(display, XKeysymToKeycode(display, XK_a), True, 0);
    key(display, XKeysymToKeycode(display, XK_a), False, 0);
    XFlush(display);
    dlclose(xtst);
    if (!wait(live, [&] { return live.Diagnostics().find("native-key-a") != std::string::npos; }))
      { std::fprintf(stderr, "Live keyboard: %s\n", live.Diagnostics().c_str()); return false; }
    if (!state->png.empty() || state->capturing) return false;
    live.RequestScreenshot();
    if (!wait(live, [&] { return live.Ready(); }) || live.Screenshot().empty() ||
        live.Pixels().width != 800 || live.Pixels().height != 600 ||
        live.Pixels().rgba[0] != 0x11) return false;
    live.SetViewport(900, 700);
    live.Present(150, 120);
    live.EndFrame();
    if (!state->png.empty() || state->want_capture) return false;
    live.RequestScreenshot();
    if (!wait(live, [&] { return live.Ready(); }) || live.Pixels().width != 900 ||
        live.Pixels().height != 700 || live.Pixels().rgba[0] != 0x11) return false;
    live.BeginFrame();
    live.EndFrame();
    if (state->visible || gtk_widget_get_visible(live.impl_->window)) return false;
    // canvas_check must also work while Code/History or another screen is open.
    live.RequestScreenshot();
    if (!wait(live, [&] { return live.Ready(); }) || live.Screenshot().empty()) return false;
    live.BeginFrame();
    live.Present(200, 100);
    live.EndFrame();
    if (!state->visible) return false;
    live.EndFrame(true);
    if (state->visible) return false;
    live.BeginFrame();
    live.Present(200, 100);
    live.EndFrame();
    live.RequestScreenshot();
    live.Ready();
    live.Close();
    PumpCanvasBrowser();
    return true;
  }();
  SetCanvasBrowserHost(nullptr);
  gtk_widget_destroy(host);
  PumpCanvasBrowser();
  if (!direct_passed) {
    std::fprintf(stderr, "Canvas native embedding/input/on-demand capture failed\n");
    return 1;
  }
  std::fprintf(stdout, "Linux native child: animation, native mouse/keyboard, on-demand PNG, resize/state, hide/show and pending close passed.\n");
#endif
#if defined(GEM16_WITH_WEBVIEW2)
  // Exercise the production windowed controller, not just the diagnostic
  // composition controller. Normal rendering must never request a PNG.
  HWND host = CreateWindowExW(0, L"STATIC", L"Canvas test", WS_POPUP,
                              100, 100, 1600, 1000, nullptr, nullptr,
                              GetModuleHandle(nullptr), nullptr);
  if (!host) return 1;
  SetCanvasBrowserHost(host);
  const bool direct_passed = [&] {
    CanvasBrowser live;
    live.Load(d);
    live.BeginFrame();
    if (!live.Present(200, 100) || !wait(live, [&] { return live.Ready(); })) return false;
    auto state = live.impl_->state;
    if (!state->direct || state->composition || !state->png.empty() || state->capturing)
      return false;
    RECT rect{};
    GetWindowRect(state->window, &rect);
    MapWindowPoints(nullptr, host, reinterpret_cast<POINT*>(&rect), 2);
    if (rect.left != 200 || rect.top != 100) return false;
    live.RequestScreenshot();
    if (!wait(live, [&] { return live.Ready(); }) || live.Screenshot().empty()) return false;
    live.SetViewport(900, 700);
    live.Present(250, 120);
    if (!state->png.empty() || state->want_capture) return false;
    live.BeginFrame();
    live.EndFrame();
    return !state->visible && !IsWindowVisible(state->window);
  }();
  SetCanvasBrowserHost(nullptr);
  DestroyWindow(host);
  if (!direct_passed) {
    std::fprintf(stderr, "Canvas direct embedding/on-demand capture failed\n");
    return 1;
  }
#endif
  std::fprintf(stdout,
               "Canvas system WebView: HTML/JS, 1024x768 PNG (%zu bytes), "
               "mouse interaction, blocked network/file, JS/SVG diagnostics, "
               "800x600/1440x900 resize without reload, no outer overflow, inner scrolling, revision invalidation and close passed.\n",
               image.size());
  return 0;
}
}  // namespace gem16::studio
