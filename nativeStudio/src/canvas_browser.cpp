#include "canvas_browser.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <thread>

#include "canvas_page.h"
#include "util/json.h"
#if defined(GEM16_WITH_WEBKIT)
#include <webkit2/webkit2.h>
#elif defined(GEM16_WITH_WEBVIEW2)
#include <WebView2.h>
#include <dcomp.h>
#include <windows.h>
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
struct CanvasBrowser::Impl {
  std::shared_ptr<Preview> state = std::make_shared<Preview>();
  GtkWidget* window = nullptr;
  WebKitWebView* view = nullptr;
  GCancellable* cancel = nullptr;
  void Capture() {
    auto s = state;
    if (!view || !s->loaded || s->closed ||
        Clock::now() - s->loaded_at < std::chrono::milliseconds(500))
      return;
    if (!s->observing) {
      s->observing = true;
      webkit_web_view_evaluate_javascript(
          view, kCanvasObservationScript, -1, nullptr, nullptr, cancel,
          [](GObject* object, GAsyncResult* result, gpointer data) {
            std::unique_ptr<std::shared_ptr<Preview>> hold(
                static_cast<std::shared_ptr<Preview>*>(data));
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
          new std::shared_ptr<Preview>(s));
    }
    if (s->capturing ||
        Clock::now() - s->captured_at < std::chrono::milliseconds(100))
      return;
    s->capturing = true;
    s->captured_at = Clock::now();
    webkit_web_view_get_snapshot(
        view, WEBKIT_SNAPSHOT_REGION_VISIBLE, WEBKIT_SNAPSHOT_OPTIONS_NONE,
        cancel,
        [](GObject* object, GAsyncResult* result, gpointer data) {
          std::unique_ptr<std::shared_ptr<Preview>> hold(
              static_cast<std::shared_ptr<Preview>*>(data));
          auto s = *hold;
          GError* error = nullptr;
          auto surface = webkit_web_view_get_snapshot_finish(
              WEBKIT_WEB_VIEW(object), result, &error);
          if (surface && !s->closed) {
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
              s->pixels = DecodePreviewImage(png.data(), png.size());
              s->png = std::move(png);
            }
          }
          if (surface) cairo_surface_destroy(surface);
          if (error) {
            if (!s->closed) s->diagnostics = error->message;
            g_error_free(error);
          }
          s->capturing = false;
        },
        new std::shared_ptr<Preview>(s));
  }
};
int InitializeCanvasBrowser(int, char**) {
  initialized = gtk_init_check(nullptr, nullptr);
  return -1;
}
void PumpCanvasBrowser() {
  if (initialized)
    for (int i = 0; i < 32 && g_main_context_pending(nullptr); ++i)
      g_main_context_iteration(nullptr, false);
}
void ShutdownCanvasBrowser() {
  PumpCanvasBrowser();
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
  impl_->state = std::make_shared<Preview>();
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
  // A realized GTK window is required by modern WebKitGTK. Its contents are
  // composited into ImGui via the official snapshot API (not
  // GtkOffscreenWindow).
  i.window = gtk_window_new(GTK_WINDOW_POPUP);
  gtk_window_set_accept_focus(GTK_WINDOW(i.window), false);
  gtk_widget_set_opacity(i.window, 0);
  gtk_window_move(GTK_WINDOW(i.window), -12000, -12000);
  gtk_widget_set_size_request(GTK_WIDGET(i.view), 1024, 768);
  gtk_container_add(GTK_CONTAINER(i.window), GTK_WIDGET(i.view));
  auto raw = s.get();
  g_signal_connect(
      i.view, "load-changed",
      G_CALLBACK(+[](WebKitWebView*, WebKitLoadEvent event, gpointer p) {
        auto s = static_cast<Preview*>(p);
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
            auto s = static_cast<Preview*>(p);
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
            auto s = static_cast<Preview*>(p);
            s->ready = s->loaded = false;
            s->diagnostics = "System WebView renderer terminated.";
          }),
      raw);
  gtk_widget_show_all(i.window);
  auto empty_region = cairo_region_create();
  gdk_window_input_shape_combine_region(gtk_widget_get_window(i.window),
                                        empty_region, 0, 0);
  cairo_region_destroy(empty_region);
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
    event->scroll.direction = wheel > 0 ? GDK_SCROLL_UP : GDK_SCROLL_DOWN;
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
bool CanvasBrowser::Ready() const {
  impl_->Capture();
  auto s = impl_->state;
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
        return browser.Ready() &&
               browser.Diagnostics().find("Script error") != std::string::npos;
      })) {
    std::fprintf(stderr, "Canvas render/diagnostics failed: %s\n",
                 browser.Diagnostics().c_str());
    return 1;
  }
  auto image = browser.Screenshot();
  auto pixels = browser.Pixels();
  if (pixels.width != 1024 || pixels.height != 768 || pixels.rgba.empty() ||
      pixels.rgba[0] != 0x12) {
    std::fprintf(stderr, "Canvas screenshot dimensions/color failed\n");
    return 1;
  }
  if (browser.Diagnostics().find("Network blocked") == std::string::npos ||
      browser.Diagnostics().find("File blocked") == std::string::npos) {
    std::fprintf(stderr, "Canvas isolation diagnostics missing\n");
    return 1;
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
  // Closing while a snapshot/observation is pending must not retain callbacks
  // into the destroyed browser. The same lifecycle is used when switching
  // chats.
  browser.Close();
  PumpCanvasBrowser();
  std::fprintf(stdout,
               "Canvas system WebView: HTML/JS, 1024x768 PNG (%zu bytes), "
               "mouse interaction, blocked network/file, JS/SVG diagnostics, "
               "revision invalidation and close passed.\n",
               image.size());
  return 0;
}
}  // namespace gem16::studio
