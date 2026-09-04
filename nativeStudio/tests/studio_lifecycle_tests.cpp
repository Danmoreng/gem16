#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <thread>

#include "app.h"
#include "fonts.h"
#include "httplib.h"
#include "imgui.h"
#include "imgui_internal.h"
#include "math_renderer.h"
#include "settings.h"
#include "util/json.h"
void CaptureStudioScreenshot(const char* path);
namespace {
class Environment {
 public:
  Environment(const char* name, std::string value) : name_(name) {
    if (const char* old = std::getenv(name)) old_ = old;
    Set(value);
  }
  ~Environment() {
#ifdef _WIN32
    _putenv_s(name_, old_ ? old_->c_str() : "");
#else
    if (old_)
      setenv(name_, old_->c_str(), 1);
    else
      unsetenv(name_);
#endif
  }

 private:
  void Set(const std::string& value) {
#ifdef _WIN32
    _putenv_s(name_, value.c_str());
#else
    setenv(name_, value.c_str(), 1);
#endif
  }
  const char* name_;
  std::optional<std::string> old_;
};
void Require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
}  // namespace
namespace gem16::studio {
struct StudioAppTestAccess {
  static void Run(StudioApp& app, const std::string& id) {
    const auto frame = [&] {
      unsigned char* pixels;
      int w, h;
      ImGui::GetIO().Fonts->GetTexDataAsRGBA32(&pixels, &w, &h);
      ImGui::NewFrame();
      app.Render();
      ImGui::Render();
    };
    const auto settle = [&] {
      for (int i = 0; i < 500; ++i) {
        frame();
        if (!app.chat_listing_.valid() && !app.chat_load_.valid() &&
            app.CanNavigateChats())
          return;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
      }
      throw std::runtime_error("Studio persistence did not settle");
    };
    settle();
    Require(app.conversation_.id == id && app.messages_.size() == 2,
            "Studio restores latest conversation");
    app.conversation_.title = "Renamed and saved";
    ++app.chat_revision_;
    app.last_chat_save_ = {};
    settle();
    Require(app.chat_store_.Load(id).get().title == "Renamed and saved",
            "Studio rename checkpoint");
    app.screen_ = Screen::kChat;
    app.prompt_tokens_ = 3351;
    app.completion_tokens_ = 2235;
    app.usage_received_ = true;
    Require(app.ContextTokens() == 5586, "context includes generated output");
    for (int i = 0; i < 4; ++i) frame();
    CaptureStudioScreenshot("studio-chat-preview.bmp");
    app.prompt_tokens_ = std::numeric_limits<std::int64_t>::max();
    Require(app.ContextTokens() == std::numeric_limits<std::int64_t>::max(),
            "untrusted usage cannot overflow");
    app.screen_ = Screen::kModels;
    for (int i = 0; i < 4; ++i) frame();
    CaptureStudioScreenshot("studio-models-preview.bmp");
    Require(app.CanNavigateChats(), "idle navigation");
    app.NewConversation(true);
    Require(!app.usage_received_ && app.ContextTokens() == 0,
            "new conversation clears the previous context meter");
    const auto temporary_id = app.conversation_.id;
    ChatMessage secret;
    secret.role = "user";
    secret.content = "temporary text must not persist";
    app.messages_.push_back(secret);
    ++app.chat_revision_;
    frame();
    Require(!app.chat_save_.valid(), "temporary chat has no queued save");
    app.NewConversation(false);
    settle();
    for (const auto& saved : app.chat_store_.List().get())
      Require(saved.id != temporary_id, "temporary chat absent from SQLite");
    app.chat_load_ = app.chat_store_.Load(id);
    settle();
    Require(app.session_id_.empty() && app.messages_.size() == 2,
            "reopening restores text without GPU session");
    Require(!app.usage_received_, "restored conversation awaits its own usage");
  }

  static void RunCanvasTools(StudioApp& app) {
    httplib::Server server;
    std::atomic<int> requests{0};
    std::string failure;
    server.Post("/v1/chat/completions", [&](const httplib::Request& request,
                                            httplib::Response& response) {
      try {
        const int round = requests++;
        const auto loaded = app.chat_store_.Load(app.conversation_.id).get();
        if (round > 0)
          Require(loaded.canvases.size() == 1,
                  "canvas committed before continuation");
        std::string delta;
        if (round < 3) {
          ToolCall call;
          if (round == 0)
            call = {
                "create", "canvas_create",
                R"({"title":"Test page","type":"html","source":"<h1>First</h1>"})"};
          else if (round == 1)
            call = {
                "edit", "canvas_edit",
                "{\"id\":\"" + loaded.canvases[0].id +
                    R"(","revision":1,"old_text":"First","new_text":"Second"})"};
          else {
            Require(loaded.canvases[0].revisions.size() == 2,
                    "edited revision checkpoint");
            Require(request.body.find("tool_call_id") != std::string::npos,
                    "HTTP tool history");
            call = {"check", "canvas_check",
                    "{\"id\":\"" + loaded.canvases[0].id +
                        R"(","revision":2,"screenshot":true})"};
          }
          // Send arguments in two separate SSE chunks, as real token streaming
          // does.
          const auto split = call.arguments.size() / 2;
          delta =
              "data: "
              "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":" +
              gem16::json::Quote(call.id) +
              ",\"function\":{\"name\":" + gem16::json::Quote(call.name) +
              ",\"arguments\":" +
              gem16::json::Quote(call.arguments.substr(0, split)) +
              "}}]}}]}\n\n";
          delta +=
              "data: "
              "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,"
              "\"function\":{\"arguments\":" +
              gem16::json::Quote(call.arguments.substr(split)) + "}}]}}]}\n\n";
        } else {
          Require(round == 3, "bounded Canvas loop");
          Require(request.body.find("System WebView is unavailable") !=
                      std::string::npos,
                  "missing WebView returns an explicit tool error");
          delta =
              "data: {\"choices\":[{\"delta\":{\"content\":\"Canvas "
              "complete\"}}]}\n\n";
        }
        response.set_content(delta + "data: [DONE]\n\n", "text/event-stream");
      } catch (const std::exception& e) {
        failure = e.what();
        response.status = 500;
        response.set_content(failure, "text/plain");
      }
    });
    const int port = server.bind_to_any_port("127.0.0.1");
    Require(port > 0, "Canvas fixture bind");
    std::jthread listener([&] { server.listen_after_bind(); });
    try {
      app.NewConversation(false);
      app.pending_request_settings_ = app.settings_;
      app.pending_request_settings_.server.port = port;
      app.messages_ = {{"user", "create and edit a page"},
                       {"assistant", {}, {}, true}};
      app.canvas_cancelled_ = false;
      app.canvas_rounds_ = app.canvas_checks_ = 0;
      ++app.chat_revision_;
      app.pending_send_ = true;
      app.SaveChat();
      for (int i = 0; i < 1500; ++i) {
        app.DrainChatEvents();
        app.PollChatStore();
        app.PollCanvasTools();
        if (requests >= 4 && !app.api_.Busy() && !app.CanvasBusy() &&
            !app.pending_send_ && !app.chat_save_.valid())
          break;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
      }
      server.stop();
      listener.join();
      Require(failure.empty(), failure.c_str());
      Require(
          requests == 4 && app.messages_.back().content == "Canvas complete",
          "real SSE to edit to tool-result continuation");
      auto saved = app.chat_store_.Load(app.conversation_.id).get();
      Require(
          saved.canvases.size() == 1 &&
              saved.canvases[0].revisions.back().source == "<h1>Second</h1>",
          "agent Canvas survives SQLite reload");
    } catch (...) {
      server.stop();
      listener.join();
      throw;
    }
  }

  static void CheckCanvasRecovery(StudioApp& app) {
    app.temporary_chat_ = false;
    app.pending_request_settings_ = app.settings_;
    app.canvas_prompt_context_ = "canvas tools offered";
    app.canvas_cancelled_ = false;
    app.canvas_format_retries_ = 0;
    app.messages_ = {{"user", "draw"}, {"assistant", {}}};
    for (int attempt = 0; attempt < 2; ++attempt) {
      app.messages_.back().error = true;
      app.messages_.back().interrupted = false;
      app.messages_.back().error_message =
          "model response has an unterminated tool call";
      app.pending_send_ = false;
      Require(app.RecoverCanvasFormatError(),
              "malformed call schedules bounded repair");
      Require(app.pending_send_ && app.messages_.back().streaming &&
                  app.messages_.back().attempts.size() ==
                      static_cast<std::size_t>(attempt + 1),
              "repair preserves previous attempt");
      if (app.chat_save_.valid()) app.chat_save_.get();
    }
    app.pending_send_ = false;
    app.messages_.back().error = true;
    app.messages_.back().error_message =
        "model response has an unterminated tool call";
    Require(!app.RecoverCanvasFormatError(), "repair stops after two attempts");
    app.canvas_format_retries_ = 0;
    app.messages_.back().error_message = "Server unavailable";
    Require(!app.RecoverCanvasFormatError(),
            "network failure is not a format retry");
    app.messages_.back().error_message =
        "model response has an unterminated tool call";
    app.canvas_cancelled_ = true;
    Require(!app.RecoverCanvasFormatError(), "Stop prevents automatic repair");
    app.NewConversation(false);
  }

  static void CheckStartup(StudioApp& app, bool expected) {
    Require(app.pending_server_action_.has_value() == expected,
            "autostart respects onboarding and preference");
  }

  static void RunServerControls(StudioApp& app,
                                const std::filesystem::path& root) {
    const auto wait = [&](auto condition) {
      for (int i = 0; i < 600; ++i) {
        app.PollServerAction();
        if (!app.ServerActionPending() && condition()) return;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
      std::string logs;
      for (const auto& line : app.server_.Logs()) logs += "\n" + line;
      throw std::runtime_error(
          "server action did not complete (phase " +
          std::to_string(static_cast<int>(app.server_.Phase())) +
          "): " + app.server_.Error() + logs);
    };
    const auto start_count = [&] {
      std::ifstream input(root / "starts.txt");
      std::string line;
      int count = 0;
      while (std::getline(input, line)) ++count;
      return count;
    };
    wait([&] { return app.server_.Phase() == ServerPhase::kRunning; });
    Require(app.server_.OwnsProcess() && start_count() == 1,
            "autostart launches exactly one managed server");
    app.messages_={{"user","edit"},{"assistant",{}},{"tool","result"},{"assistant",{}}};
    app.messages_[1].tool_calls={{"retry-call","canvas_read","{}"}};
    app.messages_[2].tool_call_id="retry-call";
    app.messages_.back().error=true;app.messages_.back().error_message="failed after tools";
    app.RetryLastRequest();
    Require(app.pending_send_&&app.messages_.back().streaming&&app.messages_.back().attempts.size()==1,
            "manual Retry works after completed tools without rerunning them");
    app.pending_send_=false;if(app.chat_save_.valid())app.chat_save_.get();
    app.messages_.clear();
    app.RequestServerAction(StudioApp::ServerAction::kStart);
    Require(!app.ServerActionPending(),
            "running server cannot be started twice");
    app.session_id_ = "obsolete-session";
    app.messages_.push_back({"user", "test"});
    app.messages_.push_back({"assistant", "partial", {}, true});
    app.pending_send_ = true;
    app.RequestServerAction(StudioApp::ServerAction::kRestart);
    Require(!app.pending_send_ && !app.messages_.back().streaming &&
                app.messages_.back().error &&
                app.messages_.back().content == "partial",
            "restart cancels queued generation and preserves partial text");
    wait([&] { return app.server_.Phase() == ServerPhase::kRunning; });
    Require(start_count() == 2 && app.session_id_.empty(),
            "restart uses a new process and invalidates the GPU session");
    app.RequestServerAction(StudioApp::ServerAction::kStop);
    wait([&] { return !app.server_.OwnsProcess(); });
    Require(start_count() == 2,
            "manual stop does not trigger another autostart");
    app.RequestServerAction(StudioApp::ServerAction::kStart);
    wait([&] { return app.server_.Phase() == ServerPhase::kRunning; });
    Require(start_count() == 3, "manual start after stop");

    auto attached_settings = app.settings_;
    attached_settings.auto_start_server = true;
    StudioApp attached(attached_settings, 1.0f);
    for (int i = 0; i < 400; ++i) {
      attached.PollServerAction();
      if (!attached.ServerActionPending() &&
          attached.server_.Phase() == ServerPhase::kExternal)
        break;
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    Require(attached.server_.Phase() == ServerPhase::kExternal &&
                !attached.server_.OwnsProcess(),
            "autostart attaches to a compatible external process");
    attached.RequestServerAction(StudioApp::ServerAction::kStop);
    attached.RequestServerAction(StudioApp::ServerAction::kRestart);
    Require(
        !attached.ServerActionPending() && app.server_.OwnsProcess() &&
            start_count() == 3,
        "external stop and restart are rejected without affecting the owner");
    app.messages_.back() = {"assistant", {}, {}, true};
    app.api_.StreamChat(app.settings_.server, {}, {{"user", "test"}}, {});
    for (int i = 0; i < 400 && app.messages_.back().content.empty(); ++i) {
      app.DrainChatEvents();
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    Require(app.api_.Busy() && !app.messages_.back().content.empty(),
            "test server streams an in-flight answer");
    app.RequestServerAction(StudioApp::ServerAction::kStop);
    wait([&] { return !app.server_.OwnsProcess(); });
    Require(!app.api_.Busy() && !app.messages_.back().streaming &&
                app.messages_.back().error &&
                !app.messages_.back().content.empty(),
            "stop cancels the transport before terminating the server");
    app.SaveChat();
    app.chat_save_.get();
    const auto saved = app.chat_store_.Load(app.conversation_.id).get();
    Require(
        saved.messages.back().error && !saved.messages.back().content.empty(),
        "partial answer survives server stop in SQLite");
  }
};
}  // namespace gem16::studio
bool TestStudioLifecycle(const std::filesystem::path& executable) {
  using namespace gem16::studio;
  const auto root = std::filesystem::temp_directory_path() /
                    ("gem16-ui-lifecycle-" + NewChatId());
  std::filesystem::create_directories(root);
  Environment data("GEM16_STUDIO_DATA_ROOT", (root / "data").string()),
      cache("HF_HUB_CACHE", (root / "hub").string());
#ifdef _WIN32
  Environment config("APPDATA", (root / "config").string());
#else
  Environment config("XDG_CONFIG_HOME", (root / "config").string());
#endif
  bool ok = true;
  ImGui::CreateContext();
  auto& io = ImGui::GetIO();
  io.IniFilename = nullptr;
  io.DisplaySize = {900, 1050};
  io.DeltaTime = 1.0f / 60.0f;
  InitializeStudioFonts();
  try {
    bool rejected_reinitialization = false;
    try {
      InitializeMathFonts();
    } catch (const std::logic_error&) {
      rejected_reinitialization = true;
    }
    Require(rejected_reinitialization,
            "reject unsafe MicroTeX reinitialization");
    auto settings = DefaultSettings();
    Require(settings.auto_start_server,
            "autostart defaults on for selected installations");
    settings.auto_start_server = false;
    settings.onboarding_complete = true;
    settings.server.port = 1;
    Conversation chat;
    chat.id = NewChatId();
    chat.title = "Local conversations";
    chat.identity = ModelIdentity(settings.server);
    ChatMessage user;
    user.role = "user";
    user.content = "How are my conversations stored?";
    chat.messages.push_back(user);
    ChatMessage answer;
    answer.role = "assistant";
    answer.content =
        "Your chats are saved locally in **SQLite**.\n\n- Search messages and "
        "attached documents.\n- Export JSON and Markdown.\n- Restore a "
        "complete backup.\n\n```sql\nSELECT title FROM conversations;\n```";
    answer.generation = GenerationIdentity(settings);
    chat.messages.push_back(answer);
    {
      ChatStore store;
      store.Save(chat).get();
    }
    {
      StudioApp app(settings, 1.0f);
      StudioAppTestAccess::CheckStartup(app, false);
      StudioAppTestAccess::Run(app, chat.id);
      StudioAppTestAccess::RunCanvasTools(app);
      StudioAppTestAccess::CheckCanvasRecovery(app);
    }
    settings.auto_start_server = true;
    settings.onboarding_complete = false;
    {
      StudioApp app(settings, 1.0f);
      StudioAppTestAccess::CheckStartup(app, false);
    }
    settings.onboarding_complete = true;
    settings.server.executable = executable.string();
    settings.server.model_directory = root.string();
    settings.server.model_name = "studio-lifecycle-test";
    settings.server.mtp_draft_tokens = 0;
    settings.server.assistant_directory.clear();
    httplib::Server reservation;
    reservation.new_task_queue = [] { return new httplib::ThreadPool(1); };
    settings.server.port = reservation.bind_to_any_port("127.0.0.1");
    Require(settings.server.port > 0, "reserve test port");
    std::jthread reservation_listener([&] { reservation.listen_after_bind(); });
    reservation.wait_until_ready();
    reservation.stop();
    reservation_listener.join();
    Require(SaveSettings(settings), "save last server configuration");
    {
      StudioApp app(LoadSettings(), 1.0f);
      StudioAppTestAccess::CheckStartup(app, true);
      StudioAppTestAccess::RunServerControls(app, root);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "Studio lifecycle: %s\n", e.what());
    ok = false;
  }
  ImGui::DestroyContext();
  std::filesystem::remove_all(root);
  return ok;
}

#include "model_cache.h"
#include "model_selection.h"
namespace {
template <class F>
void MustFail(F fn) {
  bool failed = false;
  try {
    fn();
  } catch (const std::exception&) {
    failed = true;
  }
  Require(failed, "Expected rejection");
}
}  // namespace
bool TestModelLifecycle() {
  using namespace gem16::studio;
  const auto root = std::filesystem::temp_directory_path() /
                    ("gem16-model-lifecycle-" + NewChatId());
  std::filesystem::create_directories(root);
  Environment cache("HF_HUB_CACHE", (root / "hub").string());
#ifdef _WIN32
  Environment config("APPDATA", (root / "config").string());
#else
  Environment config("XDG_CONFIG_HOME", (root / "config").string());
#endif
  try {
    const ModelCatalogFile file{
        "weights.bin",
        3,
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "blob",
        "test/model",
        "revision",
        "weights.bin"};
    const ModelComponentCatalog component{
        "test", "Test", "test/model", "revision", {&file, 1}, false, ""};
    const ModelProfileComponent target{ModelComponentKind::kTarget, &component,
                                       true};
    const ModelProfileCatalog profile{
        ModelProfile::kGemma4Unified12B, "test", "test", {&target, 1}};
    const std::span<const ModelProfileCatalog> catalog{&profile, 1};
    const auto hub = HuggingFaceHubRoot(),
               blob = RepositoryDirectory(file.source_repository, hub) /
                      "blobs" / file.blob_id;
    const auto partial = std::filesystem::path(blob.string() + ".incomplete"),
               view = ComponentDirectory(component, hub) / file.path;
    std::filesystem::create_directories(blob.parent_path());
    std::filesystem::create_directories(view.parent_path());
    {
      ModelManager manager(catalog);
      const auto status = [&] {
        return manager.State().For(profile.profile).component_status[0];
      };
      Require(status() == ComponentInstallStatus::kMissing,
              "missing component");
      std::ofstream(partial) << "a";
      manager.Refresh();
      Require(status() == ComponentInstallStatus::kPartial,
              "partial component");
      Require(manager.State().For(profile.profile).required_download_bytes == 2,
              "resume disk requirement");
      std::filesystem::remove(partial);
      std::ofstream(blob) << "abc";
      std::filesystem::create_hard_link(blob, view);
      manager.Refresh();
      Require(status() == ComponentInstallStatus::kUnverified,
              "present is not verified");
      const auto wait = [&] {
        for (int i = 0; i < 1000 && manager.State().Busy(); ++i)
          std::this_thread::sleep_for(std::chrono::milliseconds(2));
        Require(!manager.State().Busy(), "model worker timeout");
      };
      manager.VerifyInstalled();
      wait();
      Require(status() == ComponentInstallStatus::kVerified,
              "verified component");
      std::ofstream(blob) << "xbc";
      manager.VerifyInstalled();
      wait();
      Require(status() == ComponentInstallStatus::kDamaged,
              "damaged component");
      manager.Refresh();
      Require(status() == ComponentInstallStatus::kDamaged,
              "damage survives refresh");
      std::ofstream(blob) << "abc";
      manager.DownloadProfile(profile.profile);
      wait();
      Require(manager.State().error.empty() &&
                  status() == ComponentInstallStatus::kVerified,
              "repair reuses valid blob without network");
      manager.RemoveProfile(profile.profile);
      Require(!std::filesystem::exists(view) && std::filesystem::exists(blob),
              "profile removal preserves blob");
    }
    auto plan = InspectModelCache(catalog);
    Require(plan.reclaimable_bytes == 3, "unused blob preview");
    const auto shared = root / "outside-hardlink";
    std::filesystem::create_hard_link(blob, shared);
    Require(CleanModelCache(plan, catalog) == 0, "preserve external hardlink");
    std::filesystem::remove(shared);
    const auto linked = hub / "other-repository-link";
    std::error_code symlink_error;
    std::filesystem::create_symlink(blob, linked, symlink_error);
    if (!symlink_error) {
      Require(CleanModelCache(plan, catalog) == 0,
              "preserve cross-repository symlink");
      std::filesystem::remove(linked);
    }
    {
      HubBlobLock lock(HubBlobLockPath(file));
      Require(lock.Locked(), "acquire cache lock");
      Require(CleanModelCache(plan, catalog) == 0, "keep locked cache blob");
    }
    std::ofstream(blob) << "changed";
    Require(CleanModelCache(plan, catalog) == 0,
            "stale cleanup preview cannot delete changed file");
    std::ofstream(blob) << "abc";
    plan = InspectModelCache(catalog);
    const auto unknown = blob.parent_path() / "unknown-client-blob";
    std::ofstream(unknown) << "keep";
    Require(CleanModelCache(plan, catalog) == 3 &&
                !std::filesystem::exists(blob) &&
                std::filesystem::exists(unknown),
            "delete only reviewed unreferenced known blob");
    auto settings = DefaultSettings();
    auto record = GenerationIdentity(settings);
    auto restored = SavedServerSelection(record, settings.server);
    Require(restored.model_directory == settings.server.model_directory &&
                restored.mtp_draft_tokens == settings.server.mtp_draft_tokens,
            "exact model selection roundtrip");
    const auto revision = std::string(CatalogForProfile(settings.server.profile)
                                          .components.front()
                                          .catalog->revision);
    auto changed = record;
    const auto at = changed.find(revision);
    Require(at != std::string::npos, "identity carries revision");
    changed.replace(at, revision.size(), revision.size(), '0');
    MustFail([&] { SavedServerSelection(changed, settings.server); });
    settings.previous_model_selection = record;
    Require(SaveSettings(settings), "atomic selection settings save");
    Require(LoadSettings().previous_model_selection == record,
            "rollback selection persists");
    settings.auto_start_server = false;
    Require(SaveSettings(settings) && !LoadSettings().auto_start_server,
            "autostart opt-out persists");
    std::filesystem::remove_all(root);
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "Model lifecycle: %s\n", e.what());
    std::filesystem::remove_all(root);
    return false;
  }
}

// The normal server command launches this lightweight child in lifecycle tests.
// It exercises real process ownership and HTTP transport without loading a
// model.
int RunStudioTestServer(int argc, char** argv) {
  std::string directory, name;
  int port = 0;
  for (int i = 1; i + 1 < argc; ++i) {
    const std::string_view flag = argv[i];
    if (flag == "--model")
      directory = argv[++i];
    else if (flag == "--model-name")
      name = argv[++i];
    else if (flag == "--port")
      port = std::stoi(argv[++i]);
  }
  if (name != "studio-lifecycle-test" || directory.empty() || port <= 0)
    return 2;
  std::ofstream(std::filesystem::path(directory) / "starts.txt", std::ios::app)
      << "start\n";
  httplib::Server server;
  server.new_task_queue = [] { return new httplib::ThreadPool(2); };
  server.Get("/health", [](const auto&, auto& response) {
    response.set_content(
        R"({"status":"ok","profile_id":"gemma4-12b-unified","decode_mode":"ordinary","text_only":false,"capabilities":{"vision":true},"mtp_draft_tokens":0,"max_context_tokens":8192})",
        "application/json");
  });
  server.Get("/metrics", [](const auto&, auto& response) {
    response.set_content("", "text/plain");
  });
  server.Post("/v1/chat/completions", [](const auto&, auto& response) {
    response.set_chunked_content_provider(
        "text/event-stream", [](size_t, httplib::DataSink& sink) {
          std::this_thread::sleep_for(std::chrono::milliseconds(20));
          const std::string chunk =
              "data: {\"choices\":[{\"delta\":{\"content\":\"partial "
              "\"}}]}\n\n";
          return sink.write(chunk.data(), chunk.size());
        });
  });
  return server.listen("127.0.0.1", port) ? 0 : 1;
}
