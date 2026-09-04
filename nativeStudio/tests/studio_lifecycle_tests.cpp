#include <cstdio>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <thread>

#include "app.h"
#include "fonts.h"
#include "imgui.h"
#include "imgui_internal.h"
#include "math_renderer.h"
#include "settings.h"
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
    for (int i = 0; i < 4; ++i) frame();
    CaptureStudioScreenshot("studio-chat-preview.bmp");
    app.screen_ = Screen::kModels;
    for (int i = 0; i < 4; ++i) frame();
    CaptureStudioScreenshot("studio-models-preview.bmp");
    Require(app.CanNavigateChats(), "idle navigation");
    app.NewConversation(true);
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
  }
};
}  // namespace gem16::studio
bool TestStudioLifecycle() {
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
      StudioAppTestAccess::Run(app, chat.id);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "Studio lifecycle: %s\n", e.what());
    ok = false;
  }
  ImGui::DestroyContext();
  std::filesystem::remove_all(root);
  return ok;
}

#include <fstream>

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
    std::filesystem::remove_all(root);
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "Model lifecycle: %s\n", e.what());
    std::filesystem::remove_all(root);
    return false;
  }
}
