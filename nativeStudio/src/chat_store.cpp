#include "chat_store.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <random>
#include <sstream>

#include "compiler/sha256.h"
#include "model_catalog.h"
#include "settings.h"
#include "sqlite3.h"
#include "util/json.h"
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace gem16::studio {
namespace {
using J = json::Value;
std::int64_t Now() {
  return std::chrono::duration_cast<std::chrono::seconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}
std::string PathText(const std::filesystem::path& p) {
  const auto s = p.u8string();
  return {s.begin(), s.end()};
}
void Check(sqlite3* db, int code) {
  if (code != SQLITE_OK && code != SQLITE_DONE && code != SQLITE_ROW)
    throw std::runtime_error(sqlite3_errmsg(db));
}
void Exec(sqlite3* db, const char* sql) {
  Check(db, sqlite3_exec(db, sql, nullptr, nullptr, nullptr));
}
class Statement {
 public:
  Statement(sqlite3* db, const char* sql) : db_(db) {
    Check(db, sqlite3_prepare_v2(db, sql, -1, &s_, nullptr));
  }
  ~Statement() { sqlite3_finalize(s_); }
  void Text(int n, const std::string& v) {
    Check(db_, sqlite3_bind_text(s_, n, v.data(), static_cast<int>(v.size()),
                                 SQLITE_TRANSIENT));
  }
  void Int(int n, std::int64_t v) { Check(db_, sqlite3_bind_int64(s_, n, v)); }
  bool Row() {
    int r = sqlite3_step(s_);
    Check(db_, r);
    return r == SQLITE_ROW;
  }
  std::string Text(int n) {
    const auto* s = sqlite3_column_text(s_, n);
    return s ? std::string(reinterpret_cast<const char*>(s),
                           sqlite3_column_bytes(s_, n))
             : std::string{};
  }
  std::int64_t Int(int n) { return sqlite3_column_int64(s_, n); }

 private:
  sqlite3* db_;
  sqlite3_stmt* s_ = nullptr;
};
std::string Str(const J& j, const char* key) {
  const J* v = j.find(key);
  if (!v || !v->is_string())
    throw std::runtime_error("Invalid saved chat string.");
  return v->as_string();
}
std::int64_t Num(const J& j, const char* key) {
  const J* v = j.find(key);
  if (!v || !v->is_integer())
    throw std::runtime_error("Invalid saved chat number.");
  return v->as_integer();
}
bool Flag(const J& j, const char* key) {
  const J* v = j.find(key);
  if (!v || !v->is_bool()) throw std::runtime_error("Invalid saved chat flag.");
  return v->as_bool();
}
const J::Array& Array(const J& j, const char* key) {
  const J* v = j.find(key);
  if (!v || !v->is_array())
    throw std::runtime_error("Invalid saved chat array.");
  return v->as_array();
}
bool SafeHash(const std::string& s) {
  return s.size() == 64 && std::all_of(s.begin(), s.end(), [](char c) {
           return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
         });
}
void WriteBlob(const std::filesystem::path& root, const std::string& hash,
               const std::vector<std::uint8_t>& bytes) {
  if (!SafeHash(hash) || bytes.size() > 64U * 1024U * 1024U)
    throw std::runtime_error("Invalid chat attachment.");
  const auto path = root / hash;
  if (std::filesystem::is_symlink(std::filesystem::symlink_status(path)))
    throw std::runtime_error("Attachment symlink rejected.");
  if (std::filesystem::exists(path)) {
    if (std::filesystem::file_size(path) != bytes.size())
      throw std::runtime_error("Stored attachment is damaged.");
    return;
  }
  const auto tmp = root / (hash + "." + NewChatId() + ".tmp");
  try {
    {
      std::ofstream out(tmp, std::ios::binary);
      out.write(reinterpret_cast<const char*>(bytes.data()),
                static_cast<std::streamsize>(bytes.size()));
      out.close();
      if (!out)
        throw std::runtime_error(
            "Could not save attachment (check free space).");
    }
#ifdef _WIN32
    HANDLE file = CreateFileW(tmp.c_str(), GENERIC_WRITE, 0, nullptr,
                              OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE)
      throw std::runtime_error("Could not flush attachment.");
    const bool flushed = FlushFileBuffers(file) != 0;
    CloseHandle(file);
    if (!flushed ||
        !MoveFileExW(tmp.c_str(), path.c_str(), MOVEFILE_WRITE_THROUGH))
      throw std::runtime_error("Could not commit attachment.");
#else
    const int file = open(tmp.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (file < 0) throw std::runtime_error("Could not flush attachment.");
    const int flushed = fsync(file);
    close(file);
    if (flushed != 0) throw std::runtime_error("Could not flush attachment.");
    std::filesystem::rename(tmp, path);
    const int directory =
        open(root.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (directory < 0)
      throw std::runtime_error("Could not flush attachment directory.");
    const int synced = fsync(directory);
    close(directory);
    if (synced != 0)
      throw std::runtime_error("Could not flush attachment directory.");
#endif
  } catch (...) {
    std::error_code ec;
    std::filesystem::remove(tmp, ec);
    throw;
  }
}
J MessageJson(const ChatMessage& m, const std::filesystem::path& root) {
  J::Array attachments, attempts;
  for (const auto& a : m.attachments) {
    const auto hash = a.missing
                          ? a.storage_hash
                          : compiler::Sha256Hex(a.bytes.data(), a.bytes.size());
    if (!SafeHash(hash))
      throw std::runtime_error("Invalid saved attachment hash.");
    if (!a.missing) WriteBlob(root, hash, a.bytes);
    attachments.emplace_back(J::Object{
        {"hash", J(hash)},
        {"kind", J(static_cast<std::int64_t>(a.kind))},
        {"name", J(a.file_name)},
        {"mime", J(a.mime_type)},
        {"format", J(a.format)},
        {"document", J(a.document_text)},
        {"size", J(static_cast<std::int64_t>(a.missing ? a.byte_size
                                                       : a.bytes.size()))},
        {"width", J(static_cast<std::int64_t>(a.image_width))},
        {"height", J(static_cast<std::int64_t>(a.image_height))}});
  }
  for (const auto& a : m.attempts)
    attempts.emplace_back(J::Object{{"content", J(a.content)},
                                    {"reasoning", J(a.reasoning)},
                                    {"error", J(a.error)},
                                    {"error_message", J(a.error_message)},
                                    {"generation", J(a.generation)},
                                    {"created", J(a.created)}});
  return J(J::Object{{"attachments", J(std::move(attachments))},
                     {"attempts", J(std::move(attempts))},
                     {"generation", J(m.generation)},
                     {"created", J(m.created)},
                     {"error_message", J(m.error_message)}});
}
}  // namespace

std::filesystem::path StudioDataDirectory() {
  if (const char* p = std::getenv("GEM16_STUDIO_DATA_ROOT"); p && *p)
    return std::filesystem::path(reinterpret_cast<const char8_t*>(p));
#ifdef _WIN32
  if (const char* p = std::getenv("LOCALAPPDATA"); p && *p)
    return std::filesystem::path(reinterpret_cast<const char8_t*>(p)) / "gem16";
#else
  if (const char* p = std::getenv("XDG_DATA_HOME"); p && *p)
    return std::filesystem::path(reinterpret_cast<const char8_t*>(p)) / "gem16";
  if (const char* p = std::getenv("HOME"); p && *p)
    return std::filesystem::path(reinterpret_cast<const char8_t*>(p)) /
           ".local/share/gem16";
#endif
  return SettingsPath().parent_path() / "data";
}
std::string NewChatId() {
  std::random_device random;
  std::uint32_t data[8];
  for (auto& v : data) v = random();
  return compiler::Sha256Hex(data, sizeof(data));
}
std::string ModelIdentity(const ServerConfig& s) {
  J::Array components;
  for (const auto& c : CatalogForProfile(s.profile).components) {
    if (c.kind == ModelComponentKind::kAssistant &&
        s.assistant_directory.empty())
      continue;
    const auto& path = c.kind == ModelComponentKind::kTarget ? s.model_directory
                       : c.kind == ModelComponentKind::kVision
                           ? s.vision_directory
                           : s.assistant_directory;
    if (path.empty()) continue;
    const bool pinned =
        std::filesystem::path(path).lexically_normal() ==
        ComponentDirectory(*c.catalog, HuggingFaceHubRoot()).lexically_normal();
    components.emplace_back(
        J::Object{{"component", J(std::string(c.catalog->id))},
                  {"repository",
                   J(pinned ? std::string(c.catalog->repository) : "custom")},
                  {"revision", J(pinned ? std::string(c.catalog->revision)
                                        : "unverified custom path")}});
  }
  return json::Stringify(
      J(J::Object{{"profile", J(std::string(ProfileWireName(s.profile)))},
                  {"model_path", J(s.model_directory)},
                  {"vision_path", J(s.vision_directory)},
                  {"assistant_path", J(s.assistant_directory)},
                  {"components", J(std::move(components))}}));
}
std::string GenerationIdentity(const StudioSettings& s) {
  return json::Stringify(J(J::Object{
      {"model", J(ModelIdentity(s.server))},
      {"model_name", J(s.server.model_name)},
      {"context", J(s.server.max_context_tokens)},
      {"greedy", J(s.server.greedy)},
      {"draft_tokens", J(static_cast<std::int64_t>(s.server.mtp_draft_tokens))},
      {"vision_budget",
       J(static_cast<std::int64_t>(s.server.vision_soft_token_budget))},
      {"adaptive", J(s.server.mtp_adaptive)},
      {"system_prompt", J(s.generation.system_prompt)},
      {"reasoning_effort", J(s.generation.reasoning_effort)},
      {"max_output_tokens", J(s.generation.max_output_tokens)}}));
}
GenerationConfig SavedGeneration(const std::string& metadata) {
  auto parsed = json::Parse(metadata);
  if (!parsed.ok())
    throw std::runtime_error("Invalid saved generation settings.");
  GenerationConfig g;
  g.system_prompt = Str(parsed.value(), "system_prompt");
  g.reasoning_effort = Str(parsed.value(), "reasoning_effort");
  g.max_output_tokens = Num(parsed.value(), "max_output_tokens");
  if (g.system_prompt.size() > 16383 || g.max_output_tokens < 1 ||
      g.max_output_tokens > 262144 ||
      (g.reasoning_effort != "none" && g.reasoning_effort != "low" &&
       g.reasoning_effort != "medium" && g.reasoning_effort != "high"))
    throw std::runtime_error("Unsupported saved generation settings.");
  return g;
}
void PreserveAttempt(ChatMessage& m) {
  m.attempts.push_back({m.content, m.reasoning, m.error_message, m.generation,
                        m.created, m.error});
  m.content.clear();
  m.reasoning.clear();
  m.error_message.clear();
  m.error = false;
  m.streaming = true;
  m.created = Now();
}

ChatStore::ChatStore(std::filesystem::path root) : root_(std::move(root)) {
  worker_ = std::thread([this] {
    try {
      Open();
    } catch (const std::exception& e) {
      open_error_ = e.what();
    }
    for (;;) {
      std::function<void()> fn;
      {
        std::unique_lock lock(mutex_);
        wake_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
        if (queue_.empty() && stopping_) break;
        fn = std::move(queue_.front());
        queue_.pop_front();
      }
      fn();
    }
    if (db_) sqlite3_close(db_);
#ifdef _WIN32
    if (lock_handle_ != -1) CloseHandle(reinterpret_cast<HANDLE>(lock_handle_));
#else
    if (lock_handle_ != -1) close(static_cast<int>(lock_handle_));
#endif
  });
}
ChatStore::~ChatStore() {
  {
    std::lock_guard lock(mutex_);
    stopping_ = true;
  }
  wake_.notify_one();
  if (worker_.joinable()) worker_.join();
}
void ChatStore::Open() {
  std::filesystem::create_directories(root_);
  if (std::filesystem::is_symlink(std::filesystem::symlink_status(root_)))
    throw std::runtime_error("Chat data directory must not be a symlink.");
#ifndef _WIN32
  std::filesystem::permissions(root_, std::filesystem::perms::owner_all);
#endif
  for (const char* file : {"studio.db", "studio.db-wal", "studio.db-shm",
                           "studio.lock", "attachments"})
    if (std::filesystem::is_symlink(
            std::filesystem::symlink_status(root_ / file)))
      throw std::runtime_error("Chat storage symlink rejected.");
  std::filesystem::create_directories(root_ / "attachments");
#ifdef _WIN32
  lock_handle_ = reinterpret_cast<std::intptr_t>(
      CreateFileW((root_ / "studio.lock").c_str(), GENERIC_READ | GENERIC_WRITE,
                  0, nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr));
  if (lock_handle_ == -1)
    throw std::runtime_error(
        "Chat storage is already open in another Studio instance or "
        "inaccessible.");
#else
  lock_handle_ = open((root_ / "studio.lock").c_str(),
                      O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600);
  if (lock_handle_ == -1 ||
      flock(static_cast<int>(lock_handle_), LOCK_EX | LOCK_NB) != 0)
    throw std::runtime_error(
        "Chat storage is already open in another Studio instance or "
        "inaccessible.");
#endif
  const int opened = sqlite3_open_v2(
      PathText(root_ / "studio.db").c_str(), &db_,
      SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_NOMUTEX,
      nullptr);
  Check(db_, opened);
  sqlite3_busy_timeout(db_, 3000);
  sqlite3_limit(db_, SQLITE_LIMIT_LENGTH, 64 * 1024 * 1024);
  Exec(db_,
       "PRAGMA foreign_keys=ON; PRAGMA journal_mode=WAL; PRAGMA "
       "synchronous=FULL; PRAGMA trusted_schema=OFF;");
  Statement version(db_, "PRAGMA user_version");
  version.Row();
  if (version.Int(0) > 1)
    throw std::runtime_error(
        "This chat database needs a newer Studio version.");
  Exec(db_,
       "BEGIN; CREATE TABLE IF NOT EXISTS conversations(id TEXT PRIMARY "
       "KEY,title TEXT NOT NULL,identity TEXT NOT NULL,profile INTEGER NOT "
       "NULL,updated INTEGER NOT NULL,pinned INTEGER NOT NULL,archived INTEGER "
       "NOT NULL);"
       "CREATE TABLE IF NOT EXISTS messages(chat TEXT NOT NULL REFERENCES "
       "conversations(id) ON DELETE CASCADE,pos INTEGER NOT NULL,role TEXT NOT "
       "NULL,content TEXT NOT NULL,reasoning TEXT NOT NULL,state INTEGER NOT "
       "NULL,details TEXT NOT NULL,PRIMARY KEY(chat,pos));"
       "PRAGMA user_version=1; COMMIT;");
  Exec(db_, "UPDATE messages SET state=2 WHERE state=1;");
}
void ChatStore::SaveNow(const Conversation& c) {
  if (!open_error_.empty()) throw std::runtime_error(open_error_);
  if (!SafeHash(c.id) || c.messages.size() > 10000 || c.title.size() > 512)
    throw std::runtime_error("Chat exceeds storage limits.");
  std::vector<std::string> details;
  std::size_t total = 0, media_bytes = 0;
  for (const auto& m : c.messages)
    for (const auto& a : m.attachments) {
      media_bytes += a.missing ? a.byte_size : a.bytes.size();
      if (media_bytes > 256U * 1024U * 1024U)
        throw std::runtime_error("Chat attachments exceed 256 MiB.");
    }
  for (const auto& m : c.messages) {
    auto d = json::Stringify(MessageJson(m, root_ / "attachments"));
    total += d.size() + m.content.size() + m.reasoning.size();
    if (total > 32U * 1024U * 1024U)
      throw std::runtime_error(
          "Chat text exceeds 32 MiB; export it and start a new chat.");
    details.push_back(std::move(d));
  }
  Exec(db_, "BEGIN IMMEDIATE");
  try {
    Statement q(db_,
                "INSERT INTO conversations VALUES(?,?,?,?,?,?,?) ON "
                "CONFLICT(id) DO UPDATE SET "
                "title=excluded.title,identity=excluded.identity,profile="
                "excluded.profile,updated=excluded.updated,pinned=excluded."
                "pinned,archived=excluded.archived");
    q.Text(1, c.id);
    q.Text(2, c.title);
    q.Text(3, c.identity);
    q.Int(4, static_cast<int>(c.profile));
    q.Int(5, Now());
    q.Int(6, c.pinned);
    q.Int(7, c.archived);
    q.Row();
    Statement erase(db_, "DELETE FROM messages WHERE chat=? AND pos>=?");
    erase.Text(1, c.id);
    erase.Int(2, c.messages.size());
    erase.Row();
    for (std::size_t i = 0; i < c.messages.size(); ++i) {
      const auto& m = c.messages[i];
      Statement insert(
          db_,
          "INSERT INTO messages VALUES(?,?,?,?,?,?,?) ON CONFLICT(chat,pos) DO "
          "UPDATE SET "
          "role=excluded.role,content=excluded.content,reasoning=excluded."
          "reasoning,state=excluded.state,details=excluded.details WHERE "
          "role!=excluded.role OR content!=excluded.content OR "
          "reasoning!=excluded.reasoning OR state!=excluded.state OR "
          "details!=excluded.details");
      insert.Text(1, c.id);
      insert.Int(2, i);
      insert.Text(3, m.role);
      insert.Text(4, m.content);
      insert.Text(5, m.reasoning);
      insert.Int(6, m.error ? 2 : m.streaming ? 1 : 0);
      insert.Text(7, details[i]);
      insert.Row();
    }
    Exec(db_, "COMMIT");
  } catch (...) {
    sqlite3_exec(db_, "ROLLBACK", nullptr, nullptr, nullptr);
    throw;
  }
}
std::future<void> ChatStore::Save(Conversation c) {
  return Submit([this, c = std::move(c)] { SaveNow(c); });
}
Conversation ChatStore::LoadNow(const std::string& id) {
  if (!open_error_.empty()) throw std::runtime_error(open_error_);
  Statement q(db_,
              "SELECT title,identity,profile,updated,pinned,archived FROM "
              "conversations WHERE id=?");
  q.Text(1, id);
  if (!q.Row()) throw std::runtime_error("Chat no longer exists.");
  Conversation c;
  c.id = id;
  c.title = q.Text(0);
  c.identity = q.Text(1);
  auto profile = q.Int(2);
  if (profile < 0 || profile >= static_cast<int>(kModelProfileCount))
    throw std::runtime_error("Unsupported saved model profile.");
  c.profile = static_cast<ModelProfile>(profile);
  c.updated = q.Int(3);
  c.pinned = q.Int(4) != 0;
  c.archived = q.Int(5) != 0;
  Statement ms(db_,
               "SELECT role,content,reasoning,state,details FROM messages "
               "WHERE chat=? ORDER BY pos");
  ms.Text(1, id);
  std::size_t total_bytes = 0, total_text = 0;
  while (ms.Row()) {
    if (c.messages.size() >= 10000)
      throw std::runtime_error("Saved chat exceeds message limit.");
    ChatMessage m;
    m.role = ms.Text(0);
    m.content = ms.Text(1);
    m.reasoning = ms.Text(2);
    m.error = ms.Int(3) != 0;
    const auto metadata = ms.Text(4);
    total_text += metadata.size() + m.content.size() + m.reasoning.size();
    if (total_text > 32U * 1024U * 1024U)
      throw std::runtime_error("Saved chat exceeds text limit.");
    if (m.role != "user" && m.role != "assistant")
      throw std::runtime_error("Unsupported saved message role.");
    auto parsed = json::Parse(metadata, {32, 100000, 32U * 1024U * 1024U});
    if (!parsed.ok())
      throw std::runtime_error("Saved chat metadata is damaged.");
    const auto& d = parsed.value();
    m.generation = Str(d, "generation");
    m.created = Num(d, "created");
    m.error_message = Str(d, "error_message");
    if (m.error && m.error_message.empty())
      m.error_message = "Interrupted before completion. Retry to continue.";
    for (const auto& a : Array(d, "attachments")) {
      MediaAttachment v;
      v.id = static_cast<std::uint64_t>(c.messages.size() * 1024 +
                                        m.attachments.size() + 1);
      auto kind = Num(a, "kind");
      if (kind < 0 || kind > 2)
        throw std::runtime_error("Invalid attachment kind.");
      v.kind = static_cast<MediaKind>(kind);
      v.file_name = Str(a, "name");
      v.mime_type = Str(a, "mime");
      v.format = Str(a, "format");
      v.document_text = Str(a, "document");
      auto size = Num(a, "size");
      auto hash = Str(a, "hash");
      if (size < 0 || size > 64 * 1024 * 1024 || !SafeHash(hash))
        throw std::runtime_error("Invalid attachment metadata.");
      total_bytes += size;
      if (total_bytes > 256U * 1024U * 1024U)
        throw std::runtime_error("Saved attachments exceed 256 MiB.");
      v.storage_hash = hash;
      v.byte_size = size;
      if (Num(a, "width") < 0 || Num(a, "width") > 65536 ||
          Num(a, "height") < 0 || Num(a, "height") > 65536)
        throw std::runtime_error("Invalid image dimensions.");
      v.image_width = static_cast<int>(Num(a, "width"));
      v.image_height = static_cast<int>(Num(a, "height"));
      const auto path = root_ / "attachments" / hash;
      std::error_code ec;
      if (std::filesystem::is_symlink(
              std::filesystem::symlink_status(path, ec)) ||
          !std::filesystem::is_regular_file(path, ec) ||
          std::filesystem::file_size(path, ec) != v.byte_size) {
        v.missing = true;
      } else {
        v.bytes.resize(static_cast<std::size_t>(size));
        std::ifstream in(path, std::ios::binary);
        in.read(reinterpret_cast<char*>(v.bytes.data()), size);
        if (!in ||
            compiler::Sha256Hex(v.bytes.data(), v.bytes.size()) != hash) {
          v.bytes.clear();
          v.missing = true;
        }
      }
      m.attachments.push_back(std::move(v));
    }
    for (const auto& a : Array(d, "attempts"))
      m.attempts.push_back({Str(a, "content"), Str(a, "reasoning"),
                            Str(a, "error_message"), Str(a, "generation"),
                            Num(a, "created"), Flag(a, "error")});
    c.messages.push_back(std::move(m));
  }
  return c;
}
std::future<Conversation> ChatStore::Load(std::string id) {
  return Submit([this, id = std::move(id)] { return LoadNow(id); });
}
std::future<std::vector<ConversationSummary>> ChatStore::List(
    bool archived, std::string query) {
  return Submit([this, archived, query = std::move(query)] {
    if (!open_error_.empty()) throw std::runtime_error(open_error_);
    (void)query;
    std::vector<ConversationSummary> rows;
    Statement q(
        db_,
        "SELECT id,title,profile,updated,pinned,archived,COALESCE((SELECT "
        "substr(content,1,100) FROM messages WHERE chat=id ORDER BY pos DESC "
        "LIMIT 1),'') FROM conversations WHERE archived=? ORDER BY pinned "
        "DESC,updated DESC LIMIT 500");
    q.Int(1, archived);
    while (q.Row()) {
      if (q.Int(2) < 0 || q.Int(2) >= static_cast<int>(kModelProfileCount))
        throw std::runtime_error("Unsupported saved model profile.");
      rows.push_back({q.Text(0), q.Text(1), q.Text(6),
                      static_cast<ModelProfile>(q.Int(2)), q.Int(3),
                      q.Int(4) != 0, q.Int(5) != 0, -1});
    }
    return rows;
  });
}
std::future<void> ChatStore::Delete(std::string id) {
  return Submit([this, id = std::move(id)] {
    if (!open_error_.empty()) throw std::runtime_error(open_error_);
    Statement q(db_, "DELETE FROM conversations WHERE id=?");
    q.Text(1, id);
    q.Row();
  });
}
}  // namespace gem16::studio
