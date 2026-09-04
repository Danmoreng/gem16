#include <cstdio>
#include <fstream>
#include <stdexcept>

#include "chat_store.h"
#include "settings.h"
#include "sqlite3.h"

namespace {
using namespace gem16::studio;
void Require(bool condition, const char* text) {
  if (!condition) throw std::runtime_error(text);
}
template <class F>
void Fails(F fn) {
  bool failed = false;
  try {
    fn();
  } catch (const std::exception&) {
    failed = true;
  }
  Require(failed, "Expected storage failure");
}
void Sql(const std::filesystem::path& path, const char* sql) {
  sqlite3* db = nullptr;
  const auto utf8 = path.u8string();
  Require(sqlite3_open(reinterpret_cast<const char*>(utf8.c_str()), &db) ==
              SQLITE_OK,
          "open fixture");
  int result = sqlite3_exec(db, sql, nullptr, nullptr, nullptr);
  sqlite3_close(db);
  Require(result == SQLITE_OK, "fixture SQL");
}
}  // namespace
bool TestChatStore() {
  const auto root = std::filesystem::temp_directory_path() /
                    ("gem16-chat-test-" + NewChatId());
  try {
    Conversation c;
    c.id = NewChatId();
    c.title = "Grüße aus Köln";
    auto settings = DefaultSettings();
    c.identity = ModelIdentity(settings.server);
    ChatMessage user;
    user.role = "user";
    user.content = "Where is the café?";
    user.created = 123;
    MediaAttachment a;
    a.file_name = "source.txt";
    a.kind = MediaKind::kDocument;
    a.bytes = {'c', 'a', 'f', 'e'};
    a.byte_size = 4;
    a.document_text = "Searchable attachment";
    user.attachments.push_back(a);
    c.messages.push_back(user);
    ChatMessage reply;
    reply.role = "assistant";
    reply.content = "Partial answer";
    reply.reasoning = "Working";
    reply.streaming = true;
    reply.generation = GenerationIdentity(settings);
    c.messages.push_back(reply);
    {
      ChatStore store(root);
      store.Save(c).get();
      ChatStore second(root);
      Fails([&] { second.List().get(); });
      auto read = store.Load(c.id).get();
      Require(read.messages[1].error, "unfinished response is recoverable");
      Require(read.messages[0].attachments[0].bytes == a.bytes,
              "attachment roundtrip");
      Require(SavedGeneration(read.messages[1].generation).max_output_tokens ==
                  settings.generation.max_output_tokens,
              "settings roundtrip");
      PreserveAttempt(read.messages[1]);
      read.messages[1].streaming = false;
      read.messages[1].content = "Complete answer";
      store.Save(read).get();
      auto other = c;
      other.id = NewChatId();
      other.title = "Pinned";
      other.pinned = true;
      store.Save(other).get();
      Require(store.List().get().front().id == other.id, "pin order");
      other.archived = true;
      store.Save(other).get();
      Require(
          store.List().get().size() == 1 && store.List(true).get().size() == 1,
          "archive filter");
      store.Delete(other.id).get();
      Fails([&] { store.Load(other.id).get(); });
      Sql(root / "studio.db",
          "CREATE TRIGGER reject_message BEFORE UPDATE ON messages BEGIN "
          "SELECT RAISE(ABORT,'injected write failure'); END;");
      read.title = "Must roll back";
      read.messages[1].content = "Rejected";
      Fails([&] { store.Save(read).get(); });
      Require(store.Load(c.id).get().title == c.title, "transaction rollback");
      Sql(root / "studio.db", "DROP TRIGGER reject_message;");
    }
    {
      ChatStore store(root);
      auto read = store.Load(c.id).get();
      Require(read.messages[1].content == "Complete answer" &&
                  read.messages[1].attempts.size() == 1,
              "persist attempt and final");
      Require(read.messages[1].attempts[0].content == "Partial answer",
              "retain partial attempt");
      auto path =
          root / "attachments" / read.messages[0].attachments[0].storage_hash;
      {
        std::ofstream out(path);
        out << "xxxx";
      }
      read = store.Load(c.id).get();
      Require(read.messages[0].attachments[0].missing, "same size corruption");
      read.title = "Readable despite damaged media";
      store.Save(read).get();
      Require(store.Load(c.id).get().messages[0].attachments[0].missing,
              "preserve missing reference");
      std::filesystem::remove(path);
      Require(store.Load(c.id).get().messages[0].content == user.content,
              "missing attachment preserves text");
      auto invalid = c;
      invalid.id = "../escape";
      Fails([&] { store.Save(invalid).get(); });
    }
    Sql(root / "studio.db", "PRAGMA user_version=999;");
    {
      ChatStore store(root);
      Fails([&] { store.List().get(); });
    }
    std::filesystem::remove_all(root);
    std::puts(
        "Chat storage: durability, isolation, recovery, attempts, metadata, "
        "media and rollback passed");
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "Chat storage test: %s\n", e.what());
    std::filesystem::remove_all(root);
    return false;
  }
}
