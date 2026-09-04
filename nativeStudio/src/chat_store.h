#pragma once

#include <condition_variable>
#include <deque>
#include <filesystem>
#include <functional>
#include <future>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "types.h"

struct sqlite3;
namespace gem16::studio {

struct Conversation {
  std::string id;
  std::string title;
  std::string identity;
  ModelProfile profile = ModelProfile::kGemma4Unified12B;
  std::int64_t updated = 0;
  std::int64_t created = 0;
  bool pinned = false;
  bool archived = false;
  std::vector<ChatMessage> messages;
};
struct ConversationSummary {
  std::string id, title, preview;
  ModelProfile profile = ModelProfile::kGemma4Unified12B;
  std::int64_t updated = 0;
  bool pinned = false, archived = false;
  std::int64_t hit_position = -1;
};
std::filesystem::path StudioDataDirectory();
std::string NewChatId();
std::string ModelIdentity(const ServerConfig& server);
std::string GenerationIdentity(const StudioSettings& settings);
void PreserveAttempt(ChatMessage& message);
GenerationConfig SavedGeneration(const std::string& metadata);

// One worker owns the SQLite connection. Futures propagate storage errors to
// UI. The UI keeps at most one checkpoint in flight; the bounded queue is a
// backstop.
class ChatStore final {
 public:
  explicit ChatStore(std::filesystem::path root = StudioDataDirectory());
  ~ChatStore();
  ChatStore(const ChatStore&) = delete;
  ChatStore& operator=(const ChatStore&) = delete;
  std::future<void> Save(Conversation chat);
  std::future<Conversation> Load(std::string id);
  std::future<std::vector<ConversationSummary>> List(bool archived = false,
                                                     std::string query = {});
  std::future<void> Delete(std::string id);
  std::future<std::filesystem::path> Export(Conversation chat,
                                            std::filesystem::path destination);
  std::future<std::filesystem::path> Backup(std::filesystem::path destination);
  std::future<std::uint64_t> CleanAttachments();
  const std::filesystem::path& Root() const { return root_; }

 private:
  template <class F>
  auto Submit(F fn) -> std::future<decltype(fn())> {
    auto task =
        std::make_shared<std::packaged_task<decltype(fn())()>>(std::move(fn));
    auto result = task->get_future();
    {
      std::lock_guard lock(mutex_);
      if (queue_.size() >= 16 || stopping_)
        throw std::runtime_error("Chat storage is busy; please retry.");
      queue_.push_back([task] { (*task)(); });
    }
    wake_.notify_one();
    return result;
  }
  void Open();
  void SaveNow(const Conversation& chat);
  Conversation LoadNow(const std::string& id);
  std::filesystem::path root_;
  sqlite3* db_ = nullptr;
  std::intptr_t lock_handle_ = -1;
  std::mutex mutex_;
  std::condition_variable wake_;
  std::deque<std::function<void()>> queue_;
  bool stopping_ = false;
  std::thread worker_;
  std::string open_error_;
};
}  // namespace gem16::studio
