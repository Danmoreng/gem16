#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string_view>
#include <type_traits>

#include "httplib.h"

#include "gem16/chat.h"
#include "server/openai_chat.h"

namespace gem16::server {

// httplib catches routing exceptions, but executes body providers afterwards.
// Never let a deferred generation/serialization exception escape its worker.
template <typename Callback, typename Failure>
bool RunStreamCallback(Callback&& callback, Failure&& failure) noexcept {
  static_assert(std::is_nothrow_invocable_v<Failure>);
  try {
    return callback();
  } catch (...) {
    failure();
    return false;
  }
}

struct StreamCancellation {
  std::atomic<bool>* requested = nullptr;
  std::atomic<std::uint64_t>* observed = nullptr;
  std::atomic<std::uint64_t>* disconnects = nullptr;
};

[[nodiscard]] bool WriteSse(httplib::DataSink& sink,
                            std::string_view payload);
[[nodiscard]] bool FinishSse(httplib::DataSink& sink);

class ChatCompletionStream {
 public:
  ChatCompletionStream(const GemmaChatProcessor& processor,
                       OpenAiResponseIdentity identity,
                       httplib::DataSink& sink,
                       StreamCancellation cancellation);
  ChatCompletionStream(const ChatCompletionStream&) = delete;
  ChatCompletionStream& operator=(const ChatCompletionStream&) = delete;
  ~ChatCompletionStream();

  [[nodiscard]] Result<ChatGenerationResponse> Generate(
      ChatSession& session, const ChatGenerationRequest& request);
  [[nodiscard]] Status WriteToolCalls(
      const ChatGenerationResponse& response);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class ResponsesStream {
 public:
  ResponsesStream(const GemmaChatProcessor& processor,
                  OpenAiResponseIdentity identity,
                  httplib::DataSink& sink,
                  std::uint64_t reasoning_token_capacity,
                  StreamCancellation cancellation);
  ResponsesStream(const ResponsesStream&) = delete;
  ResponsesStream& operator=(const ResponsesStream&) = delete;
  ~ResponsesStream();

  [[nodiscard]] Result<ChatGenerationResponse> Generate(
      ChatSession& session, const ChatGenerationRequest& request);
  [[nodiscard]] bool WriteFinalEvents(
      const OpenAiResponsesRequest& request,
      const ChatGenerationResponse& response, std::int64_t completed_at);
  [[nodiscard]] std::uint64_t sequence() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gem16::server
