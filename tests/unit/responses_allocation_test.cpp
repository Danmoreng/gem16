// Isolated executable: allocation failures must never affect other host tests.
#include <cstdlib>
#include <new>
#include <string>
#if defined(_MSC_VER)
#include <crtdbg.h>
#endif

#include "server/session_pool.h"
#include "test.h"

namespace {
thread_local int allocations_until_failure = -1;
}

void* operator new(std::size_t size) {
  if (allocations_until_failure == 0) throw std::bad_alloc();
  if (allocations_until_failure > 0) --allocations_until_failure;
  if (void* memory = std::malloc(size == 0U ? 1U : size)) return memory;
  throw std::bad_alloc();
}
void* operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void* memory) noexcept { std::free(memory); }
void operator delete[](void* memory) noexcept { std::free(memory); }
void operator delete(void* memory, std::size_t) noexcept { std::free(memory); }
void operator delete[](void* memory, std::size_t) noexcept { std::free(memory); }

int main() {
#if defined(_MSC_VER)
  _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
  _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
  _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
  _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
  _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
#endif
#if defined(_MSC_VER) && _ITERATOR_DEBUG_LEVEL != 0
  // MSVC debug containers allocate iterator proxies in noexcept constructors.
  // Injecting there terminates by design; exercise failures in the Release
  // executable, whose STL layout matches the shipped server.
  std::cout << "Run allocation injection with MSVC Release (debug iterator proxies allocate in noexcept constructors)\n";
  return 77;
#else
  using namespace gem16;
  using namespace gem16::server;
  OpenAiResponsesRequest request;
  request.generation.tools.push_back(
      {std::string(80U, 't'), std::string(120U, 'd'), "{\"type\":\"object\"}", false});
  request.generation.tool_choice.function_name = std::string(80U, 't');
  request.generation.messages.push_back(
      {"user", {GenerationContentPart::Text(std::string(200U, 'q'))}});
  ChatGenerationResponse response;
  response.assistant_text = std::string(300U, 'a');
  response.tool_calls.push_back(
      {std::string(80U, 'i'), std::string(80U, 'f'), std::string(80U, 'x')});
  ResponsesChain original;
  original.latest_response_id = std::string(80U, 'p');
  original.messages.push_back(
      {"user", {GenerationContentPart::Text("previous turn")}});
  original.initialized = true;
  bool completed = false;
  int injected = 0;
  for (int fail_at = 0; fail_at < 1000; ++fail_at) {
    ResponsesChain chain = original;
    std::string id(80U, 'n');
    allocations_until_failure = fail_at;
    try {
      CommitResponsesChain(chain, request, response, std::move(id));
      allocations_until_failure = -1;
      GEM16_CHECK(chain.latest_response_id == std::string(80U, 'n'));
      GEM16_CHECK(chain.messages.size() == 2U);
      GEM16_CHECK(chain.messages.back().content.size() == 2U);
      GEM16_CHECK(chain.initialized);
      completed = true;
      break;
    } catch (const std::bad_alloc&) {
      allocations_until_failure = -1;
      ++injected;
      GEM16_CHECK(chain.latest_response_id == original.latest_response_id);
      GEM16_CHECK(chain.messages == original.messages);
      GEM16_CHECK(chain.tools == original.tools);
      GEM16_CHECK(chain.tool_choice == original.tool_choice);
      GEM16_CHECK(chain.initialized == original.initialized);
      // A failed commit must not prevent a later successful turn.
      CommitResponsesChain(chain, request, response, std::string(80U, 'n'));
      GEM16_CHECK(chain.messages.size() == 2U);
    }
  }
  GEM16_CHECK(completed);
  GEM16_CHECK(injected > 5);
  std::cout << "Responses commit: " << injected
            << " allocation failures rolled back; subsequent commits recovered\n";
  return gem16::test::failures == 0 ? 0 : 1;
#endif
}
