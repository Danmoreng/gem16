#pragma once

#include <string>
#include <string_view>
#include <utility>
#if defined(GEM16_SERVER_TEST_FAULTS)
#include <atomic>
#include <chrono>
#include <new>
#include <stdexcept>
#include <thread>
#endif

namespace gem16::server {

// Only the explicitly built gem16-server-fault-test executable enables these
// hooks. No model substitution or fake inference implementation is involved.
#if defined(GEM16_SERVER_TEST_FAULTS)
inline thread_local std::string_view test_fault;
inline thread_local bool test_generation_token_seen = false;
inline std::atomic<unsigned long long> test_faults_observed{0};
inline std::atomic<bool> test_pause_active{false};
inline bool TestStatusFailure(std::string_view stage) {
  if (test_fault != stage) return false;
  test_fault = {};
  test_faults_observed.fetch_add(1);
  return true;
}
class TestFaultScope {
 public:
  explicit TestFaultScope(std::string value)
      : value_(std::move(value)), previous_(test_fault),
        previous_token_seen_(test_generation_token_seen) {
    test_fault = value_;
    test_generation_token_seen = false;
  }
  ~TestFaultScope() {
    test_fault = previous_;
    test_generation_token_seen = previous_token_seen_;
  }
 private:
  std::string value_;
  std::string_view previous_;
  bool previous_token_seen_;
};
inline void TestFaultPoint(std::string_view stage) {
  const auto separator = test_fault.find(':');
  if (test_fault.substr(0, separator) != stage) return;
  const auto mode = separator == std::string_view::npos
      ? std::string_view{} : test_fault.substr(separator + 1);
  if (mode == "wait") return;  // Consumed by the cooperative test pause below.
  test_fault = {};
  test_faults_observed.fetch_add(1);
  if (mode == "bad_alloc") throw std::bad_alloc();
  if (mode == "unknown") throw 42;
  throw std::runtime_error("injected server lifecycle failure");
}
template <typename Check>
auto TestPause(std::string_view stage, Check check) {
  const auto separator = test_fault.find(':');
  if (separator == std::string_view::npos || test_fault.substr(0, separator) != stage ||
      test_fault.substr(separator + 1) != "wait") return check();
  // The first token comes from prefill logits; pause after a decode step.
  if (stage == "generation" && !test_generation_token_seen) {
    test_generation_token_seen = true;
    return check();
  }
  test_fault = {};
  test_faults_observed.fetch_add(1);
  test_pause_active.store(true);
  struct Release { ~Release() { test_pause_active.store(false); } } release;
  const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(35);
  for (;;) {
    auto status = check();
    if (!status.ok() || std::chrono::steady_clock::now() >= end) return status;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
}
template <typename Request>
std::string RequestedTestFault(const Request& request) {
  return request.get_header_value("X-Gem16-Test-Fault");
}
#else
class TestFaultScope {
 public:
  explicit TestFaultScope(std::string) {}
};
inline void TestFaultPoint(std::string_view) {}
inline bool TestStatusFailure(std::string_view) { return false; }
template <typename Check>
auto TestPause(std::string_view, Check check) { return check(); }
template <typename Request>
std::string RequestedTestFault(const Request&) { return {}; }
#endif
}  // namespace gem16::server
