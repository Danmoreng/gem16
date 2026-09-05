#pragma once

#include <string>
#include <string_view>
#include <utility>
#if defined(GEM16_SERVER_TEST_FAULTS)
#include <atomic>
#include <new>
#include <stdexcept>
#endif

namespace gem16::server {

// Only the explicitly built gem16-server-fault-test executable enables these
// hooks. No model substitution or fake inference implementation is involved.
#if defined(GEM16_SERVER_TEST_FAULTS)
inline thread_local std::string_view test_fault;
inline std::atomic<unsigned long long> test_faults_observed{0};
inline bool TestStatusFailure(std::string_view stage) {
  if (test_fault != stage) return false;
  test_fault = {};
  test_faults_observed.fetch_add(1);
  return true;
}
class TestFaultScope {
 public:
  explicit TestFaultScope(std::string value)
      : value_(std::move(value)), previous_(test_fault) { test_fault = value_; }
  ~TestFaultScope() { test_fault = previous_; }
 private:
  std::string value_;
  std::string_view previous_;
};
inline void TestFaultPoint(std::string_view stage) {
  const auto separator = test_fault.find(':');
  if (test_fault.substr(0, separator) != stage) return;
  const auto mode = separator == std::string_view::npos
      ? std::string_view{} : test_fault.substr(separator + 1);
  test_fault = {};
  test_faults_observed.fetch_add(1);
  if (mode == "bad_alloc") throw std::bad_alloc();
  if (mode == "unknown") throw 42;
  throw std::runtime_error("injected server lifecycle failure");
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
template <typename Request>
std::string RequestedTestFault(const Request&) { return {}; }
#endif
}  // namespace gem16::server
