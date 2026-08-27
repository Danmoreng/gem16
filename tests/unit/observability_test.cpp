#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "server/observability.h"
#include "test.h"
#include "util/json.h"

namespace {

void TestLogParsing() {
  GEM16_CHECK(gem16::server::ParseLogLevel("debug").ok());
  GEM16_CHECK(gem16::server::ParseLogLevel("off").ok());
  GEM16_CHECK(!gem16::server::ParseLogLevel("verbose").ok());
  GEM16_CHECK(gem16::server::ParseLogFormat("text").ok());
  GEM16_CHECK(gem16::server::ParseLogFormat("json").ok());
  GEM16_CHECK(!gem16::server::ParseLogFormat("xml").ok());
}

void TestJsonLogRecord() {
  std::ostringstream output;
  gem16::server::StructuredLogger logger(
      gem16::server::LogLevel::kInfo, gem16::server::LogFormat::kJson,
      output);
  logger.Log(gem16::server::LogLevel::kDebug, "hidden");
  logger.Log(gem16::server::LogLevel::kInfo, "request_completed",
             {{"request_id", "req_test"}, {"status", "200"}});
  GEM16_CHECK(output.str().find("hidden") == std::string::npos);
  auto parsed = gem16::json::Parse(output.str());
  GEM16_CHECK(parsed.ok());
  if (parsed.ok()) {
    GEM16_CHECK(parsed.value().find("event")->as_string() ==
                "request_completed");
    GEM16_CHECK(parsed.value().find("request_id")->as_string() ==
                "req_test");
  }
}

void TestConcurrentRecordsStayAtomic() {
  std::ostringstream output;
  gem16::server::StructuredLogger logger(
      gem16::server::LogLevel::kInfo, gem16::server::LogFormat::kJson,
      output);
  std::vector<std::thread> writers;
  for (int writer = 0; writer < 4; ++writer) {
    writers.emplace_back([&logger, writer] {
      for (int record = 0; record < 20; ++record) {
        logger.Log(gem16::server::LogLevel::kInfo, "concurrent",
                   {{"writer", std::to_string(writer)},
                    {"record", std::to_string(record)}});
      }
    });
  }
  for (std::thread& writer : writers) writer.join();
  std::size_t records = 0U;
  for (const char value : output.str()) {
    if (value == '\n') ++records;
  }
  GEM16_CHECK(records == 80U);
}

}  // namespace

void RunObservabilityTests() {
  TestLogParsing();
  TestJsonLogRecord();
  TestConcurrentRecordsStayAtomic();
}
