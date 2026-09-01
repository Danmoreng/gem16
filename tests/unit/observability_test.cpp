#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "server/observability.h"
#include "server/session_pool.h"
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

void TestVisionPrometheusMetrics() {
  gem16::server::ServerMetrics metrics;
  metrics.vision_requests.store(3U);
  metrics.vision_failures.store(1U);
  metrics.vision_budget_140.store(2U);
  metrics.selected_vision_soft_token_budget.store(140U);
  metrics.vision_tower_microseconds.store(252000U);
  const std::string output = gem16::server::VisionMetricsText(metrics);
  GEM16_CHECK(output.find("# TYPE gem16_vision_requests_total counter\n") !=
              std::string::npos);
  GEM16_CHECK(output.find("gem16_vision_requests_total 3\n") !=
              std::string::npos);
  GEM16_CHECK(output.find("gem16_vision_failures_total 1\n") !=
              std::string::npos);
  GEM16_CHECK(output.find("gem16_vision_budget_140_total 2\n") !=
              std::string::npos);
  GEM16_CHECK(output.find(
                  "# TYPE gem16_selected_vision_soft_token_budget gauge\n") !=
              std::string::npos);
  GEM16_CHECK(output.find("gem16_selected_vision_soft_token_budget 140\n") !=
              std::string::npos);
  GEM16_CHECK(output.find("gem16_vision_tower_microseconds_total 252000\n") !=
              std::string::npos);
}

void TestVisionErrorCodes() {
  const auto code = [](std::string message) {
    return gem16::server::VisionErrorCode(gem16::Status(
        gem16::StatusCode::kUnsupported, std::move(message)));
  };
  GEM16_CHECK(code("Vision module is not loaded") ==
              "vision_module_not_loaded");
  GEM16_CHECK(code("a Vision profile is required for image input") ==
              "vision_profile_required");
  GEM16_CHECK(code("the active Vision profile supports exactly one image") ==
              "vision_multiple_images_unsupported");
  GEM16_CHECK(code("Vision with fixed-D2 is not qualified") ==
              "vision_mtp_unqualified");
  GEM16_CHECK(code("Vision soft-token budget is unsupported") ==
              "vision_budget_unsupported");
  GEM16_CHECK(code("Vision context is outside the measured profile limit") ==
              "vision_context_unqualified");
  GEM16_CHECK(code("Vision artifact identity mismatch") ==
              "vision_artifact_mismatch");
  GEM16_CHECK(code("unrelated failure").empty());
}

}  // namespace

void RunObservabilityTests() {
  TestLogParsing();
  TestJsonLogRecord();
  TestConcurrentRecordsStayAtomic();
  TestVisionPrometheusMetrics();
  TestVisionErrorCodes();
}
