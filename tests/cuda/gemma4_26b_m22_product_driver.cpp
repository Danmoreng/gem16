#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "gem16/engine.h"

namespace {

struct Options {
  std::filesystem::path model;
  std::filesystem::path output;
  std::uint64_t context = 32768U;
  int device = 0;
};

bool ParseUnsigned(std::string_view text, std::uint64_t* output) {
  try {
    std::size_t used = 0U;
    const std::uint64_t value = std::stoull(std::string(text), &used);
    if (used != text.size()) return false;
    *output = value;
    return true;
  } catch (...) {
    return false;
  }
}

bool Parse(int argc, char** argv, Options* options) {
  for (int index = 1; index < argc; ++index) {
    if (index + 1 >= argc) return false;
    const std::string_view key(argv[index]);
    const std::string_view value(argv[++index]);
    if (key == "--model") {
      options->model = value;
    } else if (key == "--output") {
      options->output = value;
    } else if (key == "--context") {
      if (!ParseUnsigned(value, &options->context)) return false;
    } else if (key == "--device") {
      std::uint64_t parsed = 0U;
      if (!ParseUnsigned(value, &parsed) ||
          parsed > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        return false;
      }
      options->device = static_cast<int>(parsed);
    } else {
      return false;
    }
  }
  return !options->model.empty() && !options->output.empty() &&
         options->context >= 64U;
}

int Fail(std::string_view operation, const gem16::Status& status, int code) {
  std::cerr << operation << ": status_code="
            << static_cast<int>(status.code()) << ": " << status.message()
            << '\n';
  return code;
}

int FailCheck(std::string_view message, int code) {
  std::cerr << "M22 product check failed: " << message << '\n';
  return code;
}

gem16::Status CancelOnFirstToken(void* context, std::uint32_t) {
  auto* callback_count = static_cast<std::uint64_t*>(context);
  ++*callback_count;
  return gem16::Status(gem16::StatusCode::kCancelled,
                       "intentional M22 cancellation");
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!Parse(argc, argv, &options)) {
    std::cerr << "usage: gem16-26b-m22-product-driver --model DIR --output "
                 "JSON [--context N] [--device N]\n";
    return 2;
  }

  gem16::ModelRuntimeOptions runtime_options;
  runtime_options.model_directory = options.model;
  runtime_options.max_context_tokens = options.context;
  runtime_options.device = options.device;
  auto runtime_result = gem16::ModelRuntime::Load(runtime_options);
  if (!runtime_result.ok()) {
    return Fail("load M22 resident runtime", runtime_result.status(), 3);
  }
  std::shared_ptr<gem16::ModelRuntime> runtime =
      std::move(runtime_result).value();
  if (std::string_view(runtime->model_variant_name()) !=
          "gemma4_moe_26b_a4b" ||
      std::string_view(runtime->selected_native_path()) !=
          "sm120_integrated_nvfp4_moe_fp8_kv" ||
      runtime->supports_audio() || runtime->supports_vision() ||
      runtime->supports_mtp() || runtime->maximum_execution_slots() != 1U ||
      runtime->max_context_tokens() != options.context) {
    return FailCheck("resident runtime capability metadata is inaccurate", 4);
  }

  gem16::ConversationSessionOptions session_options;
  session_options.model_directory = options.model;
  session_options.max_context_tokens = options.context;
  session_options.kv_cache_mode = gem16::KvCacheMode::kCheckpointFp8;

  constexpr std::array<std::uint32_t, 20> kPrompt = {
      2U,     105U,  2364U, 107U, 40654U, 607U, 7121U,
      506U,   3658U, 3730U, 236761U, 106U, 107U, 105U,
      4368U,  107U,  100U,  45518U, 107U, 101U};
  std::uint64_t first_output_tokens = 0U;
  std::uint64_t second_output_tokens = 0U;
  std::uint64_t second_cached_tokens = 0U;
  std::uint64_t second_cache_write_tokens = 0U;
  std::int64_t second_slot_free_delta_bytes = 0;

  {
    auto session_result =
        gem16::ConversationSession::Create(runtime, session_options);
    if (!session_result.ok()) {
      return Fail("create first M22 session", session_result.status(), 5);
    }
    gem16::ConversationSession session = std::move(session_result).value();

    auto free_before_result = gem16::QueryDeviceMemoryInfo();
    if (!free_before_result.ok()) {
      return Fail("measure memory before second slot",
                  free_before_result.status(), 5);
    }
    auto second_slot =
        gem16::ConversationSession::Create(runtime, session_options);
    if (second_slot.ok() ||
        second_slot.status().code() != gem16::StatusCode::kResourceExhausted) {
      return FailCheck("second 26B slot was not rejected precisely", 6);
    }
    auto free_after_result = gem16::QueryDeviceMemoryInfo();
    if (!free_after_result.ok()) {
      return Fail("measure memory after second slot",
                  free_after_result.status(), 6);
    }
    second_slot_free_delta_bytes =
        static_cast<std::int64_t>(free_after_result.value().free_bytes) -
        static_cast<std::int64_t>(free_before_result.value().free_bytes);
    if (second_slot_free_delta_bytes < -(1LL << 20)) {
      return FailCheck("rejected second slot consumed device memory", 6);
    }

    gem16::ConversationSessionOptions bf16_options = session_options;
    bf16_options.kv_cache_mode = gem16::KvCacheMode::kBf16Correctness;
    auto bf16_session =
        gem16::ConversationSession::Create(runtime, bf16_options);
    if (bf16_session.ok() ||
        bf16_session.status().code() != gem16::StatusCode::kUnsupported) {
      return FailCheck("unsupported 26B BF16 KV mode was not rejected", 7);
    }
    gem16::ConversationSessionOptions mtp_options = session_options;
    mtp_options.mtp_draft_tokens = 1U;
    auto mtp_session = gem16::ConversationSession::Create(runtime, mtp_options);
    if (mtp_session.ok() ||
        mtp_session.status().code() != gem16::StatusCode::kUnsupported) {
      return FailCheck("unsupported 26B MTP request was not rejected", 7);
    }

    auto first = session.Generate(kPrompt, 3U);
    if (!first.ok()) return Fail("run first M22 turn", first.status(), 8);
    if (first.value().output_token_ids.size() != 3U ||
        first.value().prompt_cached_tokens != 0U ||
        first.value().prompt_cache_write_tokens != kPrompt.size() ||
        first.value().token_loop_allocations || !first.value().decode_graphs ||
        first.value().fallback_count != 0U) {
      return FailCheck("first resident turn telemetry is inconsistent", 8);
    }
    first_output_tokens = first.value().output_token_ids.size();

    std::vector<std::uint32_t> continuation(kPrompt.begin(), kPrompt.end());
    continuation.insert(continuation.end(),
                        first.value().output_token_ids.begin(),
                        first.value().output_token_ids.end());
    continuation.push_back(107U);
    const std::uint64_t expected_cached =
        kPrompt.size() + first.value().output_token_ids.size() - 1U;
    const std::uint64_t expected_write =
        continuation.size() - expected_cached;
    auto second = session.Generate(continuation, 2U);
    if (!second.ok()) return Fail("run resident M22 continuation", second.status(), 9);
    if (second.value().output_token_ids.size() != 2U ||
        second.value().prompt_cached_tokens != expected_cached ||
        second.value().prompt_cache_write_tokens != expected_write ||
        second.value().token_loop_allocations || !second.value().decode_graphs ||
        second.value().fallback_count != 0U) {
      return FailCheck("resident continuation did not preserve prefix ownership",
                       9);
    }
    second_output_tokens = second.value().output_token_ids.size();
    second_cached_tokens = second.value().prompt_cached_tokens;
    second_cache_write_tokens = second.value().prompt_cache_write_tokens;

    std::vector<std::uint32_t> mismatched_prompt = continuation;
    mismatched_prompt.front() = 3U;
    auto mismatch = session.Generate(mismatched_prompt, 1U);
    if (mismatch.ok() ||
        mismatch.status().code() != gem16::StatusCode::kInvalidArgument ||
        session.is_poisoned()) {
      return FailCheck("resident prefix mismatch was not rejected safely", 10);
    }

    const std::array<float, 1> invalid_audio = {0.0F};
    const gem16::AudioEmbeddingSegment audio_segment = {
        .prompt_offset = 0U,
        .frames = invalid_audio,
    };
    const std::array<gem16::AudioEmbeddingSegment, 1> audio_segments = {
        audio_segment};
    auto media = session.Generate(
        continuation, 1U, {}, nullptr, nullptr, audio_segments, {});
    if (media.ok() || media.status().code() != gem16::StatusCode::kUnsupported ||
        session.is_poisoned()) {
      return FailCheck("26B media request was not rejected safely", 11);
    }
  }

  std::uint64_t cancellation_callbacks = 0U;
  {
    auto session_result =
        gem16::ConversationSession::Create(runtime, session_options);
    if (!session_result.ok()) {
      return Fail("relaunch M22 session before cancellation",
                  session_result.status(), 12);
    }
    gem16::ConversationSession session = std::move(session_result).value();
    auto cancelled = session.Generate(kPrompt, 2U, {}, CancelOnFirstToken,
                                      &cancellation_callbacks);
    if (cancelled.ok() ||
        cancelled.status().code() != gem16::StatusCode::kCancelled ||
        cancellation_callbacks != 1U || !session.is_poisoned()) {
      return FailCheck("cancelled session did not enter poisoned state", 13);
    }
    auto retry = session.Generate(kPrompt, 1U);
    if (retry.ok() || retry.status().code() != gem16::StatusCode::kInternal) {
      return FailCheck("poisoned session accepted a continuation", 13);
    }
  }

  {
    auto session_result =
        gem16::ConversationSession::Create(runtime, session_options);
    if (!session_result.ok()) {
      return Fail("relaunch M22 session after cancellation",
                  session_result.status(), 14);
    }
    gem16::ConversationSession session = std::move(session_result).value();
    auto relaunched = session.Generate(kPrompt, 1U);
    if (!relaunched.ok()) {
      return Fail("run M22 post-cancellation relaunch", relaunched.status(),
                  14);
    }
  }

  std::ofstream output(options.output, std::ios::binary | std::ios::trunc);
  if (!output) {
    std::cerr << "cannot open M22 output: " << options.output << '\n';
    return 15;
  }
  output << "{\n"
         << "  \"schema_version\": 1,\n"
         << "  \"milestone\": \"M22\",\n"
         << "  \"model_variant\": \"" << runtime->model_variant_name()
         << "\",\n"
         << "  \"native_path\": \"" << runtime->selected_native_path()
         << "\",\n"
         << "  \"max_context_tokens\": " << runtime->max_context_tokens()
         << ",\n"
         << "  \"maximum_execution_slots\": "
         << runtime->maximum_execution_slots() << ",\n"
         << "  \"supports_audio\": false,\n"
         << "  \"supports_vision\": false,\n"
         << "  \"supports_mtp\": false,\n"
         << "  \"first_output_tokens\": " << first_output_tokens << ",\n"
         << "  \"second_output_tokens\": " << second_output_tokens << ",\n"
         << "  \"second_prompt_cached_tokens\": " << second_cached_tokens
         << ",\n"
         << "  \"second_prompt_cache_write_tokens\": "
         << second_cache_write_tokens << ",\n"
         << "  \"second_slot_free_delta_bytes\": "
         << second_slot_free_delta_bytes << ",\n"
         << "  \"second_slot_rejected\": true,\n"
         << "  \"unsupported_bf16_kv_rejected\": true,\n"
         << "  \"unsupported_mtp_rejected\": true,\n"
         << "  \"unsupported_media_rejected\": true,\n"
         << "  \"prefix_mismatch_rejected\": true,\n"
         << "  \"cancellation_callbacks\": " << cancellation_callbacks
         << ",\n"
         << "  \"cancelled_session_poisoned\": true,\n"
         << "  \"post_cancellation_relaunch_passed\": true,\n"
         << "  \"passed\": true\n"
         << "}\n";
  if (!output) {
    std::cerr << "failed to write M22 output: " << options.output << '\n';
    return 15;
  }
  return 0;
}
