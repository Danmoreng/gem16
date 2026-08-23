#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "compiler/sha256.h"
#include "cuda/engine/gemma4_26b_reference.h"
#include "gem16/tokenizer.h"
#include "util/json.h"

namespace {

struct Options {
  std::filesystem::path model;
  std::filesystem::path corpus;
  std::filesystem::path reference;
  std::filesystem::path output;
  std::uint64_t context = 32768U;
  std::uint64_t scored_tokens = 32U;
  std::uint64_t generated_tokens = 256U;
  std::string record_id;
  bool tokenwise_prefill = false;
  gem16::internal::Gemma4Moe26BBackend backend =
      gem16::internal::Gemma4Moe26BBackend::kSm120Integrated;
  int device = 0;
};

const gem16::json::Value* Field(const gem16::json::Value& value,
                                std::string_view name) {
  return value.is_object() ? value.find(name) : nullptr;
}

bool ParseUnsigned(std::string_view text, std::uint64_t* output) {
  try {
    std::size_t used = 0U;
    *output = std::stoull(std::string(text), &used);
    return used == text.size();
  } catch (...) {
    return false;
  }
}

bool Parse(int argc, char** argv, Options* options) {
  for (int index = 1; index < argc; ++index) {
    if (index + 1 >= argc) return false;
    const std::string_view key(argv[index]);
    const std::string_view value(argv[++index]);
    if (key == "--model") options->model = value;
    else if (key == "--corpus") options->corpus = value;
    else if (key == "--q4-reference") options->reference = value;
    else if (key == "--output") options->output = value;
    else if (key == "--record-id") options->record_id = value;
    else if (key == "--backend") {
      if (value == "reference") {
        options->backend = gem16::internal::Gemma4Moe26BBackend::kReference;
      } else if (value == "sm120-moe-head") {
        options->backend =
            gem16::internal::Gemma4Moe26BBackend::kSm120MoeHead;
      } else if (value == "sm120") {
        options->backend =
            gem16::internal::Gemma4Moe26BBackend::kSm120Integrated;
      } else {
        return false;
      }
    }
    else if (key == "--prefill") {
      if (value == "native") options->tokenwise_prefill = false;
      else if (value == "decode") options->tokenwise_prefill = true;
      else return false;
    }
    else if (key == "--context") {
      if (!ParseUnsigned(value, &options->context)) return false;
    } else if (key == "--scored-tokens") {
      if (!ParseUnsigned(value, &options->scored_tokens)) return false;
    } else if (key == "--generated-tokens") {
      if (!ParseUnsigned(value, &options->generated_tokens)) return false;
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
  return !options->model.empty() && !options->corpus.empty() &&
         !options->reference.empty() && !options->output.empty() &&
         options->context > 1U && options->scored_tokens > 0U &&
         options->generated_tokens > 0U;
}

gem16::Result<std::string> ReadRegular(const std::filesystem::path& path,
                                       std::uint64_t limit) {
  std::error_code error;
  const auto status = std::filesystem::symlink_status(path, error);
  if (error || std::filesystem::is_symlink(status) ||
      !std::filesystem::is_regular_file(status)) {
    return gem16::Status(gem16::StatusCode::kInvalidArgument,
                         "M19 input is not a regular file: " + path.string());
  }
  const std::uint64_t size = std::filesystem::file_size(path, error);
  if (error || size > limit) {
    return gem16::Status(gem16::StatusCode::kDataLoss,
                         "M19 input exceeds its size limit: " + path.string());
  }
  std::string result(static_cast<std::size_t>(size), '\0');
  std::ifstream input(path, std::ios::binary);
  input.read(result.data(), static_cast<std::streamsize>(result.size()));
  if (!input || input.gcount() != static_cast<std::streamsize>(result.size())) {
    return gem16::Status(gem16::StatusCode::kIoError,
                         "cannot read M19 input: " + path.string());
  }
  return result;
}

int Fail(std::string_view operation, const gem16::Status& status, int code) {
  std::cerr << operation << ": status_code="
            << static_cast<int>(status.code()) << ": " << status.message()
            << '\n';
  return code;
}

int FailCheck(std::string_view message, int code) {
  std::cerr << "M19 Q4 comparison failed: " << message << '\n';
  return code;
}

gem16::Result<std::vector<std::uint32_t>> Tokens(
    const gem16::json::Value* value, std::string_view label) {
  if (value == nullptr || !value->is_array() || value->as_array().empty()) {
    return gem16::Status(gem16::StatusCode::kDataLoss,
                         std::string(label) + " is not a token array");
  }
  std::vector<std::uint32_t> result;
  result.reserve(value->as_array().size());
  for (const auto& entry : value->as_array()) {
    if (!entry.is_integer() || entry.as_integer() < 0 ||
        entry.as_integer() >= 262144) {
      return gem16::Status(gem16::StatusCode::kDataLoss,
                           std::string(label) + " has an invalid token");
    }
    result.push_back(static_cast<std::uint32_t>(entry.as_integer()));
  }
  return result;
}

std::string TokenHash(std::span<const std::uint32_t> tokens) {
  return gem16::compiler::Sha256Hex(
      tokens.data(), tokens.size() * sizeof(std::uint32_t));
}

double LogProbability(std::span<const float> logits, std::uint32_t token) {
  const float maximum = *std::max_element(logits.begin(), logits.end());
  double denominator = 0.0;
  for (const float value : logits) {
    denominator += std::exp(static_cast<double>(value - maximum));
  }
  return static_cast<double>(logits[token] - maximum) - std::log(denominator);
}

bool InTopFive(std::span<const float> logits, std::uint32_t token) {
  std::uint32_t better = 0U;
  for (std::uint32_t index = 0U; index < logits.size(); ++index) {
    if (logits[index] > logits[token] ||
        (logits[index] == logits[token] && index < token)) {
      ++better;
      if (better >= 5U) return false;
    }
  }
  return true;
}

gem16::Result<double> Q4SelectedLogProbability(
    const gem16::json::Value& step, std::uint32_t target) {
  const auto* top = Field(step, "top_logprobs");
  if (top == nullptr || !top->is_array()) {
    return gem16::Status(gem16::StatusCode::kDataLoss,
                         "official Q4 step has no top-logprobs");
  }
  for (const auto& entry : top->as_array()) {
    const auto* token = Field(entry, "token_id");
    const auto* logprob = Field(entry, "logprob");
    if (token != nullptr && token->is_integer() &&
        token->as_integer() == static_cast<std::int64_t>(target) &&
        logprob != nullptr && logprob->is_number() &&
        std::isfinite(logprob->as_number())) {
      return logprob->as_number();
    }
  }
  return gem16::Status(gem16::StatusCode::kDataLoss,
                       "official Q4 top-logprobs omit its selected token");
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!Parse(argc, argv, &options)) {
    std::cerr << "usage: gem16-26b-m19-q4-driver --model DIR --corpus JSON "
                 "--q4-reference JSON --output JSON [--context N] "
                 "[--scored-tokens N] [--generated-tokens N] [--record-id ID] "
                 "[--backend reference|sm120-moe-head|sm120] "
                 "[--prefill native|decode] "
                 "[--device N]\n";
    return 2;
  }
  if (std::filesystem::exists(options.output)) {
    return FailCheck("refusing to overwrite output", 2);
  }
  auto corpus_text = ReadRegular(options.corpus, 4U * 1024U * 1024U);
  auto reference_text = ReadRegular(options.reference, 16U * 1024U * 1024U);
  if (!corpus_text.ok()) return Fail("read M19 corpus", corpus_text.status(), 3);
  if (!reference_text.ok()) return Fail("read M19 Q4 reference", reference_text.status(), 3);
  auto corpus = gem16::json::Parse(corpus_text.value());
  auto reference = gem16::json::Parse(reference_text.value());
  const auto* corpus_records = corpus.ok() ? Field(corpus.value(), "records") : nullptr;
  const auto* reference_records = reference.ok() ? Field(reference.value(), "records") : nullptr;
  if (!corpus.ok() || !reference.ok() || corpus_records == nullptr ||
      reference_records == nullptr || !corpus_records->is_array() ||
      !reference_records->is_array() ||
      corpus_records->as_array().size() != reference_records->as_array().size()) {
    return FailCheck("corpus/reference schema or record count changed", 3);
  }
  const std::string corpus_hash = gem16::compiler::Sha256Hex(
      corpus_text.value().data(), corpus_text.value().size());
  const auto* reference_identity = Field(reference.value(), "reference");
  if (Field(reference.value(), "corpus_sha256") == nullptr ||
      !Field(reference.value(), "corpus_sha256")->is_string() ||
      Field(reference.value(), "corpus_sha256")->as_string() != corpus_hash ||
      reference_identity == nullptr || !reference_identity->is_object()) {
    return FailCheck("official Q4 capture identity does not match corpus", 3);
  }

  auto processor = gem16::GemmaChatProcessor::Load(options.model);
  if (!processor.ok()) return Fail("load M19 tokenizer", processor.status(), 4);
  auto engine = gem16::internal::Gemma4Moe26BReferenceEngine::Create(
      options.model, options.context, options.device, options.backend);
  if (!engine.ok()) return Fail("create M19 candidate engine", engine.status(), 4);
  gem16::Status status = engine.value().ConfigureTokenSelection(
      {}, processor.value().generation_controls().suppressed_token_ids);
  if (!status.ok()) return Fail("configure M19 candidate selection", status, 4);
  std::vector<float> logits(262144U);

  auto lock_text = ReadRegular(
      options.model.parent_path() /
          (options.model.filename().string() + ".lock.json"),
      16U * 1024U * 1024U);
  auto lock = lock_text.ok() ? gem16::json::Parse(lock_text.value())
                             : gem16::Result<gem16::json::Value>(lock_text.status());
  if (!lock.ok()) return Fail("read M19 candidate lock", lock.status(), 4);
  const auto* artifact_hash = Field(lock.value(), "artifact_content_sha256");
  if (artifact_hash == nullptr || !artifact_hash->is_string()) {
    return FailCheck("candidate lock has no artifact hash", 4);
  }

  struct RecordResult {
    std::string id;
    std::string category;
    std::string input_hash;
    std::string target_hash;
    std::string reference_capture_hash;
    std::string candidate_capture_hash;
    std::string candidate_response_hash;
    std::string candidate_text;
    std::uint64_t scored = 0U;
    std::uint64_t top1 = 0U;
    std::uint64_t top5 = 0U;
    std::uint64_t logprob_compared = 0U;
    double sum_delta = 0.0;
    double max_delta = 0.0;
    bool finite = true;
  };
  std::vector<RecordResult> results;

  for (std::size_t record_index = 0U;
       record_index < corpus_records->as_array().size(); ++record_index) {
    const auto& source = corpus_records->as_array()[record_index];
    const auto& q4 = reference_records->as_array()[record_index];
    const auto* source_id = Field(source, "id");
    const auto* q4_id = Field(q4, "id");
    const auto* category = Field(source, "category");
    const auto* input_hash = Field(source, "input_token_ids_sha256_u32le");
    const auto* q4_steps = Field(q4, "steps");
    const auto* q4_record_hash = Field(q4, "record_sha256");
    if (source_id == nullptr || q4_id == nullptr || category == nullptr ||
        input_hash == nullptr || q4_steps == nullptr || q4_record_hash == nullptr ||
        !source_id->is_string() || !q4_id->is_string() ||
        source_id->as_string() != q4_id->as_string() ||
        !category->is_string() || !input_hash->is_string() ||
        !q4_steps->is_array() || !q4_record_hash->is_string() ||
        q4_steps->as_array().size() < options.scored_tokens) {
      return FailCheck("official Q4 record identity or extent changed", 5);
    }
    if (!options.record_id.empty() &&
        source_id->as_string() != options.record_id) {
      continue;
    }
    std::cout << "M19 candidate " << source_id->as_string() << '\n'
              << std::flush;
    auto prompt = Tokens(Field(source, "input_token_ids"), "M19 prompt");
    if (!prompt.ok()) return Fail("parse M19 prompt", prompt.status(), 5);
    if (TokenHash(prompt.value()) != input_hash->as_string()) {
      return FailCheck("M19 prompt token hash changed", 5);
    }

    RecordResult result;
    result.id = source_id->as_string();
    result.category = category->as_string();
    result.input_hash = input_hash->as_string();
    result.reference_capture_hash = q4_record_hash->as_string();
    std::vector<std::uint32_t> targets;
    std::string candidate_capture;

    status = engine.value().Reset();
    if (!status.ok()) return Fail("reset M19 candidate", status, 6);
    status = engine.value().ConfigureTokenSelection(
        {}, {});
    if (!status.ok()) return Fail("configure M19 candidate", status, 6);
    if (options.backend == gem16::internal::Gemma4Moe26BBackend::kReference ||
        options.tokenwise_prefill) {
      std::size_t prompt_index = 0U;
      for (const std::uint32_t token : prompt.value()) {
        status = engine.value().ForwardToken(token);
        if (!status.ok()) break;
        if (options.tokenwise_prefill) {
          auto prediction = engine.value().Prediction();
          if (!prediction.ok()) {
            std::cerr << "M19 tokenwise prefill failed after prompt token "
                      << prompt_index << " (id " << token << ")\n";
            std::vector<float> layer_output(2816U);
            for (const std::uint32_t layer : {0U, 5U, 6U, 29U}) {
              const auto capture =
                  engine.value().CopyLayerOutput(layer, layer_output);
              std::size_t non_finite = 0U;
              float maximum = 0.0F;
              if (capture.ok()) {
                for (const float value : layer_output) {
                  if (!std::isfinite(value)) ++non_finite;
                  else maximum = std::max(maximum, std::abs(value));
                }
              }
              std::cerr << "  layer " << layer
                        << " capture_status=" << capture.ok()
                        << " non_finite=" << non_finite
                        << " max_abs=" << maximum << '\n';
            }
            return Fail("inspect M19 tokenwise prefill", prediction.status(), 6);
          }
        }
        ++prompt_index;
      }
    } else {
      status = engine.value().PrefillTokens(prompt.value());
    }
    if (!status.ok()) return Fail("prefill M19 candidate", status, 6);
    for (std::uint64_t step_index = 0U; step_index < options.scored_tokens;
         ++step_index) {
      const auto& q4_step = q4_steps->as_array()[step_index];
      const auto* target_value = Field(q4_step, "token_id");
      if (target_value == nullptr || !target_value->is_integer() ||
          target_value->as_integer() < 0 || target_value->as_integer() >= 262144) {
        return FailCheck("official Q4 target token is invalid", 6);
      }
      const std::uint32_t target =
          static_cast<std::uint32_t>(target_value->as_integer());
      auto prediction = engine.value().Prediction();
      if (!prediction.ok()) return Fail("read M19 prediction", prediction.status(), 6);
      status = engine.value().CopyLogits(logits);
      if (!status.ok()) return Fail("copy M19 logits", status, 6);
      auto q4_logprob = Q4SelectedLogProbability(q4_step, target);
      if (!q4_logprob.ok()) return Fail("read official Q4 logprob", q4_logprob.status(), 6);
      const double candidate_logprob = LogProbability(logits, target);
      const double delta = std::abs(candidate_logprob - q4_logprob.value());
      result.finite = result.finite && prediction.value().all_logits_finite &&
                      std::isfinite(candidate_logprob);
      result.top1 += prediction.value().token == target ? 1U : 0U;
      result.top5 += InTopFive(logits, target) ? 1U : 0U;
      result.sum_delta += delta;
      result.max_delta = std::max(result.max_delta, delta);
      ++result.scored;
      ++result.logprob_compared;
      targets.push_back(target);
      candidate_capture += std::to_string(target) + ":" +
                           std::to_string(prediction.value().token) + ":" +
                           std::to_string(candidate_logprob) + "\n";
      if (step_index + 1U < options.scored_tokens) {
        status = engine.value().ForwardToken(target);
        if (!status.ok()) return Fail("teacher-force M19 candidate", status, 6);
      }
    }
    result.target_hash = TokenHash(targets);
    result.candidate_capture_hash = gem16::compiler::Sha256Hex(
        candidate_capture.data(), candidate_capture.size());

    status = engine.value().Reset();
    if (!status.ok()) return Fail("reset M19 generation", status, 7);
    status = engine.value().ConfigureTokenSelection(
        {}, processor.value().generation_controls().suppressed_token_ids);
    if (!status.ok()) return Fail("configure M19 generation", status, 7);
    if (options.backend == gem16::internal::Gemma4Moe26BBackend::kReference ||
        options.tokenwise_prefill) {
      std::size_t prompt_index = 0U;
      for (const std::uint32_t token : prompt.value()) {
        status = engine.value().ForwardToken(token);
        if (!status.ok()) break;
        if (options.tokenwise_prefill) {
          auto prediction = engine.value().Prediction();
          if (!prediction.ok()) {
            std::cerr << "M19 generation prefill failed after prompt token "
                      << prompt_index << " (id " << token << ")\n";
            return Fail("inspect M19 generation prefill", prediction.status(),
                        7);
          }
        }
        ++prompt_index;
      }
    } else {
      status = engine.value().PrefillTokens(prompt.value());
    }
    if (!status.ok()) return Fail("prefill M19 generation", status, 7);
    std::vector<std::uint32_t> generated;
    generated.reserve(static_cast<std::size_t>(options.generated_tokens));
    for (std::uint64_t index = 0U; index < options.generated_tokens; ++index) {
      auto selected = engine.value().SelectToken();
      if (!selected.ok()) return Fail("select M19 generated token", selected.status(), 7);
      generated.push_back(selected.value());
      if (std::find(processor.value().generation_controls().stop_token_ids.begin(),
                    processor.value().generation_controls().stop_token_ids.end(),
                    selected.value()) !=
          processor.value().generation_controls().stop_token_ids.end()) {
        break;
      }
      if (index + 1U < options.generated_tokens) {
        status = engine.value().ForwardToken(selected.value());
        if (!status.ok()) return Fail("decode M19 candidate", status, 7);
      }
    }
    std::vector<std::uint32_t> content = generated;
    if (!content.empty() &&
        std::find(processor.value().generation_controls().stop_token_ids.begin(),
                  processor.value().generation_controls().stop_token_ids.end(),
                  content.back()) !=
            processor.value().generation_controls().stop_token_ids.end()) {
      content.pop_back();
    }
    // Keep incomplete reasoning/channel markers visible for the blind quality
    // review instead of turning a truncated response into a runner failure.
    auto decoded = processor.value().Decode(content, false);
    if (!decoded.ok()) return Fail("decode M19 candidate response", decoded.status(), 7);
    result.candidate_text = std::move(decoded).value();
    result.candidate_response_hash = gem16::compiler::Sha256Hex(
        result.candidate_text.data(), result.candidate_text.size());
    results.push_back(std::move(result));
  }
  if (results.empty()) return FailCheck("selected M19 record was not found", 7);

  std::ofstream output(options.output, std::ios::binary | std::ios::trunc);
  if (!output) return FailCheck("cannot open M19 output", 8);
  output << "{\n  \"schema_version\": 1,\n"
         << "  \"kind\": \"gemma4_26b_m19_numerical\",\n"
         << "  \"status\": \"complete\",\n"
         << "  \"artifact_content_sha256\": "
         << gem16::json::Quote(artifact_hash->as_string()) << ",\n"
         << "  \"runtime_profile\": \"native_sm120_integrated_prefill_decode_head\",\n"
         << "  \"corpus_sha256\": " << gem16::json::Quote(corpus_hash) << ",\n"
         << "  \"reference\": "
         << gem16::json::Stringify(*reference_identity) << ",\n"
         << "  \"records\": [\n";
  for (std::size_t index = 0U; index < results.size(); ++index) {
    const auto& row = results[index];
    output << "    {\"id\":" << gem16::json::Quote(row.id)
           << ",\"category\":" << gem16::json::Quote(row.category)
           << ",\"input_token_ids_sha256_u32le\":"
           << gem16::json::Quote(row.input_hash)
           << ",\"target_token_ids_sha256_u32le\":"
           << gem16::json::Quote(row.target_hash)
           << ",\"target_token_source\":\"official_q4_greedy_seed_0\""
           << ",\"q4_reference_capture_sha256\":"
           << gem16::json::Quote(row.reference_capture_hash)
           << ",\"candidate_capture_sha256\":"
           << gem16::json::Quote(row.candidate_capture_hash)
           << ",\"candidate_response_sha256\":"
           << gem16::json::Quote(row.candidate_response_hash)
           << ",\"candidate_response_text\":"
           << gem16::json::Quote(row.candidate_text)
           << ",\"all_logits_finite\":" << (row.finite ? "true" : "false")
           << ",\"teacher_forced\":{\"scored_token_count\":" << row.scored
           << ",\"q4_reference_token_candidate_top5_fraction\":"
           << static_cast<double>(row.top5) / static_cast<double>(row.scored)
           << ",\"top1_agreement_fraction\":"
           << static_cast<double>(row.top1) / static_cast<double>(row.scored)
           << ",\"selected_logprob_compared_fraction\":"
           << static_cast<double>(row.logprob_compared) /
                  static_cast<double>(row.scored)
           << ",\"mean_selected_logprob_absolute_delta\":"
           << row.sum_delta / static_cast<double>(row.scored)
           << ",\"maximum_selected_logprob_absolute_delta\":"
           << row.max_delta << "}}" << (index + 1U == results.size() ? "\n" : ",\n");
  }
  output << "  ]\n}\n";
  return output ? 0 : 8;
}
