#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "compiler/sha256.h"
#include "cuda/engine/gemma4_26b_reference.h"
#include "gem16/image.h"
#include "gem16/tokenizer.h"
#include "model/gemma4_26b_vision_d2_diagnostic.h"
#include "util/json.h"

namespace {

constexpr std::uint64_t kContext = 32768U;
constexpr std::uint32_t kImageBegin = 255999U;
constexpr std::uint32_t kImagePlaceholder = 258880U;
constexpr std::uint32_t kImageEnd = 258882U;

struct Fixture {
  gem16::Gemma4Moe26BVisionImage image;
  std::vector<std::uint32_t> prompt;
  std::uint64_t vision_offset = 0U;
};

struct StateDigest {
  gem16::internal::Gemma4Moe26BVisionD2DiagnosticState state;
  std::string final_hidden_sha256;
  std::string visible_kv_sha256;
};

using LayerOutputHashes =
    std::array<std::array<std::string, 30U>, 3U>;

struct OrdinaryResult {
  std::array<std::uint32_t, 4U> tokens{};
  std::uint64_t position_before_first_proposal = 0U;
  StateDigest pre_proposal;
  std::array<StateDigest, 3U> post_forward;
  LayerOutputHashes layer_outputs;
};

struct GroupRun {
  std::uint32_t initial = 0U;
  gem16::internal::MtpGroupResult group;
  std::uint64_t position_before_first_proposal = 0U;
  std::uint64_t position_after_group = 0U;
  StateDigest pre_proposal;
  StateDigest post_group;
  LayerOutputHashes layer_outputs;
};

struct TrajectoryCheckpoint {
  gem16::internal::MtpGroupResult group;
  StateDigest state;
  std::uint64_t processed_tokens = 0U;
};

struct TrajectoryRun {
  std::vector<std::uint32_t> tokens;
  std::vector<TrajectoryCheckpoint> checkpoints;
  StateDigest pre_proposal;
  LayerOutputHashes first_group_layer_outputs;
  bool first_group_layer_outputs_captured = false;
  std::uint64_t position_before_first_proposal = 0U;
};

struct MatrixCaseResult {
  std::string name;
  std::string sampling;
  std::uint64_t context = 0U;
  std::uint64_t prompt_tokens = 0U;
  std::uint64_t vision_offset = 0U;
  std::uint64_t vision_end = 0U;
  TrajectoryRun ordinary;
  TrajectoryRun d2;
};

struct ContinuationRun {
  std::vector<std::uint32_t> first_turn_tokens;
  TrajectoryRun continuation;
};

struct ContinuationCaseResult {
  ContinuationRun ordinary;
  ContinuationRun d2;
};

struct CancellationCaseResult {
  bool vision_cancel_requested_during_prefill = false;
  bool vision_prefill_completed_safely = false;
  bool vision_d2_not_started = false;
  bool d2_cancelled = false;
  std::uint64_t d2_callback_count = 0U;
  std::uint64_t d2_output_count = 0U;
  std::uint64_t d2_group_count = 0U;
};

gem16::Result<std::vector<std::byte>> ReadRegular(
    const std::filesystem::path& path) {
  std::error_code error;
  const auto status = std::filesystem::symlink_status(path, error);
  if (error || std::filesystem::is_symlink(status) ||
      !std::filesystem::is_regular_file(status)) {
    return gem16::Status(gem16::StatusCode::kInvalidArgument,
                         "V11 identity input is not a regular file: " +
                             path.string());
  }
  const std::uintmax_t bytes = std::filesystem::file_size(path, error);
  if (error || bytes > std::numeric_limits<std::size_t>::max()) {
    return gem16::Status(gem16::StatusCode::kDataLoss,
                         "V11 identity input has an invalid size: " +
                             path.string());
  }
  std::vector<std::byte> result(static_cast<std::size_t>(bytes));
  std::ifstream input(path, std::ios::binary);
  input.read(reinterpret_cast<char*>(result.data()),
             static_cast<std::streamsize>(result.size()));
  if (!input || input.gcount() !=
                    static_cast<std::streamsize>(result.size())) {
    return gem16::Status(gem16::StatusCode::kIoError,
                         "cannot read V11 identity input: " + path.string());
  }
  return result;
}

gem16::Result<std::string> HashFile(const std::filesystem::path& path) {
  auto bytes = ReadRegular(path);
  if (!bytes.ok()) return bytes.status();
  return gem16::compiler::Sha256Hex(bytes.value().data(),
                                    bytes.value().size());
}

gem16::Result<Fixture> BuildFixture(const std::filesystem::path& target,
                                    const std::filesystem::path& image_path,
                                    std::uint32_t budget) {
  auto image = gem16::LoadGemma4Moe26BVisionImage(
      image_path, gem16::Gemma4Moe26BVisionImageOptions{budget});
  if (!image.ok()) return image.status();
  auto processor = gem16::GemmaChatProcessor::Load(target);
  if (!processor.ok()) return processor.status();
  std::string content = "<|image>";
  for (std::uint32_t index = 0U; index < image.value().soft_token_count;
       ++index) {
    content.append("<|image|>");
  }
  content.append("<image|>Describe the image precisely.");
  gem16::ChatMessage message;
  message.role = "user";
  message.content = std::move(content);
  const std::array messages{std::move(message)};
  auto prompt = processor.value().Encode(messages, false, true);
  if (!prompt.ok()) return prompt.status();
  const auto begin = std::find(prompt.value().begin(), prompt.value().end(),
                               kImageBegin);
  if (begin == prompt.value().end()) {
    return gem16::Status(gem16::StatusCode::kDataLoss,
                         "V11 prompt is missing the image-begin token");
  }
  const std::uint64_t offset =
      static_cast<std::uint64_t>(begin - prompt.value().begin()) + 1U;
  if (image.value().soft_token_count > prompt.value().size() - offset ||
      !std::all_of(prompt.value().begin() + offset,
                   prompt.value().begin() + offset +
                       image.value().soft_token_count,
                   [](std::uint32_t token) {
                     return token == kImagePlaceholder;
                   }) ||
      offset + image.value().soft_token_count >= prompt.value().size() ||
      prompt.value()[offset + image.value().soft_token_count] != kImageEnd) {
    return gem16::Status(gem16::StatusCode::kDataLoss,
                         "V11 rendered image span is inconsistent");
  }
  return Fixture{std::move(image).value(), std::move(prompt).value(), offset};
}

gem16::Status ShapeFixture(Fixture* fixture,
                           std::uint64_t target_vision_offset,
                           std::uint64_t target_prompt_tokens) {
  if (fixture == nullptr || fixture->prompt.empty() ||
      fixture->vision_offset == 0U ||
      target_vision_offset < fixture->vision_offset ||
      target_vision_offset > std::numeric_limits<std::size_t>::max() ||
      target_prompt_tokens > std::numeric_limits<std::size_t>::max()) {
    return gem16::Status(gem16::StatusCode::kInvalidArgument,
                         "V11 fixture-shaping request is invalid");
  }
  const std::uint64_t image_end =
      fixture->vision_offset + fixture->image.soft_token_count;
  if (image_end + 1U >= fixture->prompt.size()) {
    return gem16::Status(gem16::StatusCode::kDataLoss,
                         "V11 fixture has no stable text filler token");
  }
  const std::uint32_t filler = fixture->prompt[image_end + 1U];
  const std::uint64_t prefix_count =
      target_vision_offset - fixture->vision_offset;
  fixture->prompt.insert(
      fixture->prompt.begin() +
          static_cast<std::ptrdiff_t>(fixture->vision_offset - 1U),
      static_cast<std::size_t>(prefix_count), filler);
  fixture->vision_offset += prefix_count;
  if (target_prompt_tokens != 0U) {
    if (target_prompt_tokens < fixture->prompt.size()) {
      return gem16::Status(
          gem16::StatusCode::kInvalidArgument,
          "V11 target prompt length is shorter than the shaped fixture");
    }
    const std::uint64_t shaped_image_end =
        fixture->vision_offset + fixture->image.soft_token_count;
    fixture->prompt.insert(
        fixture->prompt.begin() +
            static_cast<std::ptrdiff_t>(shaped_image_end + 1U),
        static_cast<std::size_t>(target_prompt_tokens - fixture->prompt.size()),
        filler);
  }
  return gem16::Status::Ok();
}

gem16::Gemma4Moe26BVisionInputSegment Segment(
    const Fixture& fixture, std::uint64_t vision_offset) {
  return {vision_offset,
          fixture.image.soft_token_count,
          fixture.image.soft_token_budget,
          fixture.image.raw_patch_count,
          fixture.image.patches,
          fixture.image.positions};
}

gem16::Status Configure(
    gem16::internal::Gemma4Moe26BReferenceEngine& engine,
    const gem16::GemmaChatProcessor& processor,
    const gem16::SamplingOptions& sampling = {}) {
  return engine.ConfigureTokenSelection(
      sampling, processor.generation_controls().suppressed_token_ids);
}

gem16::Result<StateDigest> CaptureState(
    gem16::internal::Gemma4Moe26BReferenceEngine& engine) {
  auto kv_bytes = engine.VisionD2PostPrefillDiagnosticKvBytes();
  if (!kv_bytes.ok()) return kv_bytes.status();
  if (kv_bytes.value() > std::numeric_limits<std::size_t>::max()) {
    return gem16::Status(gem16::StatusCode::kResourceExhausted,
                         "V11 visible-KV capture exceeds host size_t");
  }
  std::vector<float> final_hidden(2816U);
  std::vector<std::uint8_t> kv(static_cast<std::size_t>(kv_bytes.value()));
  StateDigest result;
  gem16::Status status = engine.CopyVisionD2PostPrefillDiagnostic(
      final_hidden, kv, &result.state);
  if (!status.ok()) return status;
  result.final_hidden_sha256 = gem16::compiler::Sha256Hex(
      final_hidden.data(), final_hidden.size() * sizeof(float));
  result.visible_kv_sha256 =
      gem16::compiler::Sha256Hex(kv.data(), kv.size());
  return result;
}

gem16::Result<std::array<std::string, 30U>> CaptureOrdinaryLayerOutputs(
    gem16::internal::Gemma4Moe26BReferenceEngine& engine) {
  std::array<std::string, 30U> result;
  std::vector<float> row(2816U);
  for (std::uint32_t layer = 0U; layer < result.size(); ++layer) {
    gem16::Status status = engine.CopyLayerOutput(layer, row);
    if (!status.ok()) return status;
    result[layer] = gem16::compiler::Sha256Hex(
        row.data(), row.size() * sizeof(float));
  }
  return result;
}

gem16::Result<LayerOutputHashes> CaptureT3LayerOutputs(
    gem16::internal::Gemma4Moe26BReferenceEngine& engine) {
  LayerOutputHashes result;
  std::vector<float> rows(3U * 2816U);
  for (std::uint32_t layer = 0U; layer < 30U; ++layer) {
    gem16::Status status =
        engine.CopyVisionD2T3LayerOutputDiagnostic(layer, rows);
    if (!status.ok()) return status;
    for (std::uint32_t row = 0U; row < 3U; ++row) {
      result[row][layer] = gem16::compiler::Sha256Hex(
          rows.data() + static_cast<std::size_t>(row) * 2816U,
          2816U * sizeof(float));
    }
  }
  return result;
}

gem16::Result<OrdinaryResult> RunOrdinary(
    const std::filesystem::path& target,
    const std::filesystem::path& vision,
    const Fixture& fixture,
    const gem16::GemmaChatProcessor& processor) {
  auto engine = gem16::internal::Gemma4Moe26BReferenceEngine::Create(
      target, kContext, 0,
      gem16::internal::Gemma4Moe26BBackend::kSm120Integrated, true, true,
      std::nullopt, vision);
  if (!engine.ok()) return engine.status();
  gem16::Status status = Configure(engine.value(), processor);
  if (!status.ok()) return status;
  status = engine.value().PrefillTokensWithVision(fixture.prompt,
                                                   Segment(fixture, fixture.vision_offset));
  if (!status.ok()) return status;
  OrdinaryResult result;
  result.position_before_first_proposal = engine.value().position();
  auto selected = engine.value().SelectToken();
  if (!selected.ok()) return selected.status();
  result.tokens[0] = selected.value();
  auto capture = CaptureState(engine.value());
  if (!capture.ok()) return capture.status();
  result.pre_proposal = std::move(capture).value();
  for (std::size_t index = 1U; index < result.tokens.size(); ++index) {
    status = engine.value().ForwardToken(result.tokens[index - 1U]);
    if (!status.ok()) return status;
    selected = engine.value().SelectToken();
    if (!selected.ok()) return selected.status();
    result.tokens[index] = selected.value();
    status = engine.value().RefreshVisionD2FinalHiddenDiagnostic();
    if (!status.ok()) return status;
    auto layer_outputs = CaptureOrdinaryLayerOutputs(engine.value());
    if (!layer_outputs.ok()) return layer_outputs.status();
    result.layer_outputs[index - 1U] = std::move(layer_outputs).value();
    capture = CaptureState(engine.value());
    if (!capture.ok()) return capture.status();
    result.post_forward[index - 1U] = std::move(capture).value();
  }
  return result;
}

gem16::Result<GroupRun> RunGroup(
    const std::filesystem::path& target,
    const std::filesystem::path& assistant,
    const std::filesystem::path& vision,
    const Fixture& fixture,
    const gem16::GemmaChatProcessor& processor,
    std::span<const std::uint32_t> forced) {
  auto engine = gem16::internal::Gemma4Moe26BReferenceEngine::Create(
      target, kContext, 0,
      gem16::internal::Gemma4Moe26BBackend::kSm120Integrated, true, true,
      std::nullopt, vision);
  if (!engine.ok()) return engine.status();
  gem16::Status status = Configure(engine.value(), processor);
  if (!status.ok()) return status;
  status = engine.value().LoadMtpAssistant(assistant);
  if (!status.ok()) return status;
  status = engine.value().ConfigureMtpStopTokens(
      processor.generation_controls().stop_token_ids);
  if (!status.ok()) return status;
  status = engine.value().ConfigureMtpVerifierBackend(
      gem16::internal::Gemma4Moe26BMtpVerifierBackend::
          kExactSharedBatchedMoe);
  if (!status.ok()) return status;
  status = engine.value().PrefillTokensWithVision(fixture.prompt,
                                                   Segment(fixture, fixture.vision_offset));
  if (!status.ok()) return status;
  GroupRun result;
  result.position_before_first_proposal = engine.value().position();
  auto selected = engine.value().SelectToken();
  if (!selected.ok()) return selected.status();
  result.initial = selected.value();
  auto capture = CaptureState(engine.value());
  if (!capture.ok()) return capture.status();
  result.pre_proposal = std::move(capture).value();
  status = forced.empty()
               ? engine.value().RunMtpAssistantGroup(result.initial, 2U,
                                                     &result.group)
               : engine.value().RunForcedMtpProposalDiagnostic(
                     result.initial, forced, &result.group);
  if (!status.ok()) return status;
  auto layer_outputs = CaptureT3LayerOutputs(engine.value());
  if (!layer_outputs.ok()) return layer_outputs.status();
  result.layer_outputs = std::move(layer_outputs).value();
  result.position_after_group = engine.value().position();
  capture = CaptureState(engine.value());
  if (!capture.ok()) return capture.status();
  result.post_group = std::move(capture).value();
  return result;
}

gem16::Result<TrajectoryRun> RunRealTrajectory(
    const std::filesystem::path& target,
    const std::filesystem::path& assistant,
    const std::filesystem::path& vision,
    const Fixture& fixture,
    const gem16::GemmaChatProcessor& processor,
    std::uint32_t group_count,
    const gem16::SamplingOptions& sampling = {},
    std::uint64_t context = kContext) {
  auto engine = gem16::internal::Gemma4Moe26BReferenceEngine::Create(
      target, context, 0,
      gem16::internal::Gemma4Moe26BBackend::kSm120Integrated, true, true,
      std::nullopt, vision);
  if (!engine.ok()) return engine.status();
  gem16::Status status = Configure(engine.value(), processor, sampling);
  if (!status.ok()) return status;
  status = engine.value().LoadMtpAssistant(assistant);
  if (!status.ok()) return status;
  status = engine.value().ConfigureMtpStopTokens({});
  if (!status.ok()) return status;
  status = engine.value().ConfigureMtpVerifierBackend(
      gem16::internal::Gemma4Moe26BMtpVerifierBackend::
          kExactSharedBatchedMoe);
  if (!status.ok()) return status;
  status = engine.value().PrefillTokensWithVision(
      fixture.prompt, Segment(fixture, fixture.vision_offset));
  if (!status.ok()) return status;

  TrajectoryRun result;
  result.position_before_first_proposal = engine.value().position();
  auto selected = engine.value().SelectToken();
  if (!selected.ok()) return selected.status();
  result.tokens.push_back(selected.value());
  auto captured = CaptureState(engine.value());
  if (!captured.ok()) return captured.status();
  result.pre_proposal = std::move(captured).value();
  std::uint32_t pending = selected.value();
  std::uint64_t processed = 0U;
  result.checkpoints.reserve(group_count);
  for (std::uint32_t group_index = 0U; group_index < group_count;
       ++group_index) {
    TrajectoryCheckpoint checkpoint;
    status = engine.value().RunMtpAssistantGroup(
        pending, 2U, &checkpoint.group);
    if (!status.ok()) return status;
    if (checkpoint.group.output_count == 0U ||
        checkpoint.group.output_count > 3U) {
      return gem16::Status(
          gem16::StatusCode::kDataLoss,
          "V11 real Assistant returned an invalid trajectory group");
    }
    result.tokens.insert(
        result.tokens.end(), checkpoint.group.verified.begin(),
        checkpoint.group.verified.begin() + checkpoint.group.output_count);
    pending = checkpoint.group.verified[checkpoint.group.output_count - 1U];
    processed += checkpoint.group.output_count;
    checkpoint.processed_tokens = processed;
    if (group_index == 0U) {
      auto layer_outputs = CaptureT3LayerOutputs(engine.value());
      if (!layer_outputs.ok()) return layer_outputs.status();
      result.first_group_layer_outputs = std::move(layer_outputs).value();
      result.first_group_layer_outputs_captured = true;
    }
    captured = CaptureState(engine.value());
    if (!captured.ok()) return captured.status();
    checkpoint.state = std::move(captured).value();
    result.checkpoints.push_back(std::move(checkpoint));
  }
  return result;
}

gem16::Result<TrajectoryRun> RunOrdinaryTrajectory(
    const std::filesystem::path& target,
    const std::filesystem::path& vision,
    const Fixture& fixture,
    const gem16::GemmaChatProcessor& processor,
    std::span<const TrajectoryCheckpoint> checkpoints,
    const gem16::SamplingOptions& sampling = {},
    std::uint64_t context = kContext) {
  if (checkpoints.empty()) {
    return gem16::Status(gem16::StatusCode::kInvalidArgument,
                         "V11 ordinary trajectory requires checkpoints");
  }
  auto engine = gem16::internal::Gemma4Moe26BReferenceEngine::Create(
      target, context, 0,
      gem16::internal::Gemma4Moe26BBackend::kSm120Integrated, true, true,
      std::nullopt, vision);
  if (!engine.ok()) return engine.status();
  gem16::Status status = Configure(engine.value(), processor, sampling);
  if (!status.ok()) return status;
  status = engine.value().PrefillTokensWithVision(
      fixture.prompt, Segment(fixture, fixture.vision_offset));
  if (!status.ok()) return status;

  TrajectoryRun result;
  result.position_before_first_proposal = engine.value().position();
  auto selected = engine.value().SelectToken();
  if (!selected.ok()) return selected.status();
  result.tokens.push_back(selected.value());
  auto captured = CaptureState(engine.value());
  if (!captured.ok()) return captured.status();
  result.pre_proposal = std::move(captured).value();
  std::uint32_t pending = selected.value();
  std::uint64_t processed = 0U;
  result.checkpoints.reserve(checkpoints.size());
  for (std::size_t checkpoint_index = 0U;
       checkpoint_index < checkpoints.size(); ++checkpoint_index) {
    const TrajectoryCheckpoint& expected = checkpoints[checkpoint_index];
    if (expected.processed_tokens <= processed) {
      return gem16::Status(
          gem16::StatusCode::kInvalidArgument,
          "V11 trajectory checkpoints are not strictly increasing");
    }
    while (processed < expected.processed_tokens) {
      status = engine.value().ForwardToken(pending);
      if (!status.ok()) return status;
      selected = engine.value().SelectToken();
      if (!selected.ok()) return selected.status();
      pending = selected.value();
      result.tokens.push_back(pending);
      ++processed;
    }
    status = engine.value().RefreshVisionD2FinalHiddenDiagnostic();
    if (!status.ok()) return status;
    if (checkpoint_index == 0U) {
      auto layer_outputs = CaptureOrdinaryLayerOutputs(engine.value());
      if (!layer_outputs.ok()) return layer_outputs.status();
      const std::uint32_t row = expected.group.output_count - 1U;
      result.first_group_layer_outputs[row] =
          std::move(layer_outputs).value();
      result.first_group_layer_outputs_captured = true;
    }
    TrajectoryCheckpoint checkpoint;
    checkpoint.processed_tokens = processed;
    captured = CaptureState(engine.value());
    if (!captured.ok()) return captured.status();
    checkpoint.state = std::move(captured).value();
    result.checkpoints.push_back(std::move(checkpoint));
  }
  return result;
}

gem16::Result<MatrixCaseResult> RunMatrixCase(
    std::string name,
    std::string sampling_name,
    const std::filesystem::path& target,
    const std::filesystem::path& assistant,
    const std::filesystem::path& vision,
    const Fixture& fixture,
    const gem16::GemmaChatProcessor& processor,
    std::uint32_t group_count,
    const gem16::SamplingOptions& sampling,
    std::uint64_t context) {
  auto d2 = RunRealTrajectory(target, assistant, vision, fixture, processor,
                              group_count, sampling, context);
  if (!d2.ok()) return d2.status();
  auto ordinary = RunOrdinaryTrajectory(
      target, vision, fixture, processor, d2.value().checkpoints, sampling,
      context);
  if (!ordinary.ok()) return ordinary.status();
  MatrixCaseResult result;
  result.name = std::move(name);
  result.sampling = std::move(sampling_name);
  result.context = context;
  result.prompt_tokens = fixture.prompt.size();
  result.vision_offset = fixture.vision_offset;
  result.vision_end =
      fixture.vision_offset + fixture.image.soft_token_count;
  result.ordinary = std::move(ordinary).value();
  result.d2 = std::move(d2).value();
  return result;
}

gem16::Result<ContinuationRun> RunRealContinuation(
    const std::filesystem::path& target,
    const std::filesystem::path& assistant,
    const std::filesystem::path& vision,
    const Fixture& fixture,
    const gem16::GemmaChatProcessor& processor) {
  auto engine = gem16::internal::Gemma4Moe26BReferenceEngine::Create(
      target, kContext, 0,
      gem16::internal::Gemma4Moe26BBackend::kSm120Integrated, true, true,
      std::nullopt, vision);
  if (!engine.ok()) return engine.status();
  gem16::Status status = Configure(engine.value(), processor);
  if (!status.ok()) return status;
  status = engine.value().LoadMtpAssistant(assistant);
  if (!status.ok()) return status;
  status = engine.value().ConfigureMtpStopTokens({});
  if (!status.ok()) return status;
  status = engine.value().ConfigureMtpVerifierBackend(
      gem16::internal::Gemma4Moe26BMtpVerifierBackend::
          kExactSharedBatchedMoe);
  if (!status.ok()) return status;
  status = engine.value().PrefillTokensWithVision(
      fixture.prompt, Segment(fixture, fixture.vision_offset));
  if (!status.ok()) return status;

  ContinuationRun result;
  auto selected = engine.value().SelectToken();
  if (!selected.ok()) return selected.status();
  std::uint32_t pending = selected.value();
  result.first_turn_tokens.push_back(pending);
  for (std::uint32_t index = 0U; index < 2U; ++index) {
    gem16::internal::MtpGroupResult group;
    status = engine.value().RunMtpAssistantGroup(pending, 2U, &group);
    if (!status.ok()) return status;
    result.first_turn_tokens.insert(
        result.first_turn_tokens.end(), group.verified.begin(),
        group.verified.begin() + group.output_count);
    pending = group.verified[group.output_count - 1U];
  }
  const std::uint64_t image_end =
      fixture.vision_offset + fixture.image.soft_token_count;
  const std::uint32_t filler = fixture.prompt[image_end + 1U];
  std::array<std::uint32_t, 8U> continuation_tokens;
  continuation_tokens.fill(filler);
  continuation_tokens.front() = pending;
  status = engine.value().PrefillTokens(continuation_tokens);
  if (!status.ok()) return status;
  result.continuation.position_before_first_proposal =
      engine.value().position();
  selected = engine.value().SelectToken();
  if (!selected.ok()) return selected.status();
  pending = selected.value();
  result.continuation.tokens.push_back(pending);
  auto captured = CaptureState(engine.value());
  if (!captured.ok()) return captured.status();
  result.continuation.pre_proposal = std::move(captured).value();
  TrajectoryCheckpoint checkpoint;
  status = engine.value().RunMtpAssistantGroup(
      pending, 2U, &checkpoint.group);
  if (!status.ok()) return status;
  result.continuation.tokens.insert(
      result.continuation.tokens.end(), checkpoint.group.verified.begin(),
      checkpoint.group.verified.begin() + checkpoint.group.output_count);
  checkpoint.processed_tokens = checkpoint.group.output_count;
  captured = CaptureState(engine.value());
  if (!captured.ok()) return captured.status();
  checkpoint.state = std::move(captured).value();
  result.continuation.checkpoints.push_back(std::move(checkpoint));
  return result;
}

gem16::Result<ContinuationRun> RunOrdinaryContinuation(
    const std::filesystem::path& target,
    const std::filesystem::path& vision,
    const Fixture& fixture,
    const gem16::GemmaChatProcessor& processor,
    const ContinuationRun& d2) {
  if (d2.first_turn_tokens.size() < 2U ||
      d2.continuation.checkpoints.size() != 1U) {
    return gem16::Status(gem16::StatusCode::kInvalidArgument,
                         "V11 continuation oracle is incomplete");
  }
  auto engine = gem16::internal::Gemma4Moe26BReferenceEngine::Create(
      target, kContext, 0,
      gem16::internal::Gemma4Moe26BBackend::kSm120Integrated, true, true,
      std::nullopt, vision);
  if (!engine.ok()) return engine.status();
  gem16::Status status = Configure(engine.value(), processor);
  if (!status.ok()) return status;
  status = engine.value().PrefillTokensWithVision(
      fixture.prompt, Segment(fixture, fixture.vision_offset));
  if (!status.ok()) return status;

  ContinuationRun result;
  auto selected = engine.value().SelectToken();
  if (!selected.ok()) return selected.status();
  std::uint32_t pending = selected.value();
  result.first_turn_tokens.push_back(pending);
  while (result.first_turn_tokens.size() < d2.first_turn_tokens.size()) {
    status = engine.value().ForwardToken(pending);
    if (!status.ok()) return status;
    selected = engine.value().SelectToken();
    if (!selected.ok()) return selected.status();
    pending = selected.value();
    result.first_turn_tokens.push_back(pending);
  }
  const std::uint64_t image_end =
      fixture.vision_offset + fixture.image.soft_token_count;
  const std::uint32_t filler = fixture.prompt[image_end + 1U];
  std::array<std::uint32_t, 8U> continuation_tokens;
  continuation_tokens.fill(filler);
  continuation_tokens.front() = pending;
  status = engine.value().PrefillTokens(continuation_tokens);
  if (!status.ok()) return status;
  result.continuation.position_before_first_proposal =
      engine.value().position();
  selected = engine.value().SelectToken();
  if (!selected.ok()) return selected.status();
  pending = selected.value();
  result.continuation.tokens.push_back(pending);
  auto captured = CaptureState(engine.value());
  if (!captured.ok()) return captured.status();
  result.continuation.pre_proposal = std::move(captured).value();
  const std::uint32_t outputs =
      d2.continuation.checkpoints.front().group.output_count;
  for (std::uint32_t index = 0U; index < outputs; ++index) {
    status = engine.value().ForwardToken(pending);
    if (!status.ok()) return status;
    selected = engine.value().SelectToken();
    if (!selected.ok()) return selected.status();
    pending = selected.value();
    result.continuation.tokens.push_back(pending);
  }
  status = engine.value().RefreshVisionD2FinalHiddenDiagnostic();
  if (!status.ok()) return status;
  TrajectoryCheckpoint checkpoint;
  checkpoint.processed_tokens = outputs;
  captured = CaptureState(engine.value());
  if (!captured.ok()) return captured.status();
  checkpoint.state = std::move(captured).value();
  result.continuation.checkpoints.push_back(std::move(checkpoint));
  return result;
}

gem16::Result<ContinuationCaseResult> RunContinuationCase(
    const std::filesystem::path& target,
    const std::filesystem::path& assistant,
    const std::filesystem::path& vision,
    const Fixture& fixture,
    const gem16::GemmaChatProcessor& processor) {
  auto d2 = RunRealContinuation(target, assistant, vision, fixture, processor);
  if (!d2.ok()) return d2.status();
  auto ordinary = RunOrdinaryContinuation(target, vision, fixture, processor,
                                           d2.value());
  if (!ordinary.ok()) return ordinary.status();
  return ContinuationCaseResult{std::move(ordinary).value(),
                                std::move(d2).value()};
}

gem16::Status CancelFirstD2Token(void* context, std::uint32_t) {
  auto* count = static_cast<std::uint64_t*>(context);
  ++*count;
  return gem16::Status(gem16::StatusCode::kCancelled,
                       "intentional V11 D2 cancellation");
}

gem16::Result<CancellationCaseResult> RunCancellationCase(
    const std::filesystem::path& target,
    const std::filesystem::path& assistant,
    const std::filesystem::path& vision,
    const Fixture& fixture,
    const gem16::GemmaChatProcessor& processor) {
  CancellationCaseResult result;
  {
    auto engine = gem16::internal::Gemma4Moe26BReferenceEngine::Create(
        target, kContext, 0,
        gem16::internal::Gemma4Moe26BBackend::kSm120Integrated, true, false,
        std::nullopt, vision);
    if (!engine.ok()) return engine.status();
    gem16::Status status = Configure(engine.value(), processor);
    if (!status.ok()) return status;
    std::atomic<bool> started{false};
    std::atomic<bool> finished{false};
    gem16::Status prefill_status(
        gem16::StatusCode::kInternal,
        "V11 Vision cancellation worker did not run");
    std::thread worker([&] {
      started.store(true, std::memory_order_release);
      prefill_status = engine.value().PrefillTokensWithVision(
          fixture.prompt, Segment(fixture, fixture.vision_offset));
      finished.store(true, std::memory_order_release);
    });
    while (!started.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    result.vision_cancel_requested_during_prefill =
        !finished.load(std::memory_order_acquire);
    worker.join();
    if (!prefill_status.ok()) return prefill_status;
    result.vision_prefill_completed_safely = true;
    // A cancellation latched during the synchronous tower discards this
    // engine before selection, Assistant proposal, or Target verification.
    result.vision_d2_not_started =
        engine.value().mtp_group_graph_launches() == 0U;
  }
  {
    auto engine = gem16::internal::Gemma4Moe26BReferenceEngine::Create(
        target, kContext, 0,
        gem16::internal::Gemma4Moe26BBackend::kSm120Integrated, true, false,
        std::nullopt, vision);
    if (!engine.ok()) return engine.status();
    gem16::Status status = Configure(engine.value(), processor);
    if (!status.ok()) return status;
    status = engine.value().LoadMtpAssistant(assistant);
    if (!status.ok()) return status;
    status = engine.value().ConfigureMtpStopTokens({});
    if (!status.ok()) return status;
    status = engine.value().ConfigureMtpVerifierBackend(
        gem16::internal::Gemma4Moe26BMtpVerifierBackend::
            kExactSharedBatchedMoe);
    if (!status.ok()) return status;
    status = engine.value().PrefillTokensWithVision(
        fixture.prompt, Segment(fixture, fixture.vision_offset));
    if (!status.ok()) return status;
    auto selected = engine.value().SelectToken();
    if (!selected.ok()) return selected.status();
    std::array<std::uint32_t, 12U> output{};
    gem16::internal::MtpChainResult chain;
    status = engine.value().RunFixedMtpGraphChain(
        selected.value(), 2U, output, &chain, CancelFirstD2Token,
        &result.d2_callback_count);
    result.d2_cancelled =
        !status.ok() && status.code() == gem16::StatusCode::kCancelled;
    result.d2_output_count = chain.output_count;
    result.d2_group_count = chain.group_count;
    if (!result.d2_cancelled) {
      return status.ok()
                 ? gem16::Status(
                       gem16::StatusCode::kDataLoss,
                       "V11 D2 cancellation unexpectedly completed")
                 : status;
    }
  }
  return result;
}

void WriteTokens(std::span<const std::uint32_t> tokens) {
  std::cout << '[';
  for (std::size_t index = 0U; index < tokens.size(); ++index) {
    if (index != 0U) std::cout << ',';
    std::cout << tokens[index];
  }
  std::cout << ']';
}

void WriteGroup(const GroupRun& run) {
  std::cout << "{\"initial\":" << run.initial
            << ",\"position_before_first_proposal\":"
            << run.position_before_first_proposal
            << ",\"position_after_group\":" << run.position_after_group
            << ",\"proposals\":";
  WriteTokens(std::span(run.group.proposed.data(), run.group.proposal_count));
  std::cout << ",\"verified\":";
  WriteTokens(std::span(run.group.verified.data(), run.group.output_count));
  std::cout << ",\"proposal_count\":" << run.group.proposal_count
            << ",\"accepted_count\":" << run.group.accepted_count
            << ",\"output_count\":" << run.group.output_count
            << ",\"stopped\":" << run.group.stopped << '}';
}

void WriteState(const StateDigest& digest) {
  const auto& state = digest.state;
  std::cout << "{\"position\":" << state.position
            << ",\"final_hidden_sha256\":"
            << gem16::json::Quote(digest.final_hidden_sha256)
            << ",\"visible_kv_bytes\":" << state.kv_bytes
            << ",\"visible_kv_sha256\":"
            << gem16::json::Quote(digest.visible_kv_sha256)
            << ",\"sliding_ring_position\":"
            << state.sliding_ring_position
            << ",\"sampling_step\":" << state.sampling_step
            << ",\"prediction_token\":" << state.prediction_token
            << ",\"prediction_logit_bits\":"
            << state.prediction_logit_bits
            << ",\"prediction_finite\":" << state.prediction_finite
            << ",\"routing_finite\":" << state.routing_finite
            << ",\"decode_control_token\":"
            << state.decode_control_token
            << ",\"decode_control_suppressed_token_count\":"
            << state.decode_control_suppressed_token_count
            << ",\"decode_control_position\":"
            << state.decode_control_position
            << ",\"decode_control_sampling_step\":"
            << state.decode_control_sampling_step
            << ",\"vision_encode_calls\":" << state.vision_encode_calls
            << ",\"pending_decode_self_feed\":"
            << state.pending_decode_self_feed
            << ",\"decode_self_feed_valid\":"
            << state.decode_self_feed_valid << '}';
}

bool StateExact(const StateDigest& left, const StateDigest& right) {
  return left.state.position == right.state.position &&
         left.state.kv_bytes == right.state.kv_bytes &&
         left.state.sliding_ring_position ==
             right.state.sliding_ring_position &&
         left.state.sampling_step == right.state.sampling_step &&
         left.state.prediction_token == right.state.prediction_token &&
         // The selected-token logit is diagnostic output from two different
         // reduction shapes.  It is not committed decode state and may differ
         // by a few FP32 bits even when the selected token, hidden state, KV,
         // position, and RNG step are identical.
         left.state.prediction_finite == right.state.prediction_finite &&
         left.state.routing_finite == right.state.routing_finite &&
         left.state.vision_encode_calls == right.state.vision_encode_calls &&
         left.final_hidden_sha256 == right.final_hidden_sha256 &&
         left.visible_kv_sha256 == right.visible_kv_sha256;
}

bool TrajectoryExact(const TrajectoryRun& ordinary,
                     const TrajectoryRun& d2) {
  if (ordinary.position_before_first_proposal !=
          d2.position_before_first_proposal ||
      ordinary.tokens != d2.tokens ||
      ordinary.checkpoints.size() != d2.checkpoints.size() ||
      !StateExact(ordinary.pre_proposal, d2.pre_proposal)) {
    return false;
  }
  for (std::size_t index = 0U; index < ordinary.checkpoints.size(); ++index) {
    if (ordinary.checkpoints[index].processed_tokens !=
            d2.checkpoints[index].processed_tokens ||
        !StateExact(ordinary.checkpoints[index].state,
                    d2.checkpoints[index].state)) {
      return false;
    }
  }
  return true;
}

void WriteTrajectory(const TrajectoryRun& ordinary,
                     const TrajectoryRun& d2) {
  std::cout << "{\"exact\":"
            << (TrajectoryExact(ordinary, d2) ? "true" : "false")
            << ",\"tokens\":";
  WriteTokens(d2.tokens);
  std::cout << ",\"first_group_layer_mismatches\":[";
  bool first_layer_mismatch = true;
  if (ordinary.first_group_layer_outputs_captured &&
      d2.first_group_layer_outputs_captured && !d2.checkpoints.empty()) {
    const std::uint32_t row =
        d2.checkpoints.front().group.output_count - 1U;
    for (std::uint32_t layer = 0U; layer < 30U; ++layer) {
      if (ordinary.first_group_layer_outputs[row][layer] ==
          d2.first_group_layer_outputs[row][layer]) {
        continue;
      }
      if (!first_layer_mismatch) std::cout << ',';
      std::cout << layer;
      first_layer_mismatch = false;
    }
  }
  std::cout << ']';
  std::cout << ",\"groups\":[";
  for (std::size_t index = 0U; index < d2.checkpoints.size(); ++index) {
    if (index != 0U) std::cout << ',';
    const auto& checkpoint = d2.checkpoints[index];
    std::cout << "{\"index\":" << index
              << ",\"processed_tokens\":"
              << checkpoint.processed_tokens << ",\"group\":";
    GroupRun view;
    view.group = checkpoint.group;
    WriteGroup(view);
    std::cout << ",\"state_exact\":"
              << (StateExact(ordinary.checkpoints[index].state,
                             checkpoint.state)
                      ? "true"
                      : "false");
    if (!StateExact(ordinary.checkpoints[index].state, checkpoint.state)) {
      std::cout << ",\"ordinary_state\":";
      WriteState(ordinary.checkpoints[index].state);
      std::cout << ",\"d2_state\":";
      WriteState(checkpoint.state);
    }
    std::cout << '}';
  }
  std::cout << "]}";
}

void WriteMatrixCase(const MatrixCaseResult& value) {
  std::cout << "{\"name\":" << gem16::json::Quote(value.name)
            << ",\"sampling\":" << gem16::json::Quote(value.sampling)
            << ",\"context\":" << value.context
            << ",\"prompt_tokens\":" << value.prompt_tokens
            << ",\"vision_offset\":" << value.vision_offset
            << ",\"vision_end\":" << value.vision_end
            << ",\"trajectory\":";
  WriteTrajectory(value.ordinary, value.d2);
  std::cout << '}';
}

bool ContinuationExact(const ContinuationCaseResult& value) {
  return value.ordinary.first_turn_tokens == value.d2.first_turn_tokens &&
         TrajectoryExact(value.ordinary.continuation,
                         value.d2.continuation) &&
         value.ordinary.continuation.pre_proposal.state.vision_encode_calls ==
             1U &&
         value.d2.continuation.pre_proposal.state.vision_encode_calls == 1U &&
         value.ordinary.continuation.checkpoints.front()
                 .state.state.vision_encode_calls == 1U &&
         value.d2.continuation.checkpoints.front()
                 .state.state.vision_encode_calls == 1U;
}

void WriteContinuation(const ContinuationCaseResult& value) {
  std::cout << "{\"exact\":"
            << (ContinuationExact(value) ? "true" : "false")
            << ",\"first_turn_tokens\":";
  WriteTokens(value.d2.first_turn_tokens);
  std::cout << ",\"trajectory\":";
  WriteTrajectory(value.ordinary.continuation, value.d2.continuation);
  std::cout << '}';
}

bool CancellationExact(const CancellationCaseResult& value) {
  return value.vision_cancel_requested_during_prefill &&
         value.vision_prefill_completed_safely &&
         value.vision_d2_not_started && value.d2_cancelled &&
         value.d2_callback_count == 1U && value.d2_output_count > 0U &&
         value.d2_group_count > 0U;
}

void WriteCancellation(const CancellationCaseResult& value) {
  std::cout << "{\"exact\":"
            << (CancellationExact(value) ? "true" : "false")
            << ",\"vision_cancel_requested_during_prefill\":"
            << (value.vision_cancel_requested_during_prefill ? "true"
                                                               : "false")
            << ",\"vision_prefill_completed_safely\":"
            << (value.vision_prefill_completed_safely ? "true" : "false")
            << ",\"vision_d2_not_started\":"
            << (value.vision_d2_not_started ? "true" : "false")
            << ",\"d2_cancelled\":"
            << (value.d2_cancelled ? "true" : "false")
            << ",\"d2_callback_count\":" << value.d2_callback_count
            << ",\"d2_output_count\":" << value.d2_output_count
            << ",\"d2_group_count\":" << value.d2_group_count << '}';
}

int Fail(const gem16::Status& status) {
  std::cerr << "error: " << status.message() << '\n';
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 6 ||
      !gem16::internal::Gemma4Moe26BVisionD2DiagnosticEnabled()) {
    std::cerr
        << "usage: GEM16_VISION_D2_DIAGNOSTIC=1 "
           "gem16-26b-vision-d2-diagnostic <target> <assistant> "
           "<vision-module> <image> <70|140|280>\n";
    return 64;
  }
  std::uint32_t budget = 0U;
  try {
    budget = static_cast<std::uint32_t>(std::stoul(argv[5]));
  } catch (...) {
    return 64;
  }
  if (budget != 70U && budget != 140U && budget != 280U) return 64;
  if constexpr (std::endian::native != std::endian::little) {
    std::cerr << "error: V11 prompt hashes require a little-endian host\n";
    return 2;
  }
  const std::filesystem::path target = argv[1];
  const std::filesystem::path assistant = argv[2];
  const std::filesystem::path vision = argv[3];
  const std::filesystem::path image_path = argv[4];
  auto fixture = BuildFixture(target, image_path, budget);
  if (!fixture.ok()) return Fail(fixture.status());
  auto processor = gem16::GemmaChatProcessor::Load(target);
  if (!processor.ok()) return Fail(processor.status());
  auto ordinary = RunOrdinary(target, vision, fixture.value(),
                              processor.value());
  if (!ordinary.ok()) return Fail(ordinary.status());
  const std::array forced_proposals{ordinary.value().tokens[1],
                                    ordinary.value().tokens[2]};
  auto forced = RunGroup(target, assistant, vision, fixture.value(),
                         processor.value(), forced_proposals);
  if (!forced.ok()) return Fail(forced.status());
  auto real = RunGroup(target, assistant, vision, fixture.value(),
                       processor.value(), {});
  if (!real.ok()) return Fail(real.status());
  auto real_trajectory = RunRealTrajectory(
      target, assistant, vision, fixture.value(), processor.value(), 8U);
  if (!real_trajectory.ok()) return Fail(real_trajectory.status());
  auto ordinary_trajectory = RunOrdinaryTrajectory(
      target, vision, fixture.value(), processor.value(),
      real_trajectory.value().checkpoints);
  if (!ordinary_trajectory.ok()) return Fail(ordinary_trajectory.status());

  std::vector<MatrixCaseResult> matrix_cases;
  std::optional<ContinuationCaseResult> continuation_case;
  std::optional<CancellationCaseResult> cancellation_case;
  const auto run_case = [&](std::string name, std::string sampling_name,
                            Fixture shaped, std::uint32_t groups,
                            const gem16::SamplingOptions& sampling,
                            std::uint64_t context)
      -> gem16::Status {
    auto value = RunMatrixCase(
        std::move(name), std::move(sampling_name), target, assistant, vision,
        shaped, processor.value(), groups, sampling, context);
    if (!value.ok()) return value.status();
    matrix_cases.push_back(std::move(value).value());
    return gem16::Status::Ok();
  };
  if (budget == 70U) {
    const char* matrix_filter = std::getenv("GEM16_V11_MATRIX_FILTER");
    const bool short_only = matrix_filter != nullptr &&
                            std::string_view(matrix_filter) == "short";
    const bool cancel_only = matrix_filter != nullptr &&
                             std::string_view(matrix_filter) == "cancel";
    gem16::Status status = gem16::Status::Ok();
    if (!cancel_only) {
      status = run_case("short_context", "greedy", fixture.value(), 2U, {},
                        1024U);
      if (!status.ok()) return Fail(status);
    }
    if (!short_only && !cancel_only) {
      Fixture ring = fixture.value();
      status = ShapeFixture(&ring, ring.vision_offset, 1023U);
      if (!status.ok()) return Fail(status);
      status = run_case("local_ring_wrap", "greedy", std::move(ring), 2U,
                        {}, kContext);
      if (!status.ok()) return Fail(status);
      Fixture long_context = fixture.value();
      status = ShapeFixture(&long_context, 16320U, 16416U);
      if (!status.ok()) return Fail(status);
      status = run_case("over_16k_image_boundary", "greedy",
                        std::move(long_context), 1U, {}, kContext);
      if (!status.ok()) return Fail(status);
    }
    if (!short_only) {
      auto cancellation = RunCancellationCase(
          target, assistant, vision, fixture.value(), processor.value());
      if (!cancellation.ok()) return Fail(cancellation.status());
      cancellation_case = std::move(cancellation).value();
    }
  } else if (budget == 140U) {
    Fixture middle = fixture.value();
    gem16::Status status = ShapeFixture(&middle, 512U, 768U);
    if (!status.ok()) return Fail(status);
    status = run_case("image_middle", "greedy", std::move(middle), 2U,
                      {}, kContext);
    if (!status.ok()) return Fail(status);
    gem16::SamplingOptions sampled;
    sampled.enabled = true;
    sampled.temperature = 0.8F;
    sampled.top_p = 0.95F;
    sampled.top_k = 40U;
    sampled.repetition_penalty = 1.05F;
    sampled.seed = 0x5a17c9e3ULL;
    status = run_case("fixed_seed_sampled", "fixed_seed_sampled",
                      fixture.value(), 6U, sampled, kContext);
    if (!status.ok()) return Fail(status);
  } else {
    Fixture chunk_boundary = fixture.value();
    gem16::Status status = ShapeFixture(&chunk_boundary, 1984U, 2400U);
    if (!status.ok()) return Fail(status);
    status = run_case("near_chunk_boundary", "greedy",
                      std::move(chunk_boundary), 2U, {}, kContext);
    if (!status.ok()) return Fail(status);
    auto continuation = RunContinuationCase(
        target, assistant, vision, fixture.value(), processor.value());
    if (!continuation.ok()) return Fail(continuation.status());
    continuation_case = std::move(continuation).value();
  }

  const bool initial_exact =
      forced.value().initial == ordinary.value().tokens[0] &&
      real.value().initial == ordinary.value().tokens[0];
  const bool pre_proposal_exact =
      forced.value().pre_proposal.final_hidden_sha256 ==
          ordinary.value().pre_proposal.final_hidden_sha256 &&
      forced.value().pre_proposal.visible_kv_sha256 ==
          ordinary.value().pre_proposal.visible_kv_sha256 &&
      real.value().pre_proposal.final_hidden_sha256 ==
          ordinary.value().pre_proposal.final_hidden_sha256 &&
      real.value().pre_proposal.visible_kv_sha256 ==
          ordinary.value().pre_proposal.visible_kv_sha256;
  const bool forced_exact =
      forced.value().group.proposal_count == 2U &&
      forced.value().group.accepted_count == 2U &&
      forced.value().group.output_count == 3U &&
      std::equal(forced.value().group.verified.begin(),
                 forced.value().group.verified.begin() + 3U,
                 ordinary.value().tokens.begin() + 1U);
  const bool real_exact =
      real.value().group.output_count > 0U &&
      std::equal(real.value().group.verified.begin(),
                 real.value().group.verified.begin() +
                     real.value().group.output_count,
                 ordinary.value().tokens.begin() + 1U);
  const bool forced_transaction_exact =
      StateExact(forced.value().post_group,
                 ordinary.value().post_forward[2U]);
  const bool real_transaction_exact =
      real.value().group.output_count > 0U &&
      real.value().group.output_count <= 3U &&
      StateExact(real.value().post_group,
                 ordinary.value().post_forward[
                     real.value().group.output_count - 1U]);
  const bool full_trajectory_exact =
      TrajectoryExact(ordinary_trajectory.value(), real_trajectory.value());
  const bool matrix_exact = std::all_of(
      matrix_cases.begin(), matrix_cases.end(), [](const auto& value) {
        return TrajectoryExact(value.ordinary, value.d2);
      });
  const bool continuation_exact =
      !continuation_case.has_value() ||
      ContinuationExact(*continuation_case);
  const bool cancellation_exact =
      !cancellation_case.has_value() ||
      CancellationExact(*cancellation_case);
  std::array<std::vector<std::uint32_t>, 3U> layer_mismatches;
  for (std::uint32_t row = 0U; row < 3U; ++row) {
    for (std::uint32_t layer = 0U; layer < 30U; ++layer) {
      if (forced.value().layer_outputs[row][layer] !=
          ordinary.value().layer_outputs[row][layer]) {
        layer_mismatches[row].push_back(layer);
      }
    }
  }
  const bool layer_outputs_exact = std::all_of(
      layer_mismatches.begin(), layer_mismatches.end(),
      [](const auto& mismatches) { return mismatches.empty(); });

  auto image_hash = HashFile(image_path);
  auto target_hash = HashFile(target / "FINAL_SHA256SUMS");
  auto assistant_hash = HashFile(assistant / "gem16_compilation.json");
  auto vision_hash = HashFile(vision / "vision.lock.json");
  auto binary_hash = HashFile(argv[0]);
  if (!image_hash.ok()) return Fail(image_hash.status());
  if (!target_hash.ok()) return Fail(target_hash.status());
  if (!assistant_hash.ok()) return Fail(assistant_hash.status());
  if (!vision_hash.ok()) return Fail(vision_hash.status());
  if (!binary_hash.ok()) return Fail(binary_hash.status());
  const std::string prompt_hash = gem16::compiler::Sha256Hex(
      fixture.value().prompt.data(), fixture.value().prompt.size() *
                                         sizeof(std::uint32_t));

  std::cout << "{\"schema_version\":1,\"qualification\":\""
               "vision_mtp_unqualified\",\"diagnostic_gate\":true,"
               "\"manifest\":{\"image_sha256\":"
            << gem16::json::Quote(image_hash.value())
            << ",\"budget\":" << budget
            << ",\"processed_width\":"
            << fixture.value().image.processed_width
            << ",\"processed_height\":"
            << fixture.value().image.processed_height
            << ",\"raw_patch_count\":"
            << fixture.value().image.raw_patch_count
            << ",\"soft_token_count\":"
            << fixture.value().image.soft_token_count
            << ",\"prompt_token_count\":" << fixture.value().prompt.size()
            << ",\"prompt_token_sha256\":" << gem16::json::Quote(prompt_hash)
            << ",\"image_begin_token\":" << kImageBegin
            << ",\"image_begin_offset\":"
            << fixture.value().vision_offset - 1U
            << ",\"image_end_token\":" << kImageEnd
            << ",\"image_end_offset\":"
            << fixture.value().vision_offset +
                   fixture.value().image.soft_token_count
            << ",\"sampling\":\"greedy\",\"target_identity_file\":"
            << gem16::json::Quote("FINAL_SHA256SUMS")
            << ",\"target_sha256\":" << gem16::json::Quote(target_hash.value())
            << ",\"assistant_identity_file\":"
            << gem16::json::Quote("gem16_compilation.json")
            << ",\"assistant_sha256\":"
            << gem16::json::Quote(assistant_hash.value())
            << ",\"vision_identity_file\":"
            << gem16::json::Quote("vision.lock.json")
            << ",\"vision_sha256\":" << gem16::json::Quote(vision_hash.value())
            << ",\"binary_sha256\":" << gem16::json::Quote(binary_hash.value())
            << "},\"ordinary\":{\"position_before_first_proposal\":"
            << ordinary.value().position_before_first_proposal
            << ",\"pre_proposal_state\":";
  WriteState(ordinary.value().pre_proposal);
  std::cout << ",\"post_forward_states\":[";
  for (std::size_t index = 0U; index < ordinary.value().post_forward.size();
       ++index) {
    if (index != 0U) std::cout << ',';
    WriteState(ordinary.value().post_forward[index]);
  }
  std::cout << "],\"tokens\":";
  WriteTokens(ordinary.value().tokens);
  std::cout << "},\"forced_exact\":";
  WriteGroup(forced.value());
  std::cout << ",\"forced_pre_proposal_state\":";
  WriteState(forced.value().pre_proposal);
  std::cout << ",\"forced_post_group_state\":";
  WriteState(forced.value().post_group);
  std::cout << ",\"real_assistant\":";
  WriteGroup(real.value());
  std::cout << ",\"real_pre_proposal_state\":";
  WriteState(real.value().pre_proposal);
  std::cout << ",\"real_post_group_state\":";
  WriteState(real.value().post_group);
  std::cout << ",\"full_greedy_trajectory\":";
  WriteTrajectory(ordinary_trajectory.value(), real_trajectory.value());
  std::cout << ",\"matrix_cases\":[";
  for (std::size_t index = 0U; index < matrix_cases.size(); ++index) {
    if (index != 0U) std::cout << ',';
    WriteMatrixCase(matrix_cases[index]);
  }
  std::cout << ']';
  if (continuation_case.has_value()) {
    std::cout << ",\"cached_continuation\":";
    WriteContinuation(*continuation_case);
  }
  if (cancellation_case.has_value()) {
    std::cout << ",\"cancellation\":";
    WriteCancellation(*cancellation_case);
  }
  std::cout << ",\"layer_output_differential\":{\"all_exact\":"
            << (layer_outputs_exact ? "true" : "false")
            << ",\"rows\":[";
  for (std::uint32_t row = 0U; row < 3U; ++row) {
    if (row != 0U) std::cout << ',';
    std::cout << "{\"row\":" << row << ",\"mismatch_layers\":";
    WriteTokens(layer_mismatches[row]);
    if (!layer_mismatches[row].empty()) {
      const std::uint32_t layer = layer_mismatches[row].front();
      std::cout << ",\"first_ordinary_sha256\":"
                << gem16::json::Quote(
                       ordinary.value().layer_outputs[row][layer])
                << ",\"first_t3_sha256\":"
                << gem16::json::Quote(
                       forced.value().layer_outputs[row][layer]);
    }
    std::cout << '}';
  }
  std::cout << "]}";
  std::cout << ",\"verdict\":{\"initial_exact\":"
            << (initial_exact ? "true" : "false")
            << ",\"pre_proposal_state_exact\":"
            << (pre_proposal_exact ? "true" : "false")
            << ",\"forced_exact\":" << (forced_exact ? "true" : "false")
            << ",\"forced_transaction_exact\":"
            << (forced_transaction_exact ? "true" : "false")
            << ",\"real_assistant_exact\":"
            << (real_exact ? "true" : "false")
            << ",\"real_transaction_exact\":"
            << (real_transaction_exact ? "true" : "false")
            << ",\"full_trajectory_exact\":"
            << (full_trajectory_exact ? "true" : "false")
            << ",\"matrix_exact\":"
            << (matrix_exact ? "true" : "false")
            << ",\"cached_continuation_exact\":"
            << (continuation_exact ? "true" : "false")
            << ",\"cancellation_exact\":"
            << (cancellation_exact ? "true" : "false") << "}}\n";
  return initial_exact && pre_proposal_exact && forced_exact &&
                 forced_transaction_exact && real_exact &&
                 real_transaction_exact && full_trajectory_exact &&
                 matrix_exact && continuation_exact
                 && cancellation_exact
             ? 0
             : 1;
}
