#include <array>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>

#include "gem16gb/memory.h"
#include "gem16gb/engine.h"
#if GEM16GB_HAS_CUDA
#include "cuda/fp8/checkpoint_probe.h"
#include "cuda/layer/checkpoint_probe.h"
#include "cuda/nvfp4/checkpoint_probe.h"
#endif

namespace {

void Usage(std::ostream& output) {
  output << "Usage: gem16gb-bench <model-load|prefill|decode|end-to-end|kernel|memory|quality|mtp> [options]\n"
         << "\n"
         << "Memory mode:\n"
         << "  gem16gb-bench memory --model <checkpoint-dir>\n"
         << "      --profile <interactive|standard|long|xlong|max>\n"
         << "      --kv-storage separate [--kv-element-bytes <1|2>]\n"
         << "\n"
         << "Kernel mode (CUDA):\n"
         << "  gem16gb-bench kernel --model <checkpoint-dir>\n"
         << "      [--projection <gate|up|down|mlp|q|k|v|o|local-attention|global-attention|decoder-layer>]\n"
         << "      [--warmups <count>] [--iterations <count>]\n"
         << "\n"
         << "Decode mode (CUDA):\n"
         << "  gem16gb-bench decode --model <checkpoint-dir>\n"
         << "      [--context <tokens>] [--tokens <count>] [--seed <integer>]\n"
         << "      [--warmups <count>] [--repetitions <count>]\n"
         << "      [--kv-cache <fp8|bf16>] [--projection-path <native|reference>]\n"
         << "      [--enable-fused-gate-up]\n";
}

bool ParsePositiveU32(std::string_view value, std::uint32_t& parsed) {
  std::uint32_t candidate = 0;
  const auto result = std::from_chars(value.data(), value.data() + value.size(), candidate);
  if (result.ec != std::errc{} || result.ptr != value.data() + value.size() || candidate == 0) {
    return false;
  }
  parsed = candidate;
  return true;
}

bool ParseU32(std::string_view value, std::uint32_t& parsed) {
  std::uint32_t candidate = 0;
  const auto result = std::from_chars(value.data(), value.data() + value.size(), candidate);
  if (result.ec != std::errc{} || result.ptr != value.data() + value.size()) return false;
  parsed = candidate;
  return true;
}

int RunDecodeMode(int argc, char** argv) {
  gem16gb::DecodeBenchmarkOptions options;
  for (int index = 2; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--model" && index + 1 < argc) {
      options.model_directory = std::filesystem::path(argv[++index]);
    } else if (argument == "--context" && index + 1 < argc) {
      if (!ParsePositiveU32(argv[++index], options.context_tokens)) {
        std::cerr << "error: --context must be a positive integer\n";
        return 64;
      }
    } else if (argument == "--tokens" && index + 1 < argc) {
      if (!ParsePositiveU32(argv[++index], options.generated_tokens)) {
        std::cerr << "error: --tokens must be a positive integer\n";
        return 64;
      }
    } else if (argument == "--seed" && index + 1 < argc) {
      if (!ParseU32(argv[++index], options.prompt_seed)) {
        std::cerr << "error: --seed must be an unsigned integer\n";
        return 64;
      }
    } else if (argument == "--warmups" && index + 1 < argc) {
      if (!ParsePositiveU32(argv[++index], options.warmup_runs)) {
        std::cerr << "error: --warmups must be a positive integer\n";
        return 64;
      }
    } else if (argument == "--repetitions" && index + 1 < argc) {
      if (!ParsePositiveU32(argv[++index], options.measured_runs)) {
        std::cerr << "error: --repetitions must be a positive integer\n";
        return 64;
      }
    } else if (argument == "--kv-cache" && index + 1 < argc) {
      const std::string_view value(argv[++index]);
      if (value == "fp8") options.kv_cache_mode = gem16gb::KvCacheMode::kCheckpointFp8;
      else if (value == "bf16") options.kv_cache_mode = gem16gb::KvCacheMode::kBf16Correctness;
      else {
        std::cerr << "error: --kv-cache must be fp8 or bf16\n";
        return 64;
      }
    } else if (argument == "--projection-path" && index + 1 < argc) {
      const std::string_view value(argv[++index]);
      if (value == "native") options.projection_path = gem16gb::ProjectionPath::kNativeSm120;
      else if (value == "reference") options.projection_path = gem16gb::ProjectionPath::kCudaReference;
      else {
        std::cerr << "error: --projection-path must be native or reference\n";
        return 64;
      }
    } else if (argument == "--enable-fused-gate-up") {
      options.enable_fused_gate_up = true;
    } else {
      std::cerr << "error: unknown or incomplete decode option: " << argument << '\n';
      Usage(std::cerr);
      return 64;
    }
  }
  if (options.model_directory.empty()) {
    std::cerr << "error: decode mode requires --model\n";
    return 64;
  }

  auto result = gem16gb::RunDecodeBenchmark(options);
  if (!result.ok()) {
    std::cerr << "error: " << result.status().message() << '\n';
    return 1;
  }
  const auto& benchmark = result.value();
  std::cerr << std::fixed << std::setprecision(3)
            << "Decode characterization\n"
            << "  context/generated: " << options.context_tokens << '/' << options.generated_tokens << '\n'
            << "  runs: " << options.warmup_runs << " warm-up, " << options.measured_runs << " measured\n"
            << "  median throughput: " << benchmark.decode_tokens_per_second.median << " tokens/s\n"
            << "  p50/p95/p99 latency: " << benchmark.inter_token_latency_milliseconds.median << "/"
            << benchmark.inter_token_latency_milliseconds.p95 << "/"
            << benchmark.inter_token_latency_milliseconds.p99 << " ms\n";
  const auto status = gem16gb::WriteDecodeBenchmarkJson(benchmark, std::cout);
  if (!status.ok()) {
    std::cerr << "error: " << status.message() << '\n';
    return 1;
  }
  return 0;
}

bool ParseProfile(std::string_view value, gem16gb::ContextProfile& profile) {
  if (value == "interactive") profile = gem16gb::ContextProfile::kInteractive;
  else if (value == "standard") profile = gem16gb::ContextProfile::kStandard;
  else if (value == "long") profile = gem16gb::ContextProfile::kLong;
  else if (value == "xlong") profile = gem16gb::ContextProfile::kXlong;
  else if (value == "max") profile = gem16gb::ContextProfile::kMax;
  else return false;
  return true;
}

bool ParseKvStorage(std::string_view value, gem16gb::KvStorage& storage) {
  if (value == "shared") storage = gem16gb::KvStorage::kShared;
  else if (value == "separate") storage = gem16gb::KvStorage::kSeparate;
  else return false;
  return true;
}

int RunMemoryMode(int argc, char** argv) {
  std::filesystem::path model_directory;
  gem16gb::MemoryPlanOptions options;
  for (int index = 2; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--model" && index + 1 < argc) {
      model_directory = std::filesystem::path(argv[++index]);
    } else if (argument == "--profile" && index + 1 < argc) {
      if (!ParseProfile(argv[++index], options.context_profile)) {
        std::cerr << "error: unknown context profile\n";
        return 64;
      }
    } else if (argument == "--kv-storage" && index + 1 < argc) {
      if (!ParseKvStorage(argv[++index], options.kv_storage)) {
        std::cerr << "error: --kv-storage must be shared or separate\n";
        return 64;
      }
    } else if (argument == "--kv-element-bytes" && index + 1 < argc) {
      const std::string_view value(argv[++index]);
      if (value == "1") options.kv_element_bytes = 1;
      else if (value == "2") options.kv_element_bytes = 2;
      else {
        std::cerr << "error: --kv-element-bytes must be 1 or 2\n";
        return 64;
      }
    } else {
      std::cerr << "error: unknown or incomplete memory option: " << argument << '\n';
      Usage(std::cerr);
      return 64;
    }
  }
  if (model_directory.empty()) {
    std::cerr << "error: memory mode requires --model\n";
    return 64;
  }

  auto plan = gem16gb::PlanCheckpointMemory(model_directory, options);
  if (!plan.ok()) {
    std::cerr << "error: " << plan.status().message() << '\n';
    return 1;
  }
  gem16gb::PrintMemoryPlanSummary(plan.value(), std::cerr);
  const auto write_status = gem16gb::WriteMemoryPlanJson(plan.value(), std::cout);
  if (!write_status.ok()) {
    std::cerr << "error: " << write_status.message() << '\n';
    return 1;
  }
  return 0;
}

int RunKernelMode(int argc, char** argv) {
  std::filesystem::path model_directory;
  std::string projection = "gate";
  std::uint32_t warmups = 3;
  std::uint32_t iterations = 10;
  for (int index = 2; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--model" && index + 1 < argc) {
      model_directory = std::filesystem::path(argv[++index]);
    } else if (argument == "--projection" && index + 1 < argc) {
      projection = argv[++index];
      if (projection != "gate" && projection != "up" && projection != "down" &&
          projection != "mlp" && projection != "q" && projection != "k" &&
          projection != "v" && projection != "o" && projection != "local-attention" &&
          projection != "global-attention" && projection != "decoder-layer") {
        std::cerr << "error: unsupported kernel projection\n";
        return 64;
      }
    } else if (argument == "--warmups" && index + 1 < argc) {
      if (!ParsePositiveU32(argv[++index], warmups)) {
        std::cerr << "error: --warmups must be a positive integer\n";
        return 64;
      }
    } else if (argument == "--iterations" && index + 1 < argc) {
      if (!ParsePositiveU32(argv[++index], iterations)) {
        std::cerr << "error: --iterations must be a positive integer\n";
        return 64;
      }
    } else {
      std::cerr << "error: unknown or incomplete kernel option: " << argument << '\n';
      Usage(std::cerr);
      return 64;
    }
  }
  if (model_directory.empty()) {
    std::cerr << "error: kernel mode requires --model\n";
    return 64;
  }

#if GEM16GB_HAS_CUDA
  if (projection == "decoder-layer") {
    auto result = gem16gb::internal::RunLayer0DecoderCheckpointProbe(model_directory);
    if (!result.ok()) {
      std::cerr << "error: " << result.status().message() << '\n';
      return 1;
    }
    const auto& probe = result.value();
    std::cerr << "Complete Layer-0 decoder checkpoint probe\n"
              << "  path: local attention residual -> pre-MLP RMSNorm -> NVFP4 Gate/Up -> "
                 "GELU-tanh -> NVFP4 Down -> post-MLP RMSNorm -> residual -> layer scalar\n"
              << "  context: " << probe.context_tokens << " tokens\n"
              << "  MLP-input mismatched bytes: " << probe.mlp_input_mismatched_bytes << '\n'
              << "  Down-input mismatched bytes: " << probe.down_input_mismatched_bytes << '\n'
              << "  reference/native max abs: " << probe.reference_native_max_abs << '\n'
              << "  reference/native cosine: " << probe.reference_native_cosine << '\n';
    std::cout << std::setprecision(17)
              << "{\"schema_version\":1,\"status\":\"characterization\","
              << "\"benchmark_qualified\":false,\"mode\":\"kernel\",\"fallbacks\":0,"
              << "\"operator\":\"layer0_decoder\",\"precision\":\"fp8_attention_nvfp4_mlp\","
              << "\"context_tokens\":" << probe.context_tokens
              << ",\"source_layout_direct\":true,\"persistent_repack_bytes\":0,"
              << "\"no_host_roundtrip_between_sublayers\":true,\"layer_scalar_applied\":"
              << (probe.layer_scalar_applied ? "true" : "false")
              << ",\"mlp_input_mismatched_bytes\":" << probe.mlp_input_mismatched_bytes
              << ",\"down_input_mismatched_bytes\":" << probe.down_input_mismatched_bytes
              << ",\"device_bytes\":" << probe.device_bytes
              << ",\"error\":{\"reference_native_max_abs\":"
              << probe.reference_native_max_abs << ",\"reference_native_rms\":"
              << probe.reference_native_rms << ",\"reference_native_cosine\":"
              << probe.reference_native_cosine << "},\"samples\":[";
    for (std::size_t index = 0; index < probe.samples.size(); ++index) {
      if (index != 0) std::cout << ',';
      const auto& sample = probe.samples[index];
      std::cout << "{\"element\":" << sample.element << ",\"cuda_reference\":"
                << sample.cuda_reference << ",\"sm120_direct\":" << sample.sm120_direct << '}';
    }
    std::cout << "]}\n";
    return 0;
  }
  if (projection == "global-attention") {
    auto result = gem16gb::internal::RunLayer5GlobalAttentionCheckpointProbe(model_directory);
    if (!result.ok()) {
      std::cerr << "error: " << result.status().message() << '\n';
      return 1;
    }
    const auto& probe = result.value();
    std::cerr << "Layer-5 global-attention checkpoint probe\n"
              << "  path: RMSNorm -> Q/K -> K reuse for V -> Q/K/V norm -> proportional RoPE -> "
                 "KV append/read -> softmax -> O -> RMSNorm -> residual\n"
              << "  context: " << probe.context_tokens << " tokens\n"
              << "  reference/native max abs: " << probe.reference_native_max_abs << '\n'
              << "  reference/native cosine: " << probe.reference_native_cosine << '\n';
    std::cout << std::setprecision(17)
              << "{\"schema_version\":1,\"status\":\"characterization\","
              << "\"benchmark_qualified\":false,\"mode\":\"kernel\",\"fallbacks\":0,"
              << "\"operator\":\"layer5_global_attention\",\"precision\":\"fp8_attention\","
              << "\"context_tokens\":" << probe.context_tokens
              << ",\"k_projection_reused_for_v\":"
              << (probe.reused_k_projection_for_v ? "true" : "false")
              << ",\"source_layout_direct\":true,\"persistent_repack_bytes\":0,\"device_bytes\":"
              << probe.device_bytes << ",\"error\":{\"reference_native_max_abs\":"
              << probe.reference_native_max_abs << ",\"reference_native_rms\":"
              << probe.reference_native_rms << ",\"reference_native_cosine\":"
              << probe.reference_native_cosine << "},\"samples\":[";
    for (std::size_t index = 0; index < probe.samples.size(); ++index) {
      if (index != 0) std::cout << ',';
      const auto& sample = probe.samples[index];
      std::cout << "{\"element\":" << sample.element << ",\"cuda_reference\":"
                << sample.cuda_reference << ",\"sm120_direct\":" << sample.sm120_direct << '}';
    }
    std::cout << "]}\n";
    return 0;
  }
  if (projection == "local-attention") {
    auto result = gem16gb::internal::RunLayer0LocalAttentionCheckpointProbe(model_directory);
    if (!result.ok()) {
      std::cerr << "error: " << result.status().message() << '\n';
      return 1;
    }
    const auto& probe = result.value();
    std::cerr << "Layer-0 local-attention checkpoint probe\n"
              << "  path: RMSNorm -> Q/K/V -> Q/K/V norm -> RoPE -> KV append/read -> "
                 "softmax -> O -> RMSNorm -> residual\n"
              << "  context: " << probe.context_tokens << " tokens\n"
              << "  reference/native max abs: " << probe.reference_native_max_abs << '\n'
              << "  reference/native cosine: " << probe.reference_native_cosine << '\n';
    std::cout << std::setprecision(17)
              << "{\"schema_version\":1,\"status\":\"characterization\","
              << "\"benchmark_qualified\":false,\"mode\":\"kernel\",\"fallbacks\":0,"
              << "\"operator\":\"layer0_local_attention\",\"precision\":\"fp8_attention\","
              << "\"context_tokens\":" << probe.context_tokens
              << ",\"source_layout_direct\":true,\"persistent_repack_bytes\":0,\"device_bytes\":"
              << probe.device_bytes << ",\"error\":{\"reference_native_max_abs\":"
              << probe.reference_native_max_abs << ",\"reference_native_rms\":"
              << probe.reference_native_rms << ",\"reference_native_cosine\":"
              << probe.reference_native_cosine << "},\"samples\":[";
    for (std::size_t index = 0; index < probe.samples.size(); ++index) {
      if (index != 0) std::cout << ',';
      const auto& sample = probe.samples[index];
      std::cout << "{\"element\":" << sample.element << ",\"cuda_reference\":"
                << sample.cuda_reference << ",\"sm120_direct\":" << sample.sm120_direct << '}';
    }
    std::cout << "]}\n";
    return 0;
  }
  if (projection == "mlp") {
    auto result = gem16gb::internal::RunLayer0Nvfp4MlpCheckpointProbe(
        model_directory, warmups, iterations);
    if (!result.ok()) {
      std::cerr << "error: " << result.status().message() << '\n';
      return 1;
    }
    const auto& probe = result.value();
    std::cerr << "NVFP4 complete Layer-0 MLP checkpoint probe\n"
              << "  path: quantize -> Gate/Up -> GELU-tanh product -> quantize -> Down -> residual\n"
              << "  CPU/GPU input bytes: "
              << (probe.input_activation_bytes_match ? "exact match" : "MISMATCH") << '\n'
              << "  CPU/GPU reference Down-input bytes: "
              << (probe.reference_down_activation_bytes_match ? "exact match" : "MISMATCH")
              << '\n'
              << "  reference/native Down-input differing bytes: "
              << probe.native_down_activation_mismatched_bytes << '\n'
              << "  reference/native final max abs: " << probe.reference_native_max_abs << '\n'
              << "  SM120 direct MLP average: " << probe.sm120_direct_ms << " ms\n"
              << "  unfused Gate/Up average: " << probe.sm120_unfused_gate_up_ms << " ms\n"
              << "  fused Gate/Up average: " << probe.sm120_fused_gate_up_ms << " ms\n";
    std::cout << std::setprecision(17)
              << "{\"schema_version\":1,\"status\":\"characterization\","
              << "\"benchmark_qualified\":false,\"mode\":\"kernel\",\"fallbacks\":0,"
              << "\"operator\":\"nvfp4_layer0_mlp\",\"shape\":{\"hidden\":3840,"
              << "\"intermediate\":15360},\"precision\":\"w4a4_nvfp4\","
              << "\"instruction\":" << std::quoted(probe.instruction)
              << ",\"source_layout_direct\":true,\"persistent_repack_bytes\":0"
              << ",\"device_bytes\":" << probe.device_bytes
              << ",\"input_activation_bytes_match\":"
              << (probe.input_activation_bytes_match ? "true" : "false")
              << ",\"reference_down_activation_bytes_match\":"
              << (probe.reference_down_activation_bytes_match ? "true" : "false")
              << ",\"native_down_activation_mismatched_bytes\":"
              << probe.native_down_activation_mismatched_bytes
              << ",\"timing_ms\":{\"warmups\":" << warmups
              << ",\"iterations\":" << iterations
              << ",\"cuda_reference_single\":" << probe.cuda_reference_ms
              << ",\"sm120_direct_average\":" << probe.sm120_direct_ms
              << ",\"sm120_unfused_gate_up_average\":"
              << probe.sm120_unfused_gate_up_ms
              << ",\"sm120_fused_gate_up_average\":"
              << probe.sm120_fused_gate_up_ms << '}'
              << ",\"error\":{\"reference_native_max_abs\":"
              << probe.reference_native_max_abs
              << ",\"reference_native_rms\":" << probe.reference_native_rms
              << ",\"reference_native_cosine\":" << probe.reference_native_cosine
              << ",\"oracle_reference_max_abs\":" << probe.oracle_reference_max_abs
              << ",\"oracle_native_max_abs\":" << probe.oracle_native_max_abs << '}'
              << ",\"samples\":[";
    for (std::size_t index = 0; index < probe.samples.size(); ++index) {
      if (index != 0) std::cout << ',';
      const auto& sample = probe.samples[index];
      std::cout << "{\"row\":" << sample.row << ",\"oracle\":" << sample.oracle
                << ",\"cuda_reference\":" << sample.cuda_reference
                << ",\"sm120_direct\":" << sample.sm120_direct << '}';
    }
    std::cout << "]}\n";
    return probe.input_activation_bytes_match &&
                   probe.reference_down_activation_bytes_match
               ? 0
               : 1;
  }
  if (projection == "q" || projection == "k" || projection == "v" || projection == "o") {
    auto result = gem16gb::internal::RunLayer0Fp8CheckpointProbe(
        model_directory, projection, warmups, iterations);
    if (!result.ok()) {
      std::cerr << "error: " << result.status().message() << '\n';
      return 1;
    }
    const auto& probe = result.value();
    std::cerr << "FP8 layer-0 " << projection << " checkpoint probe\n"
              << "  tensor: " << probe.tensor_name << '\n'
              << "  shape: " << probe.rows << " x " << probe.contracting_elements << '\n'
              << "  CPU/GPU activation bytes and scale: "
              << (probe.activation_bytes_match && probe.activation_scale_match ? "exact match"
                                                                                : "MISMATCH")
              << '\n'
              << "  CUDA reference/native max abs: " << probe.reference_native_max_abs << '\n'
              << "  SM120 direct average: " << probe.sm120_direct_ms << " ms\n";
    std::cout << std::setprecision(17)
              << "{\"schema_version\":1,\"status\":\"characterization\","
              << "\"benchmark_qualified\":false,\"mode\":\"kernel\",\"fallbacks\":0,"
              << "\"operator\":" << std::quoted("fp8_layer0_" + projection)
              << ",\"tensor\":" << std::quoted(probe.tensor_name)
              << ",\"shape\":[" << probe.rows << ',' << probe.contracting_elements << ']'
              << ",\"precision\":\"w8a8_fp8_e4m3\",\"instruction\":"
              << std::quoted(probe.instruction)
              << ",\"source_layout_direct\":true,\"persistent_repack_bytes\":0"
              << ",\"weight_bytes\":" << probe.weight_bytes
              << ",\"weight_scale_bytes\":" << probe.weight_scale_bytes
              << ",\"device_bytes\":" << probe.device_bytes
              << ",\"activation_scale\":" << probe.activation_scale
              << ",\"activation_bytes_match\":"
              << (probe.activation_bytes_match ? "true" : "false")
              << ",\"activation_scale_match\":"
              << (probe.activation_scale_match ? "true" : "false")
              << ",\"timing_ms\":{\"warmups\":" << warmups
              << ",\"iterations\":" << iterations
              << ",\"activation_quantize_average\":" << probe.activation_quantize_ms
              << ",\"cuda_reference_single\":" << probe.cuda_reference_ms
              << ",\"sm120_direct_average\":" << probe.sm120_direct_ms << '}'
              << ",\"error\":{\"reference_native_max_abs\":"
              << probe.reference_native_max_abs
              << ",\"reference_native_rms\":" << probe.reference_native_rms
              << ",\"reference_native_cosine\":" << probe.reference_native_cosine
              << ",\"oracle_reference_max_abs\":" << probe.oracle_reference_max_abs
              << ",\"oracle_native_max_abs\":" << probe.oracle_native_max_abs << '}'
              << ",\"samples\":[";
    for (std::size_t index = 0; index < probe.samples.size(); ++index) {
      if (index != 0) std::cout << ',';
      const auto& sample = probe.samples[index];
      std::cout << "{\"row\":" << sample.row << ",\"oracle\":" << sample.oracle
                << ",\"cuda_reference\":" << sample.cuda_reference
                << ",\"sm120_direct\":" << sample.sm120_direct << '}';
    }
    std::cout << "]}\n";
    return probe.activation_bytes_match && probe.activation_scale_match ? 0 : 1;
  }
  auto result = gem16gb::internal::RunLayer0Nvfp4CheckpointProbe(
      model_directory, projection, warmups, iterations);
  if (!result.ok()) {
    std::cerr << "error: " << result.status().message() << '\n';
    return 1;
  }
  const auto& probe = result.value();
  std::cerr << "NVFP4 layer-0 " << projection << " checkpoint probe\n"
            << "  tensor: " << probe.tensor_name << '\n'
            << "  shape: " << probe.rows << " x " << probe.contracting_elements << '\n'
            << "  CPU/GPU activation bytes: "
            << (probe.activation_bytes_match ? "exact match" : "MISMATCH") << '\n'
            << "  CUDA reference/native max abs: " << probe.reference_native_max_abs << '\n'
            << "  CUDA reference/SIMT max abs: " << probe.reference_simt_max_abs << '\n'
            << "  SM120 direct average: " << probe.sm120_direct_ms << " ms\n"
            << "  SIMT GEMV average: " << probe.simt_gemv_ms << " ms\n";

  std::cout << std::setprecision(17)
            << "{\"schema_version\":1,\"status\":\"characterization\","
            << "\"benchmark_qualified\":false,\"mode\":\"kernel\",\"fallbacks\":0,"
            << "\"operator\":" << std::quoted("nvfp4_layer0_" + projection)
            << ",\"tensor\":"
            << std::quoted(probe.tensor_name)
            << ",\"shape\":[" << probe.rows << ',' << probe.contracting_elements << ']'
            << ",\"precision\":\"w4a4_nvfp4\",\"instruction\":"
            << std::quoted(probe.instruction)
            << ",\"source_layout_direct\":true,\"persistent_repack_bytes\":0"
            << ",\"packed_weight_bytes\":" << probe.packed_weight_bytes
            << ",\"weight_scale_bytes\":" << probe.weight_scale_bytes
            << ",\"device_bytes\":" << probe.device_bytes
            << ",\"input_global_divisor\":" << probe.input_global_divisor
            << ",\"weight_global_divisor\":" << probe.weight_global_divisor
            << ",\"activation_bytes_match\":"
            << (probe.activation_bytes_match ? "true" : "false")
            << ",\"timing_ms\":{\"warmups\":" << warmups
            << ",\"iterations\":" << iterations
            << ",\"activation_quantize_average\":" << probe.activation_quantize_ms
            << ",\"cuda_reference_single\":" << probe.cuda_reference_ms
            << ",\"sm120_direct_average\":" << probe.sm120_direct_ms
            << ",\"simt_gemv_average\":" << probe.simt_gemv_ms << '}'
            << ",\"error\":{\"reference_native_max_abs\":"
            << probe.reference_native_max_abs
            << ",\"reference_native_rms\":" << probe.reference_native_rms
            << ",\"reference_native_cosine\":" << probe.reference_native_cosine
            << ",\"oracle_reference_max_abs\":" << probe.oracle_reference_max_abs
            << ",\"oracle_native_max_abs\":" << probe.oracle_native_max_abs
            << ",\"reference_simt_max_abs\":" << probe.reference_simt_max_abs
            << ",\"reference_simt_rms\":" << probe.reference_simt_rms
            << ",\"reference_simt_cosine\":" << probe.reference_simt_cosine
            << ",\"oracle_simt_max_abs\":" << probe.oracle_simt_max_abs << '}'
            << ",\"samples\":[";
  for (std::size_t index = 0; index < probe.samples.size(); ++index) {
    if (index != 0) std::cout << ',';
    const auto& sample = probe.samples[index];
    std::cout << "{\"row\":" << sample.row << ",\"oracle\":" << sample.oracle
              << ",\"cuda_reference\":" << sample.cuda_reference
              << ",\"sm120_direct\":" << sample.sm120_direct
              << ",\"simt_gemv\":" << sample.simt_gemv << '}';
  }
  std::cout << "]}\n";
  return probe.activation_bytes_match ? 0 : 1;
#else
  std::cerr << "error: kernel mode requires a CUDA build\n";
  return 2;
#endif
}

}  // namespace

int main(int argc, char** argv) {
  constexpr std::array modes = {
      "model-load", "prefill", "decode", "end-to-end", "kernel", "memory", "quality", "mtp"};
  if (argc == 2 && (std::string_view(argv[1]) == "--help" || std::string_view(argv[1]) == "-h")) {
    Usage(std::cout);
    return 0;
  }
  const std::string_view requested = argc > 1 ? std::string_view(argv[1]) : std::string_view{};
  if (requested == "memory") return RunMemoryMode(argc, argv);
  if (requested == "kernel") return RunKernelMode(argc, argv);
  if (requested == "decode") return RunDecodeMode(argc, argv);

  bool known = false;
  for (const auto mode : modes) known = known || requested == mode;
  std::cerr << "{\"schema_version\":1,\"status\":\"not_implemented\",\"mode\":\""
            << (known ? requested : "unknown") << "\",\"fallbacks\":0}\n";
  return known ? 2 : 64;
}
