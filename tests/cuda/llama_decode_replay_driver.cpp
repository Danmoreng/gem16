// Standalone diagnostic, compiled against the pinned external llama headers and
// libraries. It is not linked into gem16 or exposed by either product CLI.
#include "decode_replay_support.h"
#include "llama.h"
#include "ggml-backend.h"
#include <memory>

namespace {
using namespace decode_replay;
struct Batch {
  llama_batch value = llama_batch_init(2048, 0, 1);
  ~Batch() { llama_batch_free(value); }
  void Set(std::span<const std::uint32_t> tokens, std::uint32_t position,
           bool all_outputs = false) {
    Require(!tokens.empty() && tokens.size() <= 2048U, "invalid diagnostic batch");
    value.n_tokens = static_cast<std::int32_t>(tokens.size());
    for (std::size_t i = 0U; i < tokens.size(); ++i) {
      value.token[i] = static_cast<llama_token>(tokens[i]);
      value.pos[i] = static_cast<llama_pos>(position + i);
      value.n_seq_id[i] = 1;
      value.seq_id[i][0] = 0;
      value.logits[i] = all_outputs || i + 1U == tokens.size();
    }
  }
};
void Decode(llama_context* ctx, Batch& batch,
            std::span<const std::uint32_t> tokens, std::uint32_t position,
            bool all_outputs = false) {
  batch.Set(tokens, position, all_outputs);
  Require(llama_decode(ctx, batch.value) == 0, "llama diagnostic decode failed");
  llama_synchronize(ctx);
}
void Prefill(llama_context* ctx, Batch& batch,
             std::span<const std::uint32_t> prompt) {
  llama_memory_clear(llama_get_memory(ctx), true);
  for (std::size_t i = 0U; i < prompt.size(); i += 2048U) {
    Decode(ctx, batch, prompt.subspan(i, std::min<std::size_t>(2048U, prompt.size() - i)),
           static_cast<std::uint32_t>(i));
  }
}
std::uint32_t Prediction(llama_context* ctx, int row = -1) {
  const float* values = llama_get_logits_ith(ctx, row);
  Require(values != nullptr, "llama diagnostic logits unavailable");
  return Winner({values, kVocabulary});
}
}  // namespace

int main(int argc, char** argv) {
  using namespace decode_replay;
  if (argc != 5) return 2;
  try {
    const auto prompt = ReadTokens(argv[2], 16384U);
    const auto history = ReadTokens(argv[3], 1135U);
    const bool windows = std::string_view(argv[4]) == "d2-windows";
    const bool d2 = windows || std::string_view(argv[4]) == "d2";
    Require(d2 || std::string_view(argv[4]) == "ordinary", "invalid replay mode");
    const char* backend_dir = std::getenv("GEM16_LLAMA_BACKEND_DIR");
    Require(backend_dir != nullptr, "pinned GEM16_LLAMA_BACKEND_DIR is required");
    ggml_backend_load_all_from_path(backend_dir);
    llama_backend_init();
    auto gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    Require(gpu != nullptr, "diagnostic requires the CUDA GPU");
    const std::array<llama_model_tensor_buft_override, 2> overrides{{
        {"token_embd.weight", ggml_backend_dev_buffer_type(gpu)}, {nullptr, nullptr}}};
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    mp.tensor_buft_overrides = overrides.data();
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(argv[1], mp), llama_model_free);
    Require(model != nullptr, "pinned llama model load failed");
    Require(llama_vocab_n_tokens(llama_model_get_vocab(model.get())) == kVocabulary,
            "wrong llama vocabulary");
    auto cp = llama_context_default_params();
    cp.n_ctx = kContext;
    cp.n_batch = 2048U;
    cp.n_ubatch = 512U;
    cp.n_threads = 8;
    cp.n_threads_batch = 8;
    // Match the pinned server's bounded SWA cache, rather than C API defaults.
    cp.swa_full = false;
    cp.kv_unified = false;
    cp.n_seq_max = 1U;
    cp.n_outputs_max = d2 ? 3U : 1U;
    cp.n_outputs_max_per_seq = cp.n_outputs_max;
    cp.type_k = GGML_TYPE_Q8_0;
    cp.type_v = GGML_TYPE_Q8_0;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.offload_kqv = true;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
        llama_init_from_model(model.get(), cp), llama_free);
    Require(ctx != nullptr && llama_n_ctx(ctx.get()) >= kContext,
            "llama diagnostic context creation failed");
    Batch batch;
    std::vector<Row> rows(1134U);
    if (!d2) {
      Prefill(ctx.get(), batch, prompt);
      const auto initial = Prediction(ctx.get());
      for (std::size_t i = 0U; i < rows.size(); ++i) {
        const auto start = Clock::now();
        Decode(ctx.get(), batch, std::span(history).subspan(i, 1U),
               static_cast<std::uint32_t>(prompt.size() + i));
        rows[i] = {history[i], Prediction(ctx.get()), Ns(start)};
      }
      WriteRows("llama-b11552-q4_0", rows, initial);
    } else if (windows) {
      constexpr std::array<std::size_t, 3> starts{0U, 567U, 1128U};
      std::cout << "{\"diagnostic_only\":true,\"engine\":\"llama-b11552-q4_0\","
                   "\"mode\":\"forced_d2_windows\",\"cases\":[";
      for (unsigned phase = 0U; phase < starts.size(); ++phase) {
        Prefill(ctx.get(), batch, prompt);
        for (std::size_t i = 0U; i < starts[phase]; ++i) {
          Decode(ctx.get(), batch, std::span(history).subspan(i, 1U),
                 static_cast<std::uint32_t>(16384U + i));
        }
        std::array<std::uint32_t, 3> inputs{
            history[starts[phase]], history[starts[phase] + 1U], history[starts[phase] + 2U]};
        Decode(ctx.get(), batch, inputs, static_cast<std::uint32_t>(16384U + starts[phase]), true);
        unsigned accepted = 0U;
        std::array<std::uint32_t, 3> verified{};
        for (unsigned row = 0U; row < 3U; ++row) {
          verified[row] = Prediction(ctx.get(), static_cast<int>(row));
          if (row < 2U && verified[row] == inputs[row + 1U]) ++accepted; else break;
        }
        const auto committed = static_cast<llama_pos>(16384U + starts[phase] + accepted + 1U);
        Require(llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, committed, -1), "window rollback failed");
        Decode(ctx.get(), batch, std::span(verified).subspan(accepted, 1U), static_cast<std::uint32_t>(committed));
        std::cout << (phase ? "," : "") << "{\"position\":" << 16384U + starts[phase]
                  << ",\"accepted\":" << accepted << ",\"outputs\":[";
        for (unsigned i = 0U; i <= accepted; ++i) std::cout << (i ? "," : "") << verified[i];
        std::cout << "],\"after_commit_prediction\":" << Prediction(ctx.get()) << '}';
      }
      std::cout << "]}\n";
    } else {
      for (std::uint32_t accepted = 0U; accepted < 3U; ++accepted) {
        Prefill(ctx.get(), batch, prompt);
        std::array<std::uint32_t, 3> inputs{history[0], history[1], history[2]};
        if (accepted < 2U) inputs[accepted + 1U] = (inputs[accepted + 1U] + 1U) % kVocabulary;
        Decode(ctx.get(), batch, inputs, 16384U, true);
        std::array<std::uint32_t, 3> verified{};
        std::uint32_t actual = 0U;
        for (std::uint32_t row = 0U; row < 3U; ++row) {
          verified[row] = Prediction(ctx.get(), static_cast<int>(row));
          if (row < 2U && verified[row] == inputs[row + 1U]) ++actual;
          else break;
        }
        Require(actual == accepted, "unexpected llama forced acceptance");
        for (std::uint32_t i = 0U; i <= accepted; ++i) {
          Require(verified[i] == history[i + 1U], "llama forced output mismatch");
        }
        const auto committed = 16384U + accepted + 1U;
        Require(llama_memory_seq_rm(llama_get_memory(ctx.get()), 0,
                                   static_cast<llama_pos>(committed), -1),
                "llama diagnostic rollback unsupported");
        Decode(ctx.get(), batch, std::span(verified).subspan(accepted, 1U), committed);
        Require(Prediction(ctx.get()) == history[accepted + 2U],
                "llama post-rollback continuation mismatch");
      }
      std::cout << "{\"diagnostic_only\":true,\"engine\":\"llama-b11552-q4_0\","
                   "\"mode\":\"forced_d2\",\"accepted\":[0,1,2],"
                   "\"commit_and_continuation_exact\":true}\n";
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "llama diagnostic replay: " << error.what() << '\n';
    return 1;
  }
}
