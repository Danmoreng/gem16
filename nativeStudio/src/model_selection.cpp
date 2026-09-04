#include "model_selection.h"

#include <stdexcept>

#include "chat_store.h"
#include "settings.h"
#include "util/json.h"
namespace gem16::studio {
namespace {
std::string Text(const json::Value& value, const char* key) {
  auto p = value.find(key);
  if (!p || !p->is_string())
    throw std::runtime_error("Invalid saved model selection.");
  return p->as_string();
}
std::int64_t Number(const json::Value& value, const char* key) {
  auto p = value.find(key);
  if (!p || !p->is_integer())
    throw std::runtime_error("Invalid saved model setting.");
  return p->as_integer();
}
bool Flag(const json::Value& value, const char* key) {
  auto p = value.find(key);
  if (!p || !p->is_bool())
    throw std::runtime_error("Invalid saved model flag.");
  return p->as_bool();
}
}  // namespace
bool UsesCatalogPaths(const ServerConfig& c) {
  return c.model_directory == ProfileTargetDirectory(c.profile).string() &&
         (c.assistant_directory.empty() ||
          c.assistant_directory ==
              ProfileAssistantDirectory(c.profile).string()) &&
         c.vision_directory == ProfileVisionDirectory(c.profile).string();
}
ServerConfig SavedServerSelection(const std::string& record,
                                  const ServerConfig& local) {
  auto parsed = json::Parse(record, {32, 10000, 128U * 1024U});
  if (!parsed.ok()) throw std::runtime_error("Invalid saved model selection.");
  const auto& data = parsed.value();
  auto identity = Text(data, "model");
  auto model = json::Parse(identity);
  if (!model.ok()) throw std::runtime_error("Invalid saved model identity.");
  const auto profile = Text(model.value(), "profile");
  ModelProfile selected;
  if (profile == "gemma4_12b")
    selected = ModelProfile::kGemma4Unified12B;
  else if (profile == "gemma4_26b_trellis35_vision_fp8")
    selected = ModelProfile::kGemma4Moe26BTrellis35VisionFp8;
  else
    throw std::runtime_error(
        "This saved profile is not a public Studio selection.");
  ServerConfig c = local;
  ApplyProfileDefaults(c, selected);
  c.model_directory = Text(model.value(), "model_path");
  c.assistant_directory = Text(model.value(), "assistant_path");
  c.vision_directory = Text(model.value(), "vision_path");
  if (!UsesCatalogPaths(c) || ModelIdentity(c) != identity)
    throw std::runtime_error(
        "This revision is not in the current qualified catalog. Its chat "
        "remains readable; no model will be substituted.");
  c.model_name = Text(data, "model_name");
  c.max_context_tokens = Number(data, "context");
  const auto draft = Number(data, "draft_tokens"),
             vision = Number(data, "vision_budget");
  if (c.max_context_tokens < 1 || c.max_context_tokens > 262144 ||
      (draft != 0 && draft != 2) ||
      (vision != 70 && vision != 140 && vision != 280) ||
      c.model_name.size() > 255)
    throw std::runtime_error("Unsupported saved model configuration.");
  c.mtp_draft_tokens = static_cast<int>(draft);
  c.vision_soft_token_budget = static_cast<int>(vision);
  c.greedy = Flag(data, "greedy");
  c.mtp_adaptive = Flag(data, "adaptive");
  if (const auto sessions = data.find("max_sessions")) {
    if (!sessions->is_integer() || sessions->as_integer() < 1 ||
        sessions->as_integer() > 2)
      throw std::runtime_error("Unsupported saved session count.");
    c.max_sessions = static_cast<int>(sessions->as_integer());
  }
  return c;
}
}  // namespace gem16::studio
