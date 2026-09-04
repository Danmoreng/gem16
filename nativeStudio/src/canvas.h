#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace gem16::studio {
inline constexpr std::size_t kCanvasSourceLimit = 1024U * 1024U;
struct CanvasRevision {
  std::int64_t number = 0;
  std::string source;
};
struct CanvasDocument {
  std::string id, title, type;
  std::vector<CanvasRevision> revisions;
};
struct ToolCall {
  std::string id, name, arguments;
};
void MergeToolDelta(std::vector<ToolCall>& calls, const std::string& delta);
std::string ToolCallsJson(const std::vector<ToolCall>& calls);
std::vector<ToolCall> ParseToolCalls(const std::string& json);
void ValidateCanvases(const std::vector<CanvasDocument>& documents);
std::string CanvasTools();
std::string CanvasInstructions(const std::vector<CanvasDocument>& documents);
// create/read/edit are deterministic host operations. Check is handled by the
// browser.
std::string ExecuteCanvasTool(std::vector<CanvasDocument>& documents,
                              const ToolCall& call);
CanvasDocument& FindCanvas(std::vector<CanvasDocument>& documents,
                           const std::string& id);
}  // namespace gem16::studio
