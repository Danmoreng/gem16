#include <cstdio>
#include <stdexcept>

#include "api_client.h"
#include "canvas.h"
#include "chat_history.h"
#include "chat_store.h"
#include "util/json.h"
using namespace gem16::studio;
namespace {
void Require(bool v, const char* m) {
  if (!v) throw std::runtime_error(m);
}
template <class F>
void Reject(F f) {
  bool failed = false;
  try {
    f();
  } catch (const std::exception&) {
    failed = true;
  }
  Require(failed, "Expected canvas rejection");
}
}  // namespace
bool TestCanvas() {
  auto root = std::filesystem::temp_directory_path() /
              ("gem16-canvas-" + NewChatId().substr(0, 16));
  try {
    std::vector<CanvasDocument> docs;
    ToolCall create{
        "c1", "canvas_create",
        R"({"title":"Hello","type":"html","source":"<h1>Hello</h1>"})"};
    ExecuteCanvasTool(docs, create);
    const auto id = docs.front().id;
    ToolCall edit{
        "c2", "canvas_edit",
        "{\"id\":\"" + id +
            R"(","revision":1,"old_text":"Hello","new_text":"Grüße"})"};
    ExecuteCanvasTool(docs, edit);
    Require(docs[0].revisions.size() == 2 &&
                docs[0].revisions.back().source == "<h1>Grüße</h1>",
            "bounded Unicode edit");
    Reject([&] { ExecuteCanvasTool(docs, edit); });
    Require(docs[0].revisions.size() == 2,
            "stale edit leaves history untouched");
    auto bad = docs;
    bad[0].revisions.back().source = std::string(kCanvasSourceLimit + 1, 'x');
    Reject([&] { ValidateCanvases(bad); });
    std::vector<ToolCall> calls;
    MergeToolDelta(
        calls,
        R"([{"index":0,"id":"call_1","function":{"name":"canvas_read","arguments":"{"}}])");
    MergeToolDelta(calls, "[{\"index\":0,\"function\":{\"arguments\":" +
                              gem16::json::Quote("\"id\":\"" + id + "\"}") +
                              "}}]");
    auto decoded = ParseToolCalls(ToolCallsJson(calls));
    Require(decoded.size() == 1 && decoded[0].id == "call_1" &&
                ExecuteCanvasTool(docs, decoded[0]).find("Grüße") !=
                    std::string::npos,
            "fragmented tool call reconstruction");
    Reject([&] { MergeToolDelta(calls, R"([{"index":8}])"); });
    Conversation c;
    c.id = NewChatId();
    c.title = "canvas";
    c.canvases = docs;
    ChatMessage user{"user", "make a page"};
    ChatMessage assistant{"assistant", ""};
    assistant.tool_calls = {create};
    ChatMessage tool{"tool", "created"};
    tool.tool_call_id = "c1";
    c.messages = {user, assistant, tool, {"assistant", "Done"}};
    auto repeated = c.messages;
    repeated.push_back(assistant);
    auto second = tool;
    second.content = "second result";
    repeated.push_back(second);
    repeated.push_back(assistant);
    Require(FindToolResult(repeated, 1, "c1")->content == "created" &&
                FindToolResult(repeated, 4, "c1")->content == "second result" &&
                !FindToolResult(repeated, 6, "c1"),
            "tool output is scoped to its assistant turn despite reused IDs");
    {
      ChatStore store(root);
      store.Save(c).get();
      auto modified = c;
      modified.canvases[0].revisions[0].source = "tampered";
      Reject([&] { store.Save(modified).get(); });
      auto loaded = store.Load(c.id).get();
      Require(loaded.canvases[0].revisions.size() == 2 &&
                  loaded.messages[1].tool_calls[0].id == "c1" &&
                  loaded.messages[2].tool_call_id == "c1",
              "SQLite revisions and tool transcript");
    }
    {
      ChatStore store(root);
      auto modified = c;
      modified.canvases[0].revisions[0].source = "tampered";
      Reject([&] { store.Save(modified).get(); });
      auto loaded = store.Load(c.id).get();
      Require(loaded.canvases[0].revisions.back().source == "<h1>Grüße</h1>",
              "restart preserves code");
      auto payload = BuildChatPayload({}, {}, loaded.messages, CanvasTools());
      Require(payload.find("\"tools\"") != std::string::npos &&
                  payload.find("\"tool_call_id\":\"c1\"") != std::string::npos,
              "tool wire history");
      Require(payload.find("\"content\":null") != std::string::npos,
              "tool-only assistant has no empty text part");
      loaded.messages[2].error = true;
      loaded.messages.push_back({"user", "continue"});
      payload = BuildChatPayload({}, {}, loaded.messages, CanvasTools());
      Require(payload.find("tool_call_id") == std::string::npos &&
                  payload.find("make a page") == std::string::npos &&
                  payload.find("continue") != std::string::npos,
              "cancelled tool exchange is excluded as a unit");
      auto incomplete = c.messages;
      incomplete.resize(2);
      incomplete.push_back({"user", "after restart"});
      payload = BuildChatPayload({}, {}, incomplete);
      Require(payload.find("tool_calls") == std::string::npos,
              "incomplete recovered calls are not replayed");
      store.Delete(c.id).get();
      Require(store.List().get().empty(), "delete owning conversation");
    }
    std::filesystem::remove_all(root);
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "Canvas tests: %s\n", e.what());
    std::filesystem::remove_all(root);
    return false;
  }
}
