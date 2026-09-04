#pragma once
#include <memory>
#include <string>

#include "canvas.h"
#include "image_texture.h"
namespace gem16::studio {
// Initialize the system WebView on the UI thread. No bundled browser runtime.
int InitializeCanvasBrowser(int argc, char** argv);
void PumpCanvasBrowser();
void ShutdownCanvasBrowser();
bool CanvasBrowserAvailable();
class CanvasBrowser {
 public:
  CanvasBrowser();
  ~CanvasBrowser();
  void Load(const CanvasDocument& document);
  void Close();
  bool Ready() const;
  std::string Key() const;
  std::string Diagnostics() const;
  DecodedImage Pixels() const;
  std::vector<std::uint8_t> Screenshot() const;
  void Mouse(int x, int y, int button, bool up, bool move, int wheel = 0);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace gem16::studio
namespace gem16::studio {
int RunCanvasBrowserSmoke(const std::string& output);
}
