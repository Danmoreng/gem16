#pragma once

namespace gem16::studio {

class ShaderBackground final {
 public:
  ShaderBackground() = default;
  ~ShaderBackground();
  ShaderBackground(const ShaderBackground&) = delete;
  ShaderBackground& operator=(const ShaderBackground&) = delete;

  [[nodiscard]] bool Initialize(void* device, void* context);
  void Render(float seconds, float width, float height, bool dark_theme);
  void Shutdown();
  [[nodiscard]] bool Ready() const;

 private:
  struct Impl;
  Impl* impl_ = nullptr;
};

}  // namespace gem16::studio

