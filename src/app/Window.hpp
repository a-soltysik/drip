#pragma once

#include <GLFW/glfw3.h>
#include <vulkan/vulkan_core.h>

#include <drip/engine/Window.hpp>
#include <drip/engine/utils/Signals.hpp>
#include <glm/ext/vector_uint2.hpp>
#include <vector>

namespace drip::app
{

class Window : public engine::Window
{
public:
    [[nodiscard]] static auto makeId(GLFWwindow* window) -> Id;

    Window(glm::uvec2 size, const char* name);
    Window(const Window&) = delete;
    auto operator=(const Window&) -> Window& = delete;
    Window(Window&&) noexcept;
    auto operator=(Window&&) -> Window& = delete;
    ~Window() noexcept override;

    [[nodiscard]] auto shouldClose() const -> bool override;
    [[nodiscard]] auto isMinimized() const -> bool override;
    [[nodiscard]] auto getSize() const -> glm::uvec2 override;
    [[nodiscard]] auto getRequiredExtensions() const -> std::vector<const char*> override;
    [[nodiscard]] auto createSurface(VkInstance instance) const -> VkSurfaceKHR override;
    [[nodiscard]] auto getId() const -> Id override;
    auto processInput() -> void override;
    auto waitForInput() -> void override;

private:
    [[nodiscard]] static auto createWindow(glm::uvec2 size, const char* name) -> GLFWwindow*;
    auto setupImGui() const -> void;

    engine::signal::FrameBufferResized::ReceiverT _frameBufferResizedReceiver;
    GLFWwindow* _window;
    glm::uvec2 _size;
};

}
