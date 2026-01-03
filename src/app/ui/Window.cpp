#include "Window.hpp"

#include <GLFW/glfw3.h>
#include <backends/imgui_impl_glfw.h>
#include <imgui.h>
#include <vulkan/vulkan_core.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <drip/common/log/LogMessageBuilder.hpp>
#include <drip/common/utils/Assert.hpp>
#include <drip/engine/Window.hpp>
#include <drip/engine/utils/Signals.hpp>
#include <glm/ext/vector_uint2.hpp>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "input/KeyboardHandler.hpp"
#include "input/MouseHandler.hpp"

namespace
{

void framebufferResizeCallback(GLFWwindow* window, int width, int height)
{
    static const auto sender = drip::engine::signal::frameBufferResized.registerSender();
    const auto windowId = drip::app::Window::makeId(window);
    drip::common::log::Info("Size of window [{}] changed to {}x{}", windowId, width, height);

    sender(drip::engine::signal::FrameBufferResizedData {.id = windowId, .x = width, .y = height});
}

}

namespace drip::app
{

Window::Window(const glm::uvec2 size, const char* name)
    : _window {createWindow(size, name)},
      _size {size}
{
    _keyboardHandler = std::make_unique<KeyboardHandler>(*this);
    _mouseHandler = std::make_unique<MouseHandler>(*this);
    glfwSetFramebufferSizeCallback(_window, framebufferResizeCallback);

    _frameBufferResizedReceiver = engine::signal::frameBufferResized.connect([this](auto data) {
        if (data.id == getId())
        {
            common::log::Debug("Received framebuffer resized notif");
            _size = {data.x, data.y};
        }
    });

    setupImGui();
}

Window::Window(Window&& rhs) noexcept
    : _keyboardHandler {std::move(rhs._keyboardHandler)},
      _mouseHandler {std::move(rhs._mouseHandler)},
      _frameBufferResizedReceiver {std::move(rhs._frameBufferResizedReceiver)},
      _window {std::exchange(rhs._window, nullptr)},
      _size {std::exchange(rhs._size, {})}
{
}

Window::~Window() noexcept
{
    if (_window == nullptr)
    {
        return;
    }
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(_window);
    glfwTerminate();
    common::log::Info("Window [{}] destroyed", static_cast<void*>(_window));
}

auto Window::createWindow(glm::uvec2 size, const char* name) -> GLFWwindow*
{
    glfwSetErrorCallback([](int error, const char* description) {
        common::log::Error("GLFW Error {}: {}", error, description);
    });

    common::Expect(glfwInit(), GLFW_TRUE, "Failed to initialize GLFW");

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    auto* window = glfwCreateWindow(static_cast<int>(size.x), static_cast<int>(size.y), name, nullptr, nullptr);
    common::log::Info("Window [{}] {}x{} px created", static_cast<void*>(window), size.x, size.y);

    return window;
}

auto Window::getRequiredExtensions() const -> std::vector<const char*>
{
    auto glfwExtensionsCount = uint32_t {};
    const auto* glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionsCount);

    if (!common::ShouldNotBe(glfwExtensions,
                             static_cast<const char**>(nullptr),
                             "No extension allowing surface creation was found")
             .result())
    {
        return {};
    }

    const auto extensionsSpan = std::span(glfwExtensions, glfwExtensionsCount);

    return {extensionsSpan.begin(), extensionsSpan.end()};
}

auto Window::createSurface(VkInstance instance) const -> VkSurfaceKHR
{
    auto* newSurface = VkSurfaceKHR {};
    glfwCreateWindowSurface(instance, _window, nullptr, &newSurface);

    return common::ExpectNot(newSurface, static_cast<VkSurfaceKHR>(nullptr), "Unable to create surface").result();
}

auto Window::shouldClose() const -> bool
{
    return glfwWindowShouldClose(_window) == GLFW_TRUE;
}

auto Window::processInput() -> void
{
    glfwPollEvents();
}

auto Window::getSize() const -> glm::uvec2
{
    return _size;
}

auto Window::isMinimized() const -> bool
{
    return _size.x == 0 || _size.y == 0;
}

auto Window::waitForInput() -> void
{
    glfwWaitEvents();
}

auto Window::getId() const -> size_t
{
    return makeId(_window);
}

auto Window::makeId(GLFWwindow* window) -> engine::Window::Id
{
    return std::bit_cast<size_t>(window);
}

auto Window::setupImGui() const -> void
{
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForVulkan(_window, true);
}

auto Window::setKeyCallback(GLFWkeyfun callback) const noexcept -> GLFWkeyfun
{
    return glfwSetKeyCallback(_window, callback);
}

auto Window::setMouseButtonCallback(GLFWmousebuttonfun callback) const noexcept -> GLFWmousebuttonfun
{
    return glfwSetMouseButtonCallback(_window, callback);
}

auto Window::setCursorPositionCallback(GLFWcursorposfun callback) const noexcept -> GLFWcursorposfun
{
    return glfwSetCursorPosCallback(_window, callback);
}

auto Window::getKeyboardHandler() const noexcept -> const KeyboardHandler&
{
    return *_keyboardHandler;
}

auto Window::getMouseHandler() const noexcept -> const MouseHandler&
{
    return *_mouseHandler;
}

}
