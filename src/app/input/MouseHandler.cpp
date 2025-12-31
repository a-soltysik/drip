#include "MouseHandler.hpp"

#include <GLFW/glfw3.h>
#include <fmt/format.h>
#include <imgui.h>

#include <cstddef>
#include <drip/common/log/LogMessageBuilder.hpp>
#include <drip/common/utils/Assert.hpp>
#include <drip/engine/utils/Signals.hpp>
#include <glm/ext/vector_float2.hpp>
#include <utility>

#include "ui/Window.hpp"
#include "utils/Signals.hpp"

namespace
{

void mouseButtonStateChangedCallback(GLFWwindow* window, int key, int action, int mods)
{
    static auto sender = drip::app::signal::mouseButtonStateChanged.registerSender();
    const auto windowId = drip::app::Window::makeId(window);
    drip::common::log::Debug("Mouse button state for window [{}] changed to {};{};{}", windowId, key, action, mods);
    if (!ImGui::GetIO().WantCaptureMouse)
    {
        sender(drip::app::signal::MouseButtonStateChangedData {.id = windowId,
                                                               .button = key,
                                                               .action = action,
                                                               .mods = mods});
    }
}

void cursorPositionChangedCallback(GLFWwindow* window, double x, double y)
{
    static auto sender = drip::app::signal::cursorPositionChanged.registerSender();
    const auto windowId = drip::app::Window::makeId(window);
    drip::common::log::Debug("Cursor position for window [{}] changed to ({}, {})", windowId, x, y);

    if (!ImGui::GetIO().WantCaptureMouse)
    {
        sender(drip::app::signal::CursorPositionChangedData {.id = windowId, .x = x, .y = y});
    }
}

}

namespace drip::app
{

MouseHandler::MouseHandler(const Window& window)
    : _previousPosition {_currentPosition},
      _window {window}
{
    _states.fill(ButtonState::Released);

    _window.setMouseButtonCallback(mouseButtonStateChangedCallback);
    _window.setCursorPositionCallback(cursorPositionChangedCallback);

    _mouseButtonStateChangedReceiver = signal::mouseButtonStateChanged.connect([this](const auto& data) {
        handleMouseButtonState(data);
    });

    _cursorStateChangedReceiver = signal::cursorPositionChanged.connect([this](const auto& data) {
        handleCursorPosition(data);
    });

    _newFrameNotifReceiver = engine::signal::gameLoopIterationStarted.connect([this] {
        handleGameLoopIteration();
    });
}

auto MouseHandler::getButtonState(int button) const -> ButtonState
{
    const auto isCorrectButton = static_cast<size_t>(button) < _states.size() && button >= 0;

    common::expect(isCorrectButton, fmt::format("Button: {} is beyond the size of array", button));

    return _states[static_cast<size_t>(button)];
}

auto MouseHandler::getCursorPosition() const -> glm::vec2
{
    return glm::vec2 {_currentPosition};
}

auto MouseHandler::getCursorDeltaPosition() const -> glm::vec2
{
    return glm::vec2 {_currentPosition - _previousPosition};
}

void MouseHandler::handleMouseButtonState(const signal::MouseButtonStateChangedData& data)
{
    if (data.id != _window.getId())
    {
        return;
    }
    if (const auto isCorrectButton = data.button >= 0 && std::cmp_less(data.button, _states.size());
        !common::shouldBe(isCorrectButton, fmt::format("Button: {} is beyond the size of array", data.button)))
    {
        return;
    }

    if ((_states[static_cast<size_t>(data.button)] == ButtonState::JustReleased ||
         _states[static_cast<size_t>(data.button)] == ButtonState::Released) &&
        data.action == GLFW_PRESS)
    {
        _states[static_cast<size_t>(data.button)] = ButtonState::JustPressed;
    }
    else if ((_states[static_cast<size_t>(data.button)] == ButtonState::JustPressed ||
              _states[static_cast<size_t>(data.button)] == ButtonState::Pressed) &&
             data.action == GLFW_RELEASE)
    {
        _states[static_cast<size_t>(data.button)] = ButtonState::JustReleased;
    }
}

void MouseHandler::handleCursorPosition(const signal::CursorPositionChangedData& data)
{
    if (data.id != _window.getId())
    {
        return;
    }
    _previousPosition = _currentPosition;
    _currentPosition = {data.x, data.y};
}

void MouseHandler::handleGameLoopIteration()
{
    _previousPosition = _currentPosition;
    for (auto& state : _states)
    {
        if (state == ButtonState::JustReleased)
        {
            state = ButtonState::Released;
        }
        if (state == ButtonState::JustPressed)
        {
            state = ButtonState::Pressed;
        }
    }
}
}
