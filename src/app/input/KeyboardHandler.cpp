#include "KeyboardHandler.hpp"

#include <GLFW/glfw3.h>
#include <fmt/format.h>
#include <imgui.h>

#include <cstddef>
#include <cstdint>
#include <drip/common/log/LogMessageBuilder.hpp>
#include <drip/common/utils/Assert.hpp>
#include <drip/engine/utils/Signals.hpp>
#include <utility>

#include "ui/Window.hpp"
#include "utils/Signals.hpp"

namespace
{

void keyboardStateChangedCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    static auto sender = drip::app::signal::keyboardStateChanged.registerSender();

    const auto windowId = drip::app::Window::makeId(window);
    drip::common::log::Debug("Keyboard state for window [{}] changed to {};{};{};{}",
                             windowId,
                             key,
                             scancode,
                             action,
                             mods);

    if (!ImGui::GetIO().WantCaptureKeyboard)
    {
        sender(drip::app::signal::KeyboardStateChangedData {.id = windowId,
                                                            .key = key,
                                                            .scancode = scancode,
                                                            .action = action,
                                                            .mods = mods});
    }
}

}

namespace drip::app
{

KeyboardHandler::KeyboardHandler(const Window& window)
    : _window {window}
{
    _states.fill(State::Released);

    [[maybe_unused]] static const auto oldKeyCallback = _window.setKeyCallback(keyboardStateChangedCallback);

    _keyboardStateChangedReceiver = signal::keyboardStateChanged.connect([this](const auto& data) {
        handleKeyboardState(data);
    });

    _newFrameNotifReceiver = engine::signal::gameLoopIterationStarted.connect([this] {
        handleGameLoopIteration();
    });
}

auto KeyboardHandler::getKeyState(int key) const -> State
{
    common::expect(
        key,
        [this](auto userKey) {
            return userKey < static_cast<int>(_states.size()) && userKey >= 0;
        },
        fmt::format("Key: {} is beyond the size of array", key));

    return _states[static_cast<size_t>(key)];
}

void KeyboardHandler::handleKeyboardState(const signal::KeyboardStateChangedData& data)
{
    if (data.id != _window.getId())
    {
        return;
    }

    if (const auto isCorrectKey = data.key >= 0 && std::cmp_less(data.key, _states.size());
        !common::shouldBe(isCorrectKey, fmt::format("Key: {} is beyond the size of array", data.key)))
    {
        return;
    }

    if ((_states[static_cast<size_t>(data.key)] == State::JustReleased ||
         _states[static_cast<size_t>(data.key)] == State::Released) &&
        (data.action == GLFW_PRESS || data.action == GLFW_REPEAT))
    {
        _states[static_cast<size_t>(data.key)] = State::JustPressed;
    }
    else if ((_states[static_cast<size_t>(data.key)] == State::JustPressed ||
              _states[static_cast<size_t>(data.key)] == State::Pressed) &&
             data.action == GLFW_RELEASE)
    {
        _states[static_cast<size_t>(data.key)] = State::JustReleased;
    }
}

void KeyboardHandler::handleGameLoopIteration()
{
    for (auto i = uint32_t {}; i < _states.size(); i++)
    {
        if (_states[i] == State::JustReleased)
        {
            _states[i] = State::Released;
        }
        if (_states[i] == State::JustPressed)
        {
            _states[i] = State::Pressed;
        }
    }
}

}
