#pragma once

#include <GLFW/glfw3.h>

#include <array>
#include <cstdint>
#include <drip/common/utils/Signal.hpp>
#include <drip/engine/utils/Signals.hpp>

#include "utils/Signals.hpp"

namespace drip::app
{

class Window;

class KeyboardHandler
{
public:
    enum class State : uint8_t
    {
        JustReleased,
        Released,
        Pressed,
        JustPressed
    };

    explicit KeyboardHandler(const Window& window);

    [[nodiscard]] auto getKeyState(int key) const -> State;

private:
    void handleKeyboardState(const signal::KeyboardStateChangedData& data);
    void handleGameLoopIteration();

    std::array<State, GLFW_KEY_LAST> _states {};
    signal::KeyboardStateChanged::Signal::ReceiverT _keyboardStateChangedReceiver;
    engine::signal::GameLoopIterationStarted::ReceiverT _newFrameNotifReceiver;
    const Window& _window;
};

}
