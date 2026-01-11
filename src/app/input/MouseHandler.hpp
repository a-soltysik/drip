#pragma once

#include <GLFW/glfw3.h>

#include <array>
#include <cstdint>
#include <glm/ext/vector_double2.hpp>
#include <glm/ext/vector_float2.hpp>

#include "utils/Signals.hpp"

namespace drip::app
{

class Window;

class MouseHandler
{
public:
    enum class ButtonState : uint8_t
    {
        JustReleased,
        Released,
        Pressed,
        JustPressed
    };

    explicit MouseHandler(const Window& window);

    [[nodiscard]] auto getButtonState(int button) const -> ButtonState;
    [[nodiscard]] auto getCursorPosition() const -> glm::vec2;
    [[nodiscard]] auto getCursorDeltaPosition() const -> glm::vec2;

private:
    void handleMouseButtonState(const signal::MouseButtonStateChangedData& data);
    void handleCursorPosition(const signal::CursorPositionChangedData& data);
    void handleGameLoopIteration();

    std::array<ButtonState, GLFW_MOUSE_BUTTON_LAST> _states {};
    glm::dvec2 _currentPosition {};
    glm::dvec2 _previousPosition {};

    signal::MouseButtonStateChanged::ReceiverT _mouseButtonStateChangedReceiver;
    signal::CursorPositionChanged::ReceiverT _cursorStateChangedReceiver;
    signal::MainLoopIterationStarted::ReceiverT _newFrameNotifReceiver;

    const Window& _window;
};

}
