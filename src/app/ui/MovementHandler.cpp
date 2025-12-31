#include "MovementHandler.hpp"

#include <optional>

#include "input/KeyboardHandler.hpp"

namespace drip::app
{

MovementHandler::MovementHandler(const Window& window)
    : _window {window}
{
}

auto MovementHandler::getMovement() const -> Result
{
    return Result {.x = getXMovement(), .y = getYMovement(), .z = getZMovement()};
}

auto MovementHandler::getXMovement() const -> std::optional<float>
{
    const auto& keyboardHandler = _window.getKeyboardHandler();
    const auto rightButton = keyboardHandler.getKeyState(controls.right);
    const auto leftButton = keyboardHandler.getKeyState(controls.left);

    auto direction = std::optional<float> {};
    using enum KeyboardHandler::State;

    if (rightButton == Pressed || rightButton == JustPressed)
    {
        direction = direction.value_or(0) + 1;
    }
    if (leftButton == Pressed || rightButton == JustPressed)
    {
        direction = direction.value_or(0) - 1;
    }

    return direction;
}

auto MovementHandler::getYMovement() const -> std::optional<float>
{
    const auto& keyboardHandler = _window.getKeyboardHandler();
    const auto upButton = keyboardHandler.getKeyState(controls.up);
    const auto downButton = keyboardHandler.getKeyState(controls.down);

    auto direction = std::optional<float> {};
    using enum KeyboardHandler::State;
    if (upButton == Pressed || upButton == JustPressed)
    {
        direction = direction.value_or(0) + 1;
    }
    if (downButton == Pressed || downButton == JustPressed)
    {
        direction = direction.value_or(0) - 1;
    }

    return direction;
}

auto MovementHandler::getZMovement() const -> std::optional<float>
{
    const auto& keyboardHandler = _window.getKeyboardHandler();

    const auto forwardButton = keyboardHandler.getKeyState(controls.forward);
    const auto backButton = keyboardHandler.getKeyState(controls.backward);

    auto direction = std::optional<float> {};
    using enum KeyboardHandler::State;
    if (forwardButton == Pressed || forwardButton == JustPressed)
    {
        direction = direction.value_or(0) + 1;
    }
    if (backButton == Pressed || backButton == JustPressed)
    {
        direction = direction.value_or(0) - 1;
    }

    return direction;
}

}
