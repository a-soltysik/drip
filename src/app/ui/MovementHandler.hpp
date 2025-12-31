#pragma once

#include <GLFW/glfw3.h>

#include <optional>

#include "ui/Window.hpp"

namespace drip::app
{

class MovementHandler
{
public:
    struct Result
    {
        std::optional<float> x;
        std::optional<float> y;
        std::optional<float> z;
    };

    struct Controls
    {
        int left;
        int right;
        int up;
        int down;
        int forward;
        int backward;
    };

    static constexpr auto controls = Controls {.left = GLFW_KEY_A,
                                               .right = GLFW_KEY_D,
                                               .up = GLFW_KEY_SPACE,
                                               .down = GLFW_KEY_LEFT_SHIFT,
                                               .forward = GLFW_KEY_W,
                                               .backward = GLFW_KEY_S};

    explicit MovementHandler(const Window& window);
    [[nodiscard]] auto getMovement() const -> Result;

private:
    [[nodiscard]] auto getXMovement() const -> std::optional<float>;
    [[nodiscard]] auto getYMovement() const -> std::optional<float>;
    [[nodiscard]] auto getZMovement() const -> std::optional<float>;

    const Window& _window;
};

}
